use anyhow::{Context, Result, bail};
use futures_util::StreamExt;
use std::{fmt::Display, sync::mpsc, thread, time::Duration};
use tokio::sync::mpsc::{self as tokio_mpsc, UnboundedReceiver, UnboundedSender};

use libwebrtc::{
    MediaType,
    audio_stream::native::{NativeAudioStream, NativeAudioStreamOptions},
    data_channel::{DataChannel, DataChannelInit},
    media_stream_track::MediaStreamTrack,
    peer_connection::{IceGatheringState, OfferOptions, PeerConnection, PeerConnectionState},
    peer_connection_factory::{
        PeerConnectionFactory, RtcConfiguration, native::PeerConnectionFactoryExt,
    },
    rtp_transceiver::{RtpTransceiverDirection, RtpTransceiverInit},
    session_description::{SdpType, SessionDescription},
};

const ICE_GATHER_TIMEOUT: Duration = Duration::from_secs(15);

enum Command {
    ApplyAnswer {
        answer_sdp: String,
        reply: mpsc::Sender<Result<()>>,
    },
    Close,
}

#[derive(Clone)]
struct SessionHandle {
    command_tx: mpsc::Sender<Command>,
}

struct StartedSession {
    offer_sdp: String,
    handle: SessionHandle,
    local_audio: UnboundedReceiver<Result<Vec<i16>, String>>,
    remote_audio: UnboundedReceiver<Result<Vec<i16>, String>>,
}

struct NativePeer {
    peer_connection: PeerConnection,
    _events_channel: DataChannel,
}

/// macOS GPT-Live transport using the same architecture as Codex's native
/// implementation: Google libWebRTC owns microphone capture, AEC, adaptive
/// jitter buffering, Opus PLC, clock drift correction, and speaker playout.
pub struct GptLivePeer {
    handle: SessionHandle,
    local_audio: UnboundedReceiver<Result<Vec<i16>, String>>,
    remote_audio: UnboundedReceiver<Result<Vec<i16>, String>>,
}

impl GptLivePeer {
    pub async fn create() -> Result<(Self, String)> {
        let started = tokio::task::spawn_blocking(start_native_session)
            .await
            .context("GPT-Live native WebRTC startup task panicked")??;
        Ok((
            Self {
                handle: started.handle,
                local_audio: started.local_audio,
                remote_audio: started.remote_audio,
            },
            started.offer_sdp,
        ))
    }

    pub async fn accept_answer(&self, answer_sdp: String) -> Result<()> {
        let handle = self.handle.clone();
        tokio::task::spawn_blocking(move || {
            let (reply, reply_rx) = mpsc::channel();
            handle
                .command_tx
                .send(Command::ApplyAnswer { answer_sdp, reply })
                .map_err(|_| anyhow::anyhow!("GPT-Live native WebRTC worker stopped"))?;
            reply_rx
                .recv()
                .map_err(|_| anyhow::anyhow!("GPT-Live native WebRTC worker stopped"))?
        })
        .await
        .context("GPT-Live native WebRTC answer task panicked")?
    }

    /// Native libWebRTC captures and packetizes the microphone itself. The app
    /// deliberately does not inject a second PCM/RTP microphone stream.
    pub async fn send_pcm24k(&self, _samples: &[i16]) -> Result<()> {
        Ok(())
    }

    /// Return a capture-only copy of the platform microphone track. Native
    /// libWebRTC still owns capture, AEC, and upstream packetization.
    pub fn take_local_audio(&mut self) -> UnboundedReceiver<Result<Vec<i16>, String>> {
        let (_keepalive, replacement) = tokio_mpsc::unbounded_channel();
        std::mem::replace(&mut self.local_audio, replacement)
    }

    /// Return a capture-only copy of the decoded remote track. Native libWebRTC
    /// continues rendering the same track directly through the platform ADM.
    pub fn take_remote_audio(&mut self) -> UnboundedReceiver<Result<Vec<i16>, String>> {
        let (_keepalive, replacement) = tokio_mpsc::unbounded_channel();
        std::mem::replace(&mut self.remote_audio, replacement)
    }

    pub async fn close(&self) {
        let _ = self.handle.command_tx.send(Command::Close);
    }
}

fn start_native_session() -> Result<StartedSession> {
    let (command_tx, command_rx) = mpsc::channel();
    let (offer_tx, offer_rx) = mpsc::channel();
    let (local_audio_tx, local_audio) = tokio_mpsc::unbounded_channel();
    let (remote_audio_tx, remote_audio) = tokio_mpsc::unbounded_channel();

    thread::Builder::new()
        .name("live-assistant-gpt-live-webrtc".to_owned())
        .spawn(move || worker_main(command_rx, offer_tx, local_audio_tx, remote_audio_tx))
        .context("Could not spawn native GPT-Live WebRTC worker")?;

    let offer_sdp = offer_rx
        .recv()
        .map_err(|_| anyhow::anyhow!("GPT-Live native WebRTC worker stopped"))??;
    Ok(StartedSession {
        offer_sdp,
        handle: SessionHandle { command_tx },
        local_audio,
        remote_audio,
    })
}

fn worker_main(
    command_rx: mpsc::Receiver<Command>,
    offer_tx: mpsc::Sender<Result<String>>,
    local_audio_tx: UnboundedSender<Result<Vec<i16>, String>>,
    remote_audio_tx: UnboundedSender<Result<Vec<i16>, String>>,
) {
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_name("live-assistant-libwebrtc")
        .build()
    {
        Ok(runtime) => runtime,
        Err(error) => {
            let _ = offer_tx.send(Err(error).context("Could not start libWebRTC runtime"));
            return;
        }
    };

    let native_peer = match runtime.block_on(create_peer_connection_and_offer(
        local_audio_tx,
        remote_audio_tx,
    )) {
        Ok((peer, offer_sdp)) => {
            let _ = offer_tx.send(Ok(offer_sdp));
            peer
        }
        Err(error) => {
            let _ = offer_tx.send(Err(error));
            return;
        }
    };

    for command in command_rx {
        match command {
            Command::ApplyAnswer { answer_sdp, reply } => {
                let result =
                    runtime.block_on(apply_answer(&native_peer.peer_connection, answer_sdp));
                let _ = reply.send(result);
            }
            Command::Close => {
                native_peer.peer_connection.close();
                return;
            }
        }
    }
    native_peer.peer_connection.close();
}

async fn create_peer_connection_and_offer(
    local_audio_tx: UnboundedSender<Result<Vec<i16>, String>>,
    remote_audio_tx: UnboundedSender<Result<Vec<i16>, String>>,
) -> Result<(NativePeer, String)> {
    let factory = PeerConnectionFactory::with_platform_adm();
    let peer_connection = factory
        .create_peer_connection(RtcConfiguration::default())
        .map_err(|error| {
            message_error("Could not create native GPT-Live peer connection", error)
        })?;

    let audio_transceiver = peer_connection
        .add_transceiver_for_media(
            MediaType::Audio,
            RtpTransceiverInit {
                direction: RtpTransceiverDirection::SendRecv,
                stream_ids: vec!["realtime".to_owned()],
                send_encodings: Vec::new(),
            },
        )
        .map_err(|error| message_error("Could not add native GPT-Live audio transceiver", error))?;
    let local_audio_source = factory.create_audio_source();
    let local_audio_track = factory.create_audio_track("realtime-mic", local_audio_source);
    spawn_audio_capture(
        tokio::runtime::Handle::current(),
        local_audio_track.clone(),
        local_audio_tx,
        "microphone",
    );
    audio_transceiver
        .sender()
        .set_track(Some(local_audio_track.into()))
        .map_err(|error| message_error("Could not attach native GPT-Live microphone", error))?;

    let remote_runtime = tokio::runtime::Handle::current();
    peer_connection.on_track(Some(Box::new(move |event| {
        let MediaStreamTrack::Audio(track) = event.track else {
            return;
        };
        spawn_audio_capture(
            remote_runtime.clone(),
            track,
            remote_audio_tx.clone(),
            "assistant",
        );
    })));

    // Current app-server documentation requires this data channel in the offer,
    // even though control events are consumed through the Codex sideband stream.
    let events_channel = peer_connection
        .create_data_channel("oai-events", DataChannelInit::default())
        .map_err(|error| message_error("Could not create GPT-Live events data channel", error))?;

    let offer = peer_connection
        .create_offer(OfferOptions {
            ice_restart: false,
            offer_to_receive_audio: true,
            offer_to_receive_video: false,
        })
        .await
        .map_err(|error| message_error("Could not create native GPT-Live offer", error))?;

    let (gathered_tx, gathered_rx) = tokio::sync::oneshot::channel();
    let gathered_tx = std::sync::Arc::new(std::sync::Mutex::new(Some(gathered_tx)));
    let gathered_tx_callback = gathered_tx.clone();
    peer_connection.on_ice_gathering_state_change(Some(Box::new(move |state| {
        if state == IceGatheringState::Complete
            && let Ok(mut sender) = gathered_tx_callback.lock()
            && let Some(sender) = sender.take()
        {
            let _ = sender.send(());
        }
    })));

    peer_connection
        .set_local_description(offer.clone())
        .await
        .map_err(|error| message_error("Could not set native GPT-Live local description", error))?;

    if peer_connection.ice_gathering_state() != IceGatheringState::Complete {
        let _ = tokio::time::timeout(ICE_GATHER_TIMEOUT, gathered_rx).await;
    }
    let offer = peer_connection.current_local_description().unwrap_or(offer);
    let offer_sdp = offer.to_string();
    if !offer_sdp.starts_with("v=0") {
        bail!("Native GPT-Live WebRTC produced an invalid SDP offer");
    }

    Ok((
        NativePeer {
            peer_connection,
            _events_channel: events_channel,
        },
        offer_sdp,
    ))
}

fn spawn_audio_capture(
    runtime: tokio::runtime::Handle,
    track: libwebrtc::audio_track::RtcAudioTrack,
    output: UnboundedSender<Result<Vec<i16>, String>>,
    label: &'static str,
) {
    let mut stream = NativeAudioStream::with_options(
        track,
        24_000,
        1,
        NativeAudioStreamOptions {
            // Message audio must remain complete for replay/export. The realtime
            // control loop drains this channel continuously, so an unbounded sink
            // avoids silently dropping old 10 ms frames during brief UI stalls.
            queue_size_frames: Some(0),
        },
    );
    runtime.spawn(async move {
        while let Some(frame) = stream.next().await {
            if frame.sample_rate != 24_000 || frame.num_channels != 1 {
                let _ = output.send(Err(format!(
                    "GPT-Live native {label} capture returned {} Hz / {} channels",
                    frame.sample_rate, frame.num_channels
                )));
                continue;
            }
            let samples = frame.data.into_owned();
            if !samples.is_empty() && output.send(Ok(samples)).is_err() {
                break;
            }
        }
    });
}

async fn apply_answer(peer_connection: &PeerConnection, answer_sdp: String) -> Result<()> {
    if !answer_sdp.starts_with("v=0") {
        bail!("Codex GPT-Live returned an invalid SDP answer");
    }
    let answer = SessionDescription::parse(&answer_sdp, SdpType::Answer)
        .map_err(|error| message_error("Could not parse native GPT-Live answer", error))?;
    peer_connection
        .set_remote_description(answer)
        .await
        .map_err(|error| message_error("Could not apply native GPT-Live answer", error))?;

    let deadline = tokio::time::Instant::now() + Duration::from_secs(15);
    loop {
        match peer_connection.connection_state() {
            PeerConnectionState::Connected => return Ok(()),
            PeerConnectionState::Failed | PeerConnectionState::Closed => {
                bail!("Native GPT-Live peer connection failed")
            }
            _ if tokio::time::Instant::now() >= deadline => {
                bail!("Timed out connecting native GPT-Live WebRTC audio")
            }
            _ => tokio::time::sleep(Duration::from_millis(25)).await,
        }
    }
}

fn message_error(prefix: &str, error: impl Display) -> anyhow::Error {
    anyhow::anyhow!("{prefix}: {error}")
}
