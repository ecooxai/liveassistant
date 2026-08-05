use anyhow::{Context, Result, bail};
use futures_util::StreamExt;
use std::{collections::VecDeque, fmt::Display, sync::mpsc, thread, time::Duration};
use tokio::sync::mpsc::{self as tokio_mpsc, UnboundedReceiver, UnboundedSender};

use libwebrtc::{
    MediaType,
    audio_frame::AudioFrame,
    audio_source::{AudioSourceOptions, native::NativeAudioSource},
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
const AUDIO_SAMPLE_RATE: u32 = 24_000;
const AUDIO_FRAME_SAMPLES: usize = AUDIO_SAMPLE_RATE as usize / 100;

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
    local_audio_tx: UnboundedSender<Vec<i16>>,
    remote_audio: UnboundedReceiver<Result<Vec<i16>, String>>,
}

struct NativePeer {
    peer_connection: PeerConnection,
    _events_channel: DataChannel,
    _local_audio_source: NativeAudioSource,
}

/// macOS GPT-Live transport with app-owned audio devices and native
/// libWebRTC media processing. VoiceProcessingIO performs acoustic echo
/// cancellation while libWebRTC provides Opus, NetEQ jitter buffering,
/// packet-loss concealment, clock correction, and remote decoding.
pub struct GptLivePeer {
    handle: SessionHandle,
    local_audio_tx: UnboundedSender<Vec<i16>>,
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
                local_audio_tx: started.local_audio_tx,
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

    /// Inject VoiceProcessingIO-cleaned 24 kHz mono microphone PCM into
    /// libWebRTC's encoder.
    pub async fn send_pcm24k(&self, samples: &[i16]) -> Result<()> {
        if samples.is_empty() {
            return Ok(());
        }
        self.local_audio_tx
            .send(samples.to_vec())
            .map_err(|_| anyhow::anyhow!("GPT-Live native microphone sender stopped"))
    }

    /// Return NetEQ-decoded remote PCM for VoiceProcessingIO playout and
    /// message replay/export.
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
    let (local_audio_tx, local_audio_rx) = tokio_mpsc::unbounded_channel();
    let (remote_audio_tx, remote_audio) = tokio_mpsc::unbounded_channel();

    thread::Builder::new()
        .name("live-assistant-gpt-live-webrtc".to_owned())
        .spawn(move || worker_main(command_rx, offer_tx, local_audio_rx, remote_audio_tx))
        .context("Could not spawn native GPT-Live WebRTC worker")?;

    let offer_sdp = offer_rx
        .recv()
        .map_err(|_| anyhow::anyhow!("GPT-Live native WebRTC worker stopped"))??;
    Ok(StartedSession {
        offer_sdp,
        handle: SessionHandle { command_tx },
        local_audio_tx,
        remote_audio,
    })
}

fn worker_main(
    command_rx: mpsc::Receiver<Command>,
    offer_tx: mpsc::Sender<Result<String>>,
    local_audio_rx: UnboundedReceiver<Vec<i16>>,
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
        local_audio_rx,
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
    mut local_audio_rx: UnboundedReceiver<Vec<i16>>,
    remote_audio_tx: UnboundedSender<Result<Vec<i16>, String>>,
) -> Result<(NativePeer, String)> {
    // The app owns VoiceProcessingIO capture and playout. An external-audio
    // factory avoids a second platform ADM competing for the same devices.
    let factory = PeerConnectionFactory::default();
    eprintln!(
        "[live-assistant webrtc-native] mode=external-audio mic_rate={} remote=neteq aec=voiceprocessingio",
        AUDIO_SAMPLE_RATE
    );
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
    let local_audio_source = NativeAudioSource::new(
        AudioSourceOptions {
            // VoiceProcessingIO already performed input processing. A second
            // APM pass causes pumping, clipping, and intelligibility loss.
            echo_cancellation: false,
            noise_suppression: false,
            auto_gain_control: false,
        },
        AUDIO_SAMPLE_RATE,
        1,
        1_000,
    );
    let local_audio_track = factory.create_audio_track("realtime-mic", local_audio_source.clone());
    audio_transceiver
        .sender()
        .set_track(Some(local_audio_track.into()))
        .map_err(|error| message_error("Could not attach native GPT-Live microphone", error))?;

    let local_audio_writer = local_audio_source.clone();
    let local_audio_errors = remote_audio_tx.clone();
    tokio::spawn(async move {
        let mut pending = VecDeque::<i16>::new();
        while let Some(samples) = local_audio_rx.recv().await {
            pending.extend(samples);
            while pending.len() >= AUDIO_FRAME_SAMPLES {
                let frame_samples = pending.drain(..AUDIO_FRAME_SAMPLES).collect::<Vec<_>>();
                let frame = AudioFrame {
                    data: frame_samples.into(),
                    sample_rate: AUDIO_SAMPLE_RATE,
                    num_channels: 1,
                    samples_per_channel: AUDIO_FRAME_SAMPLES as u32,
                };
                if let Err(error) = local_audio_writer.capture_frame(&frame).await {
                    let _ = local_audio_errors.send(Err(format!(
                        "Could not inject microphone PCM into GPT-Live libWebRTC: {error}"
                    )));
                    return;
                }
            }
        }
    });

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
            _local_audio_source: local_audio_source,
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
