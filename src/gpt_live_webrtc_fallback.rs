use anyhow::{Context, Result, bail};
use bytes::Bytes;
use opus::{Application, Channels, Decoder, Encoder, Signal};
use std::{collections::VecDeque, sync::Arc, time::Duration};
use tokio::sync::mpsc::{self, UnboundedReceiver, UnboundedSender};
use webrtc::{
    api::{
        APIBuilder,
        interceptor_registry::register_default_interceptors,
        media_engine::{MIME_TYPE_OPUS, MediaEngine},
    },
    data_channel::RTCDataChannel,
    interceptor::registry::Registry,
    media::{Sample, io::sample_builder::SampleBuilder},
    peer_connection::{
        RTCPeerConnection, configuration::RTCConfiguration,
        sdp::session_description::RTCSessionDescription,
    },
    rtp::codecs::opus::OpusPacket,
    rtp_transceiver::{rtp_codec::RTCRtpCodecCapability, rtp_sender::RTCRtpSender},
    track::track_local::{TrackLocal, track_local_static_sample::TrackLocalStaticSample},
};

const INPUT_RATE: usize = 24_000;
const OPUS_RATE: usize = 48_000;
const FRAME_MS: usize = 20;
const INPUT_FRAME_SAMPLES: usize = INPUT_RATE * FRAME_MS / 1_000;
const MAX_OPUS_PACKET_BYTES: usize = 4_000;
const MAX_OPUS_DECODE_SAMPLES_PER_CHANNEL: usize = 5_760;
/// GPT-Live RTP remains open and carries low-level comfort noise between replies.
/// Keep a generous voice gate so quiet phonemes are not mistaken for transport
/// noise. RTP packets are reordered separately before reaching this gate.
const REMOTE_VOICE_RMS: f64 = 12.0;
const REMOTE_SPEECH_PRE_ROLL_PACKETS: usize = 15;
const REMOTE_SPEECH_TAIL_PACKETS: usize = 60;
/// Raw `TrackRemote::read_rtp` bypasses browser/libWebRTC playout buffering. Wait
/// briefly for reordered packets, then use Opus packet-loss concealment for gaps.
const REMOTE_JITTER_MAX_LATE_PACKETS: u16 = 12;
const REMOTE_JITTER_MAX_DELAY: Duration = Duration::from_millis(240);
const REMOTE_MAX_CONCEALED_PACKETS: u16 = 12;

#[derive(Default)]
struct RemoteSpeechGate {
    pre_roll: VecDeque<Vec<i16>>,
    active: bool,
    quiet_packets: usize,
}

impl RemoteSpeechGate {
    fn push(&mut self, samples: Vec<i16>) -> Vec<Vec<i16>> {
        if samples.is_empty() {
            return Vec::new();
        }
        let voiced = pcm_rms(&samples) >= REMOTE_VOICE_RMS;
        if !self.active {
            self.pre_roll.push_back(samples);
            while self.pre_roll.len() > REMOTE_SPEECH_PRE_ROLL_PACKETS {
                self.pre_roll.pop_front();
            }
            if !voiced {
                return Vec::new();
            }
            self.active = true;
            self.quiet_packets = 0;
            return self.pre_roll.drain(..).collect();
        }

        if voiced {
            self.quiet_packets = 0;
        } else {
            self.quiet_packets = self.quiet_packets.saturating_add(1);
        }
        let output = vec![samples];
        if self.quiet_packets >= REMOTE_SPEECH_TAIL_PACKETS {
            self.active = false;
            self.quiet_packets = 0;
            self.pre_roll.clear();
        }
        output
    }
}

fn pcm_rms(samples: &[i16]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples
        .iter()
        .map(|sample| {
            let value = *sample as f64;
            value * value
        })
        .sum::<f64>()
        / samples.len() as f64)
        .sqrt()
}

fn decode_opus_to_mono(
    decoder: &mut Decoder,
    payload: &[u8],
    channel_count: usize,
    scratch: &mut [i16],
) -> std::result::Result<Vec<i16>, opus::Error> {
    let samples_per_channel = decoder.decode(payload, scratch, false)?;
    let decoded = &scratch[..samples_per_channel.saturating_mul(channel_count)];
    Ok(if channel_count == 1 {
        decoded.to_vec()
    } else {
        decoded
            .chunks_exact(channel_count)
            .map(|frame| {
                let sum = frame.iter().map(|sample| *sample as i64).sum::<i64>();
                (sum / channel_count as i64).clamp(i16::MIN as i64, i16::MAX as i64) as i16
            })
            .collect()
    })
}

fn concealment_frames(sample_duration: Duration, decoded_samples: usize) -> usize {
    if decoded_samples == 0 {
        return 0;
    }
    let expected_samples =
        ((sample_duration.as_secs_f64() * INPUT_RATE as f64).round() as usize).max(decoded_samples);
    expected_samples
        .saturating_sub(decoded_samples)
        .div_ceil(decoded_samples)
        .min(REMOTE_MAX_CONCEALED_PACKETS as usize)
}

fn send_gated_remote_audio(
    speech_gate: &mut RemoteSpeechGate,
    samples: Vec<i16>,
    remote_audio_tx: &mpsc::UnboundedSender<Result<Vec<i16>, String>>,
    emitted_packet_count: &mut usize,
    source_packet_count: usize,
) -> bool {
    for speech_packet in speech_gate.push(samples) {
        *emitted_packet_count = emitted_packet_count.saturating_add(1);
        if *emitted_packet_count == 1 || emitted_packet_count.is_multiple_of(50) {
            eprintln!(
                "[live-assistant webrtc] emitted speech packets={} source_packets={}",
                *emitted_packet_count, source_packet_count
            );
        }
        if remote_audio_tx.send(Ok(speech_packet)).is_err() {
            return false;
        }
    }
    true
}
/// Native WebRTC audio transport used by Codex GPT-Live V3.
///
/// Codex app-server owns authentication, call creation and the sideband event
/// stream. This object owns only the peer connection's audio media path.
#[derive(Clone)]
pub(crate) struct GptLiveAudioSender {
    tx: UnboundedSender<Vec<i16>>,
}

impl GptLiveAudioSender {
    pub(crate) fn send_pcm24k(&self, samples: &[i16]) -> Result<()> {
        if samples.is_empty() {
            return Ok(());
        }
        self.tx
            .send(samples.to_vec())
            .map_err(|_| anyhow::anyhow!("GPT-Live microphone sender stopped"))
    }
}

pub struct GptLivePeer {
    peer: Arc<RTCPeerConnection>,
    _local_audio: Arc<TrackLocalStaticSample>,
    _rtp_sender: Arc<RTCRtpSender>,
    _events_channel: Arc<RTCDataChannel>,
    local_audio_tx: mpsc::UnboundedSender<Vec<i16>>,
    remote_audio: UnboundedReceiver<Result<Vec<i16>, String>>,
}

impl GptLivePeer {
    /// Creates a send/receive Opus peer and returns a fully gathered SDP offer.
    pub async fn create() -> Result<(Self, String)> {
        let mut media_engine = MediaEngine::default();
        media_engine
            .register_default_codecs()
            .context("Could not register WebRTC codecs")?;
        let registry = register_default_interceptors(Registry::new(), &mut media_engine)
            .context("Could not register WebRTC interceptors")?;
        let api = APIBuilder::new()
            .with_media_engine(media_engine)
            .with_interceptor_registry(registry)
            .build();
        let peer = Arc::new(
            api.new_peer_connection(RTCConfiguration::default())
                .await
                .context("Could not create GPT-Live WebRTC peer")?,
        );

        // The official Codex WebRTC flow requires this channel in the SDP offer.
        // App-server uses it for V3 realtime events while the sideband exposes a
        // stable thread/realtime/* notification surface to this native client.
        let events_channel = peer
            .create_data_channel("oai-events", None)
            .await
            .context("Could not create the GPT-Live realtime events data channel")?;

        let local_audio = Arc::new(TrackLocalStaticSample::new(
            RTCRtpCodecCapability {
                mime_type: MIME_TYPE_OPUS.to_owned(),
                clock_rate: OPUS_RATE as u32,
                // WebRTC advertises Opus as /2 even when the encoded signal is mono.
                channels: 2,
                sdp_fmtp_line: "minptime=10;useinbandfec=1".to_owned(),
                ..Default::default()
            },
            "audio".to_owned(),
            "live-assistant".to_owned(),
        ));
        let rtp_sender = peer
            .add_track(Arc::clone(&local_audio) as Arc<dyn TrackLocal + Send + Sync>)
            .await
            .context("Could not add microphone track to GPT-Live WebRTC")?;

        // Consume RTCP so sender feedback cannot back up the interceptor pipeline.
        let rtcp_sender = Arc::clone(&rtp_sender);
        tokio::spawn(async move { while rtcp_sender.read_rtcp().await.is_ok() {} });

        let (remote_audio_tx, remote_audio) = mpsc::unbounded_channel();
        let mut encoder = Encoder::new(INPUT_RATE as u32, Channels::Mono, Application::Voip)
            .context("Could not initialize GPT-Live Opus encoder")?;
        encoder
            .set_inband_fec(true)
            .context("Could not enable GPT-Live Opus in-band FEC")?;
        encoder
            .set_packet_loss_perc(10)
            .context("Could not configure GPT-Live Opus packet-loss resilience")?;
        encoder
            .set_complexity(10)
            .context("Could not configure GPT-Live Opus complexity")?;
        encoder
            .set_signal(Signal::Voice)
            .context("Could not configure GPT-Live Opus voice mode")?;
        let (local_audio_tx, mut local_audio_rx) = mpsc::unbounded_channel::<Vec<i16>>();
        let local_audio_writer = Arc::clone(&local_audio);
        let local_audio_errors = remote_audio_tx.clone();
        tokio::spawn(async move {
            let mut pending_input = VecDeque::<i16>::new();
            while let Some(samples) = local_audio_rx.recv().await {
                pending_input.extend(samples);
                while pending_input.len() >= INPUT_FRAME_SAMPLES {
                    let frame_24k = pending_input
                        .drain(..INPUT_FRAME_SAMPLES)
                        .collect::<Vec<_>>();
                    let encoded = match encoder.encode_vec(&frame_24k, MAX_OPUS_PACKET_BYTES) {
                        Ok(encoded) => encoded,
                        Err(error) => {
                            let _ = local_audio_errors.send(Err(format!(
                                "Could not encode microphone audio for GPT-Live: {error}"
                            )));
                            return;
                        }
                    };
                    if let Err(error) = local_audio_writer
                        .write_sample(&Sample {
                            data: Bytes::from(encoded),
                            duration: Duration::from_millis(FRAME_MS as u64),
                            ..Default::default()
                        })
                        .await
                    {
                        let _ = local_audio_errors.send(Err(format!(
                            "Could not send microphone audio to GPT-Live: {error}"
                        )));
                        return;
                    }
                }
            }
        });

        peer.on_track(Box::new(move |track, _receiver, _transceiver| {
            let remote_audio_tx = remote_audio_tx.clone();
            let negotiated_channels = track.codec().capability.channels;
            eprintln!(
                "[live-assistant webrtc] remote track id={} stream_id={} codec={} channels={} decode_rate=24000",
                track.id(),
                track.stream_id(),
                track.codec().capability.mime_type,
                negotiated_channels
            );
            Box::pin(async move {
                tokio::spawn(async move {
                    let mut packet_count = 0usize;
                    let mut emitted_packet_count = 0usize;
                    let mut concealed_packet_count = 0usize;
                    let mut speech_gate = RemoteSpeechGate::default();
                    let channel_count = if negotiated_channels == 1 { 1 } else { 2 };
                    let decoder_channels = if channel_count == 1 {
                        Channels::Mono
                    } else {
                        Channels::Stereo
                    };
                    // Decode at 24 kHz, but reorder by the negotiated 48 kHz RTP clock.
                    let mut decoder = match Decoder::new(INPUT_RATE as u32, decoder_channels) {
                        Ok(decoder) => decoder,
                        Err(error) => {
                            let _ = remote_audio_tx.send(Err(format!(
                                "Could not initialize GPT-Live Opus decoder: {error}"
                            )));
                            return;
                        }
                    };
                    let mut sample_builder = SampleBuilder::new(
                        REMOTE_JITTER_MAX_LATE_PACKETS,
                        OpusPacket,
                        OPUS_RATE as u32,
                    )
                    .with_max_time_delay(REMOTE_JITTER_MAX_DELAY);
                    let mut decoded =
                        vec![0_i16; MAX_OPUS_DECODE_SAMPLES_PER_CHANNEL * channel_count];
                    loop {
                        let (packet, _) = match track.read_rtp().await {
                            Ok(value) => value,
                            Err(error) => {
                                let _ = remote_audio_tx.send(Err(format!(
                                    "GPT-Live WebRTC audio track closed: {error}"
                                )));
                                break;
                            }
                        };
                        packet_count = packet_count.saturating_add(1);
                        sample_builder.push(packet);
                        let mut last_decoded_samples = 0usize;
                        let mut reported_dropped_packets = 0usize;
                        while let Some(sample) = sample_builder.pop() {
                            reported_dropped_packets = reported_dropped_packets.saturating_add(
                                sample
                                    .prev_dropped_packets
                                    .saturating_sub(sample.prev_padding_packets)
                                    as usize,
                            );
                            let mono_24k = match decode_opus_to_mono(
                                &mut decoder,
                                sample.data.as_ref(),
                                channel_count,
                                &mut decoded,
                            ) {
                                Ok(samples) => samples,
                                Err(error) => {
                                    let _ = remote_audio_tx.send(Err(format!(
                                        "Could not decode GPT-Live Opus audio: {error}"
                                    )));
                                    continue;
                                }
                            };
                            last_decoded_samples = mono_24k.len();
                            let missing_frames = concealment_frames(sample.duration, mono_24k.len());
                            if !send_gated_remote_audio(
                                &mut speech_gate,
                                mono_24k,
                                &remote_audio_tx,
                                &mut emitted_packet_count,
                                packet_count,
                            ) {
                                return;
                            }

                            // Only conceal timestamp holes while speech is active. Sequence
                            // gaps during DTX/comfort-noise periods must not become fake speech.
                            if speech_gate.active {
                                for _ in 0..missing_frames {
                                    match decode_opus_to_mono(
                                        &mut decoder,
                                        &[],
                                        channel_count,
                                        &mut decoded,
                                    ) {
                                        Ok(concealed) => {
                                            concealed_packet_count =
                                                concealed_packet_count.saturating_add(1);
                                            if !send_gated_remote_audio(
                                                &mut speech_gate,
                                                concealed,
                                                &remote_audio_tx,
                                                &mut emitted_packet_count,
                                                packet_count,
                                            ) {
                                                return;
                                            }
                                        }
                                        Err(error) => {
                                            eprintln!(
                                                "[live-assistant webrtc] Opus PLC failed: {error}"
                                            );
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                        if packet_count == 1 || packet_count.is_multiple_of(100) {
                            eprintln!(
                                "[live-assistant webrtc] received_packets={} decoded_samples_24k={} reported_drops={} concealed_frames={}",
                                packet_count,
                                last_decoded_samples,
                                reported_dropped_packets,
                                concealed_packet_count
                            );
                        }
                    }
                });
            })
        }));

        let offer = peer
            .create_offer(None)
            .await
            .context("Could not create GPT-Live WebRTC offer")?;
        let mut gathering_complete = peer.gathering_complete_promise().await;
        peer.set_local_description(offer)
            .await
            .context("Could not set GPT-Live local SDP")?;
        let _ = tokio::time::timeout(Duration::from_secs(15), gathering_complete.recv())
            .await
            .context("Timed out gathering GPT-Live ICE candidates")?;
        let offer_sdp = peer
            .local_description()
            .await
            .context("GPT-Live WebRTC did not produce a local SDP offer")?
            .sdp;
        if !offer_sdp.starts_with("v=0") {
            bail!("GPT-Live WebRTC produced an invalid SDP offer");
        }

        Ok((
            Self {
                peer,
                _local_audio: local_audio,
                _rtp_sender: rtp_sender,
                _events_channel: events_channel,
                local_audio_tx,
                remote_audio,
            },
            offer_sdp,
        ))
    }

    pub async fn accept_answer(&self, answer_sdp: String) -> Result<()> {
        if !answer_sdp.starts_with("v=0") {
            bail!("Codex GPT-Live returned an invalid SDP answer");
        }
        let answer = RTCSessionDescription::answer(answer_sdp)
            .context("Could not parse the GPT-Live SDP answer")?;
        self.peer
            .set_remote_description(answer)
            .await
            .context("Could not apply the GPT-Live SDP answer")
    }

    pub(crate) fn audio_sender(&self) -> GptLiveAudioSender {
        GptLiveAudioSender {
            tx: self.local_audio_tx.clone(),
        }
    }

    /// Enqueues 24 kHz mono PCM for the dedicated RTP sender task. Keeping
    /// microphone packetization off the control loop prevents upstream audio
    /// writes from delaying remote RTP receive and sideband events.
    pub async fn send_pcm24k(&self, samples: &[i16]) -> Result<()> {
        if samples.is_empty() {
            return Ok(());
        }
        self.local_audio_tx
            .send(samples.to_vec())
            .map_err(|_| anyhow::anyhow!("GPT-Live microphone sender stopped"))
    }

    pub fn take_remote_audio(&mut self) -> UnboundedReceiver<Result<Vec<i16>, String>> {
        let (_sender, replacement) = mpsc::unbounded_channel();
        std::mem::replace(&mut self.remote_audio, replacement)
    }

    pub async fn close(&self) {
        let _ = self.peer.close().await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concealment_uses_media_timestamp_duration_not_sequence_gaps() {
        assert_eq!(concealment_frames(Duration::from_millis(20), 480), 0);
        assert_eq!(concealment_frames(Duration::from_millis(40), 480), 1);
        assert_eq!(concealment_frames(Duration::from_millis(260), 480), 12);
    }

    #[test]
    fn sample_builder_reorders_opus_rtp_packets() {
        use webrtc::rtp::{header::Header, packet::Packet};

        let packet = |sequence_number, timestamp| Packet {
            header: Header {
                version: 2,
                marker: true,
                payload_type: 111,
                sequence_number,
                timestamp,
                ssrc: 1,
                ..Default::default()
            },
            payload: Bytes::from_static(&[1]),
        };
        let mut builder = SampleBuilder::new(12, OpusPacket, OPUS_RATE as u32)
            .with_max_time_delay(Duration::from_millis(240));
        builder.push(packet(11, 960));
        builder.push(packet(10, 0));
        builder.push(packet(12, 1_920));

        assert_eq!(builder.pop().unwrap().packet_timestamp, 0);
        assert_eq!(builder.pop().unwrap().packet_timestamp, 960);
    }

    #[test]
    fn remote_speech_gate_drops_comfort_noise_and_keeps_voice_tail() {
        let mut gate = RemoteSpeechGate::default();
        for _ in 0..30 {
            assert!(gate.push(vec![1; INPUT_FRAME_SAMPLES]).is_empty());
        }
        let started = gate.push(vec![2_000; INPUT_FRAME_SAMPLES]);
        assert_eq!(started.len(), REMOTE_SPEECH_PRE_ROLL_PACKETS);
        for _ in 0..REMOTE_SPEECH_TAIL_PACKETS {
            assert_eq!(gate.push(vec![0; INPUT_FRAME_SAMPLES]).len(), 1);
        }
        assert!(gate.push(vec![0; INPUT_FRAME_SAMPLES]).is_empty());
    }
}
