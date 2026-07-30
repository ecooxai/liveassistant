use anyhow::{Context, Result, bail};
use bytes::Bytes;
use opus::{Application, Channels, Decoder, Encoder};
use std::{collections::VecDeque, sync::Arc, time::Duration};
use tokio::sync::mpsc::{self, UnboundedReceiver};
use webrtc::{
    api::{
        APIBuilder,
        interceptor_registry::register_default_interceptors,
        media_engine::{MIME_TYPE_OPUS, MediaEngine},
    },
    data_channel::RTCDataChannel,
    interceptor::registry::Registry,
    media::Sample,
    peer_connection::{
        RTCPeerConnection, configuration::RTCConfiguration,
        sdp::session_description::RTCSessionDescription,
    },
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
/// Emit only real assistant speech, with a short pre-roll and natural tail.
const REMOTE_VOICE_RMS: f64 = 32.0;
const REMOTE_SPEECH_PRE_ROLL_PACKETS: usize = 10;
const REMOTE_SPEECH_TAIL_PACKETS: usize = 25;

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
/// Native WebRTC audio transport used by Codex GPT-Live V3.
///
/// Codex app-server owns authentication, call creation and the sideband event
/// stream. This object owns only the peer connection's audio media path.
pub struct GptLivePeer {
    peer: Arc<RTCPeerConnection>,
    local_audio: Arc<TrackLocalStaticSample>,
    _rtp_sender: Arc<RTCRtpSender>,
    _events_channel: Arc<RTCDataChannel>,
    encoder: Encoder,
    pending_input: VecDeque<i16>,
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
                    let mut speech_gate = RemoteSpeechGate::default();
                    let channel_count = if negotiated_channels == 1 { 1 } else { 2 };
                    let decoder_channels = if channel_count == 1 {
                        Channels::Mono
                    } else {
                        Channels::Stereo
                    };
                    // libopus performs its own high-quality native 24 kHz decode.
                    // This removes the previous box-filter 48→24 kHz conversion.
                    let mut decoder = match Decoder::new(INPUT_RATE as u32, decoder_channels) {
                        Ok(decoder) => decoder,
                        Err(error) => {
                            let _ = remote_audio_tx.send(Err(format!(
                                "Could not initialize GPT-Live Opus decoder: {error}"
                            )));
                            return;
                        }
                    };
                    let mut decoded = vec![0_i16; MAX_OPUS_DECODE_SAMPLES_PER_CHANNEL * channel_count];
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
                        let samples_per_channel = match decoder.decode(
                            packet.payload.as_ref(),
                            decoded.as_mut_slice(),
                            false,
                        ) {
                            Ok(count) => count,
                            Err(error) => {
                                let _ = remote_audio_tx.send(Err(format!(
                                    "Could not decode GPT-Live Opus audio: {error}"
                                )));
                                continue;
                            }
                        };
                        let decoded = &decoded[..samples_per_channel.saturating_mul(channel_count)];
                        let mono_24k = if channel_count == 1 {
                            decoded.to_vec()
                        } else {
                            decoded
                                .chunks_exact(2)
                                .map(|frame| {
                                    ((frame[0] as i32 + frame[1] as i32) / 2)
                                        .clamp(i16::MIN as i32, i16::MAX as i32)
                                        as i16
                                })
                                .collect::<Vec<_>>()
                        };
                        if packet_count == 1 || packet_count.is_multiple_of(100) {
                            eprintln!(
                                "[live-assistant webrtc] decoded packets={} samples_24k={}",
                                packet_count,
                                mono_24k.len()
                            );
                        }
                        for speech_packet in speech_gate.push(mono_24k) {
                            emitted_packet_count = emitted_packet_count.saturating_add(1);
                            if emitted_packet_count == 1
                                || emitted_packet_count.is_multiple_of(50)
                            {
                                eprintln!(
                                    "[live-assistant webrtc] emitted speech packets={} source_packets={}",
                                    emitted_packet_count, packet_count
                                );
                            }
                            if remote_audio_tx.send(Ok(speech_packet)).is_err() {
                                return;
                            }
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

        let encoder = Encoder::new(INPUT_RATE as u32, Channels::Mono, Application::Voip)
            .context("Could not initialize GPT-Live Opus encoder")?;
        Ok((
            Self {
                peer,
                local_audio,
                _rtp_sender: rtp_sender,
                _events_channel: events_channel,
                encoder,
                pending_input: VecDeque::new(),
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

    /// Queues 24 kHz mono PCM and sends complete 20 ms Opus frames.
    pub async fn send_pcm24k(&mut self, samples: &[i16]) -> Result<()> {
        self.pending_input.extend(samples.iter().copied());
        while self.pending_input.len() >= INPUT_FRAME_SAMPLES {
            let frame_24k = self
                .pending_input
                .drain(..INPUT_FRAME_SAMPLES)
                .collect::<Vec<_>>();
            let encoded = self
                .encoder
                .encode_vec(&frame_24k, MAX_OPUS_PACKET_BYTES)
                .context("Could not encode microphone audio for GPT-Live")?;
            self.local_audio
                .write_sample(&Sample {
                    data: Bytes::from(encoded),
                    duration: Duration::from_millis(FRAME_MS as u64),
                    ..Default::default()
                })
                .await
                .context("Could not send microphone audio to GPT-Live")?;
        }
        Ok(())
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
