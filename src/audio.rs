use crate::{realtime::Command, resample::StreamingResampler};
use anyhow::{Context, Result};
use cpal::{
    SampleFormat, Stream, StreamConfig,
    traits::{DeviceTrait, HostTrait, StreamTrait},
};
use std::{
    collections::VecDeque,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, AtomicU32, Ordering},
    },
    thread::{self, JoinHandle},
    time::Instant,
};
use sys_voice::{AecConfig, CaptureHandle, Channels};
use tokio::sync::mpsc::UnboundedSender;

const PRE_ROLL_SAMPLES: usize = 24_000 * 3;
/// Capture at 48 kHz through the OS AEC path, then resample to 24 kHz for Realtime.
const AEC_CAPTURE_RATE: u32 = 48_000;
const PLAYBACK_RATE: u32 = 24_000;
/// Roughly -40 dBFS: high enough to reject room noise after VoiceProcessingIO,
/// while still counting normal close-mic speech.
const LOUD_SPEECH_RMS: f32 = 0.01;
const PLAYBACK_PREBUFFER_MS: u32 = 100;

#[derive(Default)]
struct TurnBuffer {
    pre_roll: VecDeque<i16>,
    current: Vec<i16>,
    continuous_loud_samples: usize,
    in_speech: bool,
}

/// Shared tokio runtime for sys-voice (it spawns tasks during capture setup).
fn audio_runtime() -> &'static tokio::runtime::Runtime {
    static RT: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RT.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .thread_name("live-assistant-audio")
            .build()
            .expect("audio runtime")
    })
}

/// Microphone with OS-level acoustic echo cancellation.
///
/// On macOS this uses Core Audio **VoiceProcessingIO**, which cancels system /
/// speaker output from the mic. The mic stays open while the assistant talks;
/// assistant audio should not re-trigger VAD as user speech.
pub struct Microphone {
    stop: Arc<AtomicBool>,
    _reader: JoinHandle<()>,
    turn: Arc<Mutex<TurnBuffer>>,
    level_bits: Arc<AtomicU32>,
}

impl Microphone {
    pub fn start(sender: UnboundedSender<Command>) -> Result<Self> {
        let _enter = audio_runtime().enter();
        let capture = CaptureHandle::new(AecConfig {
            sample_rate: AEC_CAPTURE_RATE,
            channels: Channels::Mono,
        })
        .map_err(|error| {
            anyhow::anyhow!(
                "Could not open AEC microphone ({error}). On macOS, enable Microphone \
                access for Live Assistant in System Settings → Privacy & Security."
            )
        })?;

        eprintln!(
            "[live-assistant mic] rate={} channels=1 mode=voice-processing-aec full_duplex=true automatic_mute=false",
            AEC_CAPTURE_RATE
        );

        let turn = Arc::new(Mutex::new(TurnBuffer::default()));
        let level_bits = Arc::new(AtomicU32::new(0));
        let stop = Arc::new(AtomicBool::new(false));

        let turn_thread = turn.clone();
        let level_thread = level_bits.clone();
        let stop_thread = stop.clone();
        let reader = thread::Builder::new()
            .name("aec-mic-reader".into())
            .spawn(move || {
                // Owns CaptureHandle so Drop stops VoiceProcessingIO when the thread ends.
                let capture = capture;
                let mut pending_capture_sample = None;
                while !stop_thread.load(Ordering::Relaxed) {
                    match capture.recv_blocking() {
                        Some(Ok(samples_f32)) => {
                            let rms = if samples_f32.is_empty() {
                                0.0
                            } else {
                                (samples_f32.iter().map(|v| v * v).sum::<f32>()
                                    / samples_f32.len() as f32)
                                    .sqrt()
                            };
                            level_thread.store(rms.to_bits(), Ordering::Relaxed);

                            let pcm = downsample_capture_to_24k(
                                &samples_f32,
                                &mut pending_capture_sample,
                            );
                            if pcm.is_empty() {
                                continue;
                            }
                            if let Ok(mut buffer) = turn_thread.lock() {
                                // VoiceProcessingIO already performs acoustic echo cancellation.
                                // Keep the microphone fully duplex while assistant audio plays so
                                // realtime translation and overlapping user speech reach GPT-Live.
                                if rms >= LOUD_SPEECH_RMS {
                                    buffer.continuous_loud_samples =
                                        buffer.continuous_loud_samples.saturating_add(pcm.len());
                                } else {
                                    buffer.continuous_loud_samples = 0;
                                }
                                if buffer.in_speech {
                                    buffer.current.extend_from_slice(&pcm);
                                } else {
                                    for sample in &pcm {
                                        buffer.pre_roll.push_back(*sample);
                                        if buffer.pre_roll.len() > PRE_ROLL_SAMPLES {
                                            buffer.pre_roll.pop_front();
                                        }
                                    }
                                }
                            }
                            if sender.send(Command::AudioChunk(pcm)).is_err() {
                                break;
                            }
                        }
                        Some(Err(error)) => {
                            eprintln!("AEC microphone error: {error}");
                            break;
                        }
                        None => break,
                    }
                }
            })
            .context("Could not start the AEC microphone reader thread")?;

        Ok(Self {
            stop,
            _reader: reader,
            turn,
            level_bits,
        })
    }

    pub fn begin_turn(&self) {
        if let Ok(mut buffer) = self.turn.lock() {
            buffer.current = buffer.pre_roll.iter().copied().collect();
            buffer.pre_roll.clear();
            buffer.in_speech = true;
        }
    }

    pub fn finish_turn(&self) -> Vec<i16> {
        if let Ok(mut buffer) = self.turn.lock() {
            buffer.in_speech = false;
            buffer.continuous_loud_samples = 0;
            return std::mem::take(&mut buffer.current);
        }
        Vec::new()
    }

    pub fn reset_turn(&self) {
        if let Ok(mut buffer) = self.turn.lock() {
            buffer.in_speech = false;
            buffer.current.clear();
            buffer.pre_roll.clear();
            buffer.continuous_loud_samples = 0;
        }
    }

    pub fn in_speech(&self) -> bool {
        self.turn
            .lock()
            .map(|buffer| buffer.in_speech)
            .unwrap_or(false)
    }

    pub fn level(&self) -> f32 {
        f32::from_bits(self.level_bits.load(Ordering::Relaxed))
    }

    pub fn loud_speech_samples(&self) -> usize {
        self.turn
            .lock()
            .map(|buffer| buffer.continuous_loud_samples)
            .unwrap_or_default()
    }
}

impl Drop for Microphone {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        // CaptureHandle is owned by the reader thread; stopping the loop drops it.
        // recv_blocking may wait until the next audio buffer — acceptable on stop.
    }
}

/// Local playback for assistant audio and message replay.
///
/// Incoming 24 kHz PCM is converted once with a band-limited FFT resampler to
/// the output device's native rate. The device callback then performs only a
/// transparent queue read: no gain, nonlinear limiting, or second resampling.
pub struct Speaker {
    _stream: Stream,
    playback: Arc<Mutex<PlaybackBuffer>>,
    output_resampler: StreamingResampler,
    assistant_started_at: Option<Instant>,
    assistant_received_samples: usize,
    assistant_logged_seconds: usize,
}

struct PlaybackBuffer {
    samples_native: VecDeque<f32>,
    playing: bool,
    start_threshold_samples: usize,
}

impl Speaker {
    pub fn new() -> Result<Self> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .context("No audio output device is available")?;
        let supported = device
            .default_output_config()
            .context("Could not read the default audio output format")?;
        let sample_format = supported.sample_format();
        let config = supported.config();
        let playback = Arc::new(Mutex::new(PlaybackBuffer {
            samples_native: VecDeque::new(),
            playing: false,
            start_threshold_samples: (config.sample_rate.0 as usize
                * PLAYBACK_PREBUFFER_MS as usize)
                / 1_000,
        }));
        let stream = build_output_stream(&device, &config, sample_format, playback.clone())?;
        stream.play().context("Could not start the audio output")?;
        let output_resampler = StreamingResampler::new(PLAYBACK_RATE, config.sample_rate.0)?;
        eprintln!(
            "[live-assistant speaker] device={:?} rate={} channels={} format={:?} gain=unity prebuffer_ms={}",
            device.name().ok(),
            config.sample_rate.0,
            config.channels,
            sample_format,
            PLAYBACK_PREBUFFER_MS
        );

        Ok(Self {
            _stream: stream,
            playback,
            output_resampler,
            assistant_started_at: None,
            assistant_received_samples: 0,
            assistant_logged_seconds: 0,
        })
    }

    pub fn begin_assistant_response(&mut self) {
        // Reset conversion/filter history without clearing already queued audio.
        // Separate backend responses should never bleed resampler state together.
        self.assistant_started_at = None;
        self.assistant_received_samples = 0;
        self.assistant_logged_seconds = 0;
        self.output_resampler.reset();
    }

    pub fn append_assistant(&mut self, samples: Vec<i16>) {
        if samples.is_empty() {
            return;
        }
        if self.assistant_started_at.is_none() {
            self.output_resampler.reset();
        }
        self.assistant_started_at.get_or_insert_with(Instant::now);
        self.assistant_received_samples += samples.len();
        let seconds = self.assistant_received_samples / PLAYBACK_RATE as usize;
        if self.assistant_received_samples == samples.len()
            || seconds > self.assistant_logged_seconds
        {
            self.assistant_logged_seconds = seconds;
            eprintln!(
                "[live-assistant speaker] received_seconds={:.2}",
                self.assistant_received_samples as f64 / PLAYBACK_RATE as f64
            );
        }
        self.append(samples);
    }

    fn append(&mut self, samples: Vec<i16>) {
        let normalized = samples.into_iter().map(pcm_i16_to_f32).collect::<Vec<_>>();
        match self.output_resampler.process(&normalized) {
            Ok(native) => {
                if let Ok(mut playback) = self.playback.lock() {
                    playback.samples_native.extend(native);
                }
            }
            Err(error) => eprintln!("Speaker resampling error: {error:#}"),
        }
    }

    pub fn play_clip(&mut self, samples: &[i16]) -> Result<()> {
        self.reset_assistant_clock();
        let normalized = samples
            .iter()
            .copied()
            .map(pcm_i16_to_f32)
            .collect::<Vec<_>>();
        let native = self.output_resampler.process_complete(&normalized)?;
        self.replace_native_output(&native);
        Ok(())
    }

    pub fn clear(&mut self) -> Result<()> {
        self.reset_assistant_clock();
        self.replace_native_output(&[]);
        Ok(())
    }

    /// Stop assistant playback and return the approximate amount heard.
    pub fn interrupt_assistant(&mut self) -> Result<Option<u32>> {
        let played_ms = self.assistant_played_ms();
        self.clear()?;
        Ok(played_ms)
    }

    pub fn assistant_is_playing(&self) -> bool {
        self.playback
            .lock()
            .map(|playback| !playback.samples_native.is_empty())
            .unwrap_or(false)
    }

    fn assistant_played_ms(&self) -> Option<u32> {
        let started = self.assistant_started_at?;
        let total_ms =
            (self.assistant_received_samples as u64 * 1_000).div_ceil(PLAYBACK_RATE as u64);
        let elapsed_ms = started.elapsed().as_millis() as u64;
        if elapsed_ms >= total_ms {
            return None;
        }
        Some(elapsed_ms.min(u32::MAX as u64) as u32)
    }

    fn reset_assistant_clock(&mut self) {
        self.assistant_started_at = None;
        self.assistant_received_samples = 0;
        self.assistant_logged_seconds = 0;
        self.output_resampler.reset();
    }

    fn replace_native_output(&self, samples: &[f32]) {
        if let Ok(mut playback) = self.playback.lock() {
            playback.samples_native.clear();
            playback.samples_native.extend(samples.iter().copied());
            playback.playing =
                !samples.is_empty() && samples.len() < playback.start_threshold_samples;
        }
    }
}

fn pcm_i16_to_f32(sample: i16) -> f32 {
    sample as f32 / 32_768.0
}

fn build_output_stream(
    device: &cpal::Device,
    config: &StreamConfig,
    sample_format: SampleFormat,
    playback: Arc<Mutex<PlaybackBuffer>>,
) -> Result<Stream> {
    let channels = config.channels as usize;
    let error_callback = |error| eprintln!("Audio output error: {error}");

    let stream = match sample_format {
        SampleFormat::F32 => {
            let playback = playback.clone();
            device.build_output_stream(
                config,
                move |output: &mut [f32], _| write_output_f32(output, channels, &playback),
                error_callback,
                None,
            )
        }
        SampleFormat::I16 => {
            let playback = playback.clone();
            device.build_output_stream(
                config,
                move |output: &mut [i16], _| write_output_i16(output, channels, &playback),
                error_callback,
                None,
            )
        }
        SampleFormat::U16 => device.build_output_stream(
            config,
            move |output: &mut [u16], _| write_output_u16(output, channels, &playback),
            error_callback,
            None,
        ),
        format => anyhow::bail!("Unsupported audio output format: {format:?}"),
    }
    .context("Could not open the native audio output")?;
    Ok(stream)
}

fn write_output_f32(output: &mut [f32], channels: usize, playback: &Mutex<PlaybackBuffer>) {
    let Ok(mut playback) = playback.try_lock() else {
        output.fill(0.0);
        return;
    };
    for frame in output.chunks_mut(channels) {
        frame.fill(next_native_sample(&mut playback));
    }
}

fn write_output_i16(output: &mut [i16], channels: usize, playback: &Mutex<PlaybackBuffer>) {
    let Ok(mut playback) = playback.try_lock() else {
        output.fill(0);
        return;
    };
    for frame in output.chunks_mut(channels) {
        let value = next_native_sample(&mut playback).clamp(-1.0, 1.0);
        frame.fill((value * i16::MAX as f32).round() as i16);
    }
}

fn write_output_u16(output: &mut [u16], channels: usize, playback: &Mutex<PlaybackBuffer>) {
    let Ok(mut playback) = playback.try_lock() else {
        output.fill(u16::MAX / 2);
        return;
    };
    for frame in output.chunks_mut(channels) {
        let value = next_native_sample(&mut playback).clamp(-1.0, 1.0);
        frame.fill(((value * 0.5 + 0.5) * u16::MAX as f32).round() as u16);
    }
}

fn next_native_sample(playback: &mut PlaybackBuffer) -> f32 {
    if !playback.playing {
        if playback.samples_native.len() < playback.start_threshold_samples {
            return 0.0;
        }
        playback.playing = true;
    }
    let sample = playback.samples_native.pop_front().unwrap_or(0.0);
    if playback.samples_native.is_empty() {
        playback.playing = false;
    }
    sample
}

fn downsample_capture_to_24k(input: &[f32], pending: &mut Option<f32>) -> Vec<i16> {
    debug_assert_eq!(AEC_CAPTURE_RATE, PLAYBACK_RATE * 2);
    let mut output = Vec::with_capacity((input.len() + usize::from(pending.is_some())) / 2);
    let mut index = 0;

    if let Some(first) = pending.take()
        && let Some(&second) = input.first()
    {
        output.push(f32_to_pcm_i16((first + second) * 0.5));
        index = 1;
    }

    while index + 1 < input.len() {
        output.push(f32_to_pcm_i16((input[index] + input[index + 1]) * 0.5));
        index += 2;
    }
    if index < input.len() {
        *pending = Some(input[index]);
    }
    output
}

fn f32_to_pcm_i16(sample: f32) -> i16 {
    (sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn microphone_message_preroll_is_three_seconds() {
        assert_eq!(PRE_ROLL_SAMPLES, 24_000 * 3);
    }

    #[test]
    fn speaker_waits_for_jitter_prebuffer_without_modifying_samples() {
        let mut playback = PlaybackBuffer {
            samples_native: VecDeque::from([0.25, -0.5]),
            playing: false,
            start_threshold_samples: 3,
        };
        assert_eq!(next_native_sample(&mut playback), 0.0);
        playback.samples_native.push_back(0.75);
        assert_eq!(next_native_sample(&mut playback), 0.25);
        assert_eq!(next_native_sample(&mut playback), -0.5);
        assert_eq!(next_native_sample(&mut playback), 0.75);
    }

    #[test]
    fn speaker_conversion_is_unity_gain() {
        assert!((pcm_i16_to_f32(16_384) - 0.5).abs() < 0.0001);
        assert_eq!(f32_to_pcm_i16(0.5), 16_384);
    }
}
