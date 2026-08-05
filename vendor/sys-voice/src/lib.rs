mod backends;
mod resampler;

use resampler::Resampler;
use thiserror::Error;

/// Output channel configuration
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Channels {
    #[default]
    Mono,
    Stereo,
}
#[derive(Debug, Clone)]
pub struct AecConfig {
    /// Target sample rate in Hz (typically 48000)
    pub sample_rate: u32,
    /// Output channels (stereo = duplicated mono from AEC)
    pub channels: Channels,
}

impl Default for AecConfig {
    fn default() -> Self {
        Self {
            sample_rate: 48000,
            channels: Channels::Mono,
        }
    }
}

#[derive(Debug, Error)]
pub enum AecError {
    #[error("audio device unavailable")]
    DeviceUnavailable,

    #[error("microphone permission denied")]
    PermissionDenied,

    #[error("AEC not supported on this device")]
    AecNotSupported,

    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("backend error: {0}")]
    BackendError(String),
}

/// Handle for streaming audio chunks to the playback buffer.
/// Drop the handle to signal end of stream.
pub struct PlaybackStreamHandle {
    chunk_tx: flume::Sender<Vec<f32>>,
}

impl PlaybackStreamHandle {
    /// Send a chunk of audio samples to the playback buffer.
    pub fn send(&self, chunk: Vec<f32>) -> Result<(), AecError> {
        self.chunk_tx
            .send(chunk)
            .map_err(|_| AecError::BackendError("playback stream closed".to_string()))
    }

    /// Send a chunk of audio samples asynchronously.
    pub async fn send_async(&self, chunk: Vec<f32>) -> Result<(), AecError> {
        self.chunk_tx
            .send_async(chunk)
            .await
            .map_err(|_| AecError::BackendError("playback stream closed".to_string()))
    }
}

/// Handle for receiving AEC-processed audio samples.
/// Capture stops automatically when dropped (channel disconnect stops backend).
pub struct CaptureHandle {
    receiver: flume::Receiver<Result<Vec<f32>, AecError>>,
    backend: backends::BackendHandle,
    sample_rate: u32,
}

#[derive(Clone)]
pub struct CaptureControl {
    backend: backends::BackendHandle,
}

impl CaptureControl {
    /// Bypass voice processing while preserving microphone capture.
    /// On macOS this temporarily lets speaker/system audio pass through the mic.
    pub fn set_voice_processing_bypassed(&self, bypassed: bool) -> Result<(), AecError> {
        self.backend.set_voice_processing_bypassed(bypassed)
    }
}

impl CaptureHandle {
    /// Create and start a new AEC capture stream.
    /// Audio samples are received via the async recv() or blocking recv_blocking() methods.
    pub fn new(config: AecConfig) -> Result<Self, AecError> {
        if config.sample_rate == 0 {
            return Err(AecError::InvalidConfig(
                "sample_rate must be non-zero".to_string(),
            ));
        }

        let (backend_tx, backend_rx) = flume::bounded::<Vec<f32>>(32);
        let (native_rate, _buffer_size, backend_handle) = backends::create_backend(backend_tx)?;

        let (public_tx, public_rx) = flume::bounded::<Result<Vec<f32>, AecError>>(32);
        let target_rate = config.sample_rate;
        let target_channels = config.channels;

        let needs_stereo = target_channels == Channels::Stereo;
        let needs_resampling = native_rate != target_rate;

        let resampler = if needs_resampling {
            Some(
                Resampler::new(native_rate, target_rate)
                    .map_err(|e| AecError::BackendError(format!("resampler init: {e:?}")))?,
            )
        } else {
            None
        };

        tokio::spawn(async move {
            let mut resampler = resampler;

            while let Ok(samples) = backend_rx.recv_async().await {
                let processed = match process_audio_chunk(samples, &mut resampler, needs_stereo) {
                    Ok(p) => p,
                    Err(e) => {
                        let _ = public_tx.send_async(Err(AecError::BackendError(e))).await;
                        break;
                    }
                };
                if public_tx.send_async(Ok(processed)).await.is_err() {
                    break;
                }
            }
        });

        Ok(Self {
            receiver: public_rx,
            backend: backend_handle,
            sample_rate: target_rate,
        })
    }

    /// Return a lightweight control handle that remains usable while capture is
    /// owned by another thread.
    pub fn control_handle(&self) -> CaptureControl {
        CaptureControl {
            backend: self.backend.clone(),
        }
    }

    /// Receive audio samples asynchronously.
    /// Returns None when the capture stream is closed.
    pub async fn recv(&self) -> Option<Result<Vec<f32>, AecError>> {
        self.receiver.recv_async().await.ok()
    }

    /// Receive audio samples, blocking the current thread.
    /// Returns None when the capture stream is closed.
    pub fn recv_blocking(&self) -> Option<Result<Vec<f32>, AecError>> {
        self.receiver.recv().ok()
    }

    /// Try to receive audio samples without blocking.
    /// Returns None if no samples are available or stream is closed.
    pub fn try_recv(&self) -> Option<Result<Vec<f32>, AecError>> {
        self.receiver.try_recv().ok()
    }

    /// Get the actual sample rate being used by the backend.
    /// May differ from requested rate if resampling is active.
    pub fn native_sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Play audio through the same engine used for capture.
    /// This enables AEC to cancel the played audio from the recording.
    /// Audio is played at the specified sample rate.
    pub fn play_audio(&self, samples: Vec<f32>, sample_rate: u32) -> Result<(), AecError> {
        self.backend.play_audio(samples, sample_rate)
    }

    /// Start a streaming playback session.
    /// Returns a handle for sending audio chunks incrementally.
    /// The stream ends when the handle is dropped.
    pub fn start_playback_stream(
        &self,
        sample_rate: u32,
    ) -> Result<PlaybackStreamHandle, AecError> {
        let chunk_tx = self.backend.start_playback_stream(sample_rate)?;
        Ok(PlaybackStreamHandle { chunk_tx })
    }
}

// Drop on CaptureHandle drops backend, which stops capture via RAII

fn process_audio_chunk(
    samples: Vec<f32>,
    resampler: &mut Option<Resampler>,
    needs_stereo: bool,
) -> Result<Vec<f32>, String> {
    let samples = if let Some(r) = resampler {
        r.process(&samples)
            .map_err(|e| format!("resample: {e:?}"))?
    } else {
        samples
    };

    if needs_stereo {
        Ok(samples.iter().flat_map(|&s| [s, s]).collect())
    } else {
        Ok(samples)
    }
}
