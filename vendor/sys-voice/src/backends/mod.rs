#[cfg(target_os = "macos")]
mod macos;

#[cfg(target_os = "ios")]
mod ios;

#[cfg(target_os = "windows")]
mod windows;

#[cfg(target_os = "linux")]
mod linux;

#[cfg(target_os = "android")]
mod android;

use crate::resampler::Resampler;
use crate::AecError;

/// Handle for sending audio to the backend for playback.
/// Audio played through this handle goes through the same engine as capture,
/// enabling AEC to cancel it from the recorded audio.
#[derive(Clone)]
pub struct BackendHandle {
    playback_tx: flume::Sender<PlaybackCommand>,
    native_sample_rate: u32,
}
pub(crate) enum PlaybackCommand {
    OneShot(Vec<f32>),
    StartStream(flume::Receiver<Vec<f32>>),
}

impl BackendHandle {
    /// Play a complete audio buffer. For streaming audio, use `start_playback_stream`.
    pub fn play_audio(&self, samples: Vec<f32>, sample_rate: u32) -> Result<(), AecError> {
        let samples = resample_oneshot(samples, sample_rate, self.native_sample_rate)?;
        self.playback_tx
            .send(PlaybackCommand::OneShot(samples))
            .map_err(|_| AecError::BackendError("playback channel closed".to_string()))
    }

    /// Start a streaming playback session. Returns a sender for audio chunks.
    /// The stream ends when the sender is dropped.
    pub fn start_playback_stream(
        &self,
        sample_rate: u32,
    ) -> Result<flume::Sender<Vec<f32>>, AecError> {
        let (user_tx, user_rx) = flume::bounded::<Vec<f32>>(64);
        let (backend_tx, backend_rx) = flume::bounded::<Vec<f32>>(64);

        spawn_resampler(user_rx, backend_tx, sample_rate, self.native_sample_rate);

        self.playback_tx
            .send(PlaybackCommand::StartStream(backend_rx))
            .map_err(|_| AecError::BackendError("playback channel closed".to_string()))?;

        Ok(user_tx)
    }
}

fn resample_oneshot(
    samples: Vec<f32>,
    source_rate: u32,
    target_rate: u32,
) -> Result<Vec<f32>, AecError> {
    if source_rate == target_rate {
        return Ok(samples);
    }
    let mut resampler = Resampler::new(source_rate, target_rate)?;
    resampler.process(&samples)
}

fn spawn_resampler(
    user_rx: flume::Receiver<Vec<f32>>,
    backend_tx: flume::Sender<Vec<f32>>,
    source_rate: u32,
    target_rate: u32,
) {
    tokio::spawn(async move {
        let mut resampler = if source_rate != target_rate {
            Resampler::new(source_rate, target_rate).ok()
        } else {
            None
        };

        while let Ok(chunk) = user_rx.recv_async().await {
            let samples = resample_chunk(&mut resampler, chunk);
            if backend_tx.send_async(samples).await.is_err() {
                break;
            }
        }
    });
}

fn resample_chunk(resampler: &mut Option<Resampler>, chunk: Vec<f32>) -> Vec<f32> {
    let Some(r) = resampler else {
        return chunk;
    };
    r.process(&chunk).unwrap_or(chunk)
}

/// Create the appropriate platform backend.
/// Spawns a capture task that owns audio resources.
/// Returns (sample_rate, buffer_size, handle). Task stops when sender disconnects.
pub(crate) fn create_backend(
    sender: flume::Sender<Vec<f32>>,
) -> Result<(u32, usize, BackendHandle), AecError> {
    let (playback_tx, playback_rx) = flume::bounded::<PlaybackCommand>(16);

    #[cfg(target_os = "macos")]
    {
        let (rate, size) = macos::create_backend(sender, playback_rx)?;
        let handle = BackendHandle {
            playback_tx,
            native_sample_rate: rate,
        };
        Ok((rate, size, handle))
    }

    #[cfg(target_os = "ios")]
    {
        let (rate, size) = ios::create_backend(sender, playback_rx)?;
        let handle = BackendHandle {
            playback_tx,
            native_sample_rate: rate,
        };
        Ok((rate, size, handle))
    }

    #[cfg(target_os = "windows")]
    {
        let (rate, size) = windows::create_backend(sender, playback_rx)?;
        let handle = BackendHandle {
            playback_tx,
            native_sample_rate: rate,
        };
        Ok((rate, size, handle))
    }

    #[cfg(target_os = "linux")]
    {
        let (rate, size) = linux::create_backend(sender, playback_rx)?;
        let handle = BackendHandle {
            playback_tx,
            native_sample_rate: rate,
        };
        Ok((rate, size, handle))
    }

    #[cfg(target_os = "android")]
    {
        let (rate, size) = android::create_backend(sender, playback_rx)?;
        let handle = BackendHandle {
            playback_tx,
            native_sample_rate: rate,
        };
        Ok((rate, size, handle))
    }

    #[cfg(not(any(
        target_os = "macos",
        target_os = "ios",
        target_os = "windows",
        target_os = "linux",
        target_os = "android"
    )))]
    {
        let _ = (sender, playback_rx);
        Err(AecError::AecNotSupported)
    }
}
