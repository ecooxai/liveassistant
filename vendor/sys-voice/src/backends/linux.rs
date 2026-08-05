use libpulse_binding::sample::{Format, Spec};
use libpulse_binding::stream::Direction;
use libpulse_simple_binding::Simple;

use crate::backends::PlaybackCommand;
use crate::AecError;

const SAMPLE_RATE: u32 = 48000;
const BUFFER_FRAMES: usize = 480; // 10ms at 48kHz

/// Create PulseAudio capture backend.
/// Spawns a blocking task that owns all PulseAudio resources.
/// Returns (sample_rate, buffer_size).
pub fn create_backend(
    sender: flume::Sender<Vec<f32>>,
    playback_rx: flume::Receiver<PlaybackCommand>,
) -> Result<(u32, usize), AecError> {
    // Verify PulseAudio connection works before spawning task
    let simple = create_simple_stream(Direction::Record, "AEC Capture")?;

    tokio::task::spawn_blocking(move || {
        let mut buffer = vec![0.0f32; BUFFER_FRAMES];

        loop {
            let byte_slice = unsafe {
                std::slice::from_raw_parts_mut(
                    buffer.as_mut_ptr() as *mut u8,
                    buffer.len() * std::mem::size_of::<f32>(),
                )
            };

            if simple.read(byte_slice).is_err() {
                break;
            }

            // When receiver is dropped, send fails and we exit
            if sender.send(buffer.clone()).is_err() {
                break;
            }
        }
    });

    // Spawn playback task
    tokio::task::spawn_blocking(move || {
        let _ = run_playback(playback_rx);
    });

    Ok((SAMPLE_RATE, BUFFER_FRAMES))
}
fn run_playback(playback_rx: flume::Receiver<PlaybackCommand>) -> Result<(), AecError> {
    let playback_simple = create_simple_stream(Direction::Playback, "AEC Playback")?;

    while let Ok(command) = playback_rx.recv() {
        match command {
            PlaybackCommand::OneShot(samples) => {
                write_samples(&playback_simple, &samples)?;
            }
            PlaybackCommand::StartStream(chunk_rx) => {
                while let Ok(samples) = chunk_rx.recv() {
                    write_samples(&playback_simple, &samples)?;
                }
            }
            PlaybackCommand::Clear => {}
        }
    }

    Ok(())
}

fn write_samples(simple: &Simple, samples: &[f32]) -> Result<(), AecError> {
    let byte_slice = unsafe {
        std::slice::from_raw_parts(
            samples.as_ptr() as *const u8,
            samples.len() * std::mem::size_of::<f32>(),
        )
    };

    simple
        .write(byte_slice)
        .map_err(|e| AecError::BackendError(format!("PulseAudio write error: {e:?}")))
}

fn create_simple_stream(direction: Direction, description: &str) -> Result<Simple, AecError> {
    let spec = Spec {
        format: Format::F32le,
        channels: 1,
        rate: SAMPLE_RATE,
    };

    if !spec.is_valid() {
        return Err(AecError::InvalidConfig(
            "Invalid PulseAudio sample spec".into(),
        ));
    }

    Simple::new(
        None,
        "sys-voice",
        direction,
        None,
        description,
        &spec,
        None,
        None,
    )
    .map_err(|e| AecError::BackendError(format!("PulseAudio error: {e:?}")))
}
