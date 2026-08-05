use crate::backends::{ControlCommand, PlaybackCommand};
use crate::AecError;
use coreaudio::audio_unit::render_callback::{self, data};
use coreaudio::audio_unit::types::IOType;
use coreaudio::audio_unit::{AudioUnit, Element, SampleFormat, Scope, StreamFormat};

use flume::{Receiver, Sender};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

// Added in macOS 14. The currently installed SDK may predate the named C
// constants, so keep the stable values from AudioUnitProperties.h locally.
const K_AU_VOICE_IO_PROPERTY_BYPASS_VOICE_PROCESSING: u32 = 2100;
const K_AU_VOICE_IO_PROPERTY_OTHER_AUDIO_DUCKING_CONFIGURATION: u32 = 2108;
/// macOS accepts values below the documented minimum of 10. A value of 1 keeps
/// the output effectively at unity while retaining the VoiceProcessingIO echo
/// reference. This avoids the audible media-volume drop caused by normal voice
/// chat ducking.
const K_AU_VOICE_IO_OTHER_AUDIO_DUCKING_LEVEL_NEAR_UNITY: u32 = 1;

#[repr(C)]
struct VoiceIoOtherAudioDuckingConfiguration {
    enable_advanced_ducking: u8,
    ducking_level: u32,
}

/// Shared buffer for playback samples
struct PlaybackBuffer {
    samples: VecDeque<f32>,
}

/// Create macOS backend. Spawns a task that owns audio resources.
/// Returns (sample_rate, buffer_size). Task stops when sender fails.
pub fn create_backend(
    public_sender: Sender<Vec<f32>>,
    playback_rx: Receiver<PlaybackCommand>,
    control_rx: Receiver<ControlCommand>,
) -> Result<(u32, usize), AecError> {
    let (callback_tx, callback_rx) = flume::bounded::<Vec<f32>>(32);

    // Create shared playback buffer for render callback
    let playback_buffer = Arc::new(Mutex::new(PlaybackBuffer {
        samples: VecDeque::with_capacity(48000), // ~1 second at 48kHz
    }));
    // Create VoiceProcessingIO audio unit - this enables OS-level AEC
    // VoiceProcessingIO automatically monitors system output for echo reference
    let mut audio_unit = AudioUnit::new(IOType::VoiceProcessingIO).map_err(|e| {
        AecError::BackendError(format!("failed to create VoiceProcessingIO: {e:?}"))
    })?;

    // coreaudio-rs may auto-initialize; must uninitialize before configuring properties
    let _ = audio_unit.uninitialize();

    // Keep voice processing explicitly enabled. This is also the property we
    // toggle for the one-second Command-key system-audio passthrough gesture.
    let bypass_voice_processing: u32 = 0;
    audio_unit
        .set_property(
            K_AU_VOICE_IO_PROPERTY_BYPASS_VOICE_PROCESSING,
            Scope::Global,
            Element::Output,
            Some(&bypass_voice_processing),
        )
        .map_err(|e| {
            AecError::BackendError(format!("failed to enable voice processing: {e:?}"))
        })?;

    // Advanced mode limits ducking to detected speech. Use a near-unity level
    // so cancellation does not make ordinary system playback noticeably quieter.
    let ducking = VoiceIoOtherAudioDuckingConfiguration {
        enable_advanced_ducking: 1,
        ducking_level: K_AU_VOICE_IO_OTHER_AUDIO_DUCKING_LEVEL_NEAR_UNITY,
    };
    match audio_unit.set_property(
        K_AU_VOICE_IO_PROPERTY_OTHER_AUDIO_DUCKING_CONFIGURATION,
        Scope::Global,
        Element::Output,
        Some(&ducking),
    ) {
        Ok(()) => tracing::info!(
            "VoiceProcessingIO advanced ducking enabled at near-unity level"
        ),
        Err(error) => tracing::warn!(
            "VoiceProcessingIO near-unity ducking is unavailable; using system default: {error:?}"
        ),
    }

    let enable_input: u32 = 1;
    audio_unit
        .set_property(
            coreaudio::sys::kAudioOutputUnitProperty_EnableIO,
            Scope::Input,
            Element::Input,
            Some(&enable_input),
        )
        .map_err(|e| AecError::BackendError(format!("failed to enable input: {e:?}")))?;

    // let enable_output: u32 = 1;
    // audio_unit
    //     .set_property(
    //         coreaudio::sys::kAudioOutputUnitProperty_EnableIO,
    //         Scope::Output,
    //         Element::Output,
    //         Some(&enable_output),
    //     )
    //     .map_err(|e| AecError::BackendError(format!("failed to enable output: {e:?}")))?;

    // Query native format - VoiceProcessingIO has strict requirements
    let native_format = audio_unit
        .stream_format(Scope::Output, Element::Input)
        .map_err(|e| AecError::BackendError(format!("failed to get native format: {e:?}")))?;

    // Use native sample rate but request f32 mono non-interleaved (canonical for VPIO)
    let stream_format = StreamFormat {
        sample_rate: native_format.sample_rate,
        sample_format: SampleFormat::F32,
        flags: coreaudio::audio_unit::audio_format::LinearPcmFlags::IS_FLOAT
            | coreaudio::audio_unit::audio_format::LinearPcmFlags::IS_PACKED
            | coreaudio::audio_unit::audio_format::LinearPcmFlags::IS_NON_INTERLEAVED,
        channels: 1,
    };

    audio_unit
        .set_stream_format(stream_format, Scope::Output, Element::Input)
        .map_err(|e| AecError::BackendError(format!("failed to set input stream format: {e:?}")))?;

    // Also set stream format for output element (for render callback)
    audio_unit
        .set_stream_format(stream_format, Scope::Input, Element::Output)
        .map_err(|e| {
            AecError::BackendError(format!("failed to set output stream format: {e:?}"))
        })?;

    let native_rate = native_format.sample_rate as u32;

    audio_unit
        .set_input_callback(
            move |args: render_callback::Args<data::NonInterleaved<f32>>| {
                let buffer = args.data.channels().next().unwrap();
                let _ = callback_tx.try_send(buffer.to_vec());
                Ok(())
            },
        )
        .map_err(|e| AecError::BackendError(format!("failed to set input callback: {e:?}")))?;

    // Set render callback for playback output - VoiceProcessingIO AEC uses this as echo reference
    let buffer_for_render = playback_buffer.clone();
    audio_unit
        .set_render_callback(
            move |mut args: render_callback::Args<data::NonInterleaved<f32>>| {
                let output_buffer = args.data.channels_mut().next().unwrap();
                // Use try_lock to avoid blocking in audio callback
                if let Ok(mut buffer) = buffer_for_render.try_lock() {
                    for sample in output_buffer.iter_mut() {
                        *sample = buffer.samples.pop_front().unwrap_or(0.0);
                    }
                } else {
                    for sample in output_buffer.iter_mut() {
                        *sample = 0.0;
                    }
                }
                Ok(())
            },
        )
        .map_err(|e| AecError::BackendError(format!("failed to set render callback: {e:?}")))?;

    audio_unit
        .initialize()
        .map_err(|e| AecError::BackendError(format!("failed to initialize: {e:?}")))?;

    audio_unit
        .start()
        .map_err(|e| AecError::BackendError(format!("failed to start: {e:?}")))?;

    // Query buffer size from audio unit (frames per slice)
    let buffer_size: u32 = audio_unit
        .get_property(
            coreaudio::sys::kAudioUnitProperty_MaximumFramesPerSlice,
            Scope::Global,
            Element::Output,
        )
        .unwrap_or(512);

    let buffer_for_playback = playback_buffer.clone();
    tokio::spawn(async move {
        while let Ok(command) = playback_rx.recv_async().await {
            match command {
                PlaybackCommand::OneShot(samples) => {
                    let Ok(mut buf) = buffer_for_playback.lock() else {
                        continue;
                    };
                    buf.samples.extend(samples);
                }
                PlaybackCommand::StartStream(chunk_rx) => {
                    let buf = buffer_for_playback.clone();
                    tokio::spawn(async move {
                        while let Ok(samples) = chunk_rx.recv_async().await {
                            let Ok(mut b) = buf.lock() else {
                                break;
                            };
                            b.samples.extend(samples);
                        }
                    });
                }
            }
        }
    });

    // Spawn task that owns audio_unit, applies live bypass changes, and forwards
    // capture. Keeping the same unit alive avoids device restarts when the user
    // holds or releases Command.
    tokio::spawn(async move {
        let mut audio_unit = audio_unit; // Hold for RAII, Drop stops audio
        let mut control_open = true;
        loop {
            if control_open {
                tokio::select! {
                    samples = callback_rx.recv_async() => {
                        let Ok(samples) = samples else { break; };
                        if public_sender.send_async(samples).await.is_err() {
                            break;
                        }
                    }
                    command = control_rx.recv_async() => {
                        match command {
                            Ok(ControlCommand::SetVoiceProcessingBypassed(bypassed)) => {
                                let value: u32 = u32::from(bypassed);
                                if let Err(error) = audio_unit.set_property(
                                    K_AU_VOICE_IO_PROPERTY_BYPASS_VOICE_PROCESSING,
                                    Scope::Global,
                                    Element::Output,
                                    Some(&value),
                                ) {
                                    tracing::warn!(
                                        "Could not set VoiceProcessingIO bypass={bypassed}: {error:?}"
                                    );
                                } else {
                                    tracing::info!(
                                        "VoiceProcessingIO system-audio passthrough={bypassed}"
                                    );
                                }
                            }
                            Err(_) => control_open = false,
                        }
                    }
                }
            } else {
                let Ok(samples) = callback_rx.recv_async().await else { break; };
                if public_sender.send_async(samples).await.is_err() {
                    break;
                }
            }
        }
    });

    Ok((native_rate, buffer_size as usize))
}
