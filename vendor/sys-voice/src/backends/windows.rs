use crate::backends::PlaybackCommand;
use crate::AecError;

use wasapi::{
    initialize_mta, DeviceEnumerator, Direction, SampleType, ShareMode, StreamMode, WaveFormat,
};

/// Create WASAPI capture backend with AEC.
/// Spawns a blocking task that owns all WASAPI resources.
/// Returns (sample_rate, buffer_size) queried from the actual device format.
pub fn create_backend(
    sender: flume::Sender<Vec<f32>>,
    playback_rx: flume::Receiver<PlaybackCommand>,
) -> Result<(u32, usize), AecError> {
    // COM must be initialized for WASAPI
    let hr = initialize_mta();
    if hr.0 != 0 {
        return Err(AecError::BackendError(format!("COM init failed: {hr:?}")));
    }

    // Verify devices are available before spawning task
    let enumerator = DeviceEnumerator::new()
        .map_err(|e| AecError::BackendError(format!("DeviceEnumerator::new: {e:?}")))?;
    enumerator
        .get_default_device(&Direction::Capture)
        .map_err(|_| AecError::DeviceUnavailable)?;
    enumerator
        .get_default_device(&Direction::Render)
        .map_err(|_| AecError::DeviceUnavailable)?;

    let (meta_tx, meta_rx) = flume::bounded::<Result<(u32, usize), AecError>>(1);

    tokio::task::spawn_blocking(move || {
        if let Err(e) = capture_loop(sender, meta_tx.clone()) {
            let _ = meta_tx.send(Err(e));
        }
    });

    // Spawn playback task to handle outgoing audio
    tokio::task::spawn_blocking(move || {
        if let Err(e) = playback_loop(playback_rx) {
            tracing::error!("Playback loop error: {e:?}");
        }
    });

    // Wait for metadata from the capture thread
    meta_rx.recv().map_err(|_| {
        AecError::BackendError("capture thread died before sending metadata".to_string())
    })?
}

fn capture_loop(
    sender: flume::Sender<Vec<f32>>,
    meta_tx: flume::Sender<Result<(u32, usize), AecError>>,
) -> Result<(), AecError> {
    // Re-initialize COM on this thread
    let hr = initialize_mta();
    if hr.0 != 0 {
        return Err(AecError::BackendError(format!("COM init failed: {hr:?}")));
    }

    let enumerator = DeviceEnumerator::new()
        .map_err(|e| AecError::BackendError(format!("DeviceEnumerator::new: {e:?}")))?;
    let capture_device = enumerator
        .get_default_device(&Direction::Capture)
        .map_err(|_| AecError::DeviceUnavailable)?;
    let render_device = enumerator
        .get_default_device(&Direction::Render)
        .map_err(|_| AecError::DeviceUnavailable)?;

    let desired_format = WaveFormat::new(32, 32, &SampleType::Float, 48000, 1, None);

    let mut audio_client = capture_device
        .get_iaudioclient()
        .map_err(|e| AecError::BackendError(format!("get_iaudioclient: {e:?}")))?;

    let capture_format = match audio_client.is_supported(&desired_format, &ShareMode::Shared) {
        Ok(None) => desired_format,
        Ok(Some(suggested)) => suggested,
        Err(_) => audio_client
            .get_mixformat()
            .map_err(|e| AecError::BackendError(format!("get_mixformat: {e:?}")))?,
    };

    let stream_mode = StreamMode::EventsShared {
        autoconvert: true,
        buffer_duration_hns: 200_000,
    };
    audio_client
        .initialize_client(&capture_format, &Direction::Capture, &stream_mode)
        .map_err(|e| AecError::BackendError(format!("initialize_client: {e:?}")))?;

    if let Ok(aec_control) = audio_client.get_aec_control() {
        if let Ok(render_id) = render_device.get_id() {
            let _ = aec_control.set_echo_cancellation_render_endpoint(Some(render_id));
        }
    }

    let capture_client = audio_client
        .get_audiocaptureclient()
        .map_err(|e| AecError::BackendError(format!("get_audiocaptureclient: {e:?}")))?;

    let event_handle = audio_client
        .set_get_eventhandle()
        .map_err(|e| AecError::BackendError(format!("set_get_eventhandle: {e:?}")))?;

    audio_client
        .start_stream()
        .map_err(|e| AecError::BackendError(format!("start_stream: {e:?}")))?;

    let block_align = capture_format.get_blockalign() as usize;
    let native_channels = capture_format.get_nchannels() as usize;
    let bits = capture_format.get_bitspersample();
    let is_float = matches!(capture_format.get_subformat(), Ok(SampleType::Float));
    let native_sample_rate = capture_format.get_samplespersec();

    let device_buffer_frames = audio_client
        .get_buffer_size()
        .map_err(|e| AecError::BackendError(format!("get_buffer_size: {e:?}")))?;

    let _ = meta_tx.send(Ok((native_sample_rate, device_buffer_frames as usize)));

    let buffer_size = (device_buffer_frames as usize) * block_align;
    let mut buffer = vec![0u8; buffer_size];

    loop {
        // Wait for event with timeout. Timeout is normal - continue waiting for data.
        let _ = event_handle.wait_for_event(100);

        let (frames_read, _buffer_info) = match capture_client.read_from_device(&mut buffer) {
            Ok(result) => result,
            Err(_) => continue, // No data available yet
        };

        if frames_read == 0 {
            continue;
        }

        let data_bytes = (frames_read as usize) * block_align;
        let data = &buffer[..data_bytes];

        if block_align == 0 || data.len() % block_align != 0 {
            return Err(AecError::BackendError(format!(
                "misaligned audio data: {} bytes, block_align {}",
                data.len(),
                block_align
            )));
        }

        let samples = convert_to_f32(&data, bits, is_float, native_channels);
        if samples.is_empty() {
            continue;
        }

        if sender.send(samples).is_err() {
            break;
        }
    }

    audio_client
        .stop_stream()
        .map_err(|e| AecError::BackendError(format!("stop_stream: {e:?}")))?;

    Ok(())
}

fn playback_loop(playback_rx: flume::Receiver<PlaybackCommand>) -> Result<(), AecError> {
    // Re-initialize COM on this thread
    let hr = initialize_mta();
    if hr.0 != 0 {
        return Err(AecError::BackendError(format!("COM init failed: {hr:?}")));
    }

    let enumerator = DeviceEnumerator::new()
        .map_err(|e| AecError::BackendError(format!("DeviceEnumerator::new: {e:?}")))?;
    let render_device = enumerator
        .get_default_device(&Direction::Render)
        .map_err(|_| AecError::DeviceUnavailable)?;

    let desired_format = WaveFormat::new(32, 32, &SampleType::Float, 48000, 1, None);

    let mut audio_client = render_device
        .get_iaudioclient()
        .map_err(|e| AecError::BackendError(format!("get_iaudioclient: {e:?}")))?;

    let render_format = match audio_client.is_supported(&desired_format, &ShareMode::Shared) {
        Ok(None) => desired_format,
        Ok(Some(suggested)) => suggested,
        Err(_) => audio_client
            .get_mixformat()
            .map_err(|e| AecError::BackendError(format!("get_mixformat: {e:?}")))?,
    };

    let native_rate = render_format.get_samplespersec();
    let native_channels = render_format.get_nchannels() as usize;

    let stream_mode = StreamMode::EventsShared {
        autoconvert: true,
        buffer_duration_hns: 200_000,
    };
    audio_client
        .initialize_client(&render_format, &Direction::Render, &stream_mode)
        .map_err(|e| AecError::BackendError(format!("initialize_client: {e:?}")))?;

    let render_client = audio_client
        .get_audiorenderclient()
        .map_err(|e| AecError::BackendError(format!("get_audiorenderclient: {e:?}")))?;

    let event_handle = audio_client
        .set_get_eventhandle()
        .map_err(|e| AecError::BackendError(format!("set_get_eventhandle: {e:?}")))?;

    audio_client
        .start_stream()
        .map_err(|e| AecError::BackendError(format!("start_stream: {e:?}")))?;

    while let Ok(command) = playback_rx.recv() {
        match command {
            PlaybackCommand::OneShot(samples) => {
                write_samples_to_device(&samples, native_channels, &render_client, &event_handle);
            }
            PlaybackCommand::StartStream(chunk_rx) => {
                while let Ok(samples) = chunk_rx.recv() {
                    write_samples_to_device(
                        &samples,
                        native_channels,
                        &render_client,
                        &event_handle,
                    );
                }
            }
        }
    }

    audio_client
        .stop_stream()
        .map_err(|e| AecError::BackendError(format!("stop_stream: {e:?}")))?;

    Ok(())
}

fn write_samples_to_device(
    samples: &[f32],
    native_channels: usize,
    render_client: &wasapi::AudioRenderClient,
    event_handle: &wasapi::Handle,
) {
    let samples: Vec<f32> = if native_channels > 1 {
        samples
            .iter()
            .flat_map(|&s| std::iter::repeat(s).take(native_channels))
            .collect()
    } else {
        samples.to_vec()
    };

    let frames_per_write = 480;
    for chunk in samples.chunks(frames_per_write * native_channels) {
        let _ = event_handle.wait_for_event(100);

        let frames = chunk.len() / native_channels;

        let mut bytes: Vec<u8> = Vec::with_capacity(chunk.len() * 4);
        for sample in chunk {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }

        if render_client.write_to_device(frames, &bytes, None).is_err() {
            break;
        }
    }
}

fn convert_to_f32(data: &[u8], bits: u16, is_float: bool, channels: usize) -> Vec<f32> {
    if is_float && bits == 32 {
        return convert_f32_to_mono(data, channels);
    }
    if bits == 16 {
        return convert_i16_to_mono(data, channels);
    }
    if bits == 24 {
        return convert_i24_to_mono(data, channels);
    }
    if bits == 32 && !is_float {
        return convert_i32_to_mono(data, channels);
    }
    Vec::new()
}

fn convert_f32_to_mono(data: &[u8], channels: usize) -> Vec<f32> {
    let samples: Vec<f32> = data
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    if channels == 1 {
        return samples;
    }

    samples
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

fn convert_i16_to_mono(data: &[u8], channels: usize) -> Vec<f32> {
    let samples: Vec<f32> = data
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
        .collect();

    if channels == 1 {
        return samples;
    }

    samples
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

fn convert_i24_to_mono(data: &[u8], channels: usize) -> Vec<f32> {
    let samples: Vec<f32> = data
        .chunks_exact(3)
        .map(|b| {
            let val =
                i32::from_le_bytes([b[0], b[1], b[2], if b[2] & 0x80 != 0 { 0xFF } else { 0 }]);
            val as f32 / 8388608.0
        })
        .collect();

    if channels == 1 {
        return samples;
    }

    samples
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

fn convert_i32_to_mono(data: &[u8], channels: usize) -> Vec<f32> {
    let samples: Vec<f32> = data
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32 / 2147483648.0)
        .collect();

    if channels == 1 {
        return samples;
    }

    samples
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}
