use anyhow::{Context, Result, bail};
use image::{DynamicImage, ImageFormat, codecs::jpeg::JpegEncoder, imageops::FilterType};
use std::{fs::File, io::Cursor, path::Path};
use symphonia::core::{
    audio::SampleBuffer, codecs::DecoderOptions, formats::FormatOptions, io::MediaSourceStream,
    meta::MetadataOptions, probe::Hint,
};

#[derive(Clone, Debug)]
pub enum Attachment {
    Image {
        name: String,
        data_url: String,
        thumbnail: Vec<u8>,
        width: u32,
        height: u32,
        byte_size: usize,
    },
    Audio {
        name: String,
        pcm24k: Vec<i16>,
        seconds: f32,
    },
}

pub fn load_image(path: &Path) -> Result<Attachment> {
    let image =
        image::open(path).with_context(|| format!("Could not decode image {}", path.display()))?;
    let name = path
        .file_name()
        .and_then(|v| v.to_str())
        .unwrap_or("image")
        .to_owned();
    image_attachment(name, image)
}

pub fn image_from_clipboard() -> Result<Attachment> {
    let mut clipboard = arboard::Clipboard::new().context("Could not open clipboard")?;
    let image = clipboard
        .get_image()
        .context("Clipboard does not contain an image")?;
    let rgba = image::RgbaImage::from_raw(
        image.width as u32,
        image.height as u32,
        image.bytes.into_owned(),
    )
    .context("Clipboard image data was malformed")?;
    image_attachment("Pasted image".to_owned(), DynamicImage::ImageRgba8(rgba))
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScreenInfo {
    pub origin_x: i32,
    pub origin_y: i32,
    pub logical_width: u32,
    pub logical_height: u32,
    pub backing_width: u32,
    pub backing_height: u32,
    pub scale_factor: f32,
}

pub fn primary_screen_info() -> Result<ScreenInfo> {
    let screens = screenshots::Screen::all().context("Could not enumerate displays")?;
    let display = screens
        .into_iter()
        .find(|screen| screen.display_info.is_primary)
        .context("No primary display was found")?
        .display_info;
    Ok(ScreenInfo {
        origin_x: display.x,
        origin_y: display.y,
        logical_width: display.width,
        logical_height: display.height,
        backing_width: (display.width as f32 * display.scale_factor).round() as u32,
        backing_height: (display.height as f32 * display.scale_factor).round() as u32,
        scale_factor: display.scale_factor,
    })
}

pub fn primary_screen_resolution() -> Result<(u32, u32)> {
    let screen = primary_screen_info()?;
    Ok((screen.logical_width, screen.logical_height))
}

pub fn capture_screenshot(
    target_width: u32,
    target_height: u32,
    show_live_pointer: bool,
    pointer_snapshot: Option<crate::live_pointer::PointerSnapshot>,
) -> Result<Attachment> {
    anyhow::ensure!(
        target_width > 0 && target_height > 0,
        "Screenshot resolution must be larger than zero"
    );
    let screens = screenshots::Screen::all().context("Could not enumerate displays")?;
    let screen = screens
        .into_iter()
        .find(|s| s.display_info.is_primary)
        .context("No primary display was found")?;
    let display = screen.display_info;
    let captured = screen
        .capture()
        .context("Screen capture failed. On macOS, enable Screen Recording for Live Assistant.")?;
    let rgba = image::RgbaImage::from_raw(captured.width(), captured.height(), captured.into_raw())
        .context("Screen capture returned malformed pixels")?;
    // Screenshot preparation runs while the user is still speaking. Triangle
    // downsampling is substantially faster than Lanczos and remains sharp when
    // reducing a Retina backing image to the display's logical resolution.
    let mut image = DynamicImage::ImageRgba8(rgba)
        .resize_exact(target_width, target_height, FilterType::Triangle)
        .to_rgba8();

    if show_live_pointer && let Some(pointer) = pointer_snapshot {
        let relative_x = pointer.position.x - display.x as f32;
        let relative_y = pointer.position.y - display.y as f32;
        if relative_x >= 0.0
            && relative_y >= 0.0
            && relative_x < display.width as f32
            && relative_y < display.height as f32
        {
            let image_x = relative_x * target_width as f32 / display.width as f32;
            let image_y = relative_y * target_height as f32 / display.height as f32;
            crate::live_pointer::paint_image(&mut image, image_x, image_y, pointer.appearance);
        }
    }

    encode_image_attachment_with_quality(
        "Current screen".to_owned(),
        DynamicImage::ImageRgba8(image),
        85,
    )
}

fn image_attachment(name: String, image: DynamicImage) -> Result<Attachment> {
    let image = constrain_image(image, 2048);
    encode_image_attachment(name, image)
}

fn encode_image_attachment(name: String, image: DynamicImage) -> Result<Attachment> {
    encode_image_attachment_with_quality(name, image, 90)
}

fn encode_image_attachment_with_quality(
    name: String,
    image: DynamicImage,
    quality: u8,
) -> Result<Attachment> {
    let width = image.width();
    let height = image.height();
    let mut jpeg = Cursor::new(Vec::new());
    image
        .write_with_encoder(JpegEncoder::new_with_quality(&mut jpeg, quality))
        .context("Could not encode image")?;
    let bytes = jpeg.into_inner();

    let thumb = image.thumbnail(480, 300);
    let mut thumbnail = Cursor::new(Vec::new());
    thumb
        .write_to(&mut thumbnail, ImageFormat::Png)
        .context("Could not encode thumbnail")?;

    Ok(Attachment::Image {
        name,
        data_url: format!(
            "data:image/jpeg;base64,{}",
            base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &bytes)
        ),
        thumbnail: thumbnail.into_inner(),
        width,
        height,
        byte_size: bytes.len(),
    })
}

fn constrain_image(image: DynamicImage, max_side: u32) -> DynamicImage {
    if image.width() <= max_side && image.height() <= max_side {
        image
    } else {
        image.resize(max_side, max_side, FilterType::Triangle)
    }
}

pub fn load_audio(path: &Path) -> Result<Attachment> {
    let file =
        File::open(path).with_context(|| format!("Could not open audio {}", path.display()))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|v| v.to_str()) {
        hint.with_extension(ext);
    }
    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .context("Unsupported audio container")?;
    let mut format = probed.format;
    let track = format
        .default_track()
        .context("Audio file has no default track")?;
    let track_id = track.id;
    let source_rate = track
        .codec_params
        .sample_rate
        .context("Audio file has no sample rate")?;
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .context("Unsupported audio codec")?;
    let mut mono = Vec::<f32>::new();

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(symphonia::core::errors::Error::IoError(err))
                if err.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(err) => return Err(err).context("Could not read audio packet"),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = decoder.decode(&packet).context("Could not decode audio")?;
        let spec = *decoded.spec();
        let mut samples = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
        samples.copy_interleaved_ref(decoded);
        let channels = spec.channels.count();
        for frame in samples.samples().chunks(channels) {
            mono.push(frame.iter().copied().sum::<f32>() / channels as f32);
        }
    }

    if mono.is_empty() {
        bail!("Audio file was empty");
    }
    let pcm24k = resample_to_24k(&mono, source_rate);
    let seconds = pcm24k.len() as f32 / 24_000.0;
    let name = path
        .file_name()
        .and_then(|v| v.to_str())
        .unwrap_or("audio")
        .to_owned();
    Ok(Attachment::Audio {
        name,
        pcm24k,
        seconds,
    })
}

pub fn resample_to_24k(input: &[f32], source_rate: u32) -> Vec<i16> {
    if input.is_empty() {
        return Vec::new();
    }
    let output_len = ((input.len() as f64) * 24_000.0 / source_rate as f64).round() as usize;
    (0..output_len)
        .map(|i| {
            let source_pos = i as f64 * source_rate as f64 / 24_000.0;
            let left = source_pos.floor() as usize;
            let right = (left + 1).min(input.len() - 1);
            let fraction = (source_pos - left as f64) as f32;
            let value = input[left] * (1.0 - fraction) + input[right] * fraction;
            (value.clamp(-1.0, 1.0) * i16::MAX as f32) as i16
        })
        .collect()
}

pub fn save_wav(path: &Path, samples: &[i16]) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 24_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec).context("Could not create WAV")?;
    for sample in samples {
        writer.write_sample(*sample)?;
    }
    writer.finalize()?;
    Ok(())
}

pub fn wav_file_size(sample_count: usize) -> usize {
    const PCM_WAV_HEADER_BYTES: usize = 44;
    PCM_WAV_HEADER_BYTES.saturating_add(sample_count.saturating_mul(size_of::<i16>()))
}

#[cfg(test)]
mod tests {
    use super::{resample_to_24k, wav_file_size};

    #[test]
    fn resampling_preserves_duration() {
        let one_second_at_48k = vec![0.25; 48_000];
        let output = resample_to_24k(&one_second_at_48k, 48_000);
        assert_eq!(output.len(), 24_000);
        assert!(output.iter().all(|sample| *sample > 8_000));
    }

    #[test]
    fn resampling_empty_audio_is_safe() {
        assert!(resample_to_24k(&[], 44_100).is_empty());
    }

    #[test]
    fn wav_size_includes_pcm_header_and_samples() {
        assert_eq!(wav_file_size(0), 44);
        assert_eq!(wav_file_size(24_000), 48_044);
    }
}
