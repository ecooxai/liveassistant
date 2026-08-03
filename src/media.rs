use anyhow::{Context, Result, bail};
use image::{DynamicImage, ImageFormat, codecs::jpeg::JpegEncoder, imageops::FilterType};
use std::{fs::File, io::Cursor, path::Path};
use symphonia::core::{
    audio::SampleBuffer, codecs::DecoderOptions, formats::FormatOptions, io::MediaSourceStream,
    meta::MetadataOptions, probe::Hint,
};

/// Keep the base64 realtime payload comfortably below a megabyte. Oversized
/// image events can remain backpressured for the whole 10-second upload window
/// even when the JPEG itself is valid.
pub(crate) const MAX_JPEG_UPLOAD_BYTES: usize = 256 * 1024;
const MIN_UPLOAD_IMAGE_LONG_SIDE: u32 = 640;
const JPEG_QUALITY_STEPS: [u8; 6] = [82, 74, 66, 58, 50, 42];

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
    image_attachment(
        "Pasted image.jpg".to_owned(),
        DynamicImage::ImageRgba8(rgba),
    )
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
        "Current screen.jpg".to_owned(),
        DynamicImage::ImageRgba8(image),
        85,
    )
}

/// Generates a deterministic, screenshot-sized JPEG for the command-line
/// transport probe. Its visual noise keeps the payload above 64 KiB so the
/// probe exercises a genuinely large base64 request like a real desktop capture
/// without requiring Screen Recording permission.
#[cfg(test)]
pub fn jpeg_upload_probe_attachment() -> Result<Attachment> {
    let pixels = image::RgbImage::from_fn(1_280, 800, |x, y| {
        let mixed = x
            .wrapping_mul(1_664_525)
            .wrapping_add(y.wrapping_mul(1_013_904_223))
            .rotate_left((y % 31) + 1);
        image::Rgb([
            mixed as u8,
            mixed.rotate_left(9) as u8,
            mixed.rotate_left(19) as u8,
        ])
    });
    encode_image_attachment_with_quality(
        "JPEG upload probe.jpg".to_owned(),
        DynamicImage::ImageRgb8(pixels),
        85,
    )
}

/// Produces two visually unmistakable screenshot-like JPEGs for the live
/// ordering probe. The small deterministic texture prevents the images from
/// collapsing into unrealistically tiny solid-color JPEGs while preserving a
/// clearly dominant red first capture and blue second capture.
#[cfg(test)]
pub fn jpeg_latest_image_probe_attachments() -> Result<[Attachment; 2]> {
    Ok([
        jpeg_color_probe_attachment("Older red screen.jpg", 0)?,
        jpeg_color_probe_attachment("Latest blue screen.jpg", 2)?,
    ])
}

/// Real photographic fixtures for the live multi-turn freshness probe. Keep
/// these as JPEGs in the repository so the probe exercises the same decode,
/// size-budget, MIME, and upload path as a captured desktop image.
pub fn jpeg_animal_probe_attachments() -> Result<[Attachment; 2]> {
    Ok([
        jpeg_animal_probe_attachment(
            "First animal screen.jpg",
            include_bytes!("../tests/fixtures/cat.jpg"),
        )?,
        jpeg_animal_probe_attachment(
            "Second animal screen.jpg",
            include_bytes!("../tests/fixtures/dog.jpg"),
        )?,
    ])
}

fn jpeg_animal_probe_attachment(name: &str, bytes: &[u8]) -> Result<Attachment> {
    let image = image::load_from_memory_with_format(bytes, ImageFormat::Jpeg)
        .with_context(|| format!("Could not decode {name} probe fixture"))?;
    encode_image_attachment_with_quality(name.to_owned(), image, 85)
}

#[cfg(test)]
fn jpeg_color_probe_attachment(name: &str, dominant_channel: usize) -> Result<Attachment> {
    let pixels = image::RgbImage::from_fn(1_280, 800, |x, y| {
        let mixed = x
            .wrapping_mul(2_246_822_519)
            .wrapping_add(y.wrapping_mul(3_266_489_917))
            .rotate_left((x.wrapping_add(y) % 31) + 1);
        let mut rgb = [
            18_u8.saturating_add((mixed & 31) as u8),
            18_u8.saturating_add(((mixed >> 8) & 31) as u8),
            18_u8.saturating_add(((mixed >> 16) & 31) as u8),
        ];
        rgb[dominant_channel] = 205_u8.saturating_add(((mixed >> 24) & 31) as u8);
        image::Rgb(rgb)
    });
    encode_image_attachment_with_quality(name.to_owned(), DynamicImage::ImageRgb8(pixels), 85)
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
    let (image, bytes) = encode_jpeg_with_budget(image, quality)?;
    let width = image.width();
    let height = image.height();

    let thumb = image.thumbnail(480, 300);
    let mut thumbnail = Cursor::new(Vec::new());
    thumb
        .write_to(&mut thumbnail, ImageFormat::Png)
        .context("Could not encode thumbnail")?;

    Ok(Attachment::Image {
        name: jpeg_attachment_name(&name),
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

fn encode_jpeg_with_budget(
    mut image: DynamicImage,
    initial_quality: u8,
) -> Result<(DynamicImage, Vec<u8>)> {
    loop {
        let mut last_encoded = Vec::new();
        for quality in std::iter::once(initial_quality).chain(
            JPEG_QUALITY_STEPS
                .into_iter()
                .filter(|quality| *quality < initial_quality),
        ) {
            let mut jpeg = Cursor::new(Vec::new());
            image
                .write_with_encoder(JpegEncoder::new_with_quality(&mut jpeg, quality))
                .context("Could not encode image as JPEG")?;
            last_encoded = jpeg.into_inner();
            if last_encoded.len() <= MAX_JPEG_UPLOAD_BYTES {
                return Ok((image, last_encoded));
            }
        }

        let long_side = image.width().max(image.height());
        if long_side <= MIN_UPLOAD_IMAGE_LONG_SIDE {
            return Ok((image, last_encoded));
        }
        let next_long_side = (long_side * 4 / 5).max(MIN_UPLOAD_IMAGE_LONG_SIDE);
        let scale = next_long_side as f64 / long_side as f64;
        let next_width = ((image.width() as f64 * scale).round() as u32).max(1);
        let next_height = ((image.height() as f64 * scale).round() as u32).max(1);
        image = image.resize_exact(next_width, next_height, FilterType::Triangle);
    }
}

fn jpeg_attachment_name(name: &str) -> String {
    let stem = Path::new(name)
        .file_stem()
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty())
        .unwrap_or("image");
    format!("{stem}.jpg")
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
    use super::{
        Attachment, DynamicImage, MAX_JPEG_UPLOAD_BYTES, encode_image_attachment,
        jpeg_animal_probe_attachments, jpeg_attachment_name, jpeg_latest_image_probe_attachments,
        jpeg_upload_probe_attachment, resample_to_24k, wav_file_size,
    };
    use base64::{Engine, engine::general_purpose::STANDARD};
    use image::{ImageFormat, Rgba, RgbaImage};

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

    #[test]
    fn image_transport_is_real_jpeg_with_a_jpg_name() {
        let pixels = RgbaImage::from_pixel(8, 6, Rgba([12, 34, 56, 255]));
        let attachment = encode_image_attachment(
            "screen capture.png".to_owned(),
            DynamicImage::ImageRgba8(pixels),
        )
        .unwrap();

        let Attachment::Image {
            name,
            data_url,
            thumbnail,
            byte_size,
            ..
        } = attachment
        else {
            panic!("expected an image attachment");
        };
        assert_eq!(name, "screen capture.jpg");
        let encoded = data_url
            .strip_prefix("data:image/jpeg;base64,")
            .expect("JPEG data URI");
        let bytes = STANDARD.decode(encoded).unwrap();
        assert_eq!(byte_size, bytes.len());
        assert_eq!(image::guess_format(&bytes).unwrap(), ImageFormat::Jpeg);
        assert_eq!(&bytes[..2], &[0xff, 0xd8]);
        assert_eq!(&bytes[bytes.len() - 2..], &[0xff, 0xd9]);
        assert_eq!(image::guess_format(&thumbnail).unwrap(), ImageFormat::Png);
    }

    #[test]
    fn jpeg_name_replaces_or_adds_the_extension() {
        assert_eq!(jpeg_attachment_name("screen.png"), "screen.jpg");
        assert_eq!(jpeg_attachment_name("Current screen"), "Current screen.jpg");
        assert_eq!(jpeg_attachment_name("photo.JPEG"), "photo.jpg");
    }

    #[test]
    fn upload_probe_exercises_a_large_jpeg_message() {
        let Attachment::Image {
            name,
            data_url,
            byte_size,
            ..
        } = jpeg_upload_probe_attachment().unwrap()
        else {
            panic!("expected an image attachment");
        };
        assert_eq!(name, "JPEG upload probe.jpg");
        assert!(byte_size > 65_536, "probe JPEG was only {byte_size} bytes");
        assert!(byte_size <= MAX_JPEG_UPLOAD_BYTES);
        assert!(data_url.starts_with("data:image/jpeg;base64,"));
    }

    #[test]
    fn latest_image_probe_is_red_then_blue() {
        let images = jpeg_latest_image_probe_attachments().unwrap();
        for (attachment, dominant_channel) in images.into_iter().zip([0_usize, 2_usize]) {
            let Attachment::Image {
                data_url,
                byte_size,
                ..
            } = attachment
            else {
                panic!("expected an image attachment");
            };
            assert!(byte_size <= MAX_JPEG_UPLOAD_BYTES);
            let bytes = STANDARD
                .decode(data_url.strip_prefix("data:image/jpeg;base64,").unwrap())
                .unwrap();
            let decoded = image::load_from_memory(&bytes).unwrap().to_rgb8();
            let channel_totals = decoded.pixels().fold([0_u64; 3], |mut totals, pixel| {
                for (total, channel) in totals.iter_mut().zip(pixel.0) {
                    *total += channel as u64;
                }
                totals
            });
            assert!(
                channel_totals[dominant_channel] > channel_totals[(dominant_channel + 1) % 3] * 4
            );
            assert!(
                channel_totals[dominant_channel] > channel_totals[(dominant_channel + 2) % 3] * 4
            );
        }
    }

    #[test]
    fn animal_probe_fixtures_use_the_production_jpeg_budget() {
        for attachment in jpeg_animal_probe_attachments().unwrap() {
            let Attachment::Image {
                name,
                data_url,
                width,
                height,
                byte_size,
                ..
            } = attachment
            else {
                panic!("expected an image attachment");
            };
            assert!(name.ends_with(".jpg"));
            assert!(data_url.starts_with("data:image/jpeg;base64,"));
            assert_eq!((width, height), (1024, 1024));
            assert!(byte_size <= MAX_JPEG_UPLOAD_BYTES);
        }
    }
}
