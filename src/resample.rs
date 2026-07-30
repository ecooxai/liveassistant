use anyhow::{Context, Result};
use rubato::{FftFixedIn, Resampler};

const FRAME_DURATION_MS: usize = 10;

enum ResamplerInner {
    Passthrough,
    Fft {
        engine: FftFixedIn<f32>,
        input_buffer: Vec<Vec<f32>>,
        output_buffer: Vec<Vec<f32>>,
        accumulator: Vec<f32>,
        chunk_size: usize,
    },
}

/// Stateful mono sample-rate converter for realtime audio streams.
///
/// FFT resampling avoids the aliasing and repeated-sample artifacts produced by
/// the previous box-filter/linear-interpolation path. Equal-rate streams pass
/// through without buffering.
pub(crate) struct StreamingResampler {
    from_rate: u32,
    to_rate: u32,
    inner: ResamplerInner,
}

impl StreamingResampler {
    pub(crate) fn new(from_rate: u32, to_rate: u32) -> Result<Self> {
        anyhow::ensure!(
            from_rate > 0 && to_rate > 0,
            "Audio sample rates must be non-zero"
        );
        let inner = if from_rate == to_rate {
            ResamplerInner::Passthrough
        } else {
            let chunk_size = (from_rate as usize * FRAME_DURATION_MS) / 1_000;
            let engine = FftFixedIn::new(
                from_rate as usize,
                to_rate as usize,
                chunk_size.max(1),
                1,
                1,
            )
            .context("Could not initialize FFT audio resampler")?;
            let input_buffer = engine.input_buffer_allocate(true);
            let output_buffer = engine.output_buffer_allocate(true);
            ResamplerInner::Fft {
                engine,
                input_buffer,
                output_buffer,
                accumulator: Vec::with_capacity(chunk_size.max(1) * 2),
                chunk_size: chunk_size.max(1),
            }
        };
        Ok(Self {
            from_rate,
            to_rate,
            inner,
        })
    }

    pub(crate) fn process(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        match &mut self.inner {
            ResamplerInner::Passthrough => Ok(input.to_vec()),
            ResamplerInner::Fft {
                engine,
                input_buffer,
                output_buffer,
                accumulator,
                chunk_size,
            } => {
                accumulator.extend_from_slice(input);
                let mut output = Vec::new();
                while accumulator.len() >= *chunk_size {
                    input_buffer[0].clear();
                    input_buffer[0].extend(accumulator.drain(..*chunk_size));
                    let (_, samples_out) = engine
                        .process_into_buffer(input_buffer, output_buffer, None)
                        .context("Could not resample realtime audio")?;
                    output.extend_from_slice(&output_buffer[0][..samples_out]);
                }
                Ok(output)
            }
        }
    }

    pub(crate) fn process_complete(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        self.reset();
        let expected =
            ((input.len() as f64) * self.to_rate as f64 / self.from_rate as f64).round() as usize;
        let mut output = self.process(input)?;
        if let ResamplerInner::Fft {
            accumulator,
            chunk_size,
            ..
        } = &mut self.inner
            && !accumulator.is_empty()
        {
            let padding = chunk_size.saturating_sub(accumulator.len());
            accumulator.extend(std::iter::repeat_n(0.0, padding));
            output.extend(self.process(&[])?);
        }
        output.truncate(expected.min(output.len()));
        Ok(output)
    }

    pub(crate) fn reset(&mut self) {
        if let ResamplerInner::Fft {
            engine,
            accumulator,
            ..
        } = &mut self.inner
        {
            engine.reset();
            accumulator.clear();
            // Rubato's allocated channel buffers retain required output length.
            // Clearing them here makes process_into_buffer reject the buffer.
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_resampler_preserves_duration() {
        let input = vec![0.25; 48_000];
        let mut resampler = StreamingResampler::new(48_000, 24_000).unwrap();
        let output = resampler.process_complete(&input).unwrap();
        assert_eq!(output.len(), 24_000);
    }

    #[test]
    fn equal_rate_audio_is_transparent() {
        let input = vec![-0.75, -0.25, 0.0, 0.25, 0.75];
        let mut resampler = StreamingResampler::new(24_000, 24_000).unwrap();
        assert_eq!(resampler.process(&input).unwrap(), input);
    }
}
