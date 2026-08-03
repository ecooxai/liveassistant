use rubato::{FftFixedIn, Resampler as RubatoResampler};

use crate::AecError;

const FRAME_DURATION_MS: usize = 10;

/// Wraps rubato for sample rate conversion when the OS backend
/// uses a different rate than requested by the user.
/// Handles variable-sized input by accumulating samples.
pub(crate) struct Resampler {
    resampler: FftFixedIn<f32>,
    input_buffer: Vec<Vec<f32>>,
    output_buffer: Vec<Vec<f32>>,
    accumulator: Vec<f32>,
    chunk_size: usize,
}
impl Resampler {
    /// Create a new resampler.
    ///
    /// - `from_rate`: Native sample rate from the backend
    /// - `to_rate`: Target sample rate requested by user
    pub fn new(from_rate: u32, to_rate: u32) -> Result<Self, AecError> {
        let chunk_size = (from_rate as usize * FRAME_DURATION_MS) / 1000;

        let resampler = FftFixedIn::new(from_rate as usize, to_rate as usize, chunk_size, 1, 1)
            .map_err(|e| AecError::BackendError(format!("resampler init failed: {e}")))?;

        let input_buffer = resampler.input_buffer_allocate(true);
        let output_buffer = resampler.output_buffer_allocate(true);

        Ok(Self {
            resampler,
            input_buffer,
            output_buffer,
            accumulator: Vec::with_capacity(chunk_size * 2),
            chunk_size,
        })
    }

    /// Process samples and return resampled output.
    /// Accumulates input until enough for a fixed chunk, then processes.
    /// May return empty Vec if not enough samples accumulated yet.
    pub fn process(&mut self, input: &[f32]) -> Result<Vec<f32>, AecError> {
        self.accumulator.extend_from_slice(input);

        let mut output = Vec::new();

        while self.accumulator.len() >= self.chunk_size {
            self.input_buffer[0].clear();
            self.input_buffer[0].extend(self.accumulator.drain(..self.chunk_size));

            let (_, samples_out) = self
                .resampler
                .process_into_buffer(&self.input_buffer, &mut self.output_buffer, None)
                .map_err(|e| AecError::BackendError(format!("resampling failed: {e}")))?;

            output.extend_from_slice(&self.output_buffer[0][..samples_out]);
        }

        Ok(output)
    }
}
