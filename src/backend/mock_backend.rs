//! Deterministic mock backend for testing and benchmarking without external C runtime dependencies.

use crate::backend::ModelBackend;
use crate::types::{InferenceError, TensorPayload};

use std::sync::atomic::{AtomicUsize, Ordering};

static MOCK_OOM_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Sets the number of consecutive OOM failures before mock allocation succeeds.
pub fn set_mock_oom_failures(count: usize) {
    MOCK_OOM_COUNT.store(count, Ordering::SeqCst);
}

/// Consumes one simulated OOM failure. Returns true if failure should occur.
#[must_use]
pub fn check_and_consume_mock_oom() -> bool {
    let mut current = MOCK_OOM_COUNT.load(Ordering::SeqCst);
    while current > 0 {
        match MOCK_OOM_COUNT.compare_exchange_weak(
            current,
            current.saturating_sub(1),
            Ordering::SeqCst,
            Ordering::SeqCst,
        ) {
            Ok(_) => return true,
            Err(actual) => current = actual,
        }
    }
    false
}

/// A mock backend that simulates model execution for testing, verification, and benchmarking.
#[derive(Debug, Clone, Default)]
pub struct MockBackend {
    provider: &'static str,
}

impl MockBackend {
    /// Creates a new `MockBackend` with a default provider label.
    #[must_use]
    pub fn new() -> Self {
        Self {
            provider: "mock-engine",
        }
    }

    /// Creates a new `MockBackend` with a custom provider label.
    #[must_use]
    pub fn with_provider(provider: &'static str) -> Self {
        Self { provider }
    }
}

impl ModelBackend for MockBackend {
    /// Simulates batched execution.
    /// Produces an output tensor named `"output"` of shape `[batch_size, 4]` with computed values.
    ///
    /// # Errors
    /// Returns [`InferenceError::InvalidInput`] if no inputs are supplied or batch size is 0.
    fn infer(&self, inputs: &[TensorPayload]) -> Result<Vec<TensorPayload>, InferenceError> {
        let first_input = inputs.first().ok_or_else(|| {
            InferenceError::InvalidInput("No input tensors supplied for mock inference".into())
        })?;

        let batch_size = first_input.batch_size();
        if batch_size == 0 {
            return Err(InferenceError::InvalidInput(
                "Input tensor has 0 batch size".into(),
            ));
        }

        // Generate output shape [batch_size, 4]
        let out_shape = vec![batch_size, 4];
        let mut out_data = Vec::with_capacity(batch_size.saturating_mul(4));

        for i in 0..batch_size {
            for j in 0..4 {
                // Deterministic calculation based on indices
                let val = (i as f32) * 10.0 + (j as f32);
                out_data.push(val);
            }
        }

        let payload = TensorPayload::from_shape_vec("output", out_shape, out_data)?;
        Ok(vec![payload])
    }

    fn provider_name(&self) -> &'static str {
        self.provider
    }
}
