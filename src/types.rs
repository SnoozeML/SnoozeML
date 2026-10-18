//! Core domain types and error representations for SnoozeML inference server.

use ndarray::ArrayD;
use serde::{Deserialize, Serialize};
use std::time::Instant;
use tokio::sync::oneshot;

/// Errors that can occur during inference, batch collation, model lifecycle, or payload processing.
#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum InferenceError {
    /// The requested model identifier is not registered.
    #[error("Model not found: {0}")]
    NotFound(String),

    /// A failure occurred in the underlying inference engine or runtime.
    #[error("Inference runtime failure: {0}")]
    Runtime(String),

    /// Tensors within a micro-batch have incompatible dimensions or channel counts.
    #[error("Batch collation shape mismatch: {0}")]
    ShapeMismatch(String),

    /// The inference request exceeded its allotted processing or queuing deadline.
    #[error("Service timeout")]
    Timeout,

    /// Input payload structure, shapes, or element counts are invalid.
    #[error("Invalid tensor input: {0}")]
    InvalidInput(String),

    /// The model is currently loading weights into memory.
    #[error("Model is currently loading weights")]
    Loading,

    /// The batch accumulation queue is either full or closed.
    #[error("Batch queue error: {0}")]
    QueueError(String),

    /// Model architecture in safetensors is not supported natively by Candle; fallback or use ONNX.
    #[error("Unsupported Safetensors architecture: '{0}'. Please convert and serve this model using ONNX format.")]
    UnsupportedSafetensorsArchitecture(String),

    /// Allocation failed due to host RAM or device VRAM exhaustion.
    #[error("Device/System out of memory: {0}")]
    OutOfMemory(String),

    /// All deallocatable models have been purged, but insufficient memory remains.
    #[error(
        "Resource limit reached: cannot allocate model even after deallocating idle sessions: {0}"
    )]
    ResourceExhausted(String),
}

impl InferenceError {
    /// Strictly verifies whether an error represents a genuine out-of-memory condition.
    ///
    /// Returns `false` for generic runtime errors, invalid inputs, or model format errors.
    #[must_use]
    pub fn is_out_of_memory(&self) -> bool {
        match self {
            Self::OutOfMemory(_) => true,
            Self::Runtime(msg) => {
                let lower = msg.to_ascii_lowercase();

                // 1. Explicit C++ std::bad_alloc
                if lower.contains("bad_alloc") {
                    return true;
                }

                // 2. Explicit CUDA driver OOM codes
                if lower.contains("cuda failure 2") // CUDA_ERROR_OUT_OF_MEMORY
                    || lower.contains("cuda error: out of memory")
                    || lower.contains("cudamalloc failed")
                    || lower.contains("cuda_error_out_of_memory")
                {
                    return true;
                }

                // 3. Explicit device/host out-of-memory assertions
                if (lower.contains("out of memory") || lower.contains("ran out of memory"))
                    && !lower.contains("invalid")
                // Reject "invalid memory access"
                {
                    return true;
                }

                // Any generic message (e.g. "allocation failed for shape [0]", "ORT_FAIL", etc.)
                // is intentionally rejected to prevent false deallocations.
                false
            }
            _ => false,
        }
    }
}

/// Represents a multi-dimensional tensor payload for inference inputs and outputs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorPayload {
    /// Identifier or binding name of the tensor input/output node.
    pub name: String,
    /// Multi-dimensional dimensions of the tensor.
    pub shape: Vec<usize>,
    /// Flattened buffer elements as a dynamic-dimensional array.
    pub data: ArrayD<f32>,
}

impl TensorPayload {
    /// Creates a new `TensorPayload` from an array of dimensions and a flat float vector.
    ///
    /// # Errors
    /// Returns [`InferenceError::InvalidInput`] if:
    /// - Multiplying the dimension elements overflows `usize`.
    /// - The computed total element count does not match `data.len()`.
    /// - Any float value in `data` is non-finite (`NaN`, `+Inf`, or `-Inf`).
    /// - The dynamic array cannot be constructed from the given shape.
    pub fn from_shape_vec(
        name: impl Into<String>,
        shape: Vec<usize>,
        data: Vec<f32>,
    ) -> Result<Self, InferenceError> {
        let name = name.into();
        let total_elements = shape.iter().try_fold(1usize, |acc, &dim| {
            acc.checked_mul(dim).ok_or_else(|| {
                InferenceError::InvalidInput("Shape dimensions overflow usize".into())
            })
        })?;

        if total_elements != data.len() {
            return Err(InferenceError::InvalidInput(format!(
                "Shape {shape:?} expects {total_elements} elements, but provided buffer has {} elements",
                data.len()
            )));
        }

        if !data.iter().all(|f| f.is_finite()) {
            return Err(InferenceError::InvalidInput(
                "Tensor data contains non-finite float values (NaN, +Inf, or -Inf)".into(),
            ));
        }

        let array = ArrayD::from_shape_vec(ndarray::IxDyn(&shape), data)
            .map_err(|e| InferenceError::InvalidInput(format!("Failed to reshape array: {e}")))?;

        Ok(Self {
            name,
            shape,
            data: array,
        })
    }

    /// Returns the batch size (leading dimension), or 0 if the shape is empty.
    #[must_use]
    pub fn batch_size(&self) -> usize {
        self.shape.first().copied().unwrap_or(0)
    }

    /// Returns the trailing dimensions after the leading batch dimension.
    #[must_use]
    pub fn trailing_shape(&self) -> &[usize] {
        if self.shape.is_empty() {
            &[]
        } else {
            // Using get to satisfy zero-panic bounds check
            self.shape.get(1..).unwrap_or(&[])
        }
    }

    /// Consumes the payload and decomposes it into its name, shape, and raw float vector components.
    ///
    /// If the tensor array has standard contiguous layout with zero offset,
    /// the buffer is extracted directly with zero allocations or data copying.
    /// Otherwise, contiguous slices or iterator fallbacks are utilized safely.
    #[must_use]
    pub fn into_components(self) -> (String, Vec<usize>, Vec<f32>) {
        let name = self.name;
        let shape = self.shape;
        let expected_len = shape
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .unwrap_or(0);
        let data = if self.data.is_standard_layout() {
            let (vec, offset) = self.data.into_raw_vec_and_offset();
            if offset == Some(0) && vec.len() == expected_len {
                vec
            } else if let Some(off) = offset {
                let end = off.saturating_add(expected_len);
                if let Some(slice) = vec.get(off..end) {
                    slice.to_vec()
                } else {
                    vec
                }
            } else {
                vec
            }
        } else if let Some(slice) = self.data.as_slice() {
            slice.to_vec()
        } else {
            self.data.into_iter().collect()
        };

        (name, shape, data)
    }

    /// Consumes the payload and returns the underlying float vector.
    ///
    /// If the tensor array has standard contiguous layout with zero offset,
    /// the buffer is extracted directly with zero allocations or data copying.
    /// Otherwise, contiguous slices or iterator fallbacks are utilized safely.
    #[must_use]
    pub fn into_data(self) -> Vec<f32> {
        self.into_components().2
    }
}

/// A client inference job queued for dynamic micro-batching.
#[derive(Debug)]
pub struct InferenceJob {
    /// Input tensor payloads for this individual request.
    pub inputs: Vec<TensorPayload>,
    /// One-shot response channel back to the awaiting client handler.
    pub response_tx: oneshot::Sender<Result<Vec<TensorPayload>, InferenceError>>,
    /// Timestamp when this job entered the accumulator.
    pub submitted_at: Instant,
}

/// Operational state of a model within the registry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelState {
    /// Configuration registered, but weights are deallocated / not in RAM/VRAM.
    Idle,
    /// Weights are currently being loaded from storage into memory.
    Loading,
    /// Weights are loaded and session is ready for inference execution.
    Ready,
    /// Inference execution is currently actively progressing.
    Busy,
}

/// Information and telemetry for a registered model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Unique identifier for the model.
    pub id: String,
    /// Operational status of the model session.
    pub state: ModelState,
    /// Filesystem or URI path to model weights.
    pub onnx_path: String,
    /// Duration in seconds of inactivity before unloading session.
    pub idle_timeout_secs: u64,
    /// Maximum batch size for dynamic collation.
    pub max_batch_size: usize,
    /// Maximum delay in milliseconds to wait for a full batch before flush.
    pub max_batch_delay_ms: u64,
    /// Number of seconds elapsed since last served inference request.
    pub last_accessed_secs_ago: u64,
    /// Optional human-readable description of the model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Whether the model is pinned in memory and exempt from automatic deallocation.
    #[serde(default)]
    pub pinned: bool,
}

/// Validates whether a model identifier contains only safe alphanumeric characters,
/// underscores, and hyphens (`^[a-zA-Z0-9_-]+$`).
///
/// Disallows empty strings, whitespace, slashes (`/`, `\`), path traversal elements (`..`),
/// and control characters.
#[must_use]
pub fn is_valid_model_id(id: &str) -> bool {
    !id.is_empty()
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}
