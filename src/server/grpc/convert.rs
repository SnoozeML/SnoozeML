//! Defensive conversion utilities between Protobuf DTOs and internal engine types.

#![allow(clippy::result_large_err)]

use crate::server::grpc::proto::{DataType, TensorData};
use crate::types::{InferenceError, TensorPayload};
use tonic::Status;

/// Converts an incoming protobuf [`TensorData`] DTO into an internal [`TensorPayload`].
///
/// # Errors
/// Returns [`tonic::Status`] with `Code::InvalidArgument` if:
/// - Shape dimensions are non-positive or overflow `usize`.
/// - Data type is unsupported (only `DataType::Fp32` and `DataType::Unspecified` are accepted).
/// - Raw byte buffer length does not match `total_elements * 4`.
/// - Buffer cannot be decoded into floats.
///
/// # Panics
/// This function does not panic.
pub fn tensor_data_to_payload(tensor: TensorData) -> Result<TensorPayload, Status> {
    let mut shape_dims = Vec::with_capacity(tensor.shape.len());
    for &dim in &tensor.shape {
        if dim <= 0 {
            return Err(Status::invalid_argument(format!(
                "Tensor '{}' dimension must be positive, got {dim}",
                tensor.name
            )));
        }
        let udim = usize::try_from(dim).map_err(|_| {
            Status::invalid_argument(format!(
                "Tensor '{}' dimension {dim} overflows usize",
                tensor.name
            ))
        })?;
        shape_dims.push(udim);
    }

    if shape_dims.is_empty() {
        return Err(Status::invalid_argument(format!(
            "Tensor '{}' shape cannot be empty",
            tensor.name
        )));
    }

    let total_elements = shape_dims.iter().try_fold(1usize, |acc, &d| {
        acc.checked_mul(d).ok_or_else(|| {
            Status::invalid_argument(format!(
                "Tensor '{}' dimensions overflow usize during multiplication",
                tensor.name
            ))
        })
    })?;

    match DataType::try_from(tensor.data_type).unwrap_or(DataType::Unspecified) {
        DataType::Fp32 | DataType::Unspecified => {}
        other => {
            return Err(Status::invalid_argument(format!(
                "Tensor '{}' specifies unsupported data type {other:?}; only FP32 is supported",
                tensor.name
            )));
        }
    }

    let expected_bytes = total_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            Status::invalid_argument(format!(
                "Tensor '{}' element count causes byte length overflow",
                tensor.name
            ))
        })?;

    if tensor.raw_data.len() != expected_bytes {
        return Err(Status::invalid_argument(format!(
            "Tensor '{}' expects {expected_bytes} bytes for {total_elements} floats, but received {} bytes",
            tensor.name,
            tensor.raw_data.len()
        )));
    }

    let mut floats = Vec::with_capacity(total_elements);
    let (chunks, _) = tensor.raw_data.as_chunks::<4>();
    for &chunk in chunks {
        floats.push(f32::from_le_bytes(chunk));
    }

    TensorPayload::from_shape_vec(tensor.name, shape_dims, floats)
        .map_err(inference_error_to_status)
}

/// Converts an internal [`TensorPayload`] into an outgoing protobuf [`TensorData`] DTO.
///
/// # Panics
/// This function does not panic.
#[must_use]
pub fn payload_to_tensor_data(payload: TensorPayload) -> TensorData {
    let (name, shape, floats) = payload.into_components();
    let shape_i64 = shape.into_iter().map(|d| d as i64).collect();
    let byte_slice: &[u8] = bytemuck::cast_slice(&floats);
    let raw_data = byte_slice.to_vec();

    TensorData {
        name,
        shape: shape_i64,
        data_type: DataType::Fp32 as i32,
        raw_data,
    }
}

/// Maps an internal [`InferenceError`] domain error to a canonical [`tonic::Status`].
///
/// # Panics
/// This function does not panic.
#[must_use]
pub fn inference_error_to_status(err: InferenceError) -> Status {
    match err {
        InferenceError::NotFound(msg) => Status::not_found(msg),
        InferenceError::InvalidInput(msg) | InferenceError::ShapeMismatch(msg) => {
            Status::invalid_argument(msg)
        }
        InferenceError::Timeout => {
            Status::deadline_exceeded("Request processing deadline exceeded")
        }
        InferenceError::Loading => Status::unavailable("Model is currently loading weights"),
        InferenceError::ResourceExhausted(msg) => {
            let mut status = Status::resource_exhausted(msg);
            let val = tonic::metadata::MetadataValue::from_static("5");
            status.metadata_mut().insert("retry-after", val);
            status
        }
        InferenceError::OutOfMemory(msg) => Status::resource_exhausted(msg),
        InferenceError::UnsupportedSafetensorsArchitecture(msg) => {
            Status::unimplemented(format!("Unsupported Safetensors architecture: {msg}"))
        }
        InferenceError::Runtime(msg) | InferenceError::QueueError(msg) => {
            Status::internal(crate::server::http::sanitize_error_message(&msg))
        }
    }
}
