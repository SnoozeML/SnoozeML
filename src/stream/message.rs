//! Serialization, wire formats, and message envelopes for stream-based inference jobs.

use crate::config::StreamWireFormat;
use crate::types::{InferenceError, TensorPayload};
use serde::{Deserialize, Serialize};

/// Tensor representation inside stream request and response payloads.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StreamTensorData {
    /// Node or tensor input/output binding name.
    pub name: String,
    /// Multi-dimensional shape dimensions.
    pub shape: Vec<usize>,
    /// Flattened buffer elements as a vector of 32-bit floats.
    pub data: Vec<f32>,
}

impl StreamTensorData {
    /// Creates a new `StreamTensorData` descriptor.
    #[must_use]
    pub fn new(name: impl Into<String>, shape: Vec<usize>, data: Vec<f32>) -> Self {
        Self {
            name: name.into(),
            shape,
            data,
        }
    }

    /// Converts this stream tensor into an internal validated [`TensorPayload`].
    ///
    /// # Errors
    /// Returns [`InferenceError::InvalidInput`] if dimensions overflow, element count mismatches,
    /// or floats are non-finite (`NaN`, `+Inf`, `-Inf`).
    pub fn into_payload(self) -> Result<TensorPayload, InferenceError> {
        TensorPayload::from_shape_vec(self.name, self.shape, self.data)
    }

    /// Constructs a `StreamTensorData` instance from a validated [`TensorPayload`].
    #[must_use]
    pub fn from_payload(payload: TensorPayload) -> Self {
        let (name, shape, data) = payload.into_components();
        Self { name, shape, data }
    }
}

/// Incoming streaming inference request envelope.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StreamInferRequest {
    /// Client-provided correlation ID to match response messages to original requests.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Optional model identifier override if consuming from a shared multi-model stream.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    /// Tensor inputs for this inference request.
    pub inputs: Vec<StreamTensorData>,
    /// Optional dynamic destination topic/queue override for routing this response.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reply_to: Option<String>,
}

impl StreamInferRequest {
    /// Deserializes a request from raw bytes based on the configured [`StreamWireFormat`].
    ///
    /// # Errors
    /// Returns [`InferenceError::InvalidInput`] if decoding fails or payload is malformed.
    pub fn from_bytes(bytes: &[u8], format: StreamWireFormat) -> Result<Self, InferenceError> {
        match format {
            StreamWireFormat::Json => serde_json::from_slice(bytes).map_err(|e| {
                InferenceError::InvalidInput(format!("Failed to parse stream JSON payload: {e}"))
            }),
            StreamWireFormat::Protobuf => {
                // Parse snoozeml.v1.ModelInferRequest protobuf bytes
                use prost::Message;
                let proto_req = crate::server::grpc::proto::ModelInferRequest::decode(bytes)
                    .map_err(|e| {
                        InferenceError::InvalidInput(format!(
                            "Failed to decode protobuf payload: {e}"
                        ))
                    })?;

                let mut inputs = Vec::with_capacity(proto_req.inputs.len());
                for proto_tensor in proto_req.inputs {
                    let shape: Vec<usize> = proto_tensor
                        .shape
                        .iter()
                        .map(|&d| usize::try_from(d).unwrap_or(0))
                        .collect();

                    let element_count = shape
                        .iter()
                        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
                        .unwrap_or(0);
                    let expected_bytes = element_count.saturating_mul(std::mem::size_of::<f32>());

                    if proto_tensor.raw_data.len() != expected_bytes {
                        return Err(InferenceError::InvalidInput(format!(
                            "Protobuf tensor '{}' raw byte length {} does not match expected element count {} ({} bytes)",
                            proto_tensor.name,
                            proto_tensor.raw_data.len(),
                            element_count,
                            expected_bytes
                        )));
                    }

                    let float_slice: &[f32] = bytemuck::try_cast_slice(&proto_tensor.raw_data)
                        .map_err(|e| {
                            InferenceError::InvalidInput(format!(
                                "Byte alignment error in protobuf tensor: {e}"
                            ))
                        })?;

                    inputs.push(StreamTensorData {
                        name: proto_tensor.name,
                        shape,
                        data: float_slice.to_vec(),
                    });
                }

                let id = proto_req
                    .parameters
                    .get("id")
                    .or_else(|| proto_req.parameters.get("correlation_id"))
                    .cloned();

                let reply_to = proto_req.parameters.get("reply_to").cloned();

                Ok(Self {
                    id,
                    model_id: if proto_req.model_id.is_empty() {
                        None
                    } else {
                        Some(proto_req.model_id)
                    },
                    inputs,
                    reply_to,
                })
            }
        }
    }

    /// Converts all enclosed stream tensor inputs into internal [`TensorPayload`] buffers.
    ///
    /// # Errors
    /// Returns [`InferenceError::InvalidInput`] if any tensor shape or data elements are invalid.
    pub fn extract_payloads(self) -> Result<Vec<TensorPayload>, InferenceError> {
        let mut payloads = Vec::with_capacity(self.inputs.len());
        for tensor in self.inputs {
            payloads.push(tensor.into_payload()?);
        }
        Ok(payloads)
    }
}

/// Outgoing streaming inference response envelope published to output topics or DLQs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StreamInferResponse {
    /// Correlation ID matching the input request.
    pub id: String,
    /// Model identifier that executed this inference request.
    pub model_id: String,
    /// Execution status (`"success"` or `"error"`).
    pub status: String,
    /// Output tensor outputs if execution succeeded.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub outputs: Vec<StreamTensorData>,
    /// Error message diagnostic if execution failed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Execution duration in milliseconds.
    pub duration_ms: f64,
}

impl StreamInferResponse {
    /// Constructs a successful response envelope from output [`TensorPayload`] buffers.
    #[must_use]
    pub fn success(
        id: impl Into<String>,
        model_id: impl Into<String>,
        outputs: Vec<TensorPayload>,
        duration_ms: f64,
    ) -> Self {
        let output_dtos = outputs
            .into_iter()
            .map(StreamTensorData::from_payload)
            .collect();
        Self {
            id: id.into(),
            model_id: model_id.into(),
            status: "success".to_string(),
            outputs: output_dtos,
            error: None,
            duration_ms,
        }
    }

    /// Constructs an error response envelope for failure reporting or dead-letter queue routing.
    #[must_use]
    pub fn error(
        id: impl Into<String>,
        model_id: impl Into<String>,
        err_msg: impl Into<String>,
        duration_ms: f64,
    ) -> Self {
        Self {
            id: id.into(),
            model_id: model_id.into(),
            status: "error".to_string(),
            outputs: Vec::new(),
            error: Some(err_msg.into()),
            duration_ms,
        }
    }

    /// Encodes this response into raw bytes based on the configured [`StreamWireFormat`].
    ///
    /// # Errors
    /// Returns [`InferenceError::Runtime`] if encoding fails.
    pub fn to_bytes(&self, format: StreamWireFormat) -> Result<Vec<u8>, InferenceError> {
        match format {
            StreamWireFormat::Json => serde_json::to_vec(self).map_err(|e| {
                InferenceError::Runtime(format!("Failed to serialize stream JSON response: {e}"))
            }),
            StreamWireFormat::Protobuf => {
                use prost::Message;
                let proto_outputs = self
                    .outputs
                    .iter()
                    .map(|out| {
                        let shape_i64: Vec<i64> = out
                            .shape
                            .iter()
                            .map(|&d| i64::try_from(d).unwrap_or(0))
                            .collect();
                        let raw_data = bytemuck::cast_slice(&out.data).to_vec();
                        crate::server::grpc::proto::TensorData {
                            name: out.name.clone(),
                            shape: shape_i64,
                            data_type: crate::server::grpc::proto::DataType::Fp32 as i32,
                            raw_data,
                        }
                    })
                    .collect();

                let proto_resp = crate::server::grpc::proto::ModelInferResponse {
                    model_id: self.model_id.clone(),
                    outputs: proto_outputs,
                    duration_ms: self.duration_ms,
                };

                let mut buf = Vec::with_capacity(proto_resp.encoded_len());
                proto_resp.encode(&mut buf).map_err(|e| {
                    InferenceError::Runtime(format!("Failed to encode protobuf response: {e}"))
                })?;
                Ok(buf)
            }
        }
    }
}
