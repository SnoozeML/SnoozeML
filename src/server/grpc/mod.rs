//! gRPC serving subsystem for SnoozeML.
//!
//! Provides high-performance tensor inference and standard `grpc.health.v1` health checking.

pub mod convert;
pub mod service;

#[allow(missing_docs)]
pub mod proto {
    //! Generated Protocol Buffer bindings for SnoozeML v1 services and DTOs.
    tonic::include_proto!("snoozeml.v1");
}

pub use proto::inference_service_server::{InferenceService, InferenceServiceServer};
pub use proto::{DataType, ModelInferRequest, ModelInferResponse, TensorData};
pub use service::SnoozeInferenceService;
