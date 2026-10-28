//! gRPC serving subsystem for SnoozeML.
//!
//! Provides high-performance tensor inference (SnoozeML v1 and Triton / Open Inference Protocol v2)
//! and standard `grpc.health.v1` health checking.

pub mod convert;
pub mod service;
pub mod triton;

#[allow(missing_docs)]
pub mod proto {
    //! Generated Protocol Buffer bindings for SnoozeML v1 services and DTOs.
    tonic::include_proto!("snoozeml.v1");
}

#[allow(missing_docs, clippy::all)]
pub mod triton_proto {
    //! Generated Protocol Buffer bindings for NVIDIA Triton / Open Inference Protocol v2.
    tonic::include_proto!("inference");
}

pub use proto::inference_service_server::{InferenceService, InferenceServiceServer};
pub use proto::{DataType, ModelInferRequest, ModelInferResponse, TensorData};
pub use service::SnoozeInferenceService;
pub use triton::TritonGrpcService;
pub use triton_proto::grpc_inference_service_server::{
    GrpcInferenceService, GrpcInferenceServiceServer,
};
