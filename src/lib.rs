//! # SnoozeML: High-Performance Bare-Metal ONNX Inference Daemon
//!
//! SnoozeML is an inference service engineered for deep learning workloads.
//! It features dynamic micro-batching, LRU scale-to-zero memory reclamation, and unified REST
//! and telemetry endpoints.

#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![warn(missing_docs)]
#![allow(clippy::result_large_err)]

pub mod auth;
pub mod backend;
pub mod batcher;
pub mod cli;
pub mod config;
pub mod model;
pub mod registry;
pub mod routing;
pub mod server;
#[cfg(feature = "streams")]
pub mod stream;
pub mod sweeper;
pub mod types;

pub use auth::{
    auth_middleware, require_role, ApiKeyConfig, AuthConfig, AuthError, AuthIdentity, AuthMethod,
    AuthService, JwtAlgorithm, JwtClaims, JwtConfig, Role,
};
pub use backend::ModelBackend;
pub use batcher::{DynamicBatcher, DynamicBatcherConfig};
pub use config::{
    default_models_dir, GlobalModelDefaults, ModelDirectoryConfig, ModelEntryConfig,
    DEFAULT_MODELS_DIR, ENV_MODELS_DIR, ENV_MODELS_DIR_FALLBACK,
};
pub use model::{compute_replica_idle_timeout, ModelConfig, ModelController};
pub use registry::{ModelRegistry, ModelReplicaGroup};
pub use routing::{
    create_routing_strategy, LeastQueueRouting, RandomRouting, ReplicaRouter, RoundRobinRouting,
    RoutingStrategy, RoutingStrategyType,
};
pub use server::{run_server, ServerConfig, TlsConfig};
#[cfg(feature = "streams")]
pub use stream::{
    driver::{ConsumedMessage, MessageAcknowledger, StreamDriver},
    message::{StreamInferRequest, StreamInferResponse, StreamTensorData},
    mock_stream::MockStreamDriver,
    StreamBindingConfig, StreamSupervisor, StreamWireFormat,
};
pub use sweeper::spawn_scale_to_zero_sweeper;
pub use types::{InferenceError, InferenceJob, ModelMetadata, ModelState, TensorPayload};
