//! Triton Inference Server / Open Inference Protocol (OIP v2) gRPC service implementation.

#![allow(clippy::result_large_err)]

use crate::auth::{AuthIdentity, Role};
use crate::registry::ModelReplicaGroup;
use crate::server::grpc::convert::inference_error_to_status;
use crate::server::grpc::service::authenticate_grpc_metadata;
use crate::server::grpc::triton_proto::grpc_inference_service_server::GrpcInferenceService;
use crate::server::grpc::triton_proto::model_config::SchedulingChoice;
use crate::server::grpc::triton_proto::model_instance_group::Kind as InstanceKind;
use crate::server::grpc::triton_proto::model_metadata_response::TensorMetadata;
use crate::server::grpc::triton_proto::model_version_policy::{Latest, PolicyChoice};
use crate::server::grpc::triton_proto::repository_index_response::ModelIndex;
use crate::server::grpc::triton_proto::{
    CudaSharedMemoryRegisterRequest, CudaSharedMemoryRegisterResponse,
    CudaSharedMemoryStatusRequest, CudaSharedMemoryStatusResponse,
    CudaSharedMemoryUnregisterRequest, CudaSharedMemoryUnregisterResponse, DataType,
    InferStatistics, InferTensorContents, LogSettingsRequest, LogSettingsResponse,
    ModelConfig as TritonModelConfig, ModelConfigRequest, ModelConfigResponse,
    ModelDynamicBatching, ModelInferRequest, ModelInferResponse, ModelInput, ModelInstanceGroup,
    ModelMetadataRequest, ModelMetadataResponse, ModelOutput, ModelReadyRequest,
    ModelReadyResponse, ModelStatistics, ModelStatisticsRequest, ModelStatisticsResponse,
    ModelStreamInferResponse, ModelVersionPolicy, RepositoryIndexRequest, RepositoryIndexResponse,
    RepositoryModelLoadRequest, RepositoryModelLoadResponse, RepositoryModelUnloadRequest,
    RepositoryModelUnloadResponse, ServerLiveRequest, ServerLiveResponse, ServerMetadataRequest,
    ServerMetadataResponse, ServerReadyRequest, ServerReadyResponse, StatisticDuration,
    SystemSharedMemoryRegisterRequest, SystemSharedMemoryRegisterResponse,
    SystemSharedMemoryStatusRequest, SystemSharedMemoryStatusResponse,
    SystemSharedMemoryUnregisterRequest, SystemSharedMemoryUnregisterResponse, TraceSettingRequest,
    TraceSettingResponse,
};
use crate::server::http::AppState;
use crate::types::{ModelState, TensorPayload};
use std::collections::HashMap;
use std::time::Instant;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use tonic::{Request, Response, Status, Streaming};

/// Infers the Triton platform and backend identifiers from a SnoozeML model path and provider.
#[must_use]
pub fn infer_triton_platform_and_backend(
    model_path: &str,
    provider: Option<&str>,
) -> (&'static str, &'static str) {
    if model_path.starts_with("mock:") || model_path == "mock" {
        return ("mock", "mock");
    }
    if matches!(provider, Some("python" | "worker"))
        || model_path.ends_with("snooze_model.py")
        || model_path.ends_with("snooze_model.json")
    {
        return ("python", "worker");
    }
    if matches!(provider, Some("tree" | "tree-cpu" | "xgboost" | "gbdt"))
        || model_path.ends_with(".xgb")
        || model_path.ends_with(".xgb.json")
        || model_path.ends_with(".tree")
        || model_path.ends_with(".tree.json")
    {
        return ("xgboost_json", "hessboost");
    }
    if model_path.ends_with(".safetensors") || matches!(provider, Some("candle" | "safetensors")) {
        return ("pytorch_safetensors", "candle");
    }
    ("onnxruntime_onnx", "onnxruntime")
}

/// Validates and converts a Triton shape array (`&[i64]`) into `(Vec<usize>, usize)`.
fn validate_triton_shape(tensor_name: &str, shape: &[i64]) -> Result<(Vec<usize>, usize), Status> {
    if shape.is_empty() {
        return Err(Status::invalid_argument(format!(
            "Tensor '{tensor_name}' shape cannot be empty"
        )));
    }
    let mut dims = Vec::with_capacity(shape.len());
    for &d in shape {
        if d <= 0 {
            return Err(Status::invalid_argument(format!(
                "Tensor '{tensor_name}' dimension must be positive, got {d}"
            )));
        }
        let udim = usize::try_from(d).map_err(|_| {
            Status::invalid_argument(format!(
                "Tensor '{tensor_name}' dimension {d} overflows usize"
            ))
        })?;
        dims.push(udim);
    }
    let total = dims.iter().try_fold(1usize, |acc, &d| {
        acc.checked_mul(d).ok_or_else(|| {
            Status::invalid_argument(format!(
                "Tensor '{tensor_name}' dimensions overflow usize during multiplication"
            ))
        })
    })?;
    Ok((dims, total))
}

/// Decodes a single Triton input tensor from raw bytes or structured `InferTensorContents`.
fn decode_triton_input_tensor(
    input: &crate::server::grpc::triton_proto::model_infer_request::InferInputTensor,
    raw_bytes: Option<&[u8]>,
) -> Result<TensorPayload, Status> {
    let (shape_dims, total_elements) = validate_triton_shape(&input.name, &input.shape)?;
    let dtype = input.datatype.trim().to_ascii_uppercase();

    let floats: Vec<f32> = if let Some(raw) = raw_bytes {
        if input.contents.is_some() {
            return Err(Status::invalid_argument(format!(
                "Tensor '{}' cannot specify both InferInputTensor.contents and raw_input_contents",
                input.name
            )));
        }
        match dtype.as_str() {
            "FP32" | "" => {
                let expected = total_elements.checked_mul(4).ok_or_else(|| {
                    Status::invalid_argument(format!(
                        "Tensor '{}' byte length overflows usize",
                        input.name
                    ))
                })?;
                if raw.len() != expected {
                    return Err(Status::invalid_argument(format!(
                        "Tensor '{}' expects {expected} raw bytes for {total_elements} FP32 elements, got {}",
                        input.name,
                        raw.len()
                    )));
                }
                let (chunks, _) = raw.as_chunks::<4>();
                chunks.iter().map(|&c| f32::from_le_bytes(c)).collect()
            }
            "FP64" => {
                let expected = total_elements.checked_mul(8).ok_or_else(|| {
                    Status::invalid_argument(format!(
                        "Tensor '{}' byte length overflows usize",
                        input.name
                    ))
                })?;
                if raw.len() != expected {
                    return Err(Status::invalid_argument(format!(
                        "Tensor '{}' expects {expected} raw bytes for {total_elements} FP64 elements, got {}",
                        input.name,
                        raw.len()
                    )));
                }
                let (chunks, _) = raw.as_chunks::<8>();
                chunks
                    .iter()
                    .map(|&c| f64::from_le_bytes(c) as f32)
                    .collect()
            }
            "INT32" => {
                let expected = total_elements.checked_mul(4).ok_or_else(|| {
                    Status::invalid_argument(format!(
                        "Tensor '{}' byte length overflows usize",
                        input.name
                    ))
                })?;
                if raw.len() != expected {
                    return Err(Status::invalid_argument(format!(
                        "Tensor '{}' expects {expected} raw bytes for {total_elements} INT32 elements, got {}",
                        input.name,
                        raw.len()
                    )));
                }
                let (chunks, _) = raw.as_chunks::<4>();
                chunks
                    .iter()
                    .map(|&c| i32::from_le_bytes(c) as f32)
                    .collect()
            }
            "INT64" => {
                let expected = total_elements.checked_mul(8).ok_or_else(|| {
                    Status::invalid_argument(format!(
                        "Tensor '{}' byte length overflows usize",
                        input.name
                    ))
                })?;
                if raw.len() != expected {
                    return Err(Status::invalid_argument(format!(
                        "Tensor '{}' expects {expected} raw bytes for {total_elements} INT64 elements, got {}",
                        input.name,
                        raw.len()
                    )));
                }
                let (chunks, _) = raw.as_chunks::<8>();
                chunks
                    .iter()
                    .map(|&c| i64::from_le_bytes(c) as f32)
                    .collect()
            }
            "UINT8" | "BOOL" => {
                if raw.len() != total_elements {
                    return Err(Status::invalid_argument(format!(
                        "Tensor '{}' expects {total_elements} raw bytes for {dtype}, got {}",
                        input.name,
                        raw.len()
                    )));
                }
                raw.iter().map(|&b| f32::from(b)).collect()
            }
            other => {
                return Err(Status::invalid_argument(format!(
                    "Tensor '{}' specifies unsupported datatype '{other}'",
                    input.name
                )));
            }
        }
    } else {
        let contents = input.contents.as_ref().ok_or_else(|| {
            Status::invalid_argument(format!(
                "Tensor '{}' is missing both raw_input_contents and InferInputTensor.contents",
                input.name
            ))
        })?;
        match dtype.as_str() {
            "FP32" | "" => contents.fp32_contents.clone(),
            "FP64" => contents.fp64_contents.iter().map(|&v| v as f32).collect(),
            "INT32" | "INT16" | "INT8" => contents.int_contents.iter().map(|&v| v as f32).collect(),
            "INT64" => contents.int64_contents.iter().map(|&v| v as f32).collect(),
            "UINT32" | "UINT16" | "UINT8" => {
                contents.uint_contents.iter().map(|&v| v as f32).collect()
            }
            "UINT64" => contents.uint64_contents.iter().map(|&v| v as f32).collect(),
            "BOOL" => contents
                .bool_contents
                .iter()
                .map(|&b| if b { 1.0 } else { 0.0 })
                .collect(),
            other => {
                return Err(Status::invalid_argument(format!(
                    "Tensor '{}' specifies unsupported datatype '{other}'",
                    input.name
                )));
            }
        }
    };

    TensorPayload::from_shape_vec(input.name.clone(), shape_dims, floats)
        .map_err(inference_error_to_status)
}

/// Builds a Triton [`ModelStatistics`] protobuf message from a [`ModelReplicaGroup`].
pub async fn build_triton_model_statistics(group: &ModelReplicaGroup) -> ModelStatistics {
    let snap = group.stats_snapshot().await;
    ModelStatistics {
        name: group.id().to_string(),
        version: "1".to_string(),
        last_inference: snap.last_inference_ms,
        inference_count: snap.inference_count,
        execution_count: snap.execution_count,
        inference_stats: Some(InferStatistics {
            success: Some(StatisticDuration {
                count: snap.success_count,
                ns: snap.success_ns,
            }),
            fail: Some(StatisticDuration {
                count: snap.fail_count,
                ns: snap.fail_ns,
            }),
            queue: Some(StatisticDuration {
                count: snap.success_count,
                ns: snap.queue_ns,
            }),
            compute_input: Some(StatisticDuration { count: 0, ns: 0 }),
            compute_infer: Some(StatisticDuration {
                count: snap.execution_count,
                ns: snap.compute_infer_ns,
            }),
            compute_output: Some(StatisticDuration { count: 0, ns: 0 }),
            cache_hit: Some(StatisticDuration { count: 0, ns: 0 }),
            cache_miss: Some(StatisticDuration { count: 0, ns: 0 }),
        }),
        batch_stats: Vec::new(),
        memory_usage: Vec::new(),
        response_stats: HashMap::new(),
    }
}

/// Implementation of the NVIDIA Triton / Open Inference Protocol v2 `GRPCInferenceService`.
#[derive(Clone)]
pub struct TritonGrpcService {
    state: AppState,
}

impl TritonGrpcService {
    /// Creates a new [`TritonGrpcService`] backed by shared [`AppState`].
    #[must_use]
    pub fn new(state: AppState) -> Self {
        Self { state }
    }

    async fn execute_triton_infer(
        state: &AppState,
        caller: &AuthIdentity,
        req: ModelInferRequest,
    ) -> Result<ModelInferResponse, Status> {
        let model_name = req.model_name;
        if !crate::types::is_valid_model_id(&model_name) {
            return Err(Status::invalid_argument(format!(
                "Invalid model_name '{model_name}'. Model names must only contain alphanumeric characters, underscores, and hyphens."
            )));
        }

        if state.auth.as_ref().is_some_and(|a| a.is_enabled()) {
            tracing::debug!(
                caller = %caller.subject,
                role = %caller.role,
                model_name = %model_name,
                "Triton gRPC inference request received from authenticated caller"
            );
        }

        let use_raw = !req.raw_input_contents.is_empty();
        if use_raw && req.raw_input_contents.len() != req.inputs.len() {
            return Err(Status::invalid_argument(format!(
                "raw_input_contents count ({}) must match inputs count ({})",
                req.raw_input_contents.len(),
                req.inputs.len()
            )));
        }

        let mut inputs = Vec::with_capacity(req.inputs.len());
        for (idx, input_tensor) in req.inputs.iter().enumerate() {
            let raw_slice = if use_raw {
                req.raw_input_contents.get(idx).map(Vec::as_slice)
            } else {
                None
            };
            let payload = decode_triton_input_tensor(input_tensor, raw_slice)?;
            inputs.push(payload);
        }

        let start = Instant::now();
        let controller = state
            .registry
            .route_for_inference(&model_name)
            .await
            .map_err(inference_error_to_status)?;

        let infer_res = controller.infer(inputs).await;
        let elapsed = start.elapsed();
        let elapsed_us = u64::try_from(elapsed.as_micros()).unwrap_or(u64::MAX);

        match infer_res {
            Ok(output_tensors) => {
                state.metrics.record_inference(true, elapsed_us);

                // Filter outputs if requested by caller
                let selected_tensors: Vec<TensorPayload> = if req.outputs.is_empty() {
                    output_tensors
                } else {
                    let mut by_name: HashMap<String, TensorPayload> = output_tensors
                        .into_iter()
                        .map(|t| (t.name.clone(), t))
                        .collect();
                    let mut filtered = Vec::with_capacity(req.outputs.len());
                    for requested in &req.outputs {
                        let tensor = by_name.remove(&requested.name).ok_or_else(|| {
                            Status::invalid_argument(format!(
                                "Requested output tensor '{}' not produced by model '{model_name}'",
                                requested.name
                            ))
                        })?;
                        filtered.push(tensor);
                    }
                    filtered
                };

                let mut outputs = Vec::with_capacity(selected_tensors.len());
                let mut raw_output_contents = if use_raw {
                    Vec::with_capacity(selected_tensors.len())
                } else {
                    Vec::new()
                };

                for tensor in selected_tensors {
                    let (name, shape, floats) = tensor.into_components();
                    let shape_i64: Vec<i64> = shape.into_iter().map(|d| d as i64).collect();

                    let contents = if use_raw {
                        let bytes: &[u8] = bytemuck::cast_slice(&floats);
                        raw_output_contents.push(bytes.to_vec());
                        None
                    } else {
                        Some(InferTensorContents {
                            fp32_contents: floats,
                            ..Default::default()
                        })
                    };

                    outputs.push(
                        crate::server::grpc::triton_proto::model_infer_response::InferOutputTensor {
                            name,
                            datatype: "FP32".to_string(),
                            shape: shape_i64,
                            parameters: HashMap::new(),
                            contents,
                        },
                    );
                }

                let model_version = if req.model_version.is_empty() {
                    "1".to_string()
                } else {
                    req.model_version
                };

                Ok(ModelInferResponse {
                    model_name,
                    model_version,
                    id: req.id,
                    parameters: HashMap::new(),
                    outputs,
                    raw_output_contents,
                })
            }
            Err(err) => {
                state.metrics.record_inference(false, elapsed_us);
                Err(inference_error_to_status(err))
            }
        }
    }
}

#[tonic::async_trait]
impl GrpcInferenceService for TritonGrpcService {
    type ModelStreamInferStream = ReceiverStream<Result<ModelStreamInferResponse, Status>>;

    async fn server_live(
        &self,
        _request: Request<ServerLiveRequest>,
    ) -> Result<Response<ServerLiveResponse>, Status> {
        Ok(Response::new(ServerLiveResponse { live: true }))
    }

    async fn server_ready(
        &self,
        _request: Request<ServerReadyRequest>,
    ) -> Result<Response<ServerReadyResponse>, Status> {
        Ok(Response::new(ServerReadyResponse { ready: true }))
    }

    async fn model_ready(
        &self,
        request: Request<ModelReadyRequest>,
    ) -> Result<Response<ModelReadyResponse>, Status> {
        authenticate_grpc_metadata(&self.state, request.metadata(), Role::ReadOnly)?;
        let req = request.into_inner();
        if !crate::types::is_valid_model_id(&req.name) {
            return Err(Status::invalid_argument(format!(
                "Invalid model name '{}'",
                req.name
            )));
        }

        let ready = match self.state.registry.get_group(&req.name).await {
            Ok(group) => group.count_loaded_replicas().await > 0,
            Err(_) => false,
        };

        Ok(Response::new(ModelReadyResponse { ready }))
    }

    async fn server_metadata(
        &self,
        request: Request<ServerMetadataRequest>,
    ) -> Result<Response<ServerMetadataResponse>, Status> {
        authenticate_grpc_metadata(&self.state, request.metadata(), Role::ReadOnly)?;
        Ok(Response::new(ServerMetadataResponse {
            name: "snooze-ml".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            extensions: vec![
                "classification".to_string(),
                "model_repository".to_string(),
                "statistics".to_string(),
                "model_configuration".to_string(),
                "binary_tensor_data".to_string(),
            ],
        }))
    }

    async fn model_metadata(
        &self,
        request: Request<ModelMetadataRequest>,
    ) -> Result<Response<ModelMetadataResponse>, Status> {
        authenticate_grpc_metadata(&self.state, request.metadata(), Role::ReadOnly)?;
        let req = request.into_inner();
        if !crate::types::is_valid_model_id(&req.name) {
            return Err(Status::invalid_argument(format!(
                "Invalid model name '{}'",
                req.name
            )));
        }

        let group = self
            .state
            .registry
            .get_group(&req.name)
            .await
            .map_err(inference_error_to_status)?;
        let cfg = group.config();
        let (platform, _) =
            infer_triton_platform_and_backend(&cfg.onnx_path, cfg.provider.as_deref());

        Ok(Response::new(ModelMetadataResponse {
            name: cfg.id.clone(),
            versions: vec!["1".to_string()],
            platform: platform.to_string(),
            inputs: vec![TensorMetadata {
                name: "input".to_string(),
                datatype: "FP32".to_string(),
                shape: vec![-1, -1],
            }],
            outputs: vec![TensorMetadata {
                name: "output".to_string(),
                datatype: "FP32".to_string(),
                shape: vec![-1, -1],
            }],
        }))
    }

    async fn model_infer(
        &self,
        request: Request<ModelInferRequest>,
    ) -> Result<Response<ModelInferResponse>, Status> {
        let caller = authenticate_grpc_metadata(&self.state, request.metadata(), Role::Inference)?;
        let req = request.into_inner();
        let resp = Self::execute_triton_infer(&self.state, &caller, req).await?;
        Ok(Response::new(resp))
    }

    async fn model_stream_infer(
        &self,
        request: Request<Streaming<ModelInferRequest>>,
    ) -> Result<Response<Self::ModelStreamInferStream>, Status> {
        let caller = authenticate_grpc_metadata(&self.state, request.metadata(), Role::Inference)?;
        let mut in_stream = request.into_inner();
        let state = self.state.clone();

        let (tx, rx) = tokio::sync::mpsc::channel(32);

        tokio::spawn(async move {
            while let Some(item) = in_stream.next().await {
                match item {
                    Ok(req) => {
                        let stream_resp =
                            match Self::execute_triton_infer(&state, &caller, req).await {
                                Ok(infer_resp) => ModelStreamInferResponse {
                                    error_message: String::new(),
                                    infer_response: Some(infer_resp),
                                },
                                Err(status) => ModelStreamInferResponse {
                                    error_message: status.message().to_string(),
                                    infer_response: None,
                                },
                            };
                        if tx.send(Ok(stream_resp)).await.is_err() {
                            break;
                        }
                    }
                    Err(status) => {
                        let _ = tx.send(Err(status)).await;
                        break;
                    }
                }
            }
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }

    async fn model_config(
        &self,
        request: Request<ModelConfigRequest>,
    ) -> Result<Response<ModelConfigResponse>, Status> {
        authenticate_grpc_metadata(&self.state, request.metadata(), Role::ReadOnly)?;
        let req = request.into_inner();
        if !crate::types::is_valid_model_id(&req.name) {
            return Err(Status::invalid_argument(format!(
                "Invalid model name '{}'",
                req.name
            )));
        }

        let group = self
            .state
            .registry
            .get_group(&req.name)
            .await
            .map_err(inference_error_to_status)?;
        let cfg = group.config();
        let replica_count = i32::try_from(group.count_replicas().await).unwrap_or(1);
        let max_batch_size = i32::try_from(cfg.max_batch_size).unwrap_or(i32::MAX);
        let delay_us = u64::try_from(cfg.max_batch_delay.as_micros()).unwrap_or(u64::MAX);
        let (platform, backend) =
            infer_triton_platform_and_backend(&cfg.onnx_path, cfg.provider.as_deref());

        let kind = match cfg.provider.as_deref() {
            Some("cuda" | "gpu" | "tensorrt") => InstanceKind::Gpu as i32,
            Some("cpu") => InstanceKind::Cpu as i32,
            _ => InstanceKind::Auto as i32,
        };

        let triton_cfg = TritonModelConfig {
            name: cfg.id.clone(),
            platform: platform.to_string(),
            backend: backend.to_string(),
            runtime: String::new(),
            version_policy: Some(ModelVersionPolicy {
                policy_choice: Some(PolicyChoice::Latest(Latest { num_versions: 1 })),
            }),
            max_batch_size,
            input: vec![ModelInput {
                name: "input".to_string(),
                data_type: DataType::TypeFp32 as i32,
                dims: vec![-1],
                ..Default::default()
            }],
            output: vec![ModelOutput {
                name: "output".to_string(),
                data_type: DataType::TypeFp32 as i32,
                dims: vec![-1],
                ..Default::default()
            }],
            instance_group: vec![ModelInstanceGroup {
                name: format!("{}_0", cfg.id),
                kind,
                count: replica_count,
                ..Default::default()
            }],
            default_model_filename: cfg.onnx_path.clone(),
            scheduling_choice: Some(SchedulingChoice::DynamicBatching(ModelDynamicBatching {
                preferred_batch_size: vec![max_batch_size],
                max_queue_delay_microseconds: delay_us,
                preserve_ordering: true,
                ..Default::default()
            })),
            ..Default::default()
        };

        Ok(Response::new(ModelConfigResponse {
            config: Some(triton_cfg),
        }))
    }

    async fn model_statistics(
        &self,
        request: Request<ModelStatisticsRequest>,
    ) -> Result<Response<ModelStatisticsResponse>, Status> {
        authenticate_grpc_metadata(&self.state, request.metadata(), Role::ReadOnly)?;
        let req = request.into_inner();

        let mut model_stats = Vec::new();
        if req.name.is_empty() {
            let groups = self.state.registry.all_groups().await;
            for group in groups {
                model_stats.push(build_triton_model_statistics(&group).await);
            }
        } else {
            if !crate::types::is_valid_model_id(&req.name) {
                return Err(Status::invalid_argument(format!(
                    "Invalid model name '{}'",
                    req.name
                )));
            }
            let group = self
                .state
                .registry
                .get_group(&req.name)
                .await
                .map_err(inference_error_to_status)?;
            model_stats.push(build_triton_model_statistics(&group).await);
        }

        Ok(Response::new(ModelStatisticsResponse { model_stats }))
    }

    async fn repository_index(
        &self,
        request: Request<RepositoryIndexRequest>,
    ) -> Result<Response<RepositoryIndexResponse>, Status> {
        authenticate_grpc_metadata(&self.state, request.metadata(), Role::ReadOnly)?;
        let req = request.into_inner();
        let groups = self.state.registry.all_groups().await;
        let mut models = Vec::with_capacity(groups.len());

        for group in groups {
            let loaded_count = group.count_loaded_replicas().await;
            let primary_state = match group.primary().await {
                Ok(p) => p.metadata().await.state,
                Err(_) => ModelState::Idle,
            };
            let is_ready =
                loaded_count > 0 || matches!(primary_state, ModelState::Ready | ModelState::Busy);

            if req.ready && !is_ready {
                continue;
            }

            let (state_str, reason) = if is_ready {
                ("READY".to_string(), String::new())
            } else if primary_state == ModelState::Loading {
                ("LOADING".to_string(), "Loading model weights".to_string())
            } else {
                (
                    "UNAVAILABLE".to_string(),
                    "Unloaded (scale-to-zero)".to_string(),
                )
            };

            models.push(ModelIndex {
                name: group.id().to_string(),
                version: "1".to_string(),
                state: state_str,
                reason,
            });
        }

        Ok(Response::new(RepositoryIndexResponse { models }))
    }

    async fn repository_model_load(
        &self,
        request: Request<RepositoryModelLoadRequest>,
    ) -> Result<Response<RepositoryModelLoadResponse>, Status> {
        authenticate_grpc_metadata(&self.state, request.metadata(), Role::Admin)?;
        let req = request.into_inner();
        if !crate::types::is_valid_model_id(&req.model_name) {
            return Err(Status::invalid_argument(format!(
                "Invalid model_name '{}'",
                req.model_name
            )));
        }

        if self
            .state
            .registry
            .get_group(&req.model_name)
            .await
            .is_err()
        {
            let _ = self
                .state
                .registry
                .refresh_from_dir(&self.state.models_dir)
                .await;
        }

        self.state
            .registry
            .ensure_loaded_with_deallocation(&req.model_name)
            .await
            .map_err(inference_error_to_status)?;

        Ok(Response::new(RepositoryModelLoadResponse {}))
    }

    async fn repository_model_unload(
        &self,
        request: Request<RepositoryModelUnloadRequest>,
    ) -> Result<Response<RepositoryModelUnloadResponse>, Status> {
        authenticate_grpc_metadata(&self.state, request.metadata(), Role::Admin)?;
        let req = request.into_inner();
        if !crate::types::is_valid_model_id(&req.model_name) {
            return Err(Status::invalid_argument(format!(
                "Invalid model_name '{}'",
                req.model_name
            )));
        }

        let group = self
            .state
            .registry
            .get_group(&req.model_name)
            .await
            .map_err(inference_error_to_status)?;
        group.unload_all().await;

        Ok(Response::new(RepositoryModelUnloadResponse {}))
    }

    async fn system_shared_memory_status(
        &self,
        request: Request<SystemSharedMemoryStatusRequest>,
    ) -> Result<Response<SystemSharedMemoryStatusResponse>, Status> {
        authenticate_grpc_metadata(&self.state, request.metadata(), Role::Admin)?;
        Err(Status::unimplemented(
            "SystemSharedMemoryStatus is out of scope and not supported by SnoozeML",
        ))
    }

    async fn system_shared_memory_register(
        &self,
        request: Request<SystemSharedMemoryRegisterRequest>,
    ) -> Result<Response<SystemSharedMemoryRegisterResponse>, Status> {
        authenticate_grpc_metadata(&self.state, request.metadata(), Role::Admin)?;
        Err(Status::unimplemented(
            "SystemSharedMemoryRegister is out of scope and not supported by SnoozeML",
        ))
    }

    async fn system_shared_memory_unregister(
        &self,
        request: Request<SystemSharedMemoryUnregisterRequest>,
    ) -> Result<Response<SystemSharedMemoryUnregisterResponse>, Status> {
        authenticate_grpc_metadata(&self.state, request.metadata(), Role::Admin)?;
        Err(Status::unimplemented(
            "SystemSharedMemoryUnregister is out of scope and not supported by SnoozeML",
        ))
    }

    async fn cuda_shared_memory_status(
        &self,
        request: Request<CudaSharedMemoryStatusRequest>,
    ) -> Result<Response<CudaSharedMemoryStatusResponse>, Status> {
        authenticate_grpc_metadata(&self.state, request.metadata(), Role::Admin)?;
        Err(Status::unimplemented(
            "CudaSharedMemoryStatus is out of scope and not supported by SnoozeML",
        ))
    }

    async fn cuda_shared_memory_register(
        &self,
        request: Request<CudaSharedMemoryRegisterRequest>,
    ) -> Result<Response<CudaSharedMemoryRegisterResponse>, Status> {
        authenticate_grpc_metadata(&self.state, request.metadata(), Role::Admin)?;
        Err(Status::unimplemented(
            "CudaSharedMemoryRegister is out of scope and not supported by SnoozeML",
        ))
    }

    async fn cuda_shared_memory_unregister(
        &self,
        request: Request<CudaSharedMemoryUnregisterRequest>,
    ) -> Result<Response<CudaSharedMemoryUnregisterResponse>, Status> {
        authenticate_grpc_metadata(&self.state, request.metadata(), Role::Admin)?;
        Err(Status::unimplemented(
            "CudaSharedMemoryUnregister is out of scope and not supported by SnoozeML",
        ))
    }

    async fn trace_setting(
        &self,
        request: Request<TraceSettingRequest>,
    ) -> Result<Response<TraceSettingResponse>, Status> {
        authenticate_grpc_metadata(&self.state, request.metadata(), Role::Admin)?;
        Err(Status::unimplemented(
            "TraceSetting is out of scope and not supported by SnoozeML",
        ))
    }

    async fn log_settings(
        &self,
        request: Request<LogSettingsRequest>,
    ) -> Result<Response<LogSettingsResponse>, Status> {
        authenticate_grpc_metadata(&self.state, request.metadata(), Role::Admin)?;
        Err(Status::unimplemented(
            "LogSettings is out of scope and not supported by SnoozeML",
        ))
    }
}
