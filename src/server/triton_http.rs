//! Open Inference Protocol (OIP v2) and NVIDIA Triton HTTP REST endpoint handlers.

use crate::auth::{AuthIdentity, Role};
use crate::registry::ModelReplicaGroup;
use crate::server::grpc::triton::infer_triton_platform_and_backend;
use crate::server::http::{AppError, AppState, ErrorResponse};
use crate::types::{InferenceError, ModelState, TensorPayload};
use axum::{
    body::Bytes,
    extract::{Extension, Path, State},
    http::StatusCode,
    middleware::from_fn,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

/// Input tensor descriptor in an OIP v2 HTTP inference request.
#[derive(Debug, Deserialize)]
pub struct TritonHttpInputTensor {
    /// Name of the input tensor.
    pub name: String,
    /// Shape dimensions of the tensor.
    pub shape: Vec<i64>,
    /// Data type identifier (e.g. `"FP32"`, `"INT32"`, `"INT64"`, `"FP64"`, `"BOOL"`).
    pub datatype: String,
    /// Optional tensor-level parameters.
    #[serde(default)]
    pub parameters: HashMap<String, serde_json::Value>,
    /// Flattened or nested JSON array containing tensor elements.
    pub data: serde_json::Value,
}

/// Requested output tensor filter in an OIP v2 HTTP inference request.
#[derive(Debug, Deserialize)]
pub struct TritonHttpRequestedOutput {
    /// Name of the requested output tensor.
    pub name: String,
    /// Optional output parameters.
    #[serde(default)]
    pub parameters: HashMap<String, serde_json::Value>,
}

/// Incoming JSON payload for OIP v2 HTTP inference (`POST /v2/models/{name}/infer`).
#[derive(Debug, Deserialize)]
pub struct TritonHttpInferRequest {
    /// Optional request identifier echoed in the response.
    #[serde(default)]
    pub id: Option<String>,
    /// Optional request-level parameters.
    #[serde(default)]
    pub parameters: HashMap<String, serde_json::Value>,
    /// Input tensors.
    pub inputs: Vec<TritonHttpInputTensor>,
    /// Optional output tensor filter list.
    #[serde(default)]
    pub outputs: Vec<TritonHttpRequestedOutput>,
}

/// Output tensor descriptor in an OIP v2 HTTP inference response.
#[derive(Debug, Serialize)]
pub struct TritonHttpOutputTensor {
    /// Name of the output tensor.
    pub name: String,
    /// Data type identifier (`"FP32"`).
    pub datatype: String,
    /// Dimensions of the output tensor.
    pub shape: Vec<usize>,
    /// Flattened tensor data in row-major order.
    pub data: Vec<f32>,
}

/// Outgoing JSON payload for OIP v2 HTTP inference.
#[derive(Debug, Serialize)]
pub struct TritonHttpInferResponse {
    /// Name of the model that executed the request.
    pub model_name: String,
    /// Model version string.
    pub model_version: String,
    /// Echoed request identifier if provided by caller.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Produced output tensors.
    pub outputs: Vec<TritonHttpOutputTensor>,
}

/// Request payload for `POST /v2/repository/index`.
#[derive(Debug, Default, Deserialize)]
pub struct TritonHttpRepoIndexRequest {
    /// Optional repository name filter.
    #[serde(default)]
    pub repository_name: String,
    /// When `true`, only returns models currently loaded and ready for inference.
    #[serde(default)]
    pub ready: bool,
}

/// Entry returned in `POST /v2/repository/index`.
#[derive(Debug, Serialize)]
pub struct TritonHttpRepoIndexEntry {
    /// Model identifier.
    pub name: String,
    /// Model version (`"1"`).
    #[serde(skip_serializing_if = "String::is_empty")]
    pub version: String,
    /// Model state (`"READY"`, `"LOADING"`, `"UNAVAILABLE"`).
    pub state: String,
    /// Optional human-readable reason for the current state.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub reason: String,
}

/// Flattens an arbitrarily nested or flat JSON array of numbers/booleans into `Vec<f32>`.
fn flatten_json_tensor_data(
    tensor_name: &str,
    val: &serde_json::Value,
    out: &mut Vec<f32>,
) -> Result<(), InferenceError> {
    match val {
        serde_json::Value::Array(items) => {
            for item in items {
                flatten_json_tensor_data(tensor_name, item, out)?;
            }
            Ok(())
        }
        serde_json::Value::Number(num) => {
            let f = num.as_f64().ok_or_else(|| {
                InferenceError::InvalidInput(format!(
                    "Tensor '{tensor_name}' contains non-numeric element"
                ))
            })? as f32;
            out.push(f);
            Ok(())
        }
        serde_json::Value::Bool(b) => {
            out.push(if *b { 1.0 } else { 0.0 });
            Ok(())
        }
        _ => Err(InferenceError::InvalidInput(format!(
            "Tensor '{tensor_name}' data must be a JSON array of numbers or booleans"
        ))),
    }
}

/// Builds public OIP v2 health routes (`/v2/health/live`, `/v2/health/ready`).
pub fn create_triton_public_routes() -> Router<AppState> {
    Router::new()
        .route("/v2/health/live", get(v2_health_live_handler))
        .route("/v2/health/ready", get(v2_health_ready_handler))
}

/// Builds `Role::ReadOnly` OIP v2 metadata, config, readiness, stats, and repository index routes.
pub fn create_triton_readonly_routes() -> Router<AppState> {
    Router::new()
        .route("/v2", get(v2_server_metadata_handler))
        .route("/v2/models/stats", get(v2_all_models_stats_handler))
        .route("/v2/models/:model_name", get(v2_model_metadata_handler))
        .route(
            "/v2/models/:model_name/versions/:version",
            get(v2_model_version_metadata_handler),
        )
        .route("/v2/models/:model_name/ready", get(v2_model_ready_handler))
        .route(
            "/v2/models/:model_name/versions/:version/ready",
            get(v2_model_version_ready_handler),
        )
        .route(
            "/v2/models/:model_name/config",
            get(v2_model_config_handler),
        )
        .route(
            "/v2/models/:model_name/versions/:version/config",
            get(v2_model_version_config_handler),
        )
        .route("/v2/models/:model_name/stats", get(v2_model_stats_handler))
        .route(
            "/v2/models/:model_name/versions/:version/stats",
            get(v2_model_version_stats_handler),
        )
        .route("/v2/repository/index", post(v2_repository_index_handler))
        .layer(from_fn(crate::auth::require_role(Role::ReadOnly)))
}

/// Builds `Role::Inference` OIP v2 inference routes.
pub fn create_triton_infer_routes() -> Router<AppState> {
    Router::new()
        .route("/v2/models/:model_name/infer", post(v2_model_infer_handler))
        .route(
            "/v2/models/:model_name/versions/:version/infer",
            post(v2_model_version_infer_handler),
        )
        .layer(axum::extract::DefaultBodyLimit::max(128 * 1024 * 1024))
        .layer(from_fn(crate::auth::require_role(Role::Inference)))
}

/// Builds `Role::Admin` OIP v2 repository load/unload and out-of-scope `501 Not Implemented` routes.
pub fn create_triton_admin_routes() -> Router<AppState> {
    Router::new()
        .route(
            "/v2/repository/models/:model_name/load",
            post(v2_repository_load_handler),
        )
        .route(
            "/v2/repository/models/:model_name/unload",
            post(v2_repository_unload_handler),
        )
        .route(
            "/v2/systemsharedmemory/status",
            get(v2_unimplemented_handler).post(v2_unimplemented_handler),
        )
        .route(
            "/v2/systemsharedmemory/region/:name/register",
            post(v2_unimplemented_handler),
        )
        .route(
            "/v2/systemsharedmemory/region/:name/unregister",
            post(v2_unimplemented_handler),
        )
        .route(
            "/v2/systemsharedmemory/unregister",
            post(v2_unimplemented_handler),
        )
        .route(
            "/v2/cudasharedmemory/status",
            get(v2_unimplemented_handler).post(v2_unimplemented_handler),
        )
        .route(
            "/v2/cudasharedmemory/region/:name/register",
            post(v2_unimplemented_handler),
        )
        .route(
            "/v2/cudasharedmemory/region/:name/unregister",
            post(v2_unimplemented_handler),
        )
        .route(
            "/v2/cudasharedmemory/unregister",
            post(v2_unimplemented_handler),
        )
        .route(
            "/v2/trace",
            get(v2_unimplemented_handler).post(v2_unimplemented_handler),
        )
        .route(
            "/v2/models/:model_name/trace",
            get(v2_unimplemented_handler).post(v2_unimplemented_handler),
        )
        .route(
            "/v2/logging",
            get(v2_unimplemented_handler).post(v2_unimplemented_handler),
        )
        .layer(from_fn(crate::auth::require_role(Role::Admin)))
}

async fn v2_health_live_handler() -> StatusCode {
    StatusCode::OK
}

async fn v2_health_ready_handler() -> StatusCode {
    StatusCode::OK
}

async fn v2_server_metadata_handler() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "name": "snooze-ml",
        "version": env!("CARGO_PKG_VERSION"),
        "extensions": [
            "classification",
            "model_repository",
            "statistics",
            "model_configuration",
            "binary_tensor_data"
        ]
    }))
}

async fn v2_model_ready_handler(
    State(state): State<AppState>,
    Path(model_name): Path<String>,
) -> Response {
    check_model_ready(&state, &model_name).await
}

async fn v2_model_version_ready_handler(
    State(state): State<AppState>,
    Path((model_name, _version)): Path<(String, String)>,
) -> Response {
    check_model_ready(&state, &model_name).await
}

async fn check_model_ready(state: &AppState, model_name: &str) -> Response {
    if !crate::types::is_valid_model_id(model_name) {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: format!("Invalid model name '{model_name}'"),
            }),
        )
            .into_response();
    }

    match state.registry.get_group(model_name).await {
        Ok(group) => {
            if group.count_loaded_replicas().await > 0 {
                StatusCode::OK.into_response()
            } else {
                (
                    StatusCode::BAD_REQUEST,
                    Json(ErrorResponse {
                        error: format!("Model '{model_name}' is not ready (unloaded)"),
                    }),
                )
                    .into_response()
            }
        }
        Err(_) => (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: format!("Model '{model_name}' is not registered"),
            }),
        )
            .into_response(),
    }
}

async fn v2_model_metadata_handler(
    State(state): State<AppState>,
    Path(model_name): Path<String>,
) -> Result<Json<serde_json::Value>, AppError> {
    build_model_metadata_json(&state, &model_name).await
}

async fn v2_model_version_metadata_handler(
    State(state): State<AppState>,
    Path((model_name, _version)): Path<(String, String)>,
) -> Result<Json<serde_json::Value>, AppError> {
    build_model_metadata_json(&state, &model_name).await
}

async fn build_model_metadata_json(
    state: &AppState,
    model_name: &str,
) -> Result<Json<serde_json::Value>, AppError> {
    if !crate::types::is_valid_model_id(model_name) {
        return Err(AppError::from(InferenceError::InvalidInput(format!(
            "Invalid model name '{model_name}'"
        ))));
    }
    let group = state.registry.get_group(model_name).await?;
    let cfg = group.config();
    let (platform, _) = infer_triton_platform_and_backend(&cfg.onnx_path, cfg.provider.as_deref());

    Ok(Json(serde_json::json!({
        "name": cfg.id,
        "versions": ["1"],
        "platform": platform,
        "inputs": [
            {
                "name": "input",
                "datatype": "FP32",
                "shape": [-1, -1]
            }
        ],
        "outputs": [
            {
                "name": "output",
                "datatype": "FP32",
                "shape": [-1, -1]
            }
        ]
    })))
}

async fn v2_model_config_handler(
    State(state): State<AppState>,
    Path(model_name): Path<String>,
) -> Result<Json<serde_json::Value>, AppError> {
    build_model_config_json(&state, &model_name).await
}

async fn v2_model_version_config_handler(
    State(state): State<AppState>,
    Path((model_name, _version)): Path<(String, String)>,
) -> Result<Json<serde_json::Value>, AppError> {
    build_model_config_json(&state, &model_name).await
}

async fn build_model_config_json(
    state: &AppState,
    model_name: &str,
) -> Result<Json<serde_json::Value>, AppError> {
    if !crate::types::is_valid_model_id(model_name) {
        return Err(AppError::from(InferenceError::InvalidInput(format!(
            "Invalid model name '{model_name}'"
        ))));
    }
    let group = state.registry.get_group(model_name).await?;
    let cfg = group.config();
    let replica_count = group.count_replicas().await;
    let delay_us = u64::try_from(cfg.max_batch_delay.as_micros()).unwrap_or(u64::MAX);
    let (platform, backend) =
        infer_triton_platform_and_backend(&cfg.onnx_path, cfg.provider.as_deref());

    let kind = match cfg.provider.as_deref() {
        Some("cuda" | "gpu" | "tensorrt") => "KIND_GPU",
        Some("cpu") => "KIND_CPU",
        _ => "KIND_AUTO",
    };

    Ok(Json(serde_json::json!({
        "name": cfg.id,
        "platform": platform,
        "backend": backend,
        "max_batch_size": cfg.max_batch_size,
        "input": [
            {
                "name": "input",
                "data_type": "TYPE_FP32",
                "dims": [-1]
            }
        ],
        "output": [
            {
                "name": "output",
                "data_type": "TYPE_FP32",
                "dims": [-1]
            }
        ],
        "dynamic_batching": {
            "preferred_batch_size": [cfg.max_batch_size],
            "max_queue_delay_microseconds": delay_us,
            "preserve_ordering": true
        },
        "instance_group": [
            {
                "name": format!("{}_0", cfg.id),
                "kind": kind,
                "count": replica_count
            }
        ],
        "default_model_filename": cfg.onnx_path
    })))
}

fn format_group_stats_json(
    group: &ModelReplicaGroup,
    snap: &crate::model::ModelStatsSnapshot,
) -> serde_json::Value {
    serde_json::json!({
        "name": group.id(),
        "version": "1",
        "last_inference": snap.last_inference_ms,
        "inference_count": snap.inference_count,
        "execution_count": snap.execution_count,
        "inference_stats": {
            "success": { "count": snap.success_count, "ns": snap.success_ns },
            "fail": { "count": snap.fail_count, "ns": snap.fail_ns },
            "queue": { "count": snap.success_count, "ns": snap.queue_ns },
            "compute_input": { "count": 0, "ns": 0 },
            "compute_infer": { "count": snap.execution_count, "ns": snap.compute_infer_ns },
            "compute_output": { "count": 0, "ns": 0 },
            "cache_hit": { "count": 0, "ns": 0 },
            "cache_miss": { "count": 0, "ns": 0 }
        },
        "batch_stats": []
    })
}

async fn v2_all_models_stats_handler(State(state): State<AppState>) -> Json<serde_json::Value> {
    let groups = state.registry.all_groups().await;
    let mut stats = Vec::with_capacity(groups.len());
    for group in groups {
        let snap = group.stats_snapshot().await;
        stats.push(format_group_stats_json(&group, &snap));
    }
    Json(serde_json::json!({ "model_stats": stats }))
}

async fn v2_model_stats_handler(
    State(state): State<AppState>,
    Path(model_name): Path<String>,
) -> Result<Json<serde_json::Value>, AppError> {
    build_single_model_stats_json(&state, &model_name).await
}

async fn v2_model_version_stats_handler(
    State(state): State<AppState>,
    Path((model_name, _version)): Path<(String, String)>,
) -> Result<Json<serde_json::Value>, AppError> {
    build_single_model_stats_json(&state, &model_name).await
}

async fn build_single_model_stats_json(
    state: &AppState,
    model_name: &str,
) -> Result<Json<serde_json::Value>, AppError> {
    if !crate::types::is_valid_model_id(model_name) {
        return Err(AppError::from(InferenceError::InvalidInput(format!(
            "Invalid model name '{model_name}'"
        ))));
    }
    let group = state.registry.get_group(model_name).await?;
    let snap = group.stats_snapshot().await;
    Ok(Json(serde_json::json!({
        "model_stats": [format_group_stats_json(&group, &snap)]
    })))
}

async fn v2_repository_index_handler(
    State(state): State<AppState>,
    body: Bytes,
) -> Result<Json<Vec<TritonHttpRepoIndexEntry>>, AppError> {
    let req: TritonHttpRepoIndexRequest = if body.is_empty() {
        TritonHttpRepoIndexRequest::default()
    } else {
        serde_json::from_slice(&body).map_err(|e| {
            InferenceError::InvalidInput(format!("Invalid repository index JSON body: {e}"))
        })?
    };

    let groups = state.registry.all_groups().await;
    let mut entries = Vec::with_capacity(groups.len());

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

        entries.push(TritonHttpRepoIndexEntry {
            name: group.id().to_string(),
            version: "1".to_string(),
            state: state_str,
            reason,
        });
    }

    Ok(Json(entries))
}

async fn v2_repository_load_handler(
    State(state): State<AppState>,
    Path(model_name): Path<String>,
) -> Result<Json<serde_json::Value>, AppError> {
    if !crate::types::is_valid_model_id(&model_name) {
        return Err(AppError::from(InferenceError::InvalidInput(format!(
            "Invalid model name '{model_name}'"
        ))));
    }

    if state.registry.get_group(&model_name).await.is_err() {
        let _ = state.registry.refresh_from_dir(&state.models_dir).await;
    }

    state
        .registry
        .ensure_loaded_with_deallocation(&model_name)
        .await?;

    Ok(Json(serde_json::json!({})))
}

async fn v2_repository_unload_handler(
    State(state): State<AppState>,
    Path(model_name): Path<String>,
) -> Result<Json<serde_json::Value>, AppError> {
    if !crate::types::is_valid_model_id(&model_name) {
        return Err(AppError::from(InferenceError::InvalidInput(format!(
            "Invalid model name '{model_name}'"
        ))));
    }

    let group = state.registry.get_group(&model_name).await?;
    group.unload_all().await;

    Ok(Json(serde_json::json!({})))
}

async fn v2_model_infer_handler(
    State(state): State<AppState>,
    Path(model_name): Path<String>,
    auth: Option<Extension<AuthIdentity>>,
    Json(payload): Json<TritonHttpInferRequest>,
) -> Result<Json<TritonHttpInferResponse>, AppError> {
    execute_v2_http_infer(state, model_name, "1".to_string(), auth, payload).await
}

async fn v2_model_version_infer_handler(
    State(state): State<AppState>,
    Path((model_name, version)): Path<(String, String)>,
    auth: Option<Extension<AuthIdentity>>,
    Json(payload): Json<TritonHttpInferRequest>,
) -> Result<Json<TritonHttpInferResponse>, AppError> {
    execute_v2_http_infer(state, model_name, version, auth, payload).await
}

async fn execute_v2_http_infer(
    state: AppState,
    model_name: String,
    model_version: String,
    auth: Option<Extension<AuthIdentity>>,
    payload: TritonHttpInferRequest,
) -> Result<Json<TritonHttpInferResponse>, AppError> {
    if !crate::types::is_valid_model_id(&model_name) {
        return Err(AppError::from(InferenceError::InvalidInput(format!(
            "Invalid model name '{model_name}'"
        ))));
    }

    if let Some(Extension(ref id)) = auth {
        tracing::debug!(
            caller = %id.subject,
            role = %id.role,
            model_name = %model_name,
            "OIP v2 HTTP inference request received from authenticated caller"
        );
    }

    let mut inputs = Vec::with_capacity(payload.inputs.len());
    for input in payload.inputs {
        if input.shape.is_empty() {
            return Err(AppError::from(InferenceError::InvalidInput(format!(
                "Tensor '{}' shape cannot be empty",
                input.name
            ))));
        }
        let mut shape_dims = Vec::with_capacity(input.shape.len());
        for d in input.shape {
            if d <= 0 {
                return Err(AppError::from(InferenceError::InvalidInput(format!(
                    "Tensor '{}' dimension must be positive, got {d}",
                    input.name
                ))));
            }
            let udim = usize::try_from(d).map_err(|_| {
                InferenceError::InvalidInput(format!(
                    "Tensor '{}' dimension {d} overflows usize",
                    input.name
                ))
            })?;
            shape_dims.push(udim);
        }

        let total_elements = shape_dims.iter().try_fold(1usize, |acc, &d| {
            acc.checked_mul(d).ok_or_else(|| {
                InferenceError::InvalidInput(format!(
                    "Tensor '{}' dimensions overflow usize",
                    input.name
                ))
            })
        })?;

        let mut floats = Vec::with_capacity(total_elements);
        flatten_json_tensor_data(&input.name, &input.data, &mut floats)?;

        let tensor = TensorPayload::from_shape_vec(input.name, shape_dims, floats)?;
        inputs.push(tensor);
    }

    let start = Instant::now();
    let controller = state.registry.route_for_inference(&model_name).await?;
    let infer_res = controller.infer(inputs).await;
    let elapsed_us = u64::try_from(start.elapsed().as_micros()).unwrap_or(u64::MAX);

    match infer_res {
        Ok(output_tensors) => {
            state.metrics.record_inference(true, elapsed_us);

            let selected_tensors: Vec<TensorPayload> = if payload.outputs.is_empty() {
                output_tensors
            } else {
                let mut by_name: HashMap<String, TensorPayload> = output_tensors
                    .into_iter()
                    .map(|t| (t.name.clone(), t))
                    .collect();
                let mut filtered = Vec::with_capacity(payload.outputs.len());
                for req_out in &payload.outputs {
                    let tensor = by_name.remove(&req_out.name).ok_or_else(|| {
                        InferenceError::InvalidInput(format!(
                            "Requested output tensor '{}' not produced by model '{model_name}'",
                            req_out.name
                        ))
                    })?;
                    filtered.push(tensor);
                }
                filtered
            };

            let mut outputs = Vec::with_capacity(selected_tensors.len());
            for tensor in selected_tensors {
                let (name, shape, data) = tensor.into_components();
                outputs.push(TritonHttpOutputTensor {
                    name,
                    datatype: "FP32".to_string(),
                    shape,
                    data,
                });
            }

            Ok(Json(TritonHttpInferResponse {
                model_name,
                model_version,
                id: payload.id,
                outputs,
            }))
        }
        Err(err) => {
            state.metrics.record_inference(false, elapsed_us);
            Err(AppError::from(err))
        }
    }
}

async fn v2_unimplemented_handler() -> Response {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(ErrorResponse {
            error: "This Triton shared-memory/trace/logging endpoint is out of scope and not implemented by SnoozeML".to_string(),
        }),
    )
        .into_response()
}
