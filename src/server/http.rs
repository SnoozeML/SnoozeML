//! HTTP / REST API routes implemented with Axum.

use crate::auth::{auth_middleware, require_role, AuthIdentity, AuthService, Role};
use crate::registry::ModelRegistry;
use crate::server::metrics::ServerMetrics;
use crate::types::{InferenceError, ModelMetadata, TensorPayload};
use axum::{
    extract::{Extension, Path, State},
    http::StatusCode,
    middleware::{from_fn, from_fn_with_state},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tower_http::trace::{DefaultMakeSpan, DefaultOnResponse, TraceLayer};
use tower_http::LatencyUnit;
use tracing::Level;

/// Shared application state injected into Axum route handlers.
#[derive(Clone)]
pub struct AppState {
    /// Registry containing all active model controllers.
    pub registry: ModelRegistry,
    /// Telemetry and Prometheus metrics tracker.
    pub metrics: Arc<ServerMetrics>,
    /// Optional authentication and role-based access control service.
    pub auth: Option<Arc<AuthService>>,
    /// Models directory containing model files and configurations.
    pub models_dir: PathBuf,
}

impl AppState {
    /// Creates a new [`AppState`] with authentication disabled.
    #[must_use]
    pub fn new(registry: ModelRegistry, metrics: Arc<ServerMetrics>) -> Self {
        Self {
            registry,
            metrics,
            auth: None,
            models_dir: PathBuf::from(crate::config::default_models_dir()),
        }
    }

    /// Attaches an authentication service to the application state.
    #[must_use]
    pub fn with_auth(mut self, auth: Arc<AuthService>) -> Self {
        self.auth = Some(auth);
        self
    }

    /// Configures the base directory for model storage and discovery.
    #[must_use]
    pub fn with_models_dir(mut self, models_dir: impl Into<PathBuf>) -> Self {
        self.models_dir = models_dir.into();
        self
    }
}

/// Response payload returned by the runtime models refresh endpoint.
#[derive(Debug, Serialize)]
pub struct RefreshModelsResponse {
    /// Operational status ("refreshed").
    pub status: String,
    /// Discovered and registered models metadata.
    pub models: Vec<ModelMetadata>,
}

/// Incoming JSON payload for model inference requests.
#[derive(Debug, Deserialize)]
pub struct InferRequest {
    /// List of named input tensors.
    pub inputs: Vec<TensorInputDto>,
}

/// DTO representing a single input tensor in a REST request.
#[derive(Debug, Deserialize)]
pub struct TensorInputDto {
    /// Name of the input tensor node.
    pub name: String,
    /// Dimensions of the tensor (e.g. `[1, 3, 224, 224]`).
    pub shape: Vec<usize>,
    /// Flat array of floating-point tensor elements.
    pub data: Vec<f32>,
}

/// Outgoing JSON response containing inference results.
#[derive(Debug, Serialize)]
pub struct InferResponse {
    /// Identifier of the executed model.
    pub model_id: String,
    /// List of named output tensors.
    pub outputs: Vec<TensorOutputDto>,
    /// Execution duration in milliseconds.
    pub duration_ms: f64,
}

/// DTO representing an output tensor returned to the client.
#[derive(Debug, Serialize)]
pub struct TensorOutputDto {
    /// Name of the output tensor node.
    pub name: String,
    /// Dimensions of the resulting tensor.
    pub shape: Vec<usize>,
    /// Flattened tensor values.
    pub data: Vec<f32>,
}

/// Generic error response wrapper.
#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    /// Error message detailing failure cause.
    pub error: String,
}

/// App error type mapping internal `InferenceError` to Axum `Response`.
pub struct AppError(InferenceError);

impl From<InferenceError> for AppError {
    fn from(err: InferenceError) -> Self {
        Self(err)
    }
}

/// Sanitizes sensitive server filesystem paths from client-facing error messages,
/// replacing internal paths with `[redacted_path]`.
#[must_use]
pub fn sanitize_error_message(msg: &str) -> String {
    let mut sanitized = String::with_capacity(msg.len());
    let mut changed = false;

    for (i, word) in msg.split(' ').enumerate() {
        if i > 0 {
            sanitized.push(' ');
        }
        let trimmed = word.trim_matches(|c| {
            c == '\'' || c == '"' || c == '(' || c == ')' || c == ':' || c == ',' || c == ';'
        });
        let is_win_drive = trimmed.len() >= 3
            && trimmed
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_alphabetic())
            && (trimmed.get(1..3) == Some(":\\") || trimmed.get(1..3) == Some(":/"));
        let is_unix_abs = trimmed.starts_with("/home/")
            || trimmed.starts_with("/Users/")
            || trimmed.starts_with("/tmp/")
            || trimmed.starts_with("/var/")
            || trimmed.starts_with("/usr/")
            || trimmed.starts_with("/root/")
            || trimmed.starts_with("/private/")
            || trimmed.starts_with("file:///");
        let is_unc_path = trimmed.starts_with("\\\\");

        if is_win_drive || is_unix_abs || is_unc_path {
            let prefix: String = word
                .chars()
                .take_while(|c| *c == '\'' || *c == '"' || *c == '(')
                .collect();
            let suffix: String = word
                .chars()
                .rev()
                .take_while(|c| {
                    *c == '\'' || *c == '"' || *c == ')' || *c == ':' || *c == ',' || *c == ';'
                })
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();
            sanitized.push_str(&prefix);
            sanitized.push_str("[redacted_path]");
            sanitized.push_str(&suffix);
            changed = true;
        } else {
            sanitized.push_str(word);
        }
    }

    if changed {
        tracing::error!(
            raw_error = %msg,
            "Sanitized internal filesystem path from client error response"
        );
    }

    sanitized
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, msg) = match self.0 {
            InferenceError::NotFound(m) => (StatusCode::NOT_FOUND, sanitize_error_message(&m)),
            InferenceError::InvalidInput(m) | InferenceError::ShapeMismatch(m) => {
                (StatusCode::BAD_REQUEST, sanitize_error_message(&m))
            }
            InferenceError::Timeout => (
                StatusCode::GATEWAY_TIMEOUT,
                "Request processing deadline exceeded".to_string(),
            ),
            InferenceError::Loading => (
                StatusCode::SERVICE_UNAVAILABLE,
                "Model is currently loading weights".to_string(),
            ),
            InferenceError::ResourceExhausted(m) => {
                let mut resp = (
                    StatusCode::SERVICE_UNAVAILABLE,
                    Json(ErrorResponse {
                        error: sanitize_error_message(&m),
                    }),
                )
                    .into_response();
                resp.headers_mut().insert(
                    axum::http::header::RETRY_AFTER,
                    axum::http::HeaderValue::from_static("5"),
                );
                return resp;
            }
            InferenceError::OutOfMemory(m) => {
                (StatusCode::SERVICE_UNAVAILABLE, sanitize_error_message(&m))
            }
            ref err @ InferenceError::UnsupportedSafetensorsArchitecture(_) => {
                (StatusCode::NOT_IMPLEMENTED, err.to_string())
            }
            InferenceError::Runtime(m) | InferenceError::QueueError(m) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                sanitize_error_message(&m),
            ),
        };

        (status, Json(ErrorResponse { error: msg })).into_response()
    }
}

/// Constructs the Axum application router with all REST routes mounted.
pub fn create_router(state: AppState) -> Router {
    // Unauthenticated public endpoints
    let public_routes = Router::new()
        .route("/", get(home_handler))
        .route("/health/live", get(liveness_handler))
        .route("/health/ready", get(readiness_handler));

    // Endpoints requiring ReadOnly role or higher
    let readonly_routes = Router::new()
        .route("/v1/models", get(list_models_handler))
        .layer(from_fn(require_role(Role::ReadOnly)));

    // Endpoints requiring Inference role or higher
    let infer_routes = Router::new()
        .route("/v1/models/:model_id/infer", post(infer_handler))
        .layer(axum::extract::DefaultBodyLimit::max(128 * 1024 * 1024))
        .layer(from_fn(require_role(Role::Inference)));

    // Administrative endpoints requiring Admin role
    let admin_routes = Router::new()
        .route(
            "/v1/models/upload",
            post(upload_model_handler)
                .layer(axum::extract::DefaultBodyLimit::max(1024 * 1024 * 1024)),
        )
        .route("/v1/models/refresh", post(refresh_models_handler))
        .route("/v1/models/:model_id/load", post(load_model_handler))
        .route("/v1/models/:model_id/unload", post(unload_model_handler))
        .route("/v1/models/:model_id/pin", post(pin_model_handler))
        .route("/v1/models/:model_id/unpin", post(unpin_model_handler))
        .layer(from_fn(require_role(Role::Admin)));

    // Metrics endpoint: protected if protect_metrics is configured, otherwise public
    let protect_metrics = state.auth.as_ref().is_some_and(|a| a.protect_metrics());

    let (public_metrics, protected_metrics) = if protect_metrics {
        (
            Router::new(),
            Router::new()
                .route("/metrics", get(metrics_handler))
                .layer(from_fn(require_role(Role::Admin))),
        )
    } else {
        (
            Router::new().route("/metrics", get(metrics_handler)),
            Router::new(),
        )
    };

    // Protected routes wrapped in authentication middleware
    let protected_routes = Router::new()
        .merge(readonly_routes)
        .merge(infer_routes)
        .merge(admin_routes)
        .merge(protected_metrics)
        .layer(from_fn_with_state(state.clone(), auth_middleware));

    public_routes
        .merge(public_metrics)
        .merge(protected_routes)
        .layer(
            TraceLayer::new_for_http()
                .make_span_with(DefaultMakeSpan::new().level(Level::INFO))
                .on_response(
                    DefaultOnResponse::new()
                        .level(Level::INFO)
                        .latency_unit(LatencyUnit::Millis),
                ),
        )
        .with_state(state)
}

/// Handles inference requests (`POST /v1/models/:model_id/infer`).
async fn infer_handler(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
    auth: Option<Extension<AuthIdentity>>,
    Json(payload): Json<InferRequest>,
) -> Result<Json<InferResponse>, AppError> {
    if !crate::types::is_valid_model_id(&model_id) {
        return Err(AppError(InferenceError::InvalidInput(format!(
            "Invalid model ID '{model_id}'. Model IDs must only contain alphanumeric characters, underscores, and hyphens."
        ))));
    }

    if let Some(Extension(ref id)) = auth {
        tracing::debug!(
            caller = %id.subject,
            role = %id.role,
            model_id = %model_id,
            "Inference request received from authenticated caller"
        );
    }
    let start = Instant::now();
    let controller = state.registry.route_for_inference(&model_id).await?;

    // Convert DTOs to internal TensorPayloads with strict bounds and size validation
    let mut inputs = Vec::with_capacity(payload.inputs.len());
    for tensor_dto in payload.inputs {
        let tensor =
            TensorPayload::from_shape_vec(tensor_dto.name, tensor_dto.shape, tensor_dto.data)?;
        inputs.push(tensor);
    }

    let infer_res = controller.infer(inputs).await;
    let elapsed = start.elapsed();
    let elapsed_us = elapsed.as_micros() as u64;

    match infer_res {
        Ok(output_tensors) => {
            state.metrics.record_inference(true, elapsed_us);

            let mut output_dtos = Vec::with_capacity(output_tensors.len());
            for tensor in output_tensors {
                let (name, shape, data) = tensor.into_components();
                output_dtos.push(TensorOutputDto { name, shape, data });
            }

            Ok(Json(InferResponse {
                model_id,
                outputs: output_dtos,
                duration_ms: (elapsed_us as f64) / 1000.0,
            }))
        }
        Err(e) => {
            state.metrics.record_inference(false, elapsed_us);
            Err(AppError(e))
        }
    }
}

/// Lists all registered models and their telemetry (`GET /v1/models`).
async fn list_models_handler(State(state): State<AppState>) -> Json<Vec<ModelMetadata>> {
    let list = state.registry.list_metadata().await;
    Json(list)
}

/// Forces loading a model into memory (`POST /v1/models/:model_id/load`).
async fn load_model_handler(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
) -> Result<Json<serde_json::Value>, AppError> {
    if !crate::types::is_valid_model_id(&model_id) {
        return Err(AppError(InferenceError::InvalidInput(format!(
            "Invalid model ID '{model_id}'. Model IDs must only contain alphanumeric characters, underscores, and hyphens."
        ))));
    }

    state
        .registry
        .ensure_loaded_with_deallocation(&model_id)
        .await?;
    Ok(Json(serde_json::json!({
        "model_id": model_id,
        "status": "loaded"
    })))
}

/// Forces unloading a model from memory (`POST /v1/models/:model_id/unload`).
async fn unload_model_handler(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
) -> Result<Json<serde_json::Value>, AppError> {
    if !crate::types::is_valid_model_id(&model_id) {
        return Err(AppError(InferenceError::InvalidInput(format!(
            "Invalid model ID '{model_id}'. Model IDs must only contain alphanumeric characters, underscores, and hyphens."
        ))));
    }

    if let Ok(group) = state.registry.get_group(&model_id).await {
        group.unload_all().await;
    } else {
        let controller = state.registry.get(&model_id).await?;
        controller.unload().await;
    }
    Ok(Json(serde_json::json!({
        "model_id": model_id,
        "status": "unloaded"
    })))
}

/// Pins a model in memory so it is eagerly loaded and exempt from automatic deallocation (`POST /v1/models/:model_id/pin`).
async fn pin_model_handler(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
) -> Result<Json<serde_json::Value>, AppError> {
    if !crate::types::is_valid_model_id(&model_id) {
        return Err(AppError(InferenceError::InvalidInput(format!(
            "Invalid model ID '{model_id}'. Model IDs must only contain alphanumeric characters, underscores, and hyphens."
        ))));
    }

    state.registry.pin_model(&model_id).await?;
    Ok(Json(serde_json::json!({
        "model_id": model_id,
        "status": "pinned",
        "pinned": true
    })))
}

/// Unpins a model so it can return to normal scale-to-zero and LRU lifecycle management (`POST /v1/models/:model_id/unpin`).
async fn unpin_model_handler(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
) -> Result<Json<serde_json::Value>, AppError> {
    if !crate::types::is_valid_model_id(&model_id) {
        return Err(AppError(InferenceError::InvalidInput(format!(
            "Invalid model ID '{model_id}'. Model IDs must only contain alphanumeric characters, underscores, and hyphens."
        ))));
    }

    state.registry.unpin_model(&model_id).await?;
    Ok(Json(serde_json::json!({
        "model_id": model_id,
        "status": "unpinned",
        "pinned": false
    })))
}

/// Handles model file upload and dynamic registration (`POST /v1/models/upload`).
async fn upload_model_handler(
    State(state): State<AppState>,
    mut multipart: axum::extract::Multipart,
) -> Result<Json<ModelMetadata>, Response> {
    let mut file_bytes: Option<bytes::Bytes> = None;
    let mut original_filename: Option<String> = None;
    let mut model_id: Option<String> = None;
    let mut description: Option<String> = None;
    let mut pinned: Option<bool> = None;
    let mut max_batch_size: Option<usize> = None;
    let mut max_batch_delay_ms: Option<u64> = None;
    let mut idle_timeout_secs: Option<u64> = None;
    let mut min_residency_secs: Option<u64> = None;
    let mut autoscale_queue_threshold: Option<usize> = None;
    let mut max_replicas: Option<usize> = None;
    let mut routing_strategy: Option<String> = None;
    let mut device: Option<String> = None;
    let mut overwrite: Option<bool> = None;

    while let Some(field) = multipart.next_field().await.map_err(|e| {
        AppError(InferenceError::InvalidInput(format!(
            "Multipart stream error: {e}"
        )))
        .into_response()
    })? {
        let field_name = field.name().unwrap_or("").to_string();
        if field_name == "file" || field_name == "model" {
            original_filename = field.file_name().map(|s| s.to_string());
            let bytes = field.bytes().await.map_err(|e| {
                AppError(InferenceError::InvalidInput(format!(
                    "Failed to read uploaded file bytes: {e}"
                )))
                .into_response()
            })?;
            file_bytes = Some(bytes);
        } else {
            let text = field.text().await.map_err(|e| {
                AppError(InferenceError::InvalidInput(format!(
                    "Failed to read field '{field_name}': {e}"
                )))
                .into_response()
            })?;
            let trimmed = text.trim();
            match field_name.as_str() {
                "model_id" | "id" if !trimmed.is_empty() => {
                    model_id = Some(trimmed.to_string());
                }
                "description" if !trimmed.is_empty() => {
                    description = Some(trimmed.to_string());
                }
                "pinned" => {
                    pinned = match trimmed.to_ascii_lowercase().as_str() {
                        "true" | "1" | "yes" => Some(true),
                        _ => Some(false),
                    };
                }
                "overwrite" => {
                    overwrite = match trimmed.to_ascii_lowercase().as_str() {
                        "true" | "1" | "yes" => Some(true),
                        _ => Some(false),
                    };
                }
                "max_batch_size" => {
                    max_batch_size = trimmed.parse().ok();
                }
                "max_batch_delay_ms" => {
                    max_batch_delay_ms = trimmed.parse().ok();
                }
                "idle_timeout_secs" => {
                    idle_timeout_secs = trimmed.parse().ok();
                }
                "min_residency_secs" => {
                    min_residency_secs = trimmed.parse().ok();
                }
                "autoscale_queue_threshold" => {
                    autoscale_queue_threshold = trimmed.parse().ok();
                }
                "max_replicas" => {
                    max_replicas = trimmed.parse().ok();
                }
                "routing_strategy" if !trimmed.is_empty() => {
                    routing_strategy = Some(trimmed.to_string());
                }
                "device" | "provider" if !trimmed.is_empty() => {
                    device = Some(trimmed.to_string());
                }
                _ => {}
            }
        }
    }

    let bytes = file_bytes.ok_or_else(|| {
        AppError(InferenceError::InvalidInput(
            "Missing required 'file' field in multipart upload".to_string(),
        ))
        .into_response()
    })?;

    let raw_name = original_filename.ok_or_else(|| {
        AppError(InferenceError::InvalidInput(
            "Uploaded file missing filename parameter".to_string(),
        ))
        .into_response()
    })?;

    let filename = std::path::Path::new(&raw_name)
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| {
            AppError(InferenceError::InvalidInput(format!(
                "Invalid file path or filename '{raw_name}'"
            )))
            .into_response()
        })?;

    let is_valid_format = filename.ends_with(".onnx")
        || filename.ends_with(".safetensors")
        || filename.ends_with(".xgb.json")
        || filename.ends_with(".json");

    if !is_valid_format {
        return Err(AppError(InferenceError::InvalidInput(format!(
            "Unsupported model format for '{filename}'. Supported extensions: .onnx, .safetensors, .xgb.json, .json"
        ))).into_response());
    }

    let computed_id = if filename.ends_with(".xgb.json") {
        filename.trim_end_matches(".xgb.json")
    } else {
        std::path::Path::new(filename)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(filename)
    };

    let final_model_id = model_id.unwrap_or_else(|| computed_id.to_string());

    if !crate::types::is_valid_model_id(&final_model_id) {
        return Err(AppError(InferenceError::InvalidInput(format!(
            "Invalid model ID '{final_model_id}'. Model IDs must only contain alphanumeric characters, underscores, and hyphens."
        ))).into_response());
    }

    tokio::fs::create_dir_all(&state.models_dir)
        .await
        .map_err(|e| {
            AppError(InferenceError::Runtime(format!(
                "Failed to create models directory '{}': {e}",
                state.models_dir.display()
            )))
            .into_response()
        })?;

    let target_path = state.models_dir.join(filename);
    if target_path.exists() && !overwrite.unwrap_or(false) {
        return Err((
            StatusCode::CONFLICT,
            Json(ErrorResponse {
                error: format!(
                    "Model file '{filename}' already exists. Pass overwrite=true in multipart form to overwrite."
                ),
            }),
        )
            .into_response());
    }

    tokio::fs::write(&target_path, &bytes).await.map_err(|e| {
        AppError(InferenceError::Runtime(format!(
            "Failed to write uploaded model weights to '{}': {e}",
            target_path.display()
        )))
        .into_response()
    })?;

    let strategy = routing_strategy
        .as_deref()
        .and_then(|s| s.parse().ok())
        .unwrap_or(crate::routing::RoutingStrategyType::LeastQueue);

    let batch_size = max_batch_size.unwrap_or(crate::config::DEFAULT_MAX_BATCH_SIZE);
    let queue_threshold = autoscale_queue_threshold.unwrap_or(batch_size);

    let config = crate::model::ModelConfig {
        id: final_model_id.clone(),
        onnx_path: target_path.to_string_lossy().to_string(),
        idle_timeout: std::time::Duration::from_secs(
            idle_timeout_secs.unwrap_or(crate::config::DEFAULT_IDLE_TIMEOUT_SECS),
        ),
        max_batch_size: batch_size,
        max_batch_delay: std::time::Duration::from_millis(
            max_batch_delay_ms.unwrap_or(crate::config::DEFAULT_MAX_BATCH_DELAY_MS),
        ),
        provider: device,
        description,
        min_residency: std::time::Duration::from_secs(
            min_residency_secs.unwrap_or(crate::config::DEFAULT_MIN_RESIDENCY_SECS),
        ),
        autoscale_queue_threshold: queue_threshold,
        max_replicas,
        routing_strategy: strategy,
        pinned: pinned.unwrap_or(false),
    };

    let controller = crate::model::ModelController::new(config);
    state.registry.register(controller.clone()).await;

    let meta = controller.metadata().await;
    tracing::info!(
        model_id = %final_model_id,
        path = %target_path.display(),
        size_bytes = bytes.len(),
        pinned = meta.pinned,
        "Model successfully uploaded and registered via REST API"
    );

    Ok(Json(meta))
}

/// Triggers directory scan and runtime refresh of models (`POST /v1/models/refresh`).
async fn refresh_models_handler(
    State(state): State<AppState>,
) -> Result<Json<RefreshModelsResponse>, AppError> {
    let dir_path = &state.models_dir;
    let refreshed = state.registry.refresh_from_dir(dir_path).await?;
    tracing::info!(
        dir = %dir_path.display(),
        model_count = refreshed.len(),
        "Refreshed models directory via REST API"
    );
    Ok(Json(RefreshModelsResponse {
        status: "refreshed".to_string(),
        models: refreshed,
    }))
}

/// Home endpoint (`GET /`) providing basic service metadata and discoverable endpoints.
async fn home_handler() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "service": "snooze-ml",
        "version": env!("CARGO_PKG_VERSION"),
        "status": "ok",
        "message": "SnoozeML High-Performance ONNX Inference Daemon",
        "endpoints": {
            "home": "/",
            "models": "/v1/models",
            "models_upload": "/v1/models/upload",
            "models_refresh": "/v1/models/refresh",
            "health_live": "/health/live",
            "health_ready": "/health/ready",
            "metrics": "/metrics"
        }
    }))
}

/// Server liveness health probe (`GET /health/live`).
async fn liveness_handler() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "live" }))
}

/// Server readiness probe (`GET /health/ready`).
async fn readiness_handler() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "ready" }))
}

/// Prometheus metrics exposition endpoint (`GET /metrics`).
async fn metrics_handler(State(state): State<AppState>) -> String {
    state.metrics.format_prometheus()
}
