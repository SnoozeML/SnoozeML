//! gRPC InferenceService implementation for SnoozeML.

#![allow(clippy::result_large_err)]

use crate::auth::{AuthIdentity, Role};
use crate::server::grpc::convert::{
    inference_error_to_status, payload_to_tensor_data, tensor_data_to_payload,
};
use crate::server::grpc::proto::inference_service_server::InferenceService;
use crate::server::grpc::proto::{ModelInferRequest, ModelInferResponse};
use crate::server::http::AppState;
use std::time::{Duration, Instant};
use tonic::{Request, Response, Status};

/// Implementation of the SnoozeML gRPC Inference Service.
#[derive(Clone)]
pub struct SnoozeInferenceService {
    state: AppState,
}

impl SnoozeInferenceService {
    /// Creates a new [`SnoozeInferenceService`] backed by shared [`AppState`].
    ///
    /// # Panics
    /// This function does not panic.
    #[must_use]
    pub fn new(state: AppState) -> Self {
        Self { state }
    }

    /// Authenticates metadata headers (`x-api-key` or `authorization: Bearer <token>`).
    ///
    /// # Errors
    /// Returns [`Status::unauthenticated`] if credentials are missing or invalid,
    /// or [`Status::permission_denied`] if the caller lacks the required [`Role::Inference`].
    fn authenticate_metadata(
        &self,
        req: &Request<ModelInferRequest>,
    ) -> Result<AuthIdentity, Status> {
        let Some(ref auth) = self.state.auth else {
            return Ok(AuthIdentity::anonymous_admin());
        };

        if !auth.is_enabled() {
            return Ok(AuthIdentity::anonymous_admin());
        }

        // 1. Check X-API-Key metadata header
        if let Some(val) = req.metadata().get("x-api-key") {
            let key_str = val
                .to_str()
                .map_err(|_| Status::unauthenticated("Invalid x-api-key header encoding"))?;
            let identity = auth
                .authenticate_api_key(key_str.trim())
                .map_err(|_| Status::unauthenticated("Invalid API key"))?;

            if identity.role < Role::Inference {
                return Err(Status::permission_denied(
                    "Insufficient permissions: requires at least inference role",
                ));
            }
            return Ok(identity);
        }

        // 2. Check Authorization metadata header (API key or JWT Bearer)
        if let Some(val) = req.metadata().get("authorization") {
            let auth_str = val
                .to_str()
                .map_err(|_| Status::unauthenticated("Invalid authorization header encoding"))?;

            let token = if let Some(stripped) = auth_str.strip_prefix("Bearer ") {
                stripped.trim()
            } else if let Some(stripped) = auth_str.strip_prefix("bearer ") {
                stripped.trim()
            } else {
                return Err(Status::unauthenticated(
                    "Malformed authorization header; must start with 'Bearer '",
                ));
            };

            // Attempt static API key validation first
            if let Ok(identity) = auth.authenticate_api_key(token) {
                if identity.role < Role::Inference {
                    return Err(Status::permission_denied(
                        "Insufficient permissions: requires at least inference role",
                    ));
                }
                return Ok(identity);
            }

            // Fallback to JWT cryptographic verification
            let identity = auth
                .authenticate_jwt(token)
                .map_err(|e| Status::unauthenticated(format!("Invalid JWT token: {e}")))?;

            if identity.role < Role::Inference {
                return Err(Status::permission_denied(
                    "Insufficient permissions: requires at least inference role",
                ));
            }
            return Ok(identity);
        }

        Err(Status::unauthenticated(
            "Authentication credentials are required",
        ))
    }
}

#[tonic::async_trait]
impl InferenceService for SnoozeInferenceService {
    /// Executes unary model inference over gRPC.
    ///
    /// # Errors
    /// Returns [`Status`] on authentication, validation, queue timeout, or runtime failures.
    async fn infer(
        &self,
        request: Request<ModelInferRequest>,
    ) -> Result<Response<ModelInferResponse>, Status> {
        let caller = self.authenticate_metadata(&request)?;

        let req = request.into_inner();
        let model_id = req.model_id;
        if !crate::types::is_valid_model_id(&model_id) {
            return Err(Status::invalid_argument(format!(
                "Invalid model ID '{model_id}'. Model IDs must only contain alphanumeric characters, underscores, and hyphens."
            )));
        }

        if !req.parameters.is_empty() {
            let mut keys: Vec<&String> = req.parameters.keys().collect();
            keys.sort();
            return Err(Status::invalid_argument(format!(
                "Unsupported request parameters: {keys:?}. Runtime parameters are reserved for future generative sampling options."
            )));
        }

        if self.state.auth.as_ref().is_some_and(|a| a.is_enabled()) {
            tracing::debug!(
                caller = %caller.subject,
                role = %caller.role,
                model_id = %model_id,
                "Inference request received from authenticated caller"
            );
        }

        let start = Instant::now();
        let controller = self
            .state
            .registry
            .route_for_inference(&model_id)
            .await
            .map_err(inference_error_to_status)?;

        let mut inputs = Vec::with_capacity(req.inputs.len());
        for tensor_data in req.inputs {
            let payload = tensor_data_to_payload(tensor_data)?;
            inputs.push(payload);
        }

        let infer_fut = controller.infer(inputs);
        let infer_res = if let Some(ms) = req.client_timeout_ms {
            match tokio::time::timeout(Duration::from_millis(ms), infer_fut).await {
                Ok(res) => res,
                Err(_) => {
                    let elapsed = start.elapsed();
                    let elapsed_us = elapsed.as_micros() as u64;
                    self.state.metrics.record_inference(false, elapsed_us);
                    return Err(Status::deadline_exceeded(
                        "Client request deadline exceeded",
                    ));
                }
            }
        } else {
            infer_fut.await
        };
        let elapsed = start.elapsed();
        let elapsed_us = elapsed.as_micros() as u64;

        match infer_res {
            Ok(output_tensors) => {
                self.state.metrics.record_inference(true, elapsed_us);
                let mut outputs = Vec::with_capacity(output_tensors.len());
                for tensor in output_tensors {
                    outputs.push(payload_to_tensor_data(tensor));
                }

                Ok(Response::new(ModelInferResponse {
                    model_id,
                    outputs,
                    duration_ms: (elapsed_us as f64) / 1000.0,
                }))
            }
            Err(e) => {
                self.state.metrics.record_inference(false, elapsed_us);
                Err(inference_error_to_status(e))
            }
        }
    }
}
