---
trigger: always_on
---

# Transport Parity, Security & RBAC Rules

SnoozeML exposes dual network transports (**HTTP REST** via Axum in `src/server/http.rs` and **gRPC** via Tonic in `src/server/grpc/`) protected by a unified authentication and RBAC layer (`src/auth.rs`). All changes to endpoints, error variants, or security logic must maintain strict parity across transports.

---

## 1. Error Mapping Parity (`InferenceError` $\rightarrow$ HTTP & gRPC)
Whenever adding or modifying a variant on `InferenceError` (`src/types.rs`), you MUST update both `AppError::into_response` (`src/server/http.rs`) and `inference_error_to_status` (`src/server/grpc/convert.rs`) according to this canonical mapping:

| `InferenceError` Variant | HTTP `StatusCode` (`src/server/http.rs`) | gRPC `tonic::Code` (`src/server/grpc/convert.rs`) | Notes |
| :--- | :--- | :--- | :--- |
| `NotFound(msg)` | `404 NOT_FOUND` | `Code::NotFound` | Unregistered `model_id` or missing weight files |
| `InvalidInput(msg)` | `400 BAD_REQUEST` | `Code::InvalidArgument` | Malformed tensor shapes, overflow, bad dtype |
| `ShapeMismatch(msg)` | `400 BAD_REQUEST` | `Code::InvalidArgument` | Incompatible tensor dimensions during collation/infer |
| `Timeout` | `504 GATEWAY_TIMEOUT` | `Code::DeadlineExceeded` | Request exceeded processing or queue deadline |
| `Loading` | `503 SERVICE_UNAVAILABLE` | `Code::Unavailable` | Model weights are currently loading |
| `ResourceExhausted(msg)` | `503 SERVICE_UNAVAILABLE` | `Code::ResourceExhausted` | Includes `retry-after: 5` header / metadata |
| `OutOfMemory(msg)` | `503 SERVICE_UNAVAILABLE` | `Code::ResourceExhausted` | Device VRAM or host RAM exhausted |
| `UnsupportedSafetensorsArchitecture(arch)` | `501 NOT_IMPLEMENTED` | `Code::Unimplemented` | Suggests converting model to ONNX |
| `Runtime(msg)` / `QueueError(msg)` | `500 INTERNAL_SERVER_ERROR` | `Code::Internal` | Engine failure or closed batcher channel |

---

## 2. Authentication & 3-Tier RBAC (`src/auth.rs`)
* **Role Hierarchy (`Role`):** `ReadOnly (1) < Inference (2) < Admin (3)`.
  * **Public (Unauthenticated):** `GET /`, `GET /health/live`, `GET /health/ready`, gRPC `grpc.health.v1.Health/Check`, and `GET /metrics` (unless `protect_metrics = true`).
  * **`Role::ReadOnly`:** `GET /v1/models` (`snooze status`).
  * **`Role::Inference`:** `POST /v1/models/:model_id/infer` and gRPC `snoozeml.v1.InferenceService/Infer`.
  * **`Role::Admin`:** `POST /v1/models/:model_id/{load,unload,pin,unpin}` and `GET /metrics` (when `protect_metrics = true`).
* **Constant-Time API Key Verification:** API keys MUST be pre-hashed with SHA-256 (`sha2::Sha256`) and compared using `subtle::ConstantTimeEq` (`ct_eq`) across all configured keys to prevent timing side-channel attacks.
* **Dual Header Support:** Both HTTP middleware (`auth_middleware`) and gRPC metadata authentication (`SnoozeInferenceService::authenticate_metadata`) must accept credentials via:
  1. `x-api-key: <key>`
  2. `authorization: Bearer <api-key-or-jwt>` (case-insensitive `Bearer ` / `bearer ` prefix)
* **Credential Sanitization:** Never log raw API keys, `jwt.secret`, `HF_TOKEN`, or TLS private keys in `tracing` spans, error messages, or `Debug` implementations.

---

## 3. Tensor Wire Validation & Shared TLS
* **Overflow-Safe Tensor Validation:** Both REST (`TensorPayload::from_shape_vec`) and gRPC (`tensor_data_to_payload`) must validate that every dimension is positive, `usize::try_from` succeeds, multiplying dimensions via `checked_mul` does not overflow `usize`, and byte/element lengths match before allocating buffers.
* **Shared TLS (`src/server/mod.rs`):** When `TlsConfig` is provided (`--tls-cert` & `--tls-key`), apply the PEM certificate and key to both the Axum HTTPS listener (`axum_server::tls_rustls::RustlsConfig`) and the Tonic gRPC server (`tonic::transport::ServerTlsConfig`).
* **Telemetry Consistency:** Every inference request on both HTTP and gRPC must record execution metrics via `state.metrics.record_inference(success, elapsed_us)`.
