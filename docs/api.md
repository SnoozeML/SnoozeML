# SnoozeML REST & Telemetry API Specification

The daemon exposes an HTTP/1.1 and HTTP/2 REST interface powered by Axum, alongside a high-performance binary gRPC service powered by Tonic on port 50051.

> [!NOTE]
> For the complete gRPC Protocol Buffer specification, client guides, and error code mappings, see the [gRPC Reference](src/content/docs/reference/grpc-api-reference.md) and [gRPC Client Guide](src/content/docs/client/grpc-inference.md).

---

## Authentication & Authorization

When authentication is enabled, requests to protected routes require credentials via:
- `X-API-Key: <key>`
- `Authorization: Bearer <key-or-jwt>`

### Role Permissions
- **Public**: `/`, `/health/live`, `/health/ready`, `/metrics` (by default)
- **`ReadOnly`**: `GET /v1/models`
- **`Inference`**: `POST /v1/models/{model_id}/infer`, `GET /v1/models`
- **`Admin`**: `POST /v1/models/{model_id}/load`, `POST /v1/models/{model_id}/unload` (and `/metrics` if `protect_metrics = true`)

### Authentication Errors
- `401 Unauthorized`: Missing, invalid, or expired credentials (`{"error": "..."}` with `WWW-Authenticate: Bearer realm="snoozeml"` header).
- `403 Forbidden`: Insufficient role permissions for the target endpoint.

See [docs/authentication.md](authentication.md) for full configuration details and token schemas.

---

## 1. Inference Endpoints

### Run Model Inference
```http
POST /v1/models/{model_id}/infer
Content-Type: application/json
```

#### Request Payload
```json
{
  "inputs": [
    {
      "name": "input_ids",
      "shape": [1, 4],
      "data": [101.0, 2054.0, 2003.0, 102.0]
    }
  ]
}
```

#### Successful Response (`200 OK`)
```json
{
  "model_id": "bert-base",
  "outputs": [
    {
      "name": "output",
      "shape": [1, 4],
      "data": [0.12, 0.45, 0.88, 0.03]
    }
  ],
  "duration_ms": 2.45
}
```

#### Error Responses
- `400 Bad Request`: Tensor dimensions or payload length mismatch (`{"error": "..."}`).
- `404 Not Found`: Target `model_id` is not registered (`{"error": "Model not found: ..."}`).
- `504 Gateway Timeout`: Processing deadline was exceeded.

---

## 2. Model Management Endpoints

### List Registered Models
```http
GET /v1/models
```
Returns an array of metadata objects describing each model:
```json
[
  {
    "id": "resnet50",
    "state": "idle",
    "onnx_path": "models/resnet50.onnx",
    "description": "ResNet-50 deep convolutional network for image classification",
    "idle_timeout_secs": 300,
    "max_batch_size": 16,
    "max_batch_delay_ms": 10,
    "last_accessed_secs_ago": 45
  }
]
```

### Explicit Model Preload
```http
POST /v1/models/{model_id}/load
```
Loads weights into RAM/VRAM ahead of time.

### Explicit Model Deallocation
```http
POST /v1/models/{model_id}/unload
```
Immediately deallocates the model session to free RAM and VRAM.

---

## 3. Health & Observability

### Liveness Probe
```http
GET /health/live
```
Returns `{"status": "live"}` when the HTTP listener is operational.

### Readiness Probe
```http
GET /health/ready
```
Returns `{"status": "ready"}` when the daemon is prepared to accept connections.

### Prometheus Metrics
```http
GET /metrics
```
Returns standard Prometheus plaintext format:
```text
# HELP snoozeml_requests_total Total number of inference requests received
# TYPE snoozeml_requests_total counter
snoozeml_requests_total{status="success"} 152
snoozeml_requests_total{status="error"} 0
snoozeml_requests_total{status="all"} 152
# HELP snoozeml_inference_latency_seconds_total Total duration of inference execution in seconds
# TYPE snoozeml_inference_latency_seconds_total counter
snoozeml_inference_latency_seconds_total 0.384500
# HELP snoozeml_batches_dispatched_total Total micro-batches processed by the execution engine
# TYPE snoozeml_batches_dispatched_total counter
snoozeml_batches_dispatched_total 12
```

---

## 4. Service Discovery & Home

### Home / Service Info
```http
GET /
```
Returns high-level daemon information, version, and links to primary endpoints:
```json
{
  "service": "snooze-ml",
  "version": "0.1.0",
  "status": "ok",
  "message": "SnoozeML High-Performance ONNX Inference Daemon",
  "endpoints": {
    "home": "/",
    "models": "/v1/models",
    "health_live": "/health/live",
    "health_ready": "/health/ready",
    "metrics": "/metrics"
  }
}
```


