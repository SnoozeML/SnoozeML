---
title: "REST API Reference"
description: "This page provides the complete HTTP REST API specification for SnoozeML."
---

This page provides the complete HTTP REST API specification for SnoozeML.

:::note
For the Protocol Buffer schema, gRPC methods, and binary tensor transmission specifications, see the [gRPC API Reference](/SnoozeML/reference/grpc-api-reference/).
:::

---

## 1. Endpoints Quick Reference

| Method | Endpoint | Required Role | Summary |
| :--- | :--- | :--- | :--- |
| `GET` | `/` | **Public** | Service information and endpoint links. |
| `GET` | `/health/live` | **Public** | Liveness probe for process health. |
| `GET` | `/health/ready` | **Public** | Readiness probe for request acceptance. |
| `GET` | `/metrics` | **Public** / `Admin` | Prometheus exposition telemetry. |
| `GET` | `/v1/models` | `ReadOnly` | List registered models and metadata. |
| `POST` | `/v1/models/:id/infer` | `Inference` | Execute dynamic batch tensor inference. |
| `POST` | `/v1/models/upload` | `Admin` | Upload model weights directly via multipart form. |
| `POST` | `/v1/models/refresh` | `Admin` | Discover and refresh models in models directory. |
| `POST` | `/v1/models/:id/load` | `Admin` | Explicitly preload model into memory. |
| `POST` | `/v1/models/:id/unload`| `Admin` | Deallocate model to scale to zero. |
| `POST` | `/v1/models/:id/pin`   | `Admin` | Pin model replica in memory (exempt from eviction). |
| `POST` | `/v1/models/:id/unpin` | `Admin` | Unpin model replica (restore scale-to-zero). |


---

## 2. API Endpoints Specification

### Service Information
```http
GET /
```
Returns high-level daemon information, version, and links to primary routes.

#### Response (`200 OK`)
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

---

### Liveness Probe
```http
GET /health/live
```
Evaluates process health. Always returns `200 OK` as long as the HTTP listener is operational.

#### Response (`200 OK`)
```json
{
  "status": "live"
}
```

---

### Readiness Probe
```http
GET /health/ready
```
Evaluates server readiness to accept network traffic.

#### Response (`200 OK`)
```json
{
  "status": "ready"
}
```

---

### Prometheus Telemetry
```http
GET /metrics
```
Returns plaintext Prometheus exposition metrics.

#### Headers
- When `protect_metrics = true`: requires `X-API-Key: <admin-key>` or `Authorization: Bearer <admin-token>`.

#### Response (`200 OK`)
```text
# HELP snoozeml_requests_total Total number of inference requests received
# TYPE snoozeml_requests_total counter
snoozeml_requests_total{status="success"} 42
```

---

### List Models
```http
GET /v1/models
```
Returns an array of metadata objects describing each registered model.

#### Required Role: `ReadOnly`

#### Response (`200 OK`)
```json
[
  {
    "id": "resnet50",
    "state": "ready",
    "onnx_path": "models/resnet50.onnx",
    "description": "ResNet-50 image classifier",
    "device": "cuda",
    "idle_timeout_secs": 300,
    "max_batch_size": 16,
    "max_batch_delay_ms": 10,
    "last_accessed_secs_ago": 25
  }
]
```

---

### Execute Model Inference
```http
POST /v1/models/{model_id}/infer
```
Dispatches input tensors to the dynamic batch accumulator for inference.

#### Required Role: `Inference`
#### Request Headers:
- `Content-Type: application/json`
- `X-API-Key: <key>` or `Authorization: Bearer <key-or-jwt>`

#### Request Body
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

#### Response (`200 OK`)
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
- `400 Bad Request`: Tensor dimensions or payload length mismatch.
- `401 Unauthorized`: Missing or invalid credentials.
- `403 Forbidden`: Insufficient role permissions.
- `404 Not Found`: Model ID not recognized.
- `503 Service Unavailable`: Device/host memory saturated with busy models (`Retry-After: 5`).

---

### Preload Model
```http
POST /v1/models/{model_id}/load
```
Loads weights into memory ahead of time to eliminate cold-start latency.

#### Required Role: `Admin`

#### Response (`200 OK`)
```json
{
  "status": "loaded",
  "model_id": "resnet50"
}
```

---

### Deallocate Model (Scale to Zero)
```http
POST /v1/models/{model_id}/unload
```
Immediately drops model weights from memory.

#### Required Role: `Admin`

#### Response (`200 OK`)
```json
{
  "status": "unloaded",
  "model_id": "resnet50"
}
```

---

### Upload Model Directly via HTTP
```http
POST /v1/models/upload
```
Uploads model weights directly via multipart form-data without mounting external volumes or restarting the daemon. Automatically persists the weights to the configured `models_dir` and registers the model in the active `ModelRegistry`.

#### Required Role: `Admin`
#### Request Headers:
- `Content-Type: multipart/form-data`
- `X-API-Key: <admin-key>` or `Authorization: Bearer <admin-jwt>`

#### Form Fields:
| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `file` | Binary | **Yes** | Model binary weight file (`.onnx`, `.safetensors`, `.xgb.json`, or `.json`). Maximum size 1 GB. |
| `model_id` | String | No | Custom identifier. Defaults to the uploaded filename without extension. |
| `description` | String | No | Optional human-readable description for model catalog listings. |
| `pinned` | Boolean | No | If `true`, pins the primary model replica in memory, exempting it from scale-to-zero and LRU eviction. Defaults to `false`. |
| `max_batch_size` | Integer | No | Maximum dynamic micro-batch batch size along Axis(0). Defaults to `16`. |
| `max_batch_delay_ms` | Integer | No | Dynamic micro-batch coalescing window in milliseconds. Defaults to `10`. |
| `idle_timeout_secs` | Integer | No | Inactivity duration in seconds before automatic scale-to-zero deallocation. Defaults to `300`. |
| `min_residency_secs` | Integer | No | Anti-thrashing residency time in seconds before eviction eligibility. Defaults to `60`. |
| `autoscale_queue_threshold`| Integer | No | Request queue depth triggering vertical replica scale-up. Defaults to `8`. |
| `max_replicas` | Integer | No | Maximum horizontal replica autoscaling limit. Defaults to `1`. |
| `routing_strategy` | String | No | Multi-replica load balancing strategy (`least_queue`, `round_robin`, `random`). Defaults to `least_queue`. |
| `device` | String | No | Hardware acceleration target (`cpu`, `cuda`, `tensorrt`, `coreml`). Defaults to `cpu`. |

#### Response (`200 OK`)
```json
{
  "status": "created",
  "model_id": "iris_classifier",
  "path": "models/iris_classifier.onnx",
  "message": "Model 'iris_classifier' uploaded and registered successfully."
}
```

---

### Runtime Refresh Models
```http
POST /v1/models/refresh
```
Performs a live filesystem scan of the configured `models_dir` to discover newly added or modified model weight files, instantly registering them into the running registry without dropping active connections.

#### Required Role: `Admin`
#### Request Headers:
- `X-API-Key: <admin-key>` or `Authorization: Bearer <admin-jwt>`

#### Response (`200 OK`)
```json
{
  "status": "ok",
  "message": "Discovered 4 models",
  "models": [
    "iris_classifier",
    "california_housing",
    "credit_risk_tree",
    "resnet50"
  ]
}
```

---

### Pin Model
```http
POST /v1/models/{model_id}/pin
```
Pins a model in memory, ensuring its primary replica is never deallocated by scale-to-zero idle timers or capacity LRU eviction.

#### Required Role: `Admin`

#### Response (`200 OK`)
```json
{
  "status": "pinned",
  "model_id": "resnet50"
}
```

---

### Unpin Model
```http
POST /v1/models/{model_id}/unpin
```
Unpins a model, restoring normal scale-to-zero idle sweeping and LRU eviction eligibility.

#### Required Role: `Admin`

#### Response (`200 OK`)
```json
{
  "status": "unpinned",
  "model_id": "resnet50"
}
```
