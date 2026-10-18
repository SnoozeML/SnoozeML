---
title: "Configuring API Keys"
description: "Static API keys are the simplest and most common method for authenticating microservices, backend applications, and scripts interacting with SnoozeML."
---

Static API keys are the simplest and most common method for authenticating microservices, backend applications, and scripts interacting with SnoozeML.

---

## 1. Configuring API Keys

API keys are mapped directly to one of the three RBAC roles: `admin`, `inference` (or `infer`), or `read_only` (or `ro`).

> [!IMPORTANT]
> **Do not store plaintext secrets in `models.toml`**. Storing credentials in version-controlled files risks credential leakage. SnoozeML encourages enabling authentication with `[auth] enabled = true` and resolving keys dynamically from environment variables.

### Option A: Via Environment Variables (Recommended)
Simply enable authentication in `models.toml`:

```toml
[auth]
enabled = true
```

Then provide your keys through environment variables:

```bash
# 1. Single API key (defaults to Admin role):
export SNOOZE_API_KEY="sk-admin-production-key-987"

# Or with an explicit role prefix (<role>:<key>):
export SNOOZE_API_KEY="inference:sk-inference-web-app-654"

# 2. Multiple API keys with roles via comma-separated list:
export SNOOZE_API_KEYS="admin:sk-admin-987,inference:sk-infer-654,ro:sk-monitor-321"

# 3. Role-specific environment variables:
export SNOOZE_API_KEY_ADMIN="sk-admin-production-key-987"
export SNOOZE_API_KEY_INFER="sk-inference-web-app-654"
export SNOOZE_API_KEY_READONLY="sk-readonly-monitor-321"
```

### Option B: Referencing Environment Variables in `models.toml`
If you need to assign explicit descriptive names to keys in `models.toml`, use `key_env` to reference environment variable names, or prefix with `$` to expand variables:

```toml
[auth]
enabled = true

# Cluster administrator key (full access)
[[auth.api_keys]]
name = "cluster-admin"
key_env = "SNOOZE_API_KEY_ADMIN"
role = "admin"

# Web backend key (can query inference and view models)
[[auth.api_keys]]
name = "web-client"
key_env = "SNOOZE_API_KEY_INFER"
role = "inference"

# Monitoring crawler key (can only list models)
[[auth.api_keys]]
name = "datadog-crawler"
key = "$SNOOZE_API_KEY_READONLY"
role = "read_only"
```

### Option C: Via CLI Flags on `snooze serve`
You can provide repeatable `--api-key` arguments using the format `<role>:<key>` (or just `<key>`, which defaults to `admin`):

```bash
snooze serve \
  --models-dir ./models \
  --api-key admin:sk-admin-production-key-987 \
  --api-key infer:sk-inference-web-app-654 \
  --api-key ro:sk-readonly-monitor-321
```

> [!TIP]
> Both `infer` and `inference` are accepted for the Inference role. Both `ro` and `read_only` are accepted for the ReadOnly role. Omitting the role prefix defaults to `admin`.


---

## 2. Authenticating Client Requests

Clients can submit their API key using either of two standard HTTP headers:

### Method 1: Header `X-API-Key` (HTTP) or `x-api-key` (gRPC)
**HTTP**:
```bash
curl -X POST http://127.0.0.1:8080/v1/models/demo-model/infer \
  -H "X-API-Key: sk-inference-web-app-654" \
  -H "Content-Type: application/json" \
  -d '{"inputs": [{"name": "input", "shape": [1, 2], "data": [1.0, 2.0]}]}'
```

**gRPC (`grpcurl`)**:
```bash
grpcurl -plaintext \
  -proto proto/snoozeml/v1/inference.proto \
  -H "x-api-key: sk-inference-web-app-654" \
  -d '{"model_id": "demo-model", "inputs": [{"name": "input", "shape": [1, 2], "raw_data": "AADAQAAAQEAA"}]}' \
  127.0.0.1:50051 snoozeml.v1.InferenceService/Infer
```

### Method 2: Header `Authorization: Bearer <key>`
**HTTP**:
```bash
curl -X POST http://127.0.0.1:8080/v1/models/demo-model/infer \
  -H "Authorization: Bearer sk-inference-web-app-654" \
  -H "Content-Type: application/json" \
  -d '{"inputs": [{"name": "input", "shape": [1, 2], "data": [1.0, 2.0]}]}'
```

**gRPC (`grpcurl`)**:
```bash
grpcurl -plaintext \
  -proto proto/snoozeml/v1/inference.proto \
  -H "authorization: Bearer sk-inference-web-app-654" \
  -d '{"model_id": "demo-model", "inputs": [{"name": "input", "shape": [1, 2], "raw_data": "AADAQAAAQEAA"}]}' \
  127.0.0.1:50051 snoozeml.v1.InferenceService/Infer
```

**Python gRPC Client Metadata**:
```python
# Pass metadata as a list of tuples
metadata = [("x-api-key", "sk-inference-web-app-654")]
response = client.Infer(request, metadata=metadata)
```

---

## 3. Role Enforcement Examples

### 1. Read-Only Key Listing Models (`200 OK`)
```bash
curl -s http://127.0.0.1:8080/v1/models \
  -H "X-API-Key: sk-readonly-monitor-321" | jq .
```
Successfully returns the array of registered models.

### 2. Read-Only Key Attempting Inference (`403 Forbidden`)
```bash
curl -X POST http://127.0.0.1:8080/v1/models/demo-model/infer \
  -H "X-API-Key: sk-readonly-monitor-321" \
  -H "Content-Type: application/json" \
  -d '{"inputs": [{"name": "input", "shape": [1, 2], "data": [1.0, 2.0]}]}'
```
Response:
```json
{
  "error": "Forbidden: requires Inference role, caller has ReadOnly"
}
```

### 3. Inference Key Attempting Preload / Unload (`403 Forbidden`)
```bash
curl -X POST http://127.0.0.1:8080/v1/models/demo-model/unload \
  -H "X-API-Key: sk-inference-web-app-654"
```
Response:
```json
{
  "error": "Forbidden: requires Admin role, caller has Inference"
}
```

### 4. Admin Key Preloading Model (`200 OK`)
```bash
curl -X POST http://127.0.0.1:8080/v1/models/demo-model/load \
  -H "X-API-Key: sk-admin-production-key-987"
```
Response:
```json
{
  "status": "loaded",
  "model_id": "demo-model"
}
```

---

## 4. Best Practices for Production

- **Generate High-Entropy Keys**: Use a cryptographically secure random generator:
  ```bash
  openssl rand -hex 32
  ```
- **Environment Secret Injection**: In Kubernetes or Docker Compose, inject keys via command-line arguments derived from secrets:
  ```yaml
  args: ["serve", "--models-dir", "/models", "--api-key", "admin:$(ADMIN_KEY)"]
  ```
- **Key Rotation**: You can configure multiple keys with the same role in `models.toml` to support zero-downtime key rotation.
