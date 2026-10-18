---
title: "Securing Endpoints & Probes"
description: "SnoozeML enforces endpoint security using a strict permissions matrix, ensuring sensitive operations are protected while cluster health checks remain frictio..."
---

SnoozeML enforces endpoint security using a strict permissions matrix, ensuring sensitive operations are protected while cluster health checks remain frictionless.

---

## 1. Complete Endpoint Permission Matrix

| Protocol | Endpoint / RPC | Method | Required Role | Description |
| :--- | :--- | :--- | :--- | :--- |
| **HTTP** | `/` | `GET` | **Public** | Daemon metadata, version, and route links |
| **HTTP** | `/health/live` | `GET` | **Public** | Process liveness probe for orchestrators |
| **HTTP** | `/health/ready` | `GET` | **Public** | Model server readiness probe |
| **HTTP** | `/metrics` | `GET` | **Public** (or **`Admin`**) | Prometheus metrics (governed by `protect_metrics`) |
| **HTTP** | `/v1/models` | `GET` | **`ReadOnly`** | List registered models, states, and telemetry |
| **HTTP** | `/v1/models/:id/infer` | `POST` | **`Inference`** | Execute batched tensor inference |
| **HTTP** | `/v1/models/:id/load` | `POST` | **`Admin`** | Force model session into memory (pre-warm) |
| **HTTP** | `/v1/models/:id/unload`| `POST` | **`Admin`** | Immediately deallocate model weights (scale-to-zero) |
| **gRPC** | `grpc.health.v1.Health/Check` | RPC | **Public** | Standard gRPC health checking probe |
| **gRPC** | `grpc.health.v1.Health/Watch` | RPC | **Public** | Streaming gRPC health checking probe |
| **gRPC** | `snoozeml.v1.InferenceService/Infer` | RPC | **`Inference`** | Execute binary batched tensor inference |

*Note: Roles are cumulative. An `Admin` caller is authorized for all endpoints.*

---

## 2. Why Health Probes Are Public

In Kubernetes, Nomad, and Docker Swarm environments, cluster control planes periodically ping `/health/live`, `/health/ready`, or query `grpc.health.v1.Health/Check` to evaluate container health and route network traffic.

If health probes required authentication:
- Orchestrators would need API secrets mounted into kubelet or ingress controllers.
- Secret rotations could accidentally cause false container restarts or drop pods from service endpoints.

SnoozeML deliberately keeps `/health/live`, `/health/ready`, and `grpc.health.v1.Health` public. They return minimal status payloads (`{"status":"live"}` or `SERVING`) and leak no model data or intellectual property.

---

## 3. Protecting Prometheus Metrics (`/metrics`)

By default, `/metrics` is accessible without authentication so standard Prometheus scrapers can collect metrics easily.

If your Prometheus metrics contain sensitive information (such as internal model identifiers or request volumes) and you wish to protect them:

### Enable Metrics Protection
In `models.toml`:
```toml
[auth]
enabled = true
protect_metrics = true
```

Or via CLI:
```bash
snooze serve --models-dir ./models --protect-metrics --api-key admin:sk-admin-secret-987
```

### Scraping Protected Metrics in Prometheus
In your `prometheus.yml` scrape configuration, supply the `Admin` API key or Bearer token:

```yaml
scrape_configs:
  - job_name: "snoozeml"
    static_configs:
      - targets: ["snooze-daemon:8080"]
    authorization:
      type: Bearer
      credentials: "sk-admin-production-key-987"
```

---

## 4. Troubleshooting Authentication Failures

### `401 Unauthorized`
The request did not provide valid credentials.
- Ensure the header name is exact: `X-API-Key` or `Authorization: Bearer <token>`.
- If using JWT, verify that the token has not expired and that the shared HMAC secret matches.

### `403 Forbidden`
The credentials were valid, but the associated role lacks permission for the endpoint.
- For example, an `Inference` key cannot call `/v1/models/{id}/load`. Use an `Admin` key for lifecycle management.