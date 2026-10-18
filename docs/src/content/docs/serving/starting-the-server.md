---
title: "Starting the Server"
description: "The core command to run the SnoozeML inference daemon is `snooze serve`. It initializes model discovery, mounts the REST API and telemetry endpoints, launche..."
---

The core command to run the SnoozeML inference daemon is `snooze serve`. It initializes model discovery, mounts the REST API and telemetry endpoints, launches the background scale-to-zero sweeper, and begins accepting inference requests.

---

## 1. Basic Launch Examples

### Serving Models from a Directory
```bash
snooze serve --models-dir ./models
```
SnoozeML will scan `./models`, read `models.toml` if present (or auto-discover all `.onnx` and `.safetensors` files), and bind to `0.0.0.0:8080`.

### Serving with an Explicit Configuration File
```bash
snooze serve --config /etc/snoozeml/models.toml
```

### Running with Mock Testing Demo Model
```bash
snooze serve --demo
```

---

## 2. CLI Options Reference

Below is a complete reference of flags accepted by `snooze serve`:

### Network Configuration

| Flag | Default | Env Var | Description |
| :--- | :--- | :--- | :--- |
| `--host <HOST>` | `0.0.0.0` | — | IP interface address to bind. Use `127.0.0.1` for local-only access. |
| `-p, --port <PORT>` | `8080` | — | TCP port for the HTTP REST server. |
| `--grpc [BOOL]` | `true` | `SNOOZE_GRPC`, `GRPC` | Enable or disable the gRPC server alongside HTTP. |
| `--no-grpc` | `false` | — | Convenience flag to completely disable gRPC and serve only HTTP REST. |
| `--grpc-port <PORT>` | `50051` | `SNOOZE_GRPC_PORT`, `GRPC_PORT` | TCP port for the gRPC server. |

### Model Directory & Config

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--models-dir <DIR>` | *None* (or `SNOOZE_MODELS_DIR`) | Path to directory containing models to discover and register. |
| `--config <PATH>` | *None* | Path to an explicit `models.toml` or `models.json` file. |
| `--demo` | `false` | Registers a built-in mock model (`demo-model`) for smoke testing and CI verification. |

### Runtime & Performance Tuning

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--idle-timeout <SECS>` | `300` | Inactivity duration in seconds before deallocating an idle model (scale-to-zero). |
| `--max-batch-size <NUM>` | `16` | Maximum batch capacity before immediately dispatching a batch forward pass. |
| `--max-delay-ms <MS>` | `10` | Maximum latency budget in milliseconds to wait for concurrent requests to coalesce. |
| `--provider <NAME>` | `auto` | Preferred execution provider (`auto`, `gpu`, `cuda`, `mlx`, `cpu`, `tensorrt`). |
| `--max-loaded-models <N>` | *None* (unbounded) | Maximum number of model instances permitted to reside in RAM/VRAM simultaneously. Triggers LRU eviction when capacity is reached. |

### Security & Authentication

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--auth-enabled` | `false` | Explicitly enforce authentication (automatically enabled if `--api-key` or `--jwt-secret` are set). |
| `--api-key <ROLE:KEY>` | *None* | Register static API key with role in format `<role>:<key>` (repeatable flag). |
| `--jwt-secret <SECRET>` | *None* (or `SNOOZE_JWT_SECRET`) | HMAC secret key used to validate signed JSON Web Tokens. |
| `--protect-metrics` | `false` | When true, restricts access to `/metrics` to clients with the `Admin` role. |

---

## 3. Dual-Protocol Architecture (HTTP REST + gRPC)

By default, SnoozeML binds **both** an HTTP/1.1 & HTTP/2 REST listener and a Tonic gRPC listener under the same OS process:

- **HTTP REST Listener**: Binds to `0.0.0.0:8080` (customized via `--port`). Handles `/v1/models/*`, `/health/*`, `/metrics`, and OpenAPI endpoints.
- **gRPC Listener**: Binds to `0.0.0.0:50051` (customized via `--grpc-port`). Serves `snoozeml.v1.InferenceService` and `grpc.health.v1.Health`.

Both listeners share the identical `AppState`, `ModelRegistry`, dynamic micro-batcher, and scale-to-zero sweeper. A request arriving over gRPC and a request arriving over HTTP for the same model are seamlessly coalesced into the **same batch** along `Axis(0)`.

### Disabling gRPC

If you only require HTTP REST endpoints, you can disable the gRPC listener using:
```bash
snooze serve --no-grpc
# or
snooze serve --grpc=false
# or via environment
GRPC=false snooze serve
```

---

## 4. Production Launch Example

Here is a recommended production configuration running with GPU acceleration, dual REST + gRPC listeners, authentication, bounded memory residency, and structured logging:

```bash
RUST_LOG=info,snooze_ml=info \
snooze serve \
  --host 0.0.0.0 \
  --port 8080 \
  --grpc-port 50051 \
  --models-dir /opt/models \
  --provider auto \
  --idle-timeout 300 \
  --max-batch-size 32 \
  --max-delay-ms 5 \
  --max-loaded-models 4 \
  --api-key admin:sk-admin-cluster-secret \
  --api-key infer:sk-client-app-key \
  --jwt-secret "your-32-character-or-longer-production-jwt-hmac-secret" \
  --protect-metrics
```

---

## 5. Startup Sequence & Auto-Discovery

When `snooze serve` launches:

1. **Configuration Resolution**: If a config file or models directory is supplied, SnoozeML parses it. If syntax errors exist on specific lines, it logs warnings and loads valid entries gracefully.
2. **Filesystem Auto-Discovery**: If `.onnx` or `.safetensors` files exist on disk that are not explicitly defined in `models.toml`, SnoozeML discovers them and registers them automatically with default parameters.
3. **Lazy Registration**: All models start in the `Idle` state. **Weights are not loaded into memory at startup**, ensuring near-instant daemon boot times (typically under 10 ms).
4. **Sweeper Task Initiation**: A background task begins monitoring model activity timestamps according to their configured idle timeouts.
5. **Dual Listener Binding**:
   - The Axum HTTP server binds to the configured host and port (`8080`).
   - If enabled, the Tonic gRPC server binds concurrently to the configured gRPC port (`50051`).

---

## 6. Graceful Shutdown

SnoozeML listens for standard termination signals (`SIGINT` / `Ctrl+C` and `SIGTERM`). An internal `tokio::sync::watch` channel coordinates clean shutdown across both listeners:

- Incoming HTTP and gRPC listeners stop accepting new connections.
- In-flight batches and active inferences finish executing and return results to clients.
- Loaded model sessions and GPU resources are safely released.
- The daemon terminates cleanly with exit code `0`.