# SnoozeML API, CLI & TUI Surface Map

## 1. HTTP REST Endpoints (`src/server/http.rs`)

| Method & Path | Minimum Role | Handler | Description |
| :--- | :--- | :--- | :--- |
| `GET /` | Public | `home_handler` | Service metadata, version, and endpoint directory |
| `GET /health/live` | Public | `liveness_handler` | Kubernetes liveness probe (`{"status": "live"}`) |
| `GET /health/ready` | Public | `readiness_handler` | Kubernetes readiness probe (`{"status": "ready"}`) |
| `GET /metrics` | Public (or `Admin` if `--protect-metrics`) | `metrics_handler` | Prometheus text exposition format (`ServerMetrics::format_prometheus`) |
| `GET /v1/models` | `Role::ReadOnly` | `list_models_handler` | Lists `Vec<ModelMetadata>` for all registered models |
| `POST /v1/models/:model_id/infer` | `Role::Inference` | `infer_handler` | Executes micro-batched tensor inference |
| `POST /v1/models/:model_id/load` | `Role::Admin` | `load_model_handler` | Eagerly loads primary replica weights into RAM/VRAM |
| `POST /v1/models/:model_id/unload` | `Role::Admin` | `unload_model_handler` | Immediately unloads all replicas of `:model_id` |
| `POST /v1/models/:model_id/pin` | `Role::Admin` | `pin_model_handler` | Pins model in memory and eagerly loads weights if idle |
| `POST /v1/models/:model_id/unpin` | `Role::Admin` | `unpin_model_handler` | Unpins model, returning it to scale-to-zero & LRU management |

## 2. gRPC Services (`src/server/grpc/`, `proto/snoozeml/v1/inference.proto`)

| Service & Method | Minimum Role | Implementation |
| :--- | :--- | :--- |
| `grpc.health.v1.Health/Check` | Public | `tonic_health` (`""` and `"snoozeml.v1.InferenceService"`) |
| `snoozeml.v1.InferenceService/Infer` | `Role::Inference` | `SnoozeInferenceService::infer` (`src/server/grpc/service.rs`) |

## 3. CLI Subcommands (`src/cli/mod.rs`)

| Subcommand | Target HTTP Route / Local Action | Test Helper in `src/cli/mod.rs` |
| :--- | :--- | :--- |
| `snooze serve` | Starts HTTP + gRPC daemon & scale-to-zero sweeper | `build_auth_config`, `resolve_serve_models` |
| `snooze status` | `GET /v1/models` | `fetch_models`, `format_status_table` |
| `snooze load [--pin]` | `POST /v1/models/:id/load` (or `/pin` if `--pin`) | `trigger_load_with_pin` |
| `snooze unload` | `POST /v1/models/:id/unload` | `trigger_unload` |
| `snooze pin` | `POST /v1/models/:id/pin` | `trigger_pin` |
| `snooze unpin` | `POST /v1/models/:id/unpin` | `trigger_unpin` |
| `snooze monitor` | Interactive TUI polling `/v1/models` & `/metrics` | `src/cli/tui.rs` |
| `snooze init` / `snooze models init` | Scans models dir & writes `models.toml` | `ModelDirectoryConfig::init_or_update_config_file` |
| `snooze models scan` / `validate` / `sample-config` | Local config inspection | `src/config.rs` |
| `snooze download <REPO>` | Downloads from HF Hub & updates `models.toml` | `src/cli/download.rs` |

## 4. TUI Monitor Keybindings (`src/cli/tui.rs`)

| Key | Action |
| :--- | :--- |
| `q` / `Esc` | Quit monitor |
| `j` / `Down` | Select next model row |
| `k` / `Up` | Select previous model row |
| `l` | Load selected model (`POST /v1/models/:id/load`) |
| `u` | Unload selected model (`POST /v1/models/:id/unload`) |
| `p` | Toggle Pin/Unpin on selected model (`POST /v1/models/:id/{pin,unpin}`) |
| `r` | Force immediate refresh |
