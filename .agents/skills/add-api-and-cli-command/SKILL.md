---
name: add-api-and-cli-command
description: >-
  End-to-end runbook for adding a new model lifecycle operation, HTTP REST
  endpoint, gRPC RPC method, CLI subcommand/flag, or TUI dashboard keybinding
  in SnoozeML. Use this skill when extending the REST API (src/server/http.rs),
  gRPC protocol (proto/snoozeml/v1/inference.proto, src/server/grpc/), CLI
  commands (src/cli/mod.rs), or interactive TUI monitor (src/cli/tui.rs).
---

# Adding an API Endpoint, gRPC RPC, CLI Command & TUI Action

In SnoozeML, model management and inference operations span six layers:
1. **Controller & Registry (`src/model.rs`, `src/registry.rs`, `src/types.rs`)**
2. **Configuration (`src/config.rs`)**
3. **HTTP REST API (`src/server/http.rs`)**
4. **gRPC Protobuf & Service (`proto/snoozeml/v1/inference.proto`, `src/server/grpc/`)**
5. **CLI Subcommands & Flags (`src/cli/mod.rs`)**
6. **Interactive TUI Dashboard (`src/cli/tui.rs`)**

Use the checklist below to ensure every layer, RBAC role, metric counter, and integration test stays in sync. For exact route and RBAC tables, see [references/surface-map.md](./references/surface-map.md).

---

## Step-by-Step Workflow

### 1. Core Domain & Registry (`src/types.rs`, `src/model.rs`, `src/registry.rs`)
1. If new model metadata is exposed to clients, add the field to `ModelMetadata` in `src/types.rs` with `#[serde(default)]` for backward compatibility, and populate it in `ModelController::metadata()` (`src/model.rs`).
2. Add the state transition or operation method on `ModelController` (`src/model.rs`) and expose a high-level coordinator method on `ModelRegistry` (`src/registry.rs`).
   - Note whether the operation applies only to the primary replica (`group.primary().await?`) or across all replicas (`group.all_replicas().await`).

### 2. Configuration File Support (`src/config.rs`)
1. If the setting can be configured in `models.toml` / `models.json`:
   - Add `pub <field>: Option<T>` with `#[serde(default, skip_serializing_if = "Option::is_none")]` to both `GlobalModelDefaults` and `ModelEntryConfig`.
   - Update `ModelDirectoryConfig::parse_fault_tolerant` (both TOML and JSON parsers) and `ModelDirectoryConfig::build_model_configs` so per-model values override `[defaults]`, which override `DEFAULT_*` constants.

### 3. HTTP REST Endpoint (`src/server/http.rs`)
1. Write the Axum handler function returning `Result<Json<...>, AppError>` (or `impl IntoResponse`).
2. Mount the route in `create_router(state: AppState)` under the appropriate RBAC router tier:
   - `public_routes`: Unauthenticated probes (`GET /`, `/health/live`, `/health/ready`).
   - `readonly_routes` (`Role::ReadOnly`): Read-only metadata (`GET /v1/models`).
   - `infer_routes` (`Role::Inference`): Inference execution (`POST /v1/models/:model_id/infer`).
   - `admin_routes` (`Role::Admin`): Lifecycle mutations (`load`, `unload`, `pin`, `unpin`, etc.).
3. If a new `InferenceError` variant was introduced, map it in `AppError::into_response`.

### 4. gRPC Service & Protobuf (`proto/snoozeml/v1/inference.proto`, `src/server/grpc/`)
1. When modifying gRPC messages or RPCs:
   - Edit `proto/snoozeml/v1/inference.proto` (`build.rs` automatically compiles it via `tonic-build` and `protoc-bin-vendored`).
   - Implement the trait method in `src/server/grpc/service.rs` (`impl InferenceService for SnoozeInferenceService`), calling `self.authenticate_metadata(&request)?` first.
   - Map any new domain types or errors in `src/server/grpc/convert.rs` (`inference_error_to_status`).

### 5. CLI Subcommand & Flags (`src/cli/mod.rs`)
1. Add the subcommand variant to `pub enum Commands` (or flag to `Commands::Serve` / `Commands::Load`) with `///` docstrings and `#[arg(...)]` attributes.
2. Implement the client HTTP call in `run_cli(cli: Cli)` using `ureq`, and extract a testable helper function (e.g., `pub fn trigger_<action>(endpoint: &str, model_id: &str) -> Result<ModelMetadata, String>`) so `tests/cli_test.rs` can test it directly.
3. If `ModelMetadata` gained a field, update the table output in `query_status` and `models scan` if relevant.

### 6. Interactive TUI Monitor (`src/cli/tui.rs`)
1. If the operation is an interactive model action (like Load `l`, Unload `u`, Pin/Unpin `p`, Refresh `r`):
   - Add the keybinding in `run_tui_loop` (`KeyCode::Char(...)`).
   - Update the footer help bar in `draw_dashboard` and table columns if new metadata is displayed.

### 7. Integration Tests & Documentation
1. Add end-to-end tests in:
   - `tests/integration_test.rs` (HTTP endpoint + RBAC role checks in `tests/auth_test.rs`)
   - `tests/cli_test.rs` (CLI parsing and client helper execution)
   - `tests/tui_test.rs` (TUI rendering and keybinding state transitions)
   - `tests/grpc_test.rs` (if gRPC was touched)
2. Update `docs/src/reference/api-reference.md` and `docs/src/reference/cli-reference.md`.
3. Run verification:
   ```powershell
   cargo test --all-targets
   cargo clippy --all-targets -- -D warnings
   ```
