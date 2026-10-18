---
name: verification-and-benchmarking
description: >-
  Runbook for executing SnoozeML's quality assurance gates, integration tests,
  synthetic OOM/autoscaling tests, Python HTTP vs gRPC equivalence tests, load
  benchmarks, and Astro Starlight + Sphinx documentation builds. Use this skill when validating
  code changes, writing new integration tests in tests/, benchmarking throughput
  or scale-to-zero behavior, or updating docs/.
---

# Verification, Testing & Benchmarking Runbook

SnoozeML enforces zero-warning static analysis and zero-mock integration testing across `tests/` and `tools/`. Follow this runbook to verify changes thoroughly.

---

## 1. Mandatory Rust QA Gate

Before completing any code change, run all three Cargo checks:

```powershell
# 1. Run all unit, behavioral, and integration tests across all targets
cargo test --all-targets

# 2. Run strict Clippy lints (enforces #![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)] and #![warn(missing_docs)])
cargo clippy --all-targets -- -D warnings

# 3. Verify rustfmt formatting
cargo fmt --check

# 4. Run Python SDK, converters, and worker pytest suite
pytest python/tests -v
```

### Feature Matrix Checks (When Modifying `src/backend/`, `src/stream/` or `Cargo.toml`)
```powershell
# Verify compilation with all optional backends disabled (mock backend only)
cargo check --no-default-features

# Verify pure-Rust tree backend in isolation
cargo check --no-default-features --features tree-backend

# Verify Candle Safetensors backend in isolation
cargo check --no-default-features --features candle-backend

# Verify ONNX Runtime backend in isolation
cargo check --no-default-features --features ort-backend

# Verify isolated streaming broker backends
cargo check --no-default-features --features streams
cargo check --no-default-features --features streams-kafka
cargo check --no-default-features --features streams-redis
cargo check --no-default-features --features streams-rabbitmq
```

---

## 2. Writing Integration Tests in `tests/` (Zero Mocks Policy)

All behavioral and integration tests belong in `tests/` (see [references/test-patterns.md](./references/test-patterns.md) for copy-pasteable patterns):

| Test Suite | Focus Area |
| :--- | :--- |
| `tests/integration_test.rs` | End-to-end HTTP REST routes via `tower::ServiceExt::oneshot` |
| `tests/grpc_test.rs` | Tonic gRPC `InferenceService` and `grpc.health.v1.Health` over ephemeral TCP ports |
| `tests/auth_test.rs` | API Key (constant-time SHA-256), HMAC JWT (HS256/384/512), and 3-tier RBAC (`ReadOnly`, `Inference`, `Admin`) |
| `tests/tls_test.rs` | Self-signed PEM cert generation (`rcgen`) and HTTPS + gRPC TLS handshakes |
| `tests/model_test.rs` | `ModelController` state transitions (`Idle`, `Loading`, `Ready`, `Busy`), pinning, idle deallocation |
| `tests/registry_test.rs` | `ModelRegistry` LRU capacity eviction (`max_loaded_models`), reactive OOM recovery, anti-thrashing (`min_residency`) |
| `tests/autoscale_test.rs` | Multi-replica horizontal/vertical autoscaling, routing (`LeastQueue`, `RoundRobin`, `Random`), differential scale-down |
| `tests/sweeper_test.rs` | Background scale-to-zero sweeper task (`spawn_scale_to_zero_sweeper`) |
| `tests/batcher_test.rs` | `DynamicBatcher` `Axis(0)` tensor concatenation, single-job fast path, proportional output splitting |
| `tests/safetensors_test.rs` | Native Candle Safetensors loading (BERT, JinaBert, NomicBert, DistilBert, CLIP, ViT, DINOv2, Linear MLP) |
| `tests/tree_test.rs` | Pure-Rust XGBoost / `hessboost` JSON & binary tree inference (`Value`, `Margin`, `Class`) |
| `tests/worker_backend_test.rs` | Process-isolated Python worker spawning, JSON-lines / binary IPC framing, child process reaping |
| `tests/stream_test.rs` | Kafka, RabbitMQ, Redis, and Mock stream event ingestion, backpressure, and ack-after-infer |
| `tests/config_test.rs` | Fault-tolerant `models.toml` / `models.json` parsing, directory discovery, stub generation |
| `tests/cli_test.rs` & `tests/tui_test.rs` | CLI argument parsing, HTTP client helpers, Ratatui `TestBackend` rendering & keybindings |


---

## 3. Built-in `mock:` Backend Identifiers for Deterministic Testing

Never create mock traits in tests. Instead, register a `ModelController` with one of SnoozeML's built-in `onnx_path` mock URIs (`src/backend/mod.rs` & `src/backend/mock_backend.rs`):

- `"mock:test"` or `"mock"`: Succeeds deterministically; outputs tensor `"output"` of shape `[batch_size, 4]` where `val = (batch_idx * 10.0) + col_idx`.
- `"mock:oom"`: Always returns `InferenceError::OutOfMemory("Mock allocation failed: CUDA failure 2: out of memory")`.
- `"mock:bad_alloc"`: Always returns `InferenceError::Runtime("Mock allocation failed: std::bad_alloc: out of host memory")` (recognized by `is_out_of_memory()`).
- `"mock:oom_once"` / `"mock:oom_configurable"`: Fails with `OutOfMemory` for `N` attempts configured via `snooze_ml::backend::mock_backend::set_mock_oom_failures(N)`, then succeeds on retry (ideal for testing reactive OOM eviction of idle/LRU models).
- `"mock:corrupted"`: Simulates corrupted weights (`InferenceError::Runtime`, non-OOM) to verify that non-OOM errors do not trigger false model evictions.

---

## 4. Python Tooling: gRPC Equivalence, Load Testing & Tree Training (`tools/`)

When testing against a live running daemon (`cargo run --release -- serve --demo`):

1. **HTTP vs. gRPC Numerical Equivalence & Health Suite**:
   ```powershell
   python tools/test_grpc.py --compare
   python tools/test_grpc.py --benchmark --iterations 50
   ```
2. **Capacity Ramp-Up, Sustained Load & Scale-to-Zero Benchmark**:
   ```powershell
   python tools/perf_test.py --mode all --no-docker --html-report tools/report.html
   ```
3. **Train & Verify Pure-Rust XGBoost Models**:
   ```powershell
   python tools/train_tree.py --dataset iris --name iris_model --verify
   ```

---

## 5. Building the User Documentation (`docs/`)

Whenever user-facing CLI flags, `models.toml` options, or HTTP/gRPC endpoints change, update the corresponding markdown page in `docs/src/content/docs/` and verify the combined Starlight and Sphinx docs build cleanly:

```powershell
# 1. Build Astro Starlight main portal and guides
npm --prefix docs run build

# 2. Build Sphinx Python SDK reference into Starlight dist
sphinx-build -b html python/docs docs/dist/python
```
