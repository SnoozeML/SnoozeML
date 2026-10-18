# AGENTS.md — SnoozeML Engineering & Architecture Guide

This document provides cross-vendor context and engineering rules for AI coding agents working in the **SnoozeML** repository.

---

## 1. Project Overview

**SnoozeML** (`snooze-ml`, binary name `snooze`) is a single-binary, bare-metal deep learning and machine learning inference daemon written in Rust. It combines:
1. **Adaptive Dynamic Micro-Batching** along `Axis(0)` (`src/batcher.rs`) to coalesce concurrent requests within a configurable batch size or latency budget.
2. **Scale-to-Zero Memory Reclamation & LRU Eviction** (`src/sweeper.rs`, `src/registry.rs`) to automatically unload inactive models from GPU VRAM / host RAM, enforce `max_loaded_models` capacity caps, and recover from reactive OOM errors.
3. **Model Pinning** (`src/model.rs`, `src/registry.rs`) to eagerly load critical models at startup or runtime and exempt them from scale-to-zero and LRU eviction.
4. **Vertical Multi-Replica Autoscaling & Routing** (`src/routing.rs`, `src/registry.rs`) to spin up additional model replicas under queue pressure and distribute traffic via `LeastQueue`, `RoundRobin`, or `Random` strategies with differential scale-down timeouts.
5. **Multi-Runtime Execution** (`src/backend/`) supporting ONNX Runtime (`ort`), Hugging Face Safetensors (`candle`), pure-Rust XGBoost/GBDT trees (`hessboost`), and deterministic mock models (`mock:`).
6. **Dual HTTP REST & Native gRPC Transports** (`src/server/`) with optional shared TLS (`rustls`) and 3-tier Role-Based Access Control (`Admin`, `Inference`, `ReadOnly` via constant-time API keys and HMAC JWTs).

---

## 2. Repository & Module Map

```text
SnoozeML/
├── Cargo.toml                  # Crate manifest & feature flags (ort-backend, candle-backend, tree-backend, cuda, tensorrt, coreml)
├── build.rs                    # Compiles proto/snoozeml/v1/inference.proto via tonic-build + protoc-bin-vendored
├── proto/snoozeml/v1/          # SnoozeML native gRPC schema (snoozeml.v1.InferenceService)
│   └── inference.proto
├── .agents/                    # Cross-vendor AI agent rules & on-demand workflow skills
│   ├── rules/                  # Always-on rules (rust.md, architecture-invariants.md, api-and-security.md)
│   └── skills/                 # Progressive-disclosure skills (add-inference-backend, add-api-and-cli-command, verification-and-benchmarking)
├── src/
│   ├── lib.rs                  # Crate root; enforces #![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)] & #![warn(missing_docs)]
│   ├── main.rs                 # Binary entrypoint (`snooze`)
│   ├── types.rs                # TensorPayload, InferenceError, ModelState, ModelMetadata
│   ├── model.rs                # ModelConfig & ModelController (lifecycle state machine, lazy load, pin/unpin, idle unload)
│   ├── registry.rs             # ModelRegistry & ModelReplicaGroup (concurrency, autoscaling, LRU capacity/OOM eviction)
│   ├── routing.rs              # RoutingStrategy trait & implementations (LeastQueue, RoundRobin, Random)
│   ├── batcher.rs              # DynamicBatcher background worker, Axis(0) tensor collation & output splitting
│   ├── sweeper.rs              # Background scale-to-zero sweeper task with differential replica timeouts
│   ├── config.rs               # Fault-tolerant models.toml / models.json parser, model discovery & stub generator
│   ├── auth.rs                 # API Key (constant-time SHA-256) & JWT authentication + RBAC middleware
│   ├── backend/
│   │   ├── mod.rs              # ModelBackend trait & create_backend factory
│   │   ├── ort_backend.rs      # ONNX Runtime backend (CPU, CUDA, TensorRT, CoreML)
│   │   ├── candle_backend.rs   # Native Safetensors backend (sequential Linear/MLP + TransformerOrVision)
│   │   ├── local_loader.rs     # Dynamic HF Safetensors loader (BERT, JinaBERT, NomicBERT, DistilBERT, CLIP, ViT, DINOv2)
│   │   ├── tree_backend.rs     # Pure-Rust XGBoost/GBDT backend via hessboost
│   │   └── mock_backend.rs     # Deterministic mock backend ("mock:*") including simulated OOM testing
│   ├── server/
│   │   ├── mod.rs              # Dual HTTP + gRPC server orchestrator, shared Rustls/Tonic TLS & graceful shutdown
│   │   ├── http.rs             # Axum REST API (/v1/models, /infer, /load, /unload, /pin, /unpin, /health/*, /metrics)
│   │   ├── metrics.rs          # Lock-free atomic Prometheus metrics collector
│   │   └── grpc/               # Tonic gRPC server subsystem
│   │       ├── mod.rs          # Generated snoozeml.v1 proto re-exports
│   │       ├── service.rs      # SnoozeInferenceService (Infer RPC + metadata auth)
│   │       └── convert.rs      # Overflow-safe TensorData <-> TensorPayload & InferenceError -> tonic::Status conversion
│   └── cli/
│       ├── mod.rs              # Clap CLI subcommands (serve, status, load, unload, pin, unpin, monitor, models, init, download)
│       ├── download.rs         # Hugging Face Hub repository downloader (hf-hub)
│       └── tui.rs              # Interactive Ratatui + Crossterm terminal dashboard
├── tests/                      # Dedicated integration & behavioral test suite (no mocks)
├── examples/                   # Interactive Jupyter notebooks (Scikit-learn ONNX, XGBoost tree, Pipeline regression)
├── python/                     # Official `snoozeml` Python SDK (HTTP/gRPC client, converters, Sphinx docs & pytest suite)
├── tools/                      # Python load tester (perf_test.py), gRPC verifier (test_grpc.py) & XGBoost trainer (train_tree.py)
├── docs/                       # Astro Starlight user & architecture documentation
└── design_docs/                # Technical design specifications (gRPC, Python SDK)
```

---

## 3. Strict Engineering & Rust Rules

All code in `src/` is compiled under `#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]` and `#![warn(missing_docs)]`. Every contribution **must** adhere to the following rules (detailed in `.agents/rules/rust.md`, `.agents/rules/architecture-invariants.md`, and `.agents/rules/api-and-security.md`):

### 3.1 Zero-Panic Policy & Defensive Design
- **Forbidden Constructs in `src/`**: Never use `.unwrap()`, `.expect()`, `panic!()`, `unreachable!()`, or `todo!()` outside of test modules (`#[cfg(test)]` or `tests/`).
- **No Direct Indexing**: Never index slices or vectors directly (`vec[0]`, `slice[i]`) in production code. Always use `.get()`, `.first()`, `.get_mut()`, or iterators with explicit bounds handling.
- **Checked Arithmetic & Conversions**: Use `checked_add`, `checked_mul`, `saturating_sub`, and `TryFrom`/`TryInto` when validating external tensor shapes, sizes, or durations.
- **Lock Poisoning**: Never `.unwrap()` a `std::sync::Mutex` or `RwLock` guard; handle poison errors or recover via `.unwrap_or_else( std::sync::PoisonError::into_inner )` where appropriate.

### 3.2 Documentation & Rustdoc Standards
- **Document Every `pub` Item**: Because `#![warn(missing_docs)]` is active and CI runs `cargo clippy --all-targets -- -D warnings`, every `pub` struct, enum, variant, struct field, trait, and function in `src/` must have a `///` docstring.
- **`# Errors` Sections**: Include `# Errors` doc sections on fallible `pub` functions describing failure conditions.
- **No Full/Absolute Paths in Documentation**: Never add full or absolute filesystem paths (such as `file:///...`, `C:\Users\...`, or `/home/...`) in documentation, markdown files, notebooks, or docstrings. Always use relative repository paths (e.g. `examples/01_sklearn_onnx.ipynb`, `src/types.rs`).

### 3.3 Observability & Error Handling
- **Structured Telemetry**: Use `tracing` with structured key-value fields (`tracing::info!(model_id = %id, duration_ms = ?elapsed, "message")`) rather than string interpolation. Never log raw API keys, JWT secrets, or sensitive credentials.
- **Pragmatic Errors**: Map domain errors into `InferenceError` (`src/types.rs`) with clear diagnostic context so HTTP (`AppError`) and gRPC (`tonic::Status`) return accurate status codes.

### 3.4 No Premature Abstraction & Testing Policy
- **Rule of Two**: Keep logic inline unless the exact helper is needed in at least two call sites. Avoid single-use wrapper traits or unnecessary indirection.
- **Dedicated `tests/` Directory**: Place integration and behavioral tests in `tests/`. Keep inline `#[cfg(test)]` blocks in `src/` strictly limited to testing `pub(crate)` or private internals that cannot be reached via public APIs.
- **Zero Mocks**: Exercise real state machines, real HTTP/gRPC routers (`tower::ServiceExt::oneshot` or ephemeral local listeners), and the built-in `mock:` tensor backend rather than mock traits.

---

## 4. Build, Test & Lint Commands

Always run both the test suite and strict Clippy lints before completing a task:

```bash
# Run all unit and integration tests
cargo test --all-targets

# Run strict Clippy static analysis (must pass with zero warnings)
cargo clippy --all-targets -- -D warnings

# Verify formatting
cargo fmt --check
```

---

## 5. Documentation & Roadmap Pointers

To keep this file evergreen and avoid duplicate maintenance, **do not hardcode feature checklists here**. Instead, consult the single sources of truth in the repository:

- **Feature Roadmap & Status**: Check the `# TODO` section in [`README.md`](README.md) before starting new feature work and mark items `[x]` there when completed.
- **Technical Design Specifications**:
  - [`design_docs/grpc_design.md`](design_docs/grpc_design.md) — Native gRPC & future KServe v2 / Triton protocol design.
  - [`design_docs/python_sdk_design.md`](design_docs/python_sdk_design.md) — Planned `snoozeml-py` Python SDK & automatic ONNX export architecture.
- **User & Architecture Documentation**: Located under [`docs/`](docs/) (Astro Starlight format; build with `npm --prefix docs run build` and `sphinx-build -b html python/docs docs/dist/python`).
