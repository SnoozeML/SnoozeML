# SnoozeML End-to-End (E2E) System & Interoperability Tests

This directory (`e2e/`) contains black-box system and wire-interoperability tests that run external client SDKs (such as NVIDIA's official `tritonclient[http,grpc]`) against a live `snooze` binary process.

Unlike `python/tests/` (which tests the `snoozeml` Python SDK) and `tests/` (which tests Rust internals and in-process routers), `e2e/` verifies full protocol compliance from third-party ecosystems and external infrastructure.

---

## Directory Structure

```text
e2e/
├── README.md               # This guide
├── requirements.txt        # Python dependencies for E2E & Testcontainers suites
├── conftest.py             # Shared pytest fixtures (spawns ephemeral `snooze serve` instances)
├── test_triton_grpc.py     # Official `tritonclient.grpc` interoperability suite (unary + streaming + RBAC)
└── test_triton_http.py     # Official `tritonclient.http` interoperability suite (JSON + binary extension + RBAC)
```

---

## Prerequisites

1. **Build the `snooze` binary**:
   ```bash
   cargo build --bin snooze
   ```
   *(The test fixtures automatically select the newest binary between `target/release/snooze` and `target/debug/snooze`, or you can set `SNOOZE_BIN=path/to/snooze`.)*

2. **Install the E2E Python dependencies** (ideally in a virtual environment):
   ```bash
   pip install -r e2e/requirements.txt
   pip install -e ./python
   ```

---

## Running the Tests

### 1. Default Mode (Ephemeral Subprocess)

By default, `pytest` automatically exports real Scikit-Learn ONNX and XGBoost models into a temporary directory, spawns isolated `snooze serve` processes on free local ports (`127.0.0.1:0`), executes all `tritonclient.grpc` and `tritonclient.http` checks, and tears down the daemon on completion:

```bash
# Run the entire E2E suite
pytest e2e -v

# Run only the Triton gRPC client tests (including bidirectional streaming)
pytest e2e/test_triton_grpc.py -v

# Run only the Triton HTTP REST client tests (JSON & binary tensor framing)
pytest e2e/test_triton_http.py -v
```

### 2. External Target Mode (Running Daemon or Container)

To run the unauthenticated E2E suite against an already-running SnoozeML container or remote server, provide `SNOOZE_HTTP_URL` and `SNOOZE_GRPC_URL`:

```bash
SNOOZE_HTTP_URL=localhost:8080 SNOOZE_GRPC_URL=localhost:50051 pytest e2e -k "not rbac" -v
```

---

## What Is Tested

- **Server Health & Metadata**: `is_server_live()`, `is_server_ready()`, `get_server_metadata()`
- **Model Repository & Scale-to-Zero Lifecycle**: `get_model_repository_index()`, `load_model()`, `is_model_ready()`, `unload_model()` on real ONNX models (`sklearn_rf` and `sklearn_iris`)
- **Model Introspection & Telemetry**: `get_model_metadata()`, `get_model_config()`, `get_inference_statistics()`
- **Unary Inference (Zero Mocks)**:
  - Multi-dtype verification (`FP32`, `FP64`, `INT32`, `INT64`)
  - Real Scikit-Learn ONNX (`sklearn_iris`, `sklearn_rf`) and XGBoost (`xgb_iris`) numerical accuracy against ground-truth `predict_proba(X)`
  - Both pure JSON OIP v2 payloads (`binary_data=False`) and Triton's HTTP binary tensor extension (`Inference-Header-Content-Length`, `binary_data=True`)
- **Bidirectional gRPC Streaming**: `start_stream()` + `async_stream_infer()` + `stop_stream()` over `GRPCInferenceService/ModelStreamInfer` against a real ONNX model
- **Out-of-Scope Endpoints**: Asserts `UNIMPLEMENTED` / HTTP `501` for shared memory (`systemsharedmemory`, `cudasharedmemory`), `trace`, and `logging`
- **3-Tier RBAC Security**: Verifies `ReadOnly`, `Inference`, and `Admin` API keys passed via `tritonclient` HTTP/gRPC headers
