# SnoozeML

A high-performance, single-binary, bare-metal inference daemon written in Rust for deep learning and ONNX models.

SnoozeML combines **zero-idle scale-to-zero memory reclamation** (automatically deallocating inactive models from GPU VRAM and host RAM) with **adaptive dynamic micro-batching** along `Axis(0)` to maximize hardware saturation without requiring client-side coordination.

---

## Key Features

- **Dynamic Micro-Batching**: Asynchronously aggregates independent concurrent inference requests within a configurable time or capacity window into contiguous multi-dimensional batch tensors.
- **Scale-to-Zero Lifecycle**: Automatically releases idle weights from VRAM/RAM after an inactivity threshold, lazily rehydrating them with sub-millisecond coordination upon incoming traffic.
- **Multi-Runtime Execution**: Supports ONNX Runtime ([`ort`](https://github.com/pykeio/ort)), native Hugging Face Safetensors via [Candle](https://github.com/huggingface/candle), and pure-Rust Decision Trees & XGBoost via [`hessboost`](https://github.com/brndnmtthws/hessboost).
- **Modular Hardware Acceleration**: Supports CPU execution, NVIDIA CUDA, Apple Silicon Metal/MLX, and TensorRT backends.
- **Pure-Rust Tree Models**: Zero-FFI, zero-C++ XGBoost JSON model serving powered by [`hessboost`](https://github.com/brndnmtthws/hessboost) with companion Python training tooling (`tools/train_tree.py`).
- **Event Streaming & Message Brokers**: Asynchronous, non-blocking ingestion and output publishing across Apache Kafka, RabbitMQ, and Redis Streams with automatic topic creation, dead-letter queue (DLQ) routing, and micro-batching integration.
- **Production-Grade Reliability**: Enforces a strict zero-panic policy (`#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]`), bounds checks, saturating conversions, and structured `tracing` telemetry.
- **Built-in CLI & TUI Monitor**: Includes client subcommands to inspect model states, trigger manual load/unload, and view real-time terminal metrics dashboards.
- **Full Observability**: Exposes standardized Prometheus exposition metrics (`/metrics`) and Kubernetes-compatible health probes (`/health/live`, `/health/ready`).

---

## Prerequisites

- **Rust**: Version 1.75 or later with Cargo.
- **C/C++ Build Tools**: Standard build essentials (e.g. `gcc` / `clang`, `make`, `cmake`) for native dependency compilation.
- **(Optional) NVIDIA CUDA / TensorRT**: CUDA Toolkit and cuDNN/TensorRT libraries if compiling with hardware acceleration features enabled.

---

## Building

Clone the repository and compile SnoozeML with Cargo:

```bash
git clone https://github.com/SnoozeML/SnoozeML.git
cd SnoozeML
```

### Build Profiles

| Target | Command | Description |
| :--- | :--- | :--- |
| **Standard (Default)** | `cargo build --release` | Default ONNX, Candle Safetensors, and Tree backends |
| **NVIDIA CUDA** | `cargo build --release --features cuda` | Enables CUDA hardware acceleration provider |
| **NVIDIA TensorRT** | `cargo build --release --features tensorrt` | Enables TensorRT acceleration provider |
| **Tree Backend Only** | `cargo build --release --no-default-features --features tree-backend` | Ultra-lightweight pure-Rust build for tree/XGBoost models only |
| **Offline / Mock** | `cargo build --release --no-default-features` | Lightweight build without native ORT libraries |

The compiled standalone binary will be located at `target/release/snooze`.

---

## Running the Server

### 1. Quick Start

To serve models from a local directory (e.g. `./models`):

```bash
cargo run --release -- serve --models-dir ./models
# Or directly using the compiled binary:
./target/release/snooze serve --models-dir ./models
```

To run with the built-in mock demo model for testing without downloading weights, pass `--demo`:

```bash
./target/release/snooze serve --demo
```

By default:
- Binds to `0.0.0.0:8080`.
- Scans and registers models found in `--models-dir` (or registers `demo-model` if `--demo` is passed).
- Background scale-to-zero sweeper runs periodically to reclaim idle weights.

### 2. Downloading Hugging Face Models
SnoozeML includes a built-in CLI command to download repositories directly from the Hugging Face Hub using the official `hf-hub` Rust client:

```bash
# Download a model into the default models folder (./models or SNOOZE_MODELS_DIR)
# Automatically initializes or updates models.toml with the newly discovered model!
snooze download sentence-transformers/all-MiniLM-L6-v2

# Download with include filter (pass --no-init to skip updating models.toml)
snooze download sentence-transformers/all-MiniLM-L6-v2 --include "*.onnx,config.json"
```

The models folder can be configured via the `SNOOZE_MODELS_DIR` environment variable (defaulting to `./models`) or `--models-dir`. After download, SnoozeML automatically runs model discovery and updates `models.toml` unless `--no-init` is passed.

### 3. Serving Real ONNX Models

SnoozeML executes ONNX models using the [`ort`](https://github.com/pykeio/ort) Rust runtime wrapper for Microsoft's [ONNX Runtime](https://github.com/microsoft/onnxruntime). Place your `.onnx` model files in a directory (for example `./models/`):

```bash
./target/release/snooze serve \
  --host 0.0.0.0 \
  --port 8080 \
  --models-dir ./models \
  --idle-timeout 300 \
  --max-batch-size 16 \
  --max-delay-ms 10 \
  --provider cpu
```

### 4. Serving Pure-Rust Tree Models (XGBoost)

SnoozeML natively runs Gradient Boosted Decision Tree (GBDT) models with zero C/C++ dependencies and zero Python overhead, powered by [`hessboost`](https://github.com/brndnmtthws/hessboost) (a pure-Rust, zero-FFI implementation compatible with [XGBoost](https://github.com/dmlc/xgboost)).

1. **Train & Export with Python**:
   ```bash
   python tools/train_tree.py --dataset iris --name iris_model
   ```
   Or export directly with standard Python XGBoost:
   ```python
   booster.save_model("models/iris_model.xgb.json")
   ```

2. **Serve with SnoozeML**:
   ```bash
   snooze serve --models-dir ./models
   ```
   SnoozeML automatically discovers `.xgb.json`, `.tree.json`, `.xgb`, or XGBoost JSON files and dynamically micro-batches them with scale-to-zero memory reclamation!

### 5. Serving Custom User Models (Python Process Workers)

SnoozeML can serve arbitrary custom architectures at runtime via process-isolated workers (`worker-backend`):
1. Drop a folder containing `snooze_model.py` into your `models/` directory:
   ```python
   # models/my-net/snooze_model.py
   class Model:
       def load(self, model_dir: str, device: str) -> None:
           self.device = device
       def predict(self, inputs: dict) -> dict:
           return {"output": inputs["input"] * 2.0}
   ```
2. Run `snooze serve --models-dir ./models`. SnoozeML automatically spawns dedicated worker processes per replica communicating over an ultra-low-overhead binary pipe protocol. Scale-to-zero unloads kill the process, guaranteeing 100% host RAM and GPU VRAM deallocation.

> [!NOTE]
> **Performance Trade-Off (Use as Last Resort)**: Custom Python workers provide complete architectural flexibility, but they are **slower than native compiled engines** and have **cold starts in the seconds magnitude (2s–10s+)** due to Python process spawning and framework imports (e.g. PyTorch). Always prefer native formats (`.onnx`, Safetensors, or `.xgb.json`) first; use custom workers only when native export is not feasible. See [`docs/src/content/docs/serving/custom-models.md`](docs/src/content/docs/serving/custom-models.md) for full trade-offs.

### CLI Server Options

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--host <HOST>` | `0.0.0.0` | Network interface address to bind |
| `-p, --port <PORT>` | `8080` | TCP port to bind |
| `--models-dir <DIR>` | *None* | Directory of `.onnx` models to discover and register automatically |
| `--idle-timeout <SECONDS>` | `300` | Inactivity duration before deallocating weights from RAM/VRAM |
| `--max-batch-size <SIZE>` | `16` | Maximum micro-batch size before triggering immediate forward pass |
| `--max-delay-ms <MS>` | `10` | Maximum latency budget before dispatching partial batches |
| `--provider <PROVIDER>` | `cpu` | Execution provider preference (`cpu`, `cuda`, `tensorrt`) |
| `--pin <MODEL_ID>` | *None* | Pin specified model(s) in memory at startup (never automatically deallocated; repeatable) |
| `--demo` | `false` | Enable mock demo model (`demo-model`) for testing |

### 4. Running with Docker & Docker Compose

SnoozeML features a lightweight multi-stage Docker build powered by `cargo-chef` on Debian Trixie (`lukemathwalker/cargo-chef` and `debian:trixie-slim`), caching crate dependencies for rapid rebuilds with non-root execution (`appuser:10001`):

```bash
# Build the official image (Linux/macOS bash or Windows PowerShell):
docker build -t snoozeml:latest .

# Run with local models volume mount
docker run -d \
  --name snoozeml \
  -p 8080:8080 \
  -v $(pwd)/models:/models:ro \
  snoozeml:latest

# Or launch with Docker Compose
docker compose up -d

# Run mock demo mode directly
docker run -it --rm -p 8080:8080 snoozeml:latest serve --demo
```



---

## Sending Inference Requests

Inference requests are sent via HTTP `POST` to `/v1/models/{model_id}/infer`.

### Example Request (`curl`)

```bash
curl -X POST http://127.0.0.1:8080/v1/models/demo-model/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "name": "input",
        "shape": [1, 3],
        "data": [1.0, 2.0, 3.0]
      }
    ]
  }'
```

### Example Response (`200 OK`)

```json
{
  "model_id": "demo-model",
  "outputs": [
    {
      "name": "output",
      "shape": [1, 4],
      "data": [1.0, 2.0, 3.0, 0.0]
    }
  ],
  "duration_ms": 1.25
}
```

---

## Model Management & Scale-to-Zero

SnoozeML provides HTTP endpoints and CLI commands to inspect and manage model lifecycles:

### Check Status & Models

```bash
# Using CLI:
./target/release/snooze status --endpoint http://127.0.0.1:8080

# Using HTTP API:
curl -s http://127.0.0.1:8080/v1/models | jq .
```

### Preload Model into Memory

Warm up model weights in VRAM/RAM ahead of scheduled traffic:

```bash
# Using CLI (pass --pin to also pin the model in memory):
./target/release/snooze load --model-id demo-model

# Using HTTP API:
curl -X POST http://127.0.0.1:8080/v1/models/demo-model/load
```

### Pin & Unpin Models in Memory

Pin a model (`pinned = true` in `models.toml`, `--pin <MODEL_ID>` at startup, or dynamically via CLI/HTTP) so it is eagerly loaded and exempt from scale-to-zero idle sweeping and LRU eviction until explicitly unpinned or unloaded:

```bash
# Pin a model via CLI:
./target/release/snooze pin --model-id demo-model

# Unpin a model via CLI:
./target/release/snooze unpin --model-id demo-model

# Using HTTP API:
curl -X POST http://127.0.0.1:8080/v1/models/demo-model/pin
curl -X POST http://127.0.0.1:8080/v1/models/demo-model/unpin
```

### Deallocate Model from Memory

Immediately release model weights to reclaim memory (scale-to-zero):

```bash
# Using CLI:
./target/release/snooze unload --model-id demo-model

# Using HTTP API:
curl -X POST http://127.0.0.1:8080/v1/models/demo-model/unload
```

---

## Monitoring & Observability

### 1. Live Terminal Dashboard (TUI)

Launch a real-time terminal monitor to track model states, batching queues, and throughput:

```bash
./target/release/snooze monitor --endpoint http://127.0.0.1:8080
```

### 2. Prometheus Metrics

Scrape Prometheus exposition metrics at `/metrics`:

```bash
curl -s http://127.0.0.1:8080/metrics
```

Exposed metrics include:
- `snoozeml_requests_total{status="success|error|all"}`: Total inference requests received.
- `snoozeml_inference_latency_seconds_total`: Cumulative inference processing time in seconds.
- `snoozeml_batches_dispatched_total`: Total micro-batches dispatched by the batching engine.

### 3. Service Information & Health Probes

- **Home / Service Info**: `GET /` returns daemon information, version, and route links
- **Liveness Probe**: `GET /health/live` returns `{"status": "live"}`
- **Readiness Probe**: `GET /health/ready` returns `{"status": "ready"}`

---

## Testing & Quality Assurance

Run the test suite:

```bash
# Run all unit and integration tests
cargo test --all-targets

# Run strict clippy analysis
cargo clippy --all-targets -- -D warnings

# Check code formatting
cargo fmt --check
```

---

## Documentation

SnoozeML includes a comprehensive documentation portal powered by [Astro Starlight](https://starlight.astro.build/) alongside an auto-generated Python SDK reference powered by [Sphinx](https://www.sphinx-doc.org/):

```bash
# Start the interactive documentation server locally
npm --prefix docs run dev

# Or build the static site (Starlight + Sphinx) into docs/dist/
npm --prefix docs run build
sphinx-build -b html python/docs docs/dist/python
```

The documentation covers:
- **Getting Started**: Installation, 2-minute quickstart, and downloading models from Hugging Face.
- **Serving & Configuration**: `snooze serve` reference, `models.toml` schema, and hardware acceleration (CUDA/Metal/CPU).
- **Scaling Mechanics**: Deep dive into dynamic micro-batching along `Axis(0)`, scale-to-zero memory reclamation, replica autoscaling, and reactive OOM prevention.
- **Authentication & RBAC**: Setting up constant-time API keys and signed JWT tokens with 3-tier role permissions.
- **Operations & Monitoring**: Interactive TUI dashboard (`snooze monitor`), Prometheus metrics, and Kubernetes deployment guides.

For architectural design specifications:
- [Architectural Overview](docs/architecture.md): Component diagrams, batch collation algorithms, and concurrency model.
- [REST & Telemetry API Specification](docs/api.md): Complete HTTP endpoint schemas, request/response structures, and error codes.
- [Configuration & Deployment Guide](docs/configuration.md): Deployment patterns, CLI flags, and acceleration features.
- [gRPC Architecture & Schema Specification](design_docs/grpc_design.md): Protobuf communication schema, raw byte tensor ingestion, and integration design.
- [Python SDK & Monorepo Architecture](design_docs/python_sdk_design.md): Monorepo structure, decoupled Astro Starlight + Sphinx documentation strategy, client API, and ONNX conversion utilities.

---

## Official Python SDK (`snoozeml`)

SnoozeML includes an official Python SDK in [`python/`](python/) for synchronous & asynchronous HTTP/gRPC inference, model lifecycle management (`list`, `get`, `load`, `pin`, `unpin`, `evict`), structured logging with credential redaction, and automatic ONNX/XGBoost model export (`export_sklearn`, `export_torch`, `export_xgboost`):

```bash
pip install -e ./python
```

```python
import numpy as np
from snoozeml import SnoozeClient
from snoozeml.converters import export_sklearn

with SnoozeClient(endpoint="localhost:50051", use_grpc=True) as client:
    # 1. Inspect and pin models
    models = client.models.list()
    client.models.pin("demo-model")

    # 2. High-throughput raw-tensor gRPC inference
    x = np.ones((1, 4), dtype=np.float32)
    resp = client.infer("demo-model", inputs={"input": x})
    print(resp.get_numpy("output"))
```

### Interactive Examples & Notebooks

Explore end-to-end tutorial notebooks in [`examples/`](examples/):
- [**`01_sklearn_onnx.ipynb`**](examples/01_sklearn_onnx.ipynb): Train Scikit-learn Random Forest on Iris, export to ONNX, deploy to SnoozeML Docker, and run batched inference.
- [**`02_xgboost_tree.ipynb`**](examples/02_xgboost_tree.ipynb): Train XGBoost on Breast Cancer, export to `.xgb.json`, and serve with SnoozeML's native pure-Rust tree backend.
- [**`03_pipeline_regression_onnx.ipynb`**](examples/03_pipeline_regression_onnx.ipynb): Train a Scikit-learn `Pipeline` (StandardScaler + Ridge) on Diabetes and infer continuous targets.
- [**`04_chronos_bolt_custom_model.ipynb`**](examples/04_chronos_bolt_custom_model.ipynb): Deploy Amazon Chronos-Bolt foundation model with custom dependencies as a process-isolated Python worker for zero-shot probabilistic forecasting.

---

## Event Streaming & Message Brokers

SnoozeML supports asynchronous, non-blocking ingestion and response publishing from enterprise message brokers:
- **Apache Kafka**: Native pure-Rust driver powered by [`rskafka`](https://github.com/influxdata/rskafka) (zero C `librdkafka` dependencies).
- **RabbitMQ**: AMQP 0-9-1 driver with configurable prefetch QoS via [`lapin`](https://github.com/CleverCloud/lapin).
- **Redis Streams**: Scalable consumer group architecture (`XREADGROUP`, `XACK`, `XADD`) via [`redis-rs`](https://github.com/redis-rs/redis-rs).
- **In-Memory Mock**: Zero-infrastructure `mock:` driver for automated CI and integration testing.

### Key Architectural Invariants
1. **Fully Decoupled Multi-Model Transport**: Streams in SnoozeML are a first-class network transport alongside HTTP REST and gRPC. Streams are **not** restricted to a single model; all registered models can be invoked across any stream. Each incoming request specifies its target `model_id` (or uses an optional stream-level default).
2. **Fully Non-Blocking & Asynchronous**: Message consumers and publishers run on dedicated Tokio worker loops managed by `StreamSupervisor`. High stream throughput **never** blocks HTTP REST or gRPC endpoints.
3. **Integrated Dynamic Micro-Batching**: Consumed messages submit inference jobs directly to `DynamicBatcher`. Stream requests coalesce seamlessly with incoming HTTP and gRPC traffic along `Axis(0)`.
4. **Dynamic Output Routing (`reply_to`)**: Messages can dynamically route responses to custom topics or queues by specifying `reply_to` in the request header/envelope, falling back to the configured `output_topic`.
5. **Automatic Topic & Queue Creation**: With `auto_create_topics = true` (default), SnoozeML automatically provisions missing Kafka topics, RabbitMQ queues, and Redis consumer groups (`MKSTREAM`).
6. **Dead-Letter Queue (DLQ)**: Malformed JSON, Protobuf decode errors, missing models, or shape mismatches are automatically routed to a configured `dlq_topic` with detailed diagnostic context, acknowledging the broker to prevent poison-pill consumer loops.
7. **Wire Formats**: Supports both standard `json` and binary `protobuf` (compatible with `snoozeml.v1.ModelInferRequest` / `ModelInferResponse`).

### Configuration Example (`models.toml`)

```toml
# Universal Multi-Model Kafka Stream Binding (serves ALL models in the registry)
[[streams]]
id = "kafka-main-stream"
driver = "kafka"
endpoint = "127.0.0.1:9092"
input_topic = "inference.requests"
output_topic = "inference.responses"
dlq_topic = "inference.dlq"
format = "json"
auto_create_topics = true
concurrency = 32
partitions = 3
group_id = "snooze-inference-workers"

# Dedicated RabbitMQ Stream Binding (with optional default model_id)
[[streams]]
id = "rabbit-vision-stream"
model_id = "resnet50" # Optional fallback: acts as default if request omits model_id
driver = "rabbitmq"
endpoint = "amqp://guest:guest@127.0.0.1:5672/%2f"
input_topic = "images.incoming"
output_topic = "images.classified"
dlq_topic = "images.dlq"
format = "protobuf"
auto_create_topics = true
concurrency = 16

# Redis Streams Consumer Group Binding
[[streams]]
id = "redis-nlp-stream"
driver = "redis"
endpoint = "redis://127.0.0.1:6379"
input_topic = "documents.stream"
output_topic = "documents.embeddings"
dlq_topic = "documents.dlq"
format = "json"
auto_create_topics = true
group_id = "snooze-embedders"
consumer_id = "snooze-node-1"
concurrency = 16
```

# TODO
- [x] Implement a way to pin a model at startup/loading by config and never deallocate it unless explicitly told to do so by the user (via CLI/API).
- [x] Implement vertical scaling of the same model on a single device using multiple CPU cores or multiple GPUs
  - [x] Implement a proper routing strategy to distribute requests across replicas of the same model
- [x] Add gRPC interface support (high-performance SnoozeML-native Protobuf schema with raw tensor memory buffers, future Triton/KServe v2 compatibility)
- [x] Implement a plugin system for custom safetensor and user-defined model architectures (process-isolated Python worker backend with binary length-prefixed wire protocol)
- [x] Add documentation portal (Astro Starlight + Sphinx)
- [x] Add Api-key support and Role Base Authenthication (JWT tokens)
- [x] Add support for Kafka, RabbitMQ, Redis Streams and other message brokers.
- [x] Add support for streaming

- [x] **Python client**
  - [x] both model management and inference
  - [x] optional support for ONNX conversion under the hood (given a sklearn pipeline or torch model)

---

## Acknowledgements & Credits

SnoozeML is built upon and inspired by several outstanding open-source projects in the Rust and Machine Learning communities:

### Inference Backends & Model Runtimes
- **[`ort`](https://github.com/pykeio/ort)**: Safe, high-performance Rust bindings for ONNX Runtime maintained by [Pyke](https://github.com/pykeio). Powers SnoozeML's ONNX model execution and hardware acceleration providers (CPU, CUDA, TensorRT, CoreML).
- **[ONNX Runtime](https://github.com/microsoft/onnxruntime)**: Cross-platform, high-performance scoring and training engine developed by [Microsoft](https://github.com/microsoft).
- **[`hessboost`](https://github.com/brndnmtthws/hessboost)**: High-performance, pure-Rust, zero-FFI implementation of XGBoost and gradient boosted trees created by [Brandon Matthews](https://github.com/brndnmtthws). Powers SnoozeML's standalone Decision Tree and GBDT backend.
- **[XGBoost](https://github.com/dmlc/xgboost)**: Scalable, portable, and distributed gradient boosting library developed by the [DMLC](https://github.com/dmlc) community, defining the standard tree model architectures and JSON schemas supported by SnoozeML.
- **[`candle`](https://github.com/huggingface/candle)**: Minimalist ML framework for Rust developed by [Hugging Face](https://github.com/huggingface), enabling native deep learning and Safetensors model execution without Python or PyTorch runtimes.
- **[`safetensors`](https://github.com/huggingface/safetensors)** & **[`tokenizers`](https://github.com/huggingface/tokenizers)**: Fast, safe tensor serialization format and state-of-the-art tokenizers developed by [Hugging Face](https://github.com/huggingface).
- **[`hf-hub`](https://github.com/huggingface/hf-hub)**: Official Rust client library for interacting with the Hugging Face Hub.

### Core Systems & Tooling
- **[Tokio](https://github.com/tokio-rs/tokio)** & **[Axum](https://github.com/tokio-rs/axum)**: High-throughput asynchronous runtime and ergonomic HTTP framework.
- **[Rayon](https://github.com/rayon-rs/rayon)**: Work-stealing data parallelism library powering multi-core CPU tensor collations and tree evaluations.
- **[ndarray](https://github.com/rust-ndarray/ndarray)**: N-dimensional array and matrix primitives for scientific computing in Rust.
- **[Ratatui](https://github.com/ratatui/ratatui)**: Terminal UI library powering SnoozeML's interactive terminal dashboard monitor (`snooze monitor`).

---

## License

This project is licensed under the [Apache-2.0 License](LICENSE).


