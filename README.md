<div align="center">

# SnoozeML

**High-performance, scale-to-zero ML & deep learning inference daemon written in Rust.**

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-1.75%2B-orange.svg)](Cargo.toml)
[![Transports](https://img.shields.io/badge/Transports-HTTP_%7C_gRPC_%7C_Kafka_%7C_RabbitMQ_%7C_Redis-success.svg)](#event-streaming--brokers)
[![Runtimes](https://img.shields.io/badge/Backends-ONNX_%7C_Safetensors_%7C_XGBoost_%7C_Python_Workers-blueviolet.svg)](#supported-model-formats)

[Quickstart](#quickstart-3-steps) •
[Python SDK](#python-sdk--model-export) •
[Supported Formats](#supported-model-formats) •
[Configuration](#configuration-modelstoml) •
[Event Streaming](#event-streaming--brokers) •
[Documentation](#documentation--examples)

</div>

---

**SnoozeML** (`snooze`) is a single-binary, bare-metal inference server engineered for high hardware utilization and zero idle memory waste. It combines **adaptive dynamic micro-batching** along `Axis(0)` with **scale-to-zero memory reclamation**, lazily loading model weights on demand and automatically evicting inactive models from GPU VRAM and host RAM.

> [!NOTE]
> SnoozeML is purpose-built for predictive machine learning, deep learning encoders, computer vision, audio, and tabular models. Generative Large Language Models (LLMs) requiring autoregressive token decoding and KV-cache management are explicitly **out of scope**.

### Why SnoozeML?

- **Adaptive Dynamic Micro-Batching**: Coalesces concurrent HTTP, gRPC, and streaming requests along `Axis(0)` within a configurable latency budget (`--max-delay-ms`) and batch cap (`--max-batch-size`).
- **Scale-to-Zero & LRU Eviction**: Unloads idle models after a configurable timeout, enforces `max_loaded_models` capacity budgets via LRU eviction, and recovers reactively from out-of-memory (OOM) spikes.
- **Model Pinning & Multi-Replica Autoscaling**: Pin mission-critical models in memory while dynamically scaling worker replicas under queue pressure (`LeastQueue`, `RoundRobin`, `Random`) with differential scale-down timeouts.
- **Multi-Runtime Execution**: Run **ONNX** (`ort`), **Hugging Face Safetensors** (`candle`), **XGBoost / GBDT trees** (`hessboost` in pure Rust), and **custom Python models** (process-isolated workers) from a single daemon.
- **Unified Transports & Security**: Simultaneous **HTTP REST** (`:8080`) and **native gRPC** (`:50051`) servers with shared Rustls TLS, constant-time SHA-256 API keys, HMAC JWTs, and 3-tier RBAC (`Admin`, `Inference`, `ReadOnly`).
- **Built-in Observability**: Interactive terminal UI dashboard (`snooze monitor`), Prometheus metrics (`/metrics`), and Kubernetes health probes (`/health/live`, `/health/ready`).

---

## Quickstart (3 Steps)

Get a real Hugging Face embedding model (`BAAI/bge-small-en-v1.5`) running end-to-end in under two minutes.

### 1. Install or Build SnoozeML

**One-Line Install (Linux & macOS — Zero `sudo` Required):**

```bash
curl -fsSL https://raw.githubusercontent.com/SnoozeML/SnoozeML/main/install.sh | sh
```

This downloads the prebuilt `snooze` binary and bundled ONNX Runtime libraries into `~/.local/share/snoozeml`, links `~/.local/bin/snooze`, initializes `~/.local/share/snoozeml/models`, and automatically starts a rootless background service (`systemd --user` on Linux, `LaunchAgent` on macOS). Pass `-- --no-service` to skip starting the background daemon, or `-- --uninstall` to remove it.

**Windows (PowerShell — Zero Admin Required):**

```powershell
irm https://raw.githubusercontent.com/SnoozeML/SnoozeML/main/install.ps1 | iex
```

**From Source (Rust 1.75+):**

```bash
git clone https://github.com/SnoozeML/SnoozeML.git
cd SnoozeML
cargo build --release
```

The standalone binary is compiled to `./target/release/snooze` (add it to your `PATH` or invoke directly).

<details>
<summary><strong>Optional Build Profiles (CUDA, TensorRT, CoreML, Minimal)</strong></summary>

| Profile | Command | Description |
| :--- | :--- | :--- |
| **Default** | `cargo build --release` | ONNX, Candle Safetensors, XGBoost trees, Python workers, and streaming |
| **NVIDIA CUDA** | `cargo build --release --features cuda` | Enables CUDA execution provider |
| **NVIDIA TensorRT** | `cargo build --release --features tensorrt` | Enables TensorRT execution provider |
| **Apple CoreML** | `cargo build --release --features coreml` | Enables CoreML execution provider |
| **All Brokers** | `cargo build --release --features streams-all` | Enables Kafka, RabbitMQ, and Redis Streams drivers |
| **Trees Only** | `cargo build --release --no-default-features --features tree-backend` | Ultra-lightweight pure-Rust XGBoost build |

</details>

### 2. Download a Real Model

Use the built-in Hugging Face Hub client to download `BAAI/bge-small-en-v1.5` directly into `./models` and auto-generate `./models/models.toml`:

```bash
snooze download BAAI/bge-small-en-v1.5 \
  --include "*.safetensors,config.json,tokenizer.json"
```

### 3. Start the Daemon & Run Inference

If you did not enable the background daemon during installation, start the server pointing to `./models` (serves HTTP on `:8080` and gRPC on `:50051`):

```bash
snooze serve --models-dir ./models
```

In another terminal, send an inference request with token IDs (`[CLS] hello world [SEP]`):

```bash
curl -s -X POST http://127.0.0.1:8080/v1/models/bge-small-en-v1.5/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "name": "input_ids",
        "shape": [1, 4],
        "data": [101.0, 7592.0, 2088.0, 102.0]
      }
    ]
  }' | jq .
```

SnoozeML lazily loads `bge-small-en-v1.5` on the first request, runs the forward pass, and returns the L2-normalized 384-dimensional embedding tensor (`"shape": [1, 384]`). After 300 seconds of inactivity, the weights are automatically deallocated back to `idle`.

---

## Python SDK & Model Export

Install the official Python SDK (`python/`) to export models directly from Scikit-Learn, PyTorch, or XGBoost and run high-throughput REST or gRPC inference:

```bash
pip install -e "./python[grpc,onnx,xgboost]"
```

### Train, Export & Infer in One Workflow

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from snoozeml import SnoozeClient
from snoozeml.converters import export_sklearn

# 1. Train a real Scikit-Learn model and export it directly into ./models
X, y = load_iris(return_X_y=True)
clf = RandomForestClassifier(n_estimators=50, random_state=42).fit(X, y)
export_sklearn(clf, initial_types=X[:1].astype(np.float32), output_path="models/iris_rf.onnx")

# 2. Restart or run `snooze serve --models-dir ./models`, then query over gRPC
with SnoozeClient(endpoint="127.0.0.1:50051", use_grpc=True) as client:
    client.models.pin("iris_rf")  # Keep resident in memory

    batch = X[:4].astype(np.float32)
    response = client.infer("iris_rf", inputs={"float_input": batch})
    print("Predictions:", response.get_numpy("label"))
```

---

## Supported Model Formats

Drop any of the following into `--models-dir` and SnoozeML will automatically discover and register them:

| Format | Extensions / Layout | Engine | Best For |
| :--- | :--- | :--- | :--- |
| **ONNX** | `.onnx` | [`ort`](https://github.com/pykeio/ort) (CPU, CUDA, TensorRT, CoreML) | PyTorch, Scikit-Learn, TensorFlow, and general deep learning models |
| **Safetensors** | `.safetensors` + `config.json` | [`candle`](https://github.com/huggingface/candle) | Native Rust execution of BERT, RoBERTa, JinaBERT, NomicBERT, DistilBERT, CLIP, ViT, DINOv2, and MLPs |
| **XGBoost / GBDT** | `.xgb.json`, `.tree.json`, `.xgb` | [`hessboost`](https://github.com/brndnmtthws/hessboost) | Pure-Rust, zero-FFI tabular tree inference (`python tools/train_tree.py --dataset iris --name iris_xgb`) |
| **Python Worker** | Directory with `snooze_model.py` | Process-isolated binary IPC | Custom architectures (e.g. Amazon Chronos-Bolt time-series forecasting) that cannot be exported to ONNX |

<details>
<summary><strong>Writing a Custom Python Worker (<code>snooze_model.py</code>)</strong></summary>

Create a folder inside `./models/` (e.g. `models/custom-forecaster/snooze_model.py`):

```python
from typing import Any, Dict
import numpy as np

class Model:
    def load(self, model_dir: str, device: str) -> None:
        self.device = device
        # Load custom PyTorch / HuggingFace weights here

    def predict(self, inputs: Dict[str, np.ndarray]) -> Dict[str, Any]:
        x = inputs["input"]
        return {"output": x * 2.0}
```

> [!NOTE]
> Custom Python workers run in isolated child processes so scale-to-zero unloads reclaim 100% of host RAM and GPU VRAM. Because spawning Python processes incurs a multi-second cold start, prefer native `.onnx`, `.safetensors`, or `.xgb.json` formats whenever possible.

</details>

---

## CLI & Lifecycle Operations

Manage loaded models, memory residency, and real-time telemetry from the CLI or REST API:

```bash
# Inspect registered models, states (idle / loading / ready / busy), and queue depths
snooze status --endpoint http://127.0.0.1:8080

# Warm up a model in RAM/VRAM ahead of traffic (pass --pin to exempt from scale-to-zero)
snooze load --model-id bge-small-en-v1.5 --pin

# Unpin or immediately unload a model to reclaim memory
snooze unpin --model-id bge-small-en-v1.5
snooze unload --model-id bge-small-en-v1.5

# Launch the interactive terminal dashboard (Ratatui TUI)
snooze monitor --endpoint http://127.0.0.1:8080
```

| Endpoint | Method | Role | Description |
| :--- | :--- | :--- | :--- |
| `/v1/models` | `GET` | `ReadOnly` | List all models, residency states, and configuration |
| `/v1/models/{id}/infer` | `POST` | `Inference` | Execute dynamically micro-batched inference |
| `/v1/models/{id}/{load,unload,pin,unpin}` | `POST` | `Admin` | Manage model memory residency and pinning |
| `/health/live`, `/health/ready` | `GET` | *Public* | Liveness and readiness probes |
| `/metrics` | `GET` | *Public* / `Admin` | Prometheus metrics (`snoozeml_requests_total`, latency, batches) |

---

## Configuration (`models.toml`)

Running `snooze init ./models` (or `snooze download ...`) generates a declarative `models.toml` in your models directory where you can fine-tune batching, replica autoscaling, RBAC, and per-model overrides:

```toml
[defaults]
idle_timeout_secs = 300
max_batch_size = 16
max_batch_delay_ms = 10
device = "auto"
max_loaded_models = 8
min_residency_secs = 10
autoscale_queue_threshold = 32
max_replicas = 4
routing_strategy = "least_queue" # "least_queue" | "round_robin" | "random"

[models."bge-small-en-v1.5"]
path = "bge-small-en-v1.5/model.safetensors"
description = "BAAI BGE small English embedding model"
pinned = true
max_batch_size = 64
max_batch_delay_ms = 5

[auth]
enabled = true
protect_metrics = true # Optional: require Admin role on /metrics

# Securely map keys to environment variable names (no plaintext secrets in config):
[[auth.api_keys]]
name = "production-client"
key_env = "PROD_CLIENT_API_KEY"
role = "inference" # "readonly" | "inference" | "admin"
```

> [!TIP]
> Credentials should **never** be stored in plaintext inside `models.toml`. Instead, pass `[auth] enabled = true` in `models.toml` and supply your credentials via environment variables:
> - `SNOOZE_API_KEY`: Single API key (defaults to `Admin` role, or `<role>:<key>`).
> - `SNOOZE_API_KEYS`: Comma-separated list of keys (e.g. `admin:sk-admin-key,inference:sk-client-key`).
> - `SNOOZE_API_KEY_ADMIN`, `SNOOZE_API_KEY_INFER`, `SNOOZE_API_KEY_READONLY`: Role-specific keys.
> - `SNOOZE_JWT_SECRET`: Shared HMAC secret for verifying JSON Web Tokens.

---

## Docker & Multi-Architecture Deployment

SnoozeML provides an optimized multi-target [`Dockerfile`](Dockerfile) supporting **AMD64 (`x86_64`)** and **ARM64 (`aarch64`)** with layer caching via [`cargo-chef`](https://github.com/LukeMathWalker/cargo-chef):

```bash
# 1. Slim: ultra-compact, zero Python runtime (~120 MB)
docker build --target slim -t snooze-ml:slim .

# 2. CPU: standard CPU inference with lean Python worker venv (~280 MB)
docker build --target cpu -t snooze-ml:cpu .

# 3. CUDA: full GPU acceleration via ONNX Runtime CUDA + cuDNN + lean Python
docker build --target cuda -t snooze-ml:cuda .
```

### Declarative Multi-Arch Builds & Docker Compose

```bash
# Compile multi-architecture images (linux/amd64 + linux/arm64) via Buildx Bake
docker buildx bake cpu
docker buildx bake slim
docker buildx bake all

# Or start services directly via Docker Compose
docker compose up snooze-cpu   # HTTP :8081, gRPC :50052
docker compose up snooze-slim  # HTTP :8080, gRPC :50051
docker compose up snooze-cuda  # HTTP :8082, gRPC :50053 (--gpus all)
```

See the [Production Deployment Guide](docs/src/content/docs/operations/production.md) for full container hardening and security guidelines.

---

## Event Streaming & Brokers

When compiled with `--features streams-all`, SnoozeML connects directly to **Apache Kafka** (`rskafka`), **RabbitMQ** (`lapin`), and **Redis Streams** (`redis`). Stream consumers run on isolated Tokio tasks, feed directly into `DynamicBatcher` alongside HTTP/gRPC traffic, support dynamic `reply_to` routing, and enforce **commit-after-infer** at-least-once delivery with Dead-Letter Queue (`dlq_topic`) routing.

```toml
[[streams]]
id = "kafka-embeddings"
driver = "kafka"
endpoint = "127.0.0.1:9092"
model_id = "bge-small-en-v1.5" # Optional default if message envelope omits model_id
input_topic = "inference.requests"
output_topic = "inference.responses"
dlq_topic = "inference.dlq"
format = "json"                # "json" | "protobuf"
auto_create_topics = true
concurrency = 32
group_id = "snooze-workers"
```

---

## Documentation & Examples

### Interactive Jupyter Notebooks (`examples/`)

- [`examples/01_sklearn_onnx.ipynb`](examples/01_sklearn_onnx.ipynb) — Train a Scikit-Learn Random Forest on Iris, export to ONNX, and run batched inference.
- [`examples/02_xgboost_tree.ipynb`](examples/02_xgboost_tree.ipynb) — Train an XGBoost classifier on Breast Cancer, export to `.xgb.json`, and serve via the native Rust tree backend.
- [`examples/03_pipeline_regression_onnx.ipynb`](examples/03_pipeline_regression_onnx.ipynb) — Export a Scikit-Learn regression `Pipeline` (`StandardScaler` + `Ridge`) to ONNX.
- [`examples/04_chronos_bolt_custom_model.ipynb`](examples/04_chronos_bolt_custom_model.ipynb) — Serve Amazon Chronos-Bolt for zero-shot time-series forecasting using a process-isolated Python worker.

### Building the Documentation Portal

SnoozeML's documentation portal combines [Astro Starlight](https://starlight.astro.build/) (`docs/`) and [Sphinx](https://www.sphinx-doc.org/) (`python/docs/`):

```bash
# Start the local Starlight development server
npm --prefix docs run dev

# Build the complete static documentation bundle into docs/dist/
npm --prefix docs run build
sphinx-build -b html python/docs docs/dist/python
```

Architectural deep dives are also available in [`docs/architecture.md`](docs/architecture.md), [`docs/api.md`](docs/api.md), [`docs/configuration.md`](docs/configuration.md), [`design_docs/grpc_design.md`](design_docs/grpc_design.md), and [`design_docs/python_sdk_design.md`](design_docs/python_sdk_design.md).

---

## Development & Quality Assurance

All production code in `src/` enforces `#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]` and `#![warn(missing_docs)]`. Run the verification suite before submitting pull requests:

```bash
# 1. Run all Rust unit and integration tests
cargo test --all-targets

# 2. Run strict Clippy static analysis
cargo clippy --all-targets -- -D warnings

# 3. Verify formatting
cargo fmt --check

# 4. Run the Python SDK & worker test suite
pytest python/tests -v
```
---

## Roadmap

SnoozeML is actively evolving. The following initiatives represent planned architectural enhancements and future milestones:

- **gRPC Streaming & Bidirectional Inference**:
  - Server-streaming (`InferStream`) for real-time sequence outputs, continuous sensor feeds, and time-series forecasting.
  - Client-streaming and duplex channels for high-frequency continuous telemetry, audio, and video frame ingestion.
- **Rust-Native Custom Runtime & WASM Plugins**:
  - In-process dynamic plugin execution via native shared libraries (`cdylib`) and WebAssembly (Wasmtime/WASI) for zero-copy, zero-IPC pre/post-processing and custom operator kernels without Python worker overhead.
- **Embedded & Edge Deployment Considerations**:
  - Footprint minimization for resource-constrained systems, robotics, and edge gateways: static zero-dependency single-binary builds, headless minimal feature profiles, cross-compilation for ARM64 and RISC-V targets, strict host memory ceilings, and acceleration via Vulkan, WebGPU (`wgpu`), and Apple Metal (MPS).
- **KServe v2 & Triton Protocol Parity**:
  - Implementation of standard KServe v2 / Triton Data Plane specifications (REST & gRPC) for seamless drop-in compatibility with cloud-native ML mesh ecosystems (Kubernetes, Kubeflow, Istio).
- **Distributed Fleet Coordination & Clustering**:
  - Peer-to-peer cluster state synchronization and distributed scale-to-zero placement across multi-node daemon topologies with cluster-wide LRU eviction.
- **Hardware Acceleration Expansion**:
  - Dedicated native execution providers for NVIDIA TensorRT, OpenVINO, and Apple CoreML across all supported backends.
- **DAG Pipeline Orchestration & Stateful Streaming**:
  - Native multi-model directed acyclic graph (DAG) pipelines for chaining preprocessors, feature extractors, and downstream classifiers directly in-engine with zero serialization hops between stages.

---

## Acknowledgements

SnoozeML builds upon outstanding projects in the Rust and Machine Learning ecosystems:
- **[`ort`](https://github.com/pykeio/ort)** & **[ONNX Runtime](https://github.com/microsoft/onnxruntime)** for cross-platform ONNX execution and hardware acceleration.
- **[`candle`](https://github.com/huggingface/candle)**, **[`safetensors`](https://github.com/huggingface/safetensors)**, **[`tokenizers`](https://github.com/huggingface/tokenizers)**, and **[`hf-hub`](https://github.com/huggingface/hf-hub)** by Hugging Face.
- **[`hessboost`](https://github.com/brndnmtthws/hessboost)** & **[XGBoost](https://github.com/dmlc/xgboost)** for pure-Rust gradient boosted decision tree inference.
- **[Tokio](https://github.com/tokio-rs/tokio)**, **[Axum](https://github.com/tokio-rs/axum)**, **[Tonic](https://github.com/hyperium/tonic)**, **[Rayon](https://github.com/rayon-rs/rayon)**, **[ndarray](https://github.com/rust-ndarray/ndarray)**, and **[Ratatui](https://github.com/ratatui/ratatui)**.

---

## License

Licensed under the [Apache-2.0 License](LICENSE).
