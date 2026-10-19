<div align="center">

# SnoozeML

**Single-binary, scale-to-zero ML & deep learning inference daemon written in Rust.**

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-1.75%2B-orange.svg)](Cargo.toml)
[![Transports](https://img.shields.io/badge/Transports-HTTP_%7C_gRPC_%7C_Kafka_%7C_RabbitMQ_%7C_Redis-success.svg)](#event-streaming--docker-deployment)
[![Runtimes](https://img.shields.io/badge/Backends-ONNX_%7C_Safetensors_%7C_XGBoost_%7C_Python_Workers-blueviolet.svg)](#supported-model-formats)

[Why SnoozeML?](#why-snoozeml) •
[Quickstart](#quickstart-3-steps) •
[Supported Formats](#supported-model-formats) •
[Python SDK](#python-sdk--model-export) •
[Configuration](#cli-configuration--security) •
[Documentation](#documentation-examples--quality-assurance)

</div>

---

**SnoozeML** (`snooze`) is a bare-metal inference server engineered for high hardware utilization and zero idle memory waste. It combines **adaptive dynamic micro-batching** with **in-process scale-to-zero memory reclamation**, lazily loading model weights on demand and automatically evicting inactive models from GPU VRAM and host RAM.

> [!NOTE]
> SnoozeML is purpose-built for predictive machine learning, deep learning encoders, computer vision, audio, and tabular models. Generative Large Language Models (LLMs) requiring autoregressive token decoding and KV-cache management are explicitly **out of scope**.

## Why SnoozeML?

Most production inference stacks force a painful compromise: either wrestle with multi-gigabyte containers, Python dependency hell, and mandatory Kubernetes/Knative clusters just to scale idle models down, or pay for idle RAM/VRAM 24/7. 

SnoozeML delivers enterprise-grade serving primitives in a single, rootless Rust binary:

| Capability | Python Inference Frameworks | Heavyweight GPU Serving Suites | **SnoozeML (`snooze`)** |
| :--- | :--- | :--- | :--- |
| **Deployment Footprint** | Heavy `pip`/Conda envs + Docker | Multi-GB container images | **Single static binary (zero `sudo` / Docker required)** |
| **Scale-to-Zero** | Requires Kubernetes + Knative / KEDA | External orchestrator required | **Native in-process idle timer & LRU eviction** |
| **Idle Baseline RAM** | Hundreds of MBs (Python interpreter + workers) | High baseline footprint | **Near-zero (reclaims 100% of model RAM/VRAM on idle)** |
| **Execution Engine** | Python GIL workarounds / multi-process | C++ (complex build & plugin chains) | **Native Rust (`ort`, `candle`, `hessboost`) + isolated Python IPC** |
| **Event Brokers** | Requires external consumer glue code | Requires external microservices | **Built-in Kafka, RabbitMQ, and Redis Streams consumers** |

### Core Capabilities

- **In-Process Scale-to-Zero & LRU Eviction**: Unloads idle models after a configurable timeout, enforces `max_loaded_models` capacity budgets via LRU eviction, and recovers reactively from out-of-memory (OOM) spikes.
- **Adaptive Dynamic Micro-Batching**: Coalesces concurrent HTTP, gRPC, and broker stream requests along `Axis(0)` within a configurable latency budget (`--max-delay-ms`) and batch cap (`--max-batch-size`).
- **Model Pinning & Replica Autoscaling**: Pin mission-critical models in memory while dynamically scaling worker replicas under queue pressure (`LeastQueue`, `RoundRobin`, `Random`) with differential scale-down timeouts.
- **Multi-Runtime Execution**: Serve **ONNX** (`ort`), **Hugging Face Safetensors** (`candle`), **XGBoost / GBDT trees** (`hessboost` in pure Rust), and **custom Python models** (process-isolated workers that reclaim 100% of memory upon eviction) from one daemon.
- **Unified Transports & Zero-Trust Security**: Simultaneous **HTTP REST** (`:8080`), **native gRPC** (`:50051`), and **Event Stream** consumers with shared Rustls TLS, constant-time SHA-256 API keys, HMAC JWTs, and 3-tier RBAC (`Admin`, `Inference`, `ReadOnly`).
- **Built-in Observability**: Interactive terminal UI dashboard (`snooze monitor`), Prometheus metrics (`/metrics`), and health probes (`/health/live`, `/health/ready`).

---

## Quickstart (3 Steps)

Get a Hugging Face embedding model (`BAAI/bge-small-en-v1.5`) running end-to-end in under two minutes.

### 1. Install SnoozeML

**Linux & macOS (Zero `sudo` Required):**
```bash
curl -fsSL https://raw.githubusercontent.com/SnoozeML/SnoozeML/main/install.sh | sh

```

*Installs `snooze` and bundled ONNX Runtime into `~/.local/bin`, initializes `~/.local/share/snoozeml/models`, and registers a rootless background service (`-- --no-service` to skip, `-- --uninstall` to remove).*

**Windows (PowerShell):**

```powershell
irm https://raw.githubusercontent.com/SnoozeML/SnoozeML/main/install.ps1 | iex

```

**From Source:** (Rust 1.75+):

```bash
git clone https://github.com/SnoozeML/SnoozeML.git && cd SnoozeML
cargo build --release

```

| Profile | Command | Description |
| --- | --- | --- |
| **Default** | `cargo build --release` | ONNX, Candle Safetensors, XGBoost trees, Python workers, and streaming |
| **NVIDIA CUDA** | `cargo build --release --features cuda` | Enables CUDA execution provider |
| **NVIDIA TensorRT** | `cargo build --release --features tensorrt` | Enables TensorRT execution provider |
| **Apple CoreML** | `cargo build --release --features coreml` | Enables CoreML execution provider |
| **All Brokers** | `cargo build --release --features streams-all` | Enables Kafka, RabbitMQ, and Redis Streams drivers |
| **Trees Only** | `cargo build --release --no-default-features --features tree-backend` | Ultra-lightweight pure-Rust XGBoost build |

### 2. Download a Model

Download `BAAI/bge-small-en-v1.5` from Hugging Face Hub and auto-generate `./models/models.toml`:

```bash
snooze download BAAI/bge-small-en-v1.5 --include "*.safetensors,config.json,tokenizer.json"

```

### 3. Serve & Run Inference

Start the server (if not using the background service) and send a request:

```bash
snooze serve --models-dir ./models

```

```bash
curl -s -X POST http://127.0.0.1:8080/v1/models/bge-small-en-v1.5/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [{ "name": "input_ids", "shape": [1, 4], "data": [101.0, 7592.0, 2088.0, 102.0] }]
  }' | jq .
```

SnoozeML lazily loads `bge-small-en-v1.5` on the first request, runs the forward pass, and returns the 384-dimensional embedding tensor. After 300 seconds of inactivity, weights are automatically deallocated back to `idle`.

---

## Supported Model Formats

Drop any of the following into `--models-dir` for automatic discovery and registration:

| Format | Extensions / Layout | Engine | Best For |
| --- | --- | --- | --- |
| **ONNX** | `.onnx` | [`ort`](https://github.com/pykeio/ort) (CPU, CUDA, TensorRT, CoreML) | PyTorch, Scikit-Learn, TensorFlow, and general deep learning |
| **Safetensors** | `.safetensors` + `config.json` | [`candle`](https://github.com/huggingface/candle) | Native Rust execution of BERT, RoBERTa, JinaBERT, NomicBERT, DistilBERT, CLIP, ViT, DINOv2, MLPs |
| **XGBoost / GBDT** | `.xgb.json`, `.tree.json`, `.xgb` | [`hessboost`](https://github.com/brndnmtthws/hessboost) | Pure-Rust, zero-FFI tabular tree inference |
| **Python Worker** | Directory with `snooze_model.py` | Process-isolated binary IPC | Custom architectures (e.g., Chronos-Bolt) that cannot export to ONNX |

Create `models/<model-name>/snooze_model.py`. Workers run in isolated child processes so scale-to-zero unloads reclaim 100% of host RAM and GPU VRAM:

```python
from typing import Any, Dict
import numpy as np

class Model:
    def load(self, model_dir: str, device: str) -> None:
        self.device = device  # Load custom PyTorch / HF weights here

    def predict(self, inputs: Dict[str, np.ndarray]) -> Dict[str, Any]:
        return {"output": inputs["input"] * 2.0}

```

---

## Python SDK & Model Export

Install the Python SDK to export Scikit-Learn, PyTorch, or XGBoost models and run REST/gRPC inference:

```bash
pip install -e "./python[grpc,onnx,xgboost]"

```

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from snoozeml import SnoozeClient
from snoozeml.converters import export_sklearn

# 1. Train and export directly to ./models
X, y = load_iris(return_X_y=True)
clf = RandomForestClassifier(n_estimators=50, random_state=42).fit(X, y)
export_sklearn(clf, initial_types=X[:1].astype(np.float32), output_path="models/iris_rf.onnx")

# 2. Pin in memory and query over gRPC
with SnoozeClient(endpoint="127.0.0.1:50051", use_grpc=True) as client:
    client.models.pin("iris_rf")
    response = client.infer("iris_rf", inputs={"float_input": X[:4].astype(np.float32)})
    print("Predictions:", response.get_numpy("label"))

```

---

## CLI, Configuration & Security

Manage lifecycle states (`idle`, `loading`, `ready`, `busy`) and live telemetry from the CLI or REST API:

```bash
snooze status --endpoint http://127.0.0.1:8080    # Inspect states & queue depths
snooze load --model-id bge-small-en-v1.5 --pin    # Warm up & exempt from scale-to-zero
snooze unpin --model-id bge-small-en-v1.5         # Re-enable idle eviction
snooze unload --model-id bge-small-en-v1.5        # Immediately reclaim RAM/VRAM
snooze monitor --endpoint http://127.0.0.1:8080   # Launch interactive Ratatui TUI
```

| Endpoint | Method | Role | Description |
| --- | --- | --- | --- |
| `/v1/models` | `GET` | `ReadOnly` | List all models, residency states, and configuration |
| `/v1/models/{id}/infer` | `POST` | `Inference` | Execute dynamically micro-batched inference |
| `/v1/models/{id}/{load,unload,pin,unpin}` | `POST` | `Admin` | Manage model memory residency and pinning |
| `/health/live`, `/health/ready` | `GET` | *Public* | Liveness and readiness probes |
| `/metrics` | `GET` | *Public* / `Admin` | Prometheus metrics (`snoozeml_requests_total`, latency, batches) |

### Declarative Configuration (`models.toml`)

Generated via `snooze init ./models` or `snooze download`. Never store plaintext secrets in `models.toml`; map API keys and HMAC JWT secrets via environment variables (`SNOOZE_API_KEY`, `SNOOZE_API_KEYS`, `SNOOZE_JWT_SECRET`, or custom `key_env` mappings):

```toml
[defaults]
idle_timeout_secs = 300
max_batch_size = 16
max_batch_delay_ms = 10
device = "auto"
max_loaded_models = 8
autoscale_queue_threshold = 32
max_replicas = 4
routing_strategy = "least_queue" # "least_queue" | "round_robin" | "random"

[models."bge-small-en-v1.5"]
path = "bge-small-en-v1.5/model.safetensors"
pinned = true
max_batch_size = 64
max_batch_delay_ms = 5

[auth]
enabled = true
protect_metrics = true

[[auth.api_keys]]
name = "production-client"
key_env = "PROD_CLIENT_API_KEY"
role = "inference" # "readonly" | "inference" | "admin"

```

---

## Event Streaming & Docker Deployment

### Native Broker Consumers (`--features streams-all`)

Connect directly to **Apache Kafka** (`rskafka`), **RabbitMQ** (`lapin`), and **Redis Streams** (`redis`). Consumers feed straight into `DynamicBatcher` alongside HTTP/gRPC traffic with dynamic `reply_to` routing, **commit-after-infer** at-least-once delivery, and Dead-Letter Queue (`dlq_topic`) support.

```toml
[[streams]]
id = "kafka-embeddings"
driver = "kafka"
endpoint = "127.0.0.1:9092"
model_id = "bge-small-en-v1.5"
input_topic = "inference.requests"
output_topic = "inference.responses"
dlq_topic = "inference.dlq"
format = "json" # "json" | "protobuf"
concurrency = 32
group_id = "snooze-workers"

```

### Multi-Target Docker Images (`x86_64` & `aarch64`)

While SnoozeML runs natively on bare metal, hardened multi-arch containers are available for containerized environments:

```bash
docker build --target slim -t snooze-ml:slim .  # ~120 MB: Pure Rust, zero Python runtime
docker build --target cpu -t snooze-ml:cpu .    # ~280 MB: Includes lean Python worker venv
docker build --target cuda -t snooze-ml:cuda .  # Full GPU acceleration (ONNX Runtime CUDA + cuDNN)

# Or launch via Docker Compose / Buildx Bake
docker compose up snooze-slim

```

---

## Documentation, Examples & Quality Assurance

* **Interactive Examples**: Step-by-step Jupyter notebooks and custom worker templates (Scikit-Learn ONNX, pure-Rust XGBoost trees, Amazon Chronos-Bolt, and real-time Kafka event streaming) are located in the [`examples/`](examples/) directory.
* **Documentation Portal**: Run `npm --prefix docs run dev` to launch the local Astro Starlight developer portal, or build the complete documentation bundle with `npm --prefix docs run build` and `sphinx-build -b html python/docs docs/dist/python`.
* **Deep Dives**: Architectural and operational specifications are available in [`docs/architecture.md`](docs/architecture.md), [`docs/api.md`](docs/api.md), [`docs/configuration.md`](docs/configuration.md), and the [Production Deployment Guide](docs/src/content/docs/operations/production.md).
* **Strict QA**: All production Rust code enforces `#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]`. Verify with `cargo test --all-targets`, `cargo clippy --all-targets -- -D warnings`, and `pytest python/tests -v`.

---

## Roadmap
SnoozeML is actively evolving. The following initiatives represent planned architectural enhancements and future milestones:

* **gRPC Streaming**: Server-streaming (`InferStream`) and bidirectional channels for real-time telemetry, audio, and video ingestion.
* **WASM & Native Plugins**: Zero-copy in-process pre/post-processing via `cdylib` and WebAssembly (Wasmtime/WASI) without Python IPC overhead.
* **Edge & Embedded Targets**: Cross-compilation for RISC-V/ARM64, strict host memory ceilings, and WebGPU (`wgpu`) / Vulkan / Apple Metal acceleration.
* **KServe v2 Data Plane Parity**: Drop-in REST/gRPC protocol compatibility for existing cloud-native ML meshes.
* **Distributed Fleet Clustering**: Peer-to-peer state sync and cluster-wide scale-to-zero LRU placement.
* **In-Engine DAG Pipelines**: Multi-model chaining with zero serialization hops between stages.

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
