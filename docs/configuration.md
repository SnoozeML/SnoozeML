# SnoozeML Configuration & Deployment Guide

## 1. Quick Start

SnoozeML is packaged as a standalone binary with batteries included.

### Starting the Server with Default Settings
```bash
snooze serve
```
By default, SnoozeML binds to `0.0.0.0:8080`. Pass `--models-dir <DIR>` to serve models, or pass `--demo` to enable the built-in demo model `demo-model` for testing.

### Sending a Test Inference Request
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

---

## 2. CLI Command Reference

### `snooze serve`
Options:
- `--host <HOST>`: Network interface address (default: `0.0.0.0`).
- `-p, --port <PORT>`: HTTP port (default: `8080`).
- `--models-dir <DIR>`: Directory path containing models (`.onnx`, `.safetensors`, or subfolders). Automatically reads `models.toml` / `models.json` if present.
- `--config <PATH>`: Explicit path to model configuration file (e.g. `models/models.toml`).
- `--idle-timeout <SECONDS>`: Default inactivity duration before deallocating weights (default: `300` s).
- `--max-batch-size <SIZE>`: Default maximum micro-batch size before triggering immediate execution (default: `16`).
- `--max-delay-ms <MS>`: Default maximum time budget before dispatching partial batch (default: `10` ms).
- `--provider <PROVIDER>`: Default accelerator preference (`auto`, `gpu`, `cuda`, `mlx`, `cpu`, `tensorrt`, `candle`). Defaults to `auto`.
- `--demo`: Enable mock demo model (`demo-model`) for testing without external weights (default: `false`).
- `--auth-enabled`: Explicitly enable authentication enforcement (default: `false`, auto-enabled if keys or JWT secret are passed).
- `--api-key <ROLE:KEY>`: Configure static API key with role (e.g. `--api-key admin:sk-admin-123`, `--api-key infer:sk-user-456`). Repeatable.
- `--jwt-secret <SECRET>`: HMAC secret for validating JSON Web Tokens (or env `SNOOZE_JWT_SECRET`).
- `--protect-metrics`: Require `Admin` role on Prometheus `/metrics` endpoint.
- `--streams-config <PATH>`: Explicit path to separate TOML file with `[[streams]]` message broker bindings.

### `snooze models init` / `snooze init`
Scans a model directory, discovers all model files and subdirectories (e.g. Hugging Face downloads), and generates or updates a `models.toml` configuration file with stubs and default settings:
```bash
# Initialize models.toml in ./models
snooze models init models

# Top-level shortcut
snooze init models
```

### `snooze models scan`
Inspects a model directory and active configuration, displaying discovered models, target devices (`auto`, `cuda`/`gpu`, `mlx`, `cpu`), batching configurations, and status:
```bash
snooze models scan models
```

### `snooze download` / `snooze models download`
Downloads a model repository directly from the Hugging Face Hub using the official `hf-hub` client into the configured models folder (default: `./models`, customizable via `SNOOZE_MODELS_DIR`):
```bash
# Download a Hugging Face model repository into ./models
snooze download sentence-transformers/all-MiniLM-L6-v2

# Specify a custom destination models directory and name
snooze download openai-community/gpt2 --models-dir ./models --name gpt2

# Filter for specific weight formats
snooze download sentence-transformers/all-MiniLM-L6-v2 --include "*.onnx,config.json"

# Skip automatic models.toml generation/updating
snooze download sentence-transformers/all-MiniLM-L6-v2 --no-init
```
Options:
- `<REPO>`: Hugging Face repository identifier (e.g. `openai-community/gpt2`).
- `-d, --models-dir <DIR>`: Root models directory (default: `./models`, env: `SNOOZE_MODELS_DIR`).
- `-n, --name, --folder <NAME>`: Custom destination subfolder name within the models directory.
- `-r, --revision <REV>`: Git revision, branch, or tag (defaults to `main`).
- `-i, --include <GLOBS>`: Comma-separated glob patterns to include (e.g. `*.onnx,*.safetensors`).
- `-e, --exclude <GLOBS>`: Comma-separated glob patterns to exclude.
- `--token <TOKEN>`: Hugging Face authentication token (env: `HF_TOKEN`).
- `--no-init`: Skip automatically initializing or updating `models.toml` after download completes (auto-init is enabled by default).

### `snooze monitor`
Displays a real-time terminal dashboard tracking model status, queue metrics, and active requests:
```bash
snooze monitor --endpoint http://127.0.0.1:8080
```

---

## 3. Model Directory & Per-Model Configuration

SnoozeML allows defining per-model execution requirements via `models.toml` (or `models.json`) located inside your models directory or passed via `--config`.

### Device Selection & "auto" Mode
SnoozeML defaults to `"auto"` mode for hardware device selection. In `"auto"` mode, SnoozeML dynamically resolves hardware acceleration in priority order:
1. **GPU**: Attempts NVIDIA CUDA / TensorRT GPU acceleration if available.
2. **MLX (on macOS)**: Attempts Apple Silicon hardware acceleration (CoreML for ONNX Runtime, Metal for Candle) when running on macOS.
3. **CPU**: Universal fallback if hardware accelerators are unavailable or fail to initialize.

You can also explicitly set `device = "cuda"` (or `"gpu"`), `device = "mlx"` (or `"metal"` / `"coreml"`), or `device = "cpu"`.

### Sample `models.toml`
```toml
# SnoozeML Model Directory Configuration
# Global defaults applied to all models unless overridden individually
[defaults]
idle_timeout_secs = 300
max_batch_size = 16
max_batch_delay_ms = 10
device = "auto"
# Autoscaling and routing defaults:
autoscale_queue_threshold = 4   # Spin up additional replica if queue exceeds this depth
max_replicas = 4                # Max concurrent instances per model (omit or set to 0 for UNBOUNDED)
routing_strategy = "least-queue" # Options: "least-queue", "round-robin", "random"

# Per-model configuration: Embedding model with automatic device selection
[[models]]
id = "all-minilm-l6-v2"
path = "all-minilm-l6-v2.onnx"
description = "Sentence-Transformers 384-dimensional embedding model"
device = "auto"
max_batch_size = 16
max_batch_delay_ms = 10
idle_timeout_secs = 300
autoscale_queue_threshold = 8
max_replicas = 2
routing_strategy = "least-queue"

# Per-model configuration: Heavy transformer on GPU with large micro-batches
[[models]]
id = "all-minilm-l6-v2-quant"
path = "all-minilm-l6-v2-quant.onnx"
description = "Quantized INT8 model accelerated on NVIDIA CUDA GPU"
device = "cuda" # or "gpu"
max_batch_size = 64
max_batch_delay_ms = 2
idle_timeout_secs = 600
autoscale_queue_threshold = 16
max_replicas = 4
routing_strategy = "least-queue"

# Subdirectory model (e.g. downloaded from Hugging Face)
[[models]]
id = "bert-base"
path = "bert-base-uncased/model.onnx"
description = "BERT base classification model"
device = "auto"
max_batch_size = 32

# Pure-Rust XGBoost Tree model
[[models]]
id = "xgboost-classifier"
path = "classifier.xgb.json"
description = "Pure-Rust XGBoost Gradient Boosted Tree model"
device = "tree"
max_batch_size = 64
max_batch_delay_ms = 5
idle_timeout_secs = 300

# Optional Authentication Configuration (keep secrets out of config files)
[auth]
enabled = true
protect_metrics = false

[auth.jwt]
secret_env = "SNOOZE_JWT_SECRET"
algorithm = "HS256"

# Declare named keys mapped to environment variable names:
[[auth.api_keys]]
name = "cluster-admin"
key_env = "SNOOZE_API_KEY_ADMIN"
role = "admin"

[[auth.api_keys]]
name = "web-client"
key_env = "SNOOZE_API_KEY_INFER"
role = "inference"
```

### Autoscaling & Replica Limits Reference

| Setting | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `autoscale_queue_threshold` | integer | `max_batch_size` | Number of queued/active requests on a replica that triggers spinning up an additional copy. |
| `max_replicas` | integer | **Unbounded** (`None` or `0`) | Maximum concurrent replica copies allowed for a model. If omitted or set to `0`, scaling is **unbounded** (scales until GPU/host memory is exhausted or `max_loaded_models` is reached). |
| `routing_strategy` | string | `"least-queue"` | Strategy for dispatching requests among identical replicas: `"least-queue"` (lowest queue/active count), `"round-robin"`, or `"random"`. |

> [!TIP]
> **Overriding Global Limits with Unbounded Scaling**:
> If you configure a global ceiling such as `[defaults] max_replicas = 4`, you can make a specific high-priority model unbounded by specifying `max_replicas = 0` in its `[[models]]` block.

### Fault-Tolerant Design
- **Missing Configuration File**: If no `models.toml` is present, SnoozeML automatically discovers all `.onnx` and `.safetensors` files in the directory and subdirectories, registering them with sensible defaults and raising a notification.
- **Unlisted Models on Disk**: If new models exist on disk that are not listed in `models.toml`, SnoozeML discovers and registers them automatically with default settings.
- **Missing Weight Files**: If a model in `models.toml` points to a non-existent file, SnoozeML logs a warning and proceeds without crashing.
- **Graceful Parsing**: If a line or field in `models.toml` contains syntax errors or unrecognized keys, SnoozeML logs a warning and continues parsing valid entries.

---

## 4. Hardware Acceleration & Compilation Features

SnoozeML supports modular compilation profiles:

| Feature Flag | Description |
| :--- | :--- |
| `default` (`ort-backend`, `candle-backend`) | Standard ONNX Runtime and Candle/Safetensors backends enabled out-of-the-box |
| `ort-backend` | ONNX Runtime backend executing `.onnx` models |
| `candle-backend` | Hugging Face Candle backend executing `.safetensors` models |
| `cuda` | Enables NVIDIA CUDA execution provider via `ort/cuda` |
| `tensorrt` | Enables NVIDIA TensorRT execution provider via `ort/tensorrt` |
| `coreml` | Enables Apple CoreML execution provider via `ort/coreml` on macOS |

### Building with Safetensors Support
Candle and Safetensors support (`candle-backend`) is enabled by default in standard SnoozeML builds alongside ONNX Runtime (`ort-backend`). Both `.onnx` and `.safetensors` model weights can be served concurrently out-of-the-box:
```bash
cargo build --release
```

### Safetensors Model Conventions
When loading `.safetensors` files, SnoozeML automatically discovers linear and multi-layer perceptron (MLP) architectures:
- **Single Linear**: Weights named `"weight"` or `"linear.weight"` (with optional `"bias"` or `"linear.bias"`).
- **Sequential MLP**: Layers ordered by name (e.g. `0.weight`, `1.weight` or `layers.0.weight`), applying intermediate ReLU activations.
- **Companion Configuration**: Place a `{model_stem}.json` or `config.json` alongside the `.safetensors` file to explicitly configure input/output node names, custom layer keys, and activation types (`relu`, `gelu`, `none`).

---

## 5. Event Streaming & Message Broker Configuration

SnoozeML connects to enterprise streaming platforms via `[[streams]]` tables inside `models.toml` or via `--streams-config <PATH>`.

### Stream Binding Schema Reference

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `id` | string | *Required* | Unique identifier for the stream binding (e.g. `"fraud-stream"`). |
| `model_id` | string | *Optional* (`None`) | Default model identifier. If omitted (`None`), the stream dynamically routes to the `model_id` specified in each incoming request, serving all registered models. |
| `driver` | string | *Required* | Stream driver implementation: `"kafka"`, `"rabbitmq"`, `"redis"`, or `"mock"`. |
| `endpoint` | string | *Required* | Broker connection string (e.g. `"127.0.0.1:9092"`, `"amqp://localhost:5672"`, `"redis://127.0.0.1:6379"`). |
| `input_topic` | string | *Required* | Topic, queue, or stream key from which requests are consumed. |
| `output_topic` | string | *Optional* | Topic, queue, or stream key to which successful inference results are published. |
| `dlq_topic` | string | *Optional* | Dead-letter queue/topic for malformed payloads, unparseable tensors, or runtime errors. |
| `format` | string | `"json"` | Message wire codec: `"json"` or `"protobuf"` (compatible with `snoozeml.v1.ModelInferRequest`). |
| `auto_create_topics`| boolean | `true` | Automatically declare missing Kafka topics, RabbitMQ queues, and Redis consumer groups. |
| `concurrency` | integer | `16` | Maximum concurrent asynchronous inference requests permitted for this stream. |
| `partitions` | integer | `1` | Topic partition count when automatically creating Kafka topics. |
| `group_id` | string | *Optional* | Consumer group identifier for Kafka (`group_id`) and Redis Streams (`group_name`). |
| `consumer_id` | string | *Optional* | Unique consumer instance name within a consumer group (Redis Streams). |
| `prefetch_count` | integer | *Optional* | RabbitMQ basic QoS channel prefetch limit. |

