---
title: "Configuration Guide (`models.toml`)"
description: "SnoozeML uses a declarative TOML configuration file (typically named `models.toml` or `models.json`) to define model-specific execution requirements, scaling..."
---

SnoozeML uses a declarative TOML configuration file (typically named `models.toml` or `models.json`) to define model-specific execution requirements, scaling rules, hardware targets, and security policies.

---

## 1. File Structure Overview

A complete `models.toml` file consists of four primary sections:

1. **`[defaults]`**: Global fallback values applied to any model that does not explicitly override them.
2. **`[[models]]`**: An array of individual model definitions, specifying file paths, execution devices, batching settings, and replica autoscaling rules.
3. **`[auth]` & `[auth.jwt]`**: Global authentication switches and JSON Web Token verification parameters.
4. **`[[auth.api_keys]]`**: Static API key definitions and their assigned access roles.

---

## 2. Complete Example `models.toml`

```toml
# ==============================================================================
# 1. Global Defaults
# ==============================================================================
[defaults]
# Time in seconds before an idle model deallocates its weights from RAM/VRAM
idle_timeout_secs = 300

# Dynamic micro-batching defaults
max_batch_size = 16
max_batch_delay_ms = 10

# Hardware device preference: "auto", "gpu" / "cuda", "mlx", or "cpu"
device = "auto"

# Autoscaling and routing defaults
autoscale_queue_threshold = 8   # Queue backlog depth that triggers spinning up a replica
max_replicas = 4                # Max replicas per model (0 or omit for UNBOUNDED)
routing_strategy = "least-queue" # "least-queue", "round-robin", or "random"

# Preemptive memory capacity bounds (optional)
max_loaded_models = 4           # Maximum concurrently resident model instances in VRAM/RAM
min_residency_secs = 10         # Minimum residency time before LRU eviction can occur

# ==============================================================================
# 2. Model Definitions
# ==============================================================================

# Fast sentence embedding model using ONNX Runtime
[[models]]
id = "minilm-embeddings"
path = "all-MiniLM-L6-v2/model.onnx"
description = "Sentence-Transformers 384-dimensional embedding model"
device = "auto"
max_batch_size = 32
max_batch_delay_ms = 5
idle_timeout_secs = 300
autoscale_queue_threshold = 16
max_replicas = 2
routing_strategy = "least-queue"

# Heavy classification model requiring dedicated GPU acceleration
[[models]]
id = "resnet50"
path = "resnet50.onnx"
description = "ResNet-50 image classification network"
device = "cuda"
max_batch_size = 64
max_batch_delay_ms = 2
idle_timeout_secs = 600
# Unbounded replica autoscaling on high load:
max_replicas = 0
autoscale_queue_threshold = 32
routing_strategy = "least-queue"

# Pure Safetensors model loaded via Candle
[[models]]
id = "bge-small"
path = "bge-small-en-v1.5" # Directory containing model.safetensors and config.json
description = "BGE small embedding model loaded natively via Candle"
device = "auto"
max_batch_size = 16
max_batch_delay_ms = 10
idle_timeout_secs = 180

# ==============================================================================
# 3. Authentication & RBAC Configuration
# ==============================================================================
[auth]
enabled = true
protect_metrics = false # Set true to require Admin role on /metrics

[auth.jwt]
secret = "your-secure-jwt-hmac-shared-secret-at-least-32-chars"
algorithm = "HS256"      # HS256, HS384, or HS512
issuer = "snoozeml-auth" # Optional: validate "iss" claim
audience = "snoozeml"    # Optional: validate "aud" claim
leeway_secs = 60         # Clock drift allowance in seconds

[[auth.api_keys]]
key = "sk-admin-secret-987"
role = "admin"
name = "cluster-admin"

[[auth.api_keys]]
key = "sk-infer-client-654"
role = "inference"
name = "web-inference-worker"

[[auth.api_keys]]
key = "sk-ro-monitor-321"
role = "read_only"
name = "dashboard-crawler"
```

---

## 3. Configuration Fields Reference

### `[defaults]` Table

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `idle_timeout_secs` | integer | `300` | Inactivity window before a model's weights are purged from RAM/VRAM. |
| `max_batch_size` | integer | `16` | Capacity threshold triggering an immediate batch execution. |
| `max_batch_delay_ms` | integer | `10` | Maximum latency budget to wait for concurrent requests. |
| `device` | string | `"auto"` | Accelerator selection (`"auto"`, `"cuda"`, `"gpu"`, `"mlx"`, `"cpu"`). |
| `autoscale_queue_threshold` | integer | `max_batch_size` | Backlog depth on active replica triggering an additional replica allocation. |
| `max_replicas` | integer | `0` (unbounded) | Maximum concurrent replica copies per model. `0` means unbounded. |
| `routing_strategy` | string | `"least-queue"` | Load balancing strategy: `"least-queue"`, `"round-robin"`, `"random"`. |
| `max_loaded_models` | integer | `None` | Hard ceiling on loaded models across the entire daemon. |
| `min_residency_secs` | integer | `10` | Grace period preventing newly loaded models from being evicted. |

### `[[models]]` Array of Tables

| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `id` | string | **Yes** | Unique identifier used in API URLs (`/v1/models/{id}/infer`). |
| `path` | string | **Yes** | Relative path (from models dir) or absolute path to `.onnx` file or model folder. |
| `description` | string | No | Human-readable description shown in status queries and TUI. |
| `device` | string | No | Overrides global `device` preference for this model. |
| `max_batch_size` | integer | No | Overrides global micro-batch size. |
| `max_batch_delay_ms` | integer | No | Overrides global batch delay budget. |
| `idle_timeout_secs` | integer | No | Overrides global scale-to-zero idle timeout. |
| `autoscale_queue_threshold`| integer | No | Overrides queue threshold for replica autoscaling. |
| `max_replicas` | integer | No | Overrides maximum replica count (`0` for unbounded). |
| `routing_strategy` | string | No | Overrides replica dispatching strategy. |

---

## 4. Path Resolution & Directory Layout

Model paths in `[[models]]` are resolved relative to the directory containing `models.toml`:

```text
models/
├── models.toml
├── resnet50.onnx                 <-- path = "resnet50.onnx"
└── all-MiniLM-L6-v2/             <-- path = "all-MiniLM-L6-v2/model.onnx"
    ├── model.onnx
    └── config.json
```

Absolute paths (e.g. `/opt/shared-models/model.onnx`) are also supported.

---

## 5. Fault-Tolerant Parsing & Resilience

SnoozeML is engineered to never crash on configuration errors:
- **Missing File**: If `models.toml` does not exist, SnoozeML scans the directory and creates a sensible in-memory configuration automatically.
- **Syntax / Typing Errors**: If an invalid key or malformed value is encountered in a model block, SnoozeML logs a warning with line details and continues parsing the remaining models.
- **Unlisted Files**: If you drop a new `.onnx` file into your models folder while the server is running, you can reload or restart without modifying `models.toml`—SnoozeML registers unlisted models automatically with default settings.