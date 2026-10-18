---
title: "Model Lifecycle Control"
description: "SnoozeML allows operators to dynamically inspect, warm up, and deallocate model instances at runtime through both dedicated CLI subcommands and REST endpoints."
---

SnoozeML allows operators to dynamically inspect, warm up, and deallocate model instances at runtime through both dedicated CLI subcommands and REST endpoints.

---

## 1. Inspecting Model Status

### Via CLI (`snooze status`)
Query a running daemon to display a formatted summary of all registered models:

```bash
snooze status --endpoint http://127.0.0.1:8080
```

### Via HTTP REST (`GET /v1/models`)
```bash
curl -s http://127.0.0.1:8080/v1/models | jq .
```

Response:
```json
[
  {
    "id": "all-minilm-l6-v2",
    "state": "ready",
    "onnx_path": "models/all-minilm-l6-v2.onnx",
    "description": "Sentence-Transformers 384-dimensional embedding model",
    "device": "cuda",
    "idle_timeout_secs": 300,
    "max_batch_size": 16,
    "max_batch_delay_ms": 10,
    "last_accessed_secs_ago": 4
  }
]
```

---

## 2. Pre-Warming Models (`load`)

Pre-warming loads model weights from disk into memory without requiring an inference payload. Use this before scheduled traffic spikes or during server startup scripts to eliminate user-facing cold starts.

### Via CLI
```bash
snooze load --model-id all-minilm-l6-v2 --endpoint http://127.0.0.1:8080
```

### Via HTTP REST
```bash
curl -X POST http://127.0.0.1:8080/v1/models/all-minilm-l6-v2/load \
  -H "X-API-Key: sk-admin-key"
```

Response (`200 OK`):
```json
{
  "status": "loaded",
  "model_id": "all-minilm-l6-v2"
}
```

---

## 3. Explicit Deallocation (`unload`)

Immediately purges a model's weights from RAM/VRAM, forcing it to enter the `Idle` state without waiting for the `idle_timeout` window to expire.

### Via CLI
```bash
snooze unload --model-id all-minilm-l6-v2 --endpoint http://127.0.0.1:8080
```

### Via HTTP REST
```bash
curl -X POST http://127.0.0.1:8080/v1/models/all-minilm-l6-v2/unload \
  -H "X-API-Key: sk-admin-key"
```

Response (`200 OK`):
```json
{
  "status": "unloaded",
  "model_id": "all-minilm-l6-v2"
}
```

---

## 4. Model Directory Operations

### Scanning Models on Disk (`snooze models scan`)
Inspects a directory on disk, checks configuration syntax, and validates model weights files:

```bash
snooze models scan ./models
```

Output:
```text
Discovered Models in './models':
- minilm-embeddings: ONNX (all-MiniLM-L6-v2/model.onnx) -> Device: auto, Batch: 32
- resnet50: ONNX (resnet50.onnx) -> Device: cuda, Batch: 64
- bge-small: Safetensors (bge-small-en-v1.5) -> Device: auto, Batch: 16
```

### Generating Configuration Stubs (`snooze init`)
Generates or updates `models.toml` in your models directory, automatically detecting weights files and creating boilerplate configurations:

```bash
snooze init ./models
```