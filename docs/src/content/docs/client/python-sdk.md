---
title: "Official Python SDK (`snoozeml`)"
description: "SnoozeML includes an official Python package (`snoozeml`, located under `python/`) providing synchronous and asynchronous clients (`SnoozeClient`, `AsyncSnoo..."
---

SnoozeML includes an official Python package (`snoozeml`, located under `python/`) providing synchronous and asynchronous clients (`SnoozeClient`, `AsyncSnoozeClient`), model lifecycle controls (`client.models`), structured logging with automatic secret redaction, and built-in ONNX/XGBoost model converters (`snoozeml.converters`).

---

## Installation

Install `snoozeml` from the repository root:

```bash
# Core HTTP + gRPC client & native ONNX/XGBoost converters
pip install ./python

# Optional ML framework converter extras
pip install "./python[converters]"
```

---

## 1. Model Management & Scale-to-Zero Control

Query model telemetry, eagerly load weights, pin critical models in memory, or trigger explicit scale-to-zero evictions via `client.models`:

```python
from snoozeml import SnoozeClient

client = SnoozeClient(
    http_endpoint="http://localhost:8080",
    grpc_endpoint="localhost:50051",
    api_key="sk-admin-key",  # or set SNOOZE_API_KEY env var
)

# List all registered models
models = client.models.list()
for m in models:
    print(f"{m.id}: state={m.state.value}, loaded={m.is_loaded}, pinned={m.pinned}")

# Inspect a specific model
info = client.models.get("demo-model")
print(f"Path: {info.onnx_path}, Last accessed: {info.last_accessed_secs_ago}s ago")

# Pin model in memory (prevents scale-to-zero sweeping and LRU eviction)
client.models.pin("demo-model")

# Unpin and explicitly evict/unload from RAM/VRAM
client.models.unpin("demo-model")
client.models.evict("demo-model")
```

### Direct Model Upload (Files & Raw In-Memory Bytes)

Upload model weights directly to SnoozeML without volume mounting or restarting:

```python
# Option A: Upload from local file path
client.upload_model(
    file="models/iris_classifier.onnx",
    model_id="iris_classifier",
    max_batch_size=32,
    idle_timeout_secs=120,
    pinned=True,  # eagerly loaded & exempt from scale-to-zero
)

# Option B: Upload raw in-memory bytes without saving to disk
# (e.g. from an in-memory XGBoost booster, BytesIO, or serialized ONNX graph)
raw_bytes = booster.save_raw(raw_format="json")
client.upload_model(
    file=raw_bytes,
    filename="credit_risk.xgb.json",  # Required when uploading bytes to identify backend format
    model_id="credit_risk",
    max_batch_size=16,
)

# Option C: Dynamically refresh models directory on server without process restart
refreshed = client.refresh_models()
print(f"Refreshed {len(refreshed)} models on server")
```


---

## 2. Synchronous & Asynchronous Tensor Inference

`SnoozeClient` and `AsyncSnoozeClient` support both HTTP REST (`use_grpc=False`) and high-throughput native gRPC (`use_grpc=True`) with raw little-endian `FP32` byte transport:

```python
import numpy as np
from snoozeml import SnoozeClient

with SnoozeClient(endpoint="localhost:50051", use_grpc=True) as client:
    input_data = np.random.randn(2, 10).astype(np.float32)

    response = client.infer(
        model_id="demo-model",
        inputs={"input": input_data},
    )

    output = response.get_numpy("output")
    print(f"Output shape: {output.shape}, Server latency: {response.duration_ms:.2f} ms")
```

### Asynchronous Concurrent Micro-Batching

```python
import asyncio
import numpy as np
from snoozeml import AsyncSnoozeClient

async def main():
    async with AsyncSnoozeClient(endpoint="localhost:50051", use_grpc=True) as client:
        tasks = [
            client.infer("demo-model", {"input": np.ones((1, 10), dtype=np.float32)})
            for _ in range(8)
        ]
        responses = await asyncio.gather(*tasks)
        print(f"Completed {len(responses)} coalesced requests")

asyncio.run(main())
```

---

## 3. Built-in Model Converters (`snoozeml.converters`)

Export `scikit-learn` pipelines/estimators, `PyTorch` modules, and `XGBoost` boosters directly into SnoozeML's `models/` directory:

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from snoozeml.converters import export_sklearn, export_torch, export_xgboost

# 1. Scikit-learn Pipeline -> ONNX
X = np.random.randn(100, 4).astype(np.float32)
y = (X[:, 0] > 0).astype(int)
pipe = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=10, random_state=42))
pipe.fit(X, y)

export_sklearn(
    model=pipe,
    initial_types=[("float_input", (None, 4))],
    output_path="models/rf_classifier.onnx",
)

# 2. PyTorch nn.Module -> ONNX
import torch
import torch.nn as nn

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2)).eval()
export_torch(
    model=net,
    sample_input=torch.randn(1, 4),
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    output_path="models/torch_mlp.onnx",
)

# 3. XGBoost -> Pure-Rust Tree JSON (.xgb.json)
import xgboost as xgb

clf = xgb.XGBClassifier(n_estimators=20, max_depth=3).fit(X, y)
export_xgboost(clf, output_path="models/xgb_classifier.xgb.json")
```

---

## 4. Structured Logging & Credential Redaction

Configure human-readable key-value or JSON structured logs with automatic `Bearer` token and `X-API-Key` redaction:

```python
from snoozeml import configure_logging

configure_logging(level="INFO", format_json=False)
```