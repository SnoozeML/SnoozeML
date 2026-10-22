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

## 4. One-Step Model Deployment & Overwrite

Export and upload a trained model directly from memory in a single call, or overwrite an existing model registration with `overwrite=True`:

```python
# One-step export + upload + optional eager load
info = client.deploy_sklearn(
    model=pipe,
    sample_input=X[:5],
    model_id="iris_rf",
    overwrite=True,
    load=True,
    pinned=True,
)

# Wait until a model finishes loading weights into memory
client.models.wait_for_ready("iris_rf", timeout=30.0, trigger_load=True)
```

---

## 5. Dataset Batch Chunking, Retries & PyTorch Tensors

Pass PyTorch tensors directly (automatically detached and moved to CPU), convert outputs via `.get_torch()`, chunk large datasets along `Axis(0)` with `infer_dataset()`, and automatically retry transient `503` responses (`ModelLoadingError` / `ResourceExhaustedError` respecting `Retry-After`):

```python
import torch
from snoozeml import SnoozeClient

client = SnoozeClient("http://localhost:8080", max_retries=3, retry_backoff=0.25)
client.wait_until_ready(timeout=15.0)

# Direct PyTorch tensor input & output conversion
x_tensor = torch.randn(4, 16, requires_grad=True)
resp = client.infer("mlp_net", {"input": x_tensor}, retries=3)
out_tensor = resp.get_torch("output", device="cpu")

# Chunk a large dataset along Axis(0) and concatenate responses automatically
large_batch = {"input": torch.randn(1024, 16)}
merged_resp = client.infer_dataset("mlp_net", large_batch, batch_size=64)
print(merged_resp.get_numpy("output").shape)  # (1024, ...)
```

---

## 6. JWT Minting & Verification (`snoozeml.auth`)

Mint and verify RFC 7519 HMAC JSON Web Tokens (`HS256`, `HS384`, `HS512`) matching SnoozeML's 3-tier RBAC hierarchy (`Role.READ_ONLY`, `Role.INFERENCE`, `Role.ADMIN`) without external dependencies:

```python
from snoozeml import Role, SnoozeClient, create_jwt_token, decode_jwt_token

token = create_jwt_token(
    secret="my-cluster-hmac-secret",
    subject="scoring-worker-01",
    role=Role.INFERENCE,
    expires_in_secs=3600,
    issuer="snooze-control-plane",
    algorithm="HS256",
)

# Verify and inspect claims locally
claims = decode_jwt_token(token, secret="my-cluster-hmac-secret", issuer="snooze-control-plane")

# Authenticate client requests using the minted JWT
authed_client = SnoozeClient("http://localhost:8080", token=token)
```

---

## 7. Event Streaming Envelopes & Clients (`snoozeml.streams`)

Build, serialize, and consume streaming inference envelopes (`JSON` or `Protobuf`) compatible with SnoozeML's Kafka, RabbitMQ, Redis Streams, and Mock stream drivers:

```python
from snoozeml import MockStreamClient, RedisStreamClient, StreamInferRequest, StreamWireFormat

# 1. Local in-memory stream pipeline testing
mock_stream = MockStreamClient(client=client, default_model_id="iris_rf")
resp = mock_stream.infer({"input": [[5.1, 3.5, 1.4, 0.2]]})

# 2. Live Redis Streams producer / consumer (XADD & XREAD)
redis_stream = RedisStreamClient(
    endpoint="redis://127.0.0.1:6379",
    input_topic="snooze.requests",
    output_topic="snooze.responses",
    default_model_id="iris_rf",
    format=StreamWireFormat.JSON,
)
result = redis_stream.infer({"input": [[5.1, 3.5, 1.4, 0.2]]}, timeout=5.0)
```

---

## 8. Programmatic Configuration, Stream Setup & Worker Scaffolding (`snoozeml.config`)

Generate `models.toml` / `models.json` files (including `[[streams]]` and `[auth]`), scaffold custom process-isolated Python worker models (`snooze_model.py` + `snooze_model.json`), and launch a local `snooze serve` daemon subprocess directly from Python:

```python
from snoozeml import GlobalDefaultsConfig, SnoozeConfig, SnoozeDaemon, scaffold_worker

# 1. Scaffold and smoke-test a custom Python worker directory
scaffold_worker("models/custom_forecaster", tensor_format="numpy", verify=True)

# 2. Build models.toml programmatically with models, streams, and RBAC auth
cfg = (
    SnoozeConfig(defaults=GlobalDefaultsConfig(max_loaded_models=8))
    .add_model("iris_rf", "iris_rf.onnx", pinned=True, max_batch_size=32)
    .add_stream(
        name="redis-ingest",
        broker="redis",
        endpoint="redis://127.0.0.1:6379",
        input_topic="snooze.requests",
        output_topic="snooze.responses",
        error_topic="snooze.dlq",
        model_id="iris_rf",
    )
    .add_api_key(role="admin", key_env="SNOOZE_ADMIN_KEY", name="ops-admin")
    .set_jwt(secret_env="SNOOZE_JWT_SECRET", algorithm="HS256")
)
cfg.save("models/models.toml")

# 3. Manage a local snooze daemon subprocess as a context manager
with SnoozeDaemon(models_dir="models", port=8080) as daemon_client:
    metrics = daemon_client.get_metrics()
    print(f"Success rate: {metrics.success_rate:.1%}, Avg latency: {metrics.average_latency_ms:.2f} ms")
```

---

## 9. Structured Logging & Credential Redaction

Configure human-readable key-value or JSON structured logs with automatic `Bearer` token and `X-API-Key` redaction:

```python
from snoozeml import configure_logging

configure_logging(level="INFO", format_json=False)
```

---

## 10. Interactive Notebooks & End-to-End Examples

For runnable tutorials covering Scikit-Learn pipelines, XGBoost classification, Chronos-Bolt probabilistic forecasting, and real-time Kafka event streaming, explore the [Interactive Examples Guide](/SnoozeML/getting-started/examples/) or inspect the `examples/` directory in the repository.