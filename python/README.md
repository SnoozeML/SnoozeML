# `snoozeml` — Official Python SDK for SnoozeML

`snoozeml` is the official Python client library and model export toolkit for the **SnoozeML** bare-metal Rust inference daemon.

It provides:
1. **Dual-Transport Inference (`SnoozeClient` & `AsyncSnoozeClient`)**: Execute unary tensor inference over HTTP REST (JSON) or high-throughput native gRPC (raw little-endian `FP32` byte buffers) with automatic NumPy `ndarray` marshaling.
2. **Model Lifecycle & Scale-to-Zero Control (`client.models`)**: List registered models, inspect live states (`idle`, `loading`, `ready`, `busy`), eagerly preload weights (`load`), pin critical models in memory (`pin` / `unpin`), and explicitly evict/unload models (`unload` / `evict`).
3. **Built-in Model Converters (`snoozeml.converters`)**: Export scikit-learn estimators & `Pipeline`s (`export_sklearn`), PyTorch `nn.Module`s (`export_torch`), and XGBoost boosters (`export_xgboost`) directly into SnoozeML-ready `.onnx` and `.xgb.json` artifacts.
4. **Structured Logging & Credential Redaction (`snoozeml.logging`)**: Built-in `RedactingFilter` and `StructuredFormatter` ensure API keys and JWT Bearer tokens are never leaked in logs or `__repr__` outputs.

---

## Installation

Install `snoozeml` directly from PyPI:

```bash
# Lightweight HTTP REST client (zero compiled C-extensions, pure Python)
pip install snoozeml

# High-performance native gRPC inference + NumPy array support
pip install "snoozeml[grpc]"

# Complete toolkit: gRPC client, custom worker, and all model converters
pip install "snoozeml[full]"
```

### Modular Dependency Extras

| Extra | Command | Included Packages | Purpose |
| :--- | :--- | :--- | :--- |
| *(None / Base)* | `pip install snoozeml` | `requests` | Ultra-lightweight HTTP client for model lifecycle management (`load`, `unload`, `pin`, `unpin`), status polling, health probes, and metrics. Zero binary C-extensions. |
| `[grpc]` | `pip install "snoozeml[grpc]"` | `grpcio`, `protobuf`, `numpy` | Native binary gRPC transport with automatic NumPy array serialization and low-latency inference. |
| `[converters]` | `pip install "snoozeml[converters]"` | `numpy`, `scikit-learn`, `onnx`, `skl2onnx`, `xgboost[-cpu]` | Offline model export toolkit (`export_sklearn`, `export_xgboost`) targeting ONNX and pure-Rust tree formats. |
| `[full]` | `pip install "snoozeml[full]"` | `grpc`, `converters`, `worker` | Full-featured installation for client, gRPC, custom workers, and model converters with zero CUDA bloat. |
| `[onnx]` | `pip install "snoozeml[onnx]"` | `numpy`, `scikit-learn`, `onnx`, `skl2onnx` | Scikit-learn estimator and pipeline export to ONNX runtime format. |
| `[xgboost]` | `pip install "snoozeml[xgboost]"` | `numpy`, `xgboost-cpu` (Linux) / `xgboost` (macOS/Win) | XGBoost model export to `.xgb.json` for pure-Rust `hessboost` tree execution. Uses `xgboost-cpu` on Linux to avoid pulling heavy NVIDIA CUDA packages. |
| `[torch]` | `pip install "snoozeml[torch]"` | `torch` | PyTorch `nn.Module` export to ONNX. *(Note: If PyTorch is already installed in your training environment, `snoozeml` automatically duck-types PyTorch tensors without needing this extra).* |
| `[worker]` | `pip install "snoozeml[worker]"` | `numpy` | Minimal dependency profile for process-isolated custom Python worker models (`snooze_model.py`). |
| `[dev]` | `pip install "snoozeml[dev]"` | `pytest`, `mypy`, `ruff`, `sphinx` | Development, linting, testing, and documentation building. |

> [!NOTE]
> For editable local development from the repository root:
> ```bash
> pip install -e "./python[full,dev]"
> ```

---

## Quickstart

### 1. Model Lifecycle Management

```python
from snoozeml import SnoozeClient

client = SnoozeClient(endpoint="http://localhost:8080", api_key="sk-admin-key")

# List all registered models
models = client.models.list()
for m in models:
    print(f"{m.id}: state={m.state.value}, pinned={m.pinned}, loaded={m.is_loaded}")

# Inspect a specific model
info = client.models.get("demo-model")
print(f"Path: {info.onnx_path}, Idle Timeout: {info.idle_timeout_secs}s")

# Pin model in memory (exempts from scale-to-zero and LRU eviction)
client.models.pin("demo-model")

# Unpin and explicitly evict (scale-to-zero)
client.models.unpin("demo-model")
client.models.evict("demo-model")
```

### 2. Synchronous & Asynchronous Inference (HTTP & gRPC)

```python
import numpy as np
from snoozeml import SnoozeClient, AsyncSnoozeClient

# High-throughput gRPC inference
with SnoozeClient(endpoint="localhost:50051", use_grpc=True) as client:
    x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    response = client.infer(model_id="demo-model", inputs={"input": x})

    output = response.get_numpy("output")
    print(f"Output shape: {output.shape}, Server latency: {response.duration_ms:.2f} ms")
```

Asynchronous usage with `AsyncSnoozeClient`:

```python
import asyncio
import numpy as np
from snoozeml import AsyncSnoozeClient

async def main():
    async with AsyncSnoozeClient(endpoint="localhost:50051", use_grpc=True) as client:
        x = np.ones((2, 4), dtype=np.float32)
        resp = await client.infer("demo-model", inputs={"input": x})
        print(resp["output"])

asyncio.run(main())
```

### 3. Exporting scikit-learn, PyTorch & XGBoost Models

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from snoozeml.converters import export_sklearn, export_torch, export_xgboost

# 1. Export a scikit-learn Pipeline to ONNX
X = np.random.randn(100, 4).astype(np.float32)
y = (X[:, 0] + X[:, 1] > 0).astype(int)
pipe = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=10, random_state=42))
pipe.fit(X, y)

onnx_path = export_sklearn(
    model=pipe,
    initial_types=[("float_input", (None, 4))],
    output_path="models/rf_pipeline.onnx",
)

# 2. Export a PyTorch nn.Module to ONNX
import torch
import torch.nn as nn

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2)).eval()
export_torch(
    model=net,
    sample_input=torch.randn(1, 4),
    output_path="models/mlp_torch.onnx",
)
```

### 4. Structured Logging

```python
from snoozeml import configure_logging

# Enable human-readable or JSON structured logs (with automatic secret redaction)
configure_logging(level="DEBUG", format_json=False)
```

---

## Interactive Examples & Jupyter Notebooks

Check out the interactive tutorial notebooks under [`examples/`](../examples):
* [**`01_sklearn_onnx.ipynb`**](../examples/01_sklearn_onnx.ipynb): Train a Scikit-learn Random Forest classifier on Iris, export to ONNX, deploy to SnoozeML Docker, and run batched inference.
* [**`02_xgboost_tree.ipynb`**](../examples/02_xgboost_tree.ipynb): Train an XGBoost classifier on Breast Cancer, export to native `.xgb.json`, and run inference via SnoozeML's pure-Rust tree backend.
* [**`03_pipeline_regression_onnx.ipynb`**](../examples/03_pipeline_regression_onnx.ipynb): Train a Scikit-learn `Pipeline` (StandardScaler + Ridge) on Diabetes and infer continuous targets.

