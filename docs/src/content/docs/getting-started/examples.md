---
title: "Interactive Examples & Notebooks"
description: "Explore end-to-end interactive Jupyter notebooks and custom worker templates for SnoozeML."
---

SnoozeML provides interactive Jupyter notebooks in the [`examples/`](https://github.com/SnoozeML/SnoozeML/tree/main/examples) directory demonstrating how to train models using standard machine learning libraries, export them to SnoozeML-compatible formats, deploy them into a running daemon, and run low-latency micro-batched inference using the official `snoozeml` Python SDK.

---

## Available Notebooks

| Notebook | Model Type | Dataset | Target Backend | Task |
| :--- | :--- | :--- | :--- | :--- |
| **`01_sklearn_onnx.ipynb`** | Scikit-learn `RandomForestClassifier` | Iris | ONNX Runtime (`ort`) | Multiclass Classification |
| **`02_xgboost_tree.ipynb`** | XGBoost `XGBClassifier` | Breast Cancer | Pure-Rust Trees (`hessboost`) | Binary Classification |
| **`03_pipeline_regression_onnx.ipynb`** | Scikit-learn `Pipeline` (`StandardScaler` + `Ridge`) | Diabetes | ONNX Runtime (`ort`) | Continuous Regression |
| **`04_chronos_bolt_custom_model.ipynb`** | Amazon Chronos-Bolt Foundation Model | Synthetic / Time Series | Custom Worker (`worker-backend`) | Zero-Shot Probabilistic Forecasting |
| **`05_streaming_xgboost.ipynb`** | XGBoost `XGBClassifier` | Breast Cancer | Pure-Rust Trees (`hessboost`) | Real-Time Event Streaming (Redpanda / Kafka) |

---

## Prerequisites & Setup

### 1. Install Python SDK & Dependencies

Install `snoozeml` from PyPI with all runtime extras (gRPC, ONNX, XGBoost, and converters) along with Jupyter:

```bash
# Install SnoozeML Python SDK from PyPI with all extras (gRPC, ONNX, XGBoost, converters)
pip install "snoozeml[full]"
pip install jupyter

# (Optional) For editable development from a cloned repository:
# pip install -e "./python[full]"

# Optional: For Chronos-Bolt custom worker (Notebook 04):
pip install -r examples/chronos_bolt/requirements.txt

# Optional: For event streaming with Redpanda / Kafka (Notebook 05):
pip install kafka-python-ng
```

### 2. Launch the SnoozeML Daemon

Start SnoozeML using Docker Compose (which mounts the `./models` directory directly):

```bash
docker compose up -d
```

Verify that the daemon is live:

```bash
curl http://localhost:8080/health/live
# {"status":"live"}
```

Alternatively, run the native binary directly on bare metal:

```bash
snooze serve --models-dir ./models
```

### 3. Launch Jupyter

```bash
jupyter notebook examples/
```

---

## Key Deployment Patterns Demonstrated

### Direct HTTP Upload (No Volume Mounting Required)

Upload model weights directly over HTTP REST without touching local filesystem volumes or restarting containers:

```python
from snoozeml import SnoozeClient

client = SnoozeClient("http://localhost:8080")

# Upload an ONNX model file
info = client.upload_model(
    file="iris_classifier.onnx",
    model_id="iris_classifier",
    pinned=False,               # Keep unpinned to allow scale-to-zero memory reclamation
    max_batch_size=16,          # Max micro-batch size for Axis(0) dynamic collation
    max_batch_delay_ms=10,      # 10ms micro-batch latency accumulation budget
    idle_timeout_secs=300,      # Scale-to-zero memory unload after 5 min of inactivity
)

print(f"Registered model '{info.id}' with state '{info.state.value}'")
```

### Dynamic Model Discovery Refresh

When dropping new model weights into `./models` on disk or through container volume mounts, trigger runtime registration instantly via HTTP without daemon restarts:

```python
refreshed = client.refresh_models()
print(f"Discovered {len(refreshed)} active models")
```

### Process-Isolated Custom Python Workers

For specialized deep learning architectures (such as [Amazon Chronos-Bolt](https://github.com/amazon-science/chronos-forecasting)) that cannot be directly exported to standard ONNX graphs, SnoozeML supports custom Python workers:

1. Place `snooze_model.py` and optional `snooze_model.json` inside a model directory (e.g., `examples/chronos_bolt/`).
2. SnoozeML spawns the model inside an isolated child process via binary IPC framing, ensuring complete scale-to-zero RAM and VRAM reclamation upon idle timeout or LRU eviction.
3. Walkthrough: See `04_chronos_bolt_custom_model.ipynb` and `examples/chronos_bolt/README.md`.

> [!NOTE]
> Custom Python workers incur interpreter initialization and framework import costs during cold start (typically 2s–10s+). For maximum throughput and sub-millisecond cold starts, prefer native ONNX (`ort`), Hugging Face Safetensors (`candle`), or pure-Rust tree (`hessboost`) formats.

### Real-Time Event Streaming (Kafka / Redpanda)

Connect models directly to Kafka or Redpanda topics without writing glue microservices:

1. Configure an input topic (`inference.requests`) and output topic (`inference.responses`) in `models.toml`.
2. Stream incoming payloads into SnoozeML's dynamic batcher.
3. Automatically commit broker offsets after successful forward pass and egress dispatch (**commit-after-infer**).
4. Walkthrough: See `05_streaming_xgboost.ipynb`.

---

## Cold Start vs. Warm In-Memory Inference

Every example notebook demonstrates SnoozeML's memory residency lifecycle:

```python
# First Request: Cold Start (weights dynamically allocated into host/GPU memory)
resp_cold = client.infer(model_id="iris_classifier", inputs={"input": X_test})
print(f"Cold Start Latency: {resp_cold.duration_ms:.2f} ms")

# Subsequent Requests: Warm In-Memory (resident weights with zero re-allocation)
resp_warm = client.infer(model_id="iris_classifier", inputs={"input": X_test})
print(f"Warm Latency:       {resp_warm.duration_ms:.2f} ms")

predictions = resp_warm.get_numpy()
```
