# SnoozeML Python Examples & Notebooks

This folder contains step-by-step, interactive Jupyter notebooks demonstrating how to train models using standard machine learning libraries, export them to SnoozeML-compatible formats, deploy them into a running SnoozeML instance (Docker or bare-metal), and run low-latency micro-batched inference with the official `snoozeml` Python SDK.

---

## Available Notebooks

| Notebook | Model Type | Dataset | Target Backend | Task |
| :--- | :--- | :--- | :--- | :--- |
| [**`01_sklearn_onnx.ipynb`**](01_sklearn_onnx.ipynb) | Scikit-learn `RandomForestClassifier` | Iris | ONNX Runtime (`ort`) | Multiclass Classification |
| [**`02_xgboost_tree.ipynb`**](02_xgboost_tree.ipynb) | XGBoost `XGBClassifier` | Breast Cancer | Pure-Rust Trees (`hessboost`) | Binary Classification |
| [**`03_pipeline_regression_onnx.ipynb`**](03_pipeline_regression_onnx.ipynb) | Scikit-learn `Pipeline` (`StandardScaler` + `Ridge`) | Diabetes | ONNX Runtime (`ort`) | Continuous Regression |
| [**`04_chronos_bolt_custom_model.ipynb`**](04_chronos_bolt_custom_model.ipynb) | Amazon Chronos-Bolt Foundation Model | Synthetic / Time Series | Custom Worker (`worker-backend`) | Zero-Shot Probabilistic Forecasting |
| [**`05_streaming_xgboost.ipynb`**](05_streaming_xgboost.ipynb) | XGBoost `XGBClassifier` | Breast Cancer | Pure-Rust Trees (`hessboost`) | Real-Time Event Streaming (Redpanda / Kafka) |

---

## Prerequisites

1. **Install Python SDK & Dependencies:**
   ```bash
   # Install SnoozeML Python SDK from PyPI with all runtime extras
   pip install "snoozeml[full]"
   pip install jupyter

   # (Optional) For editable development from a cloned repository:
   # pip install -e "./python[full]"

   # Optional: For Chronos-Bolt custom worker (Notebook 04):
   pip install -r examples/chronos_bolt/requirements.txt

   # Optional: For event streaming with Redpanda / Kafka (Notebook 05):
   pip install kafka-python-ng
   ```

2. **Launch the SnoozeML Daemon (Docker):**
   SnoozeML's `docker-compose.yml` mounts the repository's `./models` folder directly into the container at `/models`:
   ```bash
   docker compose up -d
   ```

   Verify that the daemon is live:
   ```bash
   curl http://localhost:8080/health/live
   # {"status":"live"}
   ```

   *(Optional: Alternatively run the native Rust binary directly: `snooze serve --models-dir ./models`)*

3. **Launch Jupyter Notebooks:**
   ```bash
   jupyter notebook examples/
   ```

---

## How "Upload to SnoozeML" Works

SnoozeML supports **direct HTTP uploads** as well as filesystem volume mounting:

### Option A: Direct HTTP Upload (Recommended — No Volume Mounting)
Upload your model weights directly over HTTP without needing local filesystem mounts or container restarts:

```python
from snoozeml import SnoozeClient

# Connect to daemon (with optional authentication)
# client = SnoozeClient("http://localhost:8080", api_key="snooze_admin_secret_key")
client = SnoozeClient("http://localhost:8080")

# Upload from a file path:
info = client.upload_model(
    file="iris_classifier.onnx",
    model_id="iris_classifier",
    pinned=False,               # Keep unpinned to enable scale-to-zero memory reclamation
    max_batch_size=16,          # Max micro-batch size for Axis(0) dynamic collation
    max_batch_delay_ms=10,      # 10ms micro-batch latency accumulation budget
    idle_timeout_secs=300,      # Scale-to-zero memory unload after 5 min of inactivity
)

# Or upload raw in-memory bytes directly without saving to disk:
# raw_bytes = booster.save_raw(raw_format="json")
# info = client.upload_model(
#     file=raw_bytes,
#     filename="cancer_xgboost.xgb.json",
#     model_id="cancer_xgboost",
# )
```

### Option B: Runtime Refresh of Discovered Models
If you mount `./models` via Docker or drop new files onto the server's disk, trigger automatic discovery and registration instantly via HTTP without restarting the daemon:

```python
client.refresh_models()
```

### Option C: Custom Python Worker Models (Last Resort)
For models requiring custom Python libraries or deep learning frameworks (such as [Chronos-Bolt](chronos_bolt/README.md)) that cannot be exported to ONNX:
1. Place a directory containing `snooze_model.py` and optional `snooze_model.json` inside `./models` (or configure it in `models.toml`).
2. SnoozeML automatically discovers the directory and serves it in an isolated child process with 100% OS scale-to-zero memory reclamation and dynamic micro-batching along `Axis(0)`.
3. *Note on Performance*: Custom workers are slower and have **cold starts in the seconds magnitude (2s–10s+)** due to Python process creation and model imports. Always prefer native ONNX or Safetensors when possible.
See [**`04_chronos_bolt_custom_model.ipynb`**](04_chronos_bolt_custom_model.ipynb) and [`examples/chronos_bolt/`](chronos_bolt/) for a full walkthrough.

### Option D: Real-Time Event Streaming (Redpanda / Kafka / RabbitMQ / Redis)
Connect SnoozeML directly to distributed event streaming topics without writing custom glue microservices. Declare a `[[streams]]` table in `models.toml`, publish `StreamInferRequest` envelopes to an input topic, and read `StreamInferResponse` predictions from an output topic.
See [**`05_streaming_xgboost.ipynb`**](05_streaming_xgboost.ipynb) for a complete walkthrough using Redpanda (Kafka) and a pure-Rust XGBoost model.

---

## Cold Start vs. Warm In-Memory Inference

Every notebook showcases SnoozeML's dynamic lifecycle behavior:

```python
# 1. First Request: Cold Start (weights dynamically allocated into host/GPU memory)
resp_cold = client.infer(model_id="iris_classifier", inputs={"input": X_test})
print(f"❄️  Cold Start Latency: {resp_cold.duration_ms:.2f} ms (disk load + forward pass)")

# 2. Second Request: Warm In-Memory (weights resident in memory with zero allocation overhead)
resp_warm = client.infer(model_id="iris_classifier", inputs={"input": X_test})
print(f"🔥 Warm Latency:        {resp_warm.duration_ms:.2f} ms (pure in-memory micro-batch)")

predictions = resp_warm.get_numpy()
```
