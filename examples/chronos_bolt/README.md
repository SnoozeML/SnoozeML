# Chronos-Bolt Custom Worker Model for SnoozeML

This directory demonstrates how to deploy [**Amazon Chronos-Bolt**](https://github.com/amazon-science/chronos-forecasting) pretrained time-series foundation models as a custom, process-isolated Python worker model in SnoozeML.

---

## Why Chronos-Bolt + SnoozeML?

* **Zero-Shot Probabilistic Forecasting**: Chronos-Bolt produces multi-step quantile forecasts across arbitrary time series domains without task-specific training.
* **250x Faster Execution**: Built on direct multi-step patching rather than autoregressive decoding, offering low latency suitable for real-time APIs.
* **Full Process Isolation**: Because PyTorch models retain CUDA contexts and fragmented GPU memory, SnoozeML serves Chronos-Bolt in an isolated child process. Scale-to-zero memory reclamation frees 100% of host RAM and GPU VRAM upon inactivity.
* **Zero GIL Contention**: Multi-replica horizontal scaling spawns separate Python processes, scaling across CPU cores and GPU streams without Python GIL bottlenecks.
* **Adaptive Micro-Batching**: SnoozeML dynamically coalesces concurrent incoming single-series inference requests into a unified batch along `Axis(0)`.

> [!NOTE]
> **Custom Worker Trade-Off (Use as Last Resort)**: Custom Python workers allow complete flexibility for models with custom library dependencies. Keep in mind that worker **cold starts are in the seconds magnitude (typically 3–8s)** due to Python interpreter startup and deep learning framework initialization, and execution is slower than native compiled backends. Use custom workers as a last resort when native export paths (`.onnx`, native Safetensors, or `.xgb.json`) are not viable.

---

## Directory Contents

| File | Purpose |
| :--- | :--- |
| `snooze_model.py` | Duck-typed model entrypoint implementing `load(model_dir, device)` and `predict(inputs)` |
| `snooze_model.json` | Worker manifest specifying class name, execution timeouts, and environment variables |
| `requirements.txt` | Python dependencies (`chronos-forecasting`, `torch`, `pandas`, `snoozeml`) |

---

## Installation

Install the required custom libraries:

```bash
pip install -r examples/chronos_bolt/requirements.txt
```

---

## Local Smoke Testing (No Daemon Required)

Before launching the daemon, you can verify that the model loads and predicts properly using SnoozeML's built-in `--check` CLI:

```bash
# Smoke test on CPU with a mock 1x32 time-series context
python -m snoozeml.worker --check examples/chronos_bolt --input context:1x32 --device cpu
```

Output:
```text
OK: model loaded on 'cpu'
  output 'forecast': shape=[1, 9, 12] first=[...]
  output 'median': shape=[1, 12] first=[...]
```

---

## Deploying to SnoozeML

### Option A: Static Auto-Discovery

Copy or link this directory into your SnoozeML models directory:

```bash
# Place inside ./models
cp -r examples/chronos_bolt models/chronos_bolt

# Start SnoozeML
snooze serve --models-dir ./models
```

SnoozeML automatically detects `snooze_model.py` and registers the model as `chronos_bolt`.

### Option B: Explicit Configuration in `models.toml`

Add the model entry to `models.toml`:

```toml
[[models]]
id = "chronos_bolt"
path = "examples/chronos_bolt"
device = "cpu"                  # or "cuda" / "mps"
max_batch_size = 8
max_batch_delay_ms = 15
idle_timeout_secs = 120         # Scale-to-zero unload after 2 minutes of idle time
```

---

## Client Inference Example (Python SDK)

```python
import numpy as np
from snoozeml import SnoozeClient

client = SnoozeClient("http://localhost:8080")

# 32 historical time-series observations
historical_data = np.array([[112.0, 118.0, 132.0, 129.0, 121.0, 135.0, 148.0, 148.0,
                             136.0, 119.0, 104.0, 118.0, 115.0, 126.0, 141.0, 135.0,
                             125.0, 149.0, 170.0, 170.0, 158.0, 133.0, 114.0, 140.0,
                             145.0, 150.0, 178.0, 163.0, 172.0, 178.0, 199.0, 199.0]], dtype=np.float32)

# Send request to SnoozeML
resp = client.infer(
    model_id="chronos_bolt",
    inputs={"context": historical_data},
)

# Extract predictions
quantiles = resp.outputs["forecast"].as_numpy()  # Shape: (1, 9, 12)
median = resp.outputs["median"].as_numpy()        # Shape: (1, 12)

print(f"Inference latency: {resp.duration_ms:.2f} ms")
print(f"12-step median forecast: {median[0]}")
```
