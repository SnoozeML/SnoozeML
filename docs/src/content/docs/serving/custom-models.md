---
title: "Custom User Models & Process Workers"
description: "SnoozeML supports custom user models at runtime via a process-isolated worker backend (`worker-backend`)."
---

SnoozeML supports custom user models at runtime via a process-isolated worker backend (`worker-backend`).

Instead of compiling custom deep learning architectures into the Rust binary or linking against `libpython` with PyO3, SnoozeML places the plugin boundary at a **child process communicating over a high-performance binary pipe protocol**.

:::caution
**Performance Trade-Off: Use as a Last Resort**
Custom Python workers provide unmatched flexibility to run arbitrary architectures and custom third-party libraries (e.g. PyTorch pipelines, AutoGluon, or `chronos-forecasting`). However, they are **slower than native engines** due to process boundaries and pipe IPC serialization.

Most importantly, custom worker **cold starts require significantly more time (in the seconds magnitude, typically 2–10+ seconds)** as the system must spawn a new Python child process, initialize heavy frameworks like PyTorch or CUDA runtimes, and parse weights into memory.

**Always prefer native execution engines first** (exporting models to ONNX via `torch.onnx.export` / `optimum-cli` or using native Safetensors / pure-Rust tree formats). Custom Python workers should be used **only as a last resort** when model export to ONNX is impossible or custom operators are unsupported.
:::

---

## Trade-Offs: Native Backends vs. Custom Python Workers

| Dimension | Native Engines (`ort`, `candle`, `hessboost`) | Custom Python Workers (`worker-backend`) |
| :--- | :--- | :--- |
| **Primary Goal** | Maximum throughput and ultra-low latency | Maximum flexibility and framework compatibility |
| **Cold Start Latency** | **Fast** (tens to hundreds of milliseconds) | **Slow** (in the **seconds magnitude**, 2s–10s+) |
| **Inference Overhead** | Zero IPC; in-process memory pointers | Binary pipe IPC framing over stdin/stdout |
| **Runtime Dependencies** | Self-contained single binary (no Python needed) | External Python interpreter + installed libraries |
| **Scale-to-Zero** | In-process memory drop / VRAM eviction | Process termination (100% OS RAM/VRAM return) |
| **Recommended Usage** | **Default / First Choice** | **Last Resort** (when native formats are not viable) |

---

## Why Process Isolation?

| SnoozeML Property | Benefit with Process Workers |
|---|---|
| **Scale-to-Zero & LRU Eviction** | Terminating the child process guarantees 100% reclamation of GPU VRAM and host RAM by the OS. In-process PyTorch allocators frequently fragment or retain CUDA contexts. |
| **Reactive OOM Recovery** | Memory errors (`torch.cuda.OutOfMemoryError` or `MemoryError`) are mapped to `InferenceError::OutOfMemory`. SnoozeML automatically triggers LRU eviction and retries the batch. |
| **Zero-GIL Autoscaling** | Each replica runs in its own process interpreter. Concurrency scales across CPU cores and GPU streams without GIL contention. |
| **Zero-Panic Stability** | Crashes or segfaults inside custom native extensions only kill that worker; the daemon stays up and gracefully respawns the replica. |
| **Single-Binary Portability** | No compile-time Python or libpython dependencies. The single `snooze` binary serves custom models whenever Python is available. |
| **Dynamic Micro-Batching** | Unchanged. The Rust batcher collates concurrent requests along `Axis(0)`. The worker only ever receives fully collated batches. |

---

## Creating a Custom Worker Model

To serve a custom model, place a directory in `models/` containing `snooze_model.py`:

```text
models/my-custom-net/
├── snooze_model.py        # Required entrypoint
├── model.safetensors      # Weights or assets
└── snooze_model.json      # Optional manifest
```

### Python Entrypoint (`snooze_model.py`)

A worker model is duck-typed: it defines a `Model` class with `load` and `predict` methods. No `snoozeml` import is required:

```python
import numpy as np
import torch
from safetensors.torch import load_file

class Model:
    def load(self, model_dir: str, device: str) -> None:
        """Load weights and initialize the model on the requested device."""
        self.device = torch.device(device)
        # Load weights from model_dir
        # self.net = MyPyTorchNet().to(self.device).eval()

    def predict(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Perform inference on the collated batch (batch on Axis 0)."""
        x = torch.from_numpy(inputs["input"]).to(self.device)
        with torch.no_grad():
            output = x * 2.0 + 1.0  # Example transformation
            return {"output": output.cpu().numpy()}
```

### Optional Manifest (`snooze_model.json`)

To customize interpreter paths, class names, timeouts, or environment variables, add `snooze_model.json`:

```json
{
  "python": ".venv/bin/python",
  "entrypoint": "snooze_model.py",
  "class": "Model",
  "load_timeout_secs": 300,
  "request_timeout_secs": 60,
  "tensor_format": "numpy",
  "env": {
    "OMP_NUM_THREADS": "1"
  }
}
```

#### Manifest Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `python` | `string` | Auto | Interpreter executable (resolved relative to model dir, `.venv`, `SNOOZE_PYTHON`, or `python3`). |
| `entrypoint` | `string` | `"snooze_model.py"` | Python file containing the model class. |
| `class` | `string` | `"Model"` | Name of the model class. |
| `load_timeout_secs` | `integer` | `300` | Maximum seconds allowed for `load()` before killing the worker. |
| `request_timeout_secs` | `integer` | `60` | Maximum seconds allowed for a single batched `predict()` call. |
| `tensor_format` | `string` | `"numpy"` | Format delivered to `predict`: `"numpy"` or `"raw"` (`RawTensor` without numpy). |
| `env` | `object` | `{}` | Key-value environment variables injected into the child process. |

---

## Configuration in `models.toml`

SnoozeML auto-discovers worker model directories containing `snooze_model.py`. You can also configure them explicitly in `models.toml`:

```toml
[[models]]
id = "custom-net"
path = "models/my-custom-net"
device = "cuda"
max_batch_size = 16
max_batch_delay_ms = 10
idle_timeout_secs = 60
```

When `device` is `"cuda"`, `"cpu"`, `"mps"`, or `"cuda:N"`, it is passed directly to `load(model_dir, device)`.

---

## Local Smoke Testing CLI

You can smoke-test your custom model locally without running the daemon using `--check`:

```bash
python -m snoozeml.worker --check models/my-custom-net --input input:4x3 --device cpu
```

Output:
```text
OK: model loaded on 'cpu'
  output 'output': shape=[4, 3] first=[3.0, 5.0, 7.0]
```

---

## Wire Protocol (v1)

Communication between the daemon and workers flows across stdin and stdout using length-prefixed frames:

```text
[u32 LE frame_len][u8 msg_type][u32 LE header_len][header JSON][raw little-endian f32 bytes...]
```

| Type ID | Name | Direction | Header Schema | Payload |
|---|---|---|---|---|
| `1` | `READY` | Worker → Host | `{"protocol": 1, "info": { ... }}` | Empty |
| `2` | `INFER` | Host → Worker | `{"tensors": [{"name": "...", "shape": [...], "dtype": "f32"}]}` | Concatenated LE `f32` |
| `3` | `RESULT` | Worker → Host | `{"tensors": [{"name": "...", "shape": [...], "dtype": "f32"}]}` | Concatenated LE `f32` |
| `4` | `ERROR` | Worker → Host | `{"kind": "oom"|"invalid_input"|"load"|"runtime", "message": "..."}` | Empty |
| `5` | `SHUTDOWN` | Host → Worker | `{}` | Empty |

The worker duplicates `stdin`/`stdout` file descriptors for the protocol and redirects standard output to standard error, ensuring user `print()` statements and C extension messages never corrupt the binary stream.

---

## Real-World Example: Amazon Chronos-Bolt

For a complete, real-world example demonstrating how to serve Amazon Science's **Chronos-Bolt** zero-shot time series foundation model using custom dependencies (`chronos-forecasting`, `torch`), see:
- Worker model directory: [`examples/chronos_bolt/`](https://github.com/SnoozeML/SnoozeML/tree/main/examples/chronos_bolt)
- Interactive Jupyter notebook: [`examples/04_chronos_bolt_custom_model.ipynb`](https://github.com/SnoozeML/SnoozeML/blob/main/examples/04_chronos_bolt_custom_model.ipynb)
