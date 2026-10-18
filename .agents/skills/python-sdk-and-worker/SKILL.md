---
name: python-sdk-and-worker
description: >-
  Runbook for developing and testing the SnoozeML Python SDK (python/snoozeml),
  extending client APIs, implementing tensor converters (NumPy/PyTorch), and
  authoring process-isolated Python worker models (snooze_model.py). Use this
  skill when editing the Python SDK, debugging worker IPC, or writing pytest
  suites.
---

# Python SDK & Custom Worker Runbook

This skill provides step-by-step guidance for developing SnoozeML's Python ecosystem:
1. **Python SDK Client (`python/snoozeml/client.py`)**: Synchronous `SnoozeClient` and asynchronous `AsyncSnoozeClient` supporting dual HTTP REST and gRPC transports.
2. **Tensor Serialization & Converters (`python/snoozeml/inference.py`)**: Zero-copy / contiguous conversions between NumPy `ndarray`, PyTorch `Tensor`, and SnoozeML's binary wire format.
3. **Process-Isolated Custom Worker Runtime (`python/snoozeml/worker.py`)**: Serving arbitrary custom models (e.g. Amazon Chronos-Bolt) inside isolated child processes with memory reclamation and IPC framing.

---

## 1. Adding or Modifying Python SDK Client Methods

When adding a new endpoint or operation to `SnoozeClient` / `AsyncSnoozeClient`:

### Step 1: Update Synchronous & Asynchronous Methods
1. In `python/snoozeml/client.py`, define both the synchronous method on `SnoozeClient` and the asynchronous method on `AsyncSnoozeClient`.
2. Ensure strict type annotations:
   ```python
   def my_operation(self, model_id: str, timeout: Optional[float] = None) -> ModelMetadata:
       """Perform operation on a model.

       Args:
           model_id: Unique model identifier.
           timeout: Optional request timeout in seconds.

       Returns:
           ModelMetadata object containing state and replica counts.

       Raises:
           ModelNotFoundError: If the model is not registered.
           SnoozeError: On server or transport failure.
       """
   ```
3. Use `self._request()` helper to ensure automatic authentication (`x-api-key` / `Authorization: Bearer`), base URL routing, and mapping of HTTP status codes to `SnoozeError` subclasses in `python/snoozeml/exceptions.py`.

### Step 2: Add Pytest Tests in `python/tests/`
1. Add behavioral unit and mock-transport tests in `python/tests/test_client.py`.
2. If testing against a live daemon, add end-to-end assertions in `python/tests/test_integration_e2e.py`.
3. Run:
   ```bash
   pytest python/tests/test_client.py -v
   ```

---

## 2. Authoring a Process-Isolated Custom Python Worker Model

When a user model requires specialized Python dependencies not supported natively by ONNX, Candle, or HessBoost trees:

### Step 1: Model Directory Structure
Create a dedicated folder containing at minimum `snooze_model.py`:
```text
my_model_dir/
├── snooze_model.py        # Required: Duck-typed entrypoint implementing load() and predict()
├── snooze_model.json      # Optional: Manifest configuration
└── requirements.txt       # Optional: Pip requirements
```

### Step 2: Implement Duck-Typed Model Class
In `snooze_model.py`:
```python
from typing import Dict, Any
import numpy as np

class Model:
    def __init__(self):
        self.model = None

    def load(self, model_dir: str, device: str) -> None:
        """Load model weights onto the target execution device.

        Args:
            model_dir: Absolute path to directory containing model files.
            device: Target device ('cpu', 'cuda', 'mps', or 'auto').
        """
        # Initialize heavy deep learning libraries here, NOT at module scope!
        # Example: import torch; self.model = ...

    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run batch inference on named input tensors.

        Args:
            inputs: Dictionary mapping tensor names to NumPy arrays or raw buffers.

        Returns:
            Dictionary mapping tensor names to output arrays (NumPy or PyTorch).
        """
        # Inputs are batched along Axis(0)
        input_tensor = inputs["input"]
        # If returning a PyTorch tensor, worker.py will auto-detach and move to CPU
        return {"output": input_tensor * 2.0}
```

### Step 3: Local Smoke-Testing (`--check`)
Always verify that the model loads and predicts properly without starting the SnoozeML daemon:
```powershell
python -m snoozeml.worker --check my_model_dir --input input:1x4 --device cpu
```
Expected output:
```text
OK: model loaded on 'cpu'
  output 'output': shape=[1, 4] first=[...]
```

For IPC framing and manifest specifications, see [references/worker-protocol.md](./references/worker-protocol.md).

---

## 3. Running the Python Test Suite

Run the full pytest suite across all modules:
```powershell
pytest python/tests -v
```

To run a specific module or test:
```powershell
pytest python/tests/test_worker.py -v -k test_run_check
```
