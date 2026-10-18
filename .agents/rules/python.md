---
trigger: always_on
---

# Python Engineering Standards (`python/`, `tools/`, `examples/`)

SnoozeML includes a first-class Python SDK (`python/snoozeml`), interactive example notebooks (`examples/`), operational tooling (`tools/`), and the process-isolated custom worker runtime (`python -m snoozeml.worker`). All Python contributions must adhere to the following standards.

---

## 1. Type Annotations & Static Typing
* **Strict Type Annotations:** All public classes, methods, and functions in `python/snoozeml/` must have complete type signatures using standard library `typing` (`Optional`, `Union`, `List`, `Dict`, `Tuple`, `Any`, `Callable`, `Protocol`) and `numpy.typing.NDArray`.
* **Zero Untyped Call Sites:** Avoid untyped arguments (`def foo(x):`) in production SDK code. Where polymorphic duck typing is required (e.g. arbitrary inputs to `predict()`), annotate as `Dict[str, Any]` or use `typing.Protocol`.
* **Sphinx Docstring Parity:** Ensure docstrings reflect exact type signatures and describe parameter bounds and return types using Google or Sphinx docstring format.

---

## 2. Defensive Error Handling & Context Preservation
* **No Naked `except:`:** Never use bare `except:` or catch unhandled `Exception` silently without logging or re-raising. Always catch specific exceptions (`KeyError`, `ValueError`, `ConnectionError`, `RuntimeError`).
* **SnoozeML Exception Hierarchy:** Map HTTP/gRPC client errors and worker failures to domain exceptions in `python/snoozeml/exceptions.py`:
  - `SnoozeError` (base exception)
  - `ModelNotFoundError` (404 / NotFound)
  - `InvalidInputError` (400 / InvalidArgument / ShapeMismatch)
  - `ModelLoadingError` (503 / Unavailable / Loading)
  - `ResourceExhaustedError` (503 / ResourceExhausted)
  - `OutOfMemoryError` (503 / ResourceExhausted / OOM)
  - `InferenceTimeoutError` (504 / DeadlineExceeded)
  - `WorkerCrashError` (Worker process crashed or terminated)
* **Preserve Exception Causes:** When translating errors, always preserve tracebacks via `raise SpecificError(...) from err`.

---

## 3. NumPy & PyTorch Memory & Layout Invariants
* **C-Contiguous Arrays:** SnoozeML's binary wire protocol expects contiguous byte sequences along row-major order:
  - Always enforce `np.ascontiguousarray(arr, dtype=...)` before taking `.tobytes()` or reading raw pointer buffers.
  - Reject or convert non-contiguous slices or Fortran-contiguous arrays cleanly.
* **PyTorch GPU & Autograd Detach:**
  - Before converting PyTorch tensors to NumPy or raw bytes in `worker.py` or converters, check if the tensor has autograd or resides on a GPU:
    ```python
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    ```
  - This prevents `RuntimeError: Can't call numpy() on Tensor that requires grad` and avoids GPU CUDA context memory leaks.
* **Little-Endian Standard:** Floating point and integer buffers on the wire are standard little-endian (`<f4`, `<f8`, `<i4`, `<i8`). Validate endianness before byte copying.

---

## 4. Custom Python Worker Isolation & Framing
* **No Heavy Imports in Worker Wrapper:** Keep `python/snoozeml/worker.py` lean. Never import `torch`, `transformers`, or deep learning frameworks at module scope inside `worker.py`. Deep learning dependencies must be dynamically imported only inside user models (`snooze_model.py`) upon `load()`.
* **Duck-Typed Model Interface:** A valid custom Python model must implement:
  - `load(self, model_dir: str, device: str) -> None`: Initialize weights onto the designated device (`"cpu"`, `"cuda"`, `"mps"`).
  - `predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]`: Run forward inference on inputs and return named tensor dictionaries.
* **Subprocess Hygiene:** Child worker processes must flush `sys.stdout` and `sys.stderr` promptly. Binary IPC frames travel over dedicated binary streams, leaving `stderr` for diagnostic logging.

---

## 5. Testing & Verification (Zero Mocks Policy)
* **Dedicated Test Directory (`python/tests/`):** All SDK unit and integration tests reside in `python/tests/`.
* **Zero Dummy Mocks:** Avoid mocking internal business logic. Exercise real converters, real `run_check` CLI invocations, real serialization framing, and ephemeral live server connections (or `pytest-httpx` / mock gRPC channel only at the transport boundary when live servers are unavailable).
* **Mandatory Test Command:** All tests must pass cleanly:
  ```bash
  pytest python/tests -v
  ```
