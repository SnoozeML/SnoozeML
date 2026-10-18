# Python Worker IPC Protocol & Manifest Reference

This document outlines the binary framing, message types, and configuration schema used between the Rust daemon (`src/backend/worker_backend.rs`) and the child process worker (`python/snoozeml/worker.py`).

---

## 1. Binary Wire Framing (`SNZ1`)

Frames sent across `stdin` and `stdout` follow this binary layout:

```text
+-------------------+----------------+----------------------+--------------------+--------------------+-------------------------+
| Magic (4 bytes)   | Type (1 byte)  | Header Len (4 bytes) | JSON Header Bytes  | Payload Len (4 B)  | Raw Tensor Bytes Blocks |
| 'S' 'N' 'Z' '1'   | uint8 (1-5)    | uint32 (big-endian)  | UTF-8 JSON string  | uint32 (BE)        | Contiguous bytes        |
+-------------------+----------------+----------------------+--------------------+--------------------+-------------------------+
```

### Message Types
| Type ID | Constant | Sender | Description |
| :--- | :--- | :--- | :--- |
| `1` | `MSG_PING` | Rust | Liveness heartbeat; worker responds with `MSG_RESULT` containing `{"status": "ok"}` |
| `2` | `MSG_LOAD` | Rust | Instructs worker to load weights from `model_dir` on `device` |
| `3` | `MSG_INFER` | Rust | Submits batched request tensors along `Axis(0)` |
| `4` | `MSG_RESULT` | Worker | Returns execution status, metadata, and output tensor byte blocks |
| `5` | `MSG_SHUTDOWN` | Rust | Clean shutdown request; worker exits with code `0` |

---

## 2. Worker Manifest Schema (`snooze_model.json`)

An optional `snooze_model.json` file in the model directory configures worker startup:

```json
{
  "entrypoint": "snooze_model.py",
  "class": "Model",
  "tensor_format": "numpy",
  "timeout_seconds": 30.0,
  "env": {
    "OMP_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false"
  }
}
```

### Manifest Fields
* **`entrypoint`** (string, default `"snooze_model.py"`): Python file containing the model class.
* **`class`** (string, default `"Model"`): Name of the class implementing `load()` and `predict()`.
* **`tensor_format`** (string, default `"numpy"`): Tensor representation passed to `predict()`. Options:
  - `"numpy"`: Converts input wire buffers to `np.ndarray` with appropriate shape and dtype.
  - `"raw"`: Passes raw byte buffers directly without NumPy conversion.
* **`timeout_seconds`** (float, default `30.0`): Maximum seconds allowed for an inference call before the Rust backend aborts and recycles the worker process.
* **`env`** (object, optional): Key-value environment variables injected into the child worker process on spawn.

---

## 3. Worker Auto-Discovery Rules

When `snooze serve --models-dir <dir>` scans directories, a subdirectory is recognized as a Python worker model if either:
1. `snooze_model.py` exists, OR
2. `snooze_model.json` exists.

The directory name becomes the default `model_id`.
