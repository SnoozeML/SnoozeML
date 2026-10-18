# Worker Runtime Follow-Up Plan: Usability Tweaks & Zero-Copy Framing

This document outlines two targeted follow-up improvements to the process-isolated custom worker runtime (`src/backend/worker_backend.rs` and `python/snoozeml/worker.py`).

---

## Plan 1: Usability & Ergonomics Improvements

### Goal
Remove common friction points when users author and smoke-test custom Python models (`snooze_model.py` / `snooze_model.json`), without adding any new dependencies.

### 1.1 Auto-Detach and Move PyTorch Tensors Returned by `predict()`
- **File**: `python/snoozeml/worker.py` (`encode_tensors`)
- **Problem**: If a user's `predict(self, inputs)` returns a `torch.Tensor` directly that resides on a GPU (`cuda` or `mps`) or has `requires_grad=True`, calling `np.ascontiguousarray(value, dtype="<f4")` raises a `TypeError` or `RuntimeError` unless the user manually called `.detach().cpu().numpy()`.
- **Change**: Use duck-typing before converting to NumPy so `worker.py` still does not import `torch`:
  ```python
  if hasattr(value, "detach") and hasattr(value, "cpu"):
      value = value.detach().cpu().numpy()
  arr = np.ascontiguousarray(value, dtype="<f4")
  ```
- **Test**: Add a unit test in `python/tests/test_worker.py` passing a fake tensor object implementing `.detach()` and `.cpu()` to `encode_tensors()` and verifying it encodes cleanly.

### 1.2 Honor `snooze_model.json` in `python -m snoozeml.worker --check`
- **File**: `python/snoozeml/worker.py` (`run_check` and `main`)
- **Problem**: Running `python -m snoozeml.worker --check models/my-model` ignores `snooze_model.json` in `models/my-model` and always uses hardcoded CLI defaults (`entrypoint="snooze_model.py"`, `class_name="Model"`, `tensor_format="numpy"`). Models using custom manifest settings fail `--check` unless the user repeats every setting via CLI flags.
- **Change**:
  - Change `--entrypoint`, `--class`, and `--tensor-format` defaults in `main()` to `None`.
  - In `run_check()`, if `snooze_model.json` exists in `args.check`, read `"entrypoint"`, `"class"`, and `"tensor_format"` from the manifest whenever the corresponding CLI flag was not explicitly provided, falling back to `"snooze_model.py"`, `"Model"`, and `"numpy"`.
  - In `run_worker()`, fall back to `"snooze_model.py"`, `"Model"`, and `"numpy"` if `None`.
- **Test**: Add a test in `python/tests/test_worker.py` running `run_check` against a temporary model directory containing a `snooze_model.json` with a custom `"class"` and `"tensor_format": "raw"`.

### 1.3 Detect `snooze_model.json` During Worker Model Discovery
- **Files**:
  - `src/backend/worker_backend.rs` (`is_worker_model`)
  - `src/backend/mod.rs` (`create_backend`, `#[cfg(not(feature = "worker-backend"))]`)
  - `src/config.rs` (`ModelDirectoryConfig::discover_models_in_dir` and `inspect_subdir`)
- **Problem**: `WorkerManifest` allows overriding `"entrypoint"` in `snooze_model.json` (e.g. `"entrypoint": "custom_entry.py"`), but directory detection only checks if `snooze_model.py` exists. A directory with `snooze_model.json` and a custom entrypoint filename is not recognized as a worker model.
- **Change**: Treat a directory as a worker model if either `snooze_model.py` (`WORKER_ENTRYPOINT_FILE`) **or** `snooze_model.json` (`WORKER_MANIFEST_FILE`) exists as a file inside it.
- **Test**: Add a test in `tests/worker_backend_test.rs` verifying that `is_worker_model` and `discover_models_in_dir` recognize a directory containing `snooze_model.json` + a custom `.py` entrypoint.

### 1.4 Simplify Device Handling in Documentation & Clear Stale `stderr_tail`
- **Files**:
  - `docs/src/serving/custom-models.md`
  - `src/backend/worker_backend.rs` (`WorkerBackend::spawn`)
  - `tests/worker_backend_test.rs` (`test_worker_backend_drop_reaps_process`)
- **Changes**:
  - In `docs/src/serving/custom-models.md`, simplify `self.device = torch.device(device)` since `resolve_device("auto")` in `worker.py` already resolves `"auto"` to `"cuda"`, `"mps"`, or `"cpu"`.
  - In `WorkerBackend::spawn`, clear `self.stderr_tail` before spawning a new child process so respawned workers do not inherit stale stderr lines from a previously crashed process.
  - In `tests/worker_backend_test.rs`, guard or use `pid` on non-Unix platforms to eliminate the `unused variable: pid` warning on Windows.

---

## Plan 2: Single-Allocation / Zero-Extra-Copy Framing Cleanups

### Goal
Eliminate redundant memory allocations and full-payload `memcpy` passes when encoding `INFER` frames and decoding `RESULT` frames in `src/backend/worker_backend.rs` and `python/snoozeml/worker.py`.

### 2.1 Single-Allocation Request Encoding (`encode_infer_frame`)
- **File**: `src/backend/worker_backend.rs` (`encode_infer_frame`)
- **Problem**: `encode_infer_frame` currently allocates a temporary `payload: Vec<u8>`, copies all input tensor bytes into it, and then calls `encode_frame(MSG_INFER, &header, &[&payload])`, which allocates a second `Vec<u8>` (`buf`) and copies the entire `payload` a second time.
- **Change**:
  1. In the first pass over `inputs`, build the `metas` JSON array and compute `payload_len: usize` using `checked_mul(4)` and `checked_add` without copying tensor data yet.
  2. Serialize `header` (`serde_json::to_vec`) and validate `5 + header.len() + payload_len <= MAX_FRAME_BYTES`.
  3. Allocate a single `Vec<u8>` with exact capacity `4 + 5 + header.len() + payload_len`, write the 9-byte frame prefix and `header`, and append each tensor's bytes directly into `buf` (via `bytemuck::cast_slice` on little-endian targets).
- **Result**: Cuts per-request buffer allocations and payload memory copies in half.

### 2.2 Single-Allocation Frame Decoding (`read_frame`)
- **File**: `src/backend/worker_backend.rs` (`read_frame`)
- **Problem**: `read_frame` currently allocates `body = vec![0u8; frame_len]`, reads the entire frame into `body`, and then calls `let payload = body.split_off(header_end)` and `let header = body.get(5..).map(<[u8]>::to_vec)`. `Vec::split_off` allocates a brand-new `Vec<u8>` for `payload` (which is >99% of the frame) and copies all tensor bytes into it, while leaving `body` with the full `frame_len` capacity just to copy the small header out again.
- **Change**:
  1. Read the 9-byte fixed prefix (`[u8; 9]`: `frame_len: u32`, `kind: u8`, `header_len: u32`) from the `BufReader`.
  2. Validate `5 <= frame_len <= MAX_FRAME_BYTES` and `header_len + 5 <= frame_len`, computing `payload_len = frame_len - 5 - header_len`.
  3. Allocate `header = vec![0u8; header_len]` and call `reader.read_exact(&mut header)`.
  4. Allocate `payload = vec![0u8; payload_len]` and call `reader.read_exact(&mut payload)`.
- **Result**: Simpler code with zero `split_off` or slice-copy overhead; the output tensor payload is allocated and written once.

### 2.3 Minor Python Framing Cleanups
- **File**: `python/snoozeml/worker.py` (`read_frame` and `write_frame`)
- **Changes**:
  - Pre-compile `_FRAME_PREFIX = struct.Struct("<IBI")` and use `_FRAME_PREFIX.pack(frame_len, msg_type, len(header_bytes))` in `write_frame` instead of concatenating three separate byte objects.
  - In `read_frame`, pass `body[5 : 5 + header_len]` directly to `json.loads(...)` instead of `bytes(body[5 : 5 + header_len]).decode("utf-8")`.

---

## Verification Steps

```bash
# Rust unit and integration tests (including worker_backend_test)
cargo test --all-targets

# Strict Clippy lints (zero warnings across all targets)
cargo clippy --all-targets -- -D warnings

# Formatting check
cargo fmt --check

# Python worker unit tests
pytest python/tests/test_worker.py
```
