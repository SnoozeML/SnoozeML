# Custom Worker Runtime Review (`34a36edf`)

Overall, the custom worker runtime implementation in [`src/backend/worker_backend.rs`](../src/backend/worker_backend.rs) and [`python/snoozeml/worker.py`](../python/snoozeml/worker.py) is clean, well-structured, and fits SnoozeML's architecture very well.

---

## 1. What Looks Great

- **Process Isolation & Scale-to-Zero Synergy**: Running each replica as a separate child process (`python -u -c <embedded_worker>`) means scale-to-zero and LRU eviction reclaim 100% of host RAM and GPU VRAM via OS process termination—avoiding PyTorch CUDA caching-allocator retention and Python GIL contention across replicas.
- **Non-Blocking Teardown**: [`WorkerBackend::drop`](../src/backend/worker_backend.rs) sends `MSG_SHUTDOWN` and reaps the child process on a detached thread (`snooze-worker-reaper-{pid}`), so unloading inside async tasks or the background sweeper never blocks Tokio reactor threads.
- **Stdio Stream Isolation**: [`_isolate_stdio()`](../python/snoozeml/worker.py) duplicates file descriptors `0` and `1` for the binary protocol and redirects `sys.stdout` / fd `1` to `stderr`. User `print()` statements or noisy C++/CUDA extensions cannot corrupt the wire framing.
- **Zero-Panic & Defensive Bounds**: [`src/backend/worker_backend.rs`](../src/backend/worker_backend.rs) strictly follows `#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]`, validates `MAX_FRAME_BYTES` (1 GiB), uses checked arithmetic on tensor shapes/offsets, and recovers from mutex poisoning.
- **Zero-Config Packaging**: Embedding `worker.py` via `include_str!` avoids version skew between the `snooze` binary and the user's Python environment, and doesn't require `pip install snoozeml` inside every model's virtualenv.

---

## 2. Easy Usability & Ergonomics Improvements

Since ease of use is the primary goal of this component, these small tweaks remove common friction points for users writing custom models:

### 2.1 Auto-Detach and Move PyTorch Tensors Returned from `predict()`
In [`encode_tensors`](../python/snoozeml/worker.py):
```python
np = _numpy()
arr = np.ascontiguousarray(value, dtype="<f4")
```
If a user returns a PyTorch `Tensor` directly from `predict()` that is still on GPU (`cuda` / `mps`) or has `requires_grad=True` (e.g., they forgot `with torch.no_grad():` or `.detach().cpu().numpy()`), `np.ascontiguousarray(value)` fails with a `TypeError` or `RuntimeError`.

- **Easy improvement (2 lines, no `torch` import needed)**:
  ```python
  if hasattr(value, "detach") and hasattr(value, "cpu"):
      value = value.detach().cpu().numpy()
  ```
  With this duck-typed check, users can simply write `return {"output": self.net(x)}` and GPU/grad tensors work out of the box.

### 2.2 Honor `snooze_model.json` in `--check` Smoke-Test Mode
[`run_check`](../python/snoozeml/worker.py) currently uses the CLI defaults (`--entrypoint snooze_model.py`, `--class Model`, `--tensor-format numpy`) without reading `snooze_model.json` inside the model directory.
- If a model directory uses `snooze_model.json` to set `"class": "MyCustomModel"` or `"tensor_format": "raw"`, running `python -m snoozeml.worker --check models/my-model` will fail unless the user manually passes `--class` and `--tensor-format` on the command line.
- **Easy improvement**: Load `snooze_model.json` (if present in `args.check`) as the fallback defaults in `run_check` when flags aren't explicitly overridden.

### 2.3 Detect `snooze_model.json` When a Custom `entrypoint` Filename Is Used
[`WorkerManifest`](../src/backend/worker_backend.rs) supports custom entrypoints via `"entrypoint": "custom_file.py"` in `snooze_model.json`. However:
- [`is_worker_model`](../src/backend/worker_backend.rs) only checks `path_obj.join(WORKER_ENTRYPOINT_FILE).is_file()`.
- [`ModelDirectoryConfig::discover_models_in_dir`](../src/config.rs) and [`inspect_subdir`](../src/config.rs) only check for `snooze_model.py`.
- **Easy improvement**: Check `path.join(WORKER_ENTRYPOINT_FILE).is_file() || path.join(WORKER_MANIFEST_FILE).is_file()` so directories using `snooze_model.json` with a custom `entrypoint` are auto-discovered and routed to `WorkerBackend`.

### 2.4 Simplify Device Handling in Documentation Example
[`resolve_device`](../python/snoozeml/worker.py) already resolves `"auto"` to `"cuda"`, `"mps"`, or `"cpu"` before calling `model.load(model_dir, device)` whenever `torch` is imported in `snooze_model.py`.
- In [`docs/src/serving/custom-models.md`](../docs/src/serving/custom-models.md), the example still writes:
  ```python
  self.device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
  ```
- Updating the doc snippet to simply `self.device = torch.device(device)` makes the example cleaner and highlights that SnoozeML resolves `"auto"` for the user.

---

## 3. Free & Trivial Optimizations

Even though performance is not the primary concern here, there are two places in the Rust framing code that currently allocate and copy the entire tensor payload **twice** per frame, which can be simplified in ~10 lines:

### 3.1 Single-Allocation Frame Encoding in [`encode_infer_frame`](../src/backend/worker_backend.rs)
Currently:
1. `encode_infer_frame` allocates `let mut payload: Vec<u8> = Vec::new()` (growing across multiple inputs) and copies all input tensor bytes into `payload`.
2. It then calls `encode_frame(MSG_INFER, &header, &[&payload])`, which allocates a second `buf: Vec<u8>` and copies all of `payload` a second time.

- **Free optimization**: Compute `metas` and the total `payload_len` in the first loop (without copying bytes yet), serialize `header`, allocate a single `buf` of exact capacity `9 + header.len() + payload_len`, and copy each tensor's bytes once directly into `buf`. This removes an entire batch-sized allocation and `memcpy` on every request.

### 3.2 Avoid `Vec::split_off` Reallocation in [`read_frame`](../src/backend/worker_backend.rs)
Currently in `read_frame`:
```rust
let mut body = vec![0u8; frame_len];
reader.read_exact(&mut body)
    .map_err(|e| format!("truncated frame: {e}"))?;
...
let payload = body.split_off(header_end);
let header = body.get(5..).map(<[u8]>::to_vec).unwrap_or_default();
```
`body.split_off(header_end)` allocates a brand-new `Vec<u8>` for `payload` (which is >99% of the frame) and copies all the tensor bytes from `body` into `payload`, while `body` retains the huge `frame_len` capacity just to copy `body.get(5..)` into another `Vec` and immediately drop `body`.

- **Free optimization**: Since `reader` is already a `BufReader::with_capacity(64 KiB, stdout)`, read the 9-byte fixed prefix (`frame_len: u32`, `kind: u8`, `header_len: u32`), then read `header` directly into `vec![0u8; header_len]`, and read `payload` directly into `vec![0u8; frame_len - 5 - header_len]`. It is fewer lines of code and avoids allocating and copying the output tensor payload twice.

### 3.3 Tiny Python Framing Cleanups in [`python/snoozeml/worker.py`](../python/snoozeml/worker.py)
- **`read_frame`**: `json.loads(bytes(body[5 : 5 + header_len]).decode("utf-8"))` creates an extra `bytes` copy and `str` allocation. Python's `json.loads(body[5 : 5 + header_len])` accepts a `bytearray` slice directly.
- **`write_frame`**: Pre-compiling `_PREFIX = struct.Struct("<IBI")` replaces `_U32.pack(frame_len) + bytes([msg_type]) + _U32.pack(len(header_bytes))` with a single `_PREFIX.pack(frame_len, msg_type, len(header_bytes))`.

---

## 4. Minor Edge Cases & Repository Hygiene

1. **Windows Clippy Warning in `tests/worker_backend_test.rs`**:
   All 11 integration tests in `worker_backend_test` pass on Windows, but `let pid = backend.pid().unwrap();` in `test_worker_backend_drop_reaps_process` triggers `warning: unused variable: pid` on non-Unix targets because `pid` is only referenced inside `#[cfg(unix)]` blocks (which causes `cargo clippy --all-targets -- -D warnings` to fail on Windows). Using `let _pid = ...` or `assert!(backend.pid().is_some())` outside `#[cfg(unix)]` fixes it.
2. **Stale `stderr_tail` Across Respawns (`src/backend/worker_backend.rs`)**:
   When a worker crashes or times out and is later respawned on the next request, `self.stderr_tail` still holds lines from the previous process. Clearing `self.stderr_tail` at the beginning of `spawn()` prevents old tracebacks from being appended to future errors.
3. **Repository Paths**:
   Ensure all file paths across markdown files and documentation use relative repository links.
