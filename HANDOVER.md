# CUDA Execution & Release Artifact Size — Handover Document

This document summarizes the investigation into the discrepancy between documented and actual release/container file sizes, the root causes preventing CUDA execution (or causing silent CPU fallbacks), the fixes already applied to the repository, and the remaining steps to validate on a machine with an NVIDIA GPU.

---

## 1. Problems Identified

### 1.1 Documented Size Estimates vs. Actual Release & Container Sizes
- **Where the README estimates came from**:
  - The `~120 MB` (`:slim`) and `~280 MB` (`:cpu`) values in `README.md` and `docs/src/content/docs/getting-started/installation.md` were rough **uncompressed** container layer estimates (`debian:trixie-slim` ~75 MB uncompressed + `snooze` binary ~52 MB uncompressed).
  - The `~10x smaller download` comment in `docs/src/content/docs/getting-started/installation.md` assumed the standard Linux release archive bundled the ~200+ MB `libonnxruntime_providers_cuda.so` library.
- **Actual sizes on GitHub Releases (`v0.1.1`) and GHCR**:
  - `snooze-linux-x86_64-cpu.tar.gz`: **19.54 MB**
  - `snooze-linux-x86_64.tar.gz` (CUDA/TensorRT): **19.65 MB** *(only 0.11 MB larger than CPU due to the symlink bug below)*
  - `snooze-windows-x86_64-cpu.zip`: **18.24 MB**
  - `snooze-windows-x86_64.zip` (CUDA/TensorRT): **52.59 MB** (contains `snooze.exe` at 51.7 MB + `onnxruntime_providers_cuda.dll` at 59.4 MB + TensorRT DLLs)
  - `ghcr.io/snoozeml/snoozeml:slim`: **52.44 MB** compressed
  - `ghcr.io/snoozeml/snoozeml:cpu`: **82.98 MB** compressed
  - `ghcr.io/snoozeml/snoozeml:cuda`: **83.10 MB** compressed *(only 0.12 MB larger than `:cpu` due to the same symlink bug)*

### 1.2 Bug 1 — Dangling Symlinks in Linux Release Tarballs & Docker Images (`cp -P`)
- On Unix, `ort-sys` (`build/dynamic_link.rs`) does not copy `libonnxruntime_providers_*.so` into `target/release/`; it creates **symbolic links** pointing to the build host's cache directory (`$HOME/.cache/ort.pyke.io/dfbin/...`).
- Both `.github/workflows/release.yml` and `Dockerfile` copied those files using `cp -P` (`--no-dereference`).
- As a result, `snooze-linux-x86_64.tar.gz` and `ghcr.io/snoozeml/snoozeml:cuda` packaged **0-byte broken symlinks** (`libonnxruntime_providers_cuda.so`, `libonnxruntime_providers_shared.so`, `libonnxruntime_providers_tensorrt.so`) pointing to the CI runner's filesystem instead of the actual shared libraries.

### 1.3 Bug 2 — Silent CPU Fallback & False `"onnxruntime-cuda"` Reporting in `OrtBackend`
- In `ort 2.0`, `ExecutionProviderDispatch` (`ort::ep::CUDA::default().build()`) defaults to `error_on_failure = false` (`fail_silently()`).
- In `src/backend/ort_backend.rs`, `OrtBackend::try_commit_session` registered `CUDA`, `TensorRT`, and `CoreML` **without calling `.error_on_failure()`**.
- Whenever CUDA registration failed (missing GPU, broken `libonnxruntime_providers_cuda.so` symlink, or missing CUDA/cuDNN libraries), `ort` swallowed the error, returned `Ok(())`, and silently fell back to `CPUExecutionProvider`.
- Because `try_commit_session` returned `Ok(session)`, `OrtBackend::new` logged `"Auto device: initialized with CUDA GPU"`, set `self.provider = ExecutionProvider::Cuda`, and reported `"onnxruntime-cuda"` in `snooze status` and `/v1/models` while executing 100% on the host CPU.
- Furthermore, when `ExecutionProvider::Cuda`, `ExecutionProvider::TensorRt`, or `ExecutionProvider::CoreML` was **explicitly** requested by the user, `OrtBackend::new` caught errors and fell back to CPU instead of returning an error.

### 1.4 Bug 3 — Candle (`safetensors`) Never Compiled With CUDA Features in `Cargo.toml`
- In `Cargo.toml`, `cuda = ["ort-backend", "ort?/cuda"]` only enabled `ort/cuda` and never enabled `candle-core/cuda`, `candle-nn/cuda`, or `candle-transformers/cuda` (and `metal` was not wired for macOS).
- Calling `Device::new_cuda(0)` in `src/backend/candle_backend.rs` without `candle-core/cuda` unconditionally returned `Err("not compiled with cuda support")` and fell back to `candle-cpu` even when `CandleDevice::Cuda` was explicitly requested.

### 1.5 Bug 4 — Python Worker `resolve_device("auto")` Checked `sys.modules` Before `load()`
- In `python/snoozeml/worker.py`, `resolve_device("auto")` only checked `sys.modules.get("torch")` **before** calling `model.load(model_dir, resolved)`.
- For custom Python worker models that import `torch` lazily inside `load()`, `sys.modules.get("torch")` was `None`, causing `"auto"` to resolve to `"cpu"` even on machines with PyTorch + CUDA installed.

### 1.6 Bug 5 — CUDA 13 & cuDNN 9 Runtime Requirements for `ort 2.0.0-rc.13`
- The prebuilt `ort-sys 2.0.0-rc.13` binaries (`ms@1.28.0`) are linked against **CUDA 13** (`cublas64_13.dll` / `libcublas.so.13`, `cudart64_13.dll` / `libcudart.so.13`) and **cuDNN 9** (`cudnn64_9.dll` / `libcudnn.so.9`).
- On Linux container deployments, the `cuda` target in `Dockerfile` inherits from `debian:trixie-slim`, which does not bundle CUDA 13 runtime libraries or cuDNN 9 (`docker run --gpus all` only mounts the host NVIDIA driver `libcuda.so.1`, not CUDA Toolkit or cuDNN libraries).

---

## 2. What Has Been Done

1. **Enforced Zero Silent Failures Policy (`AGENTS.md`, `.agents/rules/rust.md`)**:
   - Added an explicit rule stating that silent failures are unacceptable: hardware initialization or execution provider registration failures must never be swallowed or misreported, and fallbacks are strictly limited to `auto` device discovery.
2. **Fixed `OrtBackend` Silent Fallback & False Provider Reporting (`src/backend/ort_backend.rs`)**:
   - Chained `.error_on_failure()` onto `ort::ep::CUDA::default().build()`, `ort::ep::TensorRT::default().build()`, and `ort::ep::CoreML::default().build()` inside `OrtBackend::try_commit_session`.
   - Removed the silent CPU fallback when `ExecutionProvider::Cuda`, `ExecutionProvider::TensorRt`, or `ExecutionProvider::CoreML` is explicitly requested (`device = "cuda"` / `"tensorrt"` / `"coreml"` now returns `Err(InferenceError::Runtime(...))` if the provider cannot be initialized).
   - Preserved `ExecutionProvider::Auto` probing (`CUDA -> CoreML -> CPU`), which now accurately catches failed GPU probes via `.error_on_failure()` and reports `"onnxruntime-cpu"` when running on CPU.
3. **Wired `candle-cuda` and `metal` Features & Removed Silent Fallbacks (`Cargo.toml`, `src/backend/candle_backend.rs`)**:
   - Added `candle-cuda = ["cuda", "candle-backend", "candle-core?/cuda", "candle-nn?/cuda", "candle-transformers?/cuda"]` and `metal = ["candle-backend", "candle-core?/metal", "candle-nn?/metal", "candle-transformers?/metal"]` to `Cargo.toml`.
   - Updated `CandleBackend::new` so explicit `CandleDevice::Cuda(idx)` and `CandleDevice::Mlx` return `Err(InferenceError::Runtime(...))` instead of silently falling back to CPU when the feature or hardware is unavailable.
4. **Fixed Symlink Dereferencing in Packaging Pipelines (`.github/workflows/release.yml`, `Dockerfile`)**:
   - Replaced `cp -P` with `cp -L` (and `-maxdepth 1`) in `.github/workflows/release.yml` (Linux & macOS) and `Dockerfile` (all builder stages) so real `libonnxruntime_providers_*.so` / `.dylib` binaries are bundled instead of broken symlinks.
   - Updated the Windows release step in `.github/workflows/release.yml` to copy all top-level `*.dll` files from `target/x86_64-pc-windows-msvc/release` (including `DirectML.dll`).
5. **Fixed Python Worker `resolve_device("auto")` (`python/snoozeml/worker.py`)**:
   - Updated `resolve_device("auto")` to dynamically import `torch` via `importlib.import_module("torch")` if not yet loaded in `sys.modules`, ensuring `"cuda"` / `"mps"` is detected even when `snooze_model.py` imports `torch` lazily inside `load()`.
6. **Updated Documentation & Tests (`README.md`, `docs/src/content/docs/getting-started/installation.md`, `tests/backend_test.rs`)**:
   - Updated container and binary download size figures in `README.md` and `docs/src/content/docs/getting-started/installation.md` to reflect actual compressed sizes (`~52 MB` for `:slim`, `~83 MB` for `:cpu`, `~19 MB` for CPU binary archive).
   - Updated `tests/backend_test.rs` to verify that explicit unavailable device requests return `Err(InferenceError::Runtime(_))` rather than silently falling back to CPU.

---

## 3. What Is Left (GPU PC Verification & Next Steps)

When testing on your PC with an NVIDIA GPU, run through the following checklist:

### 3.1 Verify ONNX Runtime (`ort`) CUDA Execution & Error Reporting
1. **Check CUDA & cuDNN Runtime Libraries on the GPU Host**:
   - Because `ort-sys 2.0.0-rc.13` prebuilt binaries link against **CUDA 13** (`cublas64_13.dll` / `libcublas.so.13`, `cudart64_13.dll` / `libcudart.so.13`) and **cuDNN 9** (`cudnn64_9.dll` / `libcudnn.so.9`), ensure CUDA 13 + cuDNN 9 are on your `PATH` (Windows) or `LD_LIBRARY_PATH` (Linux).
   - Build and run with `--features cuda`:
     ```bash
     cargo build --release --features cuda
     ```
2. **Verify `device = "cuda"` Fails Loudly When Libraries Are Missing and Succeeds When Present**:
   - Set `device = "cuda"` in `models.toml` for an `.onnx` model and run `snooze load --model-id <id>`.
   - With `.error_on_failure()` now active, if any CUDA/cuDNN library is missing, `snooze` will immediately return a descriptive `500 Internal Server Error` with the exact missing library (`cudart64_13.dll` / `cudnn64_9.dll`) instead of pretending to run on `"onnxruntime-cuda"`.
   - Once CUDA 13 + cuDNN 9 are in `PATH` / `LD_LIBRARY_PATH`, confirm `snooze status` reports `onnxruntime-cuda` and `nvidia-smi` shows GPU VRAM allocation during model residency and reclamation to `0 MB` upon `snooze unload`.

### 3.2 Verify Native Candle Safetensors CUDA Execution (`--features candle-cuda`)
1. **Compile with `nvcc` (NVIDIA CUDA Toolkit)**:
   - On the GPU PC with the CUDA Toolkit (`nvcc`) installed, build with:
     ```bash
     cargo build --release --features candle-cuda
     ```
2. **Run the Safetensors GPU Equivalence Test**:
   ```bash
   cargo test --test safetensors_test --features candle-cuda -- --nocapture
   ```
   - Confirm `test_device_execution_cpu_and_cuda` prints `"CUDA GPU detected; running test on CUDA Device"` and passes numerical equivalence (`diff < 1e-4`) against CPU.

### 3.3 Decide on `Dockerfile` `cuda` Base Image & CI CUDA Toolkit Installation
- **Docker `cuda` Target (`Dockerfile`)**:
  - Currently, `FROM base-python AS cuda` uses `debian:trixie-slim`, which lacks `libcudart.so.13`, `libcublas.so.13`, and `libcudnn.so.9`. To make `ghcr.io/snoozeml/snoozeml:cuda` work out-of-the-box with `docker run --gpus all`, update the `cuda` runtime stage in `Dockerfile` to install the minimal CUDA 13 + cuDNN 9 runtime packages from NVIDIA's apt repository (or base the `cuda` runtime stage on an `nvidia/cuda` runtime image).
- **CI `candle-cuda` Compilation (`.github/workflows/release.yml`, `Dockerfile`)**:
  - Decide whether the prebuilt `snooze-linux-x86_64` / `snooze-windows-x86_64` release artifacts and `builder-cuda` Docker stage should also install the CUDA Toolkit (`nvcc`) in CI so `--features candle-cuda` can be enabled alongside `cuda,tensorrt`.
