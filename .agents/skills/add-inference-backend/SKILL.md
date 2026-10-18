---
name: add-inference-backend
description: >-
  Step-by-step runbook for adding a new model runtime backend or adding a new
  Hugging Face Safetensors architecture (Candle) to SnoozeML. Use this skill
  when implementing a new ModelBackend, extending LocalModelLoader /
  SupportedArchitecture in src/backend/, or adding support for new model file
  formats in src/config.rs.
---

# Adding an Inference Backend or Model Architecture

SnoozeML supports three paths for extending model execution in `src/backend/`:
1. **Path A — Adding a Safetensors / Candle Architecture**: Extending `LocalModelLoader` (`src/backend/local_loader.rs`) to support a new Hugging Face `model_type` (e.g., ModernBERT, Whisper, ResNet, Qwen2, or custom Safetensors plugins).
2. **Path B — Adding a New Native Runtime Engine**: Implementing the `ModelBackend` trait (`src/backend/mod.rs`) for a new compiled execution framework or model format in Rust.
3. **Path C — Adding a Custom Python Worker Model**: Packaging custom models (PyTorch, Chronos-Bolt, Scikit-learn, etc.) as process-isolated Python workers via `snooze_model.py` and `snooze_model.json`.

Detailed tensor and config patterns are documented in [references/backend-checklist.md](./references/backend-checklist.md).

---

## Path A: Adding a New Safetensors Architecture (`src/backend/local_loader.rs`)

When adding a new Hugging Face Safetensors model architecture powered by `candle-transformers` or `candle-nn`:

### Step 1: Register the Architecture Identifier
1. Add a documented variant to `SupportedArchitecture` in `src/backend/local_loader.rs`.
2. Update `SupportedArchitecture::from_hf_identifier(identifier: &str)` to match the lowercase Hugging Face `model_type` string(s).
3. Update `extract_architecture_string(val: &serde_json::Value)` in `src/backend/local_loader.rs` to map the PascalCase Hugging Face `architectures[0]` class name (e.g., `"BertModel"`, `"ViTForImageClassification"`) to the canonical lowercase identifier when `model_type` is omitted from `config.json`.

### Step 2: Add the `LoadedModel` Variant & Loader Branch
1. Add a variant to `LoadedModel` in `src/backend/local_loader.rs` holding the instantiated Candle model struct, `Option<tokenizers::Tokenizer>` (if text-based), and `candle_core::Device`.
2. Update `LoadedModel::device(&self)` and `LoadedModel::tokenizer(&self)` match arms.
3. In `LocalModelLoader::load(path: &Path, device: &Device)`, add the match arm for your `SupportedArchitecture` variant:
   - Parse `config_str` into the architecture's `Config` struct (map errors to `InferenceError::InvalidInput`).
   - Construct the model using the memory-mapped `VarBuilder` (`vb`) created via `VarBuilder::from_mmaped_safetensors` (map errors to `InferenceError::Runtime`).

### Step 3: Implement Forward Pass & Pooling
1. If it is a text encoder / embedding model, wire its forward pass in `LoadedModel::infer_text_embedding` and `LoadedModel::infer`, applying `mean_pooling` and `l2_normalize` as appropriate.
2. If it is a vision, audio, or sequence-classification model, validate input tensor rank and dimensions in `LoadedModel::infer` using `.first()` / `.get()` (never direct indexing `inputs[0]` or `shape[1]`), returning `InferenceError::ShapeMismatch` on rank mismatch.
3. Convert output `candle_core::Tensor` to `TensorPayload` via `tensor_to_payload(name, &output_tensor)`.

### Step 4: Verify in `tests/safetensors_test.rs`
1. Add unit and behavioral tests in `tests/safetensors_test.rs` constructing synthetic Safetensors weights (`safetensors::serialize_to_file`) and a matching `config.json` in an ephemeral temp directory.
2. Verify `SupportedArchitecture::from_hf_identifier`, `LocalModelLoader::load`, batched `infer` along `Axis(0)`, and shape mismatch error handling.

---

## Path B: Adding a New `ModelBackend` Runtime (`src/backend/`)

When adding an entirely new runtime engine (e.g., `src/backend/<name>_backend.rs`):

### Step 1: Feature Flag in `Cargo.toml`
1. Add optional dependencies in `Cargo.toml` with `optional = true`.
2. Define a feature flag `<name>-backend = ["dep:..."]` under `[features]` and include it in `default` if appropriate.

### Step 2: Implement `ModelBackend` (`src/backend/<name>_backend.rs`)
1. Create `src/backend/<name>_backend.rs` with module-level `//!` docs and `///` docstrings on all `pub` items.
2. Implement `pub trait ModelBackend: Send + Sync`:
   - `fn infer(&self, inputs: &[TensorPayload]) -> Result<Vec<TensorPayload>, InferenceError>`
   - `fn provider_name(&self) -> &'static str`
3. Ensure `infer` preserves the leading batch dimension `Axis(0)` (`inputs.first()?.batch_size()`) so `DynamicBatcher::split_outputs` can slice outputs back to individual callers.
4. Ensure any internal session mutex uses `std::sync::Mutex` with `.lock().map_err(...)` (never `.unwrap()`).

### Step 3: Wire Factory & Model Discovery
1. Expose the module in `src/backend/mod.rs` guarded by `#[cfg(feature = "<name>-backend")]`.
2. Add format detection and initialization in `create_backend(model_path: &str, provider_preference: Option<&str>)` with both `#[cfg(feature = "<name>-backend")]` and `#[cfg(not(feature = "<name>-backend"))]`.
3. Update `ModelDirectoryConfig::inspect_file` and `inspect_subdir` in `src/config.rs` so `snooze serve --models-dir`, `snooze init`, and `snooze models scan` automatically discover the new file extension.

### Step 4: Validation Commands
Always run all three checks before finishing:
```powershell
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
cargo check --no-default-features
```

---

## Path C: Adding a Custom Python Worker Model (`snooze_model.py` & `src/backend/worker_backend.rs`)

When native export (`.onnx`, native Safetensors, or `.xgb.json`) is not viable and the model requires arbitrary Python libraries (e.g. PyTorch, Hugging Face transformers, Amazon Chronos):

### Step 1: Create Model Directory & Entrypoint
1. Create a dedicated directory (e.g. `examples/my_model/` or `models/my_model/`).
2. Write `snooze_model.py` implementing the duck-typed interface:
   ```python
   class Model:
       def load(self, model_dir: str, device: str) -> None:
           # Initialize weights onto device ('cpu', 'cuda', 'mps')
           pass

       def predict(self, inputs: dict) -> dict:
           # Run inference and return dictionary of named tensors
           # e.g. {'output': np.ndarray or torch.Tensor}
           pass
   ```
3. Optionally add `snooze_model.json` to customize class name, timeouts, or environment variables:
   ```json
   {
     "entrypoint": "snooze_model.py",
     "class": "Model",
     "tensor_format": "numpy",
     "timeout_seconds": 30.0
   }
   ```

### Step 2: Smoke-Test with Worker CLI
Verify that weights load and produce expected output shapes without starting the daemon:
```powershell
python -m snoozeml.worker --check examples/my_model --input context:1x32 --device cpu
```

### Step 3: Verify Integration with SnoozeML Daemon
1. Launch daemon pointing to the model directory:
   ```powershell
   cargo run -- serve --models-dir examples/my_model
   ```
2. Verify inference, dynamic batching along `Axis(0)`, and scale-to-zero deallocation in `tests/worker_backend_test.rs`.

