# Backend & Architecture Reference Checklist

## 1. `ModelBackend` Contract (`src/backend/mod.rs`)

```rust
pub trait ModelBackend: Send + Sync {
    fn infer(&self, inputs: &[TensorPayload]) -> Result<Vec<TensorPayload>, InferenceError>;
    fn provider_name(&self) -> &'static str;
}
```

### Batch Dimension Invariant (`Axis(0)`)
- `DynamicBatcher` concatenates `N` client requests along `Axis(0)` before calling `ModelBackend::infer`.
- If client A sends shape `[1, 3, 224, 224]` and client B sends `[2, 3, 224, 224]`, `ModelBackend::infer` receives a single `TensorPayload` of shape `[3, 3, 224, 224]`.
- Every output `TensorPayload` returned by `ModelBackend::infer` **must** have leading dimension `3` (`output.batch_size() == 3`) so `DynamicBatcher::split_outputs` can slice `[0..1]` for client A and `[1..3]` for client B.

## 2. Error Variant Selection in Backends

| Failure Condition | `InferenceError` Variant | HTTP Status | gRPC Code |
| :--- | :--- | :--- | :--- |
| Model file or `config.json` missing on disk | `InferenceError::NotFound(msg)` | `404` | `NOT_FOUND` |
| Empty inputs, bad JSON config, overflow | `InferenceError::InvalidInput(msg)` | `400` | `INVALID_ARGUMENT` |
| Wrong input rank or feature count mismatch | `InferenceError::ShapeMismatch(msg)` | `400` | `INVALID_ARGUMENT` |
| Unrecognized HF `model_type` in Safetensors | `InferenceError::UnsupportedSafetensorsArchitecture(arch)` | `501` | `UNIMPLEMENTED` |
| CUDA / host RAM allocation failure | `InferenceError::OutOfMemory(msg)` | `503` | `RESOURCE_EXHAUSTED` |
| Engine execution or weight load error | `InferenceError::Runtime(msg)` | `500` | `INTERNAL` |

> **Important**: `InferenceError::is_out_of_memory()` (`src/types.rs`) inspects `InferenceError::OutOfMemory` and specific `InferenceError::Runtime` strings (`"bad_alloc"`, `"cuda failure 2"`, `"out of memory"`) to trigger reactive LRU model eviction and retry in `ModelRegistry::ensure_loaded_with_deallocation`.

## 3. Touchpoints When Adding a New File Extension
1. `src/backend/mod.rs`: `create_backend()` extension/provider dispatch.
2. `src/config.rs`:
   - `ModelDirectoryConfig::inspect_file()` — standalone file discovery.
   - `ModelDirectoryConfig::inspect_subdir()` — nested folder / Hugging Face repo discovery.
3. `src/cli/download.rs`: If downloaded from Hugging Face Hub, ensure default file filters include any required companion files.
4. `docs/src/serving/supported-formats.md`: Document the new format or architecture in the mdBook.
