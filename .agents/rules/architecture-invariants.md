---
trigger: always_on
---

# SnoozeML Architecture & Concurrency Invariants

When modifying core daemon components (`src/batcher.rs`, `src/model.rs`, `src/registry.rs`, `src/routing.rs`, `src/sweeper.rs`, `src/backend/`), you must preserve the following invariants:

---

## 1. Async Runtime & Blocking Execution Boundaries
* **Offload Blocking Inference (`spawn_blocking`):** All model weight loading and synchronous `ModelBackend::infer` forward passes (ONNX Runtime `ort`, Candle `candle-core`, and `hessboost` trees) MUST run inside `tokio::task::spawn_blocking` (or Rayon worker pools) so they never block Tokio async reactor threads.
* **Lock Ordering & Contention:**
  * Never hold a `tokio::sync::RwLock` or `Mutex` guard across an `.await` point if a nested call can re-acquire the same lock or `ModelRegistryInner::load_lock`.
  * `ModelRegistryInner::load_lock` serializes reactive capacity and OOM eviction (`ensure_loaded_with_deallocation`). Keep critical sections inside `load_lock` minimal and deterministic.
  * When accessing `std::sync::RwLock` or `std::sync::Mutex` (such as weak registry back-references or `OrtBackend::session`), never call `.unwrap()`; recover or map poison errors explicitly.

---

## 2. Dynamic Micro-Batching (`src/batcher.rs`)
* **Collation Along `Axis(0)`:** `DynamicBatcher::collate_inputs` concatenates request tensors along `Axis(0)` (the leading batch dimension) while requiring trailing dimensions (`shape.get(1..)`) and tensor names to match across all coalesced jobs.
* **Fast-Path Single Job:** When `jobs.len() == 1`, bypass tensor concatenation and output slicing altogether to avoid unnecessary memory allocations or float buffer copies.
* **Proportional Output Slicing:** `DynamicBatcher::split_outputs` slices each output tensor along `Axis(0)` according to each caller's original input batch size (`slice_size`), not assuming batch size `1`.
* **Fail-Safe Responder Notification:** Every code path inside `DynamicBatcher::flush` and the `run_batch_fn` closure MUST send a `Result` through every `oneshot::Sender` in `responders`. Dropping a responder without sending causes client requests to hang or fail with channel closed errors.

---

## 3. Model Lifecycle, Pinning & Differential Scale-to-Zero (`src/model.rs`, `src/registry.rs`, `src/sweeper.rs`)
* **State Machine (`ModelState`):** Transitions follow `Idle` (unloaded) $\rightarrow$ `Loading` $\rightarrow$ `Ready` $\leftrightarrow$ `Busy` $\rightarrow$ `Idle`. Always restore state to `Ready` (if backend is loaded) or `Idle` (if unloaded) after inference completes or errors.
* **Active Request Guard (`ActiveRequestGuard`):** Always increment `active_requests` via an RAII guard before checking load state or submitting to `DynamicBatcher`. A model or replica with `active_requests() > 0` or `queue_len() > 0` MUST NEVER be deallocated by idle sweeping or LRU eviction.
* **Pinning Semantics:**
  * Only the primary replica (`replica_index == 0`) inherits `config.pinned` at initialization or via `pin_model()`.
  * Surplus autoscaled replicas (`replica_index > 0`) MUST remain unpinned (`pinned = false`) so they can scale down when traffic subsides.
  * Pinned models are eagerly loaded on registration, skipped by `deallocate_if_idle()`, and excluded from `find_lru_deallocation_candidate()` (capacity and reactive OOM eviction).
  * Explicit user calls to `unload()` (`POST /v1/models/:id/unload` or `snooze unload`) ARE permitted to unload a pinned model.
* **Differential Replica Timeouts:** Use `compute_replica_idle_timeout(base_idle_timeout, total_instances, replica_index)`:
  * Primary replica (`replica_index == 0`): uses full `base_idle_timeout`.
  * Surplus replicas (`replica_index > 0`): use $\max(1\text{ms}, T_{\text{base}} / N)$ where $N$ is the count of currently loaded replicas, ensuring rapid scale-down after traffic spikes.
* **Anti-Thrashing (`min_residency`):** LRU capacity eviction (`find_lru_deallocation_candidate`) must respect `min_residency` unless `ignore_min_residency` is set during emergency OOM recovery.

---

## 4. Feature Flag Hygiene (`Cargo.toml`, `src/backend/`, `src/stream/`)
* SnoozeML supports optional compilation of backends (`ort-backend`, `candle-backend`, `tree-backend`, `worker-backend`), hardware accelerators (`cuda`, `tensorrt`, `coreml`), and event streaming brokers (`streams`, `streams-rabbitmq`, `streams-redis`, `streams-kafka`, `streams-all`).
* Always guard backend-specific and broker-specific imports, structs, and match arms with `#[cfg(feature = "...")]` and provide a clean `#[cfg(not(feature = "..."))]` fallback returning `InferenceError::Runtime` so `--no-default-features` and single-feature builds compile without dead-code or unused-variable warnings.

---

## 5. Custom Python Worker Isolation & Framing (`src/backend/worker_backend.rs`)
* **Process Boundary & Sandboxing:** The parent Rust daemon never executes untrusted Python code directly inside its process or Tokio worker threads. All execution is offloaded to a child process running `python -m snoozeml.worker`.
* **Zero-Copy / Single-Allocation Framing:** IPC frames follow the binary layout: 4-byte magic (`SNZ1` / `0x534E5A31`), 1-byte message type (`MSG_PING = 1`, `MSG_LOAD = 2`, `MSG_INFER = 3`, `MSG_RESULT = 4`, `MSG_SHUTDOWN = 5`), 4-byte JSON header length $L$, JSON header bytes ($L$), and payload length + raw tensor byte blocks. Never allocate intermediate buffers when streaming tensor bytes directly across stdin/stdout pipes.
* **Child Lifecycle & Clean Reaping:**
  - Upon deallocation, idle timeout, or daemon shutdown, send `MSG_SHUTDOWN`, wait with a grace period timeout, and reap with `SIGTERM` followed by `SIGKILL` fallback.
  - Reset `stderr_tail` on worker spawn so respawned workers do not report stale crash logs from previous terminations.

---

## 6. Streaming Concurrency & Invariants (`src/stream/`)
* **Decoupled Actor Tasks:** Each configured streaming consumer (`[[streams]]` in `models.toml`) runs in its own dedicated Tokio background task (`spawn_stream_consumer`). A failure or network disconnect on one stream must never crash the server, block the Tokio runtime, or impact HTTP/gRPC listeners.
* **Commit-After-Infer (At-Least-Once Delivery):** Message offsets or broker acknowledgements (`ack`) MUST only be committed after the model's forward pass completes and output payloads have been dispatched to the egress topic/queue. On inference error, reject/nack or route to dead-letter storage.
* **Queue Backpressure:** Respect `max_queue_size` when feeding incoming messages into `ModelController` / `DynamicBatcher`. When the internal batcher queue is saturated, pause broker polling or apply channel backpressure rather than dropping frames silently.

