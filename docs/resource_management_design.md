# Design Document: Dynamic Resource Management & Reactive OOM Deallocation

* **Status:** Proposed
* **Author:** SnoozeML Systems Engineering
* **Date:** 2026-09-23
* **Target Components:** `src/registry.rs`, `src/model.rs`, `src/backend/`, `src/types.rs`, `src/server/`

---

## 1. Executive Summary & Objective

SnoozeML currently employs **passive, timer-based scale-to-zero** deallocation. If a model remains inactive beyond its `idle_timeout`, a background sweeper drops its session from memory. However, if a new model is requested when host RAM or GPU VRAM is saturated, the backend initialization crashes or returns an Out-Of-Memory (OOM) error without attempting to free idle resources.

This document proposes a **reactive, opportunistic resource manager** that:
1. Intercepts backend allocation failures across all execution providers (ONNX Runtime, Candle/Safetensors, CUDA/CPU).
2. Uses **strict, high-confidence criteria** to detect Out-Of-Memory (OOM) conditions while preventing generic runtime exceptions from triggering deallocation loops.
3. Automatically identifies and unloads Least Recently Used (LRU) idle model instances from RAM/VRAM.
4. Retries model allocation iteratively until the model successfully loads or all deallocatable candidates are exhausted.
5. Accounts for future multi-instance / replica scaling where a single model identifier may map to multiple allocations.
6. Prevents concurrent loading stampedes and cache thrashing without requiring invasive low-level GPU byte-tracking libraries (like NVML).

---

## 2. Current Architecture & Problem Statement

### 2.1 Existing Flow
```
Client Request ---> ModelController::infer()
                          │
                          ▼
                 ensure_loaded()
                          │
            [Is backend present in RAM/VRAM?]
                  ├── Yes ──> Submit to DynamicBatcher
                  └── No  ──> create_backend()
                                   │
                                   ├── Success: Store backend & run
                                   └── OOM Error: Fails immediately! (Returns 500 Internal Server Error)
```

### 2.2 Shortcomings
* **Brittle Multitenancy:** If Model A and Model B both fit in memory, but Model C needs Model A's space, the server fails with an error even if Model A has finished executing and is completely idle.
* **Passive Only:** Scale-to-zero only triggers after `idle_timeout` expires (e.g. 5 minutes). Rapid spikes in diverse model invocations cause immediate allocation failures.
* **Lack of Loading Concurrency Control:** Simultaneous cold-start requests for multiple models can simultaneously attempt heavy allocations, causing thrashing or cascading OOMs.

---

## 3. Core Design Decisions

### Decision 1: Extend `ModelRegistry` vs. Create New `ResourceManager`
* **Option A: Standalone `ResourceManager` wrapping `ModelRegistry`.**
  * *Pros:* Clear separation of concerns.
  * *Cons:* Introduces extra wrapper indirection and synchronization layers between registry lookups and controller operations, violating our *Rule of Two* and *Anti-Boilerplate* guidelines.
* **Option B: Extend `ModelRegistry` with Allocation Coordination (Recommended).**
  * *Pros:* `ModelRegistry` already holds references to model instances, tracks access timestamps, and manages discovery. Extending it with an allocation supervisor keeps logic local and avoids premature abstraction.

### Decision 2: Reactive Deallocation on OOM vs. Active VRAM Tracking
* **Hardware Telemetry (NVML / ROCm / OS Malloc APIs):**
  * Requires platform-specific C bindings, dynamic loading of `libnvidia-ml.so`/`nvml.dll`, and has high complexity across macOS (Metal/CoreML), Linux (CUDA/ROCm), and Windows (DirectML).
* **Reactive Allocation with OOM Interception (Recommended):**
  * **Zero external dependencies.**
  * Only intervenes when an allocation actually fails with an out-of-memory error.
  * Operates reliably across all backends (CPU host memory, CUDA, Apple Silicon unified memory).

---

## 4. Architecture & Data Structures

```
                           ┌──────────────────────────────────────────────┐
                           │               ModelRegistry                  │
                           │  - models / instance allocations             │
                           │  - load_lock: tokio::sync::Mutex<()>         │
                           │  - max_loaded_models: Option<usize>          │
                           └──────────────────────┬───────────────────────┘
                                                  │
                       Coordinates Reactive Load  │
                                                  ▼
                         ┌──────────────────────────────────┐
                         │      Allocation Supervisor       │
                         │                                  │
                         │  1. Acquire `load_lock`          │
                         │  2. Attempt `backend::create()`  │
                         │  3. Strictly Verify OOM?         │
                         │     ├── True OOM: Deallocate LRU │
                         │     │             Retry alloc    │
                         │     └── Other/Err: Return error  │
                         └──────────────────────────────────┘
```

### 4.1 Registry Data Structure & Future Horizontal Scaling

In the initial implementation, each registered model has a single controller. However, SnoozeML may support **horizontal scaling / multiple allocations of the same model** (e.g. running 2 or 3 parallel backend instances for high-throughput concurrency or multi-GPU placement).

To accommodate this future evolution cleanly:
* Deallocation candidate selection must operate on **individual model allocations / instances**, not simply on the model ID.
* When scaling is implemented, if Model A has 3 active allocations and memory is needed, the supervisor can deallocate *one* idle replica of Model A rather than purging the entire model.

```rust
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};

/// Enhanced registry managing model controllers, access history, and allocation coordination.
#[derive(Debug, Clone)]
pub struct ModelRegistry {
    /// Active controllers indexed by model ID.
    /// Note: Designed to evolve into `HashMap<String, Vec<Arc<ModelController>>>`
    /// when multi-instance replica scaling of the same model is enabled.
    models: Arc<RwLock<HashMap<String, Arc<ModelController>>>>,
    /// Serialization lock ensuring only one model allocation is attempted at a time.
    /// Prevents concurrent allocation stampedes and conflicting deallocations.
    load_lock: Arc<Mutex<()>>,
    /// Optional hard cap on concurrently resident model instances (preemptive deallocation).
    max_loaded_models: Option<usize>,
}
```

### 4.2 Strict Error Classification: Avoiding False OOMs

> [!CAUTION]
> **Never Treat Generic Exceptions as OOM**
> If a model fails to load due to corrupted weights, missing files, unsupported operators, or invalid tensor dimensions, it must **never** be treated as an OOM.
> Treating generic exceptions as OOM would cause the server to erroneously deallocate healthy, running idle models in a fruitless attempt to make room for an unloadable model!

To guarantee safety, error classification must be **conservative and high-confidence**:
* **Candle:** Direct inspection of `candle_core::Error::OutOfMemory` or CUDA driver error `CUDA_ERROR_OUT_OF_MEMORY` (code 2).
* **ONNX Runtime:** Specific match on `std::bad_alloc`, allocator OOM return codes, or explicit driver OOM messages (`"CUDA failure 2: out of memory"`).
* **Default Fallback:** If an error is ambiguous or cannot be proven to be an out-of-memory condition, **do not deallocate**. Bubble the error up immediately.

```rust
#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum InferenceError {
    // ... existing variants ...

    /// Allocation failed due to host RAM or device VRAM exhaustion.
    #[error("Device/System out of memory: {0}")]
    OutOfMemory(String),

    /// All deallocatable models have been purged, but insufficient memory remains.
    #[error("Resource limit reached: cannot allocate model even after deallocating idle sessions: {0}")]
    ResourceExhausted(String),
}

impl InferenceError {
    /// Strictly verifies whether an error represents a genuine out-of-memory condition.
    /// Returns `false` for generic runtime errors, invalid inputs, or model format errors.
    #[must_use]
    pub fn is_out_of_memory(&self) -> bool {
        match self {
            Self::OutOfMemory(_) => true,
            Self::Runtime(msg) => {
                let lower = msg.to_lowercase();

                // 1. Explicit C++ std::bad_alloc
                if lower.contains("bad_alloc") || lower.contains("std::bad_alloc") {
                    return true;
                }

                // 2. Explicit CUDA driver OOM codes
                if lower.contains("cuda failure 2") // CUDA_ERROR_OUT_OF_MEMORY
                    || lower.contains("cuda error: out of memory")
                    || lower.contains("cudamalloc failed")
                    || lower.contains("cuda_error_out_of_memory")
                {
                    return true;
                }

                // 3. Explicit device/host out-of-memory assertions
                if (lower.contains("out of memory") || lower.contains("ran out of memory"))
                    && !lower.contains("invalid") // Reject "invalid memory access"
                {
                    return true;
                }

                // Any generic message (e.g. "allocation failed for shape [0]", "ORT_FAIL", etc.)
                // is intentionally rejected to prevent false deallocations.
                false
            }
            _ => false,
        }
    }
}
```

#### Error Classification Matrix

| Error Scenario | Error Signature | Classified as OOM? | Action Taken |
| :--- | :--- | :--- | :--- |
| **CUDA VRAM Exhausted** | `CUDA failure 2: out of memory` | **Yes** | Deallocate LRU idle model & retry |
| **Host RAM Exhausted** | `std::bad_alloc` | **Yes** | Deallocate LRU idle model & retry |
| **Candle OOM** | `candle_core::Error::OutOfMemory` | **Yes** | Deallocate LRU idle model & retry |
| **Corrupted Safetensors** | `unexpected end of file / header mismatch` | **No** | Fail immediately (400/500), keep idle models |
| **Missing Model File** | `File not found: model.onnx` | **No** | Fail immediately (404), keep idle models |
| **Unsupported Architecture** | `Unsupported Safetensors architecture: 'mpt'` | **No** | Fail immediately (501), keep idle models |
| **Shape / Dimension Mismatch** | `Cannot reshape tensor of size 12 to [1, 16]` | **No** | Fail immediately (400), keep idle models |

---

### 4.3 LRU Candidate Selection

A model allocation/instance is a **deallocation candidate** if and only if:
1. It currently holds allocated weights in memory (`backend.is_some()`).
2. It does not belong to the request currently performing allocation.
3. Its state is `ModelState::Ready` or `ModelState::Idle` (i.e. **not `ModelState::Busy`** executing inference, and not `ModelState::Loading`).

Linear scanning across registered instances sorted by `last_accessed` millisecond timestamp is fast, allocation-lean, and avoids the lock-coupling of intrusive linked lists.

```rust
impl ModelRegistry {
    /// Identifies the best idle model instance to deallocate based on least-recent access time.
    pub async fn find_lru_deallocation_candidate(
        &self,
        target_model_id: &str,
    ) -> Option<Arc<ModelController>> {
        let guard = self.models.read().await;
        let mut candidates: Vec<Arc<ModelController>> = Vec::new();

        for (id, controller) in guard.iter() {
            if id == target_model_id {
                continue;
            }
            // Check if model instance has weights allocated in memory and is not in active inference
            if controller.is_loaded().await && !controller.is_busy().await {
                candidates.push(controller.clone());
            }
        }

        // Sort ascending by last_accessed timestamp (oldest first)
        candidates.sort_by_key(|c| c.last_accessed_millis());
        candidates.into_iter().next()
    }
}
```

---

## 5. Reactive Allocation & Deallocation Algorithm

When `ensure_loaded_with_deallocation()` is invoked:

```rust
impl ModelRegistry {
    /// Loads the target model, deallocating LRU idle instances one by one if a verified OOM error occurs.
    pub async fn ensure_loaded_with_deallocation(
        &self,
        target_model_id: &str,
    ) -> Result<(), InferenceError> {
        let controller = self.get(target_model_id).await?;
        if controller.is_loaded().await {
            controller.touch();
            return Ok(());
        }

        // 1. Serialize allocation to prevent deallocation races and stampedes
        let _guard = self.load_lock.lock().await;

        // Double-check after acquiring lock
        if controller.is_loaded().await {
            controller.touch();
            return Ok(());
        }

        // 2. Preemptive Deallocation: enforce optional max_loaded_models cap
        if let Some(max_models) = self.max_loaded_models {
            while self.count_loaded_models().await >= max_models {
                if let Some(victim) = self.find_lru_deallocation_candidate(target_model_id).await {
                    tracing::info!(
                        victim_id = %victim.id(),
                        limit = max_models,
                        "Preemptively deallocating idle model to respect max_loaded_models threshold"
                    );
                    victim.unload().await;
                } else {
                    break;
                }
            }
        }

        // 3. Reactive Deallocation Loop
        loop {
            match controller.load_backend_internal().await {
                Ok(()) => {
                    tracing::info!(model_id = %target_model_id, "Successfully loaded model backend");
                    controller.touch();
                    return Ok(());
                }
                // Only deallocate if strictly verified as out-of-memory
                Err(err) if err.is_out_of_memory() => {
                    tracing::warn!(
                        model_id = %target_model_id,
                        error = %err,
                        "Verified out-of-memory during model allocation; searching for deallocation candidate"
                    );

                    if let Some(victim) = self.find_lru_deallocation_candidate(target_model_id).await {
                        tracing::warn!(
                            victim_id = %victim.id(),
                            target_id = %target_model_id,
                            "Deallocating LRU model to reclaim memory/VRAM"
                        );
                        victim.unload().await;
                        // Yield to allow allocator / device driver to release pages
                        tokio::task::yield_now().await;
                    } else {
                        tracing::error!(
                            model_id = %target_model_id,
                            "OOM during allocation and no deallocatable idle models remaining"
                        );
                        return Err(InferenceError::ResourceExhausted(format!(
                            "Out of memory loading '{target_model_id}'. All active models are currently busy."
                        )));
                    }
                }
                // Any other error (corrupt file, unsupported op, etc.) fails immediately
                Err(fatal_err) => {
                    return Err(fatal_err);
                }
            }
        }
    }
}
```

---

## 6. Edge Cases & Safeguards

### 6.1 Thrashing Prevention (Anti-Thrashing)
* **Problem:** Model A and Model B take turns deallocating each other on consecutive requests (ping-pong effect), degrading throughput.
* **Mitigation:**
  * Add a `min_residency: Duration` (e.g. 5–10 seconds) to model configuration.
  * A deallocation candidate that was loaded less than `min_residency` ago is protected unless all other idle candidates have already been deallocated.
  * Emit a structured warning and increment telemetry metric `snoozeml_deallocation_thrashing_total`.

### 6.2 Never Deallocate Active Models
* Deallocation explicitly checks `ModelState::Busy` and the active request reference count.
* If a model is currently processing a batch or has requests awaiting in its `DynamicBatcher`, it is pinned and never eligible for deallocation.

### 6.3 Cleanup on Backend Panics
* Backend initialization in `ModelController::ensure_loaded` is wrapped in `std::panic::catch_unwind`.
* If a C++ runtime (such as ONNX Runtime) panics on memory allocation, the panic payload is inspected. If it indicates an allocation failure, it maps to `InferenceError::OutOfMemory`; otherwise, it maps to `InferenceError::Runtime` and aborts without deallocating other models.

---

## 7. HTTP & Observability Impact

### 7.1 Status Code Mapping
In `src/server/http.rs`:
* `InferenceError::ResourceExhausted` maps to `HTTP 503 Service Unavailable` with `Retry-After: 5` header, signaling to clients that the server is saturated with active models rather than failing permanently with `HTTP 500`.

### 7.2 Telemetry
* Metric `snoozeml_model_deallocations_total{model_id="...", reason="oom|idle|capacity"}`.
* Metric `snoozeml_model_load_retries_total{model_id="..."}`.
* Trace event: `tracing::info!(target = "snooze::lifecycle", action = "deallocate", victim = %victim_id, reason = "oom")`.

---

## 8. Configuration Extensions (`models.toml`)

Extend `GlobalModelDefaults` and `ModelEntryConfig` in `src/config.rs`:

```toml
[defaults]
idle_timeout_secs = 300
max_batch_size = 16
# Maximum number of simultaneously loaded model instances before LRU deallocation triggers preemptively
max_loaded_models = 3
# Minimum seconds a model must stay in memory before being eligible for LRU deallocation
min_residency_secs = 10
```

---

## 9. Summary Comparison

| Aspect | Current Architecture | Proposed Reactive Design |
| :--- | :--- | :--- |
| **Deallocation Trigger** | Exclusively timer-based (`idle_timeout`) | Timer-based **+** Reactive on verified OOM **+** Preemptive capacity cap |
| **OOM Classification** | Unhandled | **Strict & Conservative**: only verified allocator/driver OOM triggers deallocation; generic errors fail fast |
| **Multi-Instance Readiness**| Singleton model assumed | Candidate selector and deallocation loop operate on model allocations/instances |
| **Hardware Dependencies** | None | None (zero external NVML/C bindings needed) |
| **Concurrency Safety** | Independent per-model locks | Global allocation lock prevents concurrent loading storms |
| **Client Failure Behavior** | 500 Internal Server Error | 503 Service Unavailable with backoff headers |
