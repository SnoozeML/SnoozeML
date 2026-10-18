//! Model controller managing dynamic execution, lazy rehydration, and scale-to-zero memory reclamation.

use crate::backend::{create_backend, ModelBackend};
use crate::batcher::{BatchResponder, DynamicBatcher, DynamicBatcherConfig};
use crate::types::{InferenceError, InferenceJob, ModelMetadata, ModelState, TensorPayload};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::{oneshot, RwLock};

/// Configuration parameters for defining a managed model.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Unique identifier for the model (e.g. `"resnet50"`, `"embeddings-v1"`).
    pub id: String,
    /// Filesystem path to the model weights or mock identifier.
    pub onnx_path: String,
    /// Inactivity threshold before dropping the session from memory.
    pub idle_timeout: Duration,
    /// Maximum dynamic batch size.
    pub max_batch_size: usize,
    /// Maximum delay before flushing a partial batch.
    pub max_batch_delay: Duration,
    /// Target execution provider preference (e.g. `"cpu"`, `"cuda"`, `"tensorrt"`).
    pub provider: Option<String>,
    /// Optional human-readable description of the model.
    pub description: Option<String>,
    /// Minimum duration model must remain resident before eligible for LRU deallocation.
    pub min_residency: Duration,
    /// Queue threshold before triggering replica autoscaling.
    pub autoscale_queue_threshold: usize,
    /// Optional upper limit on parallel replica instances.
    pub max_replicas: Option<usize>,
    /// Replica routing strategy.
    pub routing_strategy: crate::routing::RoutingStrategyType,
    /// Whether the model is pinned in memory and never automatically deallocated.
    pub pinned: bool,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            id: String::new(),
            onnx_path: String::new(),
            idle_timeout: Duration::from_secs(300),
            max_batch_size: 16,
            max_batch_delay: Duration::from_millis(10),
            provider: Some("auto".to_string()),
            description: None,
            min_residency: Duration::from_secs(10),
            autoscale_queue_threshold: 16,
            max_replicas: None,
            routing_strategy: crate::routing::RoutingStrategyType::LeastQueue,
            pinned: false,
        }
    }
}

/// Computes the effective scale-to-zero idle timeout for a model replica instance.
///
/// # Scaling Invariant
///
/// - The primary instance (`replica_index == 0`) strictly respects the configured
///   `base_idle_timeout`.
/// - Additional copies (`replica_index > 0`) scale down more rapidly as an inverse function
///   of the total number of loaded instances $N$. The more instances currently resident,
///   the shorter their allowed idle duration before deallocation:
///   $$\text{effective\_timeout} = \max\left(1\text{ms}, \frac{T_{\text{base}}}{N}\right)$$
///
/// This ensures rapid memory reclamation under declining load while guaranteeing that
/// the base model adheres to its configured residency threshold.
#[must_use]
pub fn compute_replica_idle_timeout(
    base_idle_timeout: Duration,
    total_instances: usize,
    replica_index: usize,
) -> Duration {
    if replica_index == 0 || total_instances <= 1 {
        base_idle_timeout
    } else {
        let divisor = u32::try_from(total_instances).unwrap_or(u32::MAX);
        let scaled = base_idle_timeout
            .checked_div(divisor)
            .unwrap_or(Duration::from_millis(1));
        scaled.max(Duration::from_millis(1))
    }
}

/// RAII guard decrementing an active request counter on drop.
struct ActiveRequestGuard<'a>(&'a std::sync::atomic::AtomicUsize);

impl<'a> Drop for ActiveRequestGuard<'a> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Controller responsible for single-model execution, dynamic micro-batching, and memory lifecycle.
pub struct ModelController {
    config: ModelConfig,
    replica_index: usize,
    pinned: AtomicBool,
    backend: Arc<RwLock<Option<Box<dyn ModelBackend>>>>,
    state: Arc<RwLock<ModelState>>,
    last_accessed: AtomicU64,
    loaded_at: AtomicU64,
    active_requests: Arc<std::sync::atomic::AtomicUsize>,
    batcher: DynamicBatcher,
    registry: std::sync::RwLock<Option<std::sync::Weak<crate::registry::ModelRegistryInner>>>,
}

impl std::fmt::Debug for ModelController {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelController")
            .field("id", &self.config.id)
            .field("replica_index", &self.replica_index)
            .field("pinned", &self.pinned.load(Ordering::Relaxed))
            .field("last_accessed", &self.last_accessed.load(Ordering::Relaxed))
            .field("queue_len", &self.queue_len())
            .field("active_requests", &self.active_requests())
            .finish()
    }
}

impl ModelController {
    /// Creates and initializes a new primary `ModelController`.
    ///
    /// The model initially begins in the `Idle` state without loading weights into memory.
    #[must_use]
    pub fn new(config: ModelConfig) -> Arc<Self> {
        Self::new_replica(config, 0)
    }

    /// Creates and initializes a `ModelController` with an explicit replica index.
    #[must_use]
    pub fn new_replica(config: ModelConfig, replica_index: usize) -> Arc<Self> {
        let backend: Arc<RwLock<Option<Box<dyn ModelBackend>>>> = Arc::new(RwLock::new(None));
        let state = Arc::new(RwLock::new(ModelState::Idle));
        let backend_clone = backend.clone();
        let state_clone = state.clone();

        let batcher_config = DynamicBatcherConfig {
            max_batch_size: config.max_batch_size,
            max_delay: config.max_batch_delay,
            queue_capacity: 1024,
        };

        let run_fn = move |inputs: Vec<TensorPayload>, responders: Vec<BatchResponder>| {
            let backend_lock = backend_clone.clone();
            let state_lock = state_clone.clone();

            tokio::task::spawn_blocking(move || {
                // Update state to Busy during active inference
                {
                    let mut st = state_lock.blocking_write();
                    *st = ModelState::Busy;
                }

                let guard = backend_lock.blocking_read();
                let has_backend = guard.is_some();
                let infer_result = if let Some(ref active_backend) = *guard {
                    active_backend.infer(&inputs)
                } else {
                    Err(InferenceError::Runtime(
                        "Model backend is not loaded in memory".into(),
                    ))
                };
                drop(guard);

                // Restore state before waking up awaiting callers
                {
                    let mut st = state_lock.blocking_write();
                    *st = if has_backend {
                        ModelState::Ready
                    } else {
                        ModelState::Idle
                    };
                }

                match infer_result {
                    Ok(batched_outputs) => {
                        if responders.len() == 1 {
                            if let Some((tx, _)) = responders.into_iter().next() {
                                let _ = tx.send(Ok(batched_outputs));
                            }
                        } else {
                            let slice_sizes: Vec<usize> =
                                responders.iter().map(|(_, s)| *s).collect();
                            match DynamicBatcher::split_outputs(&batched_outputs, &slice_sizes) {
                                Ok(split_client_outputs) => {
                                    for ((tx, _), client_tensors) in
                                        responders.into_iter().zip(split_client_outputs)
                                    {
                                        let _ = tx.send(Ok(client_tensors));
                                    }
                                }
                                Err(split_err) => {
                                    for (tx, _) in responders {
                                        let _ = tx.send(Err(split_err.clone()));
                                    }
                                }
                            }
                        }
                    }
                    Err(infer_err) => {
                        for (tx, _) in responders {
                            let _ = tx.send(Err(infer_err.clone()));
                        }
                    }
                }
            });
        };

        let batcher = DynamicBatcher::spawn(batcher_config, run_fn);

        if replica_index > 0 {
            tracing::info!(
                model_id = %config.id,
                replica_index,
                "Initialized model replica controller"
            );
        }

        let is_primary_pinned = replica_index == 0 && config.pinned;

        Arc::new(Self {
            config,
            replica_index,
            pinned: AtomicBool::new(is_primary_pinned),
            backend,
            state,
            last_accessed: AtomicU64::new(Self::now_millis()),
            loaded_at: AtomicU64::new(0),
            active_requests: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            batcher,
            registry: std::sync::RwLock::new(None),
        })
    }

    /// Returns the unique model identifier.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.config.id
    }

    /// Returns the replica index (0 for primary, >0 for additional copies).
    #[must_use]
    pub fn replica_index(&self) -> usize {
        self.replica_index
    }

    /// Returns whether this controller is an autoscaled replica rather than the primary instance.
    #[must_use]
    pub fn is_replica(&self) -> bool {
        self.replica_index > 0
    }

    /// Returns whether this model instance is pinned in memory and exempt from automatic deallocation.
    #[must_use]
    pub fn is_pinned(&self) -> bool {
        self.pinned.load(Ordering::Relaxed)
    }

    /// Sets the pinned status of this model instance without modifying its current load state.
    pub fn set_pinned(&self, pinned: bool) {
        self.pinned.store(pinned, Ordering::Relaxed);
    }

    /// Pins the model in memory, ensuring its weights are loaded first.
    ///
    /// While pinned, the model is exempt from both background scale-to-zero idle sweeps
    /// and capacity/OOM LRU eviction until explicitly unpinned or unloaded.
    ///
    /// # Errors
    /// Returns [`InferenceError`] if loading the model weights into memory fails.
    pub async fn pin(&self) -> Result<(), InferenceError> {
        self.ensure_loaded().await?;
        self.set_pinned(true);
        tracing::info!(
            model_id = %self.config.id,
            "Model pinned in memory"
        );
        Ok(())
    }

    /// Unpins the model, allowing normal scale-to-zero idle timeouts and LRU eviction to apply.
    pub fn unpin(&self) {
        let was_pinned = self.pinned.swap(false, Ordering::Relaxed);
        if was_pinned {
            tracing::info!(
                model_id = %self.config.id,
                "Model unpinned from memory"
            );
        }
    }

    /// Returns the number of jobs waiting in this controller's batcher queue.
    #[must_use]
    pub fn queue_len(&self) -> usize {
        self.batcher.queue_len()
    }

    /// Returns the number of currently active in-flight requests (queued + executing).
    #[must_use]
    pub fn active_requests(&self) -> usize {
        self.active_requests.load(Ordering::Relaxed)
    }

    /// Returns a reference to the model's configuration.
    #[must_use]
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Returns whether the model backend is currently loaded in memory.
    pub async fn is_loaded(&self) -> bool {
        self.backend.read().await.is_some()
    }

    /// Returns whether the model is actively processing inference or accumulating a batch.
    pub async fn is_busy(&self) -> bool {
        let state = *self.state.read().await;
        state == ModelState::Busy
            || state == ModelState::Loading
            || self.active_requests.load(Ordering::Relaxed) > 0
            || self.batcher.queue_len() > 0
    }

    /// Returns the system epoch timestamp in milliseconds when this model was last accessed.
    #[must_use]
    pub fn last_accessed_millis(&self) -> u64 {
        self.last_accessed.load(Ordering::Relaxed)
    }

    /// Returns the duration this model session has continuously resided in memory, or `None` if idle.
    #[must_use]
    pub fn residency_duration(&self) -> Option<Duration> {
        let loaded = self.loaded_at.load(Ordering::Relaxed);
        if loaded == 0 {
            None
        } else {
            Some(Duration::from_millis(
                Self::now_millis().saturating_sub(loaded),
            ))
        }
    }

    /// Returns the configured minimum residency duration before this model is eligible for LRU deallocation.
    #[must_use]
    pub fn min_residency(&self) -> Duration {
        self.config.min_residency
    }

    /// Associates this controller with a parent registry's allocation supervisor.
    pub fn set_registry(&self, registry: std::sync::Weak<crate::registry::ModelRegistryInner>) {
        if let Ok(mut guard) = self.registry.write() {
            *guard = Some(registry);
        }
    }

    /// Returns an upgraded strong reference to the parent registry inner state, if registered.
    #[must_use]
    pub(crate) fn registry_inner(&self) -> Option<Arc<crate::registry::ModelRegistryInner>> {
        self.registry
            .read()
            .ok()
            .and_then(|guard| guard.as_ref().and_then(|w| w.upgrade()))
    }

    /// Submits inputs for batched inference, ensuring the model is loaded first.
    ///
    /// # Errors
    /// Returns [`InferenceError`] if model rehydration, queuing, or execution fails.
    pub async fn infer(
        &self,
        inputs: Vec<TensorPayload>,
    ) -> Result<Vec<TensorPayload>, InferenceError> {
        self.touch();
        self.active_requests.fetch_add(1, Ordering::SeqCst);
        let _active_guard = ActiveRequestGuard(&self.active_requests);

        self.ensure_loaded().await?;

        let (tx, rx) = oneshot::channel();
        let job = InferenceJob {
            inputs,
            response_tx: tx,
            submitted_at: std::time::Instant::now(),
        };

        self.batcher.submit(job).await?;

        rx.await
            .map_err(|_| InferenceError::Runtime("Inference response channel dropped".into()))?
    }

    /// Verifies the model backend is loaded into memory, rehydrating it from storage if idle.
    ///
    /// If registered with a [`crate::registry::ModelRegistry`], coordinates with the registry's
    /// allocation supervisor to reactively deallocate LRU models on out-of-memory errors.
    ///
    /// # Errors
    /// Returns [`InferenceError::OutOfMemory`], [`InferenceError::ResourceExhausted`],
    /// [`InferenceError::Runtime`], or [`InferenceError::NotFound`] if weight loading fails.
    pub async fn ensure_loaded(&self) -> Result<(), InferenceError> {
        if self.is_loaded().await {
            return Ok(());
        }
        if self.is_replica() {
            self.load_backend_internal().await
        } else if let Some(reg) = self.registry_inner() {
            reg.ensure_loaded_with_deallocation(&self.config.id).await
        } else {
            self.load_backend_internal().await
        }
    }

    /// Loads the model backend into memory without triggering deallocation supervisor loops.
    ///
    /// # Errors
    /// Returns [`InferenceError::OutOfMemory`], [`InferenceError::Runtime`], or [`InferenceError::NotFound`].
    pub async fn load_backend_internal(&self) -> Result<(), InferenceError> {
        {
            let read_guard = self.backend.read().await;
            if read_guard.is_some() {
                return Ok(());
            }
        }

        let mut write_guard = self.backend.write().await;
        if write_guard.is_none() {
            {
                let mut st = self.state.write().await;
                *st = ModelState::Loading;
            }

            tracing::info!(
                model_id = %self.config.id,
                path = %self.config.onnx_path,
                "Loading model backend into memory"
            );

            let onnx_path = self.config.onnx_path.clone();
            let provider = self.config.provider.clone();
            let backend_res = tokio::task::spawn_blocking(move || {
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    create_backend(&onnx_path, provider.as_deref())
                }))
            })
            .await
            .map_err(|e| {
                InferenceError::Runtime(format!("Backend initialization task failed: {e}"))
            })?;

            let new_backend = match backend_res {
                Ok(Ok(backend)) => backend,
                Ok(Err(err)) => {
                    tracing::error!(
                        model_id = %self.config.id,
                        path = %self.config.onnx_path,
                        error = %err,
                        "Failed to load model backend into memory; reverting state to Idle"
                    );
                    let mut st = self.state.write().await;
                    *st = ModelState::Idle;
                    return Err(err);
                }
                Err(panic_payload) => {
                    let mut st = self.state.write().await;
                    *st = ModelState::Idle;
                    let msg = if let Some(s) = panic_payload.downcast_ref::<&str>() {
                        (*s).to_string()
                    } else if let Some(s) = panic_payload.downcast_ref::<String>() {
                        s.clone()
                    } else {
                        "Unknown panic during backend initialization".to_string()
                    };
                    tracing::error!(
                        model_id = %self.config.id,
                        panic_msg = %msg,
                        "Panic occurred while loading model backend; reverting state to Idle"
                    );

                    let lower = msg.to_lowercase();
                    if lower.contains("bad_alloc")
                        || lower.contains("out of memory")
                        || lower.contains("cuda failure 2")
                    {
                        return Err(InferenceError::OutOfMemory(format!(
                            "Panic during backend initialization due to memory exhaustion: {msg}"
                        )));
                    }

                    return Err(InferenceError::Runtime(format!(
                        "Panic during backend initialization: {msg}"
                    )));
                }
            };

            *write_guard = Some(new_backend);
            self.loaded_at.store(Self::now_millis(), Ordering::Relaxed);

            {
                let mut st = self.state.write().await;
                *st = ModelState::Ready;
            }
        }

        Ok(())
    }

    /// Explicitly forces the model weights to be deallocated from memory.
    pub async fn unload(&self) {
        let mut write_guard = self.backend.write().await;
        if write_guard.is_some() {
            tracing::info!(
                model_id = %self.config.id,
                "Explicitly unloading model from memory"
            );
            *write_guard = None;
        }
        self.loaded_at.store(0, Ordering::Relaxed);
        let mut st = self.state.write().await;
        *st = ModelState::Idle;
    }

    /// Evaluates model inactivity and releases weights if idle duration exceeds threshold.
    ///
    /// Models that are currently pinned (`is_pinned() == true`) or actively processing requests are exempt from idle deallocation.
    pub async fn deallocate_if_idle(&self) {
        if self.is_pinned() || self.is_busy().await {
            return;
        }

        let last = self.last_accessed.load(Ordering::Relaxed);
        let elapsed = Duration::from_millis(Self::now_millis().saturating_sub(last));

        if elapsed >= self.config.idle_timeout {
            let mut write_guard = self.backend.write().await;
            if write_guard.is_some() {
                tracing::info!(
                    model_id = %self.config.id,
                    elapsed_secs = elapsed.as_secs(),
                    timeout_secs = self.config.idle_timeout.as_secs(),
                    "Model exceeded idle timeout; deallocating from memory (scale-to-zero)"
                );
                *write_guard = None;
                self.loaded_at.store(0, Ordering::Relaxed);
                let mut st = self.state.write().await;
                *st = ModelState::Idle;
                if let Some(reg) = self.registry_inner() {
                    reg.record_deallocation("idle");
                }
            }
        }
    }

    /// Returns telemetry and metadata describing the model's current operational state.
    pub async fn metadata(&self) -> ModelMetadata {
        let last = self.last_accessed.load(Ordering::Relaxed);
        let elapsed_millis = Self::now_millis().saturating_sub(last);
        let elapsed_secs = elapsed_millis / 1000;
        let state = *self.state.read().await;

        ModelMetadata {
            id: self.config.id.clone(),
            state,
            onnx_path: self.config.onnx_path.clone(),
            idle_timeout_secs: self.config.idle_timeout.as_secs(),
            max_batch_size: self.config.max_batch_size,
            max_batch_delay_ms: u64::try_from(self.config.max_batch_delay.as_millis())
                .unwrap_or(u64::MAX),
            last_accessed_secs_ago: elapsed_secs,
            description: self.config.description.clone(),
            pinned: self.is_pinned(),
        }
    }

    /// Updates the last accessed timestamp to the current system epoch millisecond.
    pub fn touch(&self) {
        self.last_accessed
            .store(Self::now_millis(), Ordering::Relaxed);
    }

    fn now_millis() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
            .unwrap_or(0)
    }
}
