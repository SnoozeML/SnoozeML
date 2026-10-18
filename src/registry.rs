//! Central registry for indexing, discovering, accessing, routing, and coordinating allocations
//! and autoscaling of active model controllers and replicas.

use crate::model::{compute_replica_idle_timeout, ModelConfig, ModelController};
use crate::routing::{create_routing_strategy, RoutingStrategy};
use crate::server::metrics::ServerMetrics;
use crate::types::{InferenceError, ModelMetadata};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::{Mutex, RwLock};

/// Internal shared state of the model registry.
#[derive(Debug)]
pub struct ModelRegistryInner {
    pub(crate) models: RwLock<HashMap<String, Arc<ModelReplicaGroup>>>,
    pub(crate) load_lock: Mutex<()>,
    pub(crate) max_loaded_models: RwLock<Option<usize>>,
    pub(crate) metrics: RwLock<Option<Arc<ServerMetrics>>>,
}

impl ModelRegistryInner {
    /// Records a model deallocation event on the configured metrics tracker.
    pub(crate) fn record_deallocation(&self, reason: &str) {
        if let Ok(guard) = self.metrics.try_read() {
            if let Some(ref m) = *guard {
                m.record_deallocation(reason);
            }
        }
    }

    /// Records an allocation retry attempt on the configured metrics tracker.
    pub(crate) fn record_load_retry(&self) {
        if let Ok(guard) = self.metrics.try_read() {
            if let Some(ref m) = *guard {
                m.record_load_retry();
            }
        }
    }

    /// Records an anti-thrashing event on the configured metrics tracker.
    pub(crate) fn record_thrashing(&self) {
        if let Ok(guard) = self.metrics.try_read() {
            if let Some(ref m) = *guard {
                m.record_thrashing();
            }
        }
    }

    /// Reactive loading supervisor invoked from [`crate::model::ModelController`].
    pub(crate) async fn ensure_loaded_with_deallocation(
        self: &Arc<Self>,
        target_model_id: &str,
    ) -> Result<(), InferenceError> {
        let registry = ModelRegistry {
            inner: self.clone(),
        };
        registry
            .ensure_loaded_with_deallocation(target_model_id)
            .await
    }
}

/// Group managing identical model replica instances, coordinating routing,
/// autoscaling, and differential scale-to-zero lifecycles.
pub struct ModelReplicaGroup {
    pub(crate) config: ModelConfig,
    pub(crate) replicas: RwLock<Vec<Arc<ModelController>>>,
    pub(crate) router: Box<dyn RoutingStrategy>,
    pub(crate) registry: std::sync::RwLock<Option<std::sync::Weak<ModelRegistryInner>>>,
}

impl std::fmt::Debug for ModelReplicaGroup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelReplicaGroup")
            .field("id", &self.config.id)
            .field("router", &self.router)
            .finish()
    }
}

impl ModelReplicaGroup {
    /// Creates a new `ModelReplicaGroup` with the specified primary controller.
    #[must_use]
    pub fn new(config: ModelConfig, primary: Arc<ModelController>) -> Self {
        let router = create_routing_strategy(config.routing_strategy);
        Self {
            config,
            replicas: RwLock::new(vec![primary]),
            router,
            registry: std::sync::RwLock::new(None),
        }
    }

    /// Creates a new `ModelReplicaGroup` with an explicit custom routing strategy.
    #[must_use]
    pub fn with_router(
        config: ModelConfig,
        primary: Arc<ModelController>,
        router: Box<dyn RoutingStrategy>,
    ) -> Self {
        Self {
            config,
            replicas: RwLock::new(vec![primary]),
            router,
            registry: std::sync::RwLock::new(None),
        }
    }

    /// Sets the parent registry inner reference for allocation coordination.
    pub fn set_registry(&self, registry: std::sync::Weak<ModelRegistryInner>) {
        if let Ok(mut guard) = self.registry.write() {
            *guard = Some(registry);
        }
    }

    /// Returns the unique model identifier for this group.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.config.id
    }

    /// Returns a reference to the group's model configuration.
    #[must_use]
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Returns the primary controller (replica index 0).
    ///
    /// # Errors
    /// Returns [`InferenceError::NotFound`] if the replica group has no registered controllers.
    pub async fn primary(&self) -> Result<Arc<ModelController>, InferenceError> {
        let guard = self.replicas.read().await;
        guard.first().cloned().ok_or_else(|| {
            InferenceError::NotFound(format!(
                "Model '{}' has no registered replicas",
                self.config.id
            ))
        })
    }

    /// Returns snapshots of all currently registered replica controllers.
    pub async fn all_replicas(&self) -> Vec<Arc<ModelController>> {
        let guard = self.replicas.read().await;
        guard.clone()
    }

    /// Returns the total count of active replicas currently allocated.
    pub async fn count_replicas(&self) -> usize {
        let guard = self.replicas.read().await;
        guard.len()
    }

    /// Returns the count of replicas currently loaded in memory.
    pub async fn count_loaded_replicas(&self) -> usize {
        let guard = self.replicas.read().await;
        let mut count = 0usize;
        for r in guard.iter() {
            if r.is_loaded().await {
                count = count.saturating_add(1);
            }
        }
        count
    }

    /// Unloads all replicas in this group from memory.
    pub async fn unload_all(&self) {
        let guard = self.replicas.read().await;
        for replica in guard.iter() {
            replica.unload().await;
        }
    }

    /// Appends a replica controller to this model's replica pool.
    pub async fn add_replica(&self, replica: Arc<ModelController>) {
        let mut guard = self.replicas.write().await;
        guard.push(replica);
    }

    /// Selects an optimal replica for inference.
    ///
    /// ## Routing Mechanism & Autoscaling Trigger
    ///
    /// 1. Inspects active replicas using the configured [`RoutingStrategy`] (default: [`LeastQueueRouting`]).
    ///    The candidate with the lowest pending queue is selected.
    /// 2. If the candidate's pending queue backlog (`queue_len` or in-flight executions) meets or exceeds
    ///    `autoscale_queue_threshold`, the requests exceed the queue capacity of all current replicas.
    /// 3. If resources are available (system capacity limits and RAM/VRAM availability permits),
    ///    a new model replica instance is automatically spun up, loaded into memory, and returned.
    /// 4. If resources are exhausted or allocation fails, it safely falls back to the lowest-queue
    ///    existing replica without failing the client request.
    ///
    /// # Errors
    /// Returns [`InferenceError::NotFound`] if no replicas are registered.
    pub async fn route_for_inference(
        &self,
        registry: &ModelRegistry,
    ) -> Result<Arc<ModelController>, InferenceError> {
        // 1. Initial selection under read lock
        let candidate = {
            let guard = self.replicas.read().await;
            if guard.len() <= 1 {
                guard.first().cloned().ok_or_else(|| {
                    InferenceError::NotFound(format!(
                        "Model '{}' has no replicas available",
                        self.config.id
                    ))
                })?
            } else {
                self.router.select_replica(&guard).cloned().ok_or_else(|| {
                    InferenceError::NotFound(format!(
                        "Model '{}' has no replicas available",
                        self.config.id
                    ))
                })?
            }
        };

        // 2. Evaluate queue depth on candidate
        let queue_backlog = candidate.queue_len().max(candidate.active_requests());
        let threshold = self.config.autoscale_queue_threshold.max(1);

        if queue_backlog < threshold {
            return Ok(candidate);
        }

        // 3. Queue exceeded on all existing replicas: check resource availability
        tracing::debug!(
            model_id = %self.config.id,
            queue_backlog,
            threshold,
            "Requests exceed queue threshold on all replicas; evaluating autoscaling capacity"
        );

        // Pre-check max_replicas ceiling (None or 0 represents unbounded)
        if let Some(max_rep) = self.config.max_replicas {
            if max_rep > 0 {
                let current_count = self.count_replicas().await;
                if current_count >= max_rep {
                    tracing::debug!(
                        model_id = %self.config.id,
                        current_count,
                        max_rep,
                        "Model reached configured max_replicas ceiling; routing to existing replica"
                    );
                    return Ok(candidate);
                }
            }
        }

        // Pre-check max_loaded_models capacity
        if let Some(max_loaded) = registry.max_loaded_models().await {
            let total_loaded = registry.count_loaded_models().await;
            if total_loaded >= max_loaded {
                // If capacity limit reached, verify if an idle victim exists before attempting scale
                if registry
                    .find_lru_deallocation_candidate(&self.config.id)
                    .await
                    .is_none()
                {
                    tracing::debug!(
                        model_id = %self.config.id,
                        total_loaded,
                        max_loaded,
                        "Capacity limit reached with no deallocatable idle models; skipping autoscale"
                    );
                    return Ok(candidate);
                }
            }
        }

        // 4. Acquire registry load lock to serialize replica allocations and prevent stampedes
        let _load_guard = registry.inner.load_lock.lock().await;

        // Double check lowest queue under load lock
        {
            let guard = self.replicas.read().await;
            if let Some(best) = self.router.select_replica(&guard) {
                let best_backlog = best.queue_len().max(best.active_requests());
                if best_backlog < threshold {
                    return Ok(best.clone());
                }
            }
        }

        // Double check max_replicas under write lock
        let new_index = {
            let guard = self.replicas.read().await;
            if let Some(max_rep) = self.config.max_replicas {
                if max_rep > 0 && guard.len() >= max_rep {
                    return Ok(candidate);
                }
            }
            guard.len()
        };

        // 5. Spin up another copy of the model
        tracing::info!(
            model_id = %self.config.id,
            replica_index = new_index,
            "Autoscaling: spinning up additional model replica due to queue saturation"
        );

        let new_controller = ModelController::new_replica(self.config.clone(), new_index);
        new_controller.set_registry(Arc::downgrade(&registry.inner));

        // Attempt loading backend weights into memory
        match new_controller.load_backend_internal().await {
            Ok(()) => {
                let mut guard = self.replicas.write().await;
                guard.push(new_controller.clone());
                let total = guard.len();
                tracing::info!(
                    model_id = %self.config.id,
                    replica_index = new_index,
                    total_replicas = total,
                    "Successfully autoscaled and loaded new model replica"
                );
                Ok(new_controller)
            }
            Err(err) if err.is_out_of_memory() => {
                // If verified OOM occurs, attempt to reclaim an idle victim once
                tracing::warn!(
                    model_id = %self.config.id,
                    error = %err,
                    "OOM during replica autoscale; attempting to reclaim idle victim"
                );

                if let Some(victim) = registry
                    .find_lru_deallocation_candidate(&self.config.id)
                    .await
                {
                    registry.inner.record_deallocation("oom_replica_autoscale");
                    victim.unload().await;
                    tokio::task::yield_now().await;

                    if new_controller.load_backend_internal().await.is_ok() {
                        let mut guard = self.replicas.write().await;
                        guard.push(new_controller.clone());
                        return Ok(new_controller);
                    }
                }

                tracing::warn!(
                    model_id = %self.config.id,
                    "Insufficient memory to autoscale replica; falling back to lowest-queue replica"
                );
                Ok(candidate)
            }
            Err(other_err) => {
                tracing::warn!(
                    model_id = %self.config.id,
                    error = %other_err,
                    "Failed to initialize autoscaled replica; falling back to lowest-queue replica"
                );
                Ok(candidate)
            }
        }
    }

    /// Sweeps idle replicas in this group according to the differential scale-to-zero mechanism.
    ///
    /// - Primary replica (index 0) strictly respects the configured `idle_timeout`.
    /// - Additional copies (indices $k \ge 1$) deallocate more rapidly as a function of the
    ///   total number of replicas ($T_{\text{base}} / N$).
    pub async fn sweep_scale_to_zero(&self, registry: &ModelRegistryInner) {
        let mut guard = self.replicas.write().await;
        let total_instances = guard.len();
        let base_timeout = self.config.idle_timeout;

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
            .unwrap_or(0);

        let mut to_retain: Vec<Arc<ModelController>> = Vec::with_capacity(total_instances);

        for (idx, controller) in guard.iter().enumerate() {
            if idx == 0 {
                // Primary replica respects the configured idle timeout
                controller.deallocate_if_idle().await;
                to_retain.push(controller.clone());
            } else {
                // Additional replica: use scaled idle timeout
                let replica_timeout =
                    compute_replica_idle_timeout(base_timeout, total_instances, idx);
                let last = controller.last_accessed_millis();
                let idle_duration = Duration::from_millis(now.saturating_sub(last));
                let is_busy = controller.is_busy().await;

                if !is_busy && idle_duration >= replica_timeout {
                    tracing::info!(
                        model_id = %self.config.id,
                        replica_index = controller.replica_index(),
                        idle_secs = idle_duration.as_secs(),
                        timeout_secs = replica_timeout.as_secs(),
                        "Deallocating and pruning idle model replica (scale-to-zero)"
                    );
                    controller.unload().await;
                    registry.record_deallocation("replica_scale_to_zero");
                } else if !controller.is_loaded().await && !is_busy {
                    tracing::info!(
                        model_id = %self.config.id,
                        replica_index = controller.replica_index(),
                        "Pruning unloaded model replica from replica pool"
                    );
                } else {
                    to_retain.push(controller.clone());
                }
            }
        }

        *guard = to_retain;
    }
}

/// Thread-safe registry managing model controllers, access history, and allocation coordination.
#[derive(Debug, Clone)]
pub struct ModelRegistry {
    pub(crate) inner: Arc<ModelRegistryInner>,
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelRegistry {
    /// Creates a new, empty `ModelRegistry`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(ModelRegistryInner {
                models: RwLock::new(HashMap::new()),
                load_lock: Mutex::new(()),
                max_loaded_models: RwLock::new(None),
                metrics: RwLock::new(None),
            }),
        }
    }

    /// Creates a new `ModelRegistry` with custom capacity limits and telemetry tracker.
    #[must_use]
    pub fn with_options(
        max_loaded_models: Option<usize>,
        metrics: Option<Arc<ServerMetrics>>,
    ) -> Self {
        Self {
            inner: Arc::new(ModelRegistryInner {
                models: RwLock::new(HashMap::new()),
                load_lock: Mutex::new(()),
                max_loaded_models: RwLock::new(max_loaded_models),
                metrics: RwLock::new(metrics),
            }),
        }
    }

    /// Attaches a Prometheus metrics tracker for deallocation and lifecycle telemetry.
    pub async fn set_metrics(&self, metrics: Arc<ServerMetrics>) {
        let mut guard = self.inner.metrics.write().await;
        *guard = Some(metrics);
    }

    /// Sets or clears the hard capacity cap on concurrently resident model sessions.
    pub async fn set_max_loaded_models(&self, max: Option<usize>) {
        let mut guard = self.inner.max_loaded_models.write().await;
        *guard = max;
    }

    /// Returns the currently configured capacity cap, or `None` if uncapped.
    pub async fn max_loaded_models(&self) -> Option<usize> {
        *self.inner.max_loaded_models.read().await
    }

    /// Registers a new `ModelController` under its configured identifier.
    ///
    /// Wraps the controller into a new [`ModelReplicaGroup`] as its primary instance.
    /// If the model is configured with `pinned = true`, its weights are eagerly loaded into memory.
    pub async fn register(&self, controller: Arc<ModelController>) {
        controller.set_registry(Arc::downgrade(&self.inner));
        let is_pinned = controller.is_pinned();
        let meta = controller.metadata().await;
        let model_id = meta.id.clone();
        let group = Arc::new(ModelReplicaGroup::new(
            controller.config().clone(),
            controller,
        ));
        group.set_registry(Arc::downgrade(&self.inner));
        {
            let mut guard = self.inner.models.write().await;
            guard.insert(model_id.clone(), group);
        }

        if is_pinned {
            tracing::info!(
                model_id = %model_id,
                "Model is configured as pinned; eagerly loading into memory at registration"
            );
            if let Err(err) = self.ensure_loaded_with_deallocation(&model_id).await {
                tracing::warn!(
                    model_id = %model_id,
                    error = %err,
                    "Failed to eagerly load pinned model during registration"
                );
            }
        }
    }

    /// Registers a pre-configured [`ModelReplicaGroup`] directly into the registry.
    pub async fn register_group(&self, group: Arc<ModelReplicaGroup>) {
        group.set_registry(Arc::downgrade(&self.inner));
        let id = group.id().to_string();
        let is_pinned = group
            .primary()
            .await
            .map(|c| c.is_pinned())
            .unwrap_or(false);
        {
            let mut guard = self.inner.models.write().await;
            guard.insert(id.clone(), group);
        }
        if is_pinned {
            let _ = self.ensure_loaded_with_deallocation(&id).await;
        }
    }

    /// Pins a model in memory and ensures its weights are loaded.
    ///
    /// While pinned, the model is exempt from background scale-to-zero idle sweeps and
    /// capacity/OOM LRU eviction until explicitly unpinned or unloaded.
    ///
    /// # Errors
    /// Returns [`InferenceError::NotFound`] if the model is not registered, or any backend
    /// loading error if weight allocation fails.
    pub async fn pin_model(&self, model_id: &str) -> Result<(), InferenceError> {
        let controller = self.get(model_id).await?;
        self.ensure_loaded_with_deallocation(model_id).await?;
        controller.set_pinned(true);
        tracing::info!(
            model_id = %model_id,
            "Pinned model in memory"
        );
        Ok(())
    }

    /// Unpins a model, allowing normal scale-to-zero idle timeouts and LRU eviction to apply.
    ///
    /// # Errors
    /// Returns [`InferenceError::NotFound`] if the model is not registered.
    pub async fn unpin_model(&self, model_id: &str) -> Result<(), InferenceError> {
        let controller = self.get(model_id).await?;
        controller.unpin();
        Ok(())
    }

    /// Retrieves the primary model controller for the specified model identifier.
    ///
    /// # Errors
    /// Returns [`InferenceError::NotFound`] if no model matches `model_id`.
    pub async fn get(&self, model_id: &str) -> Result<Arc<ModelController>, InferenceError> {
        let group = self.get_group(model_id).await?;
        group.primary().await
    }

    /// Retrieves the replica group managing the specified model identifier.
    ///
    /// # Errors
    /// Returns [`InferenceError::NotFound`] if no model matches `model_id`.
    pub async fn get_group(
        &self,
        model_id: &str,
    ) -> Result<Arc<ModelReplicaGroup>, InferenceError> {
        let guard = self.inner.models.read().await;
        guard.get(model_id).cloned().ok_or_else(|| {
            InferenceError::NotFound(format!("Model '{model_id}' is not registered"))
        })
    }

    /// Dispatches an inference request to an optimal replica controller.
    ///
    /// Routes to the replica with the lowest queue backlog, triggering autoscaling
    /// if existing replicas are saturated and resources are available.
    ///
    /// # Errors
    /// Returns [`InferenceError::NotFound`] if no model matches `model_id`.
    pub async fn route_for_inference(
        &self,
        model_id: &str,
    ) -> Result<Arc<ModelController>, InferenceError> {
        let group = self.get_group(model_id).await?;
        group.route_for_inference(self).await
    }

    /// Removes a model and all its replicas from the registry by identifier.
    pub async fn remove(&self, model_id: &str) -> Option<Arc<ModelController>> {
        let mut guard = self.inner.models.write().await;
        if let Some(group) = guard.remove(model_id) {
            group.unload_all().await;
            group.primary().await.ok()
        } else {
            None
        }
    }

    /// Returns a list of metadata for all registered models.
    pub async fn list_metadata(&self) -> Vec<ModelMetadata> {
        let guard = self.inner.models.read().await;
        let mut metadata_list = Vec::with_capacity(guard.len());
        for group in guard.values() {
            if let Ok(primary) = group.primary().await {
                metadata_list.push(primary.metadata().await);
            }
        }
        metadata_list
    }

    /// Scans a directory for model files, discovers new or updated models,
    /// loads configuration files if present, and registers them into the registry.
    ///
    /// # Errors
    /// Returns [`InferenceError::NotFound`] if the directory does not exist or is not a directory,
    /// or [`InferenceError::Runtime`] if discovery fails.
    pub async fn refresh_from_dir(
        &self,
        models_dir: &std::path::Path,
    ) -> Result<Vec<ModelMetadata>, InferenceError> {
        if !models_dir.is_dir() {
            return Err(InferenceError::NotFound(format!(
                "Models directory '{}' does not exist or is not a directory",
                models_dir.display()
            )));
        }

        let discovered = crate::config::ModelDirectoryConfig::discover_models_in_dir(models_dir)
            .await
            .map_err(|e| {
                InferenceError::Runtime(format!(
                    "Failed to discover models in '{}': {e}",
                    models_dir.display()
                ))
            })?;

        let loaded_config =
            crate::config::ModelDirectoryConfig::load_fault_tolerant(models_dir, None).await;
        let model_configs = match loaded_config {
            Some(config) => config.build_model_configs(models_dir, &discovered),
            None => {
                let fallback = crate::config::ModelDirectoryConfig::default();
                fallback.build_model_configs(models_dir, &discovered)
            }
        };

        for config in model_configs {
            let model_id = config.id.clone();
            let is_registered = self.inner.models.read().await.contains_key(&model_id);
            if !is_registered {
                let controller = ModelController::new(config);
                self.register(controller).await;
            }
        }

        Ok(self.list_metadata().await)
    }

    /// Returns snapshots of all currently registered `ModelController` references (primary and replicas).
    pub async fn all_controllers(&self) -> Vec<Arc<ModelController>> {
        let guard = self.inner.models.read().await;
        let mut all = Vec::new();
        for group in guard.values() {
            all.extend(group.all_replicas().await);
        }
        all
    }

    /// Returns the total number of model instances currently resident in memory.
    pub async fn count_loaded_models(&self) -> usize {
        let groups: Vec<Arc<ModelReplicaGroup>> = {
            let guard = self.inner.models.read().await;
            guard.values().cloned().collect()
        };
        let mut count = 0usize;
        for group in groups {
            count = count.saturating_add(group.count_loaded_replicas().await);
        }
        count
    }

    /// Identifies the best idle model instance to deallocate based on least-recent access time.
    ///
    /// Prioritizes autoscaled replica copies over primary instances, ensuring surplus replicas
    /// are reclaimed first. Pinned models (`is_pinned() == true`) are strictly excluded and never
    /// selected for deallocation. Models loaded less than their configured `min_residency` duration
    /// are protected unless all available idle unpinned models are protected.
    pub async fn find_lru_deallocation_candidate(
        &self,
        target_model_id: &str,
    ) -> Option<Arc<ModelController>> {
        let guard = self.inner.models.read().await;
        let mut best_eligible: Option<Arc<ModelController>> = None;
        let mut best_protected: Option<Arc<ModelController>> = None;

        for (id, group) in guard.iter() {
            let replicas = group.all_replicas().await;
            for replica in replicas {
                if id == target_model_id && !replica.is_replica() {
                    continue;
                }
                if replica.is_pinned() {
                    continue;
                }
                if replica.is_loaded().await && !replica.is_busy().await {
                    let is_protected = if let Some(residency) = replica.residency_duration() {
                        residency < replica.min_residency()
                    } else {
                        false
                    };

                    let key = (!replica.is_replica(), replica.last_accessed_millis());

                    if is_protected {
                        let replace = match &best_protected {
                            None => true,
                            Some(current) => {
                                key < (!current.is_replica(), current.last_accessed_millis())
                            }
                        };
                        if replace {
                            best_protected = Some(replica);
                        }
                    } else {
                        let replace = match &best_eligible {
                            None => true,
                            Some(current) => {
                                key < (!current.is_replica(), current.last_accessed_millis())
                            }
                        };
                        if replace {
                            best_eligible = Some(replica);
                        }
                    }
                }
            }
        }

        if let Some(eligible) = best_eligible {
            return Some(eligible);
        }

        if let Some(victim) = best_protected {
            self.inner.record_thrashing();
            tracing::warn!(
                victim_id = %victim.id(),
                replica_index = victim.replica_index(),
                target_id = %target_model_id,
                "Anti-thrashing safeguard bypassed: all idle models were loaded recently; deallocating oldest"
            );
            return Some(victim);
        }

        None
    }

    /// Loads the target model, deallocating LRU idle instances one by one if a verified OOM error occurs.
    ///
    /// # Errors
    /// Returns [`InferenceError::ResourceExhausted`] if memory is exhausted and no deallocatable models remain.
    /// Returns any underlying backend error if allocation failure was not caused by an out-of-memory condition.
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
        let _guard = self.inner.load_lock.lock().await;

        // Double-check after acquiring lock
        if controller.is_loaded().await {
            controller.touch();
            return Ok(());
        }

        // 2. Preemptive Deallocation: enforce optional max_loaded_models cap
        let max_models = *self.inner.max_loaded_models.read().await;
        if let Some(max_models) = max_models {
            while self.count_loaded_models().await >= max_models {
                if let Some(victim) = self.find_lru_deallocation_candidate(target_model_id).await {
                    tracing::info!(
                        victim_id = %victim.id(),
                        replica_index = victim.replica_index(),
                        limit = max_models,
                        "Preemptively deallocating idle model to respect max_loaded_models threshold"
                    );
                    self.inner.record_deallocation("capacity");
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
                    self.inner.record_load_retry();
                    tracing::warn!(
                        model_id = %target_model_id,
                        error = %err,
                        "Verified out-of-memory during model allocation; searching for deallocation candidate"
                    );

                    if let Some(victim) =
                        self.find_lru_deallocation_candidate(target_model_id).await
                    {
                        tracing::info!(
                            target: "snooze::lifecycle",
                            action = "deallocate",
                            victim = %victim.id(),
                            replica_index = victim.replica_index(),
                            reason = "oom"
                        );
                        tracing::warn!(
                            victim_id = %victim.id(),
                            replica_index = victim.replica_index(),
                            target_id = %target_model_id,
                            "Deallocating LRU model to reclaim memory/VRAM"
                        );
                        self.inner.record_deallocation("oom");
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
                // Any other error fails immediately
                Err(fatal_err) => {
                    return Err(fatal_err);
                }
            }
        }
    }

    /// Sweeps all registered models and replicas for scale-to-zero memory reclamation.
    ///
    /// - Primary model instances respect their configured `idle_timeout`.
    /// - Additional replicas deallocate more rapidly according to `compute_replica_idle_timeout`.
    pub async fn sweep_scale_to_zero(&self) {
        let groups: Vec<Arc<ModelReplicaGroup>> = {
            let guard = self.inner.models.read().await;
            guard.values().cloned().collect()
        };

        for group in groups {
            group.sweep_scale_to_zero(&self.inner).await;
        }
    }
}
