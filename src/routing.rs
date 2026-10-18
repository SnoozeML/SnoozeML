//! # Replica Routing Strategies & Queue-Aware Load Balancing
//!
//! When a model scales horizontally to multiple concurrent instances (replicas),
//! incoming inference requests must be dispatched across replicas according to a
//! routing policy.
//!
//! ## Routing Mechanism Among Same Models
//!
//! 1. **Least-Queue Routing (Active Default):**
//!    Assigns the incoming request to the replica with the shortest pending queue.
//!    Specifically, it inspects `(controller.queue_len(), controller.active_requests())`
//!    to find the replica with the least backlog.
//!    - **Why this strategy?**
//!      In deep learning inference, batching and inference execution times are variable
//!      and dependent on input sizes and dynamic batch latency windows (`max_delay`).
//!      Assigning to the replica with the lowest queue prevents request stragglers,
//!      maximizes parallel GPU/CPU throughput, and avoids head-of-line blocking.
//!
//! 2. **Extensible Architecture for Future Strategies:**
//!    The [`RoutingStrategy`] trait decouples the selection logic from model controllers
//!    and the registry. Future routing strategies can be added with zero changes to the
//!    registry or server internals:
//!    - **Round-Robin:** Cyclically dispatches requests across active replicas.
//!      Ideal for fixed-size batches and predictable, uniform inference workloads.
//!    - **Random:** Selects a replica uniformly at random. Stateless, lock-free, and
//!      efficient under extreme concurrency.
//!    - **Peak Exponentially-Weighted Moving Average (PEWMA) / Latency-Aware:**
//!      Tracks recent execution latency per replica and routes to the fastest replica,
//!      useful when replicas run on heterogeneous execution providers (e.g. CUDA vs CPU).
//!    - **Power of Two Choices:**
//!      Samples two replicas at random and picks the one with the shorter queue, reducing
//!      inspection overhead at large replica counts.

use crate::model::ModelController;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Supported replica routing strategy algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum RoutingStrategyType {
    /// Routes requests to the replica with the shortest queue depth.
    #[default]
    LeastQueue,
    /// Routes requests cyclically across all active replicas.
    RoundRobin,
    /// Routes requests uniformly at random across all active replicas.
    Random,
}

impl std::str::FromStr for RoutingStrategyType {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s.trim().to_ascii_lowercase().as_str() {
            "round_robin" | "roundrobin" => Self::RoundRobin,
            "random" => Self::Random,
            _ => Self::LeastQueue,
        })
    }
}

/// Pluggable strategy for selecting an optimal replica among identical model instances.
pub trait RoutingStrategy: Send + Sync + std::fmt::Debug {
    /// Selects an optimal replica controller from the provided candidates.
    ///
    /// Returns `None` if `replicas` is empty.
    fn select_replica<'a>(
        &self,
        replicas: &'a [Arc<ModelController>],
    ) -> Option<&'a Arc<ModelController>>;
}

/// Assigns requests to the replica with the shortest queue backlog.
///
/// Breaks ties by comparing the count of active in-flight executions.
#[derive(Debug, Default, Clone, Copy)]
pub struct LeastQueueRouting;

impl RoutingStrategy for LeastQueueRouting {
    fn select_replica<'a>(
        &self,
        replicas: &'a [Arc<ModelController>],
    ) -> Option<&'a Arc<ModelController>> {
        if replicas.len() <= 1 {
            return replicas.first();
        }
        replicas
            .iter()
            .min_by_key(|controller| (controller.queue_len(), controller.active_requests()))
    }
}

/// Assigns requests sequentially in round-robin order across replicas.
#[derive(Debug, Default)]
pub struct RoundRobinRouting {
    cursor: AtomicUsize,
}

impl RoundRobinRouting {
    /// Creates a new `RoundRobinRouting` strategy starting at index 0.
    #[must_use]
    pub fn new() -> Self {
        Self {
            cursor: AtomicUsize::new(0),
        }
    }
}

impl RoutingStrategy for RoundRobinRouting {
    fn select_replica<'a>(
        &self,
        replicas: &'a [Arc<ModelController>],
    ) -> Option<&'a Arc<ModelController>> {
        if replicas.len() <= 1 {
            return replicas.first();
        }
        let current = self.cursor.fetch_add(1, Ordering::Relaxed);
        let index = current % replicas.len();
        replicas.get(index)
    }
}

/// Assigns requests uniformly at random across active replicas.
#[derive(Debug, Default, Clone, Copy)]
pub struct RandomRouting;

impl RoutingStrategy for RandomRouting {
    fn select_replica<'a>(
        &self,
        replicas: &'a [Arc<ModelController>],
    ) -> Option<&'a Arc<ModelController>> {
        if replicas.len() <= 1 {
            return replicas.first();
        }
        // Pseudo-random index without external crates: use monotonic timestamp
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| usize::try_from(d.subsec_nanos()).unwrap_or(0))
            .unwrap_or(0);
        let index = nanos % replicas.len();
        replicas.get(index)
    }
}

/// Creates a boxed [`RoutingStrategy`] based on the specified type.
#[must_use]
pub fn create_routing_strategy(strategy_type: RoutingStrategyType) -> Box<dyn RoutingStrategy> {
    match strategy_type {
        RoutingStrategyType::LeastQueue => Box::new(LeastQueueRouting),
        RoutingStrategyType::RoundRobin => Box::new(RoundRobinRouting::new()),
        RoutingStrategyType::Random => Box::new(RandomRouting),
    }
}

/// Configurable router coordinating replica selection across models.
#[derive(Debug)]
pub struct ReplicaRouter {
    strategy: Box<dyn RoutingStrategy>,
}

impl ReplicaRouter {
    /// Creates a new `ReplicaRouter` with the provided strategy.
    #[must_use]
    pub fn new(strategy: Box<dyn RoutingStrategy>) -> Self {
        Self { strategy }
    }

    /// Creates a `ReplicaRouter` from a strategy type.
    #[must_use]
    pub fn from_type(strategy_type: RoutingStrategyType) -> Self {
        Self {
            strategy: create_routing_strategy(strategy_type),
        }
    }

    /// Selects an optimal replica using the configured strategy.
    #[must_use]
    pub fn select_replica<'a>(
        &self,
        replicas: &'a [Arc<ModelController>],
    ) -> Option<&'a Arc<ModelController>> {
        self.strategy.select_replica(replicas)
    }
}
