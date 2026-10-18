//! Background sweeper task for periodic scale-to-zero model session reclamation.

use crate::registry::ModelRegistry;
use std::time::Duration;
use tokio::task::JoinHandle;

/// Spawns a background task that periodically sweeps the model registry
/// and triggers deallocation of idle models and surplus replicas according to their
/// respective scale-to-zero lifetimes.
#[must_use]
pub fn spawn_scale_to_zero_sweeper(
    registry: ModelRegistry,
    check_interval: Duration,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(check_interval);
        loop {
            interval.tick().await;
            registry.sweep_scale_to_zero().await;
        }
    })
}
