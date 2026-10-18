//! Integration and lifecycle tests for the scale-to-zero model sweeper background task.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use ndarray::ArrayD;
use snooze_ml::model::{ModelConfig, ModelController};
use snooze_ml::registry::ModelRegistry;
use snooze_ml::sweeper::spawn_scale_to_zero_sweeper;
use snooze_ml::types::{ModelState, TensorPayload};
use std::time::Duration;

#[tokio::test]
async fn test_sweeper_deallocates_idle_model() {
    let registry = ModelRegistry::new();
    let config = ModelConfig {
        id: "deallocate-target".to_string(),
        onnx_path: "mock:test".to_string(),
        idle_timeout: Duration::from_millis(20),
        max_batch_size: 2,
        max_batch_delay: Duration::from_millis(5),
        provider: None,
        description: None,
        min_residency: Duration::from_millis(0),
        ..Default::default()
    };

    let controller = ModelController::new(config);
    registry.register(controller.clone()).await;

    // Force model load via inference
    let input = TensorPayload {
        name: "in".into(),
        shape: vec![1, 2],
        data: ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 2]), vec![1.0, 2.0]).unwrap(),
    };
    let _ = controller.infer(vec![input]).await.unwrap();
    assert_eq!(controller.metadata().await.state, ModelState::Ready);

    // Spawn sweeper with very fast check interval
    let sweeper_handle = spawn_scale_to_zero_sweeper(registry, Duration::from_millis(10));

    // Sleep longer than idle timeout
    tokio::time::sleep(Duration::from_millis(50)).await;

    let meta = controller.metadata().await;
    assert_eq!(meta.state, ModelState::Idle);

    sweeper_handle.abort();
}

#[tokio::test]
async fn test_sweeper_retains_active_model() {
    let registry = ModelRegistry::new();

    let cfg_idle = ModelConfig {
        id: "idle-one".to_string(),
        onnx_path: "mock:test".to_string(),
        idle_timeout: Duration::from_millis(30),
        max_batch_size: 2,
        max_batch_delay: Duration::from_millis(5),
        provider: None,
        description: None,
        min_residency: Duration::from_millis(0),
        ..Default::default()
    };
    let cfg_active = ModelConfig {
        id: "active-one".to_string(),
        onnx_path: "mock:test".to_string(),
        idle_timeout: Duration::from_millis(30),
        max_batch_size: 2,
        max_batch_delay: Duration::from_millis(5),
        provider: None,
        description: None,
        min_residency: Duration::from_millis(0),
        ..Default::default()
    };

    let ctrl_idle = ModelController::new(cfg_idle);
    let ctrl_active = ModelController::new(cfg_active);

    registry.register(ctrl_idle.clone()).await;
    registry.register(ctrl_active.clone()).await;

    ctrl_idle.ensure_loaded().await.unwrap();
    ctrl_active.ensure_loaded().await.unwrap();

    assert_eq!(ctrl_idle.metadata().await.state, ModelState::Ready);
    assert_eq!(ctrl_active.metadata().await.state, ModelState::Ready);

    let sweeper = spawn_scale_to_zero_sweeper(registry, Duration::from_millis(10));

    // Sleep 60ms total, but touch active model halfway through
    tokio::time::sleep(Duration::from_millis(20)).await;
    ctrl_active.touch();
    tokio::time::sleep(Duration::from_millis(25)).await;
    ctrl_active.touch();
    tokio::time::sleep(Duration::from_millis(25)).await;

    let meta_idle = ctrl_idle.metadata().await;
    let meta_active = ctrl_active.metadata().await;

    assert_eq!(
        meta_idle.state,
        ModelState::Idle,
        "idle model should be deallocated"
    );
    assert_eq!(
        meta_active.state,
        ModelState::Ready,
        "frequently accessed model should remain ready"
    );

    sweeper.abort();
}

#[tokio::test]
async fn test_sweeper_abort_cancels_cleanly() {
    let registry = ModelRegistry::new();
    let sweeper = spawn_scale_to_zero_sweeper(registry, Duration::from_millis(50));
    sweeper.abort();
    let join_res = sweeper.await;
    assert!(join_res.is_err());
    assert!(join_res.unwrap_err().is_cancelled());
}

#[tokio::test]
async fn test_sweeper_retains_pinned_model_past_idle_timeout() {
    let registry = ModelRegistry::new();

    let cfg_pinned = ModelConfig {
        id: "pinned-model".to_string(),
        onnx_path: "mock:pinned".to_string(),
        idle_timeout: Duration::from_millis(20),
        max_batch_size: 2,
        max_batch_delay: Duration::from_millis(5),
        min_residency: Duration::from_millis(0),
        pinned: true,
        ..Default::default()
    };
    let cfg_unpinned = ModelConfig {
        id: "unpinned-model".to_string(),
        onnx_path: "mock:unpinned".to_string(),
        idle_timeout: Duration::from_millis(20),
        max_batch_size: 2,
        max_batch_delay: Duration::from_millis(5),
        min_residency: Duration::from_millis(0),
        pinned: false,
        ..Default::default()
    };

    let ctrl_pinned = ModelController::new(cfg_pinned);
    let ctrl_unpinned = ModelController::new(cfg_unpinned);

    registry.register(ctrl_pinned.clone()).await;
    registry.register(ctrl_unpinned.clone()).await;

    ctrl_unpinned.ensure_loaded().await.unwrap();
    assert_eq!(ctrl_pinned.metadata().await.state, ModelState::Ready);
    assert_eq!(ctrl_unpinned.metadata().await.state, ModelState::Ready);

    let sweeper = spawn_scale_to_zero_sweeper(registry, Duration::from_millis(10));

    // Sleep well past the 20ms idle timeout without touching either model
    tokio::time::sleep(Duration::from_millis(60)).await;

    assert_eq!(
        ctrl_pinned.metadata().await.state,
        ModelState::Ready,
        "Pinned model must remain Ready even after idle_timeout expires"
    );
    assert_eq!(
        ctrl_unpinned.metadata().await.state,
        ModelState::Idle,
        "Unpinned model must scale to zero after idle_timeout expires"
    );

    sweeper.abort();
}
