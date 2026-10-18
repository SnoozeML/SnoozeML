//! Integration and behavioral unit tests for ModelController lifecycle, lazy loading, and deallocation.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use ndarray::ArrayD;
use snooze_ml::model::{ModelConfig, ModelController};
use snooze_ml::types::{InferenceError, ModelState, TensorPayload};
use std::time::Duration;

#[tokio::test]
async fn test_model_controller_lifecycle() {
    let config = ModelConfig {
        id: "test-model".to_string(),
        onnx_path: "mock:test".to_string(),
        idle_timeout: Duration::from_millis(50),
        max_batch_size: 4,
        max_batch_delay: Duration::from_millis(10),
        provider: None,
        description: Some("Lifecycle test model".to_string()),
        min_residency: Duration::from_millis(0),
        ..Default::default()
    };

    let controller = ModelController::new(config);
    let meta_initial = controller.metadata().await;
    assert_eq!(meta_initial.state, ModelState::Idle);
    assert_eq!(
        meta_initial.description.as_deref(),
        Some("Lifecycle test model")
    );

    // Perform inference: triggers lazy load
    let input_tensor = TensorPayload {
        name: "input".into(),
        shape: vec![1, 3],
        data: ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 3]), vec![1.0, 2.0, 3.0]).unwrap(),
    };

    let result = controller.infer(vec![input_tensor]).await.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].shape, vec![1, 4]);

    let meta_loaded = controller.metadata().await;
    assert_eq!(meta_loaded.state, ModelState::Ready);

    // Test explicit unload
    controller.unload().await;
    let meta_unloaded = controller.metadata().await;
    assert_eq!(meta_unloaded.state, ModelState::Idle);
}

#[tokio::test]
async fn test_model_controller_ensure_loaded_idempotent() {
    let config = ModelConfig {
        id: "idempotent-model".to_string(),
        onnx_path: "mock:test".to_string(),
        idle_timeout: Duration::from_secs(60),
        max_batch_size: 2,
        max_batch_delay: Duration::from_millis(10),
        provider: None,
        description: None,
        min_residency: Duration::from_millis(0),
        ..Default::default()
    };
    let controller = ModelController::new(config);
    assert_eq!(controller.metadata().await.state, ModelState::Idle);

    // First call loads
    controller.ensure_loaded().await.unwrap();
    assert_eq!(controller.metadata().await.state, ModelState::Ready);

    // Second call is a no-op
    controller.ensure_loaded().await.unwrap();
    assert_eq!(controller.metadata().await.state, ModelState::Ready);
}

#[tokio::test]
async fn test_model_controller_deallocate_if_idle() {
    let config = ModelConfig {
        id: "idle-deallocate".to_string(),
        onnx_path: "mock:test".to_string(),
        idle_timeout: Duration::from_millis(30),
        max_batch_size: 2,
        max_batch_delay: Duration::from_millis(5),
        provider: None,
        description: None,
        min_residency: Duration::from_millis(0),
        ..Default::default()
    };
    let controller = ModelController::new(config);
    controller.ensure_loaded().await.unwrap();
    assert_eq!(controller.metadata().await.state, ModelState::Ready);

    // Not yet timed out
    controller.deallocate_if_idle().await;
    assert_eq!(controller.metadata().await.state, ModelState::Ready);

    // Wait past idle timeout
    tokio::time::sleep(Duration::from_millis(50)).await;
    controller.deallocate_if_idle().await;
    assert_eq!(controller.metadata().await.state, ModelState::Idle);
}

#[tokio::test]
async fn test_model_controller_invalid_model_path() {
    let config = ModelConfig {
        id: "broken-model".to_string(),
        onnx_path: "/tmp/snooze_non_existent_file.onnx".to_string(),
        idle_timeout: Duration::from_secs(60),
        max_batch_size: 2,
        max_batch_delay: Duration::from_millis(10),
        provider: None,
        description: None,
        min_residency: Duration::from_millis(0),
        ..Default::default()
    };
    let controller = ModelController::new(config);
    assert_eq!(controller.metadata().await.state, ModelState::Idle);

    let load_res = controller.ensure_loaded().await;
    assert!(load_res.is_err());
    // Ensure state does not get stuck in Loading on failure, but reverts to Idle
    assert_eq!(controller.metadata().await.state, ModelState::Idle);

    let input_tensor = TensorPayload {
        name: "input".into(),
        shape: vec![1, 2],
        data: ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 2]), vec![1.0, 2.0]).unwrap(),
    };
    let infer_res = controller.infer(vec![input_tensor]).await;
    assert!(infer_res.is_err());
    // Ensure state remains Idle even after failed inference attempt
    assert_eq!(controller.metadata().await.state, ModelState::Idle);

    // Explicit unload on an unloaded/failed model safely retains Idle state
    controller.unload().await;
    assert_eq!(controller.metadata().await.state, ModelState::Idle);
}

#[tokio::test]
async fn test_model_controller_touch_and_debug() {
    let config = ModelConfig {
        id: "debug-model".to_string(),
        onnx_path: "mock:test".to_string(),
        idle_timeout: Duration::from_secs(60),
        max_batch_size: 2,
        max_batch_delay: Duration::from_millis(10),
        provider: None,
        description: None,
        min_residency: Duration::from_millis(0),
        ..Default::default()
    };
    let controller = ModelController::new(config);
    controller.touch();

    let debug_str = format!("{controller:?}");
    assert!(debug_str.contains("ModelController"));
    assert!(debug_str.contains("debug-model"));

    let meta = controller.metadata().await;
    assert_eq!(meta.id, "debug-model");
    assert_eq!(meta.last_accessed_secs_ago, 0);
}

#[tokio::test]
async fn test_model_controller_pin_and_unpin() {
    let config = ModelConfig {
        id: "pin-target".to_string(),
        onnx_path: "mock:test".to_string(),
        idle_timeout: Duration::from_millis(20),
        max_batch_size: 2,
        max_batch_delay: Duration::from_millis(5),
        min_residency: Duration::from_millis(0),
        pinned: false,
        ..Default::default()
    };
    let controller = ModelController::new(config);
    assert!(!controller.is_pinned());
    assert_eq!(controller.metadata().await.state, ModelState::Idle);
    assert!(!controller.metadata().await.pinned);

    // Pinning eagerly loads the model and marks it as pinned
    controller.pin().await.unwrap();
    assert!(controller.is_pinned());
    let meta = controller.metadata().await;
    assert_eq!(meta.state, ModelState::Ready);
    assert!(meta.pinned);

    // Wait past idle timeout — deallocate_if_idle MUST NOT unload a pinned model
    tokio::time::sleep(Duration::from_millis(40)).await;
    controller.deallocate_if_idle().await;
    assert_eq!(
        controller.metadata().await.state,
        ModelState::Ready,
        "Pinned model must not be deallocated when idle"
    );

    // Unpinning keeps the model loaded, but allows subsequent idle deallocation
    controller.unpin();
    assert!(!controller.is_pinned());
    assert!(!controller.metadata().await.pinned);
    assert_eq!(controller.metadata().await.state, ModelState::Ready);

    // Now deallocate_if_idle unloads it because idle timeout has already elapsed
    controller.deallocate_if_idle().await;
    assert_eq!(controller.metadata().await.state, ModelState::Idle);

    // Explicit unload still works even if a model is pinned
    controller.pin().await.unwrap();
    assert_eq!(controller.metadata().await.state, ModelState::Ready);
    controller.unload().await;
    assert_eq!(controller.metadata().await.state, ModelState::Idle);
}

#[tokio::test]
async fn test_model_thrashing_race_infer_and_unload() {
    let config = ModelConfig {
        id: "race-model".to_string(),
        onnx_path: "mock:test".to_string(),
        idle_timeout: Duration::from_millis(10),
        max_batch_size: 4,
        max_batch_delay: Duration::from_millis(5),
        min_residency: Duration::from_millis(0),
        ..Default::default()
    };
    let controller = ModelController::new(config);
    controller.ensure_loaded().await.unwrap();

    let mut handles = Vec::new();

    // Spawn 10 concurrent inference tasks
    for i in 0..10 {
        let c = controller.clone();
        handles.push(tokio::spawn(async move {
            let tensor = TensorPayload {
                name: "in".into(),
                shape: vec![1, 2],
                data: ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 2]), vec![i as f32, 1.0]).unwrap(),
            };
            c.infer(vec![tensor]).await
        }));
    }

    // Simultaneously spawn 10 concurrent unload tasks
    for _ in 0..10 {
        let c = controller.clone();
        tokio::spawn(async move {
            c.unload().await;
        });
    }

    // Wait for all inference calls: none must panic, and channels must not be dropped unexpectedly
    for handle in handles {
        let res = handle.await.unwrap();
        // Requests either succeed or cleanly return an InferenceError (e.g. backend was unloaded)
        match res {
            Ok(output) => {
                assert!(!output.is_empty());
            }
            Err(err) => {
                assert!(
                    matches!(
                        err,
                        InferenceError::Runtime(_)
                            | InferenceError::Loading
                            | InferenceError::Timeout
                    ),
                    "Expected clean error variant, got: {err:?}"
                );
            }
        }
    }

    // State machine must remain consistent in either Ready or Idle (never stuck in Loading or Busy)
    let final_state = controller.metadata().await.state;
    assert!(
        final_state == ModelState::Ready || final_state == ModelState::Idle,
        "Final state must be Ready or Idle, got: {final_state:?}"
    );
}

#[tokio::test]
async fn test_model_unpin_while_requests_in_flight() {
    let config = ModelConfig {
        id: "unpin-inflight".to_string(),
        onnx_path: "mock:test".to_string(),
        idle_timeout: Duration::from_millis(10),
        max_batch_size: 2,
        max_batch_delay: Duration::from_millis(50), // Keep request in-flight for 50ms
        min_residency: Duration::from_millis(0),
        pinned: true,
        ..Default::default()
    };
    let controller = ModelController::new(config);
    controller.pin().await.unwrap();
    assert!(controller.is_pinned());
    assert!(controller.is_loaded().await);

    // Launch an infer request in background
    let c = controller.clone();
    let infer_handle = tokio::spawn(async move {
        let tensor = TensorPayload {
            name: "in".into(),
            shape: vec![1, 2],
            data: ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 2]), vec![1.0, 2.0]).unwrap(),
        };
        c.infer(vec![tensor]).await
    });

    // Wait 5ms for infer request to register active_requests
    tokio::time::sleep(Duration::from_millis(5)).await;
    assert!(
        controller.active_requests() > 0,
        "Active request count must be > 0 while batcher delay is running"
    );

    // Unpin while request is actively running
    controller.unpin();
    assert!(!controller.is_pinned(), "Model is now unpinned");

    // Even though idle timeout has passed and model is unpinned, deallocate_if_idle MUST NOT unload it
    tokio::time::sleep(Duration::from_millis(15)).await;
    controller.deallocate_if_idle().await;
    assert!(
        controller.is_loaded().await,
        "Model must remain loaded while active_requests > 0 even after unpinning"
    );

    // Complete inference
    let infer_res = infer_handle.await.unwrap();
    assert!(infer_res.is_ok());

    // Once in-flight request finishes, active_requests drops to 0
    assert_eq!(controller.active_requests(), 0);

    // Advance time past idle timeout
    tokio::time::sleep(Duration::from_millis(15)).await;
    controller.deallocate_if_idle().await;

    // Now it safely deallocates to Idle
    assert!(
        !controller.is_loaded().await,
        "Model should deallocate once active requests have completed"
    );
    assert_eq!(controller.metadata().await.state, ModelState::Idle);
}
