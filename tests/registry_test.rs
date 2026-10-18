//! Integration and unit tests for the thread-safe ModelRegistry.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use snooze_ml::model::{ModelConfig, ModelController};
use snooze_ml::registry::ModelRegistry;
use snooze_ml::server::metrics::ServerMetrics;
use snooze_ml::types::InferenceError;
use std::time::Duration;

static OOM_TEST_MUTEX: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

#[tokio::test]
async fn test_registry_registration_and_lookup() {
    let registry = ModelRegistry::new();

    let config = ModelConfig {
        id: "model-1".to_string(),
        onnx_path: "mock:1".to_string(),
        idle_timeout: Duration::from_secs(60),
        max_batch_size: 4,
        max_batch_delay: Duration::from_millis(5),
        provider: None,
        description: None,
        min_residency: Duration::from_millis(0),
        ..Default::default()
    };

    let controller = ModelController::new(config);
    registry.register(controller).await;

    let retrieved = registry.get("model-1").await;
    assert!(retrieved.is_ok());

    let missing = registry.get("non-existent").await;
    assert!(matches!(missing, Err(InferenceError::NotFound(_))));

    let list = registry.list_metadata().await;
    assert_eq!(list.len(), 1);
    assert_eq!(list[0].id, "model-1");
}

#[tokio::test]
async fn test_registry_remove() {
    let registry = ModelRegistry::new();
    let config = ModelConfig {
        id: "removable".to_string(),
        onnx_path: "mock:test".to_string(),
        idle_timeout: Duration::from_secs(60),
        max_batch_size: 4,
        max_batch_delay: Duration::from_millis(5),
        provider: None,
        description: None,
        min_residency: Duration::from_millis(0),
        ..Default::default()
    };

    registry.register(ModelController::new(config)).await;
    assert!(registry.get("removable").await.is_ok());

    // Remove existing
    let removed = registry.remove("removable").await;
    assert!(removed.is_some());
    assert!(registry.get("removable").await.is_err());

    // Remove again returns None
    let removed_again = registry.remove("removable").await;
    assert!(removed_again.is_none());

    // Remove unknown returns None
    let missing = registry.remove("does-not-exist").await;
    assert!(missing.is_none());
}

#[tokio::test]
async fn test_registry_list_metadata_and_all_controllers() {
    let registry = ModelRegistry::new();
    assert!(registry.list_metadata().await.is_empty());
    assert!(registry.all_controllers().await.is_empty());

    for id in &["alpha", "beta", "gamma"] {
        let config = ModelConfig {
            id: (*id).to_string(),
            onnx_path: format!("mock:{id}"),
            idle_timeout: Duration::from_secs(60),
            max_batch_size: 2,
            max_batch_delay: Duration::from_millis(5),
            provider: None,
            description: None,
            min_residency: Duration::from_millis(0),
            ..Default::default()
        };
        registry.register(ModelController::new(config)).await;
    }

    let metadata_list = registry.list_metadata().await;
    assert_eq!(metadata_list.len(), 3);

    let controllers = registry.all_controllers().await;
    assert_eq!(controllers.len(), 3);

    let mut ids: Vec<String> = metadata_list.into_iter().map(|m| m.id).collect();
    ids.sort();
    assert_eq!(ids, vec!["alpha", "beta", "gamma"]);
}

#[tokio::test]
async fn test_registry_count_loaded_models() {
    let registry = ModelRegistry::new();
    let m1 = ModelController::new(ModelConfig {
        id: "m1".into(),
        onnx_path: "mock:1".into(),
        ..Default::default()
    });
    let m2 = ModelController::new(ModelConfig {
        id: "m2".into(),
        onnx_path: "mock:2".into(),
        ..Default::default()
    });

    registry.register(m1.clone()).await;
    registry.register(m2.clone()).await;

    assert_eq!(registry.count_loaded_models().await, 0);

    m1.ensure_loaded().await.unwrap();
    assert_eq!(registry.count_loaded_models().await, 1);

    m2.ensure_loaded().await.unwrap();
    assert_eq!(registry.count_loaded_models().await, 2);

    m1.unload().await;
    assert_eq!(registry.count_loaded_models().await, 1);
}

#[tokio::test]
async fn test_registry_lru_candidate_selection() {
    let registry = ModelRegistry::new();
    let m1 = ModelController::new(ModelConfig {
        id: "m1".into(),
        onnx_path: "mock:1".into(),
        min_residency: Duration::from_millis(0),
        ..Default::default()
    });
    let m2 = ModelController::new(ModelConfig {
        id: "m2".into(),
        onnx_path: "mock:2".into(),
        min_residency: Duration::from_millis(0),
        ..Default::default()
    });

    registry.register(m1.clone()).await;
    registry.register(m2.clone()).await;

    // Initially neither is loaded, so no candidate
    assert!(registry
        .find_lru_deallocation_candidate("target")
        .await
        .is_none());

    // Load m1, wait slightly, load m2
    m1.ensure_loaded().await.unwrap();
    tokio::time::sleep(Duration::from_millis(15)).await;
    m2.ensure_loaded().await.unwrap();

    // m1 was accessed before m2, so m1 should be the LRU candidate
    let candidate = registry.find_lru_deallocation_candidate("target").await;
    assert!(candidate.is_some());
    assert_eq!(candidate.unwrap().id(), "m1");

    // Touch m1, making m2 the least recently accessed
    tokio::time::sleep(Duration::from_millis(15)).await;
    m1.touch();

    let candidate_after_touch = registry.find_lru_deallocation_candidate("target").await;
    assert!(candidate_after_touch.is_some());
    assert_eq!(candidate_after_touch.unwrap().id(), "m2");

    // Excluding m2 selects m1
    let candidate_exclude_m2 = registry.find_lru_deallocation_candidate("m2").await;
    assert!(candidate_exclude_m2.is_some());
    assert_eq!(candidate_exclude_m2.unwrap().id(), "m1");
}

#[tokio::test]
async fn test_registry_preemptive_capacity_deallocation() {
    let registry = ModelRegistry::new();
    // Set capacity cap to 2 models
    registry.set_max_loaded_models(Some(2)).await;

    let m1 = ModelController::new(ModelConfig {
        id: "m1".into(),
        onnx_path: "mock:1".into(),
        min_residency: Duration::from_millis(0),
        ..Default::default()
    });
    let m2 = ModelController::new(ModelConfig {
        id: "m2".into(),
        onnx_path: "mock:2".into(),
        min_residency: Duration::from_millis(0),
        ..Default::default()
    });
    let m3 = ModelController::new(ModelConfig {
        id: "m3".into(),
        onnx_path: "mock:3".into(),
        min_residency: Duration::from_millis(0),
        ..Default::default()
    });

    registry.register(m1.clone()).await;
    registry.register(m2.clone()).await;
    registry.register(m3.clone()).await;

    // Load m1 and m2
    registry
        .ensure_loaded_with_deallocation("m1")
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(15)).await;
    registry
        .ensure_loaded_with_deallocation("m2")
        .await
        .unwrap();
    assert_eq!(registry.count_loaded_models().await, 2);
    assert!(m1.is_loaded().await);
    assert!(m2.is_loaded().await);

    // Now load m3, which exceeds limit of 2 -> should preemptively deallocate m1 (LRU)
    tokio::time::sleep(Duration::from_millis(15)).await;
    registry
        .ensure_loaded_with_deallocation("m3")
        .await
        .unwrap();

    assert_eq!(registry.count_loaded_models().await, 2);
    assert!(!m1.is_loaded().await, "m1 should have been deallocated");
    assert!(m2.is_loaded().await, "m2 should still be resident");
    assert!(m3.is_loaded().await, "m3 should now be resident");
}

#[tokio::test]
async fn test_registry_reactive_oom_deallocation_and_retry() {
    let _oom_lock = OOM_TEST_MUTEX.lock().await;
    let metrics = ServerMetrics::new_shared();
    let registry = ModelRegistry::new();
    registry.set_metrics(metrics.clone()).await;

    let m1 = ModelController::new(ModelConfig {
        id: "m1".into(),
        onnx_path: "mock:1".into(),
        min_residency: Duration::from_millis(0),
        ..Default::default()
    });
    let m2 = ModelController::new(ModelConfig {
        id: "m2".into(),
        onnx_path: "mock:2".into(),
        min_residency: Duration::from_millis(0),
        ..Default::default()
    });
    // m_oom fails with OOM once, then succeeds when retried
    let m_oom = ModelController::new(ModelConfig {
        id: "m_oom".into(),
        onnx_path: "mock:oom_once".into(),
        min_residency: Duration::from_millis(0),
        ..Default::default()
    });

    registry.register(m1.clone()).await;
    registry.register(m2.clone()).await;
    registry.register(m_oom.clone()).await;

    // Load m1 and m2
    m1.ensure_loaded().await.unwrap();
    tokio::time::sleep(Duration::from_millis(15)).await;
    m2.ensure_loaded().await.unwrap();
    assert!(m1.is_loaded().await);
    assert!(m2.is_loaded().await);

    // Set mock to fail once with OOM
    snooze_ml::backend::mock_backend::set_mock_oom_failures(1);

    // Load m_oom: should fail first with OOM, deallocate LRU (m1), then retry and succeed!
    registry
        .ensure_loaded_with_deallocation("m_oom")
        .await
        .unwrap();

    assert!(
        !m1.is_loaded().await,
        "m1 (LRU) should have been deallocated to resolve OOM"
    );
    assert!(m2.is_loaded().await, "m2 should remain loaded");
    assert!(
        m_oom.is_loaded().await,
        "m_oom should be loaded after deallocation retry"
    );
    snooze_ml::backend::mock_backend::set_mock_oom_failures(0);
}

#[tokio::test]
async fn test_registry_fatal_error_does_not_deallocate_healthy_models() {
    let registry = ModelRegistry::new();

    let healthy = ModelController::new(ModelConfig {
        id: "healthy".into(),
        onnx_path: "mock:healthy".into(),
        min_residency: Duration::from_millis(0),
        ..Default::default()
    });
    let corrupted = ModelController::new(ModelConfig {
        id: "corrupted".into(),
        onnx_path: "mock:corrupted".into(),
        min_residency: Duration::from_millis(0),
        ..Default::default()
    });

    registry.register(healthy.clone()).await;
    registry.register(corrupted.clone()).await;

    healthy.ensure_loaded().await.unwrap();
    assert!(healthy.is_loaded().await);

    // Attempt to load corrupted model (non-OOM failure)
    let res = registry.ensure_loaded_with_deallocation("corrupted").await;
    assert!(res.is_err());
    assert!(!res.as_ref().unwrap_err().is_out_of_memory());

    // Healthy model MUST NOT be deallocated on non-OOM errors!
    assert!(
        healthy.is_loaded().await,
        "Healthy model must NOT be deallocated for non-OOM errors"
    );
}

#[tokio::test]
async fn test_registry_resource_exhausted_when_all_candidates_busy() {
    let registry = ModelRegistry::new();

    let busy_model = ModelController::new(ModelConfig {
        id: "busy".into(),
        onnx_path: "mock:busy".into(),
        min_residency: Duration::from_millis(0),
        ..Default::default()
    });
    let oom_model = ModelController::new(ModelConfig {
        id: "oom".into(),
        onnx_path: "mock:oom".into(),
        ..Default::default()
    });

    registry.register(busy_model.clone()).await;
    registry.register(oom_model.clone()).await;

    busy_model.ensure_loaded().await.unwrap();

    // Submit an inference job with an ongoing receiver to keep the model pinned/busy
    // We can simulate busy by submitting a job or checking candidate selection
    let candidate = registry.find_lru_deallocation_candidate("oom").await;
    assert!(candidate.is_some());

    // Force failure: mock:oom will always fail with OOM.
    // First deallocation will unload busy_model.
    // Second attempt fails with OOM again, and now NO deallocatable models remain!
    let res = registry.ensure_loaded_with_deallocation("oom").await;
    assert!(matches!(res, Err(InferenceError::ResourceExhausted(_))));
}

#[tokio::test]
async fn test_registry_anti_thrashing_protection() {
    let metrics = ServerMetrics::new_shared();
    let registry = ModelRegistry::new();
    registry.set_metrics(metrics.clone()).await;

    // m_recent has a high min_residency (60s), while m_old has 0 min_residency
    let m_recent = ModelController::new(ModelConfig {
        id: "m_recent".into(),
        onnx_path: "mock:recent".into(),
        min_residency: Duration::from_secs(60),
        ..Default::default()
    });
    let m_old = ModelController::new(ModelConfig {
        id: "m_old".into(),
        onnx_path: "mock:old".into(),
        min_residency: Duration::from_millis(0),
        ..Default::default()
    });

    registry.register(m_recent.clone()).await;
    registry.register(m_old.clone()).await;

    m_recent.ensure_loaded().await.unwrap();
    tokio::time::sleep(Duration::from_millis(15)).await;
    m_old.ensure_loaded().await.unwrap();

    // Even though m_recent was loaded first and accessed earlier, m_recent is protected
    // by min_residency, so m_old must be selected as candidate!
    let candidate = registry.find_lru_deallocation_candidate("target").await;
    assert!(candidate.is_some());
    assert_eq!(candidate.unwrap().id(), "m_old");
}

#[tokio::test]
async fn test_registry_register_pinned_model_eagerly_loads() {
    let registry = ModelRegistry::new();
    let pinned_ctrl = ModelController::new(ModelConfig {
        id: "eager-pinned".into(),
        onnx_path: "mock:pinned".into(),
        pinned: true,
        ..Default::default()
    });

    assert!(!pinned_ctrl.is_loaded().await);
    registry.register(pinned_ctrl.clone()).await;

    // Pinned model should be eagerly loaded upon registration
    assert!(pinned_ctrl.is_loaded().await);
    assert!(pinned_ctrl.is_pinned());
    let list = registry.list_metadata().await;
    assert_eq!(list.len(), 1);
    assert!(list[0].pinned);
}

#[tokio::test]
async fn test_registry_lru_skips_pinned_models() {
    let registry = ModelRegistry::new();
    registry.set_max_loaded_models(Some(2)).await;

    // m_pinned is registered first (oldest access time) but is pinned
    let m_pinned = ModelController::new(ModelConfig {
        id: "m_pinned".into(),
        onnx_path: "mock:1".into(),
        min_residency: Duration::from_millis(0),
        pinned: true,
        ..Default::default()
    });
    let m_unpinned = ModelController::new(ModelConfig {
        id: "m_unpinned".into(),
        onnx_path: "mock:2".into(),
        min_residency: Duration::from_millis(0),
        pinned: false,
        ..Default::default()
    });
    let m_new = ModelController::new(ModelConfig {
        id: "m_new".into(),
        onnx_path: "mock:3".into(),
        min_residency: Duration::from_millis(0),
        pinned: false,
        ..Default::default()
    });

    registry.register(m_pinned.clone()).await;
    tokio::time::sleep(Duration::from_millis(15)).await;
    registry.register(m_unpinned.clone()).await;
    registry.register(m_new.clone()).await;

    m_unpinned.ensure_loaded().await.unwrap();
    assert!(m_pinned.is_loaded().await);
    assert!(m_unpinned.is_loaded().await);

    // LRU candidate must skip m_pinned and select m_unpinned
    let candidate = registry.find_lru_deallocation_candidate("m_new").await;
    assert!(candidate.is_some());
    assert_eq!(candidate.unwrap().id(), "m_unpinned");

    // Loading m_new under max_loaded_models = 2 must evict m_unpinned, keeping m_pinned resident
    registry
        .ensure_loaded_with_deallocation("m_new")
        .await
        .unwrap();
    assert!(
        m_pinned.is_loaded().await,
        "Pinned model must never be evicted by LRU capacity management"
    );
    assert!(
        !m_unpinned.is_loaded().await,
        "Unpinned model should have been evicted"
    );
    assert!(m_new.is_loaded().await, "New model should be loaded");
}

#[tokio::test]
async fn test_registry_pin_and_unpin_model() {
    let registry = ModelRegistry::new();
    let ctrl = ModelController::new(ModelConfig {
        id: "dyn-pin".into(),
        onnx_path: "mock:dyn".into(),
        min_residency: Duration::from_millis(0),
        pinned: false,
        ..Default::default()
    });
    registry.register(ctrl.clone()).await;

    assert!(!ctrl.is_pinned());
    assert!(!ctrl.is_loaded().await);

    // Pin via registry: loads and sets pinned = true
    registry.pin_model("dyn-pin").await.unwrap();
    assert!(ctrl.is_pinned());
    assert!(ctrl.is_loaded().await);
    assert!(
        registry
            .find_lru_deallocation_candidate("other")
            .await
            .is_none(),
        "Pinned model should not be an LRU candidate"
    );

    // Unpin via registry: remains loaded, but becomes eligible for LRU deallocation
    registry.unpin_model("dyn-pin").await.unwrap();
    assert!(!ctrl.is_pinned());
    assert!(ctrl.is_loaded().await);
    assert!(
        registry
            .find_lru_deallocation_candidate("other")
            .await
            .is_some(),
        "Unpinned model should now be an LRU candidate"
    );

    // Pin/unpin nonexistent model returns NotFound
    assert!(matches!(
        registry.pin_model("missing").await,
        Err(InferenceError::NotFound(_))
    ));
    assert!(matches!(
        registry.unpin_model("missing").await,
        Err(InferenceError::NotFound(_))
    ));
}

#[tokio::test]
async fn test_registry_concurrent_oom_storm_serialized() {
    let _oom_lock = OOM_TEST_MUTEX.lock().await;
    let metrics = ServerMetrics::new_shared();
    let registry = ModelRegistry::new();
    registry.set_metrics(metrics.clone()).await;

    let m1 = ModelController::new(ModelConfig {
        id: "storm_m1".into(),
        onnx_path: "mock:1".into(),
        min_residency: Duration::from_millis(0),
        ..Default::default()
    });
    let m2 = ModelController::new(ModelConfig {
        id: "storm_m2".into(),
        onnx_path: "mock:2".into(),
        min_residency: Duration::from_millis(0),
        ..Default::default()
    });
    let m_oom = ModelController::new(ModelConfig {
        id: "storm_oom".into(),
        onnx_path: "mock:oom_configurable".into(),
        min_residency: Duration::from_millis(0),
        ..Default::default()
    });

    registry.register(m1.clone()).await;
    registry.register(m2.clone()).await;
    registry.register(m_oom.clone()).await;

    // Pre-load m1 and m2
    m1.ensure_loaded().await.unwrap();
    tokio::time::sleep(Duration::from_millis(15)).await;
    m2.ensure_loaded().await.unwrap();
    assert!(m1.is_loaded().await);
    assert!(m2.is_loaded().await);
    assert_eq!(registry.count_loaded_models().await, 2);

    // Configure exactly 1 OOM failure before allocation succeeds
    snooze_ml::backend::mock_backend::set_mock_oom_failures(1);

    // Launch 10 concurrent requests to ensure_loaded_with_deallocation
    let mut handles = Vec::new();
    for _ in 0..10 {
        let reg = registry.clone();
        handles.push(tokio::spawn(async move {
            reg.ensure_loaded_with_deallocation("storm_oom").await
        }));
    }

    // Await all 10 handles: all must succeed without error
    for handle in handles {
        let res = handle.await.unwrap();
        assert!(res.is_ok(), "Expected success but got: {res:?}");
    }

    // Exactly 1 model should have been evicted (m1, the LRU candidate)
    assert!(
        !m1.is_loaded().await,
        "m1 (LRU) should have been deallocated once"
    );
    // m2 must still be resident (no cascade eviction)
    assert!(
        m2.is_loaded().await,
        "m2 should remain resident and not be cascade-evicted"
    );
    // m_oom must be loaded
    assert!(m_oom.is_loaded().await, "storm_oom must be loaded");
    assert_eq!(
        registry.count_loaded_models().await,
        2,
        "Expected exactly 2 loaded models (m2 and storm_oom)"
    );
    snooze_ml::backend::mock_backend::set_mock_oom_failures(0);
}
