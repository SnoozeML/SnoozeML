//! Integration and behavioral tests for model replica autoscaling, queue-aware routing,
//! and differential scale-to-zero lifecycles.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use ndarray::ArrayD;
use snooze_ml::model::{compute_replica_idle_timeout, ModelConfig, ModelController};
use snooze_ml::registry::ModelRegistry;
use snooze_ml::routing::{
    LeastQueueRouting, RandomRouting, ReplicaRouter, RoundRobinRouting, RoutingStrategy,
    RoutingStrategyType,
};
use snooze_ml::server::http::{create_router, AppState};
use snooze_ml::server::metrics::ServerMetrics;
use snooze_ml::sweeper::spawn_scale_to_zero_sweeper;
use snooze_ml::types::{ModelState, TensorPayload};
use std::time::Duration;
use tower::ServiceExt;

#[test]
fn test_compute_replica_idle_timeout_formula() {
    let base = Duration::from_secs(300);

    // Primary instance (replica_index == 0) always respects configured base timeout
    assert_eq!(
        compute_replica_idle_timeout(base, 1, 0),
        Duration::from_secs(300)
    );
    assert_eq!(
        compute_replica_idle_timeout(base, 2, 0),
        Duration::from_secs(300)
    );
    assert_eq!(
        compute_replica_idle_timeout(base, 5, 0),
        Duration::from_secs(300)
    );

    // When total_instances == 1, even if queried with non-zero index, stays at base
    assert_eq!(
        compute_replica_idle_timeout(base, 1, 1),
        Duration::from_secs(300)
    );

    // Additional replicas (replica_index > 0) scale down inversely with total instance count N
    // N = 2: 300 / 2 = 150s
    assert_eq!(
        compute_replica_idle_timeout(base, 2, 1),
        Duration::from_secs(150)
    );

    // N = 3: 300 / 3 = 100s
    assert_eq!(
        compute_replica_idle_timeout(base, 3, 1),
        Duration::from_secs(100)
    );
    assert_eq!(
        compute_replica_idle_timeout(base, 3, 2),
        Duration::from_secs(100)
    );

    // N = 5: 300 / 5 = 60s
    assert_eq!(
        compute_replica_idle_timeout(base, 5, 1),
        Duration::from_secs(60)
    );

    // N = 10: 300 / 10 = 30s
    assert_eq!(
        compute_replica_idle_timeout(base, 10, 1),
        Duration::from_secs(30)
    );

    // Millisecond resolution calculation
    let small_base = Duration::from_millis(500);
    assert_eq!(
        compute_replica_idle_timeout(small_base, 10, 1),
        Duration::from_millis(50)
    );
}

#[tokio::test]
async fn test_replica_router_least_queue_selection() {
    let cfg = ModelConfig {
        id: "routing-test".to_string(),
        onnx_path: "mock:test".to_string(),
        ..Default::default()
    };

    let r0 = ModelController::new_replica(cfg.clone(), 0);
    let r1 = ModelController::new_replica(cfg.clone(), 1);
    let r2 = ModelController::new_replica(cfg.clone(), 2);

    let replicas = vec![r0.clone(), r1.clone(), r2.clone()];
    let router = LeastQueueRouting;

    // Initially all queues are empty (queue_len == 0, active == 0): selects first
    let selected = router.select_replica(&replicas);
    assert!(selected.is_some());
    assert_eq!(selected.unwrap().replica_index(), 0);

    // Simulate load on r0: spawn an infer job so r0 has active load
    r0.ensure_loaded().await.unwrap();
    let r0_clone = r0.clone();
    let handle = tokio::spawn(async move {
        r0_clone
            .infer(vec![TensorPayload {
                name: "in".into(),
                shape: vec![1, 2],
                data: ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 2]), vec![1.0, 2.0]).unwrap(),
            }])
            .await
    });
    tokio::time::sleep(Duration::from_millis(5)).await;
    assert!(r0.active_requests() > 0);

    // With LeastQueueRouting, r1 or r2 will be picked over r0 because r0 has active load
    let selected_under_load = router.select_replica(&replicas);
    assert!(selected_under_load.is_some());
    assert_ne!(
        selected_under_load.unwrap().replica_index(),
        0,
        "r0 has higher queue/active load"
    );

    // Test wrapper ReplicaRouter
    let replica_router = ReplicaRouter::from_type(RoutingStrategyType::LeastQueue);
    let chosen = replica_router.select_replica(&replicas);
    assert!(chosen.is_some());

    let _ = handle.await;
}

#[tokio::test]
async fn test_replica_router_round_robin_selection() {
    let cfg = ModelConfig {
        id: "rr-test".to_string(),
        onnx_path: "mock:test".to_string(),
        ..Default::default()
    };

    let r0 = ModelController::new_replica(cfg.clone(), 0);
    let r1 = ModelController::new_replica(cfg.clone(), 1);
    let r2 = ModelController::new_replica(cfg.clone(), 2);

    let replicas = vec![r0.clone(), r1.clone(), r2.clone()];
    let router = RoundRobinRouting::new();

    let s0 = router.select_replica(&replicas).unwrap();
    assert_eq!(s0.replica_index(), 0);

    let s1 = router.select_replica(&replicas).unwrap();
    assert_eq!(s1.replica_index(), 1);

    let s2 = router.select_replica(&replicas).unwrap();
    assert_eq!(s2.replica_index(), 2);

    // Wraps around
    let s3 = router.select_replica(&replicas).unwrap();
    assert_eq!(s3.replica_index(), 0);
}

#[tokio::test]
async fn test_replica_router_random_selection() {
    let cfg = ModelConfig {
        id: "rand-test".to_string(),
        onnx_path: "mock:test".to_string(),
        ..Default::default()
    };

    let r0 = ModelController::new_replica(cfg.clone(), 0);
    let r1 = ModelController::new_replica(cfg.clone(), 1);

    let replicas = vec![r0.clone(), r1.clone()];
    let router = RandomRouting;

    let selected = router.select_replica(&replicas);
    assert!(selected.is_some());
}

#[tokio::test]
async fn test_autoscale_spins_up_replicas_when_queue_exceeded() {
    let registry = ModelRegistry::new();

    // Configure model with autoscale_queue_threshold = 1
    // As soon as 1 request is active on replica 0, next request exceeds queue and triggers scaling!
    let config = ModelConfig {
        id: "autoscale-model".to_string(),
        onnx_path: "mock:test".to_string(),
        idle_timeout: Duration::from_secs(60),
        max_batch_size: 4,
        max_batch_delay: Duration::from_millis(50),
        provider: None,
        description: None,
        min_residency: Duration::from_millis(0),
        autoscale_queue_threshold: 1,
        max_replicas: Some(4),
        routing_strategy: snooze_ml::routing::RoutingStrategyType::LeastQueue,
        pinned: false,
    };

    let controller = ModelController::new(config);
    registry.register(controller.clone()).await;

    let group = registry.get_group("autoscale-model").await.unwrap();
    assert_eq!(group.count_replicas().await, 1);

    // Initial route selects primary replica (index 0)
    let c1 = registry
        .route_for_inference("autoscale-model")
        .await
        .unwrap();
    assert_eq!(c1.replica_index(), 0);
    assert_eq!(group.count_replicas().await, 1);

    // Simulate active request on replica 0
    let input = TensorPayload {
        name: "in".into(),
        shape: vec![1, 2],
        data: ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 2]), vec![1.0, 2.0]).unwrap(),
    };

    // Spawn inference in background task: batcher delay is 50ms, so request stays active
    let c1_clone = c1.clone();
    let job_handle = tokio::spawn(async move { c1_clone.infer(vec![input]).await });

    // Wait 5ms for request to enter infer and register active_requests
    tokio::time::sleep(Duration::from_millis(5)).await;
    assert!(c1.active_requests() > 0);

    // Second request arrives while replica 0 has active_requests >= threshold (1)
    // Requests exceed the queue! Since resources are available, spin up another copy of the model!
    let c2 = registry
        .route_for_inference("autoscale-model")
        .await
        .unwrap();
    assert_eq!(c2.replica_index(), 1, "Should spin up replica index 1");
    assert_eq!(group.count_replicas().await, 2, "Group now has 2 replicas");

    // Wait for initial job to complete cleanly
    let job_res = job_handle.await.unwrap();
    assert!(job_res.is_ok());
}

#[tokio::test]
async fn test_autoscale_multi_replica_scaling() {
    let registry = ModelRegistry::new();

    let config = ModelConfig {
        id: "multi-scale".to_string(),
        onnx_path: "mock:test".to_string(),
        idle_timeout: Duration::from_secs(60),
        max_batch_size: 4,
        max_batch_delay: Duration::from_millis(100),
        provider: None,
        description: None,
        min_residency: Duration::from_millis(0),
        autoscale_queue_threshold: 1,
        max_replicas: Some(5),
        routing_strategy: snooze_ml::routing::RoutingStrategyType::LeastQueue,
        pinned: false,
    };

    let controller = ModelController::new(config);
    registry.register(controller).await;

    let group = registry.get_group("multi-scale").await.unwrap();
    assert_eq!(group.count_replicas().await, 1);

    let c0 = registry.route_for_inference("multi-scale").await.unwrap();
    assert_eq!(c0.replica_index(), 0);

    // Keep c0 busy
    let input0 = TensorPayload {
        name: "in".into(),
        shape: vec![1, 2],
        data: ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 2]), vec![1.0, 2.0]).unwrap(),
    };
    let c0_clone = c0.clone();
    let handle0 = tokio::spawn(async move { c0_clone.infer(vec![input0]).await });
    tokio::time::sleep(Duration::from_millis(5)).await;

    // Request 2 arrives: triggers scale-up to replica 1
    let c1 = registry.route_for_inference("multi-scale").await.unwrap();
    assert_eq!(c1.replica_index(), 1);
    assert_eq!(group.count_replicas().await, 2);

    // Keep c1 busy
    let input1 = TensorPayload {
        name: "in".into(),
        shape: vec![1, 2],
        data: ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 2]), vec![1.0, 2.0]).unwrap(),
    };
    let c1_clone = c1.clone();
    let handle1 = tokio::spawn(async move { c1_clone.infer(vec![input1]).await });
    tokio::time::sleep(Duration::from_millis(5)).await;

    // Request 3 arrives: both replica 0 and replica 1 are busy >= threshold (1)
    // "Not just one, autoscale all the ones needed!" -> spins up replica 2!
    let c2 = registry.route_for_inference("multi-scale").await.unwrap();
    assert_eq!(c2.replica_index(), 2);
    assert_eq!(group.count_replicas().await, 3);

    let _ = handle0.await;
    let _ = handle1.await;
}

#[tokio::test]
async fn test_autoscale_respects_max_replicas_limit() {
    let registry = ModelRegistry::new();

    // Cap at max_replicas: 2
    let config = ModelConfig {
        id: "capped-scale".to_string(),
        onnx_path: "mock:test".to_string(),
        idle_timeout: Duration::from_secs(60),
        max_batch_size: 2,
        max_batch_delay: Duration::from_millis(100),
        provider: None,
        description: None,
        min_residency: Duration::from_millis(0),
        autoscale_queue_threshold: 1,
        max_replicas: Some(2),
        routing_strategy: snooze_ml::routing::RoutingStrategyType::LeastQueue,
        pinned: false,
    };

    let controller = ModelController::new(config);
    registry.register(controller).await;
    let group = registry.get_group("capped-scale").await.unwrap();

    let c0 = registry.route_for_inference("capped-scale").await.unwrap();
    let c0_clone = c0.clone();
    let h0 = tokio::spawn(async move {
        c0_clone
            .infer(vec![TensorPayload {
                name: "in".into(),
                shape: vec![1, 2],
                data: ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 2]), vec![1.0, 2.0]).unwrap(),
            }])
            .await
    });
    tokio::time::sleep(Duration::from_millis(5)).await;

    // First scale-up succeeds: replica count becomes 2
    let c1 = registry.route_for_inference("capped-scale").await.unwrap();
    assert_eq!(c1.replica_index(), 1);
    assert_eq!(group.count_replicas().await, 2);

    let c1_clone = c1.clone();
    let h1 = tokio::spawn(async move {
        c1_clone
            .infer(vec![TensorPayload {
                name: "in".into(),
                shape: vec![1, 2],
                data: ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 2]), vec![1.0, 2.0]).unwrap(),
            }])
            .await
    });
    tokio::time::sleep(Duration::from_millis(5)).await;

    // Both replicas busy, but max_replicas is 2: must NOT spin up replica 3!
    // Instead, gracefully routes to existing lowest-queue replica
    let c_fallback = registry.route_for_inference("capped-scale").await.unwrap();
    assert_eq!(group.count_replicas().await, 2);
    assert!(c_fallback.replica_index() < 2);

    let _ = h0.await;
    let _ = h1.await;
}

#[tokio::test]
async fn test_autoscale_respects_max_loaded_models_limit() {
    let registry = ModelRegistry::new();
    // Hard cap on total resident models = 2
    registry.set_max_loaded_models(Some(2)).await;

    let config = ModelConfig {
        id: "capacity-capped".to_string(),
        onnx_path: "mock:test".to_string(),
        idle_timeout: Duration::from_secs(60),
        max_batch_size: 2,
        max_batch_delay: Duration::from_millis(100),
        provider: None,
        description: None,
        min_residency: Duration::from_millis(0),
        autoscale_queue_threshold: 1,
        max_replicas: None,
        routing_strategy: snooze_ml::routing::RoutingStrategyType::LeastQueue,
        pinned: false,
    };

    let controller = ModelController::new(config);
    registry.register(controller).await;

    let c0 = registry
        .route_for_inference("capacity-capped")
        .await
        .unwrap();
    c0.ensure_loaded().await.unwrap();

    // Register another model to take up the second slot
    let other_cfg = ModelConfig {
        id: "other-model".to_string(),
        onnx_path: "mock:other".to_string(),
        ..Default::default()
    };
    let other_ctrl = ModelController::new(other_cfg);
    registry.register(other_ctrl.clone()).await;
    other_ctrl.ensure_loaded().await.unwrap();

    // Now 2 models are loaded (max capacity reached)
    assert_eq!(registry.count_loaded_models().await, 2);

    // Make c0 and other_ctrl busy
    let c0_clone = c0.clone();
    let h0 = tokio::spawn(async move {
        c0_clone
            .infer(vec![TensorPayload {
                name: "in".into(),
                shape: vec![1, 2],
                data: ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 2]), vec![1.0, 2.0]).unwrap(),
            }])
            .await
    });
    let other_clone = other_ctrl.clone();
    let h_other = tokio::spawn(async move {
        other_clone
            .infer(vec![TensorPayload {
                name: "in".into(),
                shape: vec![1, 2],
                data: ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 2]), vec![1.0, 2.0]).unwrap(),
            }])
            .await
    });
    tokio::time::sleep(Duration::from_millis(5)).await;

    // Both slots occupied by busy models, so no idle models can be evicted
    // Capacity limit reached with no deallocatable idle models: cannot autoscale, routes to c0
    let routed = registry
        .route_for_inference("capacity-capped")
        .await
        .unwrap();
    assert_eq!(routed.replica_index(), 0);

    let _ = h0.await;
    let _ = h_other.await;
}

#[tokio::test]
async fn test_replica_differential_scale_to_zero_sweeper() {
    let registry = ModelRegistry::new();

    // Base idle timeout = 100ms
    // For 2 replicas:
    // - Replica 0 respects base timeout = 100ms
    // - Replica 1 deallocates more rapidly = 100ms / 2 = 50ms!
    let config = ModelConfig {
        id: "differential-s2z".to_string(),
        onnx_path: "mock:test".to_string(),
        idle_timeout: Duration::from_millis(100),
        max_batch_size: 2,
        max_batch_delay: Duration::from_millis(5),
        provider: None,
        description: None,
        min_residency: Duration::from_millis(0),
        autoscale_queue_threshold: 1,
        max_replicas: Some(3),
        routing_strategy: snooze_ml::routing::RoutingStrategyType::LeastQueue,
        pinned: false,
    };

    let controller = ModelController::new(config.clone());
    registry.register(controller.clone()).await;

    let group = registry.get_group("differential-s2z").await.unwrap();

    // Manually spawn replica 1 to have 2 replicas
    let replica1 = ModelController::new_replica(config.clone(), 1);
    replica1.ensure_loaded().await.unwrap();
    controller.ensure_loaded().await.unwrap();

    group.add_replica(replica1.clone()).await;

    assert_eq!(group.count_replicas().await, 2);
    assert_eq!(controller.metadata().await.state, ModelState::Ready);
    assert_eq!(replica1.metadata().await.state, ModelState::Ready);

    // Synchronize access timestamps
    controller.touch();
    replica1.touch();

    // Launch sweeper with 5ms check interval
    let sweeper = spawn_scale_to_zero_sweeper(registry.clone(), Duration::from_millis(5));

    // Poll until replica 1 is pruned (scaled timeout = 50ms; replica 0 base timeout = 100ms)
    let mut replica1_pruned = false;
    for _ in 0..18 {
        tokio::time::sleep(Duration::from_millis(5)).await;
        if group.count_replicas().await == 1 {
            replica1_pruned = true;
            break;
        }
    }

    assert!(
        replica1_pruned,
        "Additional replica must deallocate rapidly and be pruned from group"
    );
    assert_eq!(
        controller.metadata().await.state,
        ModelState::Ready,
        "Primary instance must still be Ready (respecting configured minimum lifetime)"
    );

    // Wait until primary instance also reaches its 100ms timeout and scales to zero
    let mut primary_idle = false;
    for _ in 0..20 {
        tokio::time::sleep(Duration::from_millis(10)).await;
        if controller.metadata().await.state == ModelState::Idle {
            primary_idle = true;
            break;
        }
    }

    assert!(
        primary_idle,
        "Primary instance deallocates to Idle after full configured idle timeout"
    );
    assert_eq!(
        group.count_replicas().await,
        1,
        "Primary instance is never pruned from group"
    );

    sweeper.abort();
}

#[tokio::test]
async fn test_e2e_autoscale_infer_http_endpoint() {
    let registry = ModelRegistry::new();
    let metrics = ServerMetrics::new_shared();

    let config = ModelConfig {
        id: "e2e-autoscale".to_string(),
        onnx_path: "mock:test".to_string(),
        idle_timeout: Duration::from_secs(60),
        max_batch_size: 4,
        max_batch_delay: Duration::from_millis(100),
        provider: None,
        description: None,
        min_residency: Duration::from_millis(0),
        autoscale_queue_threshold: 1,
        max_replicas: Some(3),
        routing_strategy: snooze_ml::routing::RoutingStrategyType::LeastQueue,
        pinned: false,
    };

    registry.register(ModelController::new(config)).await;

    let app = create_router(AppState::new(registry.clone(), metrics.clone()));

    let payload = serde_json::json!({
        "inputs": [
            {
                "name": "input",
                "shape": [1, 2],
                "data": [1.0, 2.0]
            }
        ]
    });

    // Send first request
    let app1 = app.clone();
    let payload1 = payload.clone();
    let handle1 = tokio::spawn(async move {
        app1.oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/models/e2e-autoscale/infer")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&payload1).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap()
    });

    // Give first request a moment to begin executing
    tokio::time::sleep(Duration::from_millis(5)).await;

    // Send second request while first is executing
    // Since autoscale_queue_threshold is 1, this triggers autoscaling a new replica!
    let app2 = app.clone();
    let payload2 = payload.clone();
    let handle2 = tokio::spawn(async move {
        app2.oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/models/e2e-autoscale/infer")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&payload2).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap()
    });

    let resp1 = handle1.await.unwrap();
    let resp2 = handle2.await.unwrap();

    assert_eq!(resp1.status(), StatusCode::OK);
    assert_eq!(resp2.status(), StatusCode::OK);

    let body1 = resp1.into_body().collect().await.unwrap().to_bytes();
    let body2 = resp2.into_body().collect().await.unwrap().to_bytes();

    let json1: serde_json::Value = serde_json::from_slice(&body1).unwrap();
    let json2: serde_json::Value = serde_json::from_slice(&body2).unwrap();

    assert_eq!(json1["model_id"], "e2e-autoscale");
    assert_eq!(json2["model_id"], "e2e-autoscale");

    // Verify replicas were autoscaled to handle the concurrent requests
    let group = registry.get_group("e2e-autoscale").await.unwrap();
    assert!(
        group.count_replicas().await >= 2,
        "Expected at least 2 replicas after concurrent infer"
    );
}

#[tokio::test]
async fn test_autoscale_unbounded_with_none_or_zero() {
    let registry = ModelRegistry::new();

    // Configure model with max_replicas: None (unbounded)
    let config = ModelConfig {
        id: "unbounded-model".to_string(),
        onnx_path: "mock:test".to_string(),
        idle_timeout: Duration::from_secs(60),
        max_batch_size: 2,
        max_batch_delay: Duration::from_millis(100),
        provider: None,
        description: None,
        min_residency: Duration::from_millis(0),
        autoscale_queue_threshold: 1,
        max_replicas: None, // Unbounded
        routing_strategy: snooze_ml::routing::RoutingStrategyType::LeastQueue,
        pinned: false,
    };

    let controller = ModelController::new(config);
    registry.register(controller).await;

    let group = registry.get_group("unbounded-model").await.unwrap();
    let c0 = registry
        .route_for_inference("unbounded-model")
        .await
        .unwrap();

    let input = TensorPayload {
        name: "in".into(),
        shape: vec![1, 2],
        data: ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 2]), vec![1.0, 2.0]).unwrap(),
    };

    let c0_clone = c0.clone();
    let in0 = input.clone();
    let _h0 = tokio::spawn(async move { c0_clone.infer(vec![in0]).await });
    tokio::time::sleep(Duration::from_millis(5)).await;

    // Spin up replica 1
    let c1 = registry
        .route_for_inference("unbounded-model")
        .await
        .unwrap();
    let c1_clone = c1.clone();
    let in1 = input.clone();
    let _h1 = tokio::spawn(async move { c1_clone.infer(vec![in1]).await });
    tokio::time::sleep(Duration::from_millis(5)).await;

    // Spin up replica 2
    let c2 = registry
        .route_for_inference("unbounded-model")
        .await
        .unwrap();
    let c2_clone = c2.clone();
    let in2 = input.clone();
    let _h2 = tokio::spawn(async move { c2_clone.infer(vec![in2]).await });
    tokio::time::sleep(Duration::from_millis(5)).await;

    // Spin up replica 3
    let c3 = registry
        .route_for_inference("unbounded-model")
        .await
        .unwrap();
    assert_eq!(c3.replica_index(), 3);
    assert_eq!(
        group.count_replicas().await,
        4,
        "Can scale past arbitrary limits when unbounded"
    );
}

#[tokio::test]
async fn test_autoscale_multi_replica_cascading_scale_down() {
    let registry = ModelRegistry::new();

    // Base idle timeout: 200ms
    // With 4 replicas: surplus replicas scale down after 200ms / 4 = 50ms
    let config = ModelConfig {
        id: "burst-scale-down".to_string(),
        onnx_path: "mock:test".to_string(),
        idle_timeout: Duration::from_millis(200),
        max_batch_size: 2,
        max_batch_delay: Duration::from_millis(50),
        provider: None,
        description: None,
        min_residency: Duration::from_millis(0),
        autoscale_queue_threshold: 1,
        max_replicas: Some(5),
        routing_strategy: snooze_ml::routing::RoutingStrategyType::LeastQueue,
        pinned: false,
    };

    let controller = ModelController::new(config);
    registry.register(controller.clone()).await;

    let group = registry.get_group("burst-scale-down").await.unwrap();

    let input = TensorPayload {
        name: "in".into(),
        shape: vec![1, 2],
        data: ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 2]), vec![1.0, 2.0]).unwrap(),
    };

    // Burst 4 concurrent requests to force scaling up to 4 replicas (c0, c1, c2, c3)
    let c0 = registry
        .route_for_inference("burst-scale-down")
        .await
        .unwrap();
    let c0_c = c0.clone();
    let in0 = input.clone();
    let h0 = tokio::spawn(async move { c0_c.infer(vec![in0]).await });
    tokio::time::sleep(Duration::from_millis(5)).await;

    let c1 = registry
        .route_for_inference("burst-scale-down")
        .await
        .unwrap();
    let c1_c = c1.clone();
    let in1 = input.clone();
    let h1 = tokio::spawn(async move { c1_c.infer(vec![in1]).await });
    tokio::time::sleep(Duration::from_millis(5)).await;

    let c2 = registry
        .route_for_inference("burst-scale-down")
        .await
        .unwrap();
    let c2_c = c2.clone();
    let in2 = input.clone();
    let h2 = tokio::spawn(async move { c2_c.infer(vec![in2]).await });
    tokio::time::sleep(Duration::from_millis(5)).await;

    let c3 = registry
        .route_for_inference("burst-scale-down")
        .await
        .unwrap();
    let c3_c = c3.clone();
    let in3 = input.clone();
    let h3 = tokio::spawn(async move { c3_c.infer(vec![in3]).await });

    assert_eq!(
        group.count_replicas().await,
        4,
        "Autoscaled to 4 replicas under queue pressure"
    );

    // Start background sweeper with 5ms tick
    let sweeper = spawn_scale_to_zero_sweeper(registry.clone(), Duration::from_millis(5));

    // Await all 4 inference tasks: traffic completes and collapses to zero
    let (r0, r1, r2, r3) = tokio::join!(h0, h1, h2, h3);
    assert!(r0.unwrap().is_ok());
    assert!(r1.unwrap().is_ok());
    assert!(r2.unwrap().is_ok());
    assert!(r3.unwrap().is_ok());

    // Poll until surplus replicas are pruned down to 1
    // Surplus timeout is 200ms / 4 = 50ms, so within ~80-120ms they should be deallocated and pruned
    let mut pruned_to_one = false;
    for _ in 0..30 {
        tokio::time::sleep(Duration::from_millis(10)).await;
        if group.count_replicas().await == 1 {
            pruned_to_one = true;
            break;
        }
    }

    assert!(
        pruned_to_one,
        "Surplus replicas must scale down and be pruned down to 1"
    );

    // Primary replica R0 must NOT have been pruned
    let replicas = group.all_replicas().await;
    assert_eq!(replicas.len(), 1);
    assert_eq!(replicas[0].replica_index(), 0);

    // Primary replica must still be Ready (since its full 200ms timeout has not elapsed yet)
    assert_eq!(
        controller.metadata().await.state,
        ModelState::Ready,
        "Primary replica must remain Ready while its base idle timeout has not elapsed"
    );

    // Now wait for primary replica to reach full 200ms idle timeout
    let mut primary_idle = false;
    for _ in 0..25 {
        tokio::time::sleep(Duration::from_millis(10)).await;
        if controller.metadata().await.state == ModelState::Idle {
            primary_idle = true;
            break;
        }
    }

    assert!(
        primary_idle,
        "Primary replica deallocates to Idle after full base idle timeout"
    );
    assert_eq!(
        group.count_replicas().await,
        1,
        "Primary replica is never pruned"
    );

    sweeper.abort();
}
