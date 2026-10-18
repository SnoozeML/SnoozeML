//! End-to-end integration tests verifying HTTP endpoints, dynamic batching, and scale-to-zero.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use snooze_ml::model::{ModelConfig, ModelController};
use snooze_ml::registry::ModelRegistry;
use snooze_ml::server::http::{create_router, AppState};
use snooze_ml::server::metrics::ServerMetrics;
use std::time::Duration;
use tower::ServiceExt;

#[tokio::test]
async fn test_e2e_home_endpoint() {
    let registry = ModelRegistry::new();
    let metrics = ServerMetrics::new_shared();
    let app = create_router(AppState::new(registry.clone(), metrics.clone()));

    // 1. GET /
    let response = app
        .clone()
        .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["service"], "snooze-ml");
    assert_eq!(json["status"], "ok");
    assert_eq!(json["version"], env!("CARGO_PKG_VERSION"));
    assert_eq!(json["endpoints"]["home"], "/");
    assert_eq!(json["endpoints"]["models"], "/v1/models");
    assert_eq!(json["endpoints"]["models_upload"], "/v1/models/upload");
    assert_eq!(json["endpoints"]["models_refresh"], "/v1/models/refresh");
    assert_eq!(json["endpoints"]["health_live"], "/health/live");
    assert_eq!(json["endpoints"]["health_ready"], "/health/ready");
    assert_eq!(json["endpoints"]["metrics"], "/metrics");

    // 2. HEAD /
    let head_resp = app
        .oneshot(
            Request::builder()
                .method("HEAD")
                .uri("/")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(head_resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_e2e_health_and_models_endpoints() {
    let registry = ModelRegistry::new();
    let metrics = ServerMetrics::new_shared();

    let config = ModelConfig {
        id: "mock-resnet".to_string(),
        onnx_path: "mock:test".to_string(),
        idle_timeout: Duration::from_secs(60),
        max_batch_size: 4,
        max_batch_delay: Duration::from_millis(10),
        provider: None,
        description: Some("Custom ResNet description".to_string()),
        min_residency: Duration::from_millis(0),
        ..Default::default()
    };
    registry.register(ModelController::new(config)).await;

    let app = create_router(AppState::new(registry.clone(), metrics.clone()));

    // 1. Check /health/live
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/health/live")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    // 2. Check /v1/models
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let body_str = std::str::from_utf8(&body).unwrap();
    assert!(body_str.contains("mock-resnet"));
    assert!(body_str.contains("Custom ResNet description"));
}

#[tokio::test]
async fn test_e2e_infer_and_metrics() {
    let registry = ModelRegistry::new();
    let metrics = ServerMetrics::new_shared();

    let config = ModelConfig {
        id: "mock-infer-target".to_string(),
        onnx_path: "mock:test".to_string(),
        idle_timeout: Duration::from_secs(60),
        max_batch_size: 4,
        max_batch_delay: Duration::from_millis(10),
        provider: None,
        description: None,
        min_residency: Duration::from_millis(0),
        ..Default::default()
    };
    registry.register(ModelController::new(config)).await;

    let app = create_router(AppState::new(registry.clone(), metrics.clone()));

    let payload = serde_json::json!({
        "inputs": [
            {
                "name": "input_ids",
                "shape": [1, 2],
                "data": [10.0, 20.0]
            }
        ]
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/models/mock-infer-target/infer")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();

    let response = app.clone().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let json_resp: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(json_resp["model_id"], "mock-infer-target");
    assert!(json_resp["outputs"].is_array());
    assert_eq!(json_resp["outputs"][0]["name"], "output");
    assert_eq!(json_resp["outputs"][0]["shape"], serde_json::json!([1, 4]));

    // Check /metrics
    let metrics_resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(metrics_resp.status(), StatusCode::OK);
    let metrics_body = metrics_resp.into_body().collect().await.unwrap().to_bytes();
    let metrics_str = std::str::from_utf8(&metrics_body).unwrap();
    assert!(metrics_str.contains("snoozeml_requests_total{status=\"success\"} 1"));
}

#[tokio::test]
async fn test_e2e_readiness_endpoint() {
    let registry = ModelRegistry::new();
    let metrics = ServerMetrics::new_shared();
    let app = create_router(AppState::new(registry, metrics));

    let response = app
        .oneshot(
            Request::builder()
                .uri("/health/ready")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["status"], "ready");
}

#[tokio::test]
async fn test_e2e_model_load_and_unload_lifecycle() {
    let registry = ModelRegistry::new();
    let metrics = ServerMetrics::new_shared();

    let config = ModelConfig {
        id: "lifecycle-target".to_string(),
        onnx_path: "mock:test".to_string(),
        idle_timeout: Duration::from_secs(60),
        max_batch_size: 4,
        max_batch_delay: Duration::from_millis(10),
        provider: None,
        description: None,
        min_residency: Duration::from_millis(0),
        ..Default::default()
    };
    let controller = ModelController::new(config);
    registry.register(controller.clone()).await;

    let app = create_router(AppState::new(registry, metrics));

    // Initial state: Idle
    assert_eq!(
        controller.metadata().await.state,
        snooze_ml::types::ModelState::Idle
    );

    // 1. Explicit load via API
    let load_req = Request::builder()
        .method("POST")
        .uri("/v1/models/lifecycle-target/load")
        .body(Body::empty())
        .unwrap();
    let load_resp = app.clone().oneshot(load_req).await.unwrap();
    assert_eq!(load_resp.status(), StatusCode::OK);
    let load_body = load_resp.into_body().collect().await.unwrap().to_bytes();
    let load_json: serde_json::Value = serde_json::from_slice(&load_body).unwrap();
    assert_eq!(load_json["model_id"], "lifecycle-target");
    assert_eq!(load_json["status"], "loaded");
    assert_eq!(
        controller.metadata().await.state,
        snooze_ml::types::ModelState::Ready
    );

    // 2. Explicit unload via API
    let unload_req = Request::builder()
        .method("POST")
        .uri("/v1/models/lifecycle-target/unload")
        .body(Body::empty())
        .unwrap();
    let unload_resp = app.clone().oneshot(unload_req).await.unwrap();
    assert_eq!(unload_resp.status(), StatusCode::OK);
    let unload_body = unload_resp.into_body().collect().await.unwrap().to_bytes();
    let unload_json: serde_json::Value = serde_json::from_slice(&unload_body).unwrap();
    assert_eq!(unload_json["model_id"], "lifecycle-target");
    assert_eq!(unload_json["status"], "unloaded");
    assert_eq!(
        controller.metadata().await.state,
        snooze_ml::types::ModelState::Idle
    );

    // 3. Load non-existent model -> 404
    let ghost_load_req = Request::builder()
        .method("POST")
        .uri("/v1/models/ghost-model/load")
        .body(Body::empty())
        .unwrap();
    let ghost_load_resp = app.clone().oneshot(ghost_load_req).await.unwrap();
    assert_eq!(ghost_load_resp.status(), StatusCode::NOT_FOUND);

    // 4. Unload non-existent model -> 404
    let ghost_unload_req = Request::builder()
        .method("POST")
        .uri("/v1/models/ghost-model/unload")
        .body(Body::empty())
        .unwrap();
    let ghost_unload_resp = app.clone().oneshot(ghost_unload_req).await.unwrap();
    assert_eq!(ghost_unload_resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_e2e_infer_model_not_found() {
    let registry = ModelRegistry::new();
    let metrics = ServerMetrics::new_shared();
    let app = create_router(AppState::new(registry, metrics));

    let payload = serde_json::json!({
        "inputs": [
            {
                "name": "input_ids",
                "shape": [1, 2],
                "data": [10.0, 20.0]
            }
        ]
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/models/non-existent-model/infer")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let err_json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert!(err_json["error"]
        .as_str()
        .unwrap()
        .contains("not registered"));
}

#[tokio::test]
async fn test_e2e_infer_malformed_json_payload() {
    let registry = ModelRegistry::new();
    let metrics = ServerMetrics::new_shared();

    let config = ModelConfig {
        id: "mock-target".to_string(),
        onnx_path: "mock:test".to_string(),
        idle_timeout: Duration::from_secs(60),
        max_batch_size: 4,
        max_batch_delay: Duration::from_millis(10),
        provider: None,
        description: None,
        min_residency: Duration::from_millis(0),
        ..Default::default()
    };
    registry.register(ModelController::new(config)).await;

    let app = create_router(AppState::new(registry, metrics));

    let request = Request::builder()
        .method("POST")
        .uri("/v1/models/mock-target/infer")
        .header("content-type", "application/json")
        .body(Body::from(b"{ malformed json string".to_vec()))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    // Axum Json extractor returns 400 or 422 for malformed json
    assert!(
        response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
    );
}

#[tokio::test]
async fn test_e2e_infer_shape_element_mismatch() {
    let registry = ModelRegistry::new();
    let metrics = ServerMetrics::new_shared();

    let config = ModelConfig {
        id: "mock-target".to_string(),
        onnx_path: "mock:test".to_string(),
        idle_timeout: Duration::from_secs(60),
        max_batch_size: 4,
        max_batch_delay: Duration::from_millis(10),
        provider: None,
        description: None,
        min_residency: Duration::from_millis(0),
        ..Default::default()
    };
    registry.register(ModelController::new(config)).await;

    let app = create_router(AppState::new(registry, metrics));

    // Shape specifies [1, 4] (4 elements), but data provides only 2 elements
    let payload = serde_json::json!({
        "inputs": [
            {
                "name": "input_ids",
                "shape": [1, 4],
                "data": [10.0, 20.0]
            }
        ]
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/models/mock-target/infer")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let err_json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert!(err_json["error"]
        .as_str()
        .unwrap()
        .contains("expects 4 elements"));
}

#[tokio::test]
async fn test_e2e_infer_zero_batch_size_error() {
    let registry = ModelRegistry::new();
    let metrics = ServerMetrics::new_shared();

    let config = ModelConfig {
        id: "mock-target".to_string(),
        onnx_path: "mock:test".to_string(),
        idle_timeout: Duration::from_secs(60),
        max_batch_size: 4,
        max_batch_delay: Duration::from_millis(10),
        provider: None,
        description: None,
        min_residency: Duration::from_millis(0),
        ..Default::default()
    };
    registry.register(ModelController::new(config)).await;

    let app = create_router(AppState::new(registry, metrics));

    let payload = serde_json::json!({
        "inputs": [
            {
                "name": "input_ids",
                "shape": [0, 2],
                "data": []
            }
        ]
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/models/mock-target/infer")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let err_json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert!(err_json["error"].as_str().unwrap().contains("0 batch size"));
}

#[tokio::test]
async fn test_e2e_infer_backend_load_failure_and_metrics() {
    let registry = ModelRegistry::new();
    let metrics = ServerMetrics::new_shared();

    let config = ModelConfig {
        id: "broken-onnx".to_string(),
        onnx_path: "/tmp/snooze_non_existent_unreachable_model.onnx".to_string(),
        idle_timeout: Duration::from_secs(60),
        max_batch_size: 4,
        max_batch_delay: Duration::from_millis(10),
        provider: None,
        description: None,
        min_residency: Duration::from_millis(0),
        ..Default::default()
    };
    registry.register(ModelController::new(config)).await;

    let app = create_router(AppState::new(registry, metrics.clone()));

    let payload = serde_json::json!({
        "inputs": [
            {
                "name": "in",
                "shape": [1, 2],
                "data": [1.0, 2.0]
            }
        ]
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/models/broken-onnx/infer")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();

    let response = app.clone().oneshot(request).await.unwrap();
    assert!(
        response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
    );

    // Verify model state is Idle, not stuck in Loading
    let list_req = Request::builder()
        .method("GET")
        .uri("/v1/models")
        .body(Body::empty())
        .unwrap();
    let list_resp = app.clone().oneshot(list_req).await.unwrap();
    assert_eq!(list_resp.status(), StatusCode::OK);
    let list_body = list_resp.into_body().collect().await.unwrap().to_bytes();
    let models: Vec<snooze_ml::types::ModelMetadata> = serde_json::from_slice(&list_body).unwrap();
    let broken = models.iter().find(|m| m.id == "broken-onnx").unwrap();
    assert_eq!(broken.state, snooze_ml::types::ModelState::Idle);

    let prom = metrics.format_prometheus();
    assert!(prom.contains("snoozeml_requests_total{status=\"error\"} 1"));
}

#[tokio::test]
async fn test_e2e_explicit_load_failure_reverts_state_to_idle() {
    let registry = ModelRegistry::new();
    let metrics = ServerMetrics::new_shared();

    let config = ModelConfig {
        id: "broken-explicit-load".to_string(),
        onnx_path: "/tmp/snooze_non_existent_unreachable_model.onnx".to_string(),
        idle_timeout: Duration::from_secs(60),
        max_batch_size: 4,
        max_batch_delay: Duration::from_millis(10),
        provider: None,
        description: None,
        min_residency: Duration::from_millis(0),
        ..Default::default()
    };
    let controller = ModelController::new(config);
    registry.register(controller.clone()).await;

    let app = create_router(AppState::new(registry, metrics.clone()));

    // Initial state is Idle
    assert_eq!(
        controller.metadata().await.state,
        snooze_ml::types::ModelState::Idle
    );

    // Call POST /v1/models/broken-explicit-load/load
    let load_req = Request::builder()
        .method("POST")
        .uri("/v1/models/broken-explicit-load/load")
        .body(Body::empty())
        .unwrap();
    let load_resp = app.clone().oneshot(load_req).await.unwrap();
    assert!(
        load_resp.status() == StatusCode::NOT_FOUND
            || load_resp.status() == StatusCode::INTERNAL_SERVER_ERROR
    );

    // Model controller state MUST revert to Idle, NOT remain in Loading
    assert_eq!(
        controller.metadata().await.state,
        snooze_ml::types::ModelState::Idle
    );

    // GET /v1/models endpoint must report state "idle"
    let list_req = Request::builder()
        .method("GET")
        .uri("/v1/models")
        .body(Body::empty())
        .unwrap();
    let list_resp = app.oneshot(list_req).await.unwrap();
    assert_eq!(list_resp.status(), StatusCode::OK);
    let list_body = list_resp.into_body().collect().await.unwrap().to_bytes();
    let models: Vec<snooze_ml::types::ModelMetadata> = serde_json::from_slice(&list_body).unwrap();
    let target = models
        .iter()
        .find(|m| m.id == "broken-explicit-load")
        .expect("Model should be listed");
    assert_eq!(target.state, snooze_ml::types::ModelState::Idle);
}

#[tokio::test]
async fn test_e2e_concurrent_inference_requests() {
    let registry = ModelRegistry::new();
    let metrics = ServerMetrics::new_shared();

    let config = ModelConfig {
        id: "concurrent-target".to_string(),
        onnx_path: "mock:test".to_string(),
        idle_timeout: Duration::from_secs(60),
        max_batch_size: 4,
        max_batch_delay: Duration::from_millis(50),
        provider: None,
        description: None,
        min_residency: Duration::from_millis(0),
        ..Default::default()
    };
    registry.register(ModelController::new(config)).await;

    let app = create_router(AppState::new(registry, metrics.clone()));

    let mut handles = Vec::new();
    for i in 0..4 {
        let app_clone = app.clone();
        let handle = tokio::spawn(async move {
            let payload = serde_json::json!({
                "inputs": [
                    {
                        "name": "input_ids",
                        "shape": [1, 2],
                        "data": [(i as f32), (i as f32) + 1.0]
                    }
                ]
            });

            let request = Request::builder()
                .method("POST")
                .uri("/v1/models/concurrent-target/infer")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&payload).unwrap()))
                .unwrap();

            let response = app_clone.oneshot(request).await.unwrap();
            assert_eq!(response.status(), StatusCode::OK);

            let body = response.into_body().collect().await.unwrap().to_bytes();
            let json_resp: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(json_resp["model_id"], "concurrent-target");
            assert_eq!(json_resp["outputs"][0]["shape"], serde_json::json!([1, 4]));
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.await.unwrap();
    }

    let prom = metrics.format_prometheus();
    assert!(prom.contains("snoozeml_requests_total{status=\"success\"} 4"));
}

#[tokio::test]
async fn test_e2e_real_onnx_model_skippable() {
    use std::path::PathBuf;

    let candidate_path = std::env::var("SNOOZE_TEST_ONNX_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("tests/fixtures/test_model.onnx"));

    if !candidate_path.is_file() {
        println!(
            "SKIPPED: test_e2e_real_onnx_model_skippable: no test ONNX model found at {:?}. Set SNOOZE_TEST_ONNX_PATH or place model at tests/fixtures/test_model.onnx to enable.",
            candidate_path
        );
        return;
    }

    let registry = ModelRegistry::new();
    let metrics = ServerMetrics::new_shared();

    let config = ModelConfig {
        id: "real-onnx".to_string(),
        onnx_path: candidate_path.to_string_lossy().to_string(),
        idle_timeout: Duration::from_secs(60),
        max_batch_size: 4,
        max_batch_delay: Duration::from_millis(10),
        provider: None,
        description: None,
        min_residency: Duration::from_millis(0),
        ..Default::default()
    };
    registry.register(ModelController::new(config)).await;

    let app = create_router(AppState::new(registry, metrics));

    let load_req = Request::builder()
        .method("POST")
        .uri("/v1/models/real-onnx/load")
        .body(Body::empty())
        .unwrap();

    let load_resp = app.oneshot(load_req).await.unwrap();
    assert_eq!(load_resp.status(), StatusCode::OK);
}

#[cfg(feature = "candle-backend")]
#[tokio::test]
async fn test_e2e_candle_safetensors_lifecycle_and_inference() {
    use candle_core::{Device, Tensor};
    use std::collections::HashMap;

    // 1. Create a real temporary Safetensors model: [out_features: 1, in_features: 2]
    // Weight = [2.0, 3.0], Bias = [1.0]
    // Y = 2.0 * x0 + 3.0 * x1 + 1.0
    let temp_dir = std::env::temp_dir();
    let unique_id = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let model_path = temp_dir.join(format!("snooze_test_e2e_{unique_id}.safetensors"));

    let dev = Device::Cpu;
    let weight = Tensor::from_slice(&[2.0f32, 3.0], (1, 2), &dev).unwrap();
    let bias = Tensor::from_slice(&[1.0f32], 1, &dev).unwrap();

    let mut tensors = HashMap::new();
    tensors.insert("weight".to_string(), weight);
    tensors.insert("bias".to_string(), bias);
    candle_core::safetensors::save(&tensors, &model_path).unwrap();

    // 2. Register model with registry
    let registry = ModelRegistry::new();
    let metrics = ServerMetrics::new_shared();

    let config = ModelConfig {
        id: "candle-e2e".to_string(),
        onnx_path: model_path.to_string_lossy().to_string(),
        idle_timeout: Duration::from_secs(60),
        max_batch_size: 4,
        max_batch_delay: Duration::from_millis(5),
        provider: None,
        description: None,
        min_residency: Duration::from_millis(0),
        ..Default::default()
    };
    registry.register(ModelController::new(config)).await;

    let app = create_router(AppState::new(registry.clone(), metrics.clone()));

    // 3. Check /v1/models before inference -> should be idle
    let models_resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(models_resp.status(), StatusCode::OK);
    let body = models_resp.into_body().collect().await.unwrap().to_bytes();
    let body_str = std::str::from_utf8(&body).unwrap();
    assert!(body_str.contains("\"id\":\"candle-e2e\""));
    assert!(body_str.contains("\"state\":\"idle\""));

    // 4. Perform inference request via HTTP
    // Input: [[4.0, 1.0]] -> Expected Y = 2*4 + 3*1 + 1 = 12.0
    let infer_req_body = serde_json::json!({
        "inputs": [
            {
                "name": "input",
                "shape": [1, 2],
                "data": [4.0, 1.0]
            }
        ]
    });

    let infer_req = Request::builder()
        .method("POST")
        .uri("/v1/models/candle-e2e/infer")
        .header("Content-Type", "application/json")
        .body(Body::from(serde_json::to_vec(&infer_req_body).unwrap()))
        .unwrap();

    let infer_resp = app.clone().oneshot(infer_req).await.unwrap();
    assert_eq!(infer_resp.status(), StatusCode::OK);

    let infer_bytes = infer_resp.into_body().collect().await.unwrap().to_bytes();
    let infer_json: serde_json::Value = serde_json::from_slice(&infer_bytes).unwrap();

    let outputs = infer_json.get("outputs").unwrap().as_array().unwrap();
    assert_eq!(outputs.len(), 1);
    let out_data = outputs[0].get("data").unwrap().as_array().unwrap();
    let val = out_data[0].as_f64().unwrap();
    assert!((val - 12.0).abs() < 1e-5);

    // 5. Explicitly unload model
    let unload_req = Request::builder()
        .method("POST")
        .uri("/v1/models/candle-e2e/unload")
        .body(Body::empty())
        .unwrap();
    let unload_resp = app.clone().oneshot(unload_req).await.unwrap();
    assert_eq!(unload_resp.status(), StatusCode::OK);

    // Clean up temp file
    let _ = std::fs::remove_file(&model_path);
}

#[cfg(feature = "tree-backend")]
#[tokio::test]
async fn test_e2e_tree_lifecycle_and_inference() {
    use hessboost::data::DMatrix;
    use hessboost::prelude::{train, TrainingParams};

    // 1. Create a real temporary XGBoost JSON model
    let temp_dir = std::env::temp_dir();
    let unique_id = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let model_path = temp_dir.join(format!("snooze_tree_e2e_{unique_id}.xgb.json"));

    let x = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let y = [3.0f32, 7.0, 11.0, 15.0];
    let dtrain = DMatrix::from_dense(&x, 4, 2)
        .unwrap()
        .with_labels(&y)
        .unwrap();

    let params = TrainingParams::builder()
        .objective("reg:squarederror")
        .max_depth(2)
        .eta(0.5)
        .build()
        .unwrap();

    let trained = train(&params, &dtrain, 3).unwrap();
    trained.save_xgboost_json(&model_path).unwrap();

    // 2. Register model with registry
    let registry = ModelRegistry::new();
    let metrics = ServerMetrics::new_shared();

    let config = ModelConfig {
        id: "tree-e2e".to_string(),
        onnx_path: model_path.to_string_lossy().to_string(),
        idle_timeout: Duration::from_secs(60),
        max_batch_size: 4,
        max_batch_delay: Duration::from_millis(5),
        provider: Some("tree".to_string()),
        description: Some("Tree E2E test model".to_string()),
        min_residency: Duration::from_millis(0),
        ..Default::default()
    };
    registry.register(ModelController::new(config)).await;

    let app = create_router(AppState::new(registry.clone(), metrics.clone()));

    // 3. Query /v1/models -> initially idle
    let models_resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(models_resp.status(), StatusCode::OK);
    let body = models_resp.into_body().collect().await.unwrap().to_bytes();
    let body_str = std::str::from_utf8(&body).unwrap();
    assert!(body_str.contains("\"id\":\"tree-e2e\""));
    assert!(body_str.contains("\"state\":\"idle\""));

    // 4. Perform HTTP inference request
    let infer_req_body = serde_json::json!({
        "inputs": [
            {
                "name": "features",
                "shape": [1, 2],
                "data": [3.0, 4.0]
            }
        ]
    });

    let infer_req = Request::builder()
        .method("POST")
        .uri("/v1/models/tree-e2e/infer")
        .header("Content-Type", "application/json")
        .body(Body::from(serde_json::to_vec(&infer_req_body).unwrap()))
        .unwrap();

    let infer_resp = app.clone().oneshot(infer_req).await.unwrap();
    assert_eq!(infer_resp.status(), StatusCode::OK);

    let infer_bytes = infer_resp.into_body().collect().await.unwrap().to_bytes();
    let infer_json: serde_json::Value = serde_json::from_slice(&infer_bytes).unwrap();

    let outputs = infer_json.get("outputs").unwrap().as_array().unwrap();
    assert_eq!(outputs.len(), 1);
    let out_shape = outputs[0].get("shape").unwrap().as_array().unwrap();
    assert_eq!(out_shape, &[serde_json::json!(1), serde_json::json!(1)]);
    let out_data = outputs[0].get("data").unwrap().as_array().unwrap();
    assert_eq!(out_data.len(), 1);

    // 5. Unload model
    let unload_req = Request::builder()
        .method("POST")
        .uri("/v1/models/tree-e2e/unload")
        .body(Body::empty())
        .unwrap();
    let unload_resp = app.clone().oneshot(unload_req).await.unwrap();
    assert_eq!(unload_resp.status(), StatusCode::OK);

    // Clean up
    let _ = std::fs::remove_file(&model_path);
}

#[tokio::test]
async fn test_e2e_reactive_deallocation_on_load() {
    let registry = ModelRegistry::new();
    let metrics = ServerMetrics::new_shared();
    registry.set_metrics(metrics.clone()).await;

    let app = create_router(AppState::new(registry.clone(), metrics.clone()));

    let model_a = ModelController::new(ModelConfig {
        id: "model-a".into(),
        onnx_path: "mock:a".into(),
        min_residency: Duration::from_millis(0),
        ..Default::default()
    });
    let model_b = ModelController::new(ModelConfig {
        id: "model-b".into(),
        onnx_path: "mock:b".into(),
        min_residency: Duration::from_millis(0),
        ..Default::default()
    });
    let model_c = ModelController::new(ModelConfig {
        id: "model-c".into(),
        onnx_path: "mock:oom_once".into(),
        min_residency: Duration::from_millis(0),
        ..Default::default()
    });

    registry.register(model_a.clone()).await;
    registry.register(model_b.clone()).await;
    registry.register(model_c.clone()).await;

    // Load model-a
    let req_a = Request::builder()
        .method("POST")
        .uri("/v1/models/model-a/load")
        .body(Body::empty())
        .unwrap();
    let resp_a = app.clone().oneshot(req_a).await.unwrap();
    assert_eq!(resp_a.status(), StatusCode::OK);
    tokio::time::sleep(Duration::from_millis(15)).await;

    // Load model-b
    let req_b = Request::builder()
        .method("POST")
        .uri("/v1/models/model-b/load")
        .body(Body::empty())
        .unwrap();
    let resp_b = app.clone().oneshot(req_b).await.unwrap();
    assert_eq!(resp_b.status(), StatusCode::OK);

    assert!(model_a.is_loaded().await);
    assert!(model_b.is_loaded().await);

    // Set 1 simulated OOM failure for model-c
    snooze_ml::backend::mock_backend::set_mock_oom_failures(1);

    // Load model-c: will OOM, deallocate LRU model-a, retry, and succeed
    let req_c = Request::builder()
        .method("POST")
        .uri("/v1/models/model-c/load")
        .body(Body::empty())
        .unwrap();
    let resp_c = app.clone().oneshot(req_c).await.unwrap();
    assert_eq!(resp_c.status(), StatusCode::OK);

    assert!(
        !model_a.is_loaded().await,
        "model-a should have been deallocated to resolve OOM"
    );
    assert!(model_b.is_loaded().await, "model-b should remain loaded");
    assert!(model_c.is_loaded().await, "model-c should be loaded");

    // Verify metrics
    let metrics_req = Request::builder()
        .method("GET")
        .uri("/metrics")
        .body(Body::empty())
        .unwrap();
    let metrics_resp = app.clone().oneshot(metrics_req).await.unwrap();
    let body = metrics_resp.into_body().collect().await.unwrap().to_bytes();
    let text = String::from_utf8_lossy(&body);
    assert!(text.contains("snoozeml_model_deallocations_total{reason=\"oom\"} 1"));
    assert!(text.contains("snoozeml_model_load_retries_total 1"));
}

#[tokio::test]
async fn test_e2e_resource_exhausted_returns_503_with_retry_after() {
    let registry = ModelRegistry::new();
    let metrics = ServerMetrics::new_shared();
    let app = create_router(AppState::new(registry.clone(), metrics.clone()));

    let m_oom = ModelController::new(ModelConfig {
        id: "always-oom".into(),
        onnx_path: "mock:oom".into(),
        ..Default::default()
    });
    registry.register(m_oom).await;

    let req = Request::builder()
        .method("POST")
        .uri("/v1/models/always-oom/load")
        .body(Body::empty())
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);

    let retry_after = resp.headers().get("retry-after");
    assert!(retry_after.is_some(), "Expected Retry-After header on 503");
    assert_eq!(retry_after.unwrap(), "5");
}

#[tokio::test]
async fn test_e2e_pin_and_unpin_endpoints() {
    let registry = ModelRegistry::new();
    let metrics = ServerMetrics::new_shared();

    let config = ModelConfig {
        id: "pin-api-target".to_string(),
        onnx_path: "mock:test".to_string(),
        idle_timeout: Duration::from_millis(20),
        max_batch_size: 4,
        max_batch_delay: Duration::from_millis(10),
        min_residency: Duration::from_millis(0),
        pinned: false,
        ..Default::default()
    };
    let controller = ModelController::new(config);
    registry.register(controller.clone()).await;

    let app = create_router(AppState::new(registry.clone(), metrics));

    // Initially Idle and unpinned
    assert_eq!(
        controller.metadata().await.state,
        snooze_ml::types::ModelState::Idle
    );
    assert!(!controller.is_pinned());

    // 1. POST /v1/models/pin-api-target/pin
    let pin_req = Request::builder()
        .method("POST")
        .uri("/v1/models/pin-api-target/pin")
        .body(Body::empty())
        .unwrap();
    let pin_resp = app.clone().oneshot(pin_req).await.unwrap();
    assert_eq!(pin_resp.status(), StatusCode::OK);
    let pin_body = pin_resp.into_body().collect().await.unwrap().to_bytes();
    let pin_json: serde_json::Value = serde_json::from_slice(&pin_body).unwrap();
    assert_eq!(pin_json["model_id"], "pin-api-target");
    assert_eq!(pin_json["status"], "pinned");
    assert_eq!(pin_json["pinned"], true);

    // Model is now Ready and Pinned
    assert_eq!(
        controller.metadata().await.state,
        snooze_ml::types::ModelState::Ready
    );
    assert!(controller.is_pinned());

    // 2. GET /v1/models reflects pinned: true
    let list_req = Request::builder()
        .method("GET")
        .uri("/v1/models")
        .body(Body::empty())
        .unwrap();
    let list_resp = app.clone().oneshot(list_req).await.unwrap();
    let list_body = list_resp.into_body().collect().await.unwrap().to_bytes();
    let models_json: Vec<serde_json::Value> = serde_json::from_slice(&list_body).unwrap();
    assert_eq!(models_json[0]["pinned"], true);

    // 3. POST /v1/models/pin-api-target/unpin
    let unpin_req = Request::builder()
        .method("POST")
        .uri("/v1/models/pin-api-target/unpin")
        .body(Body::empty())
        .unwrap();
    let unpin_resp = app.clone().oneshot(unpin_req).await.unwrap();
    assert_eq!(unpin_resp.status(), StatusCode::OK);
    let unpin_body = unpin_resp.into_body().collect().await.unwrap().to_bytes();
    let unpin_json: serde_json::Value = serde_json::from_slice(&unpin_body).unwrap();
    assert_eq!(unpin_json["model_id"], "pin-api-target");
    assert_eq!(unpin_json["status"], "unpinned");
    assert_eq!(unpin_json["pinned"], false);
    assert!(!controller.is_pinned());

    // 4. Pin/unpin unknown model returns 404
    let pin_missing = Request::builder()
        .method("POST")
        .uri("/v1/models/nonexistent/pin")
        .body(Body::empty())
        .unwrap();
    let resp_404 = app.oneshot(pin_missing).await.unwrap();
    assert_eq!(resp_404.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_e2e_refresh_models_endpoint() {
    let temp_dir = std::env::temp_dir().join(format!(
        "snooze_test_refresh_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    tokio::fs::create_dir_all(&temp_dir).await.unwrap();

    // Create a mock model file in temp_dir
    let model_file = temp_dir.join("discovered_model.onnx");
    tokio::fs::write(&model_file, b"mock_weights")
        .await
        .unwrap();

    let registry = ModelRegistry::new();
    let metrics = ServerMetrics::new_shared();
    let app = create_router(AppState::new(registry.clone(), metrics).with_models_dir(&temp_dir));

    // Initially registry has no models
    assert!(registry.list_metadata().await.is_empty());

    // 1. Call POST /v1/models/refresh with empty body
    let req = Request::builder()
        .method("POST")
        .uri("/v1/models/refresh")
        .body(Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["status"], "refreshed");
    let models = json["models"].as_array().unwrap();
    assert_eq!(models.len(), 1);
    assert_eq!(models[0]["id"], "discovered_model");

    // 2. Call POST /v1/models/refresh with an arbitrary/malicious body
    // Verify that the server ignores any client-supplied models_dir and still scans server's configured dir
    let malicious_req = Request::builder()
        .method("POST")
        .uri("/v1/models/refresh")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"models_dir":"/malicious/directory/traversal"}"#,
        ))
        .unwrap();
    let resp = app.clone().oneshot(malicious_req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["status"], "refreshed");
    let models = json["models"].as_array().unwrap();
    assert_eq!(models.len(), 1);
    assert_eq!(models[0]["id"], "discovered_model");

    // Clean up
    let _ = tokio::fs::remove_dir_all(&temp_dir).await;
}

#[tokio::test]
async fn test_e2e_upload_model_endpoint() {
    let temp_dir = std::env::temp_dir().join(format!(
        "snooze_test_upload_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    tokio::fs::create_dir_all(&temp_dir).await.unwrap();

    let registry = ModelRegistry::new();
    let metrics = ServerMetrics::new_shared();
    let app = create_router(AppState::new(registry.clone(), metrics).with_models_dir(&temp_dir));

    let boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW";
    let mut body = Vec::new();

    let add_field = |body: &mut Vec<u8>, name: &str, val: &str| {
        body.extend_from_slice(
            format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{val}\r\n"
            )
            .as_bytes(),
        );
    };

    add_field(&mut body, "model_id", "uploaded-custom-model");
    add_field(&mut body, "description", "Custom uploaded classifier");
    add_field(&mut body, "pinned", "true");
    add_field(&mut body, "max_batch_size", "32");
    add_field(&mut body, "max_batch_delay_ms", "25");
    add_field(&mut body, "idle_timeout_secs", "120");

    // Add file
    body.extend_from_slice(
        format!("--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"my_weights.onnx\"\r\nContent-Type: application/octet-stream\r\n\r\n")
            .as_bytes(),
    );
    body.extend_from_slice(b"fake_onnx_data");
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());

    let req = Request::builder()
        .method("POST")
        .uri("/v1/models/upload")
        .header(
            "content-type",
            format!("multipart/form-data; boundary={boundary}"),
        )
        .body(Body::from(body))
        .unwrap();

    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let resp_bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let meta_json: serde_json::Value = serde_json::from_slice(&resp_bytes).unwrap();
    assert_eq!(meta_json["id"], "uploaded-custom-model");
    assert_eq!(meta_json["description"], "Custom uploaded classifier");
    assert_eq!(meta_json["pinned"], true);
    assert_eq!(meta_json["max_batch_size"], 32);
    assert_eq!(meta_json["max_batch_delay_ms"], 25);
    assert_eq!(meta_json["idle_timeout_secs"], 120);

    // Verify model was saved in temp_dir
    let saved_file = temp_dir.join("my_weights.onnx");
    assert!(saved_file.exists());
    let saved_content = tokio::fs::read(&saved_file).await.unwrap();
    assert_eq!(saved_content, b"fake_onnx_data");

    // Clean up
    let _ = tokio::fs::remove_dir_all(&temp_dir).await;
}

#[tokio::test]
async fn test_e2e_upload_model_conflict_and_overwrite() {
    let temp_dir = std::env::temp_dir().join(format!(
        "snooze_test_conflict_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    tokio::fs::create_dir_all(&temp_dir).await.unwrap();

    let registry = ModelRegistry::new();
    let metrics = ServerMetrics::new_shared();
    let app = create_router(AppState::new(registry.clone(), metrics).with_models_dir(&temp_dir));

    let boundary = "----WebKitFormBoundaryUploadConflict";
    let make_body = |overwrite: Option<&str>| {
        let mut body = Vec::new();
        let add_field = |body: &mut Vec<u8>, name: &str, val: &str| {
            body.extend_from_slice(
                format!(
                    "--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{val}\r\n"
                )
                .as_bytes(),
            );
        };
        add_field(&mut body, "model_id", "conflict-model");
        if let Some(ow) = overwrite {
            add_field(&mut body, "overwrite", ow);
        }
        body.extend_from_slice(
            format!("--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"conflict_model.onnx\"\r\nContent-Type: application/octet-stream\r\n\r\n")
                .as_bytes(),
        );
        body.extend_from_slice(b"version_1");
        body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
        body
    };

    // 1. Initial upload: must succeed (200 OK)
    let req1 = Request::builder()
        .method("POST")
        .uri("/v1/models/upload")
        .header(
            "content-type",
            format!("multipart/form-data; boundary={boundary}"),
        )
        .body(Body::from(make_body(None)))
        .unwrap();

    let resp1 = app.clone().oneshot(req1).await.unwrap();
    assert_eq!(resp1.status(), StatusCode::OK);

    // 2. Duplicate upload without overwrite=true: must return 409 Conflict
    let req2 = Request::builder()
        .method("POST")
        .uri("/v1/models/upload")
        .header(
            "content-type",
            format!("multipart/form-data; boundary={boundary}"),
        )
        .body(Body::from(make_body(None)))
        .unwrap();

    let resp2 = app.clone().oneshot(req2).await.unwrap();
    assert_eq!(
        resp2.status(),
        StatusCode::CONFLICT,
        "Duplicate upload without overwrite must return 409 Conflict"
    );

    // 3. Upload with overwrite=true: must succeed (200 OK)
    let req3 = Request::builder()
        .method("POST")
        .uri("/v1/models/upload")
        .header(
            "content-type",
            format!("multipart/form-data; boundary={boundary}"),
        )
        .body(Body::from(make_body(Some("true"))))
        .unwrap();

    let resp3 = app.oneshot(req3).await.unwrap();
    assert_eq!(
        resp3.status(),
        StatusCode::OK,
        "Upload with overwrite=true must replace existing file"
    );

    let _ = tokio::fs::remove_dir_all(&temp_dir).await;
}

#[tokio::test]
async fn test_e2e_model_id_validation_rejects_malicious_and_empty_ids() {
    let registry = ModelRegistry::new();
    let metrics = ServerMetrics::new_shared();
    let app = create_router(AppState::new(registry, metrics));

    let dummy_payload = serde_json::json!({
        "inputs": [
            {
                "name": "in",
                "shape": [1, 2],
                "data": [1.0, 2.0]
            }
        ]
    });

    let test_cases = [
        "invalid..id",
        "model@prod",
        "model~test",
        "model+plus",
        "model=equals",
        "model$name",
    ];

    for bad_id in test_cases {
        let req = Request::builder()
            .method("POST")
            .uri(format!("/v1/models/{bad_id}/infer"))
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&dummy_payload).unwrap()))
            .unwrap();

        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::BAD_REQUEST,
            "Invalid model ID '{bad_id}' must return 400 Bad Request"
        );
    }
}

#[tokio::test]
async fn test_e2e_large_payload_greater_than_2mb() {
    let registry = ModelRegistry::new();
    let metrics = ServerMetrics::new_shared();

    let config = ModelConfig {
        id: "mock-large".to_string(),
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

    let app = create_router(AppState::new(registry, metrics));

    // 600,000 floats serialized to JSON is ~3.5MB, exceeding Axum's default 2MB limit
    let float_count = 600_000;
    let data = vec![1.25f32; float_count];

    let payload = serde_json::json!({
        "inputs": [
            {
                "name": "large_input",
                "shape": [1, float_count],
                "data": data
            }
        ]
    });

    let body_bytes = serde_json::to_vec(&payload).unwrap();
    assert!(
        body_bytes.len() > 2 * 1024 * 1024,
        "Payload size must exceed 2MB default body limit"
    );

    let req = Request::builder()
        .method("POST")
        .uri("/v1/models/mock-large/infer")
        .header("content-type", "application/json")
        .body(Body::from(body_bytes))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "Inference routes must support payloads > 2MB via 128MB DefaultBodyLimit"
    );
}

#[test]
fn test_sanitize_paths_in_error_response() {
    use snooze_ml::server::http::sanitize_error_message;

    // Windows paths
    let win_msg = "Error reading model weights from C:\\Users\\Administrator\\secrets\\model.onnx: file not found";
    let sanitized_win = sanitize_error_message(win_msg);
    assert!(
        !sanitized_win.contains("Administrator"),
        "Sanitized message must mask sensitive user directories"
    );
    assert!(sanitized_win.contains("[redacted_path]"));

    // Unix paths
    let unix_msg =
        "Failed to open weights at /home/deployer/prod/model.safetensors: permission denied";
    let sanitized_unix = sanitize_error_message(unix_msg);
    assert!(!sanitized_unix.contains("deployer"));
    assert!(sanitized_unix.contains("[redacted_path]"));

    // Quotes around paths
    let quoted_msg = "Cannot load 'C:/Users/test/model.onnx': invalid header";
    let sanitized_quoted = sanitize_error_message(quoted_msg);
    assert_eq!(
        sanitized_quoted,
        "Cannot load '[redacted_path]': invalid header"
    );
}
