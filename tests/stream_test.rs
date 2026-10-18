//! Behavioral and integration tests for streaming message broker support (auto topic creation,
//! non-blocking concurrency, micro-batching, DLQ error routing, and telemetry).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use snooze_ml::config::{ModelDirectoryConfig, StreamBindingConfig, StreamWireFormat};
use snooze_ml::model::{ModelConfig, ModelController};
use snooze_ml::registry::ModelRegistry;
use snooze_ml::routing::RoutingStrategyType;
use snooze_ml::server::http::{create_router, AppState};
use snooze_ml::server::metrics::ServerMetrics;
use snooze_ml::stream::message::{StreamInferRequest, StreamInferResponse, StreamTensorData};
use snooze_ml::stream::mock_stream::MockStreamDriver;
use snooze_ml::stream::StreamSupervisor;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::watch;
use tower::ServiceExt;

fn create_test_model(id: &str, max_batch_size: usize) -> Arc<ModelController> {
    let config = ModelConfig {
        id: id.to_string(),
        onnx_path: "mock:demo".to_string(),
        idle_timeout: Duration::from_secs(60),
        max_batch_size,
        max_batch_delay: Duration::from_millis(10),
        provider: None,
        description: Some("Mock test model".to_string()),
        min_residency: Duration::from_secs(10),
        autoscale_queue_threshold: max_batch_size,
        max_replicas: None,
        routing_strategy: RoutingStrategyType::LeastQueue,
        pinned: false,
    };
    ModelController::new(config)
}

#[tokio::test]
async fn test_auto_create_topics_and_end_to_end_stream_inference() {
    let registry = ModelRegistry::new();
    let model = create_test_model("demo-model", 4);
    registry.register(model).await;

    let metrics = ServerMetrics::new_shared();
    let driver = Arc::new(MockStreamDriver::new("snooze.in"));

    let stream_cfg = StreamBindingConfig {
        name: "test-stream".to_string(),
        broker: "mock".to_string(),
        endpoint: "memory://".to_string(),
        model_id: Some("demo-model".to_string()),
        input_topic: "snooze.in".to_string(),
        output_topic: Some("snooze.out".to_string()),
        error_topic: Some("snooze.dlq".to_string()),
        consumer_group: None,
        auto_create_topics: true,
        concurrency: 8,
        partitions: 1,
        format: StreamWireFormat::Json,
    };

    let supervisor = Arc::new(StreamSupervisor::new(
        registry.clone(),
        metrics.clone(),
        vec![stream_cfg.clone()],
    ));

    supervisor.register_driver("test-stream", driver.clone());

    let (shutdown_tx, shutdown_rx) = watch::channel(());
    supervisor.spawn_workers(shutdown_rx);

    // Give worker loop a moment to initialize and auto-create topics
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Send 12 stream messages concurrently to test micro-batching
    let msg_count = 12;
    for i in 0..msg_count {
        let req = StreamInferRequest {
            id: Some(format!("req-{i}")),
            model_id: None,
            inputs: vec![StreamTensorData::new(
                "input",
                vec![1, 4],
                vec![1.0, 2.0, 3.0, (i as f32) + 1.0],
            )],
            reply_to: None,
        };
        let bytes = serde_json::to_vec(&req).unwrap();
        driver.inject_message("snooze.in", bytes).await.unwrap();
    }

    // Await message consumption and response publication
    let mut published = Vec::new();
    for _ in 0..50 {
        published = driver.get_published("snooze.out").await;
        if published.len() == msg_count {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    assert_eq!(
        published.len(),
        msg_count,
        "All 12 responses must be published to snooze.out"
    );

    // Verify all 12 received positive acknowledgments
    assert_eq!(driver.ack_count(), msg_count as u64);

    // Verify response content and correlation IDs
    for raw in published {
        let resp: StreamInferResponse = serde_json::from_slice(&raw).unwrap();
        assert_eq!(resp.status, "success");
        assert_eq!(resp.model_id, "demo-model");
        assert!(resp.id.starts_with("req-"));
        assert!(!resp.outputs.is_empty());
        assert_eq!(resp.outputs[0].shape, vec![1, 4]);
    }

    // Verify metrics tracking
    let prom = metrics.format_prometheus();
    assert!(prom.contains("snoozeml_stream_messages_consumed_total 12"));
    assert!(prom.contains("snoozeml_stream_messages_published_total 12"));
    assert!(prom.contains("snoozeml_stream_errors_total 0"));

    // Graceful shutdown
    let _ = shutdown_tx.send(());
}

#[tokio::test]
async fn test_stream_dlq_error_routing_on_invalid_input() {
    let registry = ModelRegistry::new();
    let model = create_test_model("demo-model", 4);
    registry.register(model).await;

    let metrics = ServerMetrics::new_shared();
    let driver = Arc::new(MockStreamDriver::new("snooze.err.in"));

    let stream_cfg = StreamBindingConfig {
        name: "err-stream".to_string(),
        broker: "mock".to_string(),
        endpoint: "memory://".to_string(),
        model_id: Some("demo-model".to_string()),
        input_topic: "snooze.err.in".to_string(),
        output_topic: Some("snooze.err.out".to_string()),
        error_topic: Some("snooze.err.dlq".to_string()),
        consumer_group: None,
        auto_create_topics: true,
        concurrency: 4,
        partitions: 1,
        format: StreamWireFormat::Json,
    };

    let supervisor = Arc::new(StreamSupervisor::new(
        registry.clone(),
        metrics.clone(),
        vec![stream_cfg],
    ));

    supervisor.register_driver("err-stream", driver.clone());

    let (shutdown_tx, shutdown_rx) = watch::channel(());
    supervisor.spawn_workers(shutdown_rx);

    tokio::time::sleep(Duration::from_millis(50)).await;

    // 1. Send malformed JSON
    driver
        .inject_message("snooze.err.in", b"{ malformed json".to_vec())
        .await
        .unwrap();

    // 2. Send invalid tensor dimensions (element count mismatch: shape [1, 4] with 2 elements)
    let invalid_req = StreamInferRequest {
        id: Some("req-mismatch".to_string()),
        model_id: None,
        inputs: vec![StreamTensorData::new(
            "input",
            vec![1, 4],
            vec![1.0, 2.0], // missing 2 elements!
        )],
        reply_to: None,
    };
    driver
        .inject_message("snooze.err.in", serde_json::to_vec(&invalid_req).unwrap())
        .await
        .unwrap();

    // Await DLQ routing
    let mut dlq_messages = Vec::new();
    for _ in 0..50 {
        dlq_messages = driver.get_published("snooze.err.dlq").await;
        if dlq_messages.len() == 2 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    assert_eq!(
        dlq_messages.len(),
        2,
        "Both invalid requests must route to DLQ"
    );

    // Both messages must be positively acknowledged so poison pills do not cycle
    assert_eq!(driver.ack_count(), 2);

    let err_resp: StreamInferResponse = serde_json::from_slice(&dlq_messages[1]).unwrap();
    assert_eq!(err_resp.status, "error");
    assert_eq!(err_resp.id, "req-mismatch");
    assert!(err_resp
        .error
        .unwrap()
        .contains("Shape [1, 4] expects 4 elements"));

    // Verify error metrics
    let prom = metrics.format_prometheus();
    assert!(prom.contains("snoozeml_stream_errors_total 2"));

    let _ = shutdown_tx.send(());
}

#[tokio::test]
async fn test_non_blocking_stream_with_concurrent_http() {
    let registry = ModelRegistry::new();
    let model = create_test_model("demo-model", 16);
    registry.register(model).await;

    let metrics = ServerMetrics::new_shared();
    let driver = Arc::new(MockStreamDriver::new("snooze.load.in"));

    let stream_cfg = StreamBindingConfig {
        name: "load-stream".to_string(),
        broker: "mock".to_string(),
        endpoint: "memory://".to_string(),
        model_id: Some("demo-model".to_string()),
        input_topic: "snooze.load.in".to_string(),
        output_topic: Some("snooze.load.out".to_string()),
        error_topic: None,
        consumer_group: None,
        auto_create_topics: true,
        concurrency: 16,
        partitions: 1,
        format: StreamWireFormat::Json,
    };

    let supervisor = Arc::new(StreamSupervisor::new(
        registry.clone(),
        metrics.clone(),
        vec![stream_cfg],
    ));

    supervisor.register_driver("load-stream", driver.clone());

    let (shutdown_tx, shutdown_rx) = watch::channel(());
    supervisor.spawn_workers(shutdown_rx);

    // Setup Axum router to test concurrent HTTP non-blocking behavior
    let app_state = AppState::new(registry, metrics);
    let app = create_router(app_state);

    tokio::time::sleep(Duration::from_millis(50)).await;

    // Inject 50 stream messages in the background
    let driver_clone = driver.clone();
    tokio::spawn(async move {
        for i in 0..50 {
            let req = StreamInferRequest {
                id: Some(format!("stream-{i}")),
                model_id: None,
                inputs: vec![StreamTensorData::new(
                    "input",
                    vec![1, 4],
                    vec![1.0, 1.0, 1.0, 1.0],
                )],
                reply_to: None,
            };
            let _ = driver_clone
                .inject_message("snooze.load.in", serde_json::to_vec(&req).unwrap())
                .await;
        }
    });

    // Concurrently send HTTP request: must be non-blocking and return 200 immediately!
    let http_req = Request::builder()
        .method("POST")
        .uri("/v1/models/demo-model/infer")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"inputs":[{"name":"input","shape":[1,4],"data":[0.5,0.5,0.5,0.5]}]}"#,
        ))
        .unwrap();

    let resp = app.oneshot(http_req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let val: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(val["model_id"], "demo-model");
    assert!(val["outputs"].is_array());

    // Clean up
    let _ = shutdown_tx.send(());
}

#[test]
fn test_parse_streams_from_toml_configuration() {
    let toml = r#"
[defaults]
idle_timeout_secs = 120

[[models]]
id = "sentiment"
path = "model.onnx"

[[streams]]
name = "kafka-sentiment"
broker = "kafka"
endpoint = "localhost:9092"
model_id = "sentiment"
input_topic = "reviews.in"
output_topic = "reviews.out"
error_topic = "reviews.dlq"
auto_create_topics = true
concurrency = 32
format = "protobuf"

[[streams]]
name = "redis-cache"
broker = "redis"
endpoint = "redis://127.0.0.1:6379"
# Omitting model_id allows this stream to dynamically serve all models
input_topic = "embed:stream"
output_topic = "embed:out"
"#;

    let parsed = ModelDirectoryConfig::parse_toml_fault_tolerant(toml).unwrap();
    assert_eq!(parsed.models.len(), 1);
    assert_eq!(parsed.streams.len(), 2);

    let s1 = &parsed.streams[0];
    assert_eq!(s1.name, "kafka-sentiment");
    assert_eq!(s1.broker, "kafka");
    assert_eq!(s1.endpoint, "localhost:9092");
    assert_eq!(s1.model_id.as_deref(), Some("sentiment"));
    assert_eq!(s1.input_topic, "reviews.in");
    assert_eq!(s1.output_topic.as_deref(), Some("reviews.out"));
    assert_eq!(s1.error_topic.as_deref(), Some("reviews.dlq"));
    assert!(s1.auto_create_topics);
    assert_eq!(s1.concurrency, 32);
    assert_eq!(s1.format, StreamWireFormat::Protobuf);

    let s2 = &parsed.streams[1];
    assert_eq!(s2.name, "redis-cache");
    assert_eq!(s2.broker, "redis");
    assert_eq!(s2.endpoint, "redis://127.0.0.1:6379");
    assert_eq!(s2.model_id, None);
    assert_eq!(s2.input_topic, "embed:stream");
    assert_eq!(s2.output_topic.as_deref(), Some("embed:out"));
    assert_eq!(s2.format, StreamWireFormat::Json);
    assert!(s2.auto_create_topics); // Default true
    assert_eq!(s2.concurrency, 16); // Default 16
}

#[tokio::test]
async fn test_protobuf_wire_format_stream_roundtrip() {
    let req = StreamInferRequest {
        id: Some("proto-123".to_string()),
        model_id: Some("bert-test".to_string()),
        inputs: vec![StreamTensorData::new(
            "input_ids",
            vec![1, 3],
            vec![101.0, 2054.0, 102.0],
        )],
        reply_to: Some("replies.topic".to_string()),
    };

    // Encode to protobuf
    use prost::Message;
    let proto_inputs = req
        .inputs
        .iter()
        .map(|inp| {
            let shape_i64: Vec<i64> = inp.shape.iter().map(|&d| d as i64).collect();
            let raw_data = bytemuck::cast_slice(&inp.data).to_vec();
            snooze_ml::server::grpc::proto::TensorData {
                name: inp.name.clone(),
                shape: shape_i64,
                data_type: snooze_ml::server::grpc::proto::DataType::Fp32 as i32,
                raw_data,
            }
        })
        .collect();

    let mut params = std::collections::HashMap::new();
    params.insert("id".to_string(), "proto-123".to_string());
    params.insert("reply_to".to_string(), "replies.topic".to_string());

    let proto_req = snooze_ml::server::grpc::proto::ModelInferRequest {
        model_id: "bert-test".to_string(),
        inputs: proto_inputs,
        client_timeout_ms: None,
        parameters: params,
    };

    let mut buf = Vec::new();
    proto_req.encode(&mut buf).unwrap();

    // Decode from protobuf bytes
    let decoded = StreamInferRequest::from_bytes(&buf, StreamWireFormat::Protobuf).unwrap();
    assert_eq!(decoded.id.as_deref(), Some("proto-123"));
    assert_eq!(decoded.model_id.as_deref(), Some("bert-test"));
    assert_eq!(decoded.reply_to.as_deref(), Some("replies.topic"));
    assert_eq!(decoded.inputs.len(), 1);
    assert_eq!(decoded.inputs[0].name, "input_ids");
    assert_eq!(decoded.inputs[0].shape, vec![1, 3]);
    assert_eq!(decoded.inputs[0].data, vec![101.0, 2054.0, 102.0]);
}

#[tokio::test]
async fn test_decoupled_stream_serves_multiple_models_dynamically() {
    let registry = ModelRegistry::new();
    let model_alpha = create_test_model("model-alpha", 4);
    let model_beta = create_test_model("model-beta", 4);
    registry.register(model_alpha).await;
    registry.register(model_beta).await;

    let metrics = ServerMetrics::new_shared();
    let driver = Arc::new(MockStreamDriver::new("snooze.multi.in"));

    // Stream binding has NO model_id specified (model_id: None) - serving ALL models!
    let stream_cfg = StreamBindingConfig {
        name: "multi-model-stream".to_string(),
        broker: "mock".to_string(),
        endpoint: "memory://".to_string(),
        model_id: None,
        input_topic: "snooze.multi.in".to_string(),
        output_topic: Some("snooze.multi.out".to_string()),
        error_topic: Some("snooze.multi.dlq".to_string()),
        consumer_group: None,
        auto_create_topics: true,
        concurrency: 8,
        partitions: 1,
        format: StreamWireFormat::Json,
    };

    let supervisor = Arc::new(StreamSupervisor::new(
        registry.clone(),
        metrics.clone(),
        vec![stream_cfg],
    ));

    supervisor.register_driver("multi-model-stream", driver.clone());

    let (shutdown_tx, shutdown_rx) = watch::channel(());
    supervisor.spawn_workers(shutdown_rx);

    tokio::time::sleep(Duration::from_millis(50)).await;

    // 1. Send request for "model-alpha"
    let req_alpha = StreamInferRequest {
        id: Some("req-alpha-1".to_string()),
        model_id: Some("model-alpha".to_string()),
        inputs: vec![StreamTensorData::new(
            "input",
            vec![1, 4],
            vec![1.0, 2.0, 3.0, 4.0],
        )],
        reply_to: None,
    };
    driver
        .inject_message("snooze.multi.in", serde_json::to_vec(&req_alpha).unwrap())
        .await
        .unwrap();

    // 2. Send request for "model-beta" on the exact same stream, specifying custom reply_to
    let req_beta = StreamInferRequest {
        id: Some("req-beta-1".to_string()),
        model_id: Some("model-beta".to_string()),
        inputs: vec![StreamTensorData::new(
            "input",
            vec![1, 4],
            vec![5.0, 6.0, 7.0, 8.0],
        )],
        reply_to: Some("client.beta.responses".to_string()),
    };
    driver
        .inject_message("snooze.multi.in", serde_json::to_vec(&req_beta).unwrap())
        .await
        .unwrap();

    // 3. Send request with missing model_id -> should route to DLQ
    let req_missing_model = StreamInferRequest {
        id: Some("req-missing-id".to_string()),
        model_id: None,
        inputs: vec![StreamTensorData::new(
            "input",
            vec![1, 4],
            vec![1.0, 1.0, 1.0, 1.0],
        )],
        reply_to: None,
    };
    driver
        .inject_message(
            "snooze.multi.in",
            serde_json::to_vec(&req_missing_model).unwrap(),
        )
        .await
        .unwrap();

    // Await responses
    let mut default_out = Vec::new();
    let mut beta_out = Vec::new();
    let mut dlq_out = Vec::new();

    for _ in 0..50 {
        default_out = driver.get_published("snooze.multi.out").await;
        beta_out = driver.get_published("client.beta.responses").await;
        dlq_out = driver.get_published("snooze.multi.dlq").await;
        if default_out.len() == 1 && beta_out.len() == 1 && dlq_out.len() == 1 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    assert_eq!(
        default_out.len(),
        1,
        "Alpha response routed to default output"
    );
    let resp_alpha: StreamInferResponse = serde_json::from_slice(&default_out[0]).unwrap();
    assert_eq!(resp_alpha.id, "req-alpha-1");
    assert_eq!(resp_alpha.model_id, "model-alpha");
    assert_eq!(resp_alpha.status, "success");

    assert_eq!(
        beta_out.len(),
        1,
        "Beta response routed to dynamic reply_to"
    );
    let resp_beta: StreamInferResponse = serde_json::from_slice(&beta_out[0]).unwrap();
    assert_eq!(resp_beta.id, "req-beta-1");
    assert_eq!(resp_beta.model_id, "model-beta");
    assert_eq!(resp_beta.status, "success");

    assert_eq!(dlq_out.len(), 1, "Missing model_id routed to DLQ");
    let resp_err: StreamInferResponse = serde_json::from_slice(&dlq_out[0]).unwrap();
    assert_eq!(resp_err.id, "req-missing-id");
    assert_eq!(resp_err.status, "error");
    assert!(resp_err.error.unwrap().contains("No model_id specified"));

    let _ = shutdown_tx.send(());
}
