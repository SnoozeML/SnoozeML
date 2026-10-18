//! Integration tests for SnoozeML gRPC endpoints (Health Checking & Inference).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use clap::Parser;
use snooze_ml::auth::{ApiKeyConfig, AuthConfig, Role};
use snooze_ml::model::{ModelConfig, ModelController};
use snooze_ml::registry::ModelRegistry;
use snooze_ml::server::grpc::proto::inference_service_client::InferenceServiceClient;
use snooze_ml::server::grpc::proto::{DataType, ModelInferRequest, TensorData};
use snooze_ml::server::{run_server, ServerConfig};
use std::time::Duration;
use tonic::Code;
use tonic_health::pb::health_client::HealthClient;
use tonic_health::pb::HealthCheckRequest;

static GRPC_ENV_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// Helper to allocate an ephemeral free port from the OS.
async fn get_free_port() -> u16 {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    listener.local_addr().unwrap().port()
}

/// Helper to wait for a TCP port to become available.
async fn wait_for_port(port: u16) {
    let addr = format!("127.0.0.1:{port}");
    for _ in 0..50 {
        if tokio::net::TcpStream::connect(&addr).await.is_ok() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("Port {port} failed to become ready within timeout");
}

#[tokio::test]
async fn test_grpc_health_checking() {
    let http_port = get_free_port().await;
    let grpc_port = get_free_port().await;

    let registry = ModelRegistry::new();
    let config = ServerConfig {
        host: "127.0.0.1".to_string(),
        port: http_port,
        grpc_port,
        grpc_enabled: true,
        sweeper_interval: Duration::from_secs(60),
        auth: None,
        tls: None,
        models_dir: std::path::PathBuf::from("./models"),
        streams: Vec::new(),
    };

    tokio::spawn(async move {
        let _ = run_server(config, registry).await;
    });

    wait_for_port(grpc_port).await;

    let channel = tonic::transport::Channel::from_shared(format!("http://127.0.0.1:{grpc_port}"))
        .unwrap()
        .connect()
        .await
        .unwrap();

    let mut health_client = HealthClient::new(channel);

    // 1. Check default server health
    let resp = health_client
        .check(HealthCheckRequest {
            service: "".to_string(),
        })
        .await
        .unwrap();
    assert_eq!(resp.into_inner().status, 1); // SERVING = 1

    // 2. Check InferenceService health
    let resp = health_client
        .check(HealthCheckRequest {
            service: "snoozeml.v1.InferenceService".to_string(),
        })
        .await
        .unwrap();
    assert_eq!(resp.into_inner().status, 1); // SERVING = 1
}

#[tokio::test]
async fn test_grpc_inference_success_and_metrics() {
    let http_port = get_free_port().await;
    let grpc_port = get_free_port().await;

    let registry = ModelRegistry::new();
    let model_cfg = ModelConfig {
        id: "mock-grpc-model".to_string(),
        onnx_path: "mock:test".to_string(),
        idle_timeout: Duration::from_secs(60),
        max_batch_size: 4,
        max_batch_delay: Duration::from_millis(10),
        provider: None,
        description: Some("Mock model for gRPC testing".to_string()),
        min_residency: Duration::from_millis(0),
        ..Default::default()
    };
    registry.register(ModelController::new(model_cfg)).await;

    let config = ServerConfig {
        host: "127.0.0.1".to_string(),
        port: http_port,
        grpc_port,
        grpc_enabled: true,
        sweeper_interval: Duration::from_secs(60),
        auth: None,
        tls: None,
        models_dir: std::path::PathBuf::from("./models"),
        streams: Vec::new(),
    };

    tokio::spawn(async move {
        let _ = run_server(config, registry).await;
    });

    wait_for_port(grpc_port).await;
    wait_for_port(http_port).await;

    let channel = tonic::transport::Channel::from_shared(format!("http://127.0.0.1:{grpc_port}"))
        .unwrap()
        .connect()
        .await
        .unwrap();

    let mut client = InferenceServiceClient::new(channel);

    // Prepare input tensor: [1, 2] with floats [10.0, 20.0]
    let input_floats = vec![10.0f32, 20.0f32];
    let raw_bytes = bytemuck::cast_slice::<f32, u8>(&input_floats).to_vec();

    let req = ModelInferRequest {
        model_id: "mock-grpc-model".to_string(),
        inputs: vec![TensorData {
            name: "input_ids".to_string(),
            shape: vec![1, 2],
            data_type: DataType::Fp32 as i32,
            raw_data: raw_bytes,
        }],
        client_timeout_ms: None,
        parameters: Default::default(),
    };

    let resp = client.infer(req).await.unwrap().into_inner();
    assert_eq!(resp.model_id, "mock-grpc-model");
    assert_eq!(resp.outputs.len(), 1);

    let output = &resp.outputs[0];
    assert_eq!(output.name, "output");
    assert_eq!(output.shape, vec![1, 4]);

    // Check decoded floats from output buffer
    let output_floats: &[f32] = bytemuck::cast_slice(&output.raw_data);
    assert_eq!(output_floats.len(), 4);
    assert!(resp.duration_ms >= 0.0);

    // Check that HTTP Prometheus endpoint recorded the inference request
    let metrics_url = format!("http://127.0.0.1:{http_port}/metrics");
    let metrics_body = tokio::task::spawn_blocking(move || {
        let mut metrics_resp = ureq::get(&metrics_url).call().unwrap();
        metrics_resp.body_mut().read_to_string().unwrap()
    })
    .await
    .unwrap();
    assert!(metrics_body.contains("snoozeml_requests_total{status=\"success\"} 1"));
}

#[tokio::test]
async fn test_grpc_inference_model_not_found() {
    let http_port = get_free_port().await;
    let grpc_port = get_free_port().await;

    let registry = ModelRegistry::new();
    let config = ServerConfig {
        host: "127.0.0.1".to_string(),
        port: http_port,
        grpc_port,
        grpc_enabled: true,
        sweeper_interval: Duration::from_secs(60),
        auth: None,
        tls: None,
        models_dir: std::path::PathBuf::from("./models"),
        streams: Vec::new(),
    };

    tokio::spawn(async move {
        let _ = run_server(config, registry).await;
    });

    wait_for_port(grpc_port).await;

    let channel = tonic::transport::Channel::from_shared(format!("http://127.0.0.1:{grpc_port}"))
        .unwrap()
        .connect()
        .await
        .unwrap();

    let mut client = InferenceServiceClient::new(channel);

    let req = ModelInferRequest {
        model_id: "non-existent-model".to_string(),
        inputs: vec![],
        client_timeout_ms: None,
        parameters: Default::default(),
    };

    let err = client.infer(req).await.unwrap_err();
    assert_eq!(err.code(), Code::NotFound);
    assert!(err.message().contains("non-existent-model"));
}

#[tokio::test]
async fn test_grpc_inference_invalid_inputs() {
    let http_port = get_free_port().await;
    let grpc_port = get_free_port().await;

    let registry = ModelRegistry::new();
    let model_cfg = ModelConfig {
        id: "mock-validate-model".to_string(),
        onnx_path: "mock:test".to_string(),
        idle_timeout: Duration::from_secs(60),
        max_batch_size: 4,
        max_batch_delay: Duration::from_millis(10),
        ..Default::default()
    };
    registry.register(ModelController::new(model_cfg)).await;

    let config = ServerConfig {
        host: "127.0.0.1".to_string(),
        port: http_port,
        grpc_port,
        grpc_enabled: true,
        sweeper_interval: Duration::from_secs(60),
        auth: None,
        tls: None,
        models_dir: std::path::PathBuf::from("./models"),
        streams: Vec::new(),
    };

    tokio::spawn(async move {
        let _ = run_server(config, registry).await;
    });

    wait_for_port(grpc_port).await;

    let channel = tonic::transport::Channel::from_shared(format!("http://127.0.0.1:{grpc_port}"))
        .unwrap()
        .connect()
        .await
        .unwrap();

    let mut client = InferenceServiceClient::new(channel);

    // 1. Truncated buffer (expects 8 bytes for 2 floats, send only 5 bytes)
    let bad_buf_req = ModelInferRequest {
        model_id: "mock-validate-model".to_string(),
        inputs: vec![TensorData {
            name: "input_ids".to_string(),
            shape: vec![1, 2],
            data_type: DataType::Fp32 as i32,
            raw_data: vec![0, 1, 2, 3, 4],
        }],
        client_timeout_ms: None,
        parameters: Default::default(),
    };
    let err = client.infer(bad_buf_req).await.unwrap_err();
    assert_eq!(err.code(), Code::InvalidArgument);

    // 2. Negative dimension in shape
    let neg_dim_req = ModelInferRequest {
        model_id: "mock-validate-model".to_string(),
        inputs: vec![TensorData {
            name: "input_ids".to_string(),
            shape: vec![1, -2],
            data_type: DataType::Fp32 as i32,
            raw_data: vec![0; 8],
        }],
        client_timeout_ms: None,
        parameters: Default::default(),
    };
    let err = client.infer(neg_dim_req).await.unwrap_err();
    assert_eq!(err.code(), Code::InvalidArgument);

    // 3. Empty model ID
    let empty_id_req = ModelInferRequest {
        model_id: "   ".to_string(),
        inputs: vec![],
        client_timeout_ms: None,
        parameters: Default::default(),
    };
    let err = client.infer(empty_id_req).await.unwrap_err();
    assert_eq!(err.code(), Code::InvalidArgument);
}

#[tokio::test]
async fn test_grpc_authentication_enforcement() {
    let http_port = get_free_port().await;
    let grpc_port = get_free_port().await;

    let registry = ModelRegistry::new();
    let model_cfg = ModelConfig {
        id: "mock-auth-model".to_string(),
        onnx_path: "mock:test".to_string(),
        idle_timeout: Duration::from_secs(60),
        max_batch_size: 4,
        max_batch_delay: Duration::from_millis(10),
        ..Default::default()
    };
    registry.register(ModelController::new(model_cfg)).await;

    let auth_config = AuthConfig {
        enabled: true,
        protect_metrics: false,
        api_keys: vec![
            ApiKeyConfig {
                key: "sk-infer-key-123".to_string(),
                role: Role::Inference,
                name: Some("infer-user".to_string()),
            },
            ApiKeyConfig {
                key: "sk-read-key-456".to_string(),
                role: Role::ReadOnly,
                name: Some("read-user".to_string()),
            },
        ],
        jwt: None,
    };

    let config = ServerConfig {
        host: "127.0.0.1".to_string(),
        port: http_port,
        grpc_port,
        grpc_enabled: true,
        sweeper_interval: Duration::from_secs(60),
        auth: Some(auth_config),
        tls: None,
        models_dir: std::path::PathBuf::from("./models"),
        streams: Vec::new(),
    };

    tokio::spawn(async move {
        let _ = run_server(config, registry).await;
    });

    wait_for_port(grpc_port).await;

    let channel = tonic::transport::Channel::from_shared(format!("http://127.0.0.1:{grpc_port}"))
        .unwrap()
        .connect()
        .await
        .unwrap();

    let mut health_client = HealthClient::new(channel.clone());

    // 1. Health check is unauthenticated and still succeeds
    let health_resp = health_client
        .check(HealthCheckRequest {
            service: "".to_string(),
        })
        .await
        .unwrap();
    assert_eq!(health_resp.into_inner().status, 1);

    let mut client = InferenceServiceClient::new(channel);

    let input_floats = vec![1.0f32, 2.0f32];
    let make_req = || ModelInferRequest {
        model_id: "mock-auth-model".to_string(),
        inputs: vec![TensorData {
            name: "input_ids".to_string(),
            shape: vec![1, 2],
            data_type: DataType::Fp32 as i32,
            raw_data: bytemuck::cast_slice::<f32, u8>(&input_floats).to_vec(),
        }],
        client_timeout_ms: None,
        parameters: Default::default(),
    };

    // 2. Unauthenticated request -> fails
    let err = client.infer(make_req()).await.unwrap_err();
    assert_eq!(err.code(), Code::Unauthenticated);

    // 3. Invalid API key -> fails
    let mut bad_key_req = tonic::Request::new(make_req());
    bad_key_req.metadata_mut().insert(
        "x-api-key",
        tonic::metadata::MetadataValue::from_static("invalid-key"),
    );
    let err = client.infer(bad_key_req).await.unwrap_err();
    assert_eq!(err.code(), Code::Unauthenticated);

    // 4. Insufficient role (ReadOnly key) -> fails with PermissionDenied
    let mut read_key_req = tonic::Request::new(make_req());
    read_key_req.metadata_mut().insert(
        "x-api-key",
        tonic::metadata::MetadataValue::from_static("sk-read-key-456"),
    );
    let err = client.infer(read_key_req).await.unwrap_err();
    assert_eq!(err.code(), Code::PermissionDenied);

    // 5. Valid Inference key via x-api-key header -> succeeds
    let mut valid_key_req = tonic::Request::new(make_req());
    valid_key_req.metadata_mut().insert(
        "x-api-key",
        tonic::metadata::MetadataValue::from_static("sk-infer-key-123"),
    );
    let resp = client.infer(valid_key_req).await.unwrap().into_inner();
    assert_eq!(resp.model_id, "mock-auth-model");

    // 6. Valid Inference key via Authorization: Bearer header -> succeeds
    let mut bearer_req = tonic::Request::new(make_req());
    bearer_req.metadata_mut().insert(
        "authorization",
        tonic::metadata::MetadataValue::from_static("Bearer sk-infer-key-123"),
    );
    let resp = client.infer(bearer_req).await.unwrap().into_inner();
    assert_eq!(resp.model_id, "mock-auth-model");
}

#[tokio::test]
async fn test_grpc_disabled_flag() {
    let http_port = get_free_port().await;
    let grpc_port = get_free_port().await;

    let registry = ModelRegistry::new();
    let config = ServerConfig {
        host: "127.0.0.1".to_string(),
        port: http_port,
        grpc_port,
        grpc_enabled: false,
        sweeper_interval: Duration::from_secs(60),
        auth: None,
        tls: None,
        models_dir: std::path::PathBuf::from("./models"),
        streams: Vec::new(),
    };

    tokio::spawn(async move {
        let _ = run_server(config, registry).await;
    });

    wait_for_port(http_port).await;

    // HTTP /health/live works
    let health_url = format!("http://127.0.0.1:{http_port}/health/live");
    let resp = tokio::task::spawn_blocking(move || {
        let mut r = ureq::get(&health_url).call().unwrap();
        r.body_mut().read_to_string().unwrap()
    })
    .await
    .unwrap();
    assert!(resp.contains("\"live\""));

    // gRPC port should NOT be open
    let grpc_addr = format!("127.0.0.1:{grpc_port}");
    let connect_res = tokio::net::TcpStream::connect(&grpc_addr).await;
    assert!(connect_res.is_err());
}

#[tokio::test]
async fn test_grpc_env_var_lifecycle() {
    let _guard = GRPC_ENV_LOCK.lock().await;
    let http_port = get_free_port().await;
    let grpc_port = get_free_port().await;

    std::env::set_var("GRPC_PORT", grpc_port.to_string());
    std::env::set_var("GRPC", "true");

    let cli = snooze_ml::cli::Cli::try_parse_from([
        "snooze",
        "serve",
        "--port",
        &http_port.to_string(),
        "--demo",
    ])
    .unwrap();

    tokio::spawn(async move {
        let _ = snooze_ml::cli::run_cli(cli).await;
    });

    wait_for_port(grpc_port).await;
    wait_for_port(http_port).await;

    let channel = tonic::transport::Channel::from_shared(format!("http://127.0.0.1:{grpc_port}"))
        .unwrap()
        .connect()
        .await
        .unwrap();

    let mut health_client = HealthClient::new(channel);
    let resp = health_client
        .check(HealthCheckRequest {
            service: "".to_string(),
        })
        .await
        .unwrap();
    assert_eq!(resp.into_inner().status, 1);

    std::env::remove_var("GRPC_PORT");
    std::env::remove_var("GRPC");
}

#[tokio::test]
async fn test_grpc_env_var_disabled() {
    let _guard = GRPC_ENV_LOCK.lock().await;
    let http_port = get_free_port().await;
    let grpc_port = get_free_port().await;

    std::env::set_var("GRPC_PORT", grpc_port.to_string());
    std::env::set_var("GRPC", "false");

    let cli = snooze_ml::cli::Cli::try_parse_from([
        "snooze",
        "serve",
        "--port",
        &http_port.to_string(),
        "--demo",
    ])
    .unwrap();

    tokio::spawn(async move {
        let _ = snooze_ml::cli::run_cli(cli).await;
    });

    wait_for_port(http_port).await;

    // gRPC port should NOT be open because GRPC=false was set in env
    let grpc_addr = format!("127.0.0.1:{grpc_port}");
    let connect_res = tokio::net::TcpStream::connect(&grpc_addr).await;
    assert!(connect_res.is_err());

    std::env::remove_var("GRPC_PORT");
    std::env::remove_var("GRPC");
}

#[test]
fn test_inference_error_to_status_parity_mappings() {
    use snooze_ml::server::grpc::convert::inference_error_to_status;
    use snooze_ml::types::InferenceError;

    // NotFound -> NotFound
    let s = inference_error_to_status(InferenceError::NotFound("m".into()));
    assert_eq!(s.code(), Code::NotFound);

    // InvalidInput -> InvalidArgument
    let s = inference_error_to_status(InferenceError::InvalidInput("bad".into()));
    assert_eq!(s.code(), Code::InvalidArgument);

    // ShapeMismatch -> InvalidArgument
    let s = inference_error_to_status(InferenceError::ShapeMismatch("mismatch".into()));
    assert_eq!(s.code(), Code::InvalidArgument);

    // Timeout -> DeadlineExceeded
    let s = inference_error_to_status(InferenceError::Timeout);
    assert_eq!(s.code(), Code::DeadlineExceeded);

    // Loading -> Unavailable
    let s = inference_error_to_status(InferenceError::Loading);
    assert_eq!(s.code(), Code::Unavailable);

    // ResourceExhausted -> ResourceExhausted + retry-after: 5 metadata
    let s = inference_error_to_status(InferenceError::ResourceExhausted("limit".into()));
    assert_eq!(s.code(), Code::ResourceExhausted);
    assert_eq!(
        s.metadata().get("retry-after").unwrap().to_str().unwrap(),
        "5"
    );

    // OutOfMemory -> ResourceExhausted
    let s = inference_error_to_status(InferenceError::OutOfMemory("oom".into()));
    assert_eq!(s.code(), Code::ResourceExhausted);

    // UnsupportedSafetensorsArchitecture -> Unimplemented
    let s = inference_error_to_status(InferenceError::UnsupportedSafetensorsArchitecture(
        "llama".into(),
    ));
    assert_eq!(s.code(), Code::Unimplemented);

    // Runtime -> Internal (with internal path sanitization)
    let s = inference_error_to_status(InferenceError::Runtime(
        "Failed to load C:\\Users\\Administrator\\model.onnx: corrupt".into(),
    ));
    assert_eq!(s.code(), Code::Internal);
    assert!(!s.message().contains("Administrator"));
    assert!(s.message().contains("[redacted_path]"));

    // QueueError -> Internal
    let s = inference_error_to_status(InferenceError::QueueError("queue dropped".into()));
    assert_eq!(s.code(), Code::Internal);
}

#[tokio::test]
async fn test_grpc_client_timeout_ms() {
    let http_port = get_free_port().await;
    let grpc_port = get_free_port().await;

    let registry = ModelRegistry::new();
    // Configure with high batch delay (500ms) so timeout triggers first
    let model_cfg = ModelConfig {
        id: "timeout-model".to_string(),
        onnx_path: "mock:test".to_string(),
        idle_timeout: Duration::from_secs(60),
        max_batch_size: 10,
        max_batch_delay: Duration::from_millis(500),
        ..Default::default()
    };
    registry.register(ModelController::new(model_cfg)).await;

    let config = ServerConfig {
        host: "127.0.0.1".to_string(),
        port: http_port,
        grpc_port,
        grpc_enabled: true,
        sweeper_interval: Duration::from_secs(60),
        auth: None,
        tls: None,
        models_dir: std::path::PathBuf::from("./models"),
        streams: Vec::new(),
    };

    tokio::spawn(async move {
        let _ = run_server(config, registry).await;
    });

    wait_for_port(grpc_port).await;

    let channel = tonic::transport::Channel::from_shared(format!("http://127.0.0.1:{grpc_port}"))
        .unwrap()
        .connect()
        .await
        .unwrap();

    let mut client = InferenceServiceClient::new(channel);

    let input_floats = vec![1.0f32, 2.0f32];
    let raw_bytes = bytemuck::cast_slice::<f32, u8>(&input_floats).to_vec();

    // Pass client_timeout_ms = 25ms, which expires before 500ms batch delay
    let req = ModelInferRequest {
        model_id: "timeout-model".to_string(),
        inputs: vec![TensorData {
            name: "input_ids".to_string(),
            shape: vec![1, 2],
            data_type: DataType::Fp32 as i32,
            raw_data: raw_bytes,
        }],
        client_timeout_ms: Some(25),
        parameters: Default::default(),
    };

    let err = client.infer(req).await.unwrap_err();
    assert_eq!(err.code(), Code::DeadlineExceeded);
    assert!(err.message().contains("Client request deadline exceeded"));
}

#[tokio::test]
async fn test_grpc_parameters_map_rejection() {
    let http_port = get_free_port().await;
    let grpc_port = get_free_port().await;

    let registry = ModelRegistry::new();
    let model_cfg = ModelConfig {
        id: "params-model".to_string(),
        onnx_path: "mock:test".to_string(),
        ..Default::default()
    };
    registry.register(ModelController::new(model_cfg)).await;

    let config = ServerConfig {
        host: "127.0.0.1".to_string(),
        port: http_port,
        grpc_port,
        grpc_enabled: true,
        sweeper_interval: Duration::from_secs(60),
        auth: None,
        tls: None,
        models_dir: std::path::PathBuf::from("./models"),
        streams: Vec::new(),
    };

    tokio::spawn(async move {
        let _ = run_server(config, registry).await;
    });

    wait_for_port(grpc_port).await;

    let channel = tonic::transport::Channel::from_shared(format!("http://127.0.0.1:{grpc_port}"))
        .unwrap()
        .connect()
        .await
        .unwrap();

    let mut client = InferenceServiceClient::new(channel);

    let mut params = std::collections::HashMap::new();
    params.insert("temperature".to_string(), "0.7".to_string());

    let req = ModelInferRequest {
        model_id: "params-model".to_string(),
        inputs: vec![],
        client_timeout_ms: None,
        parameters: params,
    };

    let err = client.infer(req).await.unwrap_err();
    assert_eq!(err.code(), Code::InvalidArgument);
    assert!(err.message().contains("Unsupported request parameters"));
}

#[tokio::test]
async fn test_grpc_rejects_non_finite_floats() {
    let http_port = get_free_port().await;
    let grpc_port = get_free_port().await;

    let registry = ModelRegistry::new();
    let model_cfg = ModelConfig {
        id: "nan-model".to_string(),
        onnx_path: "mock:test".to_string(),
        ..Default::default()
    };
    registry.register(ModelController::new(model_cfg)).await;

    let config = ServerConfig {
        host: "127.0.0.1".to_string(),
        port: http_port,
        grpc_port,
        grpc_enabled: true,
        sweeper_interval: Duration::from_secs(60),
        auth: None,
        tls: None,
        models_dir: std::path::PathBuf::from("./models"),
        streams: Vec::new(),
    };

    tokio::spawn(async move {
        let _ = run_server(config, registry).await;
    });

    wait_for_port(grpc_port).await;

    let channel = tonic::transport::Channel::from_shared(format!("http://127.0.0.1:{grpc_port}"))
        .unwrap()
        .connect()
        .await
        .unwrap();

    let mut client = InferenceServiceClient::new(channel);

    let nan_floats = vec![1.0f32, f32::NAN];
    let raw_bytes = bytemuck::cast_slice::<f32, u8>(&nan_floats).to_vec();

    let req = ModelInferRequest {
        model_id: "nan-model".to_string(),
        inputs: vec![TensorData {
            name: "input_ids".to_string(),
            shape: vec![1, 2],
            data_type: DataType::Fp32 as i32,
            raw_data: raw_bytes,
        }],
        client_timeout_ms: None,
        parameters: Default::default(),
    };

    let err = client.infer(req).await.unwrap_err();
    assert_eq!(err.code(), Code::InvalidArgument);
    assert!(err.message().contains("non-finite float values"));
}

#[tokio::test]
async fn test_grpc_large_10mb_payload_and_max_message_sizes() {
    let http_port = get_free_port().await;
    let grpc_port = get_free_port().await;

    let registry = ModelRegistry::new();
    let model_cfg = ModelConfig {
        id: "large-grpc-model".to_string(),
        onnx_path: "mock:test".to_string(),
        max_batch_size: 4,
        max_batch_delay: Duration::from_millis(5),
        ..Default::default()
    };
    registry.register(ModelController::new(model_cfg)).await;

    let config = ServerConfig {
        host: "127.0.0.1".to_string(),
        port: http_port,
        grpc_port,
        grpc_enabled: true,
        sweeper_interval: Duration::from_secs(60),
        auth: None,
        tls: None,
        models_dir: std::path::PathBuf::from("./models"),
        streams: Vec::new(),
    };

    tokio::spawn(async move {
        let _ = run_server(config, registry).await;
    });

    wait_for_port(grpc_port).await;

    let channel = tonic::transport::Channel::from_shared(format!("http://127.0.0.1:{grpc_port}"))
        .unwrap()
        .connect()
        .await
        .unwrap();

    // 2,621,440 floats * 4 bytes = 10,485,760 bytes (~10MB), exceeding Tonic's default 4MB limit
    let float_count = 2_621_440;
    let large_floats = vec![0.5f32; float_count];
    let raw_bytes = bytemuck::cast_slice::<f32, u8>(&large_floats).to_vec();

    let mut client = InferenceServiceClient::new(channel)
        .max_decoding_message_size(128 * 1024 * 1024)
        .max_encoding_message_size(128 * 1024 * 1024);

    let req = ModelInferRequest {
        model_id: "large-grpc-model".to_string(),
        inputs: vec![TensorData {
            name: "input_ids".to_string(),
            shape: vec![1, float_count as i64],
            data_type: DataType::Fp32 as i32,
            raw_data: raw_bytes,
        }],
        client_timeout_ms: None,
        parameters: Default::default(),
    };

    let resp = client.infer(req).await.unwrap().into_inner();
    assert_eq!(resp.model_id, "large-grpc-model");
    assert_eq!(resp.outputs.len(), 1);
}

#[tokio::test]
async fn test_grpc_error_parity_oom_wire() {
    let http_port = get_free_port().await;
    let grpc_port = get_free_port().await;

    let registry = ModelRegistry::new();
    let model_cfg = ModelConfig {
        id: "oom-model".to_string(),
        onnx_path: "mock:oom".to_string(),
        ..Default::default()
    };
    registry.register(ModelController::new(model_cfg)).await;

    let config = ServerConfig {
        host: "127.0.0.1".to_string(),
        port: http_port,
        grpc_port,
        grpc_enabled: true,
        sweeper_interval: Duration::from_secs(60),
        auth: None,
        tls: None,
        models_dir: std::path::PathBuf::from("./models"),
        streams: Vec::new(),
    };

    tokio::spawn(async move {
        let _ = run_server(config, registry).await;
    });

    wait_for_port(grpc_port).await;

    let channel = tonic::transport::Channel::from_shared(format!("http://127.0.0.1:{grpc_port}"))
        .unwrap()
        .connect()
        .await
        .unwrap();

    let mut client = InferenceServiceClient::new(channel);

    let req = ModelInferRequest {
        model_id: "oom-model".to_string(),
        inputs: vec![TensorData {
            name: "input".to_string(),
            shape: vec![1, 2],
            data_type: DataType::Fp32 as i32,
            raw_data: bytemuck::cast_slice::<f32, u8>(&[1.0, 2.0]).to_vec(),
        }],
        client_timeout_ms: None,
        parameters: Default::default(),
    };

    let err = client.infer(req).await.unwrap_err();
    assert_eq!(err.code(), Code::ResourceExhausted);
}
