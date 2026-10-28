//! End-to-end integration tests for NVIDIA Triton & Open Inference Protocol (OIP v2)
//! across both gRPC (`inference.GRPCInferenceService`) and HTTP REST (`/v2/...`).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use snooze_ml::auth::{ApiKeyConfig, AuthConfig, Role};
use snooze_ml::model::{ModelConfig, ModelController};
use snooze_ml::registry::ModelRegistry;
use snooze_ml::server::grpc::triton_proto::grpc_inference_service_client::GrpcInferenceServiceClient;
use snooze_ml::server::grpc::triton_proto::model_infer_request::{
    InferInputTensor, InferRequestedOutputTensor,
};
use snooze_ml::server::grpc::triton_proto::{
    CudaSharedMemoryRegisterRequest, CudaSharedMemoryStatusRequest,
    CudaSharedMemoryUnregisterRequest, InferTensorContents, LogSettingsRequest, ModelConfigRequest,
    ModelInferRequest, ModelMetadataRequest, ModelReadyRequest, ModelStatisticsRequest,
    RepositoryIndexRequest, RepositoryModelLoadRequest, RepositoryModelUnloadRequest,
    ServerLiveRequest, ServerMetadataRequest, ServerReadyRequest,
    SystemSharedMemoryRegisterRequest, SystemSharedMemoryStatusRequest,
    SystemSharedMemoryUnregisterRequest, TraceSettingRequest,
};
use snooze_ml::server::{run_server, ServerConfig};
use std::collections::HashMap;
use std::time::Duration;
use tokio_stream::StreamExt;
use tonic::Code;
use tonic_health::pb::health_client::HealthClient;
use tonic_health::pb::HealthCheckRequest;

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
async fn test_triton_grpc_core_lifecycle_and_inference() {
    let http_port = get_free_port().await;
    let grpc_port = get_free_port().await;

    let registry = ModelRegistry::new();
    let model_cfg = ModelConfig {
        id: "triton-mock-model".to_string(),
        onnx_path: "mock:test".to_string(),
        idle_timeout: Duration::from_secs(60),
        max_batch_size: 8,
        max_batch_delay: Duration::from_millis(5),
        provider: Some("cpu".to_string()),
        description: Some("Triton mock model".to_string()),
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

    // 0. Check grpc.health.v1 reports inference.GRPCInferenceService as SERVING
    let mut health_client = HealthClient::new(channel.clone());
    let h_resp = health_client
        .check(HealthCheckRequest {
            service: "inference.GRPCInferenceService".to_string(),
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(h_resp.status, 1);

    let mut client = GrpcInferenceServiceClient::new(channel);

    // 1. ServerLive & ServerReady
    assert!(
        client
            .server_live(ServerLiveRequest {})
            .await
            .unwrap()
            .into_inner()
            .live
    );
    assert!(
        client
            .server_ready(ServerReadyRequest {})
            .await
            .unwrap()
            .into_inner()
            .ready
    );

    // 2. ServerMetadata
    let server_meta = client
        .server_metadata(ServerMetadataRequest {})
        .await
        .unwrap()
        .into_inner();
    assert_eq!(server_meta.name, "snooze-ml");
    assert!(server_meta
        .extensions
        .contains(&"model_repository".to_string()));

    // 3. ModelReady before loading (scale-to-zero Idle -> ready == false)
    let ready_before = client
        .model_ready(ModelReadyRequest {
            name: "triton-mock-model".to_string(),
            version: "1".to_string(),
        })
        .await
        .unwrap()
        .into_inner();
    assert!(!ready_before.ready);

    // 4. RepositoryIndex (all vs ready=true)
    let idx_all = client
        .repository_index(RepositoryIndexRequest {
            repository_name: String::new(),
            ready: false,
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(idx_all.models.len(), 1);
    assert_eq!(idx_all.models[0].name, "triton-mock-model");
    assert_eq!(idx_all.models[0].state, "UNAVAILABLE");

    let idx_ready = client
        .repository_index(RepositoryIndexRequest {
            repository_name: String::new(),
            ready: true,
        })
        .await
        .unwrap()
        .into_inner();
    assert!(idx_ready.models.is_empty());

    // 5. RepositoryModelLoad -> ModelReady becomes true
    client
        .repository_model_load(RepositoryModelLoadRequest {
            repository_name: String::new(),
            model_name: "triton-mock-model".to_string(),
            parameters: HashMap::new(),
        })
        .await
        .unwrap();

    let ready_after_load = client
        .model_ready(ModelReadyRequest {
            name: "triton-mock-model".to_string(),
            version: "1".to_string(),
        })
        .await
        .unwrap()
        .into_inner();
    assert!(ready_after_load.ready);

    // 6. ModelMetadata & ModelConfig
    let model_meta = client
        .model_metadata(ModelMetadataRequest {
            name: "triton-mock-model".to_string(),
            version: "1".to_string(),
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(model_meta.name, "triton-mock-model");
    assert_eq!(model_meta.platform, "mock");

    let model_cfg_resp = client
        .model_config(ModelConfigRequest {
            name: "triton-mock-model".to_string(),
            version: "1".to_string(),
        })
        .await
        .unwrap()
        .into_inner();
    let cfg = model_cfg_resp.config.unwrap();
    assert_eq!(cfg.name, "triton-mock-model");
    assert_eq!(cfg.max_batch_size, 8);

    // 7. ModelInfer using raw_input_contents
    let input_floats = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32];
    let raw_bytes = bytemuck::cast_slice::<f32, u8>(&input_floats).to_vec();
    let infer_raw_resp = client
        .model_infer(ModelInferRequest {
            model_name: "triton-mock-model".to_string(),
            model_version: "1".to_string(),
            id: "req-raw-1".to_string(),
            parameters: HashMap::new(),
            inputs: vec![InferInputTensor {
                name: "input".to_string(),
                datatype: "FP32".to_string(),
                shape: vec![2, 2],
                parameters: HashMap::new(),
                contents: None,
            }],
            outputs: vec![InferRequestedOutputTensor {
                name: "output".to_string(),
                parameters: HashMap::new(),
            }],
            raw_input_contents: vec![raw_bytes],
        })
        .await
        .unwrap()
        .into_inner();

    assert_eq!(infer_raw_resp.id, "req-raw-1");
    assert_eq!(infer_raw_resp.outputs.len(), 1);
    assert_eq!(infer_raw_resp.outputs[0].shape, vec![2, 4]);
    assert_eq!(infer_raw_resp.raw_output_contents.len(), 1);
    let out_floats: &[f32] = bytemuck::cast_slice(&infer_raw_resp.raw_output_contents[0]);
    assert_eq!(out_floats.len(), 8);

    // 8. ModelInfer using structured InferTensorContents.fp32_contents
    let infer_contents_resp = client
        .model_infer(ModelInferRequest {
            model_name: "triton-mock-model".to_string(),
            model_version: String::new(),
            id: "req-contents-2".to_string(),
            parameters: HashMap::new(),
            inputs: vec![InferInputTensor {
                name: "input".to_string(),
                datatype: "FP32".to_string(),
                shape: vec![1, 3],
                parameters: HashMap::new(),
                contents: Some(InferTensorContents {
                    fp32_contents: vec![5.0, 6.0, 7.0],
                    ..Default::default()
                }),
            }],
            outputs: Vec::new(),
            raw_input_contents: Vec::new(),
        })
        .await
        .unwrap()
        .into_inner();

    assert_eq!(infer_contents_resp.id, "req-contents-2");
    assert!(infer_contents_resp.raw_output_contents.is_empty());
    let out_contents = infer_contents_resp.outputs[0].contents.as_ref().unwrap();
    assert_eq!(out_contents.fp32_contents.len(), 4);

    // 9. ModelStatistics verifies cumulative counts (2 requests, 3 total batch items)
    let stats_resp = client
        .model_statistics(ModelStatisticsRequest {
            name: "triton-mock-model".to_string(),
            version: "1".to_string(),
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(stats_resp.model_stats.len(), 1);
    let st = &stats_resp.model_stats[0];
    assert_eq!(st.inference_count, 3);
    assert_eq!(st.execution_count, 2);
    assert_eq!(
        st.inference_stats.as_ref().unwrap().success.unwrap().count,
        2
    );

    // 10. RepositoryModelUnload -> ModelReady becomes false again
    client
        .repository_model_unload(RepositoryModelUnloadRequest {
            repository_name: String::new(),
            model_name: "triton-mock-model".to_string(),
            parameters: HashMap::new(),
        })
        .await
        .unwrap();

    let ready_after_unload = client
        .model_ready(ModelReadyRequest {
            name: "triton-mock-model".to_string(),
            version: "1".to_string(),
        })
        .await
        .unwrap()
        .into_inner();
    assert!(!ready_after_unload.ready);
}

#[tokio::test]
async fn test_triton_grpc_stream_infer_and_inline_errors() {
    let http_port = get_free_port().await;
    let grpc_port = get_free_port().await;

    let registry = ModelRegistry::new();
    let model_cfg = ModelConfig {
        id: "triton-stream-model".to_string(),
        onnx_path: "mock:test".to_string(),
        idle_timeout: Duration::from_secs(60),
        max_batch_size: 8,
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

    let mut client = GrpcInferenceServiceClient::new(channel);

    let valid_req_1 = ModelInferRequest {
        model_name: "triton-stream-model".to_string(),
        model_version: "1".to_string(),
        id: "stream-1".to_string(),
        parameters: HashMap::new(),
        inputs: vec![InferInputTensor {
            name: "input".to_string(),
            datatype: "FP32".to_string(),
            shape: vec![1, 2],
            parameters: HashMap::new(),
            contents: Some(InferTensorContents {
                fp32_contents: vec![1.0, 2.0],
                ..Default::default()
            }),
        }],
        outputs: Vec::new(),
        raw_input_contents: Vec::new(),
    };

    // Invalid request in the middle of the stream (missing model)
    let invalid_req = ModelInferRequest {
        model_name: "missing-model".to_string(),
        model_version: "1".to_string(),
        id: "stream-err".to_string(),
        parameters: HashMap::new(),
        inputs: Vec::new(),
        outputs: Vec::new(),
        raw_input_contents: Vec::new(),
    };

    let valid_req_2 = ModelInferRequest {
        id: "stream-2".to_string(),
        ..valid_req_1.clone()
    };

    let mut stream = client
        .model_stream_infer(tokio_stream::iter(vec![
            valid_req_1,
            invalid_req,
            valid_req_2,
        ]))
        .await
        .unwrap()
        .into_inner();

    let mut responses = Vec::new();
    while let Some(res) = stream.next().await {
        responses.push(res.unwrap());
    }

    assert_eq!(responses.len(), 3);
    assert!(responses[0].error_message.is_empty());
    assert_eq!(responses[0].infer_response.as_ref().unwrap().id, "stream-1");

    // Second frame reported inline error without breaking the stream
    assert!(!responses[1].error_message.is_empty());
    assert!(responses[1].infer_response.is_none());

    // Third frame succeeded
    assert!(responses[2].error_message.is_empty());
    assert_eq!(responses[2].infer_response.as_ref().unwrap().id, "stream-2");
}

#[tokio::test]
async fn test_triton_grpc_unimplemented_endpoints() {
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

    let mut client = GrpcInferenceServiceClient::new(channel);

    assert_eq!(
        client
            .system_shared_memory_status(SystemSharedMemoryStatusRequest::default())
            .await
            .unwrap_err()
            .code(),
        Code::Unimplemented
    );
    assert_eq!(
        client
            .system_shared_memory_register(SystemSharedMemoryRegisterRequest::default())
            .await
            .unwrap_err()
            .code(),
        Code::Unimplemented
    );
    assert_eq!(
        client
            .system_shared_memory_unregister(SystemSharedMemoryUnregisterRequest::default())
            .await
            .unwrap_err()
            .code(),
        Code::Unimplemented
    );
    assert_eq!(
        client
            .cuda_shared_memory_status(CudaSharedMemoryStatusRequest::default())
            .await
            .unwrap_err()
            .code(),
        Code::Unimplemented
    );
    assert_eq!(
        client
            .cuda_shared_memory_register(CudaSharedMemoryRegisterRequest::default())
            .await
            .unwrap_err()
            .code(),
        Code::Unimplemented
    );
    assert_eq!(
        client
            .cuda_shared_memory_unregister(CudaSharedMemoryUnregisterRequest::default())
            .await
            .unwrap_err()
            .code(),
        Code::Unimplemented
    );
    assert_eq!(
        client
            .trace_setting(TraceSettingRequest::default())
            .await
            .unwrap_err()
            .code(),
        Code::Unimplemented
    );
    assert_eq!(
        client
            .log_settings(LogSettingsRequest::default())
            .await
            .unwrap_err()
            .code(),
        Code::Unimplemented
    );
}

#[tokio::test]
async fn test_triton_grpc_rbac_enforcement() {
    let http_port = get_free_port().await;
    let grpc_port = get_free_port().await;

    let registry = ModelRegistry::new();
    registry
        .register(ModelController::new(ModelConfig {
            id: "rbac-model".to_string(),
            onnx_path: "mock:test".to_string(),
            ..Default::default()
        }))
        .await;

    let auth = AuthConfig {
        enabled: true,
        protect_metrics: false,
        api_keys: vec![
            ApiKeyConfig::new("sk-read", Role::ReadOnly),
            ApiKeyConfig::new("sk-infer", Role::Inference),
            ApiKeyConfig::new("sk-admin", Role::Admin),
        ],
        jwt: None,
    };

    let config = ServerConfig {
        host: "127.0.0.1".to_string(),
        port: http_port,
        grpc_port,
        grpc_enabled: true,
        sweeper_interval: Duration::from_secs(60),
        auth: Some(auth),
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
    let mut client = GrpcInferenceServiceClient::new(channel);

    // 1. ServerLive is public (no key required)
    assert!(
        client
            .server_live(ServerLiveRequest {})
            .await
            .unwrap()
            .into_inner()
            .live
    );

    // 2. ServerMetadata without key -> Unauthenticated; with ReadOnly key -> OK
    assert_eq!(
        client
            .server_metadata(ServerMetadataRequest {})
            .await
            .unwrap_err()
            .code(),
        Code::Unauthenticated
    );
    let mut meta_req = tonic::Request::new(ServerMetadataRequest {});
    meta_req
        .metadata_mut()
        .insert("x-api-key", "sk-read".parse().unwrap());
    assert!(client.server_metadata(meta_req).await.is_ok());

    // 3. RepositoryModelLoad with Inference key -> PermissionDenied; with Admin key -> OK
    let mut load_infer = tonic::Request::new(RepositoryModelLoadRequest {
        repository_name: String::new(),
        model_name: "rbac-model".to_string(),
        parameters: HashMap::new(),
    });
    load_infer
        .metadata_mut()
        .insert("x-api-key", "sk-infer".parse().unwrap());
    assert_eq!(
        client
            .repository_model_load(load_infer)
            .await
            .unwrap_err()
            .code(),
        Code::PermissionDenied
    );

    let mut load_admin = tonic::Request::new(RepositoryModelLoadRequest {
        repository_name: String::new(),
        model_name: "rbac-model".to_string(),
        parameters: HashMap::new(),
    });
    load_admin
        .metadata_mut()
        .insert("x-api-key", "sk-admin".parse().unwrap());
    assert!(client.repository_model_load(load_admin).await.is_ok());
}

#[tokio::test]
async fn test_triton_http_v2_endpoints() {
    let http_port = get_free_port().await;
    let grpc_port = get_free_port().await;

    let registry = ModelRegistry::new();
    registry
        .register(ModelController::new(ModelConfig {
            id: "http-v2-model".to_string(),
            onnx_path: "mock:test".to_string(),
            max_batch_size: 4,
            max_batch_delay: Duration::from_millis(5),
            ..Default::default()
        }))
        .await;

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
    let base = format!("http://127.0.0.1:{http_port}");

    tokio::task::spawn_blocking(move || {
        // 1. Liveness & Readiness
        let r = ureq::get(&format!("{base}/v2/health/live")).call().unwrap();
        assert_eq!(r.status(), 200);
        let r = ureq::get(&format!("{base}/v2/health/ready"))
            .call()
            .unwrap();
        assert_eq!(r.status(), 200);

        // 2. Server metadata
        let mut r = ureq::get(&format!("{base}/v2")).call().unwrap();
        let meta: serde_json::Value = r.body_mut().read_json().unwrap();
        assert_eq!(meta["name"], "snooze-ml");

        // 3. Model readiness before load -> 400
        let ready_before = ureq::get(&format!("{base}/v2/models/http-v2-model/ready")).call();
        assert!(ready_before.is_err());

        // 4. Repository load -> 200, then Model ready -> 200
        let r = ureq::post(&format!("{base}/v2/repository/models/http-v2-model/load"))
            .send_empty()
            .unwrap();
        assert_eq!(r.status(), 200);

        let r = ureq::get(&format!("{base}/v2/models/http-v2-model/ready"))
            .call()
            .unwrap();
        assert_eq!(r.status(), 200);

        // 5. OIP v2 HTTP Inference (with nested array data)
        let mut infer_resp = ureq::post(&format!("{base}/v2/models/http-v2-model/infer"))
            .send_json(serde_json::json!({
                "id": "http-req-1",
                "inputs": [
                    {
                        "name": "input",
                        "shape": [2, 2],
                        "datatype": "FP32",
                        "data": [[1.0, 2.0], [3.0, 4.0]]
                    }
                ]
            }))
            .unwrap();
        assert_eq!(infer_resp.status(), 200);
        let infer_json: serde_json::Value = infer_resp.body_mut().read_json().unwrap();
        assert_eq!(infer_json["id"], "http-req-1");
        assert_eq!(infer_json["outputs"][0]["shape"], serde_json::json!([2, 4]));

        // 6. Model stats & config
        let mut stats_resp = ureq::get(&format!("{base}/v2/models/http-v2-model/stats"))
            .call()
            .unwrap();
        let stats_json: serde_json::Value = stats_resp.body_mut().read_json().unwrap();
        assert_eq!(stats_json["model_stats"][0]["inference_count"], 2);

        let mut cfg_resp = ureq::get(&format!("{base}/v2/models/http-v2-model/config"))
            .call()
            .unwrap();
        let cfg_json: serde_json::Value = cfg_resp.body_mut().read_json().unwrap();
        assert_eq!(cfg_json["max_batch_size"], 4);

        // 7. Repository Index
        let mut idx_resp = ureq::post(&format!("{base}/v2/repository/index"))
            .send_json(serde_json::json!({ "ready": true }))
            .unwrap();
        let idx_json: serde_json::Value = idx_resp.body_mut().read_json().unwrap();
        assert_eq!(idx_json[0]["name"], "http-v2-model");
        assert_eq!(idx_json[0]["state"], "READY");

        // 8. Out-of-scope endpoint returns 501 Not Implemented
        let unimpl = ureq::get(&format!("{base}/v2/cudasharedmemory/status")).call();
        match unimpl {
            Err(ureq::Error::StatusCode(code)) => assert_eq!(code, 501),
            other => panic!("Expected 501 status code, got {other:?}"),
        }
    })
    .await
    .unwrap();
}
