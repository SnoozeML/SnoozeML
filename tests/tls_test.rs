//! Integration tests for SnoozeML native TLS support (HTTP & gRPC).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use rcgen::generate_simple_self_signed;
use snooze_ml::registry::ModelRegistry;
use snooze_ml::server::grpc::proto::inference_service_client::InferenceServiceClient;
use snooze_ml::server::grpc::proto::ModelInferRequest;
use snooze_ml::server::{run_server, ServerConfig, TlsConfig};
use std::time::Duration;
use tokio::net::TcpListener;
use tonic::transport::{Certificate, Channel, ClientTlsConfig};
use tonic_health::pb::health_client::HealthClient;
use tonic_health::pb::HealthCheckRequest;

/// Helper to allocate an ephemeral unused TCP port.
async fn get_free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    port
}

/// Helper to wait until a port is ready to accept connections.
async fn wait_for_port(port: u16) {
    let start = std::time::Instant::now();
    loop {
        if tokio::net::TcpStream::connect(("127.0.0.1", port))
            .await
            .is_ok()
        {
            return;
        }
        if start.elapsed() > Duration::from_secs(10) {
            panic!("Timed out waiting for port {port} to open");
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

#[tokio::test]
async fn test_tls_config_default_disabled() {
    let config = ServerConfig::default();
    assert_eq!(config.tls, None);
    assert!(!config.host.is_empty());
}

#[tokio::test]
async fn test_tls_grpc_and_https_lifecycle() {
    let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();

    // 1. Generate self-signed certificate for localhost and 127.0.0.1
    let subject_alt_names = vec!["localhost".to_string(), "127.0.0.1".to_string()];
    let cert = generate_simple_self_signed(subject_alt_names).unwrap();
    let cert_pem = cert.cert.pem();
    let key_pem = cert.signing_key.serialize_pem();

    let temp_dir = std::env::temp_dir().join(format!("snooze_tls_test_{}", std::process::id()));
    tokio::fs::create_dir_all(&temp_dir).await.unwrap();
    let cert_path = temp_dir.join("cert.pem");
    let key_path = temp_dir.join("key.pem");
    tokio::fs::write(&cert_path, &cert_pem).await.unwrap();
    tokio::fs::write(&key_path, &key_pem).await.unwrap();

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
        tls: Some(TlsConfig::new(cert_path.clone(), key_path.clone())),
        models_dir: std::path::PathBuf::from("./models"),
        streams: Vec::new(),
    };

    tokio::spawn(async move {
        let _ = run_server(config, registry).await;
    });

    wait_for_port(grpc_port).await;

    // 2. Connect to gRPC with TLS
    let ca_cert = Certificate::from_pem(cert_pem);
    let tls_config = ClientTlsConfig::new()
        .ca_certificate(ca_cert)
        .domain_name("localhost");

    let channel = Channel::from_shared(format!("https://127.0.0.1:{grpc_port}"))
        .unwrap()
        .tls_config(tls_config)
        .unwrap()
        .connect()
        .await
        .unwrap();

    let mut health_client = HealthClient::new(channel.clone());

    let resp = health_client
        .check(HealthCheckRequest {
            service: "".to_string(),
        })
        .await
        .unwrap();

    assert_eq!(resp.into_inner().status, 1); // SERVING = 1

    // 3. Inference request via TLS channel
    let mut client = InferenceServiceClient::new(channel);
    let req = ModelInferRequest {
        model_id: "test-model".to_string(),
        inputs: vec![],
        client_timeout_ms: None,
        parameters: Default::default(),
    };
    let err = client.infer(req).await.unwrap_err();
    assert_eq!(err.code(), tonic::Code::NotFound);

    // Clean up temporary files
    let _ = tokio::fs::remove_dir_all(&temp_dir).await;
}
