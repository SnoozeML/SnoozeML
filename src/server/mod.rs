//! Server lifecycle supervisor managing HTTP, Unix Domain Sockets, and graceful shutdown.

pub mod grpc;
pub mod http;
pub mod metrics;

use crate::auth::{AuthConfig, AuthService};
use crate::registry::ModelRegistry;
use crate::server::grpc::proto::inference_service_server::InferenceServiceServer;
use crate::server::grpc::SnoozeInferenceService;
use crate::server::http::{create_router, AppState};
use crate::server::metrics::ServerMetrics;
use crate::sweeper::spawn_scale_to_zero_sweeper;
use axum_server::tls_rustls::RustlsConfig;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpListener;
use tokio::sync::watch;
use tower_http::trace::{DefaultMakeSpan, DefaultOnFailure, DefaultOnResponse, TraceLayer};
use tower_http::LatencyUnit;
use tracing::Level;

/// Configuration parameters for TLS encryption on HTTP and gRPC listeners.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TlsConfig {
    /// Path to the TLS certificate chain (PEM format).
    pub cert_path: PathBuf,
    /// Path to the TLS private key (PEM format).
    pub key_path: PathBuf,
}

impl TlsConfig {
    /// Creates a new TLS configuration with the given certificate and private key paths.
    #[must_use]
    pub fn new(cert_path: impl Into<PathBuf>, key_path: impl Into<PathBuf>) -> Self {
        Self {
            cert_path: cert_path.into(),
            key_path: key_path.into(),
        }
    }
}

/// Configuration parameters for running the inference server.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// IP address or hostname to bind (e.g. `"0.0.0.0"` or `"127.0.0.1"`).
    pub host: String,
    /// TCP port to bind for HTTP REST (e.g. `8080`).
    pub port: u16,
    /// TCP port to bind for gRPC (e.g. `50051`).
    pub grpc_port: u16,
    /// Whether to enable the gRPC server alongside HTTP.
    pub grpc_enabled: bool,
    /// Periodic interval for sweeping and deallocating idle models.
    pub sweeper_interval: Duration,
    /// Optional authentication and role-based access control settings.
    pub auth: Option<AuthConfig>,
    /// Optional TLS configuration for securing HTTP and gRPC listeners.
    pub tls: Option<TlsConfig>,
    /// Base models directory path for discovery and runtime uploads.
    pub models_dir: PathBuf,
    /// Configured message stream bindings for asynchronous ingestion.
    pub streams: Vec<crate::config::StreamBindingConfig>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8080,
            grpc_port: 50051,
            grpc_enabled: true,
            sweeper_interval: Duration::from_secs(5),
            auth: None,
            tls: None,
            models_dir: PathBuf::from(crate::config::default_models_dir()),
            streams: Vec::new(),
        }
    }
}

/// Runs the SnoozeML inference server to completion or until shutdown signal.
///
/// Serves HTTP REST endpoints and optionally gRPC inference and health endpoints concurrently.
///
/// # Errors
/// Returns [`std::io::Error`] if binding the TCP listener fails, TLS credentials cannot be loaded,
/// or runtime errors occur.
pub async fn run_server(
    config: ServerConfig,
    registry: ModelRegistry,
) -> Result<(), std::io::Error> {
    let metrics = ServerMetrics::new_shared();
    registry.set_metrics(metrics.clone()).await;

    // Launch background scale-to-zero sweeper
    let _sweeper = spawn_scale_to_zero_sweeper(registry.clone(), config.sweeper_interval);

    let auth_service = match config.auth {
        Some(auth_cfg) if auth_cfg.enabled => {
            let svc = AuthService::new(auth_cfg).map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("Failed to initialize authentication: {e}"),
                )
            })?;
            Some(Arc::new(svc))
        }
        _ => None,
    };

    let mut state =
        AppState::new(registry.clone(), metrics.clone()).with_models_dir(config.models_dir);
    if let Some(auth) = auth_service {
        state = state.with_auth(auth);
    }

    let router = create_router(state.clone());

    if config.tls.is_none() {
        tracing::warn!(
            tls = false,
            "\n================================================================================\n\
             SECURITY WARNING: TLS IS CURRENTLY DISABLED (HTTP & gRPC CLEARTEXT MODE)!\n\
             All inference traffic, tokens, and payloads will be sent unencrypted.\n\
             To enable TLS encryption, provide --tls-cert and --tls-key paths or configure\n\
             SNOOZE_TLS_CERT and SNOOZE_TLS_KEY.\n\
             In production deployments, ensure traffic is routed through a TLS-terminating\n\
             reverse proxy (such as Nginx, Caddy, or Envoy).\n\
             ================================================================================"
        );
    } else {
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
        tracing::info!(
            tls = true,
            "Native TLS encryption enabled for HTTP and gRPC listeners"
        );
    }

    let http_addr: SocketAddr = format!("{}:{}", config.host, config.port)
        .parse()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;

    let (shutdown_tx, shutdown_rx_http) = watch::channel(());
    let mut shutdown_rx_grpc = shutdown_rx_http.clone();

    #[cfg(feature = "streams")]
    if !config.streams.is_empty() {
        let stream_supervisor = Arc::new(crate::stream::StreamSupervisor::new(
            registry.clone(),
            metrics.clone(),
            config.streams.clone(),
        ));
        stream_supervisor.spawn_workers(shutdown_rx_http.clone());
    }

    tokio::spawn(async move {
        shutdown_signal().await;
        let _ = shutdown_tx.send(());
    });

    let http_server = {
        let mut shutdown_rx = shutdown_rx_http.clone();
        let tls_config = config.tls.clone();
        async move {
            match tls_config {
                Some(tls_cfg) => {
                    let rustls_config =
                        RustlsConfig::from_pem_file(&tls_cfg.cert_path, &tls_cfg.key_path)
                            .await
                            .map_err(|e| {
                                std::io::Error::new(
                                    std::io::ErrorKind::InvalidInput,
                                    format!(
                                    "Failed to load TLS certificate or key from '{}' / '{}': {e}",
                                    tls_cfg.cert_path.display(),
                                    tls_cfg.key_path.display()
                                ),
                                )
                            })?;

                    let handle = axum_server::Handle::new();
                    let shutdown_handle = handle.clone();
                    tokio::spawn(async move {
                        let _ = shutdown_rx.changed().await;
                        shutdown_handle.graceful_shutdown(Some(Duration::from_secs(5)));
                    });

                    tracing::info!(
                        address = %http_addr,
                        tls = true,
                        "SnoozeML HTTPS REST Server listening for connections"
                    );

                    axum_server::bind_rustls(http_addr, rustls_config)
                        .handle(handle)
                        .serve(router.into_make_service())
                        .await
                }
                None => {
                    let http_listener = TcpListener::bind(http_addr).await?;
                    tracing::info!(
                        address = %http_addr,
                        tls = false,
                        "SnoozeML HTTP REST Server listening for connections"
                    );

                    axum::serve(http_listener, router)
                        .with_graceful_shutdown(async move {
                            let _ = shutdown_rx.changed().await;
                        })
                        .await
                }
            }
        }
    };

    if config.grpc_enabled {
        let grpc_addr: SocketAddr = format!("{}:{}", config.host, config.grpc_port)
            .parse()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;

        let (mut health_reporter, health_service) = tonic_health::server::health_reporter();
        health_reporter
            .set_serving::<InferenceServiceServer<SnoozeInferenceService>>()
            .await;
        health_reporter
            .set_service_status("", tonic_health::ServingStatus::Serving)
            .await;

        let grpc_inference_service = SnoozeInferenceService::new(state);

        tracing::info!(
            address = %grpc_addr,
            tls = config.tls.is_some(),
            "SnoozeML gRPC Server listening for connections"
        );

        let mut grpc_builder = tonic::transport::Server::builder();

        if let Some(ref tls_cfg) = config.tls {
            let cert_pem = tokio::fs::read(&tls_cfg.cert_path).await.map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!(
                        "Failed to read gRPC TLS certificate file '{}': {e}",
                        tls_cfg.cert_path.display()
                    ),
                )
            })?;
            let key_pem = tokio::fs::read(&tls_cfg.key_path).await.map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!(
                        "Failed to read gRPC TLS private key file '{}': {e}",
                        tls_cfg.key_path.display()
                    ),
                )
            })?;
            let identity = tonic::transport::Identity::from_pem(cert_pem, key_pem);
            let tls_config = tonic::transport::ServerTlsConfig::new().identity(identity);
            grpc_builder = grpc_builder.tls_config(tls_config).map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("Invalid gRPC TLS configuration: {e}"),
                )
            })?;
        }

        let grpc_server = grpc_builder
            .layer(
                TraceLayer::new_for_grpc()
                    .make_span_with(DefaultMakeSpan::new().level(Level::INFO))
                    .on_response(
                        DefaultOnResponse::new()
                            .level(Level::INFO)
                            .latency_unit(LatencyUnit::Millis),
                    )
                    .on_failure(
                        DefaultOnFailure::new()
                            .level(Level::WARN)
                            .latency_unit(LatencyUnit::Millis),
                    ),
            )
            .add_service(health_service)
            .add_service(
                InferenceServiceServer::new(grpc_inference_service)
                    .max_decoding_message_size(128 * 1024 * 1024)
                    .max_encoding_message_size(128 * 1024 * 1024),
            )
            .serve_with_shutdown(grpc_addr, async move {
                let _ = shutdown_rx_grpc.changed().await;
            });

        let (http_res, grpc_res) = tokio::join!(http_server, grpc_server);
        http_res?;
        grpc_res.map_err(std::io::Error::other)?;
    } else {
        http_server.await?;
    }

    tracing::info!("Server shut down gracefully");
    Ok(())
}

/// Awaits SIGINT / Ctrl+C for graceful server shutdown.
async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };

    #[cfg(unix)]
    let terminate = async {
        if let Ok(mut sig) =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        {
            sig.recv().await;
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            tracing::info!("Received Ctrl+C; beginning graceful termination");
        }
        _ = terminate => {
            tracing::info!("Received SIGTERM; beginning graceful termination");
        }
    }
}
