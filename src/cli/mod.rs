//! Command Line Interface (CLI) commands and runner for SnoozeML.

pub mod download;
pub mod tui;

use crate::auth::{ApiKeyConfig, AuthConfig, JwtAlgorithm, JwtConfig, Role};
use crate::config::{
    default_models_dir, ModelDirectoryConfig, DEFAULT_DEVICE, DEFAULT_IDLE_TIMEOUT_SECS,
    DEFAULT_MAX_BATCH_DELAY_MS, DEFAULT_MAX_BATCH_SIZE, ENV_MODELS_DIR_FALLBACK, STUB_DESCRIPTION,
};
use crate::model::{ModelConfig, ModelController};
use crate::registry::ModelRegistry;
use crate::server::{run_server, ServerConfig, TlsConfig};
use clap::{Parser, Subcommand};
use download::{run_download, HfDownloadOptions};
use std::path::{Path, PathBuf};
use std::time::Duration;

/// High-performance ONNX & deep learning micro-batch inference daemon.
#[derive(Parser, Debug)]
#[command(name = "snooze", author, version, about, long_about = None)]
pub struct Cli {
    /// Subcommand to execute.
    #[command(subcommand)]
    pub command: Commands,
}

/// Supported CLI subcommands.
#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Start the inference daemon server.
    Serve {
        /// Network interface to bind.
        #[arg(long, default_value = "0.0.0.0")]
        host: String,

        /// TCP port to bind.
        #[arg(short, long, default_value_t = 8080)]
        port: u16,

        /// Path to directory containing `.onnx` or `.safetensors` models to discover and register.
        #[arg(long)]
        models_dir: Option<String>,

        /// Path to an explicit model configuration file (`models.toml`, `models.json`, etc.).
        #[arg(long)]
        config: Option<String>,

        /// Idle timeout in seconds before deallocating a model to scale to zero.
        #[arg(long, default_value_t = 300)]
        idle_timeout: u64,

        /// Maximum micro-batch size before triggering immediate execution.
        #[arg(long, default_value_t = 16)]
        max_batch_size: usize,

        /// Maximum batching latency budget in milliseconds.
        #[arg(long, default_value_t = 10)]
        max_delay_ms: u64,

        /// Execution provider preference (`auto`, `gpu`, `cuda`, `mlx`, `cpu`, `tensorrt`). Defaults to `auto`.
        #[arg(long)]
        provider: Option<String>,

        /// Maximum number of simultaneously loaded models before LRU deallocation.
        #[arg(long)]
        max_loaded_models: Option<usize>,

        /// Register a demo mock model for testing without external weights.
        #[arg(long)]
        demo: bool,

        /// Explicitly enable authentication enforcement. Automatically enabled if API keys or JWT secret are passed.
        #[arg(long)]
        auth_enabled: bool,

        /// Configure API keys with roles in format '<role>:<key>' (e.g. 'admin:sk-admin123', 'infer:sk-user456'). Repeatable.
        #[arg(long = "api-key", value_name = "ROLE:KEY")]
        api_keys: Vec<String>,

        /// JWT HMAC secret for validating JSON Web Tokens (or env SNOOZE_JWT_SECRET).
        #[arg(long, env = "SNOOZE_JWT_SECRET")]
        jwt_secret: Option<String>,

        /// Require authentication on Prometheus /metrics endpoint.
        #[arg(long)]
        protect_metrics: bool,

        /// Enable or disable gRPC server alongside HTTP (default: true, env: SNOOZE_GRPC, GRPC).
        #[arg(long, env = "SNOOZE_GRPC", num_args = 0..=1, default_missing_value = "true")]
        grpc: Option<bool>,

        /// Disable gRPC server and only serve HTTP REST endpoints.
        #[arg(long)]
        no_grpc: bool,

        /// TCP port to bind the gRPC server (default: 50051, env: SNOOZE_GRPC_PORT, GRPC_PORT).
        #[arg(long, env = "SNOOZE_GRPC_PORT")]
        grpc_port: Option<u16>,

        /// Enable TLS encryption for HTTP and gRPC listeners (default: false, env: SNOOZE_TLS).
        #[arg(long, env = "SNOOZE_TLS", num_args = 0..=1, default_missing_value = "true")]
        tls: Option<bool>,

        /// Path to the TLS certificate chain in PEM format (env: SNOOZE_TLS_CERT).
        #[arg(long, env = "SNOOZE_TLS_CERT")]
        tls_cert: Option<PathBuf>,

        /// Path to the TLS private key in PEM format (env: SNOOZE_TLS_KEY).
        #[arg(long, env = "SNOOZE_TLS_KEY")]
        tls_key: Option<PathBuf>,

        /// Model identifiers to pin in memory at startup (never automatically deallocated). Repeatable.
        #[arg(long = "pin", value_name = "MODEL_ID")]
        pin: Vec<String>,

        /// Path to an explicit streaming configuration file (`streams.toml`, `streams.json`).
        #[arg(long)]
        streams_config: Option<String>,
    },

    /// Manage models directory and configuration files.
    Models {
        /// Models directory operation to execute.
        #[command(subcommand)]
        action: ModelsAction,
    },

    /// Download a model repository from Hugging Face Hub.
    #[command(name = "download", alias = "hf-download")]
    Download {
        /// Hugging Face repository identifier (e.g. "openai-community/gpt2", "sentence-transformers/all-MiniLM-L6-v2").
        repo: String,

        /// Target root models directory (default: "./models", env: SNOOZE_MODELS_DIR).
        #[arg(
            short = 'd',
            long = "models-dir",
            alias = "dir",
            env = "SNOOZE_MODELS_DIR",
            default_value = "./models"
        )]
        models_dir: String,

        /// Optional destination subfolder name within models directory (defaults to repository name).
        #[arg(short = 'n', long = "name", alias = "folder")]
        folder: Option<String>,

        /// Specific git revision, branch, or tag to download (defaults to main).
        #[arg(short = 'r', long = "revision")]
        revision: Option<String>,

        /// Glob pattern(s) selecting which files to download (e.g. "*.onnx", "*.safetensors", "config.json").
        #[arg(short = 'i', long = "include", value_delimiter = ',')]
        include: Option<Vec<String>>,

        /// Glob pattern(s) to exclude from download.
        #[arg(short = 'e', long = "exclude", value_delimiter = ',')]
        exclude: Option<Vec<String>>,

        /// Hugging Face authorization token (defaults to HF_TOKEN environment variable).
        #[arg(long, env = "HF_TOKEN")]
        token: Option<String>,

        /// Skip automatically generating or updating models.toml in the models directory after download.
        #[arg(long)]
        no_init: bool,
    },

    /// Quick shortcut to initialize or update a model directory configuration.
    Init {
        /// Target models directory path (default: "./models", env: SNOOZE_MODELS_DIR).
        #[arg(default_value = "./models", env = "SNOOZE_MODELS_DIR")]
        dir: String,

        /// Custom configuration file path (defaults to models.toml in directory).
        #[arg(short, long)]
        file: Option<String>,
    },

    /// Query the status and active models of a running SnoozeML server.
    Status {
        /// Base URL endpoint of the target server.
        #[arg(long, default_value = "http://127.0.0.1:8080")]
        endpoint: String,
    },

    /// Explicitly trigger loading a model into memory.
    Load {
        /// Target server endpoint.
        #[arg(long, default_value = "http://127.0.0.1:8080")]
        endpoint: String,

        /// Model identifier to load.
        #[arg(long)]
        model_id: String,

        /// Also pin the model in memory so it is exempt from scale-to-zero and LRU eviction.
        #[arg(long)]
        pin: bool,
    },

    /// Explicitly trigger unloading a model from memory (scale-to-zero override).
    Unload {
        /// Target server endpoint.
        #[arg(long, default_value = "http://127.0.0.1:8080")]
        endpoint: String,

        /// Model identifier to unload.
        #[arg(long)]
        model_id: String,
    },

    /// Pin a model in memory so it is eagerly loaded and never automatically deallocated.
    Pin {
        /// Target server endpoint.
        #[arg(long, default_value = "http://127.0.0.1:8080")]
        endpoint: String,

        /// Model identifier to pin.
        #[arg(long)]
        model_id: String,
    },

    /// Unpin a model so it returns to normal scale-to-zero and LRU lifecycle management.
    Unpin {
        /// Target server endpoint.
        #[arg(long, default_value = "http://127.0.0.1:8080")]
        endpoint: String,

        /// Model identifier to unpin.
        #[arg(long)]
        model_id: String,
    },

    /// Launch a live updating terminal dashboard for monitoring.
    Monitor {
        /// Target server endpoint.
        #[arg(long, default_value = "http://127.0.0.1:8080")]
        endpoint: String,

        /// Screen refresh interval in seconds.
        #[arg(long, default_value_t = 1)]
        interval_secs: u64,
    },
}

/// Subcommands for the `models` CLI tool.
#[derive(Subcommand, Debug)]
pub enum ModelsAction {
    /// Initialize or update a configuration file with model stubs.
    Init {
        /// Target directory containing models (default: "./models", env: SNOOZE_MODELS_DIR).
        #[arg(default_value = "./models", env = "SNOOZE_MODELS_DIR")]
        dir: String,

        /// Custom configuration file path (defaults to models.toml in directory).
        #[arg(short, long)]
        file: Option<String>,
    },

    /// Scan directory and inspect discovered and configured models.
    Scan {
        /// Target directory containing models (default: "./models", env: SNOOZE_MODELS_DIR).
        #[arg(default_value = "./models", env = "SNOOZE_MODELS_DIR")]
        dir: String,

        /// Explicit configuration file path.
        #[arg(short, long)]
        file: Option<String>,
    },

    /// Download a model repository from Hugging Face Hub into the models directory.
    #[command(name = "download", alias = "hf-download")]
    Download {
        /// Hugging Face repository identifier (e.g. "openai-community/gpt2").
        repo: String,

        /// Target root models directory (default: "./models", env: SNOOZE_MODELS_DIR).
        #[arg(
            short = 'd',
            long = "models-dir",
            alias = "dir",
            env = "SNOOZE_MODELS_DIR",
            default_value = "./models"
        )]
        models_dir: String,

        /// Optional destination subfolder name within models directory (defaults to repository name).
        #[arg(short = 'n', long = "name", alias = "folder")]
        folder: Option<String>,

        /// Specific git revision, branch, or tag to download.
        #[arg(short = 'r', long = "revision")]
        revision: Option<String>,

        /// Glob pattern(s) selecting which files to download.
        #[arg(short = 'i', long = "include", value_delimiter = ',')]
        include: Option<Vec<String>>,

        /// Glob pattern(s) to exclude from download.
        #[arg(short = 'e', long = "exclude", value_delimiter = ',')]
        exclude: Option<Vec<String>>,

        /// Hugging Face authorization token (defaults to HF_TOKEN environment variable).
        #[arg(long, env = "HF_TOKEN")]
        token: Option<String>,

        /// Skip automatically generating or updating models.toml after download.
        #[arg(long)]
        no_init: bool,
    },
}

/// Executes the parsed CLI command.
///
/// # Errors
/// Returns error if server fails to start, or if client commands cannot connect.
pub async fn run_cli(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    match cli.command {
        Commands::Serve {
            host,
            port,
            models_dir,
            config,
            idle_timeout,
            max_batch_size,
            max_delay_ms,
            provider,
            max_loaded_models,
            demo,
            auth_enabled,
            api_keys,
            jwt_secret,
            protect_metrics,
            grpc,
            grpc_port,
            no_grpc,
            tls,
            tls_cert,
            tls_key,
            pin,
            streams_config,
        } => {
            let registry = ModelRegistry::new();
            if let Some(limit) = max_loaded_models {
                registry.set_max_loaded_models(Some(limit)).await;
            }

            if demo {
                tracing::info!("Registering default mock demo model 'demo-model'");
                let demo_config = ModelConfig {
                    id: "demo-model".to_string(),
                    onnx_path: "mock:demo".to_string(),
                    idle_timeout: Duration::from_secs(idle_timeout),
                    max_batch_size,
                    max_batch_delay: Duration::from_millis(max_delay_ms),
                    provider: None,
                    description: Some("Built-in mock demonstration model".to_string()),
                    min_residency: Duration::from_secs(10),
                    autoscale_queue_threshold: max_batch_size,
                    max_replicas: None,
                    routing_strategy: crate::routing::RoutingStrategyType::LeastQueue,
                    pinned: pin.iter().any(|p| p == "demo-model"),
                };
                registry.register(ModelController::new(demo_config)).await;
            }

            let resolved_dir = models_dir
                .or_else(|| std::env::var(crate::config::ENV_MODELS_DIR).ok())
                .or_else(|| std::env::var(ENV_MODELS_DIR_FALLBACK).ok());
            let mut file_auth: Option<AuthConfig> = None;
            let mut configured_streams: Vec<crate::config::StreamBindingConfig> = Vec::new();

            if let Some(ref dir_path) = resolved_dir {
                let (auth, dir_streams) = scan_and_register_models(
                    &registry,
                    dir_path,
                    Duration::from_secs(idle_timeout),
                    max_batch_size,
                    Duration::from_millis(max_delay_ms),
                    provider.clone(),
                    config.as_deref(),
                )
                .await?;
                file_auth = auth;
                configured_streams.extend(dir_streams);
            }

            if let Some(ref sc_path) = streams_config {
                let path = Path::new(sc_path);
                if let Ok(content) = tokio::fs::read_to_string(path).await {
                    if let Ok(cfg) = ModelDirectoryConfig::parse_content(&content, path) {
                        configured_streams.extend(cfg.streams);
                    }
                }
            }

            for pinned_id in &pin {
                if let Err(e) = registry.pin_model(pinned_id).await {
                    tracing::warn!(
                        model_id = %pinned_id,
                        error = %e,
                        "Failed to pin model specified via --pin"
                    );
                }
            }

            // Merge file-based auth configuration with CLI arguments / environment variables
            let mut effective_auth = file_auth.unwrap_or_default();
            if auth_enabled {
                effective_auth.enabled = true;
            }
            if protect_metrics {
                effective_auth.protect_metrics = true;
            }
            for entry in api_keys {
                let (role, key_str, name) = if let Some((role_str, key_str)) = entry.split_once(':')
                {
                    let role = Role::parse_str(role_str);
                    (role, key_str.trim(), format!("cli-{role_str}"))
                } else {
                    (Role::Admin, entry.trim(), "cli-admin".to_string())
                };
                if !key_str.is_empty() {
                    effective_auth.api_keys.push(ApiKeyConfig {
                        key: Some(key_str.to_string()),
                        key_env: None,
                        role,
                        name: Some(name),
                    });
                    effective_auth.enabled = true;
                }
            }
            if let Some(secret) = jwt_secret {
                effective_auth.jwt = Some(JwtConfig {
                    secret: Some(secret),
                    secret_env: Some("SNOOZE_JWT_SECRET".to_string()),
                    algorithm: JwtAlgorithm::HS256,
                    issuer: None,
                    audience: None,
                    leeway_secs: 60,
                });
                effective_auth.enabled = true;
            }

            // Also load any environment-based API keys or JWT secrets
            effective_auth.load_from_env();

            let auth_config = if effective_auth.enabled
                || !effective_auth.api_keys.is_empty()
                || effective_auth.jwt.is_some()
            {
                effective_auth.enabled = true;
                Some(effective_auth)
            } else {
                None
            };

            let parse_bool = |val: &str| match val.trim().to_ascii_lowercase().as_str() {
                "true" | "1" | "yes" | "on" => Some(true),
                "false" | "0" | "no" | "off" => Some(false),
                _ => None,
            };

            let effective_grpc_enabled = if no_grpc {
                false
            } else if let Some(enabled) = grpc {
                enabled
            } else if let Ok(val) = std::env::var("GRPC")
                .or_else(|_| std::env::var("grpc"))
                .or_else(|_| std::env::var("SNOOZE_GRPC_ENABLED"))
                .or_else(|_| std::env::var("GRPC_ENABLED"))
            {
                parse_bool(&val).unwrap_or(true)
            } else {
                true
            };

            let effective_grpc_port = grpc_port
                .or_else(|| {
                    std::env::var("GRPC_PORT")
                        .or_else(|_| std::env::var("grpc_port"))
                        .ok()
                        .and_then(|val| val.trim().parse::<u16>().ok())
                })
                .unwrap_or(50051);

            let effective_tls_enabled = if let Some(enabled) = tls {
                enabled
            } else if let Ok(val) = std::env::var("SNOOZE_TLS")
                .or_else(|_| std::env::var("TLS"))
                .or_else(|_| std::env::var("SNOOZE_TLS_ENABLED"))
                .or_else(|_| std::env::var("TLS_ENABLED"))
            {
                parse_bool(&val).unwrap_or(false)
            } else {
                false
            };

            let effective_cert_path = tls_cert.or_else(|| {
                std::env::var("SNOOZE_TLS_CERT")
                    .or_else(|_| std::env::var("TLS_CERT"))
                    .ok()
                    .map(PathBuf::from)
            });

            let effective_key_path = tls_key.or_else(|| {
                std::env::var("SNOOZE_TLS_KEY")
                    .or_else(|_| std::env::var("TLS_KEY"))
                    .ok()
                    .map(PathBuf::from)
            });

            let tls_config = match (effective_cert_path, effective_key_path) {
                (Some(cert_path), Some(key_path)) => {
                    if tls == Some(false) {
                        None
                    } else {
                        Some(TlsConfig::new(cert_path, key_path))
                    }
                }
                (Some(_), None) => {
                    return Err(
                        "TLS certificate path was provided (--tls-cert / SNOOZE_TLS_CERT), but private key path (--tls-key / SNOOZE_TLS_KEY) is missing.".into(),
                    );
                }
                (None, Some(_)) => {
                    return Err(
                        "TLS private key path was provided (--tls-key / SNOOZE_TLS_KEY), but certificate path (--tls-cert / SNOOZE_TLS_CERT) is missing.".into(),
                    );
                }
                (None, None) => {
                    if effective_tls_enabled {
                        return Err(
                            "TLS was explicitly enabled (--tls or SNOOZE_TLS=true) but neither certificate (--tls-cert / SNOOZE_TLS_CERT) nor private key (--tls-key / SNOOZE_TLS_KEY) was provided.".into(),
                        );
                    }
                    None
                }
            };

            let server_config = ServerConfig {
                host,
                port,
                grpc_port: effective_grpc_port,
                grpc_enabled: effective_grpc_enabled,
                sweeper_interval: Duration::from_secs(5),
                auth: auth_config,
                tls: tls_config,
                models_dir: resolved_dir
                    .map(PathBuf::from)
                    .unwrap_or_else(|| PathBuf::from(crate::config::default_models_dir())),
                streams: configured_streams,
            };

            run_server(server_config, registry).await?;
            Ok(())
        }
        Commands::Download {
            repo,
            models_dir,
            folder,
            revision,
            include,
            exclude,
            token,
            no_init,
        } => {
            let effective_models_dir = if models_dir == "./models" {
                default_models_dir()
            } else {
                models_dir
            };
            let options = HfDownloadOptions {
                repo_id: repo,
                models_dir: PathBuf::from(effective_models_dir),
                folder,
                revision,
                include,
                exclude,
                token,
                no_init,
            };
            run_download(options).await?;
            Ok(())
        }
        Commands::Models { action } => match action {
            ModelsAction::Init { dir, file } => {
                let effective_dir = if dir == "./models" {
                    default_models_dir()
                } else {
                    dir
                };
                run_models_init(&effective_dir, file.as_deref()).await?;
                Ok(())
            }
            ModelsAction::Scan { dir, file } => {
                let effective_dir = if dir == "./models" {
                    default_models_dir()
                } else {
                    dir
                };
                run_models_scan(&effective_dir, file.as_deref()).await?;
                Ok(())
            }
            ModelsAction::Download {
                repo,
                models_dir,
                folder,
                revision,
                include,
                exclude,
                token,
                no_init,
            } => {
                let effective_dir = if models_dir == "./models" {
                    default_models_dir()
                } else {
                    models_dir
                };
                let options = HfDownloadOptions {
                    repo_id: repo,
                    models_dir: PathBuf::from(effective_dir),
                    folder,
                    revision,
                    include,
                    exclude,
                    token,
                    no_init,
                };
                run_download(options).await?;
                Ok(())
            }
        },
        Commands::Init { dir, file } => {
            let effective_dir = if dir == "./models" {
                default_models_dir()
            } else {
                dir
            };
            run_models_init(&effective_dir, file.as_deref()).await?;
            Ok(())
        }
        Commands::Status { endpoint } => {
            run_status(&endpoint).await?;
            Ok(())
        }
        Commands::Load {
            endpoint,
            model_id,
            pin,
        } => {
            if pin {
                run_pin(&endpoint, &model_id).await?;
            } else {
                run_load(&endpoint, &model_id).await?;
            }
            Ok(())
        }
        Commands::Unload { endpoint, model_id } => {
            run_unload(&endpoint, &model_id).await?;
            Ok(())
        }
        Commands::Pin { endpoint, model_id } => {
            run_pin(&endpoint, &model_id).await?;
            Ok(())
        }
        Commands::Unpin { endpoint, model_id } => {
            run_unpin(&endpoint, &model_id).await?;
            Ok(())
        }
        Commands::Monitor {
            endpoint,
            interval_secs,
        } => {
            run_monitor(&endpoint, interval_secs).await?;
            Ok(())
        }
    }
}

macro_rules! with_env_api_key {
    ($req:expr) => {{
        let mut r = $req;
        if let Ok(key) = std::env::var("SNOOZE_API_KEY") {
            let trimmed = key.trim();
            if !trimmed.is_empty() {
                r = r.header("x-api-key", trimmed);
            }
        }
        r
    }};
}

/// Fetches list of registered models from the SnoozeML daemon.
///
/// # Errors
/// Returns error message string if connection or JSON decoding fails.
pub(crate) fn fetch_models(endpoint: &str) -> Result<Vec<crate::types::ModelMetadata>, String> {
    let url = format!("{}/v1/models", endpoint.trim_end_matches('/'));
    let mut resp = with_env_api_key!(ureq::get(&url))
        .call()
        .map_err(|e| format!("Failed to connect to {url}: {e}"))?;
    let models: Vec<crate::types::ModelMetadata> = resp
        .body_mut()
        .read_json()
        .map_err(|e| format!("Failed to parse JSON response from {url}: {e}"))?;
    Ok(models)
}

/// Fetches raw Prometheus metrics from the SnoozeML daemon.
///
/// # Errors
/// Returns error message string if connection or reading fails.
pub(crate) fn fetch_metrics(endpoint: &str) -> Result<String, String> {
    let url = format!("{}/metrics", endpoint.trim_end_matches('/'));
    let mut resp = with_env_api_key!(ureq::get(&url))
        .call()
        .map_err(|e| format!("Failed to connect to {url}: {e}"))?;
    let metrics = resp
        .body_mut()
        .read_to_string()
        .map_err(|e| format!("Failed to read metrics response from {url}: {e}"))?;
    Ok(metrics)
}

/// Queries and prints server status and registered models.
///
/// # Errors
/// Returns error if server is unreachable or response cannot be parsed.
async fn run_status(endpoint: &str) -> Result<(), Box<dyn std::error::Error>> {
    let ep = endpoint.to_string();
    let models = tokio::task::spawn_blocking(move || fetch_models(&ep))
        .await
        .map_err(|e| format!("Internal task failure: {e}"))??;

    println!("========================================================================");
    println!("  SnoozeML Inference Daemon Status | Host: {endpoint}");
    println!("========================================================================\n");

    if models.is_empty() {
        println!("  No models currently registered on server.\n");
    } else {
        println!("Registered Models:");
        println!(
            "  {:<22} {:<10} {:<8} {:<12} {:<12} {:<16} {:<24}",
            "MODEL ID",
            "STATE",
            "PINNED",
            "MAX BATCH",
            "DELAY (ms)",
            "LAST ACTIVE (s)",
            "DESCRIPTION"
        );
        println!("  -----------------------------------------------------------------------------------------------------------");
        for m in &models {
            let state_str = format!("{:?}", m.state);
            let pin_str = if m.pinned { "yes" } else { "no" };
            let desc_str = m.description.as_deref().unwrap_or("-");
            println!(
                "  {:<22} {:<10} {:<8} {:<12} {:<12} {:<16} {:<24}",
                m.id,
                state_str,
                pin_str,
                m.max_batch_size,
                m.max_batch_delay_ms,
                m.last_accessed_secs_ago,
                desc_str
            );
        }
        println!();
    }
    println!("========================================================================");
    Ok(())
}

/// Sends a request to preload a model into memory.
///
/// # Errors
/// Returns error if model is not found, server fails, or connection error occurs.
async fn run_load(endpoint: &str, model_id: &str) -> Result<(), Box<dyn std::error::Error>> {
    let ep = endpoint.to_string();
    let mid = model_id.to_string();
    let res = tokio::task::spawn_blocking(move || {
        let url = format!("{}/v1/models/{mid}/load", ep.trim_end_matches('/'));
        match with_env_api_key!(ureq::post(&url)).send_empty() {
            Ok(_) => {
                println!("Successfully loaded model '{mid}' into memory at {ep}.");
                Ok(())
            }
            Err(ureq::Error::StatusCode(404)) => {
                Err(format!("Model '{mid}' was not found on server at {ep}"))
            }
            Err(ureq::Error::StatusCode(code)) => Err(format!(
                "Server at {ep} returned error HTTP status {code} loading model '{mid}'"
            )),
            Err(e) => Err(format!("Failed to connect to {url}: {e}")),
        }
    })
    .await
    .map_err(|e| format!("Internal task failure: {e}"))?;

    res.map_err(Into::into)
}

/// Sends a request to deallocate a model from memory (scale-to-zero).
///
/// # Errors
/// Returns error if model is not found, server fails, or connection error occurs.
async fn run_unload(endpoint: &str, model_id: &str) -> Result<(), Box<dyn std::error::Error>> {
    let ep = endpoint.to_string();
    let mid = model_id.to_string();
    let res = tokio::task::spawn_blocking(move || {
        let url = format!("{}/v1/models/{mid}/unload", ep.trim_end_matches('/'));
        match with_env_api_key!(ureq::post(&url)).send_empty() {
            Ok(_) => {
                println!("Successfully unloaded model '{mid}' from memory at {ep}.");
                Ok(())
            }
            Err(ureq::Error::StatusCode(404)) => {
                Err(format!("Model '{mid}' was not found on server at {ep}"))
            }
            Err(ureq::Error::StatusCode(code)) => Err(format!(
                "Server at {ep} returned error HTTP status {code} unloading model '{mid}'"
            )),
            Err(e) => Err(format!("Failed to connect to {url}: {e}")),
        }
    })
    .await
    .map_err(|e| format!("Internal task failure: {e}"))?;

    res.map_err(Into::into)
}

/// Sends a request to pin a model in memory so it is eagerly loaded and never automatically deallocated.
///
/// # Errors
/// Returns error if model is not found, server fails, or connection error occurs.
async fn run_pin(endpoint: &str, model_id: &str) -> Result<(), Box<dyn std::error::Error>> {
    let ep = endpoint.to_string();
    let mid = model_id.to_string();
    let res = tokio::task::spawn_blocking(move || {
        let url = format!("{}/v1/models/{mid}/pin", ep.trim_end_matches('/'));
        match with_env_api_key!(ureq::post(&url)).send_empty() {
            Ok(_) => {
                println!("Successfully pinned model '{mid}' in memory at {ep}.");
                Ok(())
            }
            Err(ureq::Error::StatusCode(404)) => {
                Err(format!("Model '{mid}' was not found on server at {ep}"))
            }
            Err(ureq::Error::StatusCode(code)) => Err(format!(
                "Server at {ep} returned error HTTP status {code} pinning model '{mid}'"
            )),
            Err(e) => Err(format!("Failed to connect to {url}: {e}")),
        }
    })
    .await
    .map_err(|e| format!("Internal task failure: {e}"))?;

    res.map_err(Into::into)
}

/// Sends a request to unpin a model so it returns to normal scale-to-zero and LRU lifecycle management.
///
/// # Errors
/// Returns error if model is not found, server fails, or connection error occurs.
async fn run_unpin(endpoint: &str, model_id: &str) -> Result<(), Box<dyn std::error::Error>> {
    let ep = endpoint.to_string();
    let mid = model_id.to_string();
    let res = tokio::task::spawn_blocking(move || {
        let url = format!("{}/v1/models/{mid}/unpin", ep.trim_end_matches('/'));
        match with_env_api_key!(ureq::post(&url)).send_empty() {
            Ok(_) => {
                println!("Successfully unpinned model '{mid}' at {ep}.");
                Ok(())
            }
            Err(ureq::Error::StatusCode(404)) => {
                Err(format!("Model '{mid}' was not found on server at {ep}"))
            }
            Err(ureq::Error::StatusCode(code)) => Err(format!(
                "Server at {ep} returned error HTTP status {code} unpinning model '{mid}'"
            )),
            Err(e) => Err(format!("Failed to connect to {url}: {e}")),
        }
    })
    .await
    .map_err(|e| format!("Internal task failure: {e}"))?;

    res.map_err(Into::into)
}

/// Runs a live interactive terminal dashboard for monitoring models and metrics.
///
/// # Errors
/// Returns error if terminal setup fails or monitor encounters unrecoverable errors.
async fn run_monitor(endpoint: &str, interval_secs: u64) -> Result<(), Box<dyn std::error::Error>> {
    tui::run_tui(endpoint, interval_secs).await
}

/// Initializes or updates a model directory configuration file.
///
/// # Errors
/// Returns error if directory cannot be read or written.
pub async fn run_models_init(
    dir: &str,
    file: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let dir_path = Path::new(dir);
    if !dir_path.exists() {
        tokio::fs::create_dir_all(dir_path).await?;
        println!("Created models directory at: {}", dir_path.display());
    }

    let custom_file = file.map(Path::new);
    let (target_path, added_count) =
        ModelDirectoryConfig::init_or_update_dir(dir_path, custom_file).await?;

    println!("========================================================================");
    println!("  SnoozeML Model Configuration Utility");
    println!("========================================================================\n");
    println!("  Directory:          {}", dir_path.display());
    println!("  Configuration file: {}", target_path.display());
    if added_count > 0 {
        println!(
            "  Status: Configured {} model(s) with stub description '{}'.",
            added_count, STUB_DESCRIPTION
        );
        println!("\n  Per-model batching, devices (CPU/GPU), and descriptions can now be");
        println!("  customized directly in the configuration file.");
    } else {
        println!("  Status: Configuration file is up to date (no new unconfigured models found).");
    }
    println!("\n========================================================================");
    Ok(())
}

/// Scans a model directory and displays discovered models and their configurations.
///
/// # Errors
/// Returns error if directory scan fails.
pub async fn run_models_scan(
    dir: &str,
    file: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let dir_path = Path::new(dir);
    if !dir_path.exists() {
        println!("Directory '{}' does not exist.", dir);
        return Ok(());
    }

    let discovered = ModelDirectoryConfig::discover_models_in_dir(dir_path).await?;
    let custom_file = file.map(Path::new);
    let loaded_config = ModelDirectoryConfig::load_fault_tolerant(dir_path, custom_file).await;

    println!("========================================================================");
    println!(
        "  SnoozeML Model Directory Scan | Directory: {}",
        dir_path.display()
    );
    println!("========================================================================\n");

    match &loaded_config {
        Some(cfg) => {
            let cfg_path = custom_file
                .map(|p| p.to_path_buf())
                .or_else(|| ModelDirectoryConfig::find_config_file(dir_path))
                .unwrap_or_else(|| PathBuf::from("configured"));
            println!("  Active Config File: {}", cfg_path.display());
            println!(
                "  Defaults: batch_size={}, delay={}ms, idle_timeout={}s, device={}\n",
                cfg.defaults
                    .max_batch_size
                    .unwrap_or(DEFAULT_MAX_BATCH_SIZE),
                cfg.defaults
                    .max_batch_delay_ms
                    .unwrap_or(DEFAULT_MAX_BATCH_DELAY_MS),
                cfg.defaults
                    .idle_timeout_secs
                    .unwrap_or(DEFAULT_IDLE_TIMEOUT_SECS),
                cfg.defaults.device.as_deref().unwrap_or(DEFAULT_DEVICE)
            );

            let built_configs = cfg.build_model_configs(dir_path, &discovered);
            if built_configs.is_empty() {
                println!("  No models configured or discovered.\n");
            } else {
                println!(
                    "  Configured & Discovered Models ({}):",
                    built_configs.len()
                );
                println!(
                    "  {:<22} {:<10} {:<8} {:<12} {:<12} {:<14} {:<24}",
                    "MODEL ID",
                    "DEVICE",
                    "PINNED",
                    "MAX BATCH",
                    "DELAY (ms)",
                    "TIMEOUT (s)",
                    "DESCRIPTION"
                );
                println!("  -----------------------------------------------------------------------------------------------------------");
                for m in &built_configs {
                    let dev_str = m.provider.as_deref().unwrap_or("cpu");
                    let pin_str = if m.pinned { "yes" } else { "no" };
                    let desc_str = m.description.as_deref().unwrap_or("-");
                    println!(
                        "  {:<22} {:<10} {:<8} {:<12} {:<12} {:<14} {:<24}",
                        m.id,
                        dev_str,
                        pin_str,
                        m.max_batch_size,
                        m.max_batch_delay.as_millis(),
                        m.idle_timeout.as_secs(),
                        desc_str
                    );
                }
                println!();
            }
        }
        None => {
            println!("  Active Config File: (none - using system defaults)\n");
            if discovered.is_empty() {
                println!("  No models discovered in directory.\n");
            } else {
                println!("  Discovered Models On Disk ({}):", discovered.len());
                println!("  {:<24} {:<40}", "MODEL ID", "RELATIVE PATH");
                println!("  --------------------------------------------------------------------");
                for d in &discovered {
                    println!("  {:<24} {:<40}", d.id, d.relative_path);
                }
                println!("\n  Run 'snooze models init' or 'snooze init' to generate a configuration file.");
            }
        }
    }
    println!("========================================================================");
    Ok(())
}

/// Scans a directory for model files and registers each model into the registry,
/// respecting any discovered configuration file and per-model settings.
async fn scan_and_register_models(
    registry: &ModelRegistry,
    dir_path: &str,
    idle_timeout: Duration,
    max_batch_size: usize,
    max_delay: Duration,
    provider: Option<String>,
    explicit_config: Option<&str>,
) -> Result<(Option<AuthConfig>, Vec<crate::config::StreamBindingConfig>), std::io::Error> {
    let path = Path::new(dir_path);
    if !path.is_dir() {
        tracing::warn!(dir = %dir_path, "Provided models directory does not exist or is not a directory");
        return Ok((None, Vec::new()));
    }

    let discovered = ModelDirectoryConfig::discover_models_in_dir(path).await?;
    let explicit_path = explicit_config.map(Path::new);
    let loaded_config = ModelDirectoryConfig::load_fault_tolerant(path, explicit_path).await;

    let (model_configs, config_max_models, file_auth, streams) = match loaded_config {
        Some(mut config) => {
            if let Some(p) = provider {
                config.defaults.device = Some(p);
            }
            let max_m = config.defaults.max_loaded_models;
            let auth = config.auth.take();
            let streams = std::mem::take(&mut config.streams);
            (
                config.build_model_configs(path, &discovered),
                max_m,
                auth,
                streams,
            )
        }
        None => {
            let mut fallback_config = ModelDirectoryConfig::default();
            fallback_config.defaults.idle_timeout_secs = Some(idle_timeout.as_secs());
            fallback_config.defaults.max_batch_size = Some(max_batch_size);
            fallback_config.defaults.max_batch_delay_ms = u64::try_from(max_delay.as_millis()).ok();
            if let Some(p) = provider {
                fallback_config.defaults.device = Some(p);
            }
            let max_m = fallback_config.defaults.max_loaded_models;
            (
                fallback_config.build_model_configs(path, &discovered),
                max_m,
                None,
                Vec::new(),
            )
        }
    };

    if registry.max_loaded_models().await.is_none() {
        if let Some(max_m) = config_max_models {
            registry.set_max_loaded_models(Some(max_m)).await;
        }
    }

    for config in model_configs {
        registry.register(ModelController::new(config)).await;
    }

    Ok((file_auth, streams))
}

// Colocated inline tests for private internal `scan_and_register_models` helper,
// which cannot be exercised via the public CLI runner API without starting a server daemon.
#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
    use super::*;

    #[tokio::test]
    async fn test_scan_and_register_models() {
        let mut temp_dir = std::env::temp_dir();
        let unique_id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        temp_dir.push(format!("snooze_models_test_{unique_id}"));
        tokio::fs::create_dir_all(&temp_dir).await.unwrap();

        // Create dummy .onnx files and a non-onnx file
        let model1 = temp_dir.join("model_one.onnx");
        let model2 = temp_dir.join("model_two.onnx");
        let non_model = temp_dir.join("readme.txt");

        tokio::fs::write(&model1, b"dummy content").await.unwrap();
        tokio::fs::write(&model2, b"dummy content").await.unwrap();
        tokio::fs::write(&non_model, b"hello").await.unwrap();

        let registry = ModelRegistry::new();
        scan_and_register_models(
            &registry,
            &temp_dir.to_string_lossy(),
            Duration::from_secs(60),
            8,
            Duration::from_millis(10),
            None,
            None,
        )
        .await
        .unwrap();

        let list = registry.list_metadata().await;
        assert_eq!(list.len(), 2);
        let mut ids: Vec<String> = list.into_iter().map(|m| m.id).collect();
        ids.sort();
        assert_eq!(ids, vec!["model_one", "model_two"]);

        // Clean up
        let _ = tokio::fs::remove_dir_all(&temp_dir).await;
    }

    #[tokio::test]
    async fn test_scan_and_register_models_with_config() {
        let mut temp_dir = std::env::temp_dir();
        let unique_id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        temp_dir.push(format!("snooze_models_cfg_test_{unique_id}"));
        tokio::fs::create_dir_all(&temp_dir).await.unwrap();

        let model1 = temp_dir.join("model_one.onnx");
        tokio::fs::write(&model1, b"dummy content").await.unwrap();

        let toml_content = r#"
[defaults]
idle_timeout_secs = 200
max_batch_size = 4
max_batch_delay_ms = 15
device = "cpu"

[[models]]
id = "custom_id"
path = "model_one.onnx"
description = "My custom model description"
device = "cuda"
max_batch_size = 8
"#;
        tokio::fs::write(temp_dir.join("models.toml"), toml_content)
            .await
            .unwrap();

        let registry = ModelRegistry::new();
        scan_and_register_models(
            &registry,
            &temp_dir.to_string_lossy(),
            Duration::from_secs(60),
            16,
            Duration::from_millis(10),
            None,
            None,
        )
        .await
        .unwrap();

        let list = registry.list_metadata().await;
        assert_eq!(list.len(), 1);
        let meta = list.first().unwrap();
        assert_eq!(meta.id, "custom_id");
        assert_eq!(meta.max_batch_size, 8);
        assert_eq!(
            meta.description.as_deref(),
            Some("My custom model description")
        );

        let _ = tokio::fs::remove_dir_all(&temp_dir).await;
    }

    #[tokio::test]
    async fn test_scan_and_register_models_non_existent_directory() {
        let registry = ModelRegistry::new();
        let res = scan_and_register_models(
            &registry,
            "/tmp/snooze_definitely_non_existent_dir_12345",
            Duration::from_secs(60),
            8,
            Duration::from_millis(10),
            None,
            None,
        )
        .await;

        assert!(res.is_ok());
        assert!(registry.list_metadata().await.is_empty());
    }
}
