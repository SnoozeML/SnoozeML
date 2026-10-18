//! Integration tests for SnoozeML CLI commands and argument parsing.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use clap::Parser;
use snooze_ml::cli::{run_cli, Cli, Commands};
use snooze_ml::model::{ModelConfig, ModelController};
use snooze_ml::registry::ModelRegistry;
use snooze_ml::server::http::{create_router, AppState};
use snooze_ml::server::metrics::ServerMetrics;
use std::time::Duration;

static CLI_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn test_cli_parse_serve_defaults() {
    let _guard = CLI_ENV_LOCK.lock().unwrap();
    std::env::remove_var("SNOOZE_MODELS_DIR");
    std::env::remove_var("MODELS_DIR");

    let cli = Cli::try_parse_from(["snooze", "serve"]).unwrap();
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
            demo,
            max_loaded_models,
            ..
        } => {
            assert_eq!(host, "0.0.0.0");
            assert_eq!(port, 8080);
            assert_eq!(models_dir, None);
            assert_eq!(config, None);
            assert_eq!(idle_timeout, 300);
            assert_eq!(max_batch_size, 16);
            assert_eq!(max_delay_ms, 10);
            assert_eq!(provider, None);
            assert_eq!(max_loaded_models, None);
            assert!(!demo);
        }
        _ => panic!("Expected Serve command"),
    }
}

#[test]
fn test_cli_parse_serve_custom_flags() {
    let cli = Cli::try_parse_from([
        "snooze",
        "serve",
        "--host",
        "127.0.0.1",
        "--port",
        "9090",
        "--models-dir",
        "/tmp/models",
        "--config",
        "/tmp/models/custom.toml",
        "--idle-timeout",
        "60",
        "--max-batch-size",
        "32",
        "--max-delay-ms",
        "5",
        "--provider",
        "cuda",
        "--max-loaded-models",
        "4",
    ])
    .unwrap();

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
            demo,
            max_loaded_models,
            ..
        } => {
            assert_eq!(host, "127.0.0.1");
            assert_eq!(port, 9090);
            assert_eq!(models_dir, Some("/tmp/models".to_string()));
            assert_eq!(config, Some("/tmp/models/custom.toml".to_string()));
            assert_eq!(idle_timeout, 60);
            assert_eq!(max_batch_size, 32);
            assert_eq!(max_delay_ms, 5);
            assert_eq!(provider, Some("cuda".to_string()));
            assert_eq!(max_loaded_models, Some(4));
            assert!(!demo);
        }
        _ => panic!("Expected Serve command"),
    }
}

#[test]
fn test_cli_parse_serve_demo_flag() {
    let cli = Cli::try_parse_from(["snooze", "serve", "--demo"]).unwrap();
    match cli.command {
        Commands::Serve { demo, .. } => {
            assert!(demo);
        }
        _ => panic!("Expected Serve command"),
    }
}

#[test]
fn test_cli_parse_status() {
    let cli =
        Cli::try_parse_from(["snooze", "status", "--endpoint", "http://10.0.0.1:8080"]).unwrap();
    match cli.command {
        Commands::Status { endpoint } => {
            assert_eq!(endpoint, "http://10.0.0.1:8080");
        }
        _ => panic!("Expected Status command"),
    }
}

#[test]
fn test_cli_parse_load_and_unload() {
    let cli_load = Cli::try_parse_from([
        "snooze",
        "load",
        "--endpoint",
        "http://127.0.0.1:8080",
        "--model-id",
        "bert",
    ])
    .unwrap();

    match cli_load.command {
        Commands::Load {
            endpoint,
            model_id,
            pin,
        } => {
            assert_eq!(endpoint, "http://127.0.0.1:8080");
            assert_eq!(model_id, "bert");
            assert!(!pin);
        }
        _ => panic!("Expected Load command"),
    }

    let cli_load_pin = Cli::try_parse_from([
        "snooze",
        "load",
        "--endpoint",
        "http://127.0.0.1:8080",
        "--model-id",
        "bert",
        "--pin",
    ])
    .unwrap();
    match cli_load_pin.command {
        Commands::Load {
            endpoint,
            model_id,
            pin,
        } => {
            assert_eq!(endpoint, "http://127.0.0.1:8080");
            assert_eq!(model_id, "bert");
            assert!(pin);
        }
        _ => panic!("Expected Load command"),
    }

    let cli_pin = Cli::try_parse_from([
        "snooze",
        "pin",
        "--endpoint",
        "http://127.0.0.1:8080",
        "--model-id",
        "bert",
    ])
    .unwrap();
    match cli_pin.command {
        Commands::Pin { endpoint, model_id } => {
            assert_eq!(endpoint, "http://127.0.0.1:8080");
            assert_eq!(model_id, "bert");
        }
        _ => panic!("Expected Pin command"),
    }

    let cli_unpin = Cli::try_parse_from([
        "snooze",
        "unpin",
        "--endpoint",
        "http://127.0.0.1:8080",
        "--model-id",
        "bert",
    ])
    .unwrap();
    match cli_unpin.command {
        Commands::Unpin { endpoint, model_id } => {
            assert_eq!(endpoint, "http://127.0.0.1:8080");
            assert_eq!(model_id, "bert");
        }
        _ => panic!("Expected Unpin command"),
    }

    let cli_unload = Cli::try_parse_from([
        "snooze",
        "unload",
        "--endpoint",
        "http://127.0.0.1:8080",
        "--model-id",
        "bert",
    ])
    .unwrap();

    match cli_unload.command {
        Commands::Unload { endpoint, model_id } => {
            assert_eq!(endpoint, "http://127.0.0.1:8080");
            assert_eq!(model_id, "bert");
        }
        _ => panic!("Expected Unload command"),
    }
}

#[test]
fn test_cli_parse_monitor() {
    let cli = Cli::try_parse_from([
        "snooze",
        "monitor",
        "--endpoint",
        "http://127.0.0.1:8080",
        "--interval-secs",
        "2",
    ])
    .unwrap();

    match cli.command {
        Commands::Monitor {
            endpoint,
            interval_secs,
        } => {
            assert_eq!(endpoint, "http://127.0.0.1:8080");
            assert_eq!(interval_secs, 2);
        }
        _ => panic!("Expected Monitor command"),
    }
}

#[test]
fn test_cli_parse_errors() {
    let err1 = Cli::try_parse_from(["snooze", "nonexistent"]);
    assert!(err1.is_err());

    let err2 = Cli::try_parse_from(["snooze", "load"]);
    assert!(err2.is_err());

    let err3 = Cli::try_parse_from(["snooze", "serve", "--port", "not_a_number"]);
    assert!(err3.is_err());
}

#[tokio::test]
async fn test_cli_client_commands_e2e() {
    let registry = ModelRegistry::new();
    let metrics = ServerMetrics::new_shared();

    let config = ModelConfig {
        id: "demo-model".to_string(),
        onnx_path: "mock:demo".to_string(),
        idle_timeout: Duration::from_secs(60),
        max_batch_size: 4,
        max_batch_delay: Duration::from_millis(10),
        provider: None,
        description: None,
        min_residency: Duration::from_millis(0),
        ..Default::default()
    };
    let ctrl = ModelController::new(config);
    registry.register(ctrl.clone()).await;

    let app = create_router(AppState::new(registry.clone(), metrics.clone()));

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let endpoint = format!("http://{addr}");

    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });

    // 1. snooze status
    let cli_status = Cli::try_parse_from(["snooze", "status", "--endpoint", &endpoint]).unwrap();
    assert!(run_cli(cli_status).await.is_ok());

    // 2. snooze load (existing model)
    let cli_load_ok = Cli::try_parse_from([
        "snooze",
        "load",
        "--endpoint",
        &endpoint,
        "--model-id",
        "demo-model",
    ])
    .unwrap();
    assert!(run_cli(cli_load_ok).await.is_ok());

    // 3. snooze load (nonexistent model -> error)
    let cli_load_err = Cli::try_parse_from([
        "snooze",
        "load",
        "--endpoint",
        &endpoint,
        "--model-id",
        "nonexistent-model",
    ])
    .unwrap();
    let err = run_cli(cli_load_err).await;
    assert!(err.is_err());
    assert!(err.unwrap_err().to_string().contains("not found"));

    // 4. snooze pin & unpin
    let cli_pin_ok = Cli::try_parse_from([
        "snooze",
        "pin",
        "--endpoint",
        &endpoint,
        "--model-id",
        "demo-model",
    ])
    .unwrap();
    assert!(run_cli(cli_pin_ok).await.is_ok());
    assert!(ctrl.is_pinned());

    let cli_unpin_ok = Cli::try_parse_from([
        "snooze",
        "unpin",
        "--endpoint",
        &endpoint,
        "--model-id",
        "demo-model",
    ])
    .unwrap();
    assert!(run_cli(cli_unpin_ok).await.is_ok());
    assert!(!ctrl.is_pinned());

    let cli_pin_err = Cli::try_parse_from([
        "snooze",
        "pin",
        "--endpoint",
        &endpoint,
        "--model-id",
        "nonexistent-model",
    ])
    .unwrap();
    assert!(run_cli(cli_pin_err).await.is_err());

    let cli_unpin_err = Cli::try_parse_from([
        "snooze",
        "unpin",
        "--endpoint",
        &endpoint,
        "--model-id",
        "nonexistent-model",
    ])
    .unwrap();
    assert!(run_cli(cli_unpin_err).await.is_err());

    // 5. snooze unload (existing model)
    let cli_unload_ok = Cli::try_parse_from([
        "snooze",
        "unload",
        "--endpoint",
        &endpoint,
        "--model-id",
        "demo-model",
    ])
    .unwrap();
    assert!(run_cli(cli_unload_ok).await.is_ok());

    // 6. snooze unload (nonexistent model -> error)
    let cli_unload_err = Cli::try_parse_from([
        "snooze",
        "unload",
        "--endpoint",
        &endpoint,
        "--model-id",
        "nonexistent-model",
    ])
    .unwrap();
    let err = run_cli(cli_unload_err).await;
    assert!(err.is_err());
    assert!(err.unwrap_err().to_string().contains("not found"));

    // 7. snooze status against unreachable endpoint
    let cli_status_unreachable =
        Cli::try_parse_from(["snooze", "status", "--endpoint", "http://127.0.0.1:1"]).unwrap();
    assert!(run_cli(cli_status_unreachable).await.is_err());
}

#[test]
fn test_cli_parse_models_subcommands() {
    let cli_init = Cli::try_parse_from(["snooze", "models", "init", "/path/to/models"]).unwrap();
    match cli_init.command {
        Commands::Models {
            action: snooze_ml::cli::ModelsAction::Init { dir, file },
        } => {
            assert_eq!(dir, "/path/to/models");
            assert_eq!(file, None);
        }
        _ => panic!("Expected Models Init command"),
    }

    let cli_init_file = Cli::try_parse_from([
        "snooze",
        "models",
        "init",
        "/path/to/models",
        "--file",
        "/custom/path.toml",
    ])
    .unwrap();
    match cli_init_file.command {
        Commands::Models {
            action: snooze_ml::cli::ModelsAction::Init { dir, file },
        } => {
            assert_eq!(dir, "/path/to/models");
            assert_eq!(file, Some("/custom/path.toml".to_string()));
        }
        _ => panic!("Expected Models Init command with file"),
    }

    let cli_scan = Cli::try_parse_from(["snooze", "models", "scan", "/path/to/models"]).unwrap();
    match cli_scan.command {
        Commands::Models {
            action: snooze_ml::cli::ModelsAction::Scan { dir, file },
        } => {
            assert_eq!(dir, "/path/to/models");
            assert_eq!(file, None);
        }
        _ => panic!("Expected Models Scan command"),
    }

    let cli_init_alias = Cli::try_parse_from(["snooze", "init", "my_models"]).unwrap();
    match cli_init_alias.command {
        Commands::Init { dir, file } => {
            assert_eq!(dir, "my_models");
            assert_eq!(file, None);
        }
        _ => panic!("Expected Init alias command"),
    }
}

#[test]
fn test_cli_parse_download_defaults() {
    let _guard = CLI_ENV_LOCK.lock().unwrap();
    std::env::remove_var("SNOOZE_MODELS_DIR");
    std::env::remove_var("MODELS_DIR");

    let cli = Cli::try_parse_from([
        "snooze",
        "download",
        "sentence-transformers/all-MiniLM-L6-v2",
    ])
    .unwrap();
    match cli.command {
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
            assert_eq!(repo, "sentence-transformers/all-MiniLM-L6-v2");
            assert_eq!(models_dir, "./models");
            assert_eq!(folder, None);
            assert_eq!(revision, None);
            assert_eq!(include, None);
            assert_eq!(exclude, None);
            assert_eq!(token, None);
            assert!(!no_init);
        }
        _ => panic!("Expected Download command"),
    }
}

#[test]
fn test_cli_parse_download_full_options() {
    let cli = Cli::try_parse_from([
        "snooze",
        "download",
        "openai-community/gpt2",
        "--models-dir",
        "/custom/models",
        "--name",
        "gpt2-custom",
        "--revision",
        "v1.0",
        "--include",
        "*.onnx,config.json",
        "--exclude",
        "*.msgpack",
        "--token",
        "hf_my_secret_token",
        "--no-init",
    ])
    .unwrap();

    match cli.command {
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
            assert_eq!(repo, "openai-community/gpt2");
            assert_eq!(models_dir, "/custom/models");
            assert_eq!(folder, Some("gpt2-custom".to_string()));
            assert_eq!(revision, Some("v1.0".to_string()));
            assert_eq!(
                include,
                Some(vec!["*.onnx".to_string(), "config.json".to_string()])
            );
            assert_eq!(exclude, Some(vec!["*.msgpack".to_string()]));
            assert_eq!(token, Some("hf_my_secret_token".to_string()));
            assert!(no_init);
        }
        _ => panic!("Expected Download command"),
    }
}

#[test]
fn test_cli_parse_models_download() {
    let cli = Cli::try_parse_from([
        "snooze",
        "models",
        "download",
        "sentence-transformers/all-MiniLM-L6-v2",
        "--models-dir",
        "/custom/models",
        "--name",
        "minilm",
    ])
    .unwrap();

    match cli.command {
        Commands::Models {
            action:
                snooze_ml::cli::ModelsAction::Download {
                    repo,
                    models_dir,
                    folder,
                    revision,
                    include,
                    exclude,
                    token,
                    no_init,
                },
        } => {
            assert_eq!(repo, "sentence-transformers/all-MiniLM-L6-v2");
            assert_eq!(models_dir, "/custom/models");
            assert_eq!(folder, Some("minilm".to_string()));
            assert_eq!(revision, None);
            assert_eq!(include, None);
            assert_eq!(exclude, None);
            assert_eq!(token, None);
            assert!(!no_init);
        }
        _ => panic!("Expected ModelsAction::Download command"),
    }
}

#[test]
fn test_cli_models_dir_env_var() {
    let _guard = CLI_ENV_LOCK.lock().unwrap();
    std::env::set_var("SNOOZE_MODELS_DIR", "/env/configured/models");

    let cli = Cli::try_parse_from(["snooze", "download", "dummy/repo"]).unwrap();
    match cli.command {
        Commands::Download { models_dir, .. } => {
            assert_eq!(models_dir, "/env/configured/models");
        }
        _ => panic!("Expected Download command"),
    }

    std::env::remove_var("SNOOZE_MODELS_DIR");
}

#[test]
fn test_cli_command_name() {
    use clap::CommandFactory;
    assert_eq!(Cli::command().get_name(), "snooze");
}

#[test]
fn test_cli_grpc_defaults() {
    let _guard = CLI_ENV_LOCK.lock().unwrap();
    std::env::remove_var("SNOOZE_GRPC");
    std::env::remove_var("SNOOZE_GRPC_PORT");
    std::env::remove_var("GRPC");
    std::env::remove_var("grpc");
    std::env::remove_var("GRPC_PORT");
    std::env::remove_var("grpc_port");

    let cli = Cli::try_parse_from(["snooze", "serve"]).unwrap();
    match cli.command {
        Commands::Serve {
            grpc,
            grpc_port,
            no_grpc,
            ..
        } => {
            assert_eq!(grpc, None);
            assert_eq!(grpc_port, None);
            assert!(!no_grpc);
        }
        _ => panic!("Expected Serve command"),
    }
}

#[test]
fn test_cli_grpc_flags() {
    let _guard = CLI_ENV_LOCK.lock().unwrap();

    // 1. --no-grpc
    let cli = Cli::try_parse_from(["snooze", "serve", "--no-grpc"]).unwrap();
    match cli.command {
        Commands::Serve { no_grpc, .. } => assert!(no_grpc),
        _ => panic!("Expected Serve command"),
    }

    // 2. --grpc
    let cli = Cli::try_parse_from(["snooze", "serve", "--grpc"]).unwrap();
    match cli.command {
        Commands::Serve { grpc, .. } => assert_eq!(grpc, Some(true)),
        _ => panic!("Expected Serve command"),
    }

    // 3. --grpc=false
    let cli = Cli::try_parse_from(["snooze", "serve", "--grpc=false"]).unwrap();
    match cli.command {
        Commands::Serve { grpc, .. } => assert_eq!(grpc, Some(false)),
        _ => panic!("Expected Serve command"),
    }

    // 4. --grpc-port
    let cli = Cli::try_parse_from(["snooze", "serve", "--grpc-port", "50058"]).unwrap();
    match cli.command {
        Commands::Serve { grpc_port, .. } => assert_eq!(grpc_port, Some(50058)),
        _ => panic!("Expected Serve command"),
    }
}

#[test]
fn test_cli_grpc_env_vars() {
    let _guard = CLI_ENV_LOCK.lock().unwrap();

    // SNOOZE_GRPC and SNOOZE_GRPC_PORT clap attributes
    std::env::set_var("SNOOZE_GRPC", "false");
    std::env::set_var("SNOOZE_GRPC_PORT", "50077");

    let cli = Cli::try_parse_from(["snooze", "serve"]).unwrap();
    match cli.command {
        Commands::Serve {
            grpc, grpc_port, ..
        } => {
            assert_eq!(grpc, Some(false));
            assert_eq!(grpc_port, Some(50077));
        }
        _ => panic!("Expected Serve command"),
    }

    // CLI overrides environment variable
    let cli_override =
        Cli::try_parse_from(["snooze", "serve", "--grpc=true", "--grpc-port", "50099"]).unwrap();
    match cli_override.command {
        Commands::Serve {
            grpc, grpc_port, ..
        } => {
            assert_eq!(grpc, Some(true));
            assert_eq!(grpc_port, Some(50099));
        }
        _ => panic!("Expected Serve command"),
    }

    std::env::remove_var("SNOOZE_GRPC");
    std::env::remove_var("SNOOZE_GRPC_PORT");
}

#[test]
fn test_cli_tls_flags_and_defaults() {
    let _guard = CLI_ENV_LOCK.lock().unwrap();
    std::env::remove_var("SNOOZE_TLS");
    std::env::remove_var("TLS");
    std::env::remove_var("SNOOZE_TLS_CERT");
    std::env::remove_var("TLS_CERT");
    std::env::remove_var("SNOOZE_TLS_KEY");
    std::env::remove_var("TLS_KEY");

    // 1. Defaults to None (disabled)
    let cli = Cli::try_parse_from(["snooze", "serve"]).unwrap();
    match cli.command {
        Commands::Serve {
            tls,
            tls_cert,
            tls_key,
            ..
        } => {
            assert_eq!(tls, None);
            assert_eq!(tls_cert, None);
            assert_eq!(tls_key, None);
        }
        _ => panic!("Expected Serve command"),
    }

    // 2. Explicit --tls, --tls-cert, --tls-key
    let cli = Cli::try_parse_from([
        "snooze",
        "serve",
        "--tls",
        "--tls-cert",
        "/etc/ssl/cert.pem",
        "--tls-key",
        "/etc/ssl/key.pem",
    ])
    .unwrap();
    match cli.command {
        Commands::Serve {
            tls,
            tls_cert,
            tls_key,
            ..
        } => {
            assert_eq!(tls, Some(true));
            assert_eq!(
                tls_cert,
                Some(std::path::PathBuf::from("/etc/ssl/cert.pem"))
            );
            assert_eq!(tls_key, Some(std::path::PathBuf::from("/etc/ssl/key.pem")));
        }
        _ => panic!("Expected Serve command"),
    }
}

#[test]
fn test_cli_tls_env_vars() {
    let _guard = CLI_ENV_LOCK.lock().unwrap();

    std::env::set_var("SNOOZE_TLS", "true");
    std::env::set_var("SNOOZE_TLS_CERT", "/env/cert.pem");
    std::env::set_var("SNOOZE_TLS_KEY", "/env/key.pem");

    let cli = Cli::try_parse_from(["snooze", "serve"]).unwrap();
    match cli.command {
        Commands::Serve {
            tls,
            tls_cert,
            tls_key,
            ..
        } => {
            assert_eq!(tls, Some(true));
            assert_eq!(tls_cert, Some(std::path::PathBuf::from("/env/cert.pem")));
            assert_eq!(tls_key, Some(std::path::PathBuf::from("/env/key.pem")));
        }
        _ => panic!("Expected Serve command"),
    }

    std::env::remove_var("SNOOZE_TLS");
    std::env::remove_var("SNOOZE_TLS_CERT");
    std::env::remove_var("SNOOZE_TLS_KEY");
}
