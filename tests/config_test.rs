//! Behavioral and integration tests for model directory configuration, per-model settings,
//! discovery, fault-tolerant parsing, and stub generation.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use snooze_ml::config::{
    DiscoveredModel, GlobalModelDefaults, ModelDirectoryConfig, ModelEntryConfig, DEFAULT_DEVICE,
    DEFAULT_IDLE_TIMEOUT_SECS, DEFAULT_MAX_BATCH_DELAY_MS, DEFAULT_MAX_BATCH_SIZE,
    STUB_DESCRIPTION,
};
use std::path::Path;
use std::time::Duration;

#[test]
fn test_default_config_values() {
    let defaults = GlobalModelDefaults::default();
    assert_eq!(defaults.idle_timeout_secs, Some(DEFAULT_IDLE_TIMEOUT_SECS));
    assert_eq!(defaults.max_batch_size, Some(DEFAULT_MAX_BATCH_SIZE));
    assert_eq!(
        defaults.max_batch_delay_ms,
        Some(DEFAULT_MAX_BATCH_DELAY_MS)
    );
    assert_eq!(defaults.device.as_deref(), Some(DEFAULT_DEVICE));

    let entry = ModelEntryConfig::default();
    assert!(entry.id.is_empty());
    assert!(entry.path.is_empty());
    assert_eq!(entry.description, None);
    assert_eq!(entry.device, None);
    assert_eq!(entry.max_batch_size, None);
    assert_eq!(entry.min_residency_secs, None);
}

#[test]
fn test_parse_toml_basic_and_per_model_settings() {
    let toml = r#"
# Global daemon defaults
[defaults]
idle_timeout_secs = 600
max_batch_size = 32
max_batch_delay_ms = 5
device = "cpu"

[[models]]
id = "model-cpu"
path = "model_cpu.onnx"
description = "CPU embedding model"
device = "cpu"
max_batch_size = 8
max_batch_delay_ms = 2
idle_timeout_secs = 120

[[models]]
id = "model-gpu"
path = "model_gpu.onnx"
description = "High-throughput GPU model"
device = "gpu"
max_batch_size = 64
max_batch_delay_ms = 20
idle_timeout_secs = 900
"#;

    let config = ModelDirectoryConfig::parse_toml_fault_tolerant(toml).unwrap();
    assert_eq!(config.defaults.idle_timeout_secs, Some(600));
    assert_eq!(config.defaults.max_batch_size, Some(32));
    assert_eq!(config.defaults.max_batch_delay_ms, Some(5));
    assert_eq!(config.defaults.device.as_deref(), Some("cpu"));

    assert_eq!(config.models.len(), 2);

    let m1 = &config.models[0];
    assert_eq!(m1.id, "model-cpu");
    assert_eq!(m1.path, "model_cpu.onnx");
    assert_eq!(m1.description.as_deref(), Some("CPU embedding model"));
    assert_eq!(m1.device.as_deref(), Some("cpu"));
    assert_eq!(m1.max_batch_size, Some(8));
    assert_eq!(m1.max_batch_delay_ms, Some(2));
    assert_eq!(m1.idle_timeout_secs, Some(120));

    let m2 = &config.models[1];
    assert_eq!(m2.id, "model-gpu");
    assert_eq!(m2.path, "model_gpu.onnx");
    assert_eq!(m2.description.as_deref(), Some("High-throughput GPU model"));
    assert_eq!(m2.device.as_deref(), Some("gpu"));
    assert_eq!(m2.max_batch_size, Some(64));
    assert_eq!(m2.max_batch_delay_ms, Some(20));
    assert_eq!(m2.idle_timeout_secs, Some(900));
}

#[test]
fn test_parse_toml_fault_tolerance_with_malformed_lines() {
    let toml = r#"
# Some comments
[defaults]
idle_timeout_secs = 150
invalid_syntax_without_equals
random gibberish %%%
max_batch_size = 12

[[models]]
id = "valid-model"
path = "valid.onnx"
# Model has typo on one line
unknown_flag = true
max_batch_size = 4
description = "A resilient model"
"#;

    let config = ModelDirectoryConfig::parse_toml_fault_tolerant(toml).unwrap();
    assert_eq!(config.defaults.idle_timeout_secs, Some(150));
    assert_eq!(config.defaults.max_batch_size, Some(12));

    assert_eq!(config.models.len(), 1);
    let m = &config.models[0];
    assert_eq!(m.id, "valid-model");
    assert_eq!(m.path, "valid.onnx");
    assert_eq!(m.max_batch_size, Some(4));
    assert_eq!(m.description.as_deref(), Some("A resilient model"));
}

#[test]
fn test_parse_toml_table_syntax() {
    let toml = r#"
[defaults]
max_batch_size = 16

[models.alpha]
path = "alpha.onnx"
description = "Alpha model"
device = "cuda"

[models.beta]
path = "beta.safetensors"
description = "Beta model"
max_batch_size = 32
"#;

    let config = ModelDirectoryConfig::parse_toml_fault_tolerant(toml).unwrap();
    assert_eq!(config.models.len(), 2);
    let alpha = config.models.iter().find(|m| m.id == "alpha").unwrap();
    assert_eq!(alpha.path, "alpha.onnx");
    assert_eq!(alpha.description.as_deref(), Some("Alpha model"));
    assert_eq!(alpha.device.as_deref(), Some("cuda"));

    let beta = config.models.iter().find(|m| m.id == "beta").unwrap();
    assert_eq!(beta.path, "beta.safetensors");
    assert_eq!(beta.max_batch_size, Some(32));
}

#[test]
fn test_parse_json_format() {
    let json = r#"{
        "defaults": {
            "idle_timeout_secs": 400,
            "max_batch_size": 24,
            "max_batch_delay_ms": 12,
            "device": "cuda"
        },
        "models": [
            {
                "id": "json-model",
                "path": "json_model.onnx",
                "description": "Configured via JSON",
                "device": "gpu",
                "max_batch_size": 16
            }
        ]
    }"#;

    let path = Path::new("models.json");
    let config = ModelDirectoryConfig::parse_content(json, path).unwrap();
    assert_eq!(config.defaults.idle_timeout_secs, Some(400));
    assert_eq!(config.defaults.max_batch_size, Some(24));
    assert_eq!(config.models.len(), 1);
    assert_eq!(config.models[0].id, "json-model");
    assert_eq!(
        config.models[0].description.as_deref(),
        Some("Configured via JSON")
    );
}

#[test]
fn test_device_normalization() {
    assert_eq!(
        ModelDirectoryConfig::normalize_device(Some("auto")),
        Some("auto".to_string())
    );
    assert_eq!(
        ModelDirectoryConfig::normalize_device(Some("gpu")),
        Some("cuda".to_string())
    );
    assert_eq!(
        ModelDirectoryConfig::normalize_device(Some("CUDA")),
        Some("cuda".to_string())
    );
    assert_eq!(
        ModelDirectoryConfig::normalize_device(Some("mlx")),
        Some("mlx".to_string())
    );
    assert_eq!(
        ModelDirectoryConfig::normalize_device(Some("metal")),
        Some("mlx".to_string())
    );
    assert_eq!(
        ModelDirectoryConfig::normalize_device(Some("coreml")),
        Some("mlx".to_string())
    );
    assert_eq!(
        ModelDirectoryConfig::normalize_device(Some("cpu")),
        Some("cpu".to_string())
    );
    assert_eq!(
        ModelDirectoryConfig::normalize_device(Some("tensorrt")),
        Some("tensorrt".to_string())
    );
    assert_eq!(
        ModelDirectoryConfig::normalize_device(Some("safetensors")),
        Some("candle".to_string())
    );
    assert_eq!(ModelDirectoryConfig::normalize_device(None), None);
}

#[tokio::test]
async fn test_discover_models_single_files_and_subdirectories() {
    let mut temp_dir = std::env::temp_dir();
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    temp_dir.push(format!("snooze_discover_test_{unique}"));
    tokio::fs::create_dir_all(&temp_dir).await.unwrap();

    // 1. Direct model file
    tokio::fs::write(temp_dir.join("standalone.onnx"), b"fake onnx")
        .await
        .unwrap();
    // 2. Direct safetensors file
    tokio::fs::write(temp_dir.join("standalone.safetensors"), b"fake st")
        .await
        .unwrap();
    // 3. Hugging Face subdirectory with model.onnx
    let hf_dir1 = temp_dir.join("bge-small");
    tokio::fs::create_dir_all(&hf_dir1).await.unwrap();
    tokio::fs::write(hf_dir1.join("model.onnx"), b"fake hf onnx")
        .await
        .unwrap();
    tokio::fs::write(hf_dir1.join("tokenizer.json"), b"fake tok")
        .await
        .unwrap();

    // 4. Hugging Face subdirectory with onnx/model.onnx
    let hf_dir2 = temp_dir.join("sentence-transformer");
    tokio::fs::create_dir_all(hf_dir2.join("onnx"))
        .await
        .unwrap();
    tokio::fs::write(hf_dir2.join("onnx").join("model.onnx"), b"nested onnx")
        .await
        .unwrap();

    // 5. Non-model files
    tokio::fs::write(temp_dir.join("README.md"), b"# Notes")
        .await
        .unwrap();
    tokio::fs::write(temp_dir.join("config.json"), b"{}")
        .await
        .unwrap();

    let discovered = ModelDirectoryConfig::discover_models_in_dir(&temp_dir)
        .await
        .unwrap();

    let ids: Vec<String> = discovered.into_iter().map(|d| d.id).collect();
    assert_eq!(
        ids,
        vec![
            "bge-small",
            "sentence-transformer",
            "standalone",
            "standalone" // stem of standalone.safetensors
        ]
    );

    let _ = tokio::fs::remove_dir_all(&temp_dir).await;
}

#[tokio::test]
async fn test_init_or_update_dir_creates_new_config_with_stubs() {
    let mut temp_dir = std::env::temp_dir();
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    temp_dir.push(format!("snooze_init_test_{unique}"));
    tokio::fs::create_dir_all(&temp_dir).await.unwrap();

    // Place models in directory
    tokio::fs::write(temp_dir.join("resnet50.onnx"), b"fake onnx")
        .await
        .unwrap();
    let hf = temp_dir.join("bert-base");
    tokio::fs::create_dir_all(&hf).await.unwrap();
    tokio::fs::write(hf.join("model.onnx"), b"fake hf onnx")
        .await
        .unwrap();

    let (target_file, added) = ModelDirectoryConfig::init_or_update_dir(&temp_dir, None)
        .await
        .unwrap();

    assert_eq!(target_file, temp_dir.join("models.toml"));
    assert_eq!(added, 2);
    assert!(target_file.is_file());

    let content = tokio::fs::read_to_string(&target_file).await.unwrap();
    assert!(content.contains("[defaults]"));
    assert!(content.contains("[[models]]"));
    assert!(content.contains("id = \"bert-base\""));
    assert!(content.contains("id = \"resnet50\""));
    assert!(content.contains(&format!("description = \"{STUB_DESCRIPTION}\"")));

    // Re-running without adding new models should result in 0 added
    let (_, added_again) = ModelDirectoryConfig::init_or_update_dir(&temp_dir, None)
        .await
        .unwrap();
    assert_eq!(added_again, 0);

    let _ = tokio::fs::remove_dir_all(&temp_dir).await;
}

#[tokio::test]
async fn test_init_or_update_dir_merges_new_model() {
    let mut temp_dir = std::env::temp_dir();
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    temp_dir.push(format!("snooze_merge_test_{unique}"));
    tokio::fs::create_dir_all(&temp_dir).await.unwrap();

    // Start with model1
    tokio::fs::write(temp_dir.join("model1.onnx"), b"fake1")
        .await
        .unwrap();
    let (_, added1) = ModelDirectoryConfig::init_or_update_dir(&temp_dir, None)
        .await
        .unwrap();
    assert_eq!(added1, 1);

    // Customize description of model1 in the file
    let config_file = temp_dir.join("models.toml");
    let initial_content = tokio::fs::read_to_string(&config_file).await.unwrap();
    let customized_content = initial_content.replace(STUB_DESCRIPTION, "Custom Description 1");
    tokio::fs::write(&config_file, customized_content.as_bytes())
        .await
        .unwrap();

    // Add model2 to the folder
    tokio::fs::write(temp_dir.join("model2.onnx"), b"fake2")
        .await
        .unwrap();

    // Update config
    let (_, added2) = ModelDirectoryConfig::init_or_update_dir(&temp_dir, None)
        .await
        .unwrap();
    assert_eq!(added2, 1);

    let final_content = tokio::fs::read_to_string(&config_file).await.unwrap();
    // Custom description of model1 is preserved
    assert!(final_content.contains("Custom Description 1"));
    // New model stub is appended
    assert!(final_content.contains("id = \"model2\""));
    assert!(final_content.contains(&format!("description = \"{STUB_DESCRIPTION}\"")));

    let _ = tokio::fs::remove_dir_all(&temp_dir).await;
}

#[tokio::test]
async fn test_build_model_configs_with_unconfigured_and_configured_models() {
    let mut temp_dir = std::env::temp_dir();
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    temp_dir.push(format!("snooze_build_test_{unique}"));
    tokio::fs::create_dir_all(&temp_dir).await.unwrap();

    let model1_path = temp_dir.join("m1.onnx");
    let model2_path = temp_dir.join("m2.onnx");
    tokio::fs::write(&model1_path, b"fake1").await.unwrap();
    tokio::fs::write(&model2_path, b"fake2").await.unwrap();

    // Config only specifies m1 with custom GPU and batch size
    let config = ModelDirectoryConfig {
        defaults: GlobalModelDefaults {
            idle_timeout_secs: Some(100),
            max_batch_size: Some(10),
            max_batch_delay_ms: Some(5),
            device: Some("cpu".to_string()),
            max_loaded_models: None,
            min_residency_secs: None,
            ..Default::default()
        },
        models: vec![ModelEntryConfig {
            id: "custom_m1".to_string(),
            path: "m1.onnx".to_string(),
            description: Some("Custom M1".to_string()),
            device: Some("gpu".to_string()), // Should normalize to cuda
            max_batch_size: Some(64),
            max_batch_delay_ms: Some(25),
            idle_timeout_secs: Some(500),
            min_residency_secs: None,
            ..Default::default()
        }],
        auth: None,
        streams: Vec::new(),
    };

    let discovered = vec![
        DiscoveredModel {
            id: "m1".to_string(),
            relative_path: "m1.onnx".to_string(),
            absolute_path: model1_path.clone(),
        },
        DiscoveredModel {
            id: "m2".to_string(),
            relative_path: "m2.onnx".to_string(),
            absolute_path: model2_path.clone(),
        },
    ];

    let built = config.build_model_configs(&temp_dir, &discovered);
    assert_eq!(built.len(), 2);

    let m1_cfg = built.iter().find(|m| m.id == "custom_m1").unwrap();
    assert_eq!(m1_cfg.max_batch_size, 64);
    assert_eq!(m1_cfg.max_batch_delay, Duration::from_millis(25));
    assert_eq!(m1_cfg.idle_timeout, Duration::from_secs(500));
    assert_eq!(m1_cfg.provider.as_deref(), Some("cuda"));
    assert_eq!(m1_cfg.description.as_deref(), Some("Custom M1"));

    // m2 was not configured, so it is auto-discovered with defaults
    let m2_cfg = built.iter().find(|m| m.id == "m2").unwrap();
    assert_eq!(m2_cfg.max_batch_size, 10);
    assert_eq!(m2_cfg.max_batch_delay, Duration::from_millis(5));
    assert_eq!(m2_cfg.idle_timeout, Duration::from_secs(100));
    assert_eq!(m2_cfg.provider.as_deref(), Some("cpu"));
    assert_eq!(m2_cfg.description, None);

    let _ = tokio::fs::remove_dir_all(&temp_dir).await;
}

#[tokio::test]
async fn test_fault_tolerant_skips_missing_model_files() {
    let mut temp_dir = std::env::temp_dir();
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    temp_dir.push(format!("snooze_missing_test_{unique}"));
    tokio::fs::create_dir_all(&temp_dir).await.unwrap();

    let real_model = temp_dir.join("real.onnx");
    tokio::fs::write(&real_model, b"fake onnx").await.unwrap();

    let config = ModelDirectoryConfig {
        defaults: GlobalModelDefaults::default(),
        models: vec![
            ModelEntryConfig {
                id: "missing-one".to_string(),
                path: "does_not_exist.onnx".to_string(),
                description: None,
                device: None,
                max_batch_size: None,
                max_batch_delay_ms: None,
                idle_timeout_secs: None,
                min_residency_secs: None,
                ..Default::default()
            },
            ModelEntryConfig {
                id: "real-one".to_string(),
                path: "real.onnx".to_string(),
                description: Some("Real model".to_string()),
                device: None,
                max_batch_size: None,
                max_batch_delay_ms: None,
                idle_timeout_secs: None,
                min_residency_secs: None,
                ..Default::default()
            },
        ],
        auth: None,
        streams: Vec::new(),
    };

    let built = config.build_model_configs(&temp_dir, &[]);
    // missing-one was skipped safely, real-one was registered
    assert_eq!(built.len(), 1);
    assert_eq!(built[0].id, "real-one");
    assert_eq!(built[0].provider.as_deref(), Some("auto"));

    let _ = tokio::fs::remove_dir_all(&temp_dir).await;
}

#[tokio::test]
async fn test_auto_device_default_in_model_configs() {
    let mut temp_dir = std::env::temp_dir();
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    temp_dir.push(format!("snooze_auto_dev_test_{unique}"));
    tokio::fs::create_dir_all(&temp_dir).await.unwrap();

    let model_path = temp_dir.join("sample.onnx");
    tokio::fs::write(&model_path, b"onnx data").await.unwrap();

    let discovered = vec![DiscoveredModel {
        id: "sample".to_string(),
        relative_path: "sample.onnx".to_string(),
        absolute_path: model_path.clone(),
    }];

    // Empty config defaults to "auto"
    let config = ModelDirectoryConfig::default();
    let built = config.build_model_configs(&temp_dir, &discovered);
    assert_eq!(built.len(), 1);
    assert_eq!(built[0].id, "sample");
    assert_eq!(built[0].provider.as_deref(), Some("auto"));

    let _ = tokio::fs::remove_dir_all(&temp_dir).await;
}

#[tokio::test]
async fn test_auto_device_in_stubs() {
    let mut temp_dir = std::env::temp_dir();
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    temp_dir.push(format!("snooze_auto_stub_test_{unique}"));
    tokio::fs::create_dir_all(&temp_dir).await.unwrap();

    let model_file = temp_dir.join("net.onnx");
    tokio::fs::write(&model_file, b"onnx").await.unwrap();

    let (cfg_path, count) = ModelDirectoryConfig::init_or_update_dir(&temp_dir, None)
        .await
        .unwrap();
    assert_eq!(count, 1);

    let content = tokio::fs::read_to_string(&cfg_path).await.unwrap();
    assert!(content.contains("device = \"auto\""));

    let _ = tokio::fs::remove_dir_all(&temp_dir).await;
}

#[test]
fn test_default_models_dir_env_resolution() {
    use snooze_ml::config::{
        default_models_dir, DEFAULT_MODELS_DIR, ENV_MODELS_DIR, ENV_MODELS_DIR_FALLBACK,
    };

    assert_eq!(DEFAULT_MODELS_DIR, "./models");
    assert_eq!(ENV_MODELS_DIR, "SNOOZE_MODELS_DIR");
    assert_eq!(ENV_MODELS_DIR_FALLBACK, "MODELS_DIR");

    // Clean any ambient env vars
    std::env::remove_var(ENV_MODELS_DIR);
    std::env::remove_var(ENV_MODELS_DIR_FALLBACK);
    assert_eq!(default_models_dir(), "./models");

    // Test SNOOZE_MODELS_DIR
    std::env::set_var(ENV_MODELS_DIR, "/custom/snooze/models");
    assert_eq!(default_models_dir(), "/custom/snooze/models");

    // Test SNOOZE_MODELS_DIR precedence over MODELS_DIR
    std::env::set_var(ENV_MODELS_DIR_FALLBACK, "/fallback/models");
    assert_eq!(default_models_dir(), "/custom/snooze/models");

    // Test fallback when SNOOZE_MODELS_DIR is empty or removed
    std::env::remove_var(ENV_MODELS_DIR);
    assert_eq!(default_models_dir(), "/fallback/models");

    // Clean up
    std::env::remove_var(ENV_MODELS_DIR_FALLBACK);
    assert_eq!(default_models_dir(), "./models");
}

#[test]
fn test_parse_toml_max_loaded_and_min_residency() {
    let toml = r#"
[defaults]
idle_timeout_secs = 300
max_batch_size = 16
max_loaded_models = 4
min_residency_secs = 15

[[models]]
id = "m1"
path = "m1.onnx"
min_residency_secs = 5

[[models]]
id = "m2"
path = "m2.onnx"
"#;

    let parsed = ModelDirectoryConfig::parse_toml_fault_tolerant(toml).unwrap();
    assert_eq!(parsed.defaults.max_loaded_models, Some(4));
    assert_eq!(parsed.defaults.min_residency_secs, Some(15));
    assert_eq!(parsed.models.len(), 2);
    assert_eq!(parsed.models[0].min_residency_secs, Some(5));
    assert_eq!(parsed.models[1].min_residency_secs, None);

    let output_toml = parsed.to_toml_string();
    assert!(output_toml.contains("max_loaded_models = 4"));
    assert!(output_toml.contains("min_residency_secs = 15"));
}

#[tokio::test]
async fn test_parse_toml_and_json_pinned_models() {
    let mut temp_dir = std::env::temp_dir();
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    temp_dir.push(format!("snooze_pinned_cfg_test_{unique}"));
    tokio::fs::create_dir_all(&temp_dir).await.unwrap();

    for name in ["m_pinned.onnx", "m_preload.onnx", "m_unpinned.onnx"] {
        tokio::fs::write(temp_dir.join(name), b"onnx")
            .await
            .unwrap();
    }

    let toml = r#"
[defaults]
pinned = false

[[models]]
id = "m_pinned"
path = "m_pinned.onnx"
pinned = true

[[models]]
id = "m_preload"
path = "m_preload.onnx"
preload = true

[[models]]
id = "m_unpinned"
path = "m_unpinned.onnx"
"#;

    let parsed = ModelDirectoryConfig::parse_toml_fault_tolerant(toml).unwrap();
    assert_eq!(parsed.defaults.pinned, Some(false));
    assert_eq!(parsed.models.len(), 3);
    assert_eq!(parsed.models[0].pinned, Some(true));
    assert_eq!(parsed.models[1].pinned, Some(true));
    assert_eq!(parsed.models[2].pinned, None);

    let output_toml = parsed.to_toml_string();
    assert!(output_toml.contains("pinned = true"));

    let built = parsed.build_model_configs(&temp_dir, &[]);
    assert_eq!(built.len(), 3);
    assert!(built[0].pinned);
    assert!(built[1].pinned);
    assert!(!built[2].pinned);

    let _ = tokio::fs::remove_dir_all(&temp_dir).await;
}
