//! Model directory configuration, per-model settings, and discovery system.
//!
//! Provides fault-tolerant configuration loading, discovery for both standalone model
//! files (`.onnx`, `.safetensors`) and multi-file directory structures (e.g. Hugging Face
//! downloads), as well as sample configuration and stub generation utilities.

use crate::auth::{ApiKeyConfig, AuthConfig, JwtAlgorithm, JwtConfig, Role};
use crate::model::ModelConfig;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Default idle timeout in seconds before unloading model weights (scale-to-zero).
pub const DEFAULT_IDLE_TIMEOUT_SECS: u64 = 300;

/// Default maximum dynamic batch size.
pub const DEFAULT_MAX_BATCH_SIZE: usize = 16;

/// Default maximum dynamic batch latency budget in milliseconds.
pub const DEFAULT_MAX_BATCH_DELAY_MS: u64 = 10;

/// Default execution device/provider preference.
pub const DEFAULT_DEVICE: &str = "auto";

/// Default minimum duration in seconds a model must stay resident before deallocation (anti-thrashing).
pub const DEFAULT_MIN_RESIDENCY_SECS: u64 = 10;

/// Placeholder description assigned when generating model configuration stubs.
pub const STUB_DESCRIPTION: &str = "Add description there";

/// Default models directory path.
pub const DEFAULT_MODELS_DIR: &str = "./models";

/// Environment variable used to override the models directory.
pub const ENV_MODELS_DIR: &str = "SNOOZE_MODELS_DIR";

/// Fallback environment variable for the models directory.
pub const ENV_MODELS_DIR_FALLBACK: &str = "MODELS_DIR";

/// Resolves the default models directory using environment variables ([`ENV_MODELS_DIR`] or
/// [`ENV_MODELS_DIR_FALLBACK`]), falling back to [`DEFAULT_MODELS_DIR`] (`"./models"`).
#[must_use]
pub fn default_models_dir() -> String {
    if let Ok(val) = std::env::var(ENV_MODELS_DIR) {
        let trimmed = val.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }
    if let Ok(val) = std::env::var(ENV_MODELS_DIR_FALLBACK) {
        let trimmed = val.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }
    DEFAULT_MODELS_DIR.to_string()
}

/// Global defaults applied to models when per-model settings are omitted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GlobalModelDefaults {
    /// Inactivity duration in seconds before memory deallocation.
    pub idle_timeout_secs: Option<u64>,
    /// Maximum micro-batch size for request collation.
    pub max_batch_size: Option<usize>,
    /// Latency window in milliseconds to accumulate requests.
    pub max_batch_delay_ms: Option<u64>,
    /// Execution provider or hardware device (e.g. `"auto"`, `"gpu"`, `"cuda"`, `"mlx"`, `"cpu"`, `"tensorrt"`).
    pub device: Option<String>,
    /// Maximum number of concurrently loaded model sessions in memory before preemptive deallocation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_loaded_models: Option<usize>,
    /// Minimum duration in seconds a model must remain resident before being eligible for LRU deallocation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_residency_secs: Option<u64>,
    /// Queue threshold before triggering replica autoscaling.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub autoscale_queue_threshold: Option<usize>,
    /// Optional upper limit on parallel replica instances.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_replicas: Option<usize>,
    /// Replica routing strategy name ("least_queue", "round_robin", "random").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub routing_strategy: Option<String>,
    /// Whether models are pinned in memory at startup by default and never auto-deallocated.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pinned: Option<bool>,
}

impl Default for GlobalModelDefaults {
    fn default() -> Self {
        Self {
            idle_timeout_secs: Some(DEFAULT_IDLE_TIMEOUT_SECS),
            max_batch_size: Some(DEFAULT_MAX_BATCH_SIZE),
            max_batch_delay_ms: Some(DEFAULT_MAX_BATCH_DELAY_MS),
            device: Some(DEFAULT_DEVICE.to_string()),
            max_loaded_models: None,
            min_residency_secs: Some(DEFAULT_MIN_RESIDENCY_SECS),
            autoscale_queue_threshold: None,
            max_replicas: None,
            routing_strategy: None,
            pinned: None,
        }
    }
}

/// Configuration settings for an individual model entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ModelEntryConfig {
    /// Unique identifier for the model in the registry and API endpoints.
    pub id: String,
    /// Path to model weights, relative to the models directory or absolute.
    pub path: String,
    /// Optional custom description of the model returned via the API.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Target hardware device or provider (e.g. `"cpu"`, `"cuda"`, `"gpu"`, `"tensorrt"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,
    /// Per-model maximum micro-batch size.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_batch_size: Option<usize>,
    /// Per-model batch latency budget in milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_batch_delay_ms: Option<u64>,
    /// Per-model idle timeout in seconds before deallocation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub idle_timeout_secs: Option<u64>,
    /// Per-model minimum residency duration in seconds before eligible for deallocation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_residency_secs: Option<u64>,
    /// Per-model queue threshold before triggering replica autoscaling.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub autoscale_queue_threshold: Option<usize>,
    /// Per-model maximum replica instances.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_replicas: Option<usize>,
    /// Per-model replica routing strategy name ("least_queue", "round_robin", "random").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub routing_strategy: Option<String>,
    /// Whether this model is pinned in memory at startup and exempt from automatic deallocation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pinned: Option<bool>,
}

fn default_stream_auto_create() -> bool {
    true
}

fn default_stream_concurrency() -> usize {
    16
}

fn default_stream_partitions() -> usize {
    1
}

/// Supported payload serialization formats across streaming transports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum StreamWireFormat {
    /// Standard JSON serialization matching HTTP REST `InferRequest` and `InferResponse` schemas.
    #[default]
    Json,
    /// Binary Protobuf serialization matching the native `snoozeml.v1` gRPC wire schema.
    Protobuf,
}

/// Configuration settings for an individual message stream binding.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamBindingConfig {
    /// Human-readable identifier for this stream worker binding.
    pub name: String,
    /// Message broker type (`"kafka"`, `"rabbitmq"`, `"redis"`, or `"mock"`).
    pub broker: String,
    /// Connection endpoint URL or bootstrap servers (e.g. `"localhost:9092"` or `"redis://127.0.0.1:6379"`).
    pub endpoint: String,
    /// Optional default model identifier. If `None` (the default), stream requests are dynamically
    /// routed to whatever `model_id` is specified inside each incoming stream message, allowing
    /// all registered models to be served over the stream. If `Some(id)`, acts as a default/fallback.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    /// Input topic, queue name, or Redis stream key to consume requests from.
    pub input_topic: String,
    /// Output topic, queue name, or Redis stream key to publish responses to (defaults to `Some("snooze.responses")`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_topic: Option<String>,
    /// Optional error or Dead-Letter Queue (DLQ) topic to publish failures to.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_topic: Option<String>,
    /// Consumer group identifier (for Kafka and Redis Streams).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub consumer_group: Option<String>,
    /// Whether to automatically create topics/queues if they do not exist on the broker (defaults to `true`).
    #[serde(default = "default_stream_auto_create")]
    pub auto_create_topics: bool,
    /// Maximum concurrent in-flight inference requests allowed for backpressure control (defaults to `16`).
    #[serde(default = "default_stream_concurrency")]
    pub concurrency: usize,
    /// Default partition count when automatically creating topics (defaults to `1`).
    #[serde(default = "default_stream_partitions")]
    pub partitions: usize,
    /// Wire serialization format for stream messages (`"json"` or `"protobuf"`, defaults to `"json"`).
    #[serde(default)]
    pub format: StreamWireFormat,
}

impl Default for StreamBindingConfig {
    fn default() -> Self {
        Self {
            name: "default-stream".to_string(),
            broker: "mock".to_string(),
            endpoint: "memory://".to_string(),
            model_id: None,
            input_topic: "snooze.requests".to_string(),
            output_topic: Some("snooze.responses".to_string()),
            error_topic: Some("snooze.dlq".to_string()),
            consumer_group: Some("snoozeml-group".to_string()),
            auto_create_topics: true,
            concurrency: 16,
            partitions: 1,
            format: StreamWireFormat::Json,
        }
    }
}

/// Root configuration structure representing a model directory's setup.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ModelDirectoryConfig {
    /// Global fallback settings.
    #[serde(default)]
    pub defaults: GlobalModelDefaults,
    /// List of per-model configurations.
    #[serde(default)]
    pub models: Vec<ModelEntryConfig>,
    /// Optional stream bindings for message broker ingestion and publication.
    #[serde(default)]
    pub streams: Vec<StreamBindingConfig>,
    /// Optional authentication and role-based access control settings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth: Option<AuthConfig>,
}

/// Discovered model artifact on the filesystem.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveredModel {
    /// Suggested unique model identifier.
    pub id: String,
    /// Path to weights relative to the models directory (or absolute if needed).
    pub relative_path: String,
    /// Full resolved filesystem path to the model weight file.
    pub absolute_path: PathBuf,
}

impl ModelDirectoryConfig {
    /// Discovers all model weight files (`.onnx`, `.safetensors`) within a directory,
    /// searching both standalone files and subdirectories (e.g. Hugging Face downloads).
    ///
    /// # Errors
    /// Returns [`std::io::Error`] if the directory cannot be read.
    pub async fn discover_models_in_dir(
        dir: &Path,
    ) -> Result<Vec<DiscoveredModel>, std::io::Error> {
        let mut discovered = Vec::new();
        if !dir.is_dir() {
            return Ok(discovered);
        }

        let mut entries = tokio::fs::read_dir(dir).await?;
        while let Some(entry) = entries.next_entry().await? {
            let entry_path = entry.path();
            let file_type = entry.file_type().await?;

            if file_type.is_file() {
                if let Some(model) = Self::inspect_file(&entry_path, dir) {
                    discovered.push(model);
                }
            } else if file_type.is_dir() {
                // If the directory directly contains snooze_model.py or snooze_model.json, treat the entire directory
                // as a worker model and do not descend into it for stray weights.
                if entry_path.join("snooze_model.py").is_file()
                    || entry_path.join("snooze_model.json").is_file()
                {
                    let folder_name = entry_path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("model")
                        .to_string();
                    let relative_path = entry_path
                        .strip_prefix(dir)
                        .ok()
                        .map(|p| p.to_string_lossy().to_string())
                        .unwrap_or_else(|| folder_name.clone());
                    discovered.push(DiscoveredModel {
                        id: folder_name,
                        relative_path,
                        absolute_path: entry_path.clone(),
                    });
                    continue;
                }
                // Subdirectory discovery: Hugging Face repos or nested model folders
                let sub_models = Self::inspect_subdir(&entry_path, dir).await?;
                discovered.extend(sub_models);
            }
        }

        // Sort by ID for deterministic discovery order
        discovered.sort_by(|a, b| a.id.cmp(&b.id));
        Ok(discovered)
    }

    /// Evaluates whether a file is a supported model weight file.
    fn inspect_file(file_path: &Path, base_dir: &Path) -> Option<DiscoveredModel> {
        let ext = file_path.extension()?.to_str()?;
        if ext == "onnx" || ext == "safetensors" || ext == "xgb" || ext == "tree" {
            let stem = file_path.file_stem()?.to_str()?;
            let relative_path = file_path
                .strip_prefix(base_dir)
                .ok()?
                .to_string_lossy()
                .to_string();

            Some(DiscoveredModel {
                id: stem.to_string(),
                relative_path,
                absolute_path: file_path.to_path_buf(),
            })
        } else if ext == "json" {
            let file_name = file_path.file_name()?.to_str()?;
            if file_name == "config.json"
                || file_name == "tokenizer.json"
                || file_name == "generation_config.json"
                || file_name == "tokenizer_config.json"
                || file_name == "special_tokens_map.json"
                || file_name.starts_with("reference_")
            {
                return None;
            }

            if Self::is_tree_json_file(file_path) {
                let stem = if let Some(s) = file_name.strip_suffix(".xgb.json") {
                    s
                } else if let Some(s) = file_name.strip_suffix(".tree.json") {
                    s
                } else {
                    file_path.file_stem()?.to_str()?
                };
                let relative_path = file_path
                    .strip_prefix(base_dir)
                    .ok()?
                    .to_string_lossy()
                    .to_string();

                Some(DiscoveredModel {
                    id: stem.to_string(),
                    relative_path,
                    absolute_path: file_path.to_path_buf(),
                })
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Evaluates whether a JSON file contains tree model ensemble metadata.
    fn is_tree_json_file(file_path: &Path) -> bool {
        let file_name = match file_path.file_name().and_then(|n| n.to_str()) {
            Some(n) => n,
            None => return false,
        };
        if file_name.ends_with(".xgb.json") || file_name.ends_with(".tree.json") {
            return true;
        }
        if let Ok(mut file) = std::fs::File::open(file_path) {
            use std::io::Read;
            let mut buffer = [0u8; 1024];
            if let Ok(n) = file.read(&mut buffer) {
                let s = String::from_utf8_lossy(buffer.get(..n).unwrap_or(&[]));
                if s.contains("gradient_booster") || s.contains("gbtree") || s.contains("\"trees\"")
                {
                    return true;
                }
            }
        }
        false
    }

    /// Inspects a subdirectory (e.g. Hugging Face format) for weight files.
    async fn inspect_subdir(
        subdir_path: &Path,
        base_dir: &Path,
    ) -> Result<Vec<DiscoveredModel>, std::io::Error> {
        let mut models = Vec::new();
        let folder_name = match subdir_path.file_name().and_then(|n| n.to_str()) {
            Some(name) => name.to_string(),
            None => return Ok(models),
        };

        if subdir_path.join("snooze_model.py").is_file()
            || subdir_path.join("snooze_model.json").is_file()
        {
            if let Ok(rel) = subdir_path.strip_prefix(base_dir) {
                models.push(DiscoveredModel {
                    id: folder_name,
                    relative_path: rel.to_string_lossy().to_string(),
                    absolute_path: subdir_path.to_path_buf(),
                });
            }
            return Ok(models);
        }

        // Check immediate children and one level of subfolders (e.g. subdir/onnx/model.onnx)
        let mut sub_entries = tokio::fs::read_dir(subdir_path).await?;
        let mut found_files = Vec::new();

        while let Some(sub_entry) = sub_entries.next_entry().await? {
            let path = sub_entry.path();
            let ft = sub_entry.file_type().await?;

            if ft.is_file() {
                if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                    if ext == "onnx"
                        || ext == "safetensors"
                        || ext == "xgb"
                        || ext == "tree"
                        || (ext == "json" && Self::is_tree_json_file(&path))
                    {
                        found_files.push(path);
                    }
                }
            } else if ft.is_dir() {
                // E.g. folder_name/onnx/model.onnx
                let mut nested_entries = tokio::fs::read_dir(&path).await?;
                while let Some(nested_entry) = nested_entries.next_entry().await? {
                    let npath = nested_entry.path();
                    if nested_entry.file_type().await?.is_file() {
                        if let Some(ext) = npath.extension().and_then(|e| e.to_str()) {
                            if ext == "onnx"
                                || ext == "safetensors"
                                || ext == "xgb"
                                || ext == "tree"
                                || (ext == "json" && Self::is_tree_json_file(&npath))
                            {
                                found_files.push(npath);
                            }
                        }
                    }
                }
            }
        }

        // If a single model file is found, assign the folder name as its identifier
        if found_files.len() == 1 {
            if let Some(file_path) = found_files.first() {
                if let Ok(rel) = file_path.strip_prefix(base_dir) {
                    models.push(DiscoveredModel {
                        id: folder_name,
                        relative_path: rel.to_string_lossy().to_string(),
                        absolute_path: file_path.clone(),
                    });
                }
            }
        } else {
            // Multiple weights in directory (e.g. model.onnx and model_quantized.onnx)
            for file_path in found_files {
                let stem = file_path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("model");
                let model_id = if stem == "model" {
                    folder_name.clone()
                } else {
                    format!("{folder_name}-{stem}")
                };

                if let Ok(rel) = file_path.strip_prefix(base_dir) {
                    models.push(DiscoveredModel {
                        id: model_id,
                        relative_path: rel.to_string_lossy().to_string(),
                        absolute_path: file_path,
                    });
                }
            }
        }

        Ok(models)
    }

    /// Searches for a configuration file in the specified directory.
    ///
    /// Checks candidates in priority: `models.toml`, `snooze.toml`, `config.toml`,
    /// `models.json`, and `config.json`.
    #[must_use]
    pub fn find_config_file(dir: &Path) -> Option<PathBuf> {
        let candidates = [
            "models.toml",
            "snooze.toml",
            "config.toml",
            "models.json",
            "config.json",
        ];

        for candidate in candidates {
            let path = dir.join(candidate);
            if path.is_file() {
                return Some(path);
            }
        }

        None
    }

    /// Loads configuration from a specific file path or discovers one in `models_dir`.
    ///
    /// This function is fault-tolerant:
    /// - If the file does not exist, returns `None` without error.
    /// - If the file contains malformed syntax, logs a structured warning and returns `None`.
    pub async fn load_fault_tolerant(
        models_dir: &Path,
        explicit_file: Option<&Path>,
    ) -> Option<Self> {
        let target_path = match explicit_file {
            Some(path) => {
                if path.is_file() {
                    path.to_path_buf()
                } else {
                    tracing::warn!(
                        path = %path.display(),
                        "Explicitly specified config file does not exist; falling back to discovery"
                    );
                    Self::find_config_file(models_dir)?
                }
            }
            None => Self::find_config_file(models_dir)?,
        };

        tracing::info!(path = %target_path.display(), "Loading model configuration file");

        let content = match tokio::fs::read_to_string(&target_path).await {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!(
                    path = %target_path.display(),
                    error = %e,
                    "Failed to read configuration file; falling back to automatic defaults"
                );
                return None;
            }
        };

        match Self::parse_content(&content, &target_path) {
            Ok(config) => Some(config),
            Err(err_msg) => {
                tracing::warn!(
                    path = %target_path.display(),
                    error = %err_msg,
                    "Failed to parse model configuration; falling back to automatic defaults"
                );
                None
            }
        }
    }

    /// Parses configuration content based on file extension (JSON or TOML).
    ///
    /// # Errors
    /// Returns error string describing parsing failure.
    pub fn parse_content(content: &str, file_path: &Path) -> Result<Self, String> {
        if file_path.extension().and_then(|e| e.to_str()) == Some("json") {
            serde_json::from_str::<Self>(content)
                .map_err(|e| format!("Invalid JSON configuration: {e}"))
        } else {
            Self::parse_toml_fault_tolerant(content)
        }
    }

    /// Fault-tolerant TOML parser handling `[defaults]` and `[[models]]` / `[models.<id>]`.
    ///
    /// Ignores malformed non-critical lines while logging warnings, preserving all valid entries.
    ///
    /// # Errors
    /// Returns error string only if parsing cannot yield a valid structure.
    pub fn parse_toml_fault_tolerant(content: &str) -> Result<Self, String> {
        let mut config = Self::default();
        let mut current_model: Option<ModelEntryConfig> = None;
        let mut current_stream: Option<StreamBindingConfig> = None;
        let mut current_api_key: Option<ApiKeyConfig> = None;
        let mut in_defaults = false;
        let mut in_auth = false;
        let mut in_auth_jwt = false;

        for (idx, line) in content.lines().enumerate() {
            let line_num = idx.saturating_add(1);
            let trimmed = line.trim();

            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            if trimmed == "[[models]]" || trimmed == "[[model]]" {
                if let Some(prev) = current_model.take() {
                    config.models.push(prev);
                }
                if let Some(prev) = current_stream.take() {
                    config.streams.push(prev);
                }
                if let Some(prev) = current_api_key.take() {
                    let auth = config.auth.get_or_insert_with(AuthConfig::default);
                    auth.api_keys.push(prev);
                }
                current_model = Some(ModelEntryConfig::default());
                in_defaults = false;
                in_auth = false;
                in_auth_jwt = false;
                continue;
            }

            if trimmed == "[[streams]]" || trimmed == "[[stream]]" {
                if let Some(prev) = current_model.take() {
                    config.models.push(prev);
                }
                if let Some(prev) = current_stream.take() {
                    config.streams.push(prev);
                }
                if let Some(prev) = current_api_key.take() {
                    let auth = config.auth.get_or_insert_with(AuthConfig::default);
                    auth.api_keys.push(prev);
                }
                current_stream = Some(StreamBindingConfig::default());
                in_defaults = false;
                in_auth = false;
                in_auth_jwt = false;
                continue;
            }

            if trimmed == "[defaults]" || trimmed == "[default]" {
                if let Some(prev) = current_model.take() {
                    config.models.push(prev);
                }
                if let Some(prev) = current_stream.take() {
                    config.streams.push(prev);
                }
                if let Some(prev) = current_api_key.take() {
                    let auth = config.auth.get_or_insert_with(AuthConfig::default);
                    auth.api_keys.push(prev);
                }
                in_defaults = true;
                in_auth = false;
                in_auth_jwt = false;
                continue;
            }

            if trimmed == "[auth]" {
                if let Some(prev) = current_model.take() {
                    config.models.push(prev);
                }
                if let Some(prev) = current_stream.take() {
                    config.streams.push(prev);
                }
                if let Some(prev) = current_api_key.take() {
                    let auth = config.auth.get_or_insert_with(AuthConfig::default);
                    auth.api_keys.push(prev);
                }
                config.auth.get_or_insert_with(AuthConfig::default);
                in_defaults = false;
                in_auth = true;
                in_auth_jwt = false;
                continue;
            }

            if trimmed == "[auth.jwt]" || trimmed == "[jwt]" {
                if let Some(prev) = current_model.take() {
                    config.models.push(prev);
                }
                if let Some(prev) = current_stream.take() {
                    config.streams.push(prev);
                }
                if let Some(prev) = current_api_key.take() {
                    let auth = config.auth.get_or_insert_with(AuthConfig::default);
                    auth.api_keys.push(prev);
                }
                let auth = config.auth.get_or_insert_with(AuthConfig::default);
                auth.jwt.get_or_insert_with(JwtConfig::default);
                in_defaults = false;
                in_auth = false;
                in_auth_jwt = true;
                continue;
            }

            if trimmed == "[[auth.api_keys]]"
                || trimmed == "[[auth.api_key]]"
                || trimmed == "[[api_keys]]"
            {
                if let Some(prev) = current_model.take() {
                    config.models.push(prev);
                }
                if let Some(prev) = current_stream.take() {
                    config.streams.push(prev);
                }
                if let Some(prev) = current_api_key.take() {
                    let auth = config.auth.get_or_insert_with(AuthConfig::default);
                    auth.api_keys.push(prev);
                }
                config.auth.get_or_insert_with(AuthConfig::default);
                current_api_key = Some(ApiKeyConfig {
                    key: None,
                    key_env: None,
                    role: Role::Inference,
                    name: None,
                });
                in_defaults = false;
                in_auth = false;
                in_auth_jwt = false;
                continue;
            }

            if trimmed.starts_with("[models.") && trimmed.ends_with(']') {
                if let Some(prev) = current_model.take() {
                    config.models.push(prev);
                }
                if let Some(prev) = current_stream.take() {
                    config.streams.push(prev);
                }
                if let Some(prev) = current_api_key.take() {
                    let auth = config.auth.get_or_insert_with(AuthConfig::default);
                    auth.api_keys.push(prev);
                }
                let id = trimmed
                    .trim_start_matches("[models.")
                    .trim_end_matches(']')
                    .trim_matches('"')
                    .trim_matches('\'')
                    .to_string();

                let new_entry = ModelEntryConfig {
                    id,
                    ..Default::default()
                };
                current_model = Some(new_entry);
                in_defaults = false;
                in_auth = false;
                in_auth_jwt = false;
                continue;
            }

            if trimmed.starts_with("[streams.") && trimmed.ends_with(']') {
                if let Some(prev) = current_model.take() {
                    config.models.push(prev);
                }
                if let Some(prev) = current_stream.take() {
                    config.streams.push(prev);
                }
                if let Some(prev) = current_api_key.take() {
                    let auth = config.auth.get_or_insert_with(AuthConfig::default);
                    auth.api_keys.push(prev);
                }
                let name = trimmed
                    .trim_start_matches("[streams.")
                    .trim_end_matches(']')
                    .trim_matches('"')
                    .trim_matches('\'')
                    .to_string();

                let new_entry = StreamBindingConfig {
                    name,
                    ..Default::default()
                };
                current_stream = Some(new_entry);
                in_defaults = false;
                in_auth = false;
                in_auth_jwt = false;
                continue;
            }

            // Key = Value assignment
            if let Some((raw_key, raw_val)) = trimmed.split_once('=') {
                let key = raw_key.trim().to_lowercase();
                let val_str = raw_val.trim();

                // Strip inline comment if value is quoted or simple token
                let val = Self::clean_toml_value(val_str);

                if in_defaults {
                    match key.as_str() {
                        "idle_timeout_secs" | "idle_timeout" => {
                            if let Ok(v) = val.parse::<u64>() {
                                config.defaults.idle_timeout_secs = Some(v);
                            }
                        }
                        "max_batch_size" => {
                            if let Ok(v) = val.parse::<usize>() {
                                config.defaults.max_batch_size = Some(v);
                            }
                        }
                        "max_batch_delay_ms" | "max_delay_ms" => {
                            if let Ok(v) = val.parse::<u64>() {
                                config.defaults.max_batch_delay_ms = Some(v);
                            }
                        }
                        "device" | "provider" => {
                            config.defaults.device = Some(val);
                        }
                        "max_loaded_models" => {
                            if let Ok(v) = val.parse::<usize>() {
                                config.defaults.max_loaded_models = Some(v);
                            }
                        }
                        "min_residency_secs" | "min_residency" => {
                            if let Ok(v) = val.parse::<u64>() {
                                config.defaults.min_residency_secs = Some(v);
                            }
                        }
                        "autoscale_queue_threshold" => {
                            if let Ok(v) = val.parse::<usize>() {
                                config.defaults.autoscale_queue_threshold = Some(v);
                            }
                        }
                        "max_replicas" => {
                            if let Ok(v) = val.parse::<usize>() {
                                config.defaults.max_replicas = Some(v);
                            }
                        }
                        "routing_strategy" => {
                            config.defaults.routing_strategy = Some(val);
                        }
                        "pinned" | "pin" | "preload" => {
                            if let Some(b) = Self::parse_bool_str(&val) {
                                config.defaults.pinned = Some(b);
                            }
                        }
                        other => {
                            tracing::debug!(line = line_num, key = %other, "Unrecognized key in [defaults]");
                        }
                    }
                } else if let Some(ref mut model) = current_model {
                    match key.as_str() {
                        "id" => model.id = val,
                        "path" => model.path = val,
                        "description" => model.description = Some(val),
                        "device" | "provider" => model.device = Some(val),
                        "max_batch_size" => {
                            if let Ok(v) = val.parse::<usize>() {
                                model.max_batch_size = Some(v);
                            }
                        }
                        "max_batch_delay_ms" | "max_delay_ms" => {
                            if let Ok(v) = val.parse::<u64>() {
                                model.max_batch_delay_ms = Some(v);
                            }
                        }
                        "idle_timeout_secs" | "idle_timeout" => {
                            if let Ok(v) = val.parse::<u64>() {
                                model.idle_timeout_secs = Some(v);
                            }
                        }
                        "min_residency_secs" | "min_residency" => {
                            if let Ok(v) = val.parse::<u64>() {
                                model.min_residency_secs = Some(v);
                            }
                        }
                        "autoscale_queue_threshold" => {
                            if let Ok(v) = val.parse::<usize>() {
                                model.autoscale_queue_threshold = Some(v);
                            }
                        }
                        "max_replicas" => {
                            if let Ok(v) = val.parse::<usize>() {
                                model.max_replicas = Some(v);
                            }
                        }
                        "routing_strategy" => {
                            model.routing_strategy = Some(val);
                        }
                        "pinned" | "pin" | "preload" => {
                            if let Some(b) = Self::parse_bool_str(&val) {
                                model.pinned = Some(b);
                            }
                        }
                        other => {
                            tracing::debug!(line = line_num, key = %other, "Unrecognized key in model entry");
                        }
                    }
                } else if let Some(ref mut stream) = current_stream {
                    match key.as_str() {
                        "name" => stream.name = val,
                        "broker" => stream.broker = val,
                        "endpoint" | "url" | "bootstrap_servers" | "address" => {
                            stream.endpoint = val
                        }
                        "model_id" | "model" => stream.model_id = Some(val),
                        "input_topic" | "topic" | "queue" | "input" | "stream" => {
                            stream.input_topic = val
                        }
                        "output_topic" | "out" | "output" => stream.output_topic = Some(val),
                        "error_topic" | "dlq" | "error" => stream.error_topic = Some(val),
                        "consumer_group" | "group" => stream.consumer_group = Some(val),
                        "auto_create_topics" | "auto_create" => {
                            if let Some(b) = Self::parse_bool_str(&val) {
                                stream.auto_create_topics = b;
                            }
                        }
                        "concurrency" | "workers" => {
                            if let Ok(v) = val.parse::<usize>() {
                                stream.concurrency = v;
                            }
                        }
                        "partitions" => {
                            if let Ok(v) = val.parse::<usize>() {
                                stream.partitions = v;
                            }
                        }
                        "format" => match val.to_lowercase().as_str() {
                            "protobuf" | "proto" => stream.format = StreamWireFormat::Protobuf,
                            _ => stream.format = StreamWireFormat::Json,
                        },
                        other => {
                            tracing::debug!(line = line_num, key = %other, "Unrecognized key in stream entry");
                        }
                    }
                } else if in_auth {
                    let auth = config.auth.get_or_insert_with(AuthConfig::default);
                    match key.as_str() {
                        "enabled" | "auth_enabled" => {
                            if let Some(b) = Self::parse_bool_str(&val) {
                                auth.enabled = b;
                            }
                        }
                        "protect_metrics" => {
                            if let Some(b) = Self::parse_bool_str(&val) {
                                auth.protect_metrics = b;
                            }
                        }
                        other => {
                            tracing::debug!(line = line_num, key = %other, "Unrecognized key in [auth]");
                        }
                    }
                } else if in_auth_jwt {
                    let auth = config.auth.get_or_insert_with(AuthConfig::default);
                    let jwt = auth.jwt.get_or_insert_with(JwtConfig::default);
                    match key.as_str() {
                        "secret" => jwt.secret = Some(val),
                        "secret_env" => jwt.secret_env = Some(val),
                        "algorithm" | "alg" => match val.to_uppercase().as_str() {
                            "HS384" => jwt.algorithm = JwtAlgorithm::HS384,
                            "HS512" => jwt.algorithm = JwtAlgorithm::HS512,
                            _ => jwt.algorithm = JwtAlgorithm::HS256,
                        },
                        "issuer" | "iss" => jwt.issuer = Some(val),
                        "audience" | "aud" => jwt.audience = Some(val),
                        "leeway_secs" | "leeway" => {
                            if let Ok(v) = val.parse::<u64>() {
                                jwt.leeway_secs = v;
                            }
                        }
                        other => {
                            tracing::debug!(line = line_num, key = %other, "Unrecognized key in [auth.jwt]");
                        }
                    }
                } else if let Some(ref mut key_cfg) = current_api_key {
                    match key.as_str() {
                        "key" => key_cfg.key = Some(val),
                        "key_env" => key_cfg.key_env = Some(val),
                        "role" => key_cfg.role = Role::parse_str(&val),
                        "name" => key_cfg.name = Some(val),
                        other => {
                            tracing::debug!(line = line_num, key = %other, "Unrecognized key in [[auth.api_keys]]");
                        }
                    }
                } else {
                    tracing::debug!(line = line_num, "Key-value pair outside section; ignoring");
                }
            }
        }

        if let Some(prev) = current_model {
            config.models.push(prev);
        }
        if let Some(prev) = current_stream {
            config.streams.push(prev);
        }
        if let Some(prev) = current_api_key {
            let auth = config.auth.get_or_insert_with(AuthConfig::default);
            auth.api_keys.push(prev);
        }

        Ok(config)
    }

    /// Parses a boolean string representation (`"true"`, `"1"`, `"yes"`, `"false"`, `"0"`, `"no"`).
    fn parse_bool_str(val: &str) -> Option<bool> {
        match val.trim().to_ascii_lowercase().as_str() {
            "true" | "1" | "yes" | "on" => Some(true),
            "false" | "0" | "no" | "off" => Some(false),
            _ => None,
        }
    }

    /// Strips enclosing quotes and trailing inline comments from a TOML value.
    fn clean_toml_value(raw: &str) -> String {
        let trimmed = raw.trim();
        if trimmed.starts_with('"') {
            // Find terminating quote
            if let Some(closing) = trimmed.get(1..).and_then(|s| s.find('"')) {
                let quote_end = closing.saturating_add(1);
                return trimmed.get(1..quote_end).unwrap_or("").to_string();
            }
        } else if trimmed.starts_with('\'') {
            if let Some(closing) = trimmed.get(1..).and_then(|s| s.find('\'')) {
                let quote_end = closing.saturating_add(1);
                return trimmed.get(1..quote_end).unwrap_or("").to_string();
            }
        }

        // Unquoted scalar (e.g. integer or identifier): strip inline comment if present
        let before_comment = trimmed.split('#').next().unwrap_or(trimmed).trim();
        before_comment.to_string()
    }

    /// Resolves the filesystem path for a configured model path.
    ///
    /// Checks:
    /// 1. Absolute path.
    /// 2. Relative to `models_dir`.
    /// 3. Relative to current working directory.
    #[must_use]
    pub fn resolve_path(models_dir: &Path, configured_path: &str) -> PathBuf {
        let raw = Path::new(configured_path);
        if raw.is_absolute() {
            return raw.to_path_buf();
        }

        let from_models_dir = models_dir.join(raw);
        if from_models_dir.exists() {
            return from_models_dir;
        }

        if raw.exists() {
            return raw.to_path_buf();
        }

        from_models_dir
    }

    /// Normalizes a device or provider string into a supported engine identifier.
    ///
    /// Recognizes `"auto"`, maps `"gpu"` to `"cuda"`, maps `"mlx"`, `"metal"`, `"coreml"` to `"mlx"`,
    /// and standardizes `"cuda"`, `"tensorrt"`, `"candle"`, and `"cpu"`.
    #[must_use]
    pub fn normalize_device(dev: Option<&str>) -> Option<String> {
        let d = dev?.trim().to_lowercase();
        match d.as_str() {
            "auto" => Some("auto".to_string()),
            "gpu" | "cuda" => Some("cuda".to_string()),
            "mlx" | "metal" | "coreml" => Some("mlx".to_string()),
            "tensorrt" => Some("tensorrt".to_string()),
            "candle" | "safetensors" => Some("candle".to_string()),
            "cpu" => Some("cpu".to_string()),
            other => Some(other.to_string()),
        }
    }

    /// Parses an optional routing strategy string into a [`crate::routing::RoutingStrategyType`].
    #[must_use]
    pub fn parse_routing_strategy(s: Option<&str>) -> crate::routing::RoutingStrategyType {
        match s.map(|val| val.trim().to_lowercase()).as_deref() {
            Some("round_robin") | Some("roundrobin") => {
                crate::routing::RoutingStrategyType::RoundRobin
            }
            Some("random") => crate::routing::RoutingStrategyType::Random,
            _ => crate::routing::RoutingStrategyType::LeastQueue,
        }
    }

    /// Converts the configured entries and unconfigured discovered models into a list
    /// of ready-to-run [`ModelConfig`] instances.
    pub fn build_model_configs(
        &self,
        models_dir: &Path,
        discovered_on_disk: &[DiscoveredModel],
    ) -> Vec<ModelConfig> {
        let mut result = Vec::new();
        let mut registered_ids: HashSet<String> = HashSet::new();
        let mut registered_paths: HashSet<PathBuf> = HashSet::new();

        let default_idle = Duration::from_secs(
            self.defaults
                .idle_timeout_secs
                .unwrap_or(DEFAULT_IDLE_TIMEOUT_SECS),
        );
        let default_batch_size = self
            .defaults
            .max_batch_size
            .unwrap_or(DEFAULT_MAX_BATCH_SIZE);
        let default_delay = Duration::from_millis(
            self.defaults
                .max_batch_delay_ms
                .unwrap_or(DEFAULT_MAX_BATCH_DELAY_MS),
        );
        let default_min_residency = Duration::from_secs(
            self.defaults
                .min_residency_secs
                .unwrap_or(DEFAULT_MIN_RESIDENCY_SECS),
        );
        let default_provider = Self::normalize_device(self.defaults.device.as_deref())
            .or_else(|| Some("auto".to_string()));
        let default_pinned = self.defaults.pinned.unwrap_or(false);

        // 1. Process explicitly configured models
        for entry in &self.models {
            if entry.id.is_empty() || entry.path.is_empty() {
                tracing::warn!(entry = ?entry, "Skipping invalid model entry with empty id or path");
                continue;
            }

            let resolved_path = Self::resolve_path(models_dir, &entry.path);
            if !resolved_path.exists() {
                tracing::warn!(
                    model_id = %entry.id,
                    path = %resolved_path.display(),
                    "Configured model weight file not found on disk; skipping registration"
                );
                continue;
            }

            let idle_timeout = entry
                .idle_timeout_secs
                .map(Duration::from_secs)
                .unwrap_or(default_idle);
            let max_batch_size = entry.max_batch_size.unwrap_or(default_batch_size);
            let max_batch_delay = entry
                .max_batch_delay_ms
                .map(Duration::from_millis)
                .unwrap_or(default_delay);
            let min_residency = entry
                .min_residency_secs
                .map(Duration::from_secs)
                .unwrap_or(default_min_residency);
            let provider = match entry.device.as_deref() {
                Some(d) => Self::normalize_device(Some(d)),
                None => default_provider.clone(),
            };
            let pinned = entry.pinned.unwrap_or(default_pinned);

            tracing::info!(
                model_id = %entry.id,
                path = %resolved_path.display(),
                provider = ?provider,
                batch_size = max_batch_size,
                pinned = pinned,
                "Registering configured model"
            );

            registered_ids.insert(entry.id.clone());
            if let Ok(canon) = resolved_path.canonicalize() {
                registered_paths.insert(canon);
            }
            registered_paths.insert(resolved_path.clone());

            let autoscale_queue_threshold = entry
                .autoscale_queue_threshold
                .or(self.defaults.autoscale_queue_threshold)
                .unwrap_or(max_batch_size);
            let max_replicas = match entry.max_replicas {
                Some(0) => None, // 0 explicitly unbinds the replica limit
                Some(n) => Some(n),
                None => self.defaults.max_replicas.filter(|&n| n > 0),
            };
            let routing_strategy = Self::parse_routing_strategy(
                entry
                    .routing_strategy
                    .as_deref()
                    .or(self.defaults.routing_strategy.as_deref()),
            );

            result.push(ModelConfig {
                id: entry.id.clone(),
                onnx_path: resolved_path.to_string_lossy().to_string(),
                idle_timeout,
                max_batch_size,
                max_batch_delay,
                provider,
                description: entry.description.clone(),
                min_residency,
                autoscale_queue_threshold,
                max_replicas,
                routing_strategy,
                pinned,
            });
        }

        // 2. Discover unconfigured models on disk (fault tolerance / ease of use)
        let default_autoscale_threshold = self
            .defaults
            .autoscale_queue_threshold
            .unwrap_or(default_batch_size);
        let default_max_replicas = self.defaults.max_replicas.filter(|&n| n > 0);
        let default_routing =
            Self::parse_routing_strategy(self.defaults.routing_strategy.as_deref());

        for disc in discovered_on_disk {
            let canon_disc = disc
                .absolute_path
                .canonicalize()
                .unwrap_or_else(|_| disc.absolute_path.clone());
            let is_already_registered = registered_ids.contains(&disc.id)
                || registered_paths.contains(&canon_disc)
                || registered_paths.contains(&disc.absolute_path);

            if !is_already_registered {
                tracing::info!(
                    model_id = %disc.id,
                    path = %disc.relative_path,
                    "Discovered unconfigured model on disk; registering with default settings"
                );

                registered_ids.insert(disc.id.clone());
                registered_paths.insert(canon_disc);
                registered_paths.insert(disc.absolute_path.clone());

                result.push(ModelConfig {
                    id: disc.id.clone(),
                    onnx_path: disc.absolute_path.to_string_lossy().to_string(),
                    idle_timeout: default_idle,
                    max_batch_size: default_batch_size,
                    max_batch_delay: default_delay,
                    provider: default_provider.clone(),
                    description: None,
                    min_residency: default_min_residency,
                    autoscale_queue_threshold: default_autoscale_threshold,
                    max_replicas: default_max_replicas,
                    routing_strategy: default_routing,
                    pinned: default_pinned,
                });
            }
        }

        result
    }

    /// Generates a well-commented TOML configuration string containing defaults and model entries.
    #[must_use]
    pub fn to_toml_string(&self) -> String {
        let mut out = String::new();
        out.push_str("# SnoozeML Model Directory Configuration\n");
        out.push_str("# Auto-generated configuration for SnoozeML inference daemon.\n");
        out.push_str("# This file defines global execution defaults and per-model settings (devices, batching, descriptions).\n\n");

        out.push_str("[defaults]\n");
        out.push_str(&format!(
            "idle_timeout_secs = {}\n",
            self.defaults
                .idle_timeout_secs
                .unwrap_or(DEFAULT_IDLE_TIMEOUT_SECS)
        ));
        out.push_str(&format!(
            "max_batch_size = {}\n",
            self.defaults
                .max_batch_size
                .unwrap_or(DEFAULT_MAX_BATCH_SIZE)
        ));
        out.push_str(&format!(
            "max_batch_delay_ms = {}\n",
            self.defaults
                .max_batch_delay_ms
                .unwrap_or(DEFAULT_MAX_BATCH_DELAY_MS)
        ));
        out.push_str(&format!(
            "device = \"{}\"\n",
            self.defaults.device.as_deref().unwrap_or(DEFAULT_DEVICE)
        ));
        if let Some(max_m) = self.defaults.max_loaded_models {
            out.push_str(&format!("max_loaded_models = {max_m}\n"));
        } else {
            out.push_str("# max_loaded_models = 3\n");
        }
        if let Some(pinned) = self.defaults.pinned {
            out.push_str(&format!("pinned = {pinned}\n"));
        }
        out.push_str(&format!(
            "min_residency_secs = {}\n\n",
            self.defaults
                .min_residency_secs
                .unwrap_or(DEFAULT_MIN_RESIDENCY_SECS)
        ));

        if self.models.is_empty() {
            out.push_str(
                "# Sample model configuration stub (uncomment and customize as needed):\n",
            );
            out.push_str("# [[models]]\n");
            out.push_str("# id = \"sample-model\"\n");
            out.push_str("# path = \"sample-model.onnx\"\n");
            out.push_str(&format!("# description = \"{STUB_DESCRIPTION}\"\n"));
            out.push_str("# device = \"auto\"\n");
            out.push_str("# max_batch_size = 16\n");
            out.push_str("# max_batch_delay_ms = 10\n");
            out.push_str("# idle_timeout_secs = 300\n");
            out.push_str("# min_residency_secs = 10\n");
            out.push_str("# pinned = false\n");
        } else {
            for m in &self.models {
                out.push_str("[[models]]\n");
                out.push_str(&format!("id = \"{}\"\n", m.id));
                out.push_str(&format!("path = \"{}\"\n", m.path));
                if let Some(ref desc) = m.description {
                    out.push_str(&format!("description = \"{desc}\"\n"));
                } else {
                    out.push_str(&format!("description = \"{STUB_DESCRIPTION}\"\n"));
                }
                out.push_str(&format!(
                    "device = \"{}\"\n",
                    m.device.as_deref().unwrap_or(DEFAULT_DEVICE)
                ));
                if let Some(bs) = m.max_batch_size {
                    out.push_str(&format!("max_batch_size = {bs}\n"));
                }
                if let Some(d) = m.max_batch_delay_ms {
                    out.push_str(&format!("max_batch_delay_ms = {d}\n"));
                }
                if let Some(t) = m.idle_timeout_secs {
                    out.push_str(&format!("idle_timeout_secs = {t}\n"));
                }
                if let Some(pinned) = m.pinned {
                    out.push_str(&format!("pinned = {pinned}\n"));
                }
                out.push('\n');
            }
        }

        out
    }

    /// Generates a stub model entry for a newly discovered model file.
    #[must_use]
    pub fn create_stub_for_discovered(disc: &DiscoveredModel) -> ModelEntryConfig {
        ModelEntryConfig {
            id: disc.id.clone(),
            path: disc.relative_path.clone(),
            description: Some(STUB_DESCRIPTION.to_string()),
            device: Some(DEFAULT_DEVICE.to_string()),
            max_batch_size: Some(DEFAULT_MAX_BATCH_SIZE),
            max_batch_delay_ms: Some(DEFAULT_MAX_BATCH_DELAY_MS),
            idle_timeout_secs: Some(DEFAULT_IDLE_TIMEOUT_SECS),
            min_residency_secs: Some(DEFAULT_MIN_RESIDENCY_SECS),
            autoscale_queue_threshold: None,
            max_replicas: None,
            routing_strategy: None,
            pinned: None,
        }
    }

    /// Synchronizes the models directory with a configuration file:
    /// - If no configuration file exists, creates `models.toml` with discovered model stubs.
    /// - If a configuration file exists, appends new stubs for any unconfigured models.
    ///
    /// Returns a tuple containing the target config path and the count of newly added model stubs.
    ///
    /// # Errors
    /// Returns [`std::io::Error`] if files cannot be read or written.
    pub async fn init_or_update_dir(
        models_dir: &Path,
        custom_file: Option<&Path>,
    ) -> Result<(PathBuf, usize), std::io::Error> {
        let discovered = Self::discover_models_in_dir(models_dir).await?;

        let target_path = match custom_file {
            Some(p) => p.to_path_buf(),
            None => match Self::find_config_file(models_dir) {
                Some(existing) => existing,
                None => models_dir.join("models.toml"),
            },
        };

        if target_path.is_file() {
            // Merge newly discovered models into existing config
            let existing_content = tokio::fs::read_to_string(&target_path).await?;
            let existing_config = Self::parse_content(&existing_content, &target_path)
                .unwrap_or_else(|_| Self::default());

            let mut known_ids: HashSet<String> = HashSet::new();
            let mut known_paths: HashSet<String> = HashSet::new();

            for m in &existing_config.models {
                known_ids.insert(m.id.clone());
                known_paths.insert(m.path.clone());
            }

            let mut new_stubs = Vec::new();
            for disc in &discovered {
                if !known_ids.contains(&disc.id) && !known_paths.contains(&disc.relative_path) {
                    new_stubs.push(Self::create_stub_for_discovered(disc));
                }
            }

            if new_stubs.is_empty() {
                return Ok((target_path, 0));
            }

            let added_count = new_stubs.len();
            let mut updated_content = existing_content;
            if !updated_content.ends_with('\n') {
                updated_content.push('\n');
            }

            for stub in new_stubs {
                updated_content.push_str(&format!(
                    "\n[[models]]\nid = \"{}\"\npath = \"{}\"\ndescription = \"{}\"\ndevice = \"{}\"\nmax_batch_size = {}\nmax_batch_delay_ms = {}\nidle_timeout_secs = {}\n",
                    stub.id,
                    stub.path,
                    stub.description.as_deref().unwrap_or(STUB_DESCRIPTION),
                    stub.device.as_deref().unwrap_or(DEFAULT_DEVICE),
                    stub.max_batch_size.unwrap_or(DEFAULT_MAX_BATCH_SIZE),
                    stub.max_batch_delay_ms.unwrap_or(DEFAULT_MAX_BATCH_DELAY_MS),
                    stub.idle_timeout_secs.unwrap_or(DEFAULT_IDLE_TIMEOUT_SECS),
                ));
            }

            tokio::fs::write(&target_path, updated_content.as_bytes()).await?;
            Ok((target_path, added_count))
        } else {
            // Generate brand new config
            let mut config = Self::default();
            for disc in &discovered {
                config.models.push(Self::create_stub_for_discovered(disc));
            }

            let toml_str = config.to_toml_string();
            if let Some(parent) = target_path.parent() {
                tokio::fs::create_dir_all(parent).await?;
            }
            tokio::fs::write(&target_path, toml_str.as_bytes()).await?;
            let added_count = config.models.len();
            Ok((target_path, added_count))
        }
    }
}
