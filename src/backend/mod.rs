//! Model execution backend abstractions and implementations.

#[cfg(feature = "candle-backend")]
pub mod candle_backend;
#[cfg(feature = "candle-backend")]
pub mod local_loader;
pub mod mock_backend;
pub mod ort_backend;

#[cfg(feature = "tree-backend")]
pub mod tree_backend;
#[cfg(feature = "worker-backend")]
pub mod worker_backend;

#[cfg(feature = "candle-backend")]
pub use candle_backend::{CandleBackend, CandleConfig, CandleDevice};
#[cfg(feature = "candle-backend")]
pub use local_loader::{LoadedModel, LocalModelLoader, SupportedArchitecture};
pub use mock_backend::MockBackend;
#[cfg(feature = "ort-backend")]
pub use ort_backend::{ExecutionProvider, OrtBackend, OrtConfig};
#[cfg(feature = "tree-backend")]
pub use tree_backend::{TreeBackend, TreeConfig, TreeDevice, TreePredictType};
#[cfg(feature = "worker-backend")]
pub use worker_backend::{WorkerBackend, WorkerConfig};

use crate::types::{InferenceError, TensorPayload};

/// Evaluates whether a given model path or provider preference corresponds to a tree model.
fn is_tree_model(model_path: &str, provider: Option<&str>) -> bool {
    if matches!(provider, Some("tree" | "tree-cpu" | "xgboost" | "gbdt")) {
        return true;
    }
    if model_path.ends_with(".xgb")
        || model_path.ends_with(".xgb.json")
        || model_path.ends_with(".tree")
        || model_path.ends_with(".tree.json")
    {
        return true;
    }
    if model_path.ends_with(".json") {
        let path = std::path::Path::new(model_path);
        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            if name == "config.json" || name == "tokenizer.json" || name == "generation_config.json"
            {
                return false;
            }
        }
        if let Ok(mut file) = std::fs::File::open(path) {
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
    }
    false
}

/// Trait defining the interface for an underlying model execution engine.
///
/// Designed to support multiple runtime providers such as ONNX Runtime (`ort`),
/// Safetensors / Candle, Tree ensembles (`hessboost` / XGBoost), and mock engines for testing.
pub trait ModelBackend: Send + Sync {
    /// Executes batched inference on the provided input tensors.
    ///
    /// # Errors
    /// Returns [`InferenceError::Runtime`] or [`InferenceError::ShapeMismatch`] if
    /// tensor shapes are incompatible or the underlying runtime encounters an error.
    fn infer(&self, inputs: &[TensorPayload]) -> Result<Vec<TensorPayload>, InferenceError>;

    /// Returns the provider name identifier (e.g. `"onnxruntime-cpu"`, `"tree-cpu"`, `"mock-engine"`).
    fn provider_name(&self) -> &'static str;
}

/// Factory helper to initialize a backend for a given model path.
///
/// Automatically selects the appropriate backend engine based on the file extension
/// (`.safetensors` for Candle, `.onnx` for ONNX Runtime, `.json`/`.xgb` for Tree) or provider preferences.
///
/// # Errors
/// Returns [`InferenceError::NotFound`] or [`InferenceError::Runtime`] if initialization fails.
pub fn create_backend(
    model_path: &str,
    provider_preference: Option<&str>,
) -> Result<Box<dyn ModelBackend>, InferenceError> {
    if model_path.starts_with("mock:") || model_path == "mock" {
        if model_path == "mock:oom" {
            return Err(InferenceError::OutOfMemory(
                "Mock allocation failed: CUDA failure 2: out of memory".to_string(),
            ));
        }
        if model_path == "mock:bad_alloc" {
            return Err(InferenceError::Runtime(
                "Mock allocation failed: std::bad_alloc: out of host memory".to_string(),
            ));
        }
        if model_path == "mock:corrupted" {
            return Err(InferenceError::Runtime(
                "Corrupted weights header: unexpected end of file".to_string(),
            ));
        }
        if (model_path == "mock:oom_configurable" || model_path == "mock:oom_once")
            && mock_backend::check_and_consume_mock_oom()
        {
            return Err(InferenceError::OutOfMemory(
                "Mock allocation failed: CUDA failure 2: out of memory".to_string(),
            ));
        }

        let provider = match provider_preference {
            Some("gpu" | "cuda") => "mock-gpu",
            Some("mlx" | "metal" | "coreml") => "mock-mlx",
            Some("cpu") => "mock-cpu",
            Some("tensorrt") => "mock-tensorrt",
            Some("auto") | None => {
                if cfg!(feature = "cuda") {
                    "mock-gpu"
                } else if cfg!(target_os = "macos") {
                    "mock-mlx"
                } else {
                    "mock-cpu"
                }
            }
            Some(_) => "mock-engine",
        };
        return Ok(Box::new(MockBackend::with_provider(provider)));
    }

    #[cfg(feature = "worker-backend")]
    if worker_backend::is_worker_model(model_path, provider_preference) {
        let config = worker_backend::WorkerConfig::from_path(model_path, provider_preference);
        let backend = worker_backend::WorkerBackend::new(&config)?;
        return Ok(Box::new(backend));
    }

    #[cfg(not(feature = "worker-backend"))]
    {
        let path_obj = std::path::Path::new(model_path);
        let is_worker = matches!(provider_preference, Some("python" | "worker"))
            || (path_obj.is_dir()
                && (path_obj.join("snooze_model.py").is_file()
                    || path_obj.join("snooze_model.json").is_file()))
            || (path_obj.is_file()
                && (path_obj.file_name().and_then(|n| n.to_str()) == Some("snooze_model.py")
                    || path_obj.file_name().and_then(|n| n.to_str()) == Some("snooze_model.json")));
        if is_worker {
            return Err(InferenceError::Runtime(format!(
                "Worker backend feature ('worker-backend') is disabled and model path '{model_path}' requires it"
            )));
        }
    }

    if is_tree_model(model_path, provider_preference) {
        #[cfg(feature = "tree-backend")]
        {
            let config = tree_backend::TreeConfig {
                model_path: model_path.to_string(),
                device: tree_backend::TreeDevice::Cpu,
                input_name: None,
                output_name: None,
                predict_type: tree_backend::TreePredictType::Value,
            };
            let backend = TreeBackend::new(&config)?;
            return Ok(Box::new(backend));
        }

        #[cfg(not(feature = "tree-backend"))]
        {
            return Err(InferenceError::Runtime(format!(
                "Tree backend feature ('tree-backend') is disabled and model path '{model_path}' requires it"
            )));
        }
    }

    let path_obj = std::path::Path::new(model_path);
    let is_safetensors = model_path.ends_with(".safetensors")
        || matches!(provider_preference, Some("candle" | "safetensors"))
        || (path_obj.is_dir()
            && (path_obj.join("model.safetensors").is_file()
                || path_obj.join("config.json").is_file()));

    if is_safetensors {
        #[cfg(feature = "candle-backend")]
        {
            let device = match provider_preference {
                Some("cuda" | "gpu") => candle_backend::CandleDevice::Cuda(0),
                Some("mlx" | "metal" | "coreml") => candle_backend::CandleDevice::Mlx,
                Some("cpu") => candle_backend::CandleDevice::Cpu,
                Some("auto") | None => candle_backend::CandleDevice::Auto,
                _ => candle_backend::CandleDevice::Auto,
            };
            let config = candle_backend::CandleConfig {
                model_path: model_path.to_string(),
                device,
            };
            let backend = CandleBackend::new(&config)?;
            return Ok(Box::new(backend));
        }

        #[cfg(not(feature = "candle-backend"))]
        {
            return Err(InferenceError::Runtime(format!(
                "Candle / Safetensors backend feature ('candle-backend') is disabled and model path '{model_path}' requires it"
            )));
        }
    }

    #[cfg(feature = "ort-backend")]
    {
        let provider = match provider_preference {
            Some("cuda" | "gpu") => ort_backend::ExecutionProvider::Cuda,
            Some("tensorrt") => ort_backend::ExecutionProvider::TensorRt,
            Some("mlx" | "metal" | "coreml") => ort_backend::ExecutionProvider::CoreML,
            Some("cpu") => ort_backend::ExecutionProvider::Cpu,
            Some("auto") | None => ort_backend::ExecutionProvider::Auto,
            _ => ort_backend::ExecutionProvider::Auto,
        };
        let config = ort_backend::OrtConfig {
            model_path: model_path.to_string(),
            provider,
            intra_threads: 1,
            inter_threads: 1,
        };
        let ort_backend = OrtBackend::new(&config)?;
        Ok(Box::new(ort_backend))
    }

    #[cfg(not(feature = "ort-backend"))]
    {
        Err(InferenceError::Runtime(format!(
            "ONNX backend feature is disabled and model path '{model_path}' is not a mock or safetensors backend"
        )))
    }
}
