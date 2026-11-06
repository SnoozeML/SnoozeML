//! ONNX Runtime backend implementation using `ort` 2.0.

#[cfg(feature = "ort-backend")]
use {
    crate::backend::ModelBackend,
    crate::types::{InferenceError, TensorPayload},
    ort::{
        environment::Environment,
        ep::{self, ExecutionProvider as OrtExecutionProvider},
        session::Session,
        value::Value,
    },
    std::path::{Path, PathBuf},
    std::sync::Arc,
};

/// Hardware execution provider selection for ONNX Runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExecutionProvider {
    /// Automatically select the fastest available hardware accelerator:
    /// tries GPU (CUDA), then MLX / CoreML (on macOS), lastly defaulting to CPU.
    #[default]
    Auto,
    /// CPU Execution Provider with OpenMP / oneDNN threads.
    Cpu,
    /// NVIDIA CUDA Execution Provider.
    Cuda,
    /// NVIDIA TensorRT Execution Provider.
    TensorRt,
    /// Apple CoreML / Metal Execution Provider (macOS Apple Silicon).
    CoreML,
}

/// Configuration options for initializing an ONNX Runtime session.
#[derive(Debug, Clone)]
pub struct OrtConfig {
    /// Filesystem path to the ONNX model file.
    pub model_path: String,
    /// Target execution provider (Auto, CPU, CUDA, TensorRT, CoreML).
    pub provider: ExecutionProvider,
    /// Number of intra-operator compute threads.
    pub intra_threads: usize,
    /// Number of inter-operator compute threads.
    pub inter_threads: usize,
}

impl Default for OrtConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            provider: ExecutionProvider::Auto,
            intra_threads: 1,
            inter_threads: 1,
        }
    }
}

/// ONNX Runtime shared environment and dynamic loader wrapper.
#[cfg(feature = "ort-backend")]
pub struct OrtRuntime {
    env: Arc<Environment>,
}

#[cfg(feature = "ort-backend")]
impl OrtRuntime {
    /// Initializes the ONNX Runtime Environment by dynamically loading `libonnxruntime`
    /// from `ORT_DYLIB_PATH`, a bundled `./lib` directory next to the binary, or the system path.
    ///
    /// # Errors
    /// Returns an error string if loading the dynamic library or building the environment fails.
    pub fn new() -> Result<Self, String> {
        let env = if let Some(dylib_path) = resolve_ort_dylib_path() {
            tracing::info!("Loading ONNX Runtime dynamically from {:?}", dylib_path);
            ort::init_from(&dylib_path)
                .map_err(|e| format!("Failed to load ORT dylib at {dylib_path:?}: {e}"))?
                .with_name("snooze")
                .build()
                .map_err(|e| format!("Failed to build ORT environment: {e}"))?
        } else {
            tracing::info!(
                "ORT_DYLIB_PATH and bundled lib not found; falling back to system libonnxruntime"
            );
            ort::init()
                .with_name("snooze")
                .build()
                .map_err(|e| format!("Failed to build default ORT environment: {e}"))?
        };

        Ok(Self { env: Arc::new(env) })
    }

    /// Loads an ONNX model session with automatic CUDA -> CPU fallback.
    ///
    /// # Errors
    /// Returns an error string if session builder configuration or model loading fails.
    pub fn create_session(
        &self,
        model_path: &Path,
        prefer_cuda: bool,
        intra_threads: usize,
    ) -> Result<Session, String> {
        let mut builder = Session::builder(&self.env)
            .map_err(|e| format!("Failed to create SessionBuilder: {e}"))?
            .with_intra_threads(intra_threads)
            .map_err(|e| format!("Failed to set intra_threads: {e}"))?;

        if prefer_cuda {
            // Note: Calling `.build()` without `.error_on_failure()` tells ONNX Runtime
            // to attempt loading `libonnxruntime_providers_cuda.so` and initializing CUDA,
            // and to gracefully fall back to the next provider (CPU) if drivers/GPUs are missing.
            match <ep::CUDA as OrtExecutionProvider>::is_available() {
                Ok(true) => {
                    tracing::info!(
                        "CUDA provider library detected; registering CUDA with CPU fallback"
                    );
                    builder = builder
                        .with_execution_providers([
                            ep::CUDA::default().build(),
                            ep::CPU::default().build(),
                        ])
                        .map_err(|e| {
                            format!("Failed to register CUDA/CPU execution providers: {e}")
                        })?;
                }
                Ok(false) => {
                    tracing::warn!("CUDA requested, but CUDA provider is unavailable on this host; falling back to CPU");
                    builder = builder
                        .with_execution_providers([ep::CPU::default().build()])
                        .map_err(|e| format!("Failed to register CPU execution provider: {e}"))?;
                }
                Err(e) => {
                    tracing::warn!("CUDA availability check failed ({e}); falling back to CPU");
                    builder = builder
                        .with_execution_providers([ep::CPU::default().build()])
                        .map_err(|e| format!("Failed to register CPU execution provider: {e}"))?;
                }
            }
        } else {
            builder = builder
                .with_execution_providers([ep::CPU::default().build()])
                .map_err(|e| format!("Failed to register CPU execution provider: {e}"))?;
        }

        builder
            .commit_from_file(model_path)
            .map_err(|e| format!("Failed to load ONNX model {:?}: {e}", model_path))
    }
}

/// Resolves the filesystem path to `libonnxruntime` shared library dynamically.
pub fn resolve_ort_dylib_path() -> Option<PathBuf> {
    if let Some(env_path) = std::env::var_os("ORT_DYLIB_PATH") {
        let path = PathBuf::from(env_path);
        if path.exists() {
            return Some(path);
        }
    }

    let exe_dir = std::env::current_exe().ok()?.parent()?.to_path_buf();

    #[cfg(target_os = "linux")]
    let candidates = [
        "libonnxruntime.so",
        "libonnxruntime.so.1.31.1",
        "libonnxruntime.so.1",
    ];
    #[cfg(target_os = "macos")]
    let candidates = [
        "libonnxruntime.dylib",
        "libonnxruntime.1.31.1.dylib",
        "libonnxruntime.1.dylib",
    ];
    #[cfg(target_os = "windows")]
    let candidates = ["onnxruntime.dll"];
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    let candidates: [&str; 0] = [];

    let search_dirs = [
        exe_dir.join("lib"),
        exe_dir.clone(),
        exe_dir.join("../lib"),
        exe_dir.join("../lib/snooze"),
    ];

    for dir in &search_dirs {
        for file in &candidates {
            let full_path = dir.join(file);
            if full_path.exists() {
                return Some(full_path);
            }
        }
    }

    None
}

#[cfg(feature = "ort-backend")]
static GLOBAL_ORT_RUNTIME: std::sync::OnceLock<Arc<OrtRuntime>> = std::sync::OnceLock::new();

/// Returns or initializes the global ONNX Runtime environment.
///
/// # Errors
/// Returns [`InferenceError::Runtime`] if initializing the ONNX Runtime fails.
#[cfg(feature = "ort-backend")]
pub fn get_or_init_ort_runtime() -> Result<Arc<OrtRuntime>, InferenceError> {
    if let Some(rt) = GLOBAL_ORT_RUNTIME.get() {
        return Ok(Arc::clone(rt));
    }
    let rt = OrtRuntime::new().map_err(InferenceError::Runtime)?;
    let arc = Arc::new(rt);
    let _ = GLOBAL_ORT_RUNTIME.set(Arc::clone(&arc));
    Ok(GLOBAL_ORT_RUNTIME.get().cloned().unwrap_or(arc))
}

/// Concrete ONNX Runtime backend wrapping an `ort::session::Session`.
#[cfg(feature = "ort-backend")]
pub struct OrtBackend {
    session: std::sync::Mutex<Session>,
    provider: ExecutionProvider,
    _runtime: Arc<OrtRuntime>,
}

#[cfg(feature = "ort-backend")]
impl OrtBackend {
    /// Instantiates a new ONNX Runtime session from configuration.
    ///
    /// When configured with [`ExecutionProvider::Auto`], attempts hardware acceleration
    /// in priority order: GPU (CUDA) -> CPU.
    ///
    /// # Errors
    /// Returns [`InferenceError::NotFound`] if the model file is missing, or
    /// [`InferenceError::Runtime`] if session creation or model loading fails.
    pub fn new(config: &OrtConfig) -> Result<Self, InferenceError> {
        let path = Path::new(&config.model_path);
        if !path.exists() {
            return Err(InferenceError::NotFound(format!(
                "ONNX model file not found at: {}",
                config.model_path
            )));
        }

        match config.provider {
            ExecutionProvider::Cuda => {
                #[cfg(not(feature = "cuda"))]
                {
                    return Err(InferenceError::Runtime(format!(
                        "CUDA execution provider requested for '{}', but snooze was compiled without the 'cuda' feature",
                        config.model_path
                    )));
                }
            }
            ExecutionProvider::TensorRt => {
                #[cfg(not(feature = "tensorrt"))]
                {
                    return Err(InferenceError::Runtime(format!(
                        "TensorRT execution provider requested for '{}', but snooze was compiled without the 'tensorrt' feature",
                        config.model_path
                    )));
                }
            }
            ExecutionProvider::CoreML => {
                #[cfg(not(all(target_os = "macos", feature = "coreml")))]
                {
                    return Err(InferenceError::Runtime(format!(
                        "CoreML execution provider requested for '{}', but CoreML is not supported or enabled in this build",
                        config.model_path
                    )));
                }
            }
            _ => {}
        }

        let runtime = get_or_init_ort_runtime()?;
        let prefer_cuda = matches!(
            config.provider,
            ExecutionProvider::Cuda | ExecutionProvider::Auto
        );

        let session = runtime
            .create_session(path, prefer_cuda, config.intra_threads)
            .map_err(InferenceError::Runtime)?;

        let active_provider = match config.provider {
            ExecutionProvider::Cpu => ExecutionProvider::Cpu,
            ExecutionProvider::Cuda => {
                let is_cuda = <ep::CUDA as OrtExecutionProvider>::is_available().unwrap_or(false);
                if is_cuda {
                    ExecutionProvider::Cuda
                } else {
                    ExecutionProvider::Cpu
                }
            }
            ExecutionProvider::Auto => {
                let is_cuda = <ep::CUDA as OrtExecutionProvider>::is_available().unwrap_or(false);
                if is_cuda {
                    ExecutionProvider::Cuda
                } else {
                    ExecutionProvider::Cpu
                }
            }
            ExecutionProvider::TensorRt => ExecutionProvider::TensorRt,
            ExecutionProvider::CoreML => ExecutionProvider::CoreML,
        };

        Ok(Self {
            session: std::sync::Mutex::new(session),
            provider: active_provider,
            _runtime: runtime,
        })
    }
}

#[cfg(feature = "ort-backend")]
impl ModelBackend for OrtBackend {
    /// Executes batched inference against the underlying ONNX Runtime session.
    ///
    /// # Errors
    /// Returns [`InferenceError::Runtime`] or [`InferenceError::ShapeMismatch`] if
    /// tensor extraction or inference execution fails.
    fn infer(&self, inputs: &[TensorPayload]) -> Result<Vec<TensorPayload>, InferenceError> {
        let mut ort_inputs = Vec::with_capacity(inputs.len());

        for payload in inputs {
            let val = Value::from_array(payload.data.clone()).map_err(|e| {
                InferenceError::Runtime(format!("Failed to convert ndarray to ort::Value: {e}"))
            })?;
            ort_inputs.push((payload.name.as_str(), val));
        }

        let mut session_guard = self
            .session
            .lock()
            .map_err(|e| InferenceError::Runtime(format!("ONNX session lock poisoned: {e}")))?;

        let outputs = session_guard
            .run(ort_inputs)
            .map_err(|e| InferenceError::Runtime(format!("ort session run failed: {e}")))?;

        let mut results = Vec::with_capacity(outputs.len());
        for (name, value) in outputs {
            let (shape_dims, raw_data) = value.try_extract_tensor::<f32>().map_err(|e| {
                InferenceError::Runtime(format!("Failed to extract output tensor as f32: {e}"))
            })?;

            let shape_vec: Vec<usize> = shape_dims.iter().map(|&d| d as usize).collect();
            let payload = TensorPayload::from_shape_vec(name, shape_vec, raw_data.to_vec())?;
            results.push(payload);
        }

        Ok(results)
    }

    fn provider_name(&self) -> &'static str {
        match self.provider {
            ExecutionProvider::Auto | ExecutionProvider::Cpu => "onnxruntime-cpu",
            ExecutionProvider::Cuda => "onnxruntime-cuda",
            ExecutionProvider::TensorRt => "onnxruntime-tensorrt",
            ExecutionProvider::CoreML => "onnxruntime-coreml",
        }
    }
}
