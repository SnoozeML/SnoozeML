//! ONNX Runtime backend implementation using `ort` 2.0 (`load-dynamic`).

#[cfg(feature = "ort-backend")]
use {
    crate::backend::ModelBackend,
    crate::types::{InferenceError, TensorPayload},
    ort::{environment::Environment, session::Session, value::Value},
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
            tracing::info!(
                dylib_path = ?dylib_path,
                "Loading ONNX Runtime dynamically"
            );
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
}

/// Resolves the filesystem path to the `libonnxruntime` shared library dynamically.
pub fn resolve_ort_dylib_path() -> Option<PathBuf> {
    if let Some(env_path) = std::env::var_os("ORT_DYLIB_PATH") {
        let path = PathBuf::from(env_path);
        if path.exists() {
            return Some(path);
        }
    }

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

    let mut search_dirs = Vec::new();
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            search_dirs.push(exe_dir.join("lib"));
            search_dirs.push(exe_dir.to_path_buf());
            search_dirs.push(exe_dir.join("../lib"));
            search_dirs.push(exe_dir.join("../lib/snooze"));
            search_dirs.push(exe_dir.join("../../lib"));
        }
    }
    search_dirs.push(PathBuf::from("ort_dist/lib"));
    search_dirs.push(PathBuf::from("/opt/ort/lib"));
    search_dirs.push(PathBuf::from("/usr/local/lib"));

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
    /// in priority order: GPU (CUDA) -> MLX / CoreML (on macOS) -> CPU.
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

        // Check compile-time feature availability before initializing the runtime
        // when an explicit accelerator is requested.
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
            ExecutionProvider::Auto | ExecutionProvider::Cpu => {}
        }

        let runtime = get_or_init_ort_runtime()?;

        let (session, active_provider) = match config.provider {
            ExecutionProvider::Auto => {
                // 1. Try GPU (CUDA)
                #[cfg(feature = "cuda")]
                let gpu_res = {
                    tracing::debug!(model = %config.model_path, "Auto device: attempting CUDA GPU initialization");
                    match Self::try_commit_session(
                        &runtime,
                        path,
                        config.intra_threads,
                        config.inter_threads,
                        ExecutionProvider::Cuda,
                    ) {
                        Ok(s) => {
                            tracing::info!(model = %config.model_path, "Auto device: initialized with CUDA GPU");
                            Some((s, ExecutionProvider::Cuda))
                        }
                        Err(e) => {
                            let err_str = e.to_string();
                            let reason = crate::backend::classify_cuda_failure_reason(&err_str);
                            tracing::warn!(
                                model = %config.model_path,
                                reason = %reason,
                                error = %err_str,
                                "Auto device: {reason} Continuing on the CPU backend"
                            );
                            None
                        }
                    }
                };
                #[cfg(not(feature = "cuda"))]
                let gpu_res: Option<(Session, ExecutionProvider)> = None;

                if let Some(res) = gpu_res {
                    res
                } else {
                    // 2. Try MLX / CoreML (on macOS)
                    #[cfg(all(target_os = "macos", feature = "coreml"))]
                    let mlx_res = {
                        tracing::debug!(model = %config.model_path, "Auto device: attempting macOS CoreML/MLX initialization");
                        match Self::try_commit_session(
                            &runtime,
                            path,
                            config.intra_threads,
                            config.inter_threads,
                            ExecutionProvider::CoreML,
                        ) {
                            Ok(s) => {
                                tracing::info!(model = %config.model_path, "Auto device: initialized with CoreML/MLX");
                                Some((s, ExecutionProvider::CoreML))
                            }
                            Err(e) => {
                                tracing::info!(model = %config.model_path, error = %e, "CoreML/MLX unavailable; falling back to CPU");
                                None
                            }
                        }
                    };
                    #[cfg(not(all(target_os = "macos", feature = "coreml")))]
                    let mlx_res: Option<(Session, ExecutionProvider)> = None;

                    if let Some(res) = mlx_res {
                        res
                    } else {
                        // 3. Universal CPU fallback
                        tracing::info!(model = %config.model_path, "Auto device: defaulting to host CPU");
                        let s = Self::try_commit_session(
                            &runtime,
                            path,
                            config.intra_threads,
                            config.inter_threads,
                            ExecutionProvider::Cpu,
                        )?;
                        (s, ExecutionProvider::Cpu)
                    }
                }
            }
            ExecutionProvider::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    let s = Self::try_commit_session(
                        &runtime,
                        path,
                        config.intra_threads,
                        config.inter_threads,
                        ExecutionProvider::Cuda,
                    )
                    .map_err(|e| {
                        let err_str = e.to_string();
                        let reason = crate::backend::classify_cuda_failure_reason(&err_str);
                        tracing::error!(
                            model = %config.model_path,
                            reason = %reason,
                            error = %err_str,
                            "Strict CUDA execution provider initialization failed: {reason} Remediation: Install a compatible NVIDIA driver, pass --gpus all to Docker, or set device = \"cpu\" / device = \"auto\" in your config"
                        );
                        InferenceError::Runtime(format!(
                            "Failed to initialize CUDA execution provider for '{}': {reason} ({err_str}). Install a compatible NVIDIA driver, CUDA 13 runtime libraries (libcudart/cublas 13), and cuDNN 9 on your host, pass --gpus all to Docker (ghcr.io/snoozeml/snoozeml:cuda), or set device = \"cpu\" / device = \"auto\" in your config",
                            config.model_path
                        ))
                    })?;
                    (s, ExecutionProvider::Cuda)
                }
                #[cfg(not(feature = "cuda"))]
                {
                    return Err(InferenceError::Runtime(format!(
                        "CUDA execution provider requested for '{}', but snooze was compiled without the 'cuda' feature",
                        config.model_path
                    )));
                }
            }
            ExecutionProvider::TensorRt => {
                #[cfg(feature = "tensorrt")]
                {
                    let s = Self::try_commit_session(
                        &runtime,
                        path,
                        config.intra_threads,
                        config.inter_threads,
                        ExecutionProvider::TensorRt,
                    )
                    .map_err(|e| {
                        let err_str = e.to_string();
                        let reason = crate::backend::classify_cuda_failure_reason(&err_str);
                        tracing::error!(
                            model = %config.model_path,
                            reason = %reason,
                            error = %err_str,
                            "Strict TensorRT execution provider initialization failed: {reason} Remediation: Install NVIDIA TensorRT and a compatible NVIDIA driver, pass --gpus all to Docker, or set device = \"cpu\" / device = \"auto\" in your config"
                        );
                        InferenceError::Runtime(format!(
                            "Failed to initialize TensorRT execution provider for '{}': {reason} ({err_str}). Ensure NVIDIA TensorRT, CUDA 13 runtime, and cuDNN 9 libraries are installed on your host, pass --gpus all to Docker (ghcr.io/snoozeml/snoozeml:cuda), or set device = \"cpu\" / device = \"auto\" in your config",
                            config.model_path
                        ))
                    })?;
                    (s, ExecutionProvider::TensorRt)
                }
                #[cfg(not(feature = "tensorrt"))]
                {
                    return Err(InferenceError::Runtime(format!(
                        "TensorRT execution provider requested for '{}', but snooze was compiled without the 'tensorrt' feature",
                        config.model_path
                    )));
                }
            }
            ExecutionProvider::CoreML => {
                #[cfg(all(target_os = "macos", feature = "coreml"))]
                {
                    let s = Self::try_commit_session(
                        &runtime,
                        path,
                        config.intra_threads,
                        config.inter_threads,
                        ExecutionProvider::CoreML,
                    )?;
                    (s, ExecutionProvider::CoreML)
                }
                #[cfg(not(all(target_os = "macos", feature = "coreml")))]
                {
                    return Err(InferenceError::Runtime(format!(
                        "CoreML execution provider requested for '{}', but CoreML is not supported or enabled in this build",
                        config.model_path
                    )));
                }
            }
            ExecutionProvider::Cpu => {
                let s = Self::try_commit_session(
                    &runtime,
                    path,
                    config.intra_threads,
                    config.inter_threads,
                    ExecutionProvider::Cpu,
                )?;
                (s, ExecutionProvider::Cpu)
            }
        };

        Ok(Self {
            session: std::sync::Mutex::new(session),
            provider: active_provider,
            _runtime: runtime,
        })
    }

    /// Internal helper to construct and commit a single ONNX session with the given provider.
    fn try_commit_session(
        runtime: &OrtRuntime,
        path: &Path,
        intra_threads: usize,
        inter_threads: usize,
        provider: ExecutionProvider,
    ) -> Result<Session, InferenceError> {
        let mut builder = Session::builder(&runtime.env).map_err(|e| {
            InferenceError::Runtime(format!("Failed to create SessionBuilder: {e}"))
        })?;

        builder = builder
            .with_intra_threads(intra_threads)
            .map_err(|e| InferenceError::Runtime(format!("Failed to set intra threads: {e}")))?
            .with_inter_threads(inter_threads)
            .map_err(|e| InferenceError::Runtime(format!("Failed to set inter threads: {e}")))?;

        match provider {
            #[cfg(feature = "cuda")]
            ExecutionProvider::Cuda => {
                builder = builder
                    .with_execution_providers([ort::ep::CUDA::default().build().error_on_failure()])
                    .map_err(|e| {
                        InferenceError::Runtime(format!("Failed to register CUDA provider: {e}"))
                    })?;
            }
            #[cfg(feature = "tensorrt")]
            ExecutionProvider::TensorRt => {
                builder = builder
                    .with_execution_providers([ort::ep::TensorRT::default()
                        .build()
                        .error_on_failure()])
                    .map_err(|e| {
                        InferenceError::Runtime(format!(
                            "Failed to register TensorRT provider: {e}"
                        ))
                    })?;
            }
            #[cfg(all(target_os = "macos", feature = "coreml"))]
            ExecutionProvider::CoreML => {
                builder = builder
                    .with_execution_providers([ort::ep::CoreML::default()
                        .build()
                        .error_on_failure()])
                    .map_err(|e| {
                        InferenceError::Runtime(format!("Failed to register CoreML provider: {e}"))
                    })?;
            }
            _ => {
                builder = builder
                    .with_execution_providers([ort::ep::CPU::default().build()])
                    .map_err(|e| {
                        InferenceError::Runtime(format!("Failed to register CPU provider: {e}"))
                    })?;
            }
        }

        builder
            .commit_from_file(path)
            .map_err(|e| InferenceError::Runtime(format!("Failed to load ONNX session: {e}")))
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
