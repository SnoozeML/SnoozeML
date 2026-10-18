//! ONNX Runtime backend implementation using `ort` 2.0.

#[cfg(feature = "ort-backend")]
use {
    crate::backend::ModelBackend,
    crate::types::{InferenceError, TensorPayload},
    ort::session::Session,
    ort::value::Value,
    std::path::Path,
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

/// Concrete ONNX Runtime backend wrapping an `ort::session::Session`.
#[cfg(feature = "ort-backend")]
pub struct OrtBackend {
    session: std::sync::Mutex<Session>,
    provider: ExecutionProvider,
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

        let (session, active_provider) = match config.provider {
            ExecutionProvider::Auto => {
                // 1. Try GPU (CUDA)
                #[cfg(feature = "cuda")]
                let gpu_res = {
                    tracing::debug!(model = %config.model_path, "Auto device: attempting CUDA GPU initialization");
                    match Self::try_commit_session(
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
                            tracing::info!(model = %config.model_path, error = %e, "CUDA GPU unavailable; checking next option");
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
                    match Self::try_commit_session(
                        path,
                        config.intra_threads,
                        config.inter_threads,
                        ExecutionProvider::Cuda,
                    ) {
                        Ok(s) => (s, ExecutionProvider::Cuda),
                        Err(e) => {
                            tracing::warn!(model = %config.model_path, error = %e, "CUDA requested but failed to initialize; falling back to CPU");
                            let s = Self::try_commit_session(
                                path,
                                config.intra_threads,
                                config.inter_threads,
                                ExecutionProvider::Cpu,
                            )?;
                            (s, ExecutionProvider::Cpu)
                        }
                    }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    tracing::warn!(model = %config.model_path, "CUDA requested but crate compiled without cuda feature; falling back to CPU");
                    let s = Self::try_commit_session(
                        path,
                        config.intra_threads,
                        config.inter_threads,
                        ExecutionProvider::Cpu,
                    )?;
                    (s, ExecutionProvider::Cpu)
                }
            }
            ExecutionProvider::TensorRt => {
                #[cfg(feature = "tensorrt")]
                {
                    match Self::try_commit_session(
                        path,
                        config.intra_threads,
                        config.inter_threads,
                        ExecutionProvider::TensorRt,
                    ) {
                        Ok(s) => (s, ExecutionProvider::TensorRt),
                        Err(e) => {
                            tracing::warn!(model = %config.model_path, error = %e, "TensorRT requested but failed to initialize; falling back to CPU");
                            let s = Self::try_commit_session(
                                path,
                                config.intra_threads,
                                config.inter_threads,
                                ExecutionProvider::Cpu,
                            )?;
                            (s, ExecutionProvider::Cpu)
                        }
                    }
                }
                #[cfg(not(feature = "tensorrt"))]
                {
                    tracing::warn!(model = %config.model_path, "TensorRT requested but crate compiled without tensorrt feature; falling back to CPU");
                    let s = Self::try_commit_session(
                        path,
                        config.intra_threads,
                        config.inter_threads,
                        ExecutionProvider::Cpu,
                    )?;
                    (s, ExecutionProvider::Cpu)
                }
            }
            ExecutionProvider::CoreML => {
                #[cfg(all(target_os = "macos", feature = "coreml"))]
                {
                    match Self::try_commit_session(
                        path,
                        config.intra_threads,
                        config.inter_threads,
                        ExecutionProvider::CoreML,
                    ) {
                        Ok(s) => (s, ExecutionProvider::CoreML),
                        Err(e) => {
                            tracing::warn!(model = %config.model_path, error = %e, "CoreML / MLX requested but failed to initialize; falling back to CPU");
                            let s = Self::try_commit_session(
                                path,
                                config.intra_threads,
                                config.inter_threads,
                                ExecutionProvider::Cpu,
                            )?;
                            (s, ExecutionProvider::Cpu)
                        }
                    }
                }
                #[cfg(not(all(target_os = "macos", feature = "coreml")))]
                {
                    tracing::warn!(model = %config.model_path, "CoreML / MLX requested but not supported in current environment; falling back to CPU");
                    let s = Self::try_commit_session(
                        path,
                        config.intra_threads,
                        config.inter_threads,
                        ExecutionProvider::Cpu,
                    )?;
                    (s, ExecutionProvider::Cpu)
                }
            }
            ExecutionProvider::Cpu => {
                let s = Self::try_commit_session(
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
        })
    }

    /// Internal helper to construct and commit a single ONNX session with given provider.
    fn try_commit_session(
        path: &Path,
        intra_threads: usize,
        inter_threads: usize,
        provider: ExecutionProvider,
    ) -> Result<Session, InferenceError> {
        let mut builder = Session::builder().map_err(|e| {
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
                    .with_execution_providers([ort::ep::CUDA::default().build()])
                    .map_err(|e| {
                        InferenceError::Runtime(format!("Failed to set CUDA provider: {e}"))
                    })?;
            }
            #[cfg(feature = "tensorrt")]
            ExecutionProvider::TensorRt => {
                builder = builder
                    .with_execution_providers([ort::ep::TensorRT::default().build()])
                    .map_err(|e| {
                        InferenceError::Runtime(format!("Failed to set TensorRT provider: {e}"))
                    })?;
            }
            #[cfg(all(target_os = "macos", feature = "coreml"))]
            ExecutionProvider::CoreML => {
                builder = builder
                    .with_execution_providers([ort::ep::CoreML::default().build()])
                    .map_err(|e| {
                        InferenceError::Runtime(format!("Failed to set CoreML provider: {e}"))
                    })?;
            }
            _ => {}
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
