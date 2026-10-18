//! Candle and Safetensors execution backend implementation.

use crate::backend::ModelBackend;
use crate::types::{InferenceError, TensorPayload};
use candle_core::{Device, Tensor};
use candle_nn::{Linear, Module};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Execution device target for the Candle backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CandleDevice {
    /// Automatically select the fastest available hardware accelerator:
    /// tries GPU (CUDA), then MLX / Metal (on macOS), lastly defaulting to CPU.
    #[default]
    Auto,
    /// Host CPU compute execution.
    Cpu,
    /// NVIDIA CUDA GPU execution with device index.
    Cuda(usize),
    /// Apple Silicon MLX / Metal GPU acceleration.
    Mlx,
}

/// Supported activation functions between linear layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ActivationType {
    /// Pass-through linear identity (no activation).
    #[default]
    None,
    /// Rectified Linear Unit activation.
    Relu,
    /// Gaussian Error Linear Unit activation.
    Gelu,
}

/// Specification of a linear feed-forward layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerSpec {
    /// Tensor key for the weight matrix (e.g. `"weight"`, `"fc1.weight"`).
    pub weight: String,
    /// Optional tensor key for the bias vector (e.g. `"bias"`, `"fc1.bias"`).
    pub bias: Option<String>,
    /// Activation applied after this layer's projection.
    #[serde(default)]
    pub activation: ActivationType,
}

/// Companion architecture configuration for a Safetensors model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandleModelConfig {
    /// Expected input tensor node name.
    #[serde(default = "default_input_name")]
    pub input_name: String,
    /// Produced output tensor node name.
    #[serde(default = "default_output_name")]
    pub output_name: String,
    /// Ordered list of sequential feed-forward layers.
    #[serde(default)]
    pub layers: Vec<LayerSpec>,
}

fn default_input_name() -> String {
    "input".to_string()
}

fn default_output_name() -> String {
    "output".to_string()
}

/// Configuration options for initializing a Candle model backend.
#[derive(Debug, Clone)]
pub struct CandleConfig {
    /// Filesystem path to the `.safetensors` model file.
    pub model_path: String,
    /// Target execution compute device.
    pub device: CandleDevice,
}

impl Default for CandleConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            device: CandleDevice::Auto,
        }
    }
}

/// Internal representation of a compiled sequential layer.
struct ExecutableLayer {
    linear: Linear,
    activation: ActivationType,
    in_features: usize,
    _out_features: usize,
}

use crate::backend::local_loader::{LoadedModel, LocalModelLoader};

/// Internal execution variant for a Candle model backend.
enum CandleBackendInner {
    /// Sequential Linear/MLP layers.
    SequentialLinear(Vec<ExecutableLayer>),
    /// Dynamic Hugging Face transformer or vision model.
    TransformerOrVision(Box<LoadedModel>),
}

/// Concrete model backend executing Safetensors models using Hugging Face Candle.
pub struct CandleBackend {
    inner: CandleBackendInner,
    device: Device,
    input_name: String,
    output_name: String,
    device_label: &'static str,
}

impl CandleBackend {
    /// Loads a `.safetensors` file, builds the model layers, and initializes the Candle engine.
    ///
    /// When configured with [`CandleDevice::Auto`], attempts hardware acceleration in priority order:
    /// GPU (CUDA) -> MLX / Metal (on macOS) -> CPU.
    ///
    /// # Errors
    /// Returns [`InferenceError::NotFound`] if the model file is missing, or
    /// [`InferenceError::Runtime`] / [`InferenceError::ShapeMismatch`] if parsing, layer resolution,
    /// or layer shape verification fails.
    pub fn new(config: &CandleConfig) -> Result<Self, InferenceError> {
        let model_file = Path::new(&config.model_path);
        if !model_file.exists() {
            return Err(InferenceError::NotFound(format!(
                "Safetensors model file not found at: {}",
                config.model_path
            )));
        }

        let (device, device_label) = match config.device {
            CandleDevice::Auto => {
                // 1. Try GPU (CUDA)
                #[cfg(feature = "cuda")]
                let gpu_dev = match Device::new_cuda(0) {
                    Ok(dev) => {
                        tracing::info!(model = %config.model_path, "Auto device: initialized Candle with CUDA GPU");
                        Some((dev, "candle-cuda"))
                    }
                    Err(e) => {
                        tracing::info!(model = %config.model_path, error = %e, "CUDA GPU unavailable for Candle; checking next option");
                        None
                    }
                };
                #[cfg(not(feature = "cuda"))]
                let gpu_dev: Option<(Device, &'static str)> = None;

                if let Some(res) = gpu_dev {
                    res
                } else {
                    // 2. Try MLX / Metal on macOS
                    let mlx_dev = if cfg!(target_os = "macos") {
                        match Device::new_metal(0) {
                            Ok(dev) => {
                                tracing::info!(model = %config.model_path, "Auto device: initialized Candle with Apple Silicon MLX/Metal");
                                Some((dev, "candle-mlx"))
                            }
                            Err(e) => {
                                tracing::info!(model = %config.model_path, error = %e, "Metal/MLX unavailable for Candle; falling back to CPU");
                                None
                            }
                        }
                    } else {
                        None
                    };

                    if let Some(res) = mlx_dev {
                        res
                    } else {
                        // 3. Fallback to CPU
                        tracing::info!(model = %config.model_path, "Auto device: defaulting Candle to host CPU");
                        (Device::Cpu, "candle-cpu")
                    }
                }
            }
            CandleDevice::Cpu => (Device::Cpu, "candle-cpu"),
            CandleDevice::Mlx => {
                if cfg!(target_os = "macos") {
                    match Device::new_metal(0) {
                        Ok(dev) => (dev, "candle-mlx"),
                        Err(e) => {
                            tracing::warn!(model = %config.model_path, error = %e, "MLX/Metal requested for Candle but failed to initialize; falling back to CPU");
                            (Device::Cpu, "candle-cpu")
                        }
                    }
                } else {
                    tracing::warn!(model = %config.model_path, "MLX/Metal requested for Candle but current OS is not macOS; falling back to CPU");
                    (Device::Cpu, "candle-cpu")
                }
            }
            #[cfg(feature = "cuda")]
            CandleDevice::Cuda(idx) => match Device::new_cuda(idx) {
                Ok(dev) => (dev, "candle-cuda"),
                Err(e) => {
                    tracing::warn!(model = %config.model_path, error = %e, "CUDA requested for Candle but device initialization failed; falling back to CPU");
                    (Device::Cpu, "candle-cpu")
                }
            },
            #[cfg(not(feature = "cuda"))]
            CandleDevice::Cuda(_) => {
                tracing::warn!(model = %config.model_path, "CUDA requested for Candle but crate compiled without CUDA; falling back to CPU");
                (Device::Cpu, "candle-cpu")
            }
        };

        // Inspect for companion configuration: {model_stem}.json or config.json
        let (companion_config, hf_arch) = Self::load_companion_config(model_file);

        let has_explicit_linear_specs = companion_config
            .as_ref()
            .map(|c| !c.layers.is_empty())
            .unwrap_or(false);

        // If not explicitly configured as a sequential linear model, attempt loading via LocalModelLoader
        if !has_explicit_linear_specs {
            let parent_dir = if model_file.is_dir() {
                model_file
            } else {
                model_file.parent().unwrap_or_else(|| Path::new("."))
            };

            if parent_dir.join("config.json").is_file() || model_file.is_dir() {
                match LocalModelLoader::load(model_file, &device) {
                    Ok(loaded_model) => {
                        tracing::info!(
                            model = %config.model_path,
                            provider = %device_label,
                            "Candle Safetensors transformer/vision model loaded successfully"
                        );

                        return Ok(Self {
                            inner: CandleBackendInner::TransformerOrVision(Box::new(loaded_model)),
                            device,
                            input_name: "input_ids".to_string(),
                            output_name: "embedding".to_string(),
                            device_label,
                        });
                    }
                    Err(InferenceError::UnsupportedSafetensorsArchitecture(arch)) => {
                        return Err(InferenceError::UnsupportedSafetensorsArchitecture(arch));
                    }
                    Err(err) => {
                        if model_file.is_dir() || hf_arch.is_some() {
                            return Err(err);
                        }
                    }
                }
            }
        }

        // Load all tensors from the safetensors file for sequential linear execution
        let raw_tensors = candle_core::safetensors::load(model_file, &device).map_err(|e| {
            InferenceError::Runtime(format!(
                "Failed to load safetensors file '{}': {e}",
                config.model_path
            ))
        })?;

        let input_name = companion_config
            .as_ref()
            .map(|c| c.input_name.clone())
            .unwrap_or_else(default_input_name);

        let output_name = companion_config
            .as_ref()
            .map(|c| c.output_name.clone())
            .unwrap_or_else(default_output_name);

        let layer_specs = match companion_config {
            Some(cfg) if !cfg.layers.is_empty() => cfg.layers,
            _ => match Self::auto_discover_layers(&raw_tensors) {
                Ok(specs) => specs,
                Err(err) => {
                    if let Some(arch) = &hf_arch {
                        return Err(InferenceError::Runtime(format!(
                            "Safetensors model '{}' uses Hugging Face architecture '{}' which cannot be automatically mapped to a sequential feed-forward network ({err}). SnoozeML's Safetensors backend currently executes sequential Linear/MLP architectures. To serve this model with SnoozeML, export it to ONNX format (e.g. via `optimum-cli export onnx`) to run on the ONNX Runtime backend, or provide a companion linear layer specification in a JSON file.",
                            config.model_path, arch
                        )));
                    }
                    return Err(err);
                }
            },
        };

        if layer_specs.is_empty() {
            return Err(InferenceError::Runtime(format!(
                "No executable layers discovered in safetensors model '{}'",
                config.model_path
            )));
        }

        let mut compiled_layers = Vec::with_capacity(layer_specs.len());
        let mut previous_out_features: Option<usize> = None;

        for (idx, spec) in layer_specs.into_iter().enumerate() {
            let weight_tensor = raw_tensors.get(&spec.weight).ok_or_else(|| {
                InferenceError::Runtime(format!(
                    "Weight tensor '{}' not found in model weights",
                    spec.weight
                ))
            })?;

            let weight_dims = weight_tensor.dims();
            if weight_dims.len() != 2 {
                if let Some(arch) = &hf_arch {
                    return Err(InferenceError::Runtime(format!(
                        "Safetensors model '{}' uses Hugging Face architecture '{}' with non-2D weight tensor '{}' (shape {weight_dims:?}). SnoozeML's Safetensors backend currently executes sequential Linear/MLP architectures. To serve this model with SnoozeML, export it to ONNX format (e.g. via `optimum-cli export onnx`) to run on the ONNX Runtime backend.",
                        config.model_path, arch, spec.weight
                    )));
                }
                return Err(InferenceError::ShapeMismatch(format!(
                    "Weight tensor '{}' must be 2D (out_features, in_features), but has shape {weight_dims:?}",
                    spec.weight
                )));
            }

            let out_features = weight_dims.first().copied().unwrap_or(0);
            let in_features = weight_dims.get(1).copied().unwrap_or(0);

            if in_features == 0 || out_features == 0 {
                return Err(InferenceError::ShapeMismatch(format!(
                    "Weight tensor '{}' has degenerate dimensions: {weight_dims:?}",
                    spec.weight
                )));
            }

            // Verify sequential layer chaining
            if let Some(prev_out) = previous_out_features {
                if prev_out != in_features {
                    if let Some(arch) = &hf_arch {
                        return Err(InferenceError::Runtime(format!(
                            "Safetensors model '{}' uses Hugging Face architecture '{}' which cannot be executed as a sequential linear pipeline (layer {idx} '{}' expects input dimension {in_features}, but previous layer output dimension was {prev_out}). SnoozeML's Safetensors backend currently executes sequential Linear/MLP architectures. To serve this transformer model with SnoozeML, export it to ONNX format (e.g. via `optimum-cli export onnx`) to run on the ONNX Runtime backend.",
                            config.model_path, arch, spec.weight
                        )));
                    }
                    return Err(InferenceError::ShapeMismatch(format!(
                        "Layer {idx} input dimension ({in_features}) does not match previous layer output dimension ({prev_out})"
                    )));
                }
            }
            previous_out_features = Some(out_features);

            // Optional bias
            let bias_tensor = if let Some(ref bias_key) = spec.bias {
                let b = raw_tensors.get(bias_key).ok_or_else(|| {
                    InferenceError::Runtime(format!(
                        "Bias tensor '{bias_key}' specified for layer {idx} but not found"
                    ))
                })?;
                let b_dims = b.dims();
                if b_dims != [out_features] {
                    return Err(InferenceError::ShapeMismatch(format!(
                        "Bias tensor '{bias_key}' shape {b_dims:?} does not match layer output dimension [{out_features}]"
                    )));
                }
                Some(b.clone())
            } else {
                None
            };

            let linear = Linear::new(weight_tensor.clone(), bias_tensor);
            compiled_layers.push(ExecutableLayer {
                linear,
                activation: spec.activation,
                in_features,
                _out_features: out_features,
            });
        }

        tracing::info!(
            model = %config.model_path,
            layers_count = compiled_layers.len(),
            provider = %device_label,
            "Candle Safetensors model loaded successfully"
        );

        Ok(Self {
            inner: CandleBackendInner::SequentialLinear(compiled_layers),
            device,
            input_name,
            output_name,
            device_label,
        })
    }

    /// Attempts to read a companion `{stem}.json` or `config.json` configuration,
    /// returning any valid CandleModelConfig and/or detected Hugging Face architecture metadata.
    fn load_companion_config(model_file: &Path) -> (Option<CandleModelConfig>, Option<String>) {
        let parent_dir = model_file.parent().unwrap_or_else(|| Path::new("."));
        let stem = model_file
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("");

        let candidate_stem = parent_dir.join(format!("{stem}.json"));
        let candidate_config = parent_dir.join("config.json");

        // 1. Explicit model stem companion file (e.g. model.safetensors -> model.json)
        if candidate_stem.is_file() {
            if let Ok(contents) = std::fs::read_to_string(&candidate_stem) {
                if let Ok(cfg) = serde_json::from_str::<CandleModelConfig>(&contents) {
                    if !cfg.layers.is_empty() {
                        return (Some(cfg), None);
                    }
                }
            }
        }

        // 2. Generic config.json
        if candidate_config.is_file() {
            if let Ok(contents) = std::fs::read_to_string(&candidate_config) {
                // Check if it defines SnoozeML sequential layers
                if let Ok(cfg) = serde_json::from_str::<CandleModelConfig>(&contents) {
                    if !cfg.layers.is_empty() {
                        return (Some(cfg), None);
                    }
                }

                // Detect Hugging Face model architecture if present
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(&contents) {
                    let mut parts = Vec::new();
                    if let Some(archs) = val.get("architectures").and_then(|a| a.as_array()) {
                        let names: Vec<_> = archs.iter().filter_map(|v| v.as_str()).collect();
                        if !names.is_empty() {
                            parts.push(names.join(", "));
                        }
                    }
                    if let Some(mt) = val.get("model_type").and_then(|v| v.as_str()) {
                        parts.push(format!("model_type: '{mt}'"));
                    }
                    if !parts.is_empty() {
                        return (None, Some(parts.join(" / ")));
                    }
                }
            }
        }

        (None, None)
    }

    /// Automatically discovers linear layer weights and biases from raw tensors.
    fn auto_discover_layers(
        tensors: &HashMap<String, Tensor>,
    ) -> Result<Vec<LayerSpec>, InferenceError> {
        // 1. Direct single-layer check: "weight" or "linear.weight"
        if tensors.contains_key("weight") {
            let bias = if tensors.contains_key("bias") {
                Some("bias".to_string())
            } else {
                None
            };
            return Ok(vec![LayerSpec {
                weight: "weight".to_string(),
                bias,
                activation: ActivationType::None,
            }]);
        }

        if tensors.contains_key("linear.weight") {
            let bias = if tensors.contains_key("linear.bias") {
                Some("linear.bias".to_string())
            } else {
                None
            };
            return Ok(vec![LayerSpec {
                weight: "linear.weight".to_string(),
                bias,
                activation: ActivationType::None,
            }]);
        }

        // 2. Discover all keys ending in ".weight"
        let mut weight_keys: Vec<String> = tensors
            .keys()
            .filter(|k| k.ends_with(".weight"))
            .cloned()
            .collect();

        if !weight_keys.is_empty() {
            // Sort naturally by key
            weight_keys.sort();

            let total_layers = weight_keys.len();
            let mut specs = Vec::with_capacity(total_layers);

            for (idx, w_key) in weight_keys.iter().enumerate() {
                let prefix = w_key.strip_suffix(".weight").unwrap_or(w_key.as_str());
                let b_key = format!("{prefix}.bias");
                let bias = if tensors.contains_key(&b_key) {
                    Some(b_key)
                } else {
                    None
                };

                // Use ReLU on hidden layers; None on final layer
                let activation = if idx.saturating_add(1) < total_layers {
                    ActivationType::Relu
                } else {
                    ActivationType::None
                };

                specs.push(LayerSpec {
                    weight: w_key.clone(),
                    bias,
                    activation,
                });
            }

            return Ok(specs);
        }

        // 3. Fallback: Check if there is exactly one 2D tensor in the file
        let mut two_d_keys: Vec<String> = tensors
            .iter()
            .filter(|(_, t)| t.dims().len() == 2)
            .map(|(k, _)| k.clone())
            .collect();

        if two_d_keys.len() == 1 {
            if let Some(single_weight) = two_d_keys.pop() {
                return Ok(vec![LayerSpec {
                    weight: single_weight,
                    bias: None,
                    activation: ActivationType::None,
                }]);
            }
        }

        Err(InferenceError::Runtime(format!(
            "Could not automatically determine layer structure from tensors: {:?}",
            tensors.keys().collect::<Vec<_>>()
        )))
    }

    /// Encodes an input slice of texts into normalized embedding vectors using mean pooling and L2 normalization.
    ///
    /// # Errors
    /// Returns [`InferenceError::InvalidInput`] if the model is not a text transformer or lacks a tokenizer,
    /// or [`InferenceError::Runtime`] if inference fails.
    pub fn encode_text(&self, texts: &[&str]) -> Result<TensorPayload, InferenceError> {
        match &self.inner {
            CandleBackendInner::TransformerOrVision(model) => model.encode_text(texts),
            CandleBackendInner::SequentialLinear(_) => Err(InferenceError::InvalidInput(
                "Sequential linear Candle model does not support text encoding".into(),
            )),
        }
    }

    /// Returns a reference to the loaded transformer or vision model if present.
    #[must_use]
    pub fn loaded_model(&self) -> Option<&crate::backend::local_loader::LoadedModel> {
        match &self.inner {
            CandleBackendInner::TransformerOrVision(model) => Some(model),
            CandleBackendInner::SequentialLinear(_) => None,
        }
    }
}

impl ModelBackend for CandleBackend {
    /// Executes batched inference across the model backend.
    ///
    /// # Errors
    /// Returns [`InferenceError::InvalidInput`] if inputs are empty or non-contiguous,
    /// [`InferenceError::ShapeMismatch`] if feature dimensions do not align with model weights,
    /// or [`InferenceError::Runtime`] if Candle forward propagation fails.
    fn infer(&self, inputs: &[TensorPayload]) -> Result<Vec<TensorPayload>, InferenceError> {
        match &self.inner {
            CandleBackendInner::TransformerOrVision(model) => model.infer(inputs),
            CandleBackendInner::SequentialLinear(layers) => {
                let first_layer = layers.first().ok_or_else(|| {
                    InferenceError::Runtime("Candle model has no compiled layers".into())
                })?;

                // Find input matching input_name, or default to first input
                let input_payload = inputs
                    .iter()
                    .find(|p| p.name == self.input_name)
                    .or_else(|| inputs.first())
                    .ok_or_else(|| {
                        InferenceError::InvalidInput(
                            "No input tensors supplied for inference".into(),
                        )
                    })?;

                let batch_size = input_payload.batch_size();
                if batch_size == 0 {
                    return Err(InferenceError::InvalidInput(
                        "Input tensor has a batch size of 0".into(),
                    ));
                }

                let slice = input_payload.data.as_slice().ok_or_else(|| {
                    InferenceError::InvalidInput(
                        "Tensor data buffer is not contiguous in memory".into(),
                    )
                })?;

                let last_dim = input_payload.shape.last().copied().unwrap_or(0);
                if last_dim != first_layer.in_features {
                    return Err(InferenceError::ShapeMismatch(format!(
                        "Input feature dimension ({last_dim}) does not match model expected input dimension ({})",
                        first_layer.in_features
                    )));
                }

                // Convert to Candle tensor
                let mut current_tensor =
                    Tensor::from_slice(slice, input_payload.shape.as_slice(), &self.device)
                        .map_err(|e| {
                            InferenceError::Runtime(format!(
                                "Failed to build Candle input tensor: {e}"
                            ))
                        })?;

                // Forward through sequential layers
                for (idx, layer) in layers.iter().enumerate() {
                    current_tensor = layer.linear.forward(&current_tensor).map_err(|e| {
                        InferenceError::Runtime(format!("Layer {idx} forward pass failed: {e}"))
                    })?;

                    match layer.activation {
                        ActivationType::None => {}
                        ActivationType::Relu => {
                            current_tensor = current_tensor.relu().map_err(|e| {
                                InferenceError::Runtime(format!(
                                    "Layer {idx} ReLU activation failed: {e}"
                                ))
                            })?;
                        }
                        ActivationType::Gelu => {
                            current_tensor = current_tensor.gelu().map_err(|e| {
                                InferenceError::Runtime(format!(
                                    "Layer {idx} GELU activation failed: {e}"
                                ))
                            })?;
                        }
                    }
                }

                // Convert output Candle tensor back to TensorPayload
                let payload = crate::backend::local_loader::tensor_to_payload(
                    &self.output_name,
                    &current_tensor,
                )?;
                Ok(vec![payload])
            }
        }
    }

    fn provider_name(&self) -> &'static str {
        self.device_label
    }
}
