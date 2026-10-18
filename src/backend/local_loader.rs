//! Dynamic loader for Hugging Face Safetensors models using Candle and candle-transformers.
//!
//! Provides direct execution of local Safetensors model directories for text embedders
//! (BERT, RoBERTa, Jina BERT, Nomic BERT, DistilBERT) and vision models (CLIP, ViT, DINOv2)
//! without requiring prior ONNX conversion.

use crate::types::{InferenceError, TensorPayload};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::{
    bert::{BertModel, Config as BertConfig},
    clip::{ClipConfig, ClipModel},
    distilbert::{Config as DistilBertConfig, DistilBertModel},
    jina_bert::{BertModel as JinaBertModel, Config as JinaBertConfig},
    nomic_bert::{Config as NomicBertConfig, NomicBertModel},
    vit::{Config as VitConfig, Model as VitModel},
};
use serde::Deserialize;
use std::path::{Path, PathBuf};

/// Enumeration of supported Safetensors model architectures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SupportedArchitecture {
    /// Standard BERT or RoBERTa embedding / encoder model.
    Bert,
    /// Jina BERT embedding model.
    JinaBert,
    /// Nomic BERT embedding model.
    NomicBert,
    /// DistilBERT transformer model.
    DistilBert,
    /// CLIP multimodal text and vision model.
    Clip,
    /// Vision Transformer (ViT) image classification model.
    Vit,
    /// DINOv2 self-supervised vision transformer.
    DinoV2,
}

impl SupportedArchitecture {
    /// Attempts to parse a Hugging Face `model_type` or architecture name into a supported enum variant.
    #[must_use]
    pub fn from_hf_identifier(identifier: &str) -> Option<Self> {
        let normalized = identifier.trim().to_ascii_lowercase();
        match normalized.as_str() {
            "bert" | "roberta" => Some(Self::Bert),
            "jina-bert" | "jina_bert" => Some(Self::JinaBert),
            "nomic_bert" => Some(Self::NomicBert),
            "distilbert" => Some(Self::DistilBert),
            "clip" | "clip_text_model" | "clip_vision_model" => Some(Self::Clip),
            "vit" => Some(Self::Vit),
            "dinov2" => Some(Self::DinoV2),
            _ => None,
        }
    }
}

/// Companion configuration parser for DINOv2 models.
#[derive(Debug, Clone, Deserialize)]
pub struct DinoV2Config {
    /// Number of transformer hidden layers (depth).
    #[serde(default = "default_dinov2_depth")]
    pub num_hidden_layers: usize,
    /// Hidden embedding dimension size.
    #[serde(default = "default_dinov2_embed_dim")]
    pub hidden_size: usize,
    /// Number of attention heads in multi-head self-attention.
    #[serde(default = "default_dinov2_num_heads")]
    pub num_attention_heads: usize,
}

fn default_dinov2_depth() -> usize {
    12
}

fn default_dinov2_embed_dim() -> usize {
    384
}

fn default_dinov2_num_heads() -> usize {
    6
}

/// In-memory representation of an instantiated Candle transformer or vision model.
pub enum LoadedModel {
    /// Standard BERT / RoBERTa model.
    Bert {
        /// Loaded Candle BERT model weights.
        model: BertModel,
        /// Optional Hugging Face tokenizer if `tokenizer.json` was present.
        tokenizer: Option<tokenizers::Tokenizer>,
        /// Device where model resides.
        device: Device,
    },
    /// Jina BERT model.
    JinaBert {
        /// Loaded Candle Jina BERT model weights.
        model: JinaBertModel,
        /// Optional Hugging Face tokenizer if `tokenizer.json` was present.
        tokenizer: Option<tokenizers::Tokenizer>,
        /// Device where model resides.
        device: Device,
    },
    /// Nomic BERT model.
    NomicBert {
        /// Loaded Candle Nomic BERT model weights.
        model: NomicBertModel,
        /// Optional Hugging Face tokenizer if `tokenizer.json` was present.
        tokenizer: Option<tokenizers::Tokenizer>,
        /// Device where model resides.
        device: Device,
    },
    /// DistilBERT model.
    DistilBert {
        /// Loaded Candle DistilBERT model weights.
        model: DistilBertModel,
        /// Optional Hugging Face tokenizer if `tokenizer.json` was present.
        tokenizer: Option<tokenizers::Tokenizer>,
        /// Device where model resides.
        device: Device,
    },
    /// CLIP text and vision model.
    Clip {
        /// Loaded Candle CLIP model weights.
        model: ClipModel,
        /// Optional Hugging Face tokenizer if `tokenizer.json` was present.
        tokenizer: Option<tokenizers::Tokenizer>,
        /// Device where model resides.
        device: Device,
    },
    /// Vision Transformer (ViT).
    Vit {
        /// Loaded Candle ViT model weights.
        model: VitModel,
        /// Device where model resides.
        device: Device,
    },
    /// DINOv2 vision model.
    DinoV2 {
        /// Loaded Candle DINOv2 vision transformer.
        model: candle_transformers::models::dinov2::DinoVisionTransformer,
        /// Device where model resides.
        device: Device,
    },
}

impl LoadedModel {
    /// Returns the target device of the loaded model.
    #[must_use]
    pub fn device(&self) -> &Device {
        match self {
            Self::Bert { device, .. }
            | Self::JinaBert { device, .. }
            | Self::NomicBert { device, .. }
            | Self::DistilBert { device, .. }
            | Self::Clip { device, .. }
            | Self::Vit { device, .. }
            | Self::DinoV2 { device, .. } => device,
        }
    }

    /// Returns a reference to the optional tokenizer if available.
    #[must_use]
    pub fn tokenizer(&self) -> Option<&tokenizers::Tokenizer> {
        match self {
            Self::Bert { tokenizer, .. }
            | Self::JinaBert { tokenizer, .. }
            | Self::NomicBert { tokenizer, .. }
            | Self::DistilBert { tokenizer, .. }
            | Self::Clip { tokenizer, .. } => tokenizer.as_ref(),
            Self::Vit { .. } | Self::DinoV2 { .. } => None,
        }
    }

    /// Encodes an input slice of texts into normalized embedding vectors using mean pooling and L2 normalization.
    ///
    /// # Errors
    /// Returns [`InferenceError::InvalidInput`] if no tokenizer is loaded, or
    /// [`InferenceError::Runtime`] if tokenization or tensor inference fails.
    pub fn encode_text(&self, texts: &[&str]) -> Result<TensorPayload, InferenceError> {
        let tokenizer = self.tokenizer().ok_or_else(|| {
            InferenceError::InvalidInput(
                "Cannot encode text: model directory does not contain 'tokenizer.json'".into(),
            )
        })?;

        let encodings = tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| InferenceError::Runtime(format!("Tokenization failed: {e}")))?;

        let batch_size = encodings.len();
        if batch_size == 0 {
            return Err(InferenceError::InvalidInput(
                "Input text batch is empty".into(),
            ));
        }

        let max_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0);

        if max_len == 0 {
            return Err(InferenceError::InvalidInput(
                "All tokenized sequences have length 0".into(),
            ));
        }

        let total_tokens = batch_size.checked_mul(max_len).ok_or_else(|| {
            InferenceError::InvalidInput("Batch token count overflows usize".into())
        })?;

        let mut all_ids = Vec::with_capacity(total_tokens);
        let mut all_mask = Vec::with_capacity(total_tokens);

        for enc in &encodings {
            let ids = enc.get_ids();
            let mask = enc.get_attention_mask();
            all_ids.extend_from_slice(ids);
            all_mask.extend_from_slice(mask);
            // Pad if necessary
            let pad_len = max_len.saturating_sub(ids.len());
            if pad_len > 0 {
                all_ids.resize(all_ids.len().saturating_add(pad_len), 0u32);
                all_mask.resize(all_mask.len().saturating_add(pad_len), 0u32);
            }
        }

        let device = self.device();
        let input_ids =
            Tensor::from_slice(&all_ids, (batch_size, max_len), device).map_err(|e| {
                InferenceError::Runtime(format!("Failed to build input_ids tensor: {e}"))
            })?;
        let attention_mask =
            Tensor::from_slice(&all_mask, (batch_size, max_len), device).map_err(|e| {
                InferenceError::Runtime(format!("Failed to build attention_mask tensor: {e}"))
            })?;

        self.infer_text_embedding(&input_ids, &attention_mask)
    }

    /// Internal forward pass for text embedding models producing pooled and L2-normalized representations.
    fn infer_text_embedding(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<TensorPayload, InferenceError> {
        let token_embeddings = match self {
            Self::Bert { model, .. } => {
                let token_type_ids = input_ids.zeros_like().map_err(|e| {
                    InferenceError::Runtime(format!("Failed to build token_type_ids: {e}"))
                })?;
                model
                    .forward(input_ids, &token_type_ids, Some(attention_mask))
                    .map_err(|e| {
                        InferenceError::Runtime(format!("BERT forward pass failed: {e}"))
                    })?
            }
            Self::JinaBert { model, .. } => model.forward(input_ids).map_err(|e| {
                InferenceError::Runtime(format!("Jina BERT forward pass failed: {e}"))
            })?,
            Self::NomicBert { model, .. } => {
                let token_type_ids = input_ids.zeros_like().map_err(|e| {
                    InferenceError::Runtime(format!("Failed to build token_type_ids: {e}"))
                })?;
                model
                    .forward(input_ids, Some(&token_type_ids), Some(attention_mask))
                    .map_err(|e| {
                        InferenceError::Runtime(format!("Nomic BERT forward pass failed: {e}"))
                    })?
            }
            Self::DistilBert { model, .. } => {
                model.forward(input_ids, attention_mask).map_err(|e| {
                    InferenceError::Runtime(format!("DistilBERT forward pass failed: {e}"))
                })?
            }
            Self::Clip { model, .. } => {
                let text_features = model.get_text_features(input_ids).map_err(|e| {
                    InferenceError::Runtime(format!("CLIP text features failed: {e}"))
                })?;
                let normalized = l2_normalize(&text_features)?;
                return tensor_to_payload("embedding", &normalized);
            }
            Self::Vit { .. } | Self::DinoV2 { .. } => {
                return Err(InferenceError::InvalidInput(
                    "Model is a vision model and cannot process text input tokens".into(),
                ));
            }
        };

        // Standard Mean Pooling
        let pooled = mean_pooling(&token_embeddings, attention_mask)?;
        // L2 Normalization
        let normalized = l2_normalize(&pooled)?;

        tensor_to_payload("embedding", &normalized)
    }

    /// Executes batched inference on the loaded model using structured tensor inputs.
    ///
    /// # Errors
    /// Returns [`InferenceError::InvalidInput`] if inputs are malformed or missing,
    /// [`InferenceError::ShapeMismatch`] if tensor dimensions do not align, or
    /// [`InferenceError::Runtime`] if the forward propagation encounters an error.
    pub fn infer(&self, inputs: &[TensorPayload]) -> Result<Vec<TensorPayload>, InferenceError> {
        let first_payload = inputs.first().ok_or_else(|| {
            InferenceError::InvalidInput("No input tensors supplied for inference".into())
        })?;

        match self {
            Self::Bert { .. }
            | Self::JinaBert { .. }
            | Self::NomicBert { .. }
            | Self::DistilBert { .. } => {
                // Look for input_ids
                let input_ids_payload = inputs
                    .iter()
                    .find(|p| p.name == "input_ids")
                    .unwrap_or(first_payload);

                let batch_size = input_ids_payload.batch_size();
                if batch_size == 0 {
                    return Err(InferenceError::InvalidInput("Input batch size is 0".into()));
                }

                let seq_len = input_ids_payload.shape.get(1).copied().unwrap_or(0);
                if seq_len == 0 {
                    return Err(InferenceError::InvalidInput(
                        "Input sequence length is 0".into(),
                    ));
                }

                let ids_slice = input_ids_payload.data.as_slice().ok_or_else(|| {
                    InferenceError::InvalidInput(
                        "Input IDs buffer is not contiguous in memory".into(),
                    )
                })?;

                // Convert float tensor values to u32 token IDs
                let u32_ids: Vec<u32> = ids_slice.iter().map(|&v| v as u32).collect();
                let device = self.device();
                let input_ids = Tensor::from_slice(&u32_ids, (batch_size, seq_len), device)
                    .map_err(|e| {
                        InferenceError::Runtime(format!("Failed to build input_ids tensor: {e}"))
                    })?;

                // Look for optional attention_mask
                let attention_mask = if let Some(mask_payload) =
                    inputs.iter().find(|p| p.name == "attention_mask")
                {
                    let mask_slice = mask_payload.data.as_slice().ok_or_else(|| {
                        InferenceError::InvalidInput(
                            "Attention mask buffer is not contiguous".into(),
                        )
                    })?;
                    let u32_mask: Vec<u32> = mask_slice.iter().map(|&v| v as u32).collect();
                    Tensor::from_slice(&u32_mask, (batch_size, seq_len), device).map_err(|e| {
                        InferenceError::Runtime(format!(
                            "Failed to build attention_mask tensor: {e}"
                        ))
                    })?
                } else {
                    input_ids.ones_like().map_err(|e| {
                        InferenceError::Runtime(format!(
                            "Failed to create default attention mask: {e}"
                        ))
                    })?
                };

                let output_payload = self.infer_text_embedding(&input_ids, &attention_mask)?;
                Ok(vec![output_payload])
            }
            Self::Clip { model, device, .. } => {
                // If 4D tensor -> image features, if 2D tensor -> text features
                let slice = first_payload.data.as_slice().ok_or_else(|| {
                    InferenceError::InvalidInput(
                        "CLIP input tensor is not contiguous in memory".into(),
                    )
                })?;
                let shape = first_payload.shape.as_slice();

                if shape.len() == 4 {
                    // Image input: [batch, channels, height, width]
                    let img_tensor = Tensor::from_slice(slice, shape, device).map_err(|e| {
                        InferenceError::Runtime(format!("Failed to build CLIP image tensor: {e}"))
                    })?;
                    let features = model.get_image_features(&img_tensor).map_err(|e| {
                        InferenceError::Runtime(format!("CLIP image forward pass failed: {e}"))
                    })?;
                    let normalized = l2_normalize(&features)?;
                    let out = tensor_to_payload("image_embeddings", &normalized)?;
                    Ok(vec![out])
                } else if shape.len() == 2 {
                    // Token input: [batch, seq_len]
                    let u32_ids: Vec<u32> = slice.iter().map(|&v| v as u32).collect();
                    let input_ids = Tensor::from_slice(&u32_ids, shape, device).map_err(|e| {
                        InferenceError::Runtime(format!(
                            "Failed to build CLIP text token tensor: {e}"
                        ))
                    })?;
                    let features = model.get_text_features(&input_ids).map_err(|e| {
                        InferenceError::Runtime(format!("CLIP text forward pass failed: {e}"))
                    })?;
                    let normalized = l2_normalize(&features)?;
                    let out = tensor_to_payload("text_embeddings", &normalized)?;
                    Ok(vec![out])
                } else {
                    Err(InferenceError::ShapeMismatch(format!(
                        "CLIP expects 4D image input [B, C, H, W] or 2D text token input [B, S], got shape {shape:?}"
                    )))
                }
            }
            Self::Vit { model, device } => {
                let slice = first_payload.data.as_slice().ok_or_else(|| {
                    InferenceError::InvalidInput(
                        "ViT input tensor is not contiguous in memory".into(),
                    )
                })?;
                let shape = first_payload.shape.as_slice();
                if shape.len() != 4 {
                    return Err(InferenceError::ShapeMismatch(format!(
                        "ViT expects 4D input tensor [batch, channels, height, width], got {shape:?}"
                    )));
                }
                let img_tensor = Tensor::from_slice(slice, shape, device).map_err(|e| {
                    InferenceError::Runtime(format!("Failed to build ViT input tensor: {e}"))
                })?;
                let logits = model.forward(&img_tensor).map_err(|e| {
                    InferenceError::Runtime(format!("ViT forward pass failed: {e}"))
                })?;
                let out = tensor_to_payload("logits", &logits)?;
                Ok(vec![out])
            }
            Self::DinoV2 { model, device } => {
                let slice = first_payload.data.as_slice().ok_or_else(|| {
                    InferenceError::InvalidInput(
                        "DINOv2 input tensor is not contiguous in memory".into(),
                    )
                })?;
                let shape = first_payload.shape.as_slice();
                if shape.len() != 4 {
                    return Err(InferenceError::ShapeMismatch(format!(
                        "DINOv2 expects 4D input tensor [batch, channels, height, width], got {shape:?}"
                    )));
                }
                let img_tensor = Tensor::from_slice(slice, shape, device).map_err(|e| {
                    InferenceError::Runtime(format!("Failed to build DINOv2 input tensor: {e}"))
                })?;
                let features = model.forward(&img_tensor).map_err(|e| {
                    InferenceError::Runtime(format!("DINOv2 forward pass failed: {e}"))
                })?;
                let out = tensor_to_payload("features", &features)?;
                Ok(vec![out])
            }
        }
    }
}

/// Converts a Candle Tensor into a SnoozeML `TensorPayload`.
pub(crate) fn tensor_to_payload(
    name: &str,
    tensor: &Tensor,
) -> Result<TensorPayload, InferenceError> {
    let out_shape = tensor.dims().to_vec();
    let flattened = tensor
        .flatten_all()
        .map_err(|e| InferenceError::Runtime(format!("Failed to flatten output tensor: {e}")))?
        .to_vec1::<f32>()
        .map_err(|e| {
            InferenceError::Runtime(format!("Failed to read output tensor buffer: {e}"))
        })?;

    TensorPayload::from_shape_vec(name, out_shape, flattened)
}

/// Dynamic local Safetensors model loader using Hugging Face Candle.
pub struct LocalModelLoader;

impl LocalModelLoader {
    /// Loads a local Safetensors model directory or file onto the specified compute device.
    ///
    /// # Process
    /// 1. Locates and reads `<dir>/config.json`.
    /// 2. Resolves architecture by inspecting `model_type` or `architectures[0]`.
    /// 3. Locates `<dir>/model.safetensors` and maps weights via [`VarBuilder::from_mmaped_safetensors`].
    /// 4. Dispatches and initializes the target Candle model struct.
    /// 5. If `model_type` is unsupported, returns [`InferenceError::UnsupportedSafetensorsArchitecture`].
    ///
    /// # Errors
    /// Returns [`InferenceError::NotFound`] if files are missing,
    /// [`InferenceError::UnsupportedSafetensorsArchitecture`] if the model architecture is unsupported,
    /// or [`InferenceError::Runtime`] / [`InferenceError::InvalidInput`] if weight mapping or model building fails.
    pub fn load(path: &Path, device: &Device) -> Result<LoadedModel, InferenceError> {
        let (dir, weights_file) = resolve_paths(path)?;

        let config_path = dir.join("config.json");
        if !config_path.is_file() {
            return Err(InferenceError::NotFound(format!(
                "Model configuration 'config.json' not found in directory: '{}'",
                dir.display()
            )));
        }

        let config_str = std::fs::read_to_string(&config_path).map_err(|e| {
            InferenceError::Runtime(format!(
                "Failed to read config.json at '{}': {e}",
                config_path.display()
            ))
        })?;

        let config_val: serde_json::Value = serde_json::from_str(&config_str).map_err(|e| {
            InferenceError::InvalidInput(format!(
                "Failed to parse config.json at '{}': {e}",
                config_path.display()
            ))
        })?;

        let raw_arch = extract_architecture_string(&config_val).ok_or_else(|| {
            InferenceError::InvalidInput(format!(
                "config.json at '{}' does not specify 'model_type' or 'architectures'",
                config_path.display()
            ))
        })?;

        let supported_arch = SupportedArchitecture::from_hf_identifier(&raw_arch)
            .ok_or_else(|| InferenceError::UnsupportedSafetensorsArchitecture(raw_arch.clone()))?;

        tracing::info!(
            model_path = %path.display(),
            architecture = %raw_arch,
            weights = %weights_file.display(),
            "Loading native Safetensors model via Candle"
        );

        // Map weights using memory mapping
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_file.as_path()], DType::F32, device)
                .map_err(|e| {
                    InferenceError::Runtime(format!(
                        "Failed to mmap safetensors weights from '{}': {e}",
                        weights_file.display()
                    ))
                })?
        };

        // Discover optional tokenizer.json in the model directory
        let tokenizer_path = dir.join("tokenizer.json");
        let tokenizer = if tokenizer_path.is_file() {
            match tokenizers::Tokenizer::from_file(&tokenizer_path) {
                Ok(tok) => {
                    tracing::info!(
                        path = %tokenizer_path.display(),
                        "Loaded companion tokenizer.json"
                    );
                    Some(tok)
                }
                Err(err) => {
                    tracing::warn!(
                        path = %tokenizer_path.display(),
                        error = %err,
                        "Found tokenizer.json but failed to parse; continuing without tokenizer"
                    );
                    None
                }
            }
        } else {
            None
        };

        match supported_arch {
            SupportedArchitecture::Bert => {
                let cfg: BertConfig = serde_json::from_str(&config_str).map_err(|e| {
                    InferenceError::InvalidInput(format!("Failed to parse BERT config: {e}"))
                })?;
                let model = BertModel::load(vb, &cfg).map_err(|e| {
                    InferenceError::Runtime(format!("Failed to build BERT model: {e}"))
                })?;
                Ok(LoadedModel::Bert {
                    model,
                    tokenizer,
                    device: device.clone(),
                })
            }
            SupportedArchitecture::JinaBert => {
                let cfg: JinaBertConfig = serde_json::from_str(&config_str).map_err(|e| {
                    InferenceError::InvalidInput(format!("Failed to parse Jina BERT config: {e}"))
                })?;
                let model = JinaBertModel::new(vb, &cfg).map_err(|e| {
                    InferenceError::Runtime(format!("Failed to build Jina BERT model: {e}"))
                })?;
                Ok(LoadedModel::JinaBert {
                    model,
                    tokenizer,
                    device: device.clone(),
                })
            }
            SupportedArchitecture::NomicBert => {
                let cfg: NomicBertConfig = serde_json::from_str(&config_str).map_err(|e| {
                    InferenceError::InvalidInput(format!("Failed to parse Nomic BERT config: {e}"))
                })?;
                let model = NomicBertModel::load(vb, &cfg).map_err(|e| {
                    InferenceError::Runtime(format!("Failed to build Nomic BERT model: {e}"))
                })?;
                Ok(LoadedModel::NomicBert {
                    model,
                    tokenizer,
                    device: device.clone(),
                })
            }
            SupportedArchitecture::DistilBert => {
                let cfg: DistilBertConfig = serde_json::from_str(&config_str).map_err(|e| {
                    InferenceError::InvalidInput(format!("Failed to parse DistilBERT config: {e}"))
                })?;
                let model = DistilBertModel::load(vb, &cfg).map_err(|e| {
                    InferenceError::Runtime(format!("Failed to build DistilBERT model: {e}"))
                })?;
                Ok(LoadedModel::DistilBert {
                    model,
                    tokenizer,
                    device: device.clone(),
                })
            }
            SupportedArchitecture::Clip => {
                let cfg = parse_clip_config(&config_val);
                let model = ClipModel::new(vb, &cfg).map_err(|e| {
                    InferenceError::Runtime(format!("Failed to build CLIP model: {e}"))
                })?;
                Ok(LoadedModel::Clip {
                    model,
                    tokenizer,
                    device: device.clone(),
                })
            }
            SupportedArchitecture::Vit => {
                let cfg: VitConfig = serde_json::from_str(&config_str).map_err(|e| {
                    InferenceError::InvalidInput(format!("Failed to parse ViT config: {e}"))
                })?;
                let num_labels = config_val
                    .get("id2label")
                    .and_then(|v| v.as_object())
                    .map(|o| o.len())
                    .or_else(|| {
                        config_val
                            .get("num_labels")
                            .and_then(|v| v.as_u64())
                            .map(|n| n as usize)
                    })
                    .unwrap_or(1000);
                let model = VitModel::new(&cfg, num_labels, vb).map_err(|e| {
                    InferenceError::Runtime(format!("Failed to build ViT model: {e}"))
                })?;
                Ok(LoadedModel::Vit {
                    model,
                    device: device.clone(),
                })
            }
            SupportedArchitecture::DinoV2 => {
                let cfg: DinoV2Config =
                    serde_json::from_str(&config_str).unwrap_or_else(|_| DinoV2Config {
                        num_hidden_layers: default_dinov2_depth(),
                        hidden_size: default_dinov2_embed_dim(),
                        num_attention_heads: default_dinov2_num_heads(),
                    });
                let model = candle_transformers::models::dinov2::DinoVisionTransformer::new(
                    vb,
                    cfg.num_hidden_layers,
                    cfg.hidden_size,
                    cfg.num_attention_heads,
                )
                .map_err(|e| {
                    InferenceError::Runtime(format!("Failed to build DINOv2 model: {e}"))
                })?;
                Ok(LoadedModel::DinoV2 {
                    model,
                    device: device.clone(),
                })
            }
        }
    }
}

/// Resolves directory and weights file from a candidate path.
fn resolve_paths(path: &Path) -> Result<(PathBuf, PathBuf), InferenceError> {
    if path.is_file() {
        let parent = path
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("."));
        Ok((parent, path.to_path_buf()))
    } else if path.is_dir() {
        let direct_file = path.join("model.safetensors");
        if direct_file.is_file() {
            return Ok((path.to_path_buf(), direct_file));
        }

        // Search for any .safetensors file in the directory
        if let Ok(entries) = std::fs::read_dir(path) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.is_file() && p.extension().map(|e| e == "safetensors").unwrap_or(false) {
                    return Ok((path.to_path_buf(), p));
                }
            }
        }

        Err(InferenceError::NotFound(format!(
            "No .safetensors weights file found in directory '{}'",
            path.display()
        )))
    } else {
        Err(InferenceError::NotFound(format!(
            "Model path '{}' does not exist on filesystem",
            path.display()
        )))
    }
}

/// Extracts the architecture identifier string from `config.json`.
fn extract_architecture_string(val: &serde_json::Value) -> Option<String> {
    // 1. Prefer explicit "model_type"
    if let Some(mt) = val.get("model_type").and_then(|v| v.as_str()) {
        let trimmed = mt.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_string());
        }
    }

    // 2. Check "architectures[0]"
    if let Some(archs) = val.get("architectures").and_then(|a| a.as_array()) {
        if let Some(first) = archs.first().and_then(|v| v.as_str()) {
            let trimmed = first.trim();
            if !trimmed.is_empty() {
                // Map common PascalCase HF class names to lowercase identifiers
                let lower = trimmed.to_ascii_lowercase();
                if lower.contains("roberta") {
                    return Some("roberta".to_string());
                } else if lower.contains("distilbert") {
                    return Some("distilbert".to_string());
                } else if lower.contains("nomic") {
                    return Some("nomic_bert".to_string());
                } else if lower.contains("jina") {
                    return Some("jina_bert".to_string());
                } else if lower.contains("dinov2") {
                    return Some("dinov2".to_string());
                } else if lower.contains("clip") {
                    return Some("clip".to_string());
                } else if lower.contains("vit") {
                    return Some("vit".to_string());
                } else if lower.contains("bert") {
                    return Some("bert".to_string());
                }
                return Some(trimmed.to_string());
            }
        }
    }

    None
}

/// Computes mean pooling over token embeddings weighted by attention mask:
///
/// $$\text{embedding} = \frac{\sum (\text{token\_embeddings} \times \text{attention\_mask})}{\sum \text{attention\_mask}}$$
///
/// # Errors
/// Returns [`InferenceError::Runtime`] if tensor arithmetic fails.
pub fn mean_pooling(
    token_embeddings: &Tensor,
    attention_mask: &Tensor,
) -> Result<Tensor, InferenceError> {
    // attention_mask shape: [batch_size, seq_len] -> unsqueeze to [batch_size, seq_len, 1]
    let mask = attention_mask
        .to_dtype(DType::F32)
        .map_err(|e| {
            InferenceError::Runtime(format!("Failed to convert attention mask to f32: {e}"))
        })?
        .unsqueeze(2)
        .map_err(|e| InferenceError::Runtime(format!("Failed to unsqueeze attention mask: {e}")))?;

    // Multiply token embeddings [B, S, H] by mask [B, S, 1]
    let masked = token_embeddings
        .broadcast_mul(&mask)
        .map_err(|e| InferenceError::Runtime(format!("Failed to apply attention mask: {e}")))?;

    // Sum token embeddings across seq_len (dim 1) -> [B, H]
    let sum_embeddings = masked
        .sum(1)
        .map_err(|e| InferenceError::Runtime(format!("Failed to sum masked embeddings: {e}")))?;

    // Sum mask across seq_len (dim 1) -> [B, 1]
    let sum_mask = mask
        .sum(1)
        .map_err(|e| InferenceError::Runtime(format!("Failed to sum attention mask: {e}")))?;

    // Clamp sum_mask to minimum 1e-9 to prevent division by zero
    let eps_tensor = Tensor::new(1e-9f32, sum_mask.device())
        .map_err(|e| InferenceError::Runtime(format!("Failed to create eps tensor: {e}")))?;
    let clamped_sum_mask = sum_mask
        .broadcast_maximum(&eps_tensor)
        .map_err(|e| InferenceError::Runtime(format!("Failed to clamp mask sum: {e}")))?;

    // Division -> [B, H]
    sum_embeddings
        .broadcast_div(&clamped_sum_mask)
        .map_err(|e| InferenceError::Runtime(format!("Failed to divide by mask sum: {e}")))
}

/// Computes L2 normalization:
///
/// $$\mathbf{v}_{\text{norm}} = \frac{\mathbf{v}}{\Vert{}\mathbf{v}\Vert{}_2}$$
///
/// # Errors
/// Returns [`InferenceError::Runtime`] if tensor arithmetic fails.
pub fn l2_normalize(embeddings: &Tensor) -> Result<Tensor, InferenceError> {
    // embeddings: [batch_size, hidden_size]
    // norm = sqrt(sum(v^2)) keep dim -> [batch_size, 1]
    let norm = embeddings
        .sqr()
        .map_err(|e| InferenceError::Runtime(format!("Failed to square embeddings: {e}")))?
        .sum_keepdim(1)
        .map_err(|e| InferenceError::Runtime(format!("Failed to sum squared embeddings: {e}")))?
        .sqrt()
        .map_err(|e| InferenceError::Runtime(format!("Failed to compute norm sqrt: {e}")))?;

    let eps_tensor = Tensor::new(1e-12f32, norm.device())
        .map_err(|e| InferenceError::Runtime(format!("Failed to create eps tensor: {e}")))?;
    let clamped_norm = norm
        .broadcast_maximum(&eps_tensor)
        .map_err(|e| InferenceError::Runtime(format!("Failed to clamp norm: {e}")))?;

    embeddings
        .broadcast_div(&clamped_norm)
        .map_err(|e| InferenceError::Runtime(format!("Failed to normalize embeddings: {e}")))
}

/// Helper to parse Hugging Face CLIP config parameters into Candle ClipConfig.
fn parse_clip_config(val: &serde_json::Value) -> ClipConfig {
    let mut cfg = ClipConfig::vit_base_patch32();
    if let Some(t_val) = val.get("text_config") {
        if let Some(v) = t_val.get("vocab_size").and_then(|v| v.as_u64()) {
            cfg.text_config.vocab_size = v as usize;
        }
        if let Some(v) = t_val.get("hidden_size").and_then(|v| v.as_u64()) {
            cfg.text_config.embed_dim = v as usize;
        }
        if let Some(v) = t_val.get("intermediate_size").and_then(|v| v.as_u64()) {
            cfg.text_config.intermediate_size = v as usize;
        }
        if let Some(v) = t_val.get("num_hidden_layers").and_then(|v| v.as_u64()) {
            cfg.text_config.num_hidden_layers = v as usize;
        }
        if let Some(v) = t_val.get("num_attention_heads").and_then(|v| v.as_u64()) {
            cfg.text_config.num_attention_heads = v as usize;
        }
        if let Some(v) = t_val
            .get("max_position_embeddings")
            .and_then(|v| v.as_u64())
        {
            cfg.text_config.max_position_embeddings = v as usize;
        }
        if let Some(v) = t_val.get("projection_dim").and_then(|v| v.as_u64()) {
            cfg.text_config.projection_dim = v as usize;
        }
    }
    if let Some(v_val) = val.get("vision_config") {
        if let Some(v) = v_val.get("hidden_size").and_then(|v| v.as_u64()) {
            cfg.vision_config.embed_dim = v as usize;
        }
        if let Some(v) = v_val.get("intermediate_size").and_then(|v| v.as_u64()) {
            cfg.vision_config.intermediate_size = v as usize;
        }
        if let Some(v) = v_val.get("num_hidden_layers").and_then(|v| v.as_u64()) {
            cfg.vision_config.num_hidden_layers = v as usize;
        }
        if let Some(v) = v_val.get("num_attention_heads").and_then(|v| v.as_u64()) {
            cfg.vision_config.num_attention_heads = v as usize;
        }
        if let Some(v) = v_val.get("image_size").and_then(|v| v.as_u64()) {
            cfg.vision_config.image_size = v as usize;
            cfg.image_size = v as usize;
        }
        if let Some(v) = v_val.get("patch_size").and_then(|v| v.as_u64()) {
            cfg.vision_config.patch_size = v as usize;
        }
        if let Some(v) = v_val.get("projection_dim").and_then(|v| v.as_u64()) {
            cfg.vision_config.projection_dim = v as usize;
        }
    }
    if let Some(scale) = val.get("logit_scale_init_value").and_then(|v| v.as_f64()) {
        cfg.logit_scale_init_value = scale as f32;
    }
    cfg
}
