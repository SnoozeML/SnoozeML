//! Tree model execution backend supporting XGBoost JSON and native gradient boosted trees.
//!
//! Provides ultra-low latency, pure-Rust decision tree and gradient-boosted ensemble inference
//! without C/C++ dependencies or FFI bindings, powered by `hessboost`.

use crate::backend::ModelBackend;
use crate::types::{InferenceError, TensorPayload};
use hessboost::data::DMatrix;
use hessboost::learner::BoostedModel;
use std::path::Path;

/// Execution device target for the Tree model backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TreeDevice {
    /// Host CPU compute execution using multi-threaded Rayon.
    #[default]
    Cpu,
    /// Automatic hardware device selection (defaults to CPU).
    Auto,
}

/// Prediction output representation for tree model inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TreePredictType {
    /// Transformed predictions (probabilities for logistic/multiclass, raw values for regression).
    #[default]
    Value,
    /// Raw un-transformed margins / log-odds prior to activation functions.
    Margin,
    /// Argmax discrete class predictions (0, 1, 2, ...).
    Class,
}

/// Configuration options for initializing a Tree model backend.
#[derive(Debug, Clone)]
pub struct TreeConfig {
    /// Filesystem path to the model file (`.json`, `.xgb.json`, `.tree.json`, `.xgb`, `.bin`).
    pub model_path: String,
    /// Target execution compute device.
    pub device: TreeDevice,
    /// Expected input tensor node name (default: `"features"`).
    pub input_name: Option<String>,
    /// Produced output tensor node name (default: `"predictions"`).
    pub output_name: Option<String>,
    /// Prediction mode: value/probabilities, raw margin, or discrete class.
    pub predict_type: TreePredictType,
}

impl Default for TreeConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            device: TreeDevice::Auto,
            input_name: None,
            output_name: None,
            predict_type: TreePredictType::Value,
        }
    }
}

/// Tree model execution backend wrapping a `hessboost::learner::BoostedModel`.
pub struct TreeBackend {
    model: BoostedModel,
    config: TreeConfig,
    n_features: usize,
    n_outputs: usize,
    objective: String,
}

impl TreeBackend {
    /// Instantiates a new Tree model backend from configuration.
    ///
    /// Supports:
    /// - Standard XGBoost JSON models exported via `booster.save_model("model.json")`.
    /// - Native `hessboost` JSON format.
    /// - Native `hessboost` binary format (`.bin`).
    ///
    /// # Errors
    /// Returns [`InferenceError::NotFound`] if the model file is missing, or
    /// [`InferenceError::Runtime`] if parsing, deserialization, or initialization fails.
    pub fn new(config: &TreeConfig) -> Result<Self, InferenceError> {
        let path = Path::new(&config.model_path);
        if !path.exists() {
            return Err(InferenceError::NotFound(format!(
                "Tree model file not found at: {}",
                config.model_path
            )));
        }

        let model = Self::load_boosted_model(path)?;
        let n_features = model.n_features();
        let n_outputs = model.n_outputs();
        let objective = model.objective().to_string();

        tracing::info!(
            path = %config.model_path,
            n_features,
            n_outputs,
            objective = %objective,
            "Initialized pure-Rust Tree model backend successfully"
        );

        Ok(Self {
            model,
            config: config.clone(),
            n_features,
            n_outputs,
            objective,
        })
    }

    /// Loads the underlying `BoostedModel` from disk with format fallback.
    fn load_boosted_model(path: &Path) -> Result<BoostedModel, InferenceError> {
        let ext = path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();

        // 1. Binary format if extension is .bin
        if ext == "bin" {
            return BoostedModel::load_binary(path).map_err(|e| {
                InferenceError::Runtime(format!(
                    "Failed to load binary tree model from '{}': {e}",
                    path.display()
                ))
            });
        }

        // 2. Read string for JSON formats
        let content = std::fs::read_to_string(path).map_err(|e| {
            InferenceError::Runtime(format!(
                "Failed to read tree model file at '{}': {e}",
                path.display()
            ))
        })?;

        // 3. Attempt XGBoost JSON format first (standard across Python / XGBoost ecosystem)
        match BoostedModel::from_xgboost_json(&content) {
            Ok(m) => Ok(m),
            Err(xgb_err) => {
                // 4. Fallback to native hessboost JSON format
                match BoostedModel::from_json(&content) {
                    Ok(m) => Ok(m),
                    Err(json_err) => {
                        // 5. Try binary fallback as last resort
                        if let Ok(bytes) = std::fs::read(path) {
                            if let Ok(m) = BoostedModel::from_bytes(&bytes) {
                                return Ok(m);
                            }
                        }

                        Err(InferenceError::Runtime(format!(
                            "Failed to parse tree model from '{}'. XGBoost JSON error: {xgb_err}; Native JSON error: {json_err}",
                            path.display()
                        )))
                    }
                }
            }
        }
    }

    /// Returns the number of input features expected by this tree model.
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Returns the number of outputs produced per instance by this tree model.
    #[must_use]
    pub fn n_outputs(&self) -> usize {
        self.n_outputs
    }

    /// Returns the objective function name configured for this tree model.
    #[must_use]
    pub fn objective(&self) -> &str {
        &self.objective
    }
}

impl ModelBackend for TreeBackend {
    /// Executes batched inference on the provided input tensor payload.
    ///
    /// Accepts 2D payloads with shape `[batch_size, n_features]` or 1D payloads with
    /// shape `[n_features]` (treated as `[1, n_features]`).
    ///
    /// # Errors
    /// Returns:
    /// - [`InferenceError::InvalidInput`] if input count is 0, batch size is 0, or feature count mismatches.
    /// - [`InferenceError::ShapeMismatch`] if tensor dimensionality is neither 1D nor 2D.
    /// - [`InferenceError::Runtime`] if matrix construction or tree traversal fails.
    fn infer(&self, inputs: &[TensorPayload]) -> Result<Vec<TensorPayload>, InferenceError> {
        let input = inputs.first().ok_or_else(|| {
            InferenceError::InvalidInput("TreeBackend requires at least 1 input tensor".to_string())
        })?;

        let (batch_size, feature_count) = match input.shape.as_slice() {
            [b, f] => (*b, *f),
            [f] => (1usize, *f),
            other => {
                return Err(InferenceError::ShapeMismatch(format!(
                    "TreeBackend expects 1D [features] or 2D [batch_size, features] tensor, got shape {other:?}"
                )));
            }
        };

        if batch_size == 0 {
            return Err(InferenceError::InvalidInput(
                "Batch size cannot be zero".to_string(),
            ));
        }

        if feature_count != self.n_features {
            return Err(InferenceError::InvalidInput(format!(
                "Feature dimension mismatch: tree model expects {} features, but input payload provided {}",
                self.n_features, feature_count
            )));
        }

        // Zero-copy continuous slice extraction if standard layout, otherwise safe owned buffer
        let flat_buffer: Vec<f32>;
        let data_slice: &[f32] = if let Some(slice) = input.data.as_slice() {
            slice
        } else {
            flat_buffer = input.data.iter().copied().collect();
            &flat_buffer
        };

        let dmatrix =
            DMatrix::from_dense(data_slice, batch_size, self.n_features).map_err(|e| {
                InferenceError::Runtime(format!(
                    "Failed to construct DMatrix for tree inference: {e}"
                ))
            })?;

        let preds = match self.config.predict_type {
            TreePredictType::Value => self.model.predict(&dmatrix),
            TreePredictType::Margin => self.model.predict_margin(&dmatrix),
            TreePredictType::Class => {
                let classes = self.model.predict_class(&dmatrix).map_err(|e| {
                    InferenceError::Runtime(format!("Tree class prediction failed: {e}"))
                })?;
                Ok(classes.into_iter().map(|c| c as f32).collect())
            }
        }
        .map_err(|e| InferenceError::Runtime(format!("Tree prediction failed: {e}")))?;

        let expected_elements = batch_size
            .checked_mul(self.n_outputs)
            .ok_or_else(|| InferenceError::InvalidInput("Batch element count overflow".into()))?;

        if preds.len() != expected_elements {
            return Err(InferenceError::Runtime(format!(
                "Prediction result size mismatch: expected {expected_elements} elements, got {}",
                preds.len()
            )));
        }

        // Micro-batcher requires leading dimension to represent batch size
        let output_shape = if self.n_outputs > 1 {
            vec![batch_size, self.n_outputs]
        } else {
            vec![batch_size, 1]
        };

        let output_name = self
            .config
            .output_name
            .as_deref()
            .unwrap_or("predictions")
            .to_string();

        let payload = TensorPayload::from_shape_vec(output_name, output_shape, preds)?;
        Ok(vec![payload])
    }

    fn provider_name(&self) -> &'static str {
        "tree-cpu"
    }
}
