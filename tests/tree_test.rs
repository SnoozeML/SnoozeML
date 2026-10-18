//! Behavioral, integration, and contract tests for the Tree model backend (`hessboost` / XGBoost).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use hessboost::data::DMatrix;
use hessboost::prelude::{train, TrainingParams};
use snooze_ml::backend::{
    create_backend, ModelBackend, TreeBackend, TreeConfig, TreeDevice, TreePredictType,
};
use snooze_ml::config::ModelDirectoryConfig;
use snooze_ml::model::{ModelConfig, ModelController};
use snooze_ml::types::{InferenceError, ModelState, TensorPayload};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Helper to create a temporary directory unique to a test.
fn temp_test_dir(name: &str) -> PathBuf {
    let mut dir = std::env::temp_dir();
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    dir.push(format!("snooze_tree_test_{name}_{unique}"));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Helper to train a small regression model and export in XGBoost JSON format.
fn train_test_regression_model(dir: &Path, filename: &str) -> PathBuf {
    let x = [
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let y = [3.0f32, 7.0, 11.0, 15.0, 19.0, 23.0];
    let dtrain = DMatrix::from_dense(&x, 6, 2)
        .unwrap()
        .with_labels(&y)
        .unwrap();

    let params = TrainingParams::builder()
        .objective("reg:squarederror")
        .max_depth(3)
        .eta(0.5)
        .build()
        .unwrap();

    let model = train(&params, &dtrain, 5).unwrap();
    let path = dir.join(filename);
    model.save_xgboost_json(&path).unwrap();
    path
}

/// Helper to train a binary classification model and export in XGBoost JSON format.
fn train_test_classification_model(dir: &Path, filename: &str) -> PathBuf {
    let x = [
        0.1f32, 0.2, 0.2f32, 0.1, 0.8f32, 0.9, 0.9f32, 0.8, 0.15f32, 0.25, 0.85f32, 0.75,
    ];
    let y = [0.0f32, 0.0, 1.0, 1.0, 0.0, 1.0];
    let dtrain = DMatrix::from_dense(&x, 6, 2)
        .unwrap()
        .with_labels(&y)
        .unwrap();

    let params = TrainingParams::builder()
        .objective("binary:logistic")
        .max_depth(2)
        .eta(1.0)
        .build()
        .unwrap();

    let model = train(&params, &dtrain, 4).unwrap();
    let path = dir.join(filename);
    model.save_xgboost_json(&path).unwrap();
    path
}

/// Helper to train a multiclass classification model (3 classes, 4 features).
fn train_test_multiclass_model(dir: &Path, filename: &str) -> PathBuf {
    // 6 rows, 4 features
    let x = [
        1.0f32, 0.1, 0.2, 0.1, 1.1f32, 0.2, 0.1, 0.2, 0.1f32, 1.0, 0.2, 0.1, 0.2f32, 1.1, 0.1, 0.2,
        0.1f32, 0.2, 1.0, 0.9, 0.2f32, 0.1, 1.1, 0.8,
    ];
    let y = [0.0f32, 0.0, 1.0, 1.0, 2.0, 2.0];
    let dtrain = DMatrix::from_dense(&x, 6, 4)
        .unwrap()
        .with_labels(&y)
        .unwrap();

    let params = TrainingParams::builder()
        .objective("multi:softprob")
        .num_class(3)
        .max_depth(2)
        .eta(1.0)
        .build()
        .unwrap();

    let model = train(&params, &dtrain, 3).unwrap();
    let path = dir.join(filename);
    model.save_xgboost_json(&path).unwrap();
    path
}

#[test]
fn test_tree_config_default() {
    let cfg = TreeConfig::default();
    assert_eq!(cfg.model_path, "");
    assert_eq!(cfg.device, TreeDevice::Auto);
    assert_eq!(cfg.input_name, None);
    assert_eq!(cfg.output_name, None);
    assert_eq!(cfg.predict_type, TreePredictType::Value);
}

#[test]
fn test_tree_backend_file_not_found() {
    let cfg = TreeConfig {
        model_path: "nonexistent_model_file.json".to_string(),
        device: TreeDevice::Cpu,
        input_name: None,
        output_name: None,
        predict_type: TreePredictType::Value,
    };

    let err = TreeBackend::new(&cfg).err().unwrap();
    match err {
        InferenceError::NotFound(msg) => {
            assert!(msg.contains("Tree model file not found"));
        }
        other => panic!("Expected NotFound error, got {other:?}"),
    }
}

#[test]
fn test_tree_backend_regression_inference() {
    let dir = temp_test_dir("regression");
    let model_path = train_test_regression_model(&dir, "reg_model.xgb.json");

    let cfg = TreeConfig {
        model_path: model_path.to_string_lossy().to_string(),
        device: TreeDevice::Cpu,
        input_name: Some("features".to_string()),
        output_name: Some("predictions".to_string()),
        predict_type: TreePredictType::Value,
    };

    let backend = TreeBackend::new(&cfg).unwrap();
    assert_eq!(backend.provider_name(), "tree-cpu");
    assert_eq!(backend.n_features(), 2);
    assert_eq!(backend.n_outputs(), 1);
    assert_eq!(backend.objective(), "reg:squarederror");

    // 2 samples, 2 features
    let input =
        TensorPayload::from_shape_vec("features", vec![2, 2], vec![2.0f32, 3.0, 6.0, 7.0]).unwrap();

    let outputs = backend.infer(&[input]).unwrap();
    assert_eq!(outputs.len(), 1);

    let out = &outputs[0];
    assert_eq!(out.name, "predictions");
    assert_eq!(out.shape, vec![2, 1]);
    assert_eq!(out.data.len(), 2);
    // Predictions should be reasonable positive numbers
    let vals: Vec<f32> = out.data.iter().copied().collect();
    assert!(vals[0] > 0.0);
    assert!(vals[1] > 0.0);
    assert!(vals[1] > vals[0]);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_tree_backend_binary_classification_inference() {
    let dir = temp_test_dir("binary_cls");
    let model_path = train_test_classification_model(&dir, "cls_model.json");

    let cfg = TreeConfig {
        model_path: model_path.to_string_lossy().to_string(),
        device: TreeDevice::Cpu,
        input_name: None,
        output_name: None,
        predict_type: TreePredictType::Value,
    };

    let backend = TreeBackend::new(&cfg).unwrap();
    assert_eq!(backend.n_features(), 2);
    assert_eq!(backend.n_outputs(), 1);
    assert_eq!(backend.objective(), "binary:logistic");

    // 2 samples: one likely class 0, one likely class 1
    let input = TensorPayload::from_shape_vec("input", vec![2, 2], vec![0.15f32, 0.15, 0.85, 0.85])
        .unwrap();

    let outputs = backend.infer(&[input]).unwrap();
    let out = &outputs[0];
    assert_eq!(out.shape, vec![2, 1]);
    let probs: Vec<f32> = out.data.iter().copied().collect();
    // Probabilities are bounded [0, 1]
    assert!(probs[0] >= 0.0 && probs[0] <= 1.0);
    assert!(probs[1] >= 0.0 && probs[1] <= 1.0);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_tree_backend_multiclass_inference() {
    let dir = temp_test_dir("multiclass");
    let model_path = train_test_multiclass_model(&dir, "multiclass.xgb.json");

    let cfg = TreeConfig {
        model_path: model_path.to_string_lossy().to_string(),
        device: TreeDevice::Cpu,
        input_name: None,
        output_name: Some("probabilities".to_string()),
        predict_type: TreePredictType::Value,
    };

    let backend = TreeBackend::new(&cfg).unwrap();
    assert_eq!(backend.n_features(), 4);
    assert_eq!(backend.n_outputs(), 3);
    assert_eq!(backend.objective(), "multi:softprob");

    // 2 samples, 4 features each
    let input = TensorPayload::from_shape_vec(
        "features",
        vec![2, 4],
        vec![
            1.0f32, 0.1, 0.2, 0.1, // matches class 0
            0.1f32, 0.2, 1.0, 0.9, // matches class 2
        ],
    )
    .unwrap();

    let outputs = backend.infer(&[input]).unwrap();
    assert_eq!(outputs.len(), 1);
    let out = &outputs[0];
    assert_eq!(out.name, "probabilities");
    assert_eq!(out.shape, vec![2, 3]);
    assert_eq!(out.data.len(), 6);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_tree_backend_1d_input_tensor_handling() {
    let dir = temp_test_dir("1d_input");
    let model_path = train_test_regression_model(&dir, "model_1d.json");

    let cfg = TreeConfig {
        model_path: model_path.to_string_lossy().to_string(),
        device: TreeDevice::Cpu,
        input_name: None,
        output_name: None,
        predict_type: TreePredictType::Value,
    };

    let backend = TreeBackend::new(&cfg).unwrap();

    // 1D tensor [2] with 2 features -> treated as 1 sample with 2 features
    let input = TensorPayload::from_shape_vec("sample", vec![2], vec![3.0f32, 4.0]).unwrap();
    let outputs = backend.infer(&[input]).unwrap();
    assert_eq!(outputs[0].shape, vec![1, 1]);
    assert_eq!(outputs[0].data.len(), 1);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_tree_backend_feature_count_mismatch() {
    let dir = temp_test_dir("feature_mismatch");
    let model_path = train_test_regression_model(&dir, "model_mismatch.json");

    let cfg = TreeConfig {
        model_path: model_path.to_string_lossy().to_string(),
        device: TreeDevice::Cpu,
        input_name: None,
        output_name: None,
        predict_type: TreePredictType::Value,
    };

    let backend = TreeBackend::new(&cfg).unwrap();

    // Model expects 2 features, provide 3 features
    let input = TensorPayload::from_shape_vec("wrong_features", vec![1, 3], vec![1.0f32, 2.0, 3.0])
        .unwrap();

    let err = backend.infer(&[input]).err().unwrap();
    match err {
        InferenceError::InvalidInput(msg) => {
            assert!(msg.contains("Feature dimension mismatch"));
        }
        other => panic!("Expected InvalidInput, got {other:?}"),
    }

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_tree_backend_invalid_dimensionality_3d() {
    let dir = temp_test_dir("dim_mismatch");
    let model_path = train_test_regression_model(&dir, "model_3d.json");

    let cfg = TreeConfig {
        model_path: model_path.to_string_lossy().to_string(),
        device: TreeDevice::Cpu,
        input_name: None,
        output_name: None,
        predict_type: TreePredictType::Value,
    };

    let backend = TreeBackend::new(&cfg).unwrap();

    // 3D tensor
    let input =
        TensorPayload::from_shape_vec("3d_tensor", vec![1, 1, 2], vec![1.0f32, 2.0]).unwrap();

    let err = backend.infer(&[input]).err().unwrap();
    match err {
        InferenceError::ShapeMismatch(msg) => {
            assert!(msg.contains("expects 1D [features] or 2D [batch_size, features]"));
        }
        other => panic!("Expected ShapeMismatch, got {other:?}"),
    }

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_tree_backend_zero_batch_size() {
    let dir = temp_test_dir("zero_batch");
    let model_path = train_test_regression_model(&dir, "model_zero.json");

    let cfg = TreeConfig {
        model_path: model_path.to_string_lossy().to_string(),
        device: TreeDevice::Cpu,
        input_name: None,
        output_name: None,
        predict_type: TreePredictType::Value,
    };

    let backend = TreeBackend::new(&cfg).unwrap();

    // 0 batch size
    let input = TensorPayload::from_shape_vec("zero_batch", vec![0, 2], vec![]).unwrap();
    let err = backend.infer(&[input]).err().unwrap();
    match err {
        InferenceError::InvalidInput(msg) => {
            assert!(msg.contains("Batch size cannot be zero"));
        }
        other => panic!("Expected InvalidInput, got {other:?}"),
    }

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_create_backend_factory_tree_selection() {
    let dir = temp_test_dir("factory");
    let model_path = train_test_regression_model(&dir, "my_model.xgb.json");

    // 1. Selection via extension (.xgb.json)
    let backend1 = create_backend(&model_path.to_string_lossy(), None).unwrap();
    assert_eq!(backend1.provider_name(), "tree-cpu");

    // 2. Selection via explicit provider "tree"
    let plain_json = dir.join("plain.json");
    std::fs::copy(&model_path, &plain_json).unwrap();
    let backend2 = create_backend(&plain_json.to_string_lossy(), Some("tree")).unwrap();
    assert_eq!(backend2.provider_name(), "tree-cpu");

    // 3. Selection via auto-detection of content in .json
    let backend3 = create_backend(&plain_json.to_string_lossy(), None).unwrap();
    assert_eq!(backend3.provider_name(), "tree-cpu");

    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn test_tree_backend_model_controller_lifecycle() {
    let dir = temp_test_dir("controller");
    let model_path = train_test_classification_model(&dir, "iris_tree.json");

    let config = ModelConfig {
        id: "iris-tree".to_string(),
        onnx_path: model_path.to_string_lossy().to_string(),
        idle_timeout: Duration::from_millis(50),
        max_batch_size: 4,
        max_batch_delay: Duration::from_millis(5),
        provider: Some("tree".to_string()),
        description: Some("Iris decision tree model".to_string()),
        min_residency: Duration::from_millis(10),
        autoscale_queue_threshold: 4,
        max_replicas: None,
        routing_strategy: snooze_ml::routing::RoutingStrategyType::LeastQueue,
        pinned: false,
    };

    let controller = ModelController::new(config);
    assert_eq!(controller.id(), "iris-tree");

    // Verify initial state is Idle
    let meta = controller.metadata().await;
    assert_eq!(meta.state, ModelState::Idle);
    assert!(!controller.is_loaded().await);

    // Run inference - triggers scale-up / rehydration
    let input =
        TensorPayload::from_shape_vec("features", vec![2, 2], vec![0.2f32, 0.3, 0.8, 0.9]).unwrap();

    let result = controller.infer(vec![input]).await.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].shape, vec![2, 1]);

    // Model is now loaded and Ready
    let meta_ready = controller.metadata().await;
    assert_eq!(meta_ready.state, ModelState::Ready);
    assert!(controller.is_loaded().await);

    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn test_tree_model_discovery_in_directory() {
    let dir = temp_test_dir("discovery");

    // Create an XGBoost model
    let _ = train_test_regression_model(&dir, "house_prices.xgb.json");

    // Create a plain config.json that should NOT be discovered as a model
    let config_file = dir.join("config.json");
    let mut f = File::create(&config_file).unwrap();
    f.write_all(b"{\"vocab_size\": 1000}").unwrap();

    let discovered = ModelDirectoryConfig::discover_models_in_dir(&dir)
        .await
        .unwrap();

    assert_eq!(discovered.len(), 1);
    assert_eq!(discovered[0].id, "house_prices");
    assert!(discovered[0]
        .relative_path
        .contains("house_prices.xgb.json"));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_tree_backend_load_python_xgboost_artifact() {
    let model_path = Path::new("models/test_iris.xgb.json");
    if !model_path.exists() {
        return;
    }

    let cfg = TreeConfig {
        model_path: model_path.to_string_lossy().to_string(),
        device: TreeDevice::Cpu,
        input_name: Some("features".to_string()),
        output_name: Some("predictions".to_string()),
        predict_type: TreePredictType::Value,
    };

    let backend = TreeBackend::new(&cfg).unwrap();
    assert_eq!(backend.n_features(), 4);
    assert_eq!(backend.n_outputs(), 3);
    assert_eq!(backend.provider_name(), "tree-cpu");

    // Single Iris sample: 5.1, 3.5, 1.4, 0.2
    let input =
        TensorPayload::from_shape_vec("features", vec![1, 4], vec![5.1f32, 3.5, 1.4, 0.2]).unwrap();

    let outputs = backend.infer(&[input]).unwrap();
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].shape, vec![1, 3]);

    let probs: Vec<f32> = outputs[0].data.iter().copied().collect();
    assert_eq!(probs.len(), 3);
    let sum: f32 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-4);

    // Clean up test files
    let _ = std::fs::remove_file("models/test_iris.xgb.json");
    let _ = std::fs::remove_file("models/test_iris_sample.json");
}
