//! Integration and unit tests for model execution backend engines (Mock, ONNX Runtime, Candle).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use snooze_ml::backend::{create_backend, MockBackend, ModelBackend};
#[cfg(feature = "ort-backend")]
use snooze_ml::backend::{ExecutionProvider, OrtBackend, OrtConfig};
use snooze_ml::types::{InferenceError, TensorPayload};

// ============================================================================
// Mock Backend Tests
// ============================================================================

#[test]
fn test_mock_backend_infer() {
    let backend = MockBackend::new();
    assert_eq!(backend.provider_name(), "mock-engine");
    let payload = TensorPayload::from_shape_vec("in", vec![2, 3], vec![1.0; 6]).unwrap();
    let result = backend.infer(&[payload]).unwrap();
    assert_eq!(result.len(), 1);
    let out = &result[0];
    assert_eq!(out.name, "output");
    assert_eq!(out.shape, vec![2, 4]);
    assert_eq!(out.data.len(), 8);
}

#[test]
fn test_mock_backend_custom_provider() {
    let backend = MockBackend::with_provider("custom-test-engine");
    assert_eq!(backend.provider_name(), "custom-test-engine");
}

#[test]
fn test_mock_backend_empty_inputs_error() {
    let backend = MockBackend::new();
    let result = backend.infer(&[]);
    assert!(
        matches!(result, Err(InferenceError::InvalidInput(msg)) if msg.contains("No input tensors"))
    );
}

#[test]
fn test_mock_backend_zero_batch_size_error() {
    let backend = MockBackend::new();
    let payload = TensorPayload::from_shape_vec("scalar", vec![], vec![1.0]).unwrap();
    assert_eq!(payload.batch_size(), 0);
    let result = backend.infer(&[payload]);
    assert!(
        matches!(result, Err(InferenceError::InvalidInput(msg)) if msg.contains("0 batch size"))
    );
}

#[test]
fn test_mock_backend_values_verification() {
    let backend = MockBackend::new();
    let payload = TensorPayload::from_shape_vec("in", vec![3, 2], vec![0.0; 6]).unwrap();
    let result = backend.infer(&[payload]).unwrap();
    assert_eq!(result.len(), 1);
    let out = &result[0];
    assert_eq!(out.shape, vec![3, 4]);
    let data: Vec<f32> = out.data.iter().copied().collect();
    assert_eq!(
        data,
        vec![0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0, 20.0, 21.0, 22.0, 23.0]
    );
}

// ============================================================================
// Factory create_backend Tests
// ============================================================================

#[test]
fn test_create_backend_mock() {
    let expected_auto = if cfg!(feature = "cuda") {
        "mock-gpu"
    } else if cfg!(target_os = "macos") {
        "mock-mlx"
    } else {
        "mock-cpu"
    };

    let b1 = create_backend("mock", None).unwrap();
    assert_eq!(b1.provider_name(), expected_auto);

    let b2 = create_backend("mock:transformer", Some("auto")).unwrap();
    assert_eq!(b2.provider_name(), expected_auto);

    let b_gpu = create_backend("mock", Some("gpu")).unwrap();
    assert_eq!(b_gpu.provider_name(), "mock-gpu");

    let b_mlx = create_backend("mock", Some("mlx")).unwrap();
    assert_eq!(b_mlx.provider_name(), "mock-mlx");

    let b_cpu = create_backend("mock", Some("cpu")).unwrap();
    assert_eq!(b_cpu.provider_name(), "mock-cpu");
}

#[test]
fn test_create_backend_non_existent_file() {
    let res = create_backend("/tmp/snooze_non_existent_model_12345.onnx", None);
    #[cfg(feature = "ort-backend")]
    assert!(matches!(res, Err(InferenceError::NotFound(_))));
    #[cfg(not(feature = "ort-backend"))]
    assert!(matches!(res, Err(InferenceError::Runtime(_))));
}

#[test]
fn test_create_backend_provider_preferences() {
    for prov in &[
        "auto", "cpu", "cuda", "gpu", "mlx", "metal", "coreml", "tensorrt", "unknown",
    ] {
        let res = create_backend("/tmp/snooze_non_existent_model_prov.onnx", Some(prov));
        #[cfg(feature = "ort-backend")]
        assert!(matches!(res, Err(InferenceError::NotFound(_))));
        #[cfg(not(feature = "ort-backend"))]
        assert!(matches!(res, Err(InferenceError::Runtime(_))));
    }
}

#[test]
fn test_create_backend_safetensors_missing_file() {
    let res = create_backend("/tmp/snooze_non_existent.safetensors", None);
    #[cfg(feature = "candle-backend")]
    assert!(matches!(res, Err(InferenceError::NotFound(_))));
    #[cfg(not(feature = "candle-backend"))]
    assert!(matches!(res, Err(InferenceError::Runtime(_))));
}

// ============================================================================
// ONNX Runtime Backend Tests
// ============================================================================

#[cfg(feature = "ort-backend")]
#[test]
fn test_ort_config_default() {
    let config = OrtConfig::default();
    assert_eq!(config.model_path, "");
    assert_eq!(config.provider, ExecutionProvider::Auto);
    assert_eq!(config.intra_threads, 1);
    assert_eq!(config.inter_threads, 1);
}

#[cfg(feature = "ort-backend")]
#[test]
fn test_ort_backend_not_found() {
    let config = OrtConfig {
        model_path: "/tmp/snooze_definitely_missing_model.onnx".into(),
        provider: ExecutionProvider::Cpu,
        intra_threads: 1,
        inter_threads: 1,
    };
    let err = OrtBackend::new(&config).err().unwrap();
    assert!(matches!(err, InferenceError::NotFound(_)));
}

#[cfg(feature = "ort-backend")]
#[test]
fn test_ort_backend_corrupt_file() {
    use std::io::Write;

    let mut temp_path = std::env::temp_dir();
    temp_path.push(format!(
        "corrupt_model_{}.onnx",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));

    let mut file = std::fs::File::create(&temp_path).unwrap();
    file.write_all(b"this is definitely not a valid protobuf ONNX model file")
        .unwrap();
    file.sync_all().unwrap();
    drop(file);

    let config = OrtConfig {
        model_path: temp_path.to_string_lossy().to_string(),
        provider: ExecutionProvider::Cpu,
        intra_threads: 1,
        inter_threads: 1,
    };
    let res = OrtBackend::new(&config);
    let _ = std::fs::remove_file(&temp_path);

    assert!(matches!(res, Err(InferenceError::Runtime(_))));
}

#[cfg(feature = "ort-backend")]
#[test]
fn test_ort_backend_real_model_skippable() {
    use std::path::PathBuf;

    let candidate_path = std::env::var("SNOOZE_TEST_ONNX_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("tests/fixtures/test_model.onnx"));

    if !candidate_path.is_file() {
        println!(
            "SKIPPED: test_ort_backend_real_model_skippable: no test ONNX model found at {:?}. Set SNOOZE_TEST_ONNX_PATH to enable.",
            candidate_path
        );
        return;
    }

    let config = OrtConfig {
        model_path: candidate_path.to_string_lossy().to_string(),
        provider: ExecutionProvider::Cpu,
        intra_threads: 1,
        inter_threads: 1,
    };

    let backend = OrtBackend::new(&config).unwrap();
    assert_eq!(backend.provider_name(), "onnxruntime-cpu");
}

// ============================================================================
// Candle / Safetensors Backend Tests
// ============================================================================

#[cfg(feature = "candle-backend")]
mod candle_tests {
    use super::*;
    use candle_core::{Device, Tensor};
    use snooze_ml::backend::{CandleBackend, CandleConfig, CandleDevice};
    use std::collections::HashMap;
    use std::fs::File;
    use std::io::Write;
    use std::path::PathBuf;

    fn temp_model_path(prefix: &str) -> PathBuf {
        let unique_id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("{prefix}_{unique_id}.safetensors"))
    }

    #[test]
    fn test_candle_single_linear_inference() {
        let model_path = temp_model_path("single_linear");
        let dev = Device::Cpu;

        let w_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let weight = Tensor::from_slice(&w_data, (2, 3), &dev).unwrap();

        let b_data = vec![0.5f32, -0.5];
        let bias = Tensor::from_slice(&b_data, 2, &dev).unwrap();

        let mut tensors = HashMap::new();
        tensors.insert("weight".to_string(), weight);
        tensors.insert("bias".to_string(), bias);

        candle_core::safetensors::save(&tensors, &model_path).unwrap();

        let config = CandleConfig {
            model_path: model_path.to_str().unwrap().to_string(),
            device: CandleDevice::Cpu,
        };

        let backend = CandleBackend::new(&config).unwrap();
        assert_eq!(backend.provider_name(), "candle-cpu");

        let input_tensor =
            TensorPayload::from_shape_vec("input", vec![1, 3], vec![2.0, 1.0, 0.0]).unwrap();

        let results = backend.infer(&[input_tensor]).unwrap();
        assert_eq!(results.len(), 1);
        let out = &results[0];
        assert_eq!(out.name, "output");
        assert_eq!(out.shape, vec![1, 2]);

        let out_slice = out.data.as_slice().unwrap();
        assert!((out_slice[0] - 4.5).abs() < 1e-5);
        assert!((out_slice[1] - 12.5).abs() < 1e-5);

        let _ = std::fs::remove_file(&model_path);
    }

    #[test]
    fn test_candle_two_layer_mlp_with_relu() {
        let model_path = temp_model_path("two_layer_mlp");
        let dev = Device::Cpu;

        let w0_data = vec![1.0f32, -1.0, 2.0, 0.0, -3.0, 1.0, 0.5, 0.5];
        let w0 = Tensor::from_slice(&w0_data, (4, 2), &dev).unwrap();

        let w1_data = vec![1.0f32, 2.0, 1.0, -1.0];
        let w1 = Tensor::from_slice(&w1_data, (1, 4), &dev).unwrap();

        let mut tensors = HashMap::new();
        tensors.insert("0.weight".to_string(), w0);
        tensors.insert("1.weight".to_string(), w1);

        candle_core::safetensors::save(&tensors, &model_path).unwrap();

        let config = CandleConfig {
            model_path: model_path.to_str().unwrap().to_string(),
            device: CandleDevice::Cpu,
        };

        let backend = CandleBackend::new(&config).unwrap();

        let input_tensor =
            TensorPayload::from_shape_vec("input", vec![1, 2], vec![1.0, 2.0]).unwrap();

        let results = backend.infer(&[input_tensor]).unwrap();
        assert_eq!(results.len(), 1);
        let out = &results[0];
        assert_eq!(out.shape, vec![1, 1]);

        let out_val = out.data.as_slice().unwrap()[0];
        assert!((out_val - 2.5).abs() < 1e-5);

        let _ = std::fs::remove_file(&model_path);
    }

    #[test]
    fn test_candle_companion_config() {
        let model_path = temp_model_path("custom_config");
        let dev = Device::Cpu;

        let w_data = vec![2.0f32, -3.0];
        let weight = Tensor::from_slice(&w_data, (1, 2), &dev).unwrap();

        let mut tensors = HashMap::new();
        tensors.insert("projection.weight".to_string(), weight);
        candle_core::safetensors::save(&tensors, &model_path).unwrap();

        let config_path = model_path.with_extension("json");
        let json_content = r#"{
            "input_name": "client_in",
            "output_name": "client_out",
            "layers": [
                {
                    "weight": "projection.weight",
                    "activation": "none"
                }
            ]
        }"#;
        let mut f = File::create(&config_path).unwrap();
        f.write_all(json_content.as_bytes()).unwrap();

        let config = CandleConfig {
            model_path: model_path.to_str().unwrap().to_string(),
            device: CandleDevice::Cpu,
        };

        let backend = CandleBackend::new(&config).unwrap();
        let input = TensorPayload::from_shape_vec("client_in", vec![1, 2], vec![3.0, 1.0]).unwrap();

        let results = backend.infer(&[input]).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "client_out");
        assert!((results[0].data.as_slice().unwrap()[0] - 3.0).abs() < 1e-5);

        let _ = std::fs::remove_file(&model_path);
        let _ = std::fs::remove_file(&config_path);
    }

    #[test]
    fn test_candle_file_not_found() {
        let config = CandleConfig {
            model_path: "/tmp/non_existent_snooze_candle_model.safetensors".into(),
            device: CandleDevice::Cpu,
        };
        let res = CandleBackend::new(&config);
        assert!(matches!(res, Err(InferenceError::NotFound(_))));
    }

    #[test]
    fn test_candle_shape_mismatch_input() {
        let model_path = temp_model_path("mismatch_test");
        let dev = Device::Cpu;

        let w_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let weight = Tensor::from_slice(&w_data, (2, 2), &dev).unwrap();

        let mut tensors = HashMap::new();
        tensors.insert("weight".to_string(), weight);
        candle_core::safetensors::save(&tensors, &model_path).unwrap();

        let config = CandleConfig {
            model_path: model_path.to_str().unwrap().to_string(),
            device: CandleDevice::Cpu,
        };
        let backend = CandleBackend::new(&config).unwrap();

        let input =
            TensorPayload::from_shape_vec("input", vec![1, 3], vec![1.0, 2.0, 3.0]).unwrap();
        let res = backend.infer(&[input]);
        assert!(matches!(res, Err(InferenceError::ShapeMismatch(_))));

        let _ = std::fs::remove_file(&model_path);
    }

    #[test]
    fn test_candle_config_default() {
        let config = CandleConfig::default();
        assert_eq!(config.model_path, "");
        assert_eq!(config.device, CandleDevice::Auto);
    }

    #[test]
    fn test_candle_auto_and_mlx_device_selection() {
        let model_path = temp_model_path("auto_mlx_test");
        let dev = Device::Cpu;

        let w_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let weight = Tensor::from_slice(&w_data, (2, 2), &dev).unwrap();

        let mut tensors = HashMap::new();
        tensors.insert("weight".to_string(), weight);
        candle_core::safetensors::save(&tensors, &model_path).unwrap();

        // Test CandleDevice::Auto
        let config_auto = CandleConfig {
            model_path: model_path.to_str().unwrap().to_string(),
            device: CandleDevice::Auto,
        };
        let backend_auto = CandleBackend::new(&config_auto).unwrap();
        assert!(
            backend_auto.provider_name() == "candle-cuda"
                || backend_auto.provider_name() == "candle-mlx"
                || backend_auto.provider_name() == "candle-cpu"
        );

        // Test CandleDevice::Mlx (falls back to CPU if not macOS/metal)
        let config_mlx = CandleConfig {
            model_path: model_path.to_str().unwrap().to_string(),
            device: CandleDevice::Mlx,
        };
        let backend_mlx = CandleBackend::new(&config_mlx).unwrap();
        assert!(
            backend_mlx.provider_name() == "candle-mlx"
                || backend_mlx.provider_name() == "candle-cpu"
        );

        let _ = std::fs::remove_file(&model_path);
    }

    #[test]
    fn test_candle_hf_transformer_error_reporting() {
        let p = std::path::Path::new("models/chronos-bolt-mini/model.safetensors");
        if !p.exists() {
            return;
        }
        let res = create_backend("models/chronos-bolt-mini/model.safetensors", None);
        assert!(res.is_err());
        let err_msg = res.err().unwrap().to_string();
        assert!(
            err_msg.contains("ChronosBoltModelForForecasting") || err_msg.contains("t5"),
            "Expected error to mention transformer architecture, got: {err_msg}"
        );
        assert!(
            err_msg.contains("optimum-cli export onnx") || err_msg.contains("ONNX"),
            "Expected error to mention ONNX export, got: {err_msg}"
        );
    }
}
