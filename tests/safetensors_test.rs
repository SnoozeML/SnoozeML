//! Integration and acceptance tests for native Safetensors model loading via Candle.
//!
//! Validates:
//! - Direct folder loading for BERT (`all-MiniLM-L6-v2`) matching Hugging Face reference within 10^-4.
//! - Loading `nomic-embed-text-v1` via `nomic_bert`.
//! - Clean error reporting for unsupported architectures (`t5`, `llama`, `chronos-bolt`).
//! - Exact numerical correctness of Mean Pooling and L2 Normalization.
//! - Compute execution on CPU and CUDA where available.

#![cfg(feature = "candle-backend")]

use candle_core::{Device, Tensor};
use snooze_ml::backend::local_loader::{
    l2_normalize, mean_pooling, LoadedModel, LocalModelLoader, SupportedArchitecture,
};
use snooze_ml::backend::{create_backend, CandleBackend, CandleConfig, CandleDevice, ModelBackend};
use snooze_ml::types::{InferenceError, TensorPayload};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Helper to generate a temporary directory path that cleans up automatically.
struct TempDirGuard {
    path: PathBuf,
}

impl TempDirGuard {
    fn new(name: &str) -> Self {
        let mut path = std::env::temp_dir();
        let unique = format!(
            "snooze_st_test_{name}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        path.push(unique);
        std::fs::create_dir_all(&path).unwrap();
        Self { path }
    }
}

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

// ============================================================================
// Acceptance Criterion 1: all-MiniLM-L6-v2 Embeddings Accuracy
// ============================================================================

#[test]
fn test_all_minilm_l6_v2_embeddings_matching_hf_reference() {
    let model_dir = Path::new("models/all-MiniLM-L6-v2");
    if !model_dir.exists() {
        eprintln!("Skipping test: models/all-MiniLM-L6-v2 not present");
        return;
    }

    let dev = Device::Cpu;
    let loaded = LocalModelLoader::load(model_dir, &dev).unwrap();

    // Verify model variant
    match &loaded {
        LoadedModel::Bert {
            model: _,
            tokenizer,
            device: _,
        } => {
            assert!(
                tokenizer.is_some(),
                "Tokenizer should be loaded from tokenizer.json"
            );
        }
        _ => panic!("Expected LoadedModel::Bert for all-MiniLM-L6-v2"),
    }

    // Load Python reference embedding
    let ref_path = Path::new("tests/fixtures/reference_all_minilm.json");
    assert!(ref_path.is_file(), "Reference embedding fixture must exist");
    let ref_str = std::fs::read_to_string(ref_path).unwrap();
    let ref_val: serde_json::Value = serde_json::from_str(&ref_str).unwrap();

    let text = ref_val.get("text").unwrap().as_str().unwrap();
    let expected_dim = ref_val.get("dim").unwrap().as_u64().unwrap() as usize;
    assert_eq!(expected_dim, 384, "all-MiniLM-L6-v2 dimension must be 384");

    let expected_emb: Vec<f32> = ref_val
        .get("embedding")
        .unwrap()
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    assert_eq!(expected_emb.len(), 384);

    // Compute embedding using Rust pipeline
    let payload = loaded.encode_text(&[text]).unwrap();
    assert_eq!(payload.shape, vec![1, 384]);

    let actual_slice = payload.data.as_slice().unwrap();
    assert_eq!(actual_slice.len(), 384);

    // Verify L2 norm is 1.0 within tolerance
    let l2_norm: f32 = actual_slice.iter().map(|&x| x * x).sum::<f32>().sqrt();
    assert!(
        (l2_norm - 1.0).abs() < 1e-4,
        "Embedding vector must be unit normalized, got norm: {l2_norm}"
    );

    // Compare each dimension against Hugging Face reference with 10^-4 tolerance
    let mut max_diff: f32 = 0.0;
    for (i, (&actual, &expected)) in actual_slice.iter().zip(expected_emb.iter()).enumerate() {
        let diff = (actual - expected).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        assert!(
            diff < 1e-4,
            "Dimension {i} mismatch: actual={actual}, expected={expected}, diff={diff} exceeds 1e-4"
        );
    }

    println!("all-MiniLM-L6-v2 maximum difference across 384 dimensions: {max_diff}");
    assert!(max_diff < 1e-4);
}

#[test]
fn test_all_minilm_via_candle_backend_infer() {
    let model_dir = Path::new("models/all-MiniLM-L6-v2");
    if !model_dir.exists() {
        eprintln!("Skipping test: models/all-MiniLM-L6-v2 not present");
        return;
    }

    let config = CandleConfig {
        model_path: "models/all-MiniLM-L6-v2".to_string(),
        device: CandleDevice::Cpu,
    };
    let backend = CandleBackend::new(&config).unwrap();

    // Verify encode_text helper
    let out = backend.encode_text(&["Hello world!"]).unwrap();
    assert_eq!(out.shape, vec![1, 384]);

    // Verify infer method with input_ids and attention_mask tensors
    let input_ids = TensorPayload::from_shape_vec(
        "input_ids",
        vec![1, 5],
        vec![101.0, 7592.0, 2088.0, 999.0, 102.0], // [CLS] Hello world! [SEP]
    )
    .unwrap();
    let attention_mask =
        TensorPayload::from_shape_vec("attention_mask", vec![1, 5], vec![1.0, 1.0, 1.0, 1.0, 1.0])
            .unwrap();

    let outputs = backend.infer(&[input_ids, attention_mask]).unwrap();
    assert_eq!(outputs.len(), 1);
    let emb_payload = &outputs[0];
    assert_eq!(emb_payload.shape, vec![1, 384]);

    let slice = emb_payload.data.as_slice().unwrap();
    let norm: f32 = slice.iter().map(|&x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-4);
}

// ============================================================================
// Acceptance Criterion 2: Nomic BERT Loading
// ============================================================================

#[test]
fn test_nomic_embed_text_v1_loading() {
    let model_dir = Path::new("models/nomic-embed-text-v1");
    if !model_dir.exists() {
        eprintln!("Skipping test: models/nomic-embed-text-v1 not present");
        return;
    }

    let dev = Device::Cpu;
    let loaded = LocalModelLoader::load(model_dir, &dev).unwrap();

    // Verify model variant is NomicBert
    match &loaded {
        LoadedModel::NomicBert {
            model: _,
            tokenizer,
            device: _,
        } => {
            assert!(
                tokenizer.is_some(),
                "Tokenizer should be loaded for nomic-embed-text-v1"
            );
        }
        _ => panic!("Expected LoadedModel::NomicBert for nomic-embed-text-v1"),
    }

    // Run inference with text input
    let out = loaded
        .encode_text(&["Testing Nomic BERT embeddings"])
        .unwrap();
    assert_eq!(out.shape.len(), 2);
    assert_eq!(out.shape[0], 1);
    assert_eq!(out.shape[1], 768, "nomic-embed-text-v1 dimension is 768");

    let slice = out.data.as_slice().unwrap();
    let norm: f32 = slice.iter().map(|&x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-4,
        "Nomic BERT output vector must be unit normalized"
    );
}

// ============================================================================
// Acceptance Criterion 3: Unsupported Architecture Error Reporting
// ============================================================================

#[test]
fn test_unsupported_architecture_chronos_bolt_t5() {
    let model_dir = Path::new("models/chronos-bolt-mini");
    if !model_dir.exists() {
        eprintln!("Skipping test: models/chronos-bolt-mini not present");
        return;
    }

    let dev = Device::Cpu;
    // 1. Direct directory path
    let err = LocalModelLoader::load(model_dir, &dev).err().unwrap();
    match err {
        InferenceError::UnsupportedSafetensorsArchitecture(arch) => {
            assert_eq!(arch, "t5");
        }
        other => panic!("Expected UnsupportedSafetensorsArchitecture, got: {other:?}"),
    }

    // 2. Direct model.safetensors file path
    let file_path = model_dir.join("model.safetensors");
    let err_file = LocalModelLoader::load(&file_path, &dev).err().unwrap();
    match err_file {
        InferenceError::UnsupportedSafetensorsArchitecture(arch) => {
            assert_eq!(arch, "t5");
        }
        other => panic!("Expected UnsupportedSafetensorsArchitecture, got: {other:?}"),
    }

    // 3. Via create_backend factory
    let res = create_backend("models/chronos-bolt-mini", None);
    assert!(res.is_err());
    let be_err = res.err().unwrap();
    let err_msg = be_err.to_string();
    assert!(
        err_msg.contains("Unsupported Safetensors architecture: 't5'"),
        "Error message should clearly report unsupported arch 't5', got: {err_msg}"
    );
    assert!(
        err_msg.contains("ONNX"),
        "Error message should direct user to ONNX format, got: {err_msg}"
    );
}

#[test]
fn test_unsupported_architecture_synthetic_llama_and_chronos() {
    let temp = TempDirGuard::new("unsupported_arch");
    let dev = Device::Cpu;

    // Create dummy weights
    let dummy_weights_path = temp.path.join("model.safetensors");
    let mut tensors = HashMap::new();
    let dummy_t = Tensor::zeros((2, 2), candle_core::DType::F32, &dev).unwrap();
    tensors.insert("weight".to_string(), dummy_t);
    candle_core::safetensors::save(&tensors, &dummy_weights_path).unwrap();

    // Case A: model_type = "llama"
    let config_path = temp.path.join("config.json");
    let llama_config = serde_json::json!({
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"]
    });
    std::fs::write(&config_path, llama_config.to_string()).unwrap();

    let res_llama = LocalModelLoader::load(&temp.path, &dev);
    assert!(matches!(
        res_llama,
        Err(InferenceError::UnsupportedSafetensorsArchitecture(arch)) if arch == "llama"
    ));

    // Case B: model_type = "chronos-bolt"
    let chronos_config = serde_json::json!({
        "model_type": "chronos-bolt",
        "architectures": ["ChronosBoltModel"]
    });
    std::fs::write(&config_path, chronos_config.to_string()).unwrap();

    let res_chronos = LocalModelLoader::load(&temp.path, &dev);
    assert!(matches!(
        res_chronos,
        Err(InferenceError::UnsupportedSafetensorsArchitecture(arch)) if arch == "chronos-bolt"
    ));
}

// ============================================================================
// Acceptance Criterion 4: Mean Pooling & L2 Normalization Mathematical Proof
// ============================================================================

#[test]
fn test_mean_pooling_and_l2_normalization_math() {
    let dev = Device::Cpu;

    // Construct known tensor:
    // Batch size 1, 3 tokens, 2 features:
    // Token 0: [1.0, 2.0], mask: 1
    // Token 1: [3.0, 4.0], mask: 1
    // Token 2: [100.0, 200.0], mask: 0 (masked out padding token)
    let token_data = vec![
        1.0f32, 2.0, // token 0
        3.0, 4.0, // token 1
        100.0, 200.0, // token 2 (ignored)
    ];
    let embeddings = Tensor::from_slice(&token_data, (1, 3, 2), &dev).unwrap();
    let mask = Tensor::from_slice(&[1u32, 1, 0], (1, 3), &dev).unwrap();

    // 1. Mean Pooling
    // Expected sum: [1.0 + 3.0, 2.0 + 4.0] = [4.0, 6.0]
    // Expected mask sum: 1 + 1 = 2
    // Expected pooled: [4.0 / 2, 6.0 / 2] = [2.0, 3.0]
    let pooled = mean_pooling(&embeddings, &mask).unwrap();
    assert_eq!(pooled.dims(), &[1, 2]);

    let pooled_vec = pooled.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert!((pooled_vec[0] - 2.0).abs() < 1e-6);
    assert!((pooled_vec[1] - 3.0).abs() < 1e-6);

    // 2. L2 Normalization
    // Norm of [2.0, 3.0] = sqrt(4 + 9) = sqrt(13) ≈ 3.605551275
    // Normalized = [2.0 / sqrt(13), 3.0 / sqrt(13)] ≈ [0.5547002, 0.8320503]
    let normalized = l2_normalize(&pooled).unwrap();
    assert_eq!(normalized.dims(), &[1, 2]);

    let norm_vec = normalized.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let expected_0 = 2.0f32 / (13.0f32).sqrt();
    let expected_1 = 3.0f32 / (13.0f32).sqrt();

    assert!((norm_vec[0] - expected_0).abs() < 1e-6);
    assert!((norm_vec[1] - expected_1).abs() < 1e-6);

    // L2 norm of the normalized vector must be exactly 1.0
    let final_norm = (norm_vec[0] * norm_vec[0] + norm_vec[1] * norm_vec[1]).sqrt();
    assert!((final_norm - 1.0).abs() < 1e-6);
}

// ============================================================================
// Supported Architecture Identifier Resolution Tests
// ============================================================================

#[test]
fn test_supported_architecture_resolution() {
    assert_eq!(
        SupportedArchitecture::from_hf_identifier("bert"),
        Some(SupportedArchitecture::Bert)
    );
    assert_eq!(
        SupportedArchitecture::from_hf_identifier("roberta"),
        Some(SupportedArchitecture::Bert)
    );
    assert_eq!(
        SupportedArchitecture::from_hf_identifier("jina-bert"),
        Some(SupportedArchitecture::JinaBert)
    );
    assert_eq!(
        SupportedArchitecture::from_hf_identifier("jina_bert"),
        Some(SupportedArchitecture::JinaBert)
    );
    assert_eq!(
        SupportedArchitecture::from_hf_identifier("nomic_bert"),
        Some(SupportedArchitecture::NomicBert)
    );
    assert_eq!(
        SupportedArchitecture::from_hf_identifier("distilbert"),
        Some(SupportedArchitecture::DistilBert)
    );
    assert_eq!(
        SupportedArchitecture::from_hf_identifier("clip"),
        Some(SupportedArchitecture::Clip)
    );
    assert_eq!(
        SupportedArchitecture::from_hf_identifier("clip_text_model"),
        Some(SupportedArchitecture::Clip)
    );
    assert_eq!(
        SupportedArchitecture::from_hf_identifier("clip_vision_model"),
        Some(SupportedArchitecture::Clip)
    );
    assert_eq!(
        SupportedArchitecture::from_hf_identifier("vit"),
        Some(SupportedArchitecture::Vit)
    );
    assert_eq!(
        SupportedArchitecture::from_hf_identifier("dinov2"),
        Some(SupportedArchitecture::DinoV2)
    );

    // Unsupported architectures
    assert_eq!(SupportedArchitecture::from_hf_identifier("t5"), None);
    assert_eq!(SupportedArchitecture::from_hf_identifier("llama"), None);
    assert_eq!(
        SupportedArchitecture::from_hf_identifier("chronos-bolt"),
        None
    );
    assert_eq!(SupportedArchitecture::from_hf_identifier("mistral"), None);
    assert_eq!(SupportedArchitecture::from_hf_identifier("gpt2"), None);
}

// ============================================================================
// Acceptance Criterion 4: Device Execution (CPU and CUDA where available)
// ============================================================================

#[test]
fn test_device_execution_cpu_and_cuda() {
    let model_dir = Path::new("models/all-MiniLM-L6-v2");
    if !model_dir.exists() {
        eprintln!("Skipping device test: models/all-MiniLM-L6-v2 not present");
        return;
    }

    // 1. CPU execution
    let cpu_dev = Device::Cpu;
    let cpu_model = LocalModelLoader::load(model_dir, &cpu_dev).unwrap();
    let cpu_emb = cpu_model.encode_text(&["CPU execution test"]).unwrap();
    assert_eq!(cpu_emb.shape, vec![1, 384]);

    // 2. CUDA execution if GPU is physically available
    #[cfg(feature = "cuda")]
    {
        if let Ok(cuda_dev) = Device::new_cuda(0) {
            println!("CUDA GPU detected; running test on CUDA Device");
            let cuda_model = LocalModelLoader::load(model_dir, &cuda_dev).unwrap();
            let cuda_emb = cuda_model.encode_text(&["CUDA execution test"]).unwrap();
            assert_eq!(cuda_emb.shape, vec![1, 384]);

            // Ensure CUDA and CPU results match within tolerance
            let cpu_slice = cpu_emb.data.as_slice().unwrap();
            let cuda_slice = cuda_emb.data.as_slice().unwrap();
            for (c, g) in cpu_slice.iter().zip(cuda_slice.iter()) {
                assert!((c - g).abs() < 1e-4);
            }
        } else {
            println!(
                "CUDA feature enabled, but no physical CUDA GPU available; skipping CUDA execution"
            );
        }
    }
}
