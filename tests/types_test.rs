//! Integration and behavioral unit tests for domain types and payloads.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use snooze_ml::types::{InferenceError, ModelMetadata, ModelState, TensorPayload};

#[test]
fn test_tensor_payload_creation_valid() {
    let payload = TensorPayload::from_shape_vec("input_1", vec![1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    assert!(payload.is_ok());
    let p = payload.unwrap();
    assert_eq!(p.batch_size(), 1);
    assert_eq!(p.trailing_shape(), &[2, 2]);
}

#[test]
fn test_tensor_payload_creation_mismatch() {
    let payload = TensorPayload::from_shape_vec("input_1", vec![1, 2, 2], vec![1.0, 2.0, 3.0]);
    assert!(matches!(payload, Err(InferenceError::InvalidInput(_))));
}

#[test]
fn test_tensor_payload_empty_shape() {
    let payload = TensorPayload::from_shape_vec("empty", vec![], vec![42.0]);
    assert!(payload.is_ok());
    let p = payload.unwrap();
    assert_eq!(p.batch_size(), 0);
    assert_eq!(p.trailing_shape(), &[] as &[usize]);
}

#[test]
fn test_tensor_payload_shape_overflow() {
    let payload = TensorPayload::from_shape_vec("overflow", vec![usize::MAX, 2], vec![1.0]);
    assert!(matches!(payload, Err(InferenceError::InvalidInput(msg)) if msg.contains("overflow")));
}

#[test]
fn test_tensor_payload_1d_shape() {
    let payload = TensorPayload::from_shape_vec("1d", vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    assert_eq!(payload.batch_size(), 4);
    assert_eq!(payload.trailing_shape(), &[] as &[usize]);
}

#[test]
fn test_tensor_payload_multi_dim() {
    let data = vec![0.5f32; 2 * 3 * 4 * 5];
    let payload = TensorPayload::from_shape_vec("4d", vec![2, 3, 4, 5], data).unwrap();
    assert_eq!(payload.batch_size(), 2);
    assert_eq!(payload.trailing_shape(), &[3, 4, 5]);
}

#[test]
fn test_tensor_payload_serde_roundtrip() {
    let original =
        TensorPayload::from_shape_vec("t", vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let json = serde_json::to_string(&original).unwrap();
    let deserialized: TensorPayload = serde_json::from_str(&json).unwrap();
    assert_eq!(original, deserialized);
}

#[test]
fn test_inference_error_display() {
    assert_eq!(
        InferenceError::NotFound("resnet".into()).to_string(),
        "Model not found: resnet"
    );
    assert_eq!(
        InferenceError::Runtime("cuda error".into()).to_string(),
        "Inference runtime failure: cuda error"
    );
    assert_eq!(
        InferenceError::ShapeMismatch("mismatch".into()).to_string(),
        "Batch collation shape mismatch: mismatch"
    );
    assert_eq!(InferenceError::Timeout.to_string(), "Service timeout");
    assert_eq!(
        InferenceError::InvalidInput("bad data".into()).to_string(),
        "Invalid tensor input: bad data"
    );
    assert_eq!(
        InferenceError::Loading.to_string(),
        "Model is currently loading weights"
    );
    assert_eq!(
        InferenceError::QueueError("full".into()).to_string(),
        "Batch queue error: full"
    );
    assert_eq!(
        InferenceError::OutOfMemory("cuda vram full".into()).to_string(),
        "Device/System out of memory: cuda vram full"
    );
    assert_eq!(
        InferenceError::ResourceExhausted("all busy".into()).to_string(),
        "Resource limit reached: cannot allocate model even after deallocating idle sessions: all busy"
    );
    assert_eq!(
        InferenceError::UnsupportedSafetensorsArchitecture("llama".into()).to_string(),
        "Unsupported Safetensors architecture: 'llama'. Please convert and serve this model using ONNX format."
    );
}

#[test]
fn test_inference_error_serde_roundtrip() {
    let errors = vec![
        InferenceError::NotFound("m".into()),
        InferenceError::Runtime("r".into()),
        InferenceError::ShapeMismatch("s".into()),
        InferenceError::Timeout,
        InferenceError::InvalidInput("i".into()),
        InferenceError::Loading,
        InferenceError::QueueError("q".into()),
        InferenceError::OutOfMemory("oom".into()),
        InferenceError::ResourceExhausted("exhausted".into()),
        InferenceError::UnsupportedSafetensorsArchitecture("falcon".into()),
    ];

    for err in errors {
        let json = serde_json::to_string(&err).unwrap();
        let de: InferenceError = serde_json::from_str(&json).unwrap();
        assert_eq!(err, de);
    }
}

#[test]
fn test_is_out_of_memory_classification() {
    // 1. Explicit OutOfMemory variant
    assert!(InferenceError::OutOfMemory("any message".into()).is_out_of_memory());

    // 2. Explicit C++ std::bad_alloc
    assert!(InferenceError::Runtime("C++ std::bad_alloc thrown".into()).is_out_of_memory());
    assert!(
        InferenceError::Runtime("Failed with bad_alloc in host memory".into()).is_out_of_memory()
    );

    // 3. Explicit CUDA driver OOM codes
    assert!(InferenceError::Runtime("CUDA failure 2: out of memory".into()).is_out_of_memory());
    assert!(InferenceError::Runtime("cuda error: out of memory".into()).is_out_of_memory());
    assert!(InferenceError::Runtime("cudaMalloc failed: out of memory".into()).is_out_of_memory());
    assert!(
        InferenceError::Runtime("CUDA_ERROR_OUT_OF_MEMORY on device 0".into()).is_out_of_memory()
    );

    // 4. Device/host out-of-memory assertions
    assert!(InferenceError::Runtime("Host system ran out of memory".into()).is_out_of_memory());
    assert!(InferenceError::Runtime("Allocation failed: out of memory".into()).is_out_of_memory());

    // 5. Must reject "invalid memory access"
    assert!(
        !InferenceError::Runtime("Fatal: invalid memory access at 0x0".into()).is_out_of_memory()
    );

    // 6. Generic errors must NOT be classified as OOM
    assert!(!InferenceError::Runtime("File not found: model.onnx".into()).is_out_of_memory());
    assert!(!InferenceError::Runtime("ORT_FAIL: failed to load tensor".into()).is_out_of_memory());
    assert!(
        !InferenceError::Runtime("Unsupported Safetensors architecture: 'mpt'".into())
            .is_out_of_memory()
    );
    assert!(
        !InferenceError::ShapeMismatch("Cannot reshape tensor of size 12 to [1, 16]".into())
            .is_out_of_memory()
    );
    assert!(!InferenceError::NotFound("model not found".into()).is_out_of_memory());
    assert!(!InferenceError::InvalidInput("invalid tensor shape".into()).is_out_of_memory());
    assert!(!InferenceError::ResourceExhausted("capacity reached".into()).is_out_of_memory());
}

#[test]
fn test_model_state_serde() {
    let states = [
        (ModelState::Idle, "\"idle\""),
        (ModelState::Loading, "\"loading\""),
        (ModelState::Ready, "\"ready\""),
        (ModelState::Busy, "\"busy\""),
    ];

    for (state, expected_json) in states {
        let json = serde_json::to_string(&state).unwrap();
        assert_eq!(json, expected_json);
        let parsed: ModelState = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, state);
    }
}

#[test]
fn test_model_metadata_serde_roundtrip() {
    let meta = ModelMetadata {
        id: "model_x".into(),
        state: ModelState::Ready,
        onnx_path: "/models/x.onnx".into(),
        idle_timeout_secs: 120,
        max_batch_size: 16,
        max_batch_delay_ms: 10,
        last_accessed_secs_ago: 42,
        description: Some("Custom model description".into()),
        pinned: true,
    };
    let json = serde_json::to_string(&meta).unwrap();
    let de: ModelMetadata = serde_json::from_str(&json).unwrap();
    assert_eq!(de.id, meta.id);
    assert_eq!(de.state, meta.state);
    assert_eq!(de.onnx_path, meta.onnx_path);
    assert_eq!(de.idle_timeout_secs, meta.idle_timeout_secs);
    assert_eq!(de.max_batch_size, meta.max_batch_size);
    assert_eq!(de.max_batch_delay_ms, meta.max_batch_delay_ms);
    assert_eq!(de.last_accessed_secs_ago, meta.last_accessed_secs_ago);
    assert_eq!(de.description, meta.description);
    assert_eq!(de.pinned, meta.pinned);
}

#[test]
fn test_tensor_payload_rejects_non_finite_floats() {
    // NaN
    let err_nan =
        TensorPayload::from_shape_vec("test_nan", vec![2], vec![1.0, f32::NAN]).unwrap_err();
    assert!(matches!(err_nan, InferenceError::InvalidInput(_)));

    // +Infinity
    let err_pos_inf =
        TensorPayload::from_shape_vec("test_pos_inf", vec![2], vec![f32::INFINITY, 0.0])
            .unwrap_err();
    assert!(matches!(err_pos_inf, InferenceError::InvalidInput(_)));

    // -Infinity
    let err_neg_inf =
        TensorPayload::from_shape_vec("test_neg_inf", vec![1], vec![f32::NEG_INFINITY])
            .unwrap_err();
    assert!(matches!(err_neg_inf, InferenceError::InvalidInput(_)));
}

#[test]
fn test_is_valid_model_id() {
    use snooze_ml::types::is_valid_model_id;

    // Valid IDs
    assert!(is_valid_model_id("resnet50"));
    assert!(is_valid_model_id("bert-base-uncased"));
    assert!(is_valid_model_id("model_v1_prod"));
    assert!(is_valid_model_id("123-456_789"));

    // Invalid IDs: empty / whitespace
    assert!(!is_valid_model_id(""));
    assert!(!is_valid_model_id("   "));
    assert!(!is_valid_model_id("model name"));

    // Invalid IDs: path traversal and slashes
    assert!(!is_valid_model_id("../secret"));
    assert!(!is_valid_model_id("..\\windows"));
    assert!(!is_valid_model_id("models/resnet"));
    assert!(!is_valid_model_id("models\\resnet"));

    // Invalid IDs: control characters & special symbols
    assert!(!is_valid_model_id("model\0id"));
    assert!(!is_valid_model_id("model\nid"));
    assert!(!is_valid_model_id("model;id"));
    assert!(!is_valid_model_id("model:id"));
    assert!(!is_valid_model_id("model$id"));
}
