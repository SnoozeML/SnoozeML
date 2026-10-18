//! Integration tests for process-isolated Python worker backend.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use ndarray::ArrayD;
use snooze_ml::backend::worker_backend::{is_worker_model, WorkerBackend, WorkerConfig};
use snooze_ml::backend::{create_backend, ModelBackend};
use snooze_ml::config::ModelDirectoryConfig;
use snooze_ml::model::{ModelConfig, ModelController};
use snooze_ml::types::{InferenceError, TensorPayload};
use std::path::{Path, PathBuf};
use std::time::Duration;

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("worker_models")
}

fn check_python_available() -> bool {
    let config = WorkerConfig {
        model_dir: fixtures_dir().join("identity"),
        device: "cpu".to_string(),
    };
    match WorkerBackend::new(&config) {
        Ok(_) => true,
        Err(e) => {
            eprintln!("SKIPPED: Python worker environment not available: {e}");
            false
        }
    }
}

#[test]
fn test_worker_backend_identity_roundtrip() {
    if !check_python_available() {
        return;
    }

    let model_dir = fixtures_dir().join("identity");
    let config = WorkerConfig {
        model_dir,
        device: "cpu".to_string(),
    };

    let backend = WorkerBackend::new(&config).unwrap();
    assert_eq!(backend.provider_name(), "python-worker");
    assert!(backend.pid().is_some());

    let input_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let payload = TensorPayload::from_shape_vec("input", vec![2, 3], input_data.clone()).unwrap();

    let outputs = backend.infer(&[payload]).unwrap();
    assert_eq!(outputs.len(), 1);
    let out = &outputs[0];
    assert_eq!(out.name, "output");
    assert_eq!(out.shape, vec![2, 3]);
    let out_vec: Vec<f32> = out.data.iter().copied().collect();
    assert_eq!(out_vec, input_data);
}

#[test]
fn test_worker_backend_affine_math() {
    if !check_python_available() {
        return;
    }

    let model_dir = fixtures_dir().join("affine");
    let config = WorkerConfig {
        model_dir,
        device: "cpu".to_string(),
    };

    let backend = WorkerBackend::new(&config).unwrap();
    let payload = TensorPayload::from_shape_vec("input", vec![1, 3], vec![1.0, 2.5, -4.0]).unwrap();

    let outputs = backend.infer(&[payload]).unwrap();
    assert_eq!(outputs.len(), 1);
    let out_vec: Vec<f32> = outputs[0].data.iter().copied().collect();
    // 2x + 1 -> [3.0, 6.0, -7.0]
    assert_eq!(out_vec, vec![3.0, 6.0, -7.0]);
}

#[test]
fn test_worker_backend_multi_io() {
    if !check_python_available() {
        return;
    }

    let model_dir = fixtures_dir().join("multi_io");
    let config = WorkerConfig {
        model_dir,
        device: "cpu".to_string(),
    };

    let backend = WorkerBackend::new(&config).unwrap();
    let a = TensorPayload::from_shape_vec("a", vec![2, 2], vec![10.0, 20.0, 30.0, 40.0]).unwrap();
    let b = TensorPayload::from_shape_vec("b", vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();

    let outputs = backend.infer(&[a, b]).unwrap();
    assert_eq!(outputs.len(), 2);

    let sum = outputs.iter().find(|t| t.name == "sum").unwrap();
    let diff = outputs.iter().find(|t| t.name == "diff").unwrap();

    let sum_vec: Vec<f32> = sum.data.iter().copied().collect();
    let diff_vec: Vec<f32> = diff.data.iter().copied().collect();

    assert_eq!(sum_vec, vec![11.0, 22.0, 33.0, 44.0]);
    assert_eq!(diff_vec, vec![9.0, 18.0, 27.0, 36.0]);
}

#[tokio::test]
async fn test_worker_backend_through_model_controller_batching() {
    if !check_python_available() {
        return;
    }

    let model_dir = fixtures_dir().join("affine");
    let config = ModelConfig {
        id: "worker-affine".to_string(),
        onnx_path: model_dir.to_string_lossy().to_string(),
        idle_timeout: Duration::from_secs(60),
        max_batch_size: 4,
        max_batch_delay: Duration::from_millis(50),
        provider: Some("cpu".to_string()),
        description: Some("Affine test".to_string()),
        min_residency: Duration::from_millis(0),
        ..Default::default()
    };

    let controller = ModelController::new(config);
    controller.ensure_loaded().await.unwrap();

    // Send two concurrent requests to test batching and output splitting along Axis(0)
    let c1 = controller.clone();
    let c2 = controller.clone();

    let h1 = tokio::spawn(async move {
        let p = TensorPayload {
            name: "input".to_string(),
            shape: vec![1, 2],
            data: ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 2]), vec![10.0, 20.0]).unwrap(),
        };
        c1.infer(vec![p]).await.unwrap()
    });

    let h2 = tokio::spawn(async move {
        let p = TensorPayload {
            name: "input".to_string(),
            shape: vec![1, 2],
            data: ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 2]), vec![30.0, 40.0]).unwrap(),
        };
        c2.infer(vec![p]).await.unwrap()
    });

    let res1 = h1.await.unwrap();
    let res2 = h2.await.unwrap();

    assert_eq!(res1.len(), 1);
    assert_eq!(res2.len(), 1);

    let v1: Vec<f32> = res1[0].data.iter().copied().collect();
    let v2: Vec<f32> = res2[0].data.iter().copied().collect();

    // 2x + 1
    assert_eq!(v1, vec![21.0, 41.0]);
    assert_eq!(v2, vec![61.0, 81.0]);

    // Test clean unload
    controller.unload().await;
}

#[test]
fn test_worker_backend_oom_classification() {
    if !check_python_available() {
        return;
    }

    let model_dir = fixtures_dir().join("oom");
    let config = WorkerConfig {
        model_dir,
        device: "cpu".to_string(),
    };

    let backend = WorkerBackend::new(&config).unwrap();
    let payload = TensorPayload::from_shape_vec("input", vec![1, 2], vec![1.0, 2.0]).unwrap();

    let err = backend.infer(&[payload]).err().unwrap();
    assert!(
        matches!(&err, InferenceError::OutOfMemory(msg) if msg.contains("out of memory")),
        "Expected OutOfMemory error, got: {:?}",
        err
    );
}

#[test]
fn test_worker_backend_crash_mid_request_and_respawn() {
    if !check_python_available() {
        return;
    }

    let model_dir = fixtures_dir().join("crash");
    let config = WorkerConfig {
        model_dir,
        device: "cpu".to_string(),
    };

    let backend = WorkerBackend::new(&config).unwrap();
    let first_pid = backend.pid().unwrap();
    let payload = TensorPayload::from_shape_vec("input", vec![1, 2], vec![1.0, 2.0]).unwrap();

    // The first infer crashes the worker via sys.exit(42)
    let err = backend.infer(std::slice::from_ref(&payload)).err().unwrap();
    assert!(
        matches!(err, InferenceError::Runtime(_)),
        "Expected Runtime error on child crash, got: {:?}",
        err
    );

    // The worker was marked dead. The next request attempts to respawn it
    let res = backend.infer(&[payload]);
    // The respawned worker will run the same crash fixture and fail again cleanly
    assert!(res.is_err());
    let second_pid = backend.pid();
    if let Some(new_pid) = second_pid {
        assert_ne!(
            first_pid, new_pid,
            "Worker PID should have changed on respawn"
        );
    }
}

#[test]
fn test_worker_backend_request_timeout() {
    if !check_python_available() {
        return;
    }

    let model_dir = fixtures_dir().join("hang");
    let config = WorkerConfig {
        model_dir,
        device: "cpu".to_string(),
    };

    let backend = WorkerBackend::new(&config).unwrap();
    let payload = TensorPayload::from_shape_vec("input", vec![1, 2], vec![1.0, 2.0]).unwrap();

    let started = std::time::Instant::now();
    let err = backend.infer(&[payload]).err().unwrap();
    let elapsed = started.elapsed();

    assert!(
        matches!(&err, InferenceError::Runtime(msg) if msg.contains("request_timeout_secs")),
        "Expected timeout error, got: {:?}",
        err
    );
    // Request timeout in snooze_model.json is 2s, should finish in ~2-4s, not 10s
    assert!(
        elapsed < Duration::from_secs(6),
        "Infer took too long ({:?}), worker timeout failed",
        elapsed
    );
}

#[test]
fn test_worker_backend_drop_reaps_process() {
    if !check_python_available() {
        return;
    }

    let model_dir = fixtures_dir().join("identity");
    let config = WorkerConfig {
        model_dir,
        device: "cpu".to_string(),
    };

    let backend = WorkerBackend::new(&config).unwrap();
    assert!(backend.pid().is_some());
    #[cfg(unix)]
    let pid = backend.pid().unwrap();

    // Verify process is alive
    #[cfg(unix)]
    {
        let status = std::process::Command::new("kill")
            .arg("-0")
            .arg(pid.to_string())
            .status()
            .unwrap();
        assert!(
            status.success(),
            "Worker process should be alive before drop"
        );
    }

    drop(backend);

    // Wait a brief moment for the shutdown grace / reaper thread
    std::thread::sleep(Duration::from_millis(500));

    #[cfg(unix)]
    {
        let mut dead = false;
        for _ in 0..20 {
            let status = std::process::Command::new("kill")
                .arg("-0")
                .arg(pid.to_string())
                .status();
            if let Ok(st) = status {
                if !st.success() {
                    dead = true;
                    break;
                }
            } else {
                dead = true;
                break;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        assert!(
            dead,
            "Worker process should have terminated after backend drop"
        );
    }
}

#[test]
fn test_create_backend_worker_precedence_over_safetensors() {
    if !check_python_available() {
        return;
    }

    let model_dir = fixtures_dir().join("hybrid_safetensors");
    // Directory contains snooze_model.py, model.safetensors, and config.json
    let backend = create_backend(model_dir.to_str().unwrap(), None).unwrap();
    assert_eq!(
        backend.provider_name(),
        "python-worker",
        "create_backend must prioritize snooze_model.py over .safetensors"
    );
}

#[test]
fn test_worker_backend_raw_tensor_format() {
    if !check_python_available() {
        return;
    }

    let model_dir = fixtures_dir().join("raw_tensor");
    let config = WorkerConfig {
        model_dir,
        device: "cpu".to_string(),
    };

    let backend = WorkerBackend::new(&config).unwrap();
    let payload =
        TensorPayload::from_shape_vec("input", vec![1, 3], vec![5.0, 10.0, 15.0]).unwrap();

    let outputs = backend.infer(&[payload]).unwrap();
    assert_eq!(outputs.len(), 1);
    let out_vec: Vec<f32> = outputs[0].data.iter().copied().collect();
    assert_eq!(out_vec, vec![5.0, 10.0, 15.0]);
}

#[tokio::test]
async fn test_config_discovery_worker_model() {
    let base_dir = fixtures_dir();
    let discovered = ModelDirectoryConfig::discover_models_in_dir(&base_dir)
        .await
        .unwrap();

    // Verify discovered contains "identity", "affine", "hybrid_safetensors" as single models
    let ids: Vec<&str> = discovered.iter().map(|m| m.id.as_str()).collect();
    assert!(
        ids.contains(&"identity"),
        "Should discover identity worker model"
    );
    assert!(
        ids.contains(&"affine"),
        "Should discover affine worker model"
    );
    assert!(
        ids.contains(&"hybrid_safetensors"),
        "Should discover hybrid_safetensors as a single worker model"
    );

    // Verify it did not discover hybrid_safetensors/model.safetensors as a separate model
    let paths: Vec<&Path> = discovered
        .iter()
        .map(|m| m.absolute_path.as_path())
        .collect();
    assert!(
        !paths
            .iter()
            .any(|p| p.ends_with("hybrid_safetensors/model.safetensors")),
        "Should not descend into worker model for internal .safetensors files"
    );
}

struct TempDirGuard {
    path: PathBuf,
}

impl TempDirGuard {
    fn new(name: &str) -> Self {
        let mut path = std::env::temp_dir();
        let unique = format!(
            "snooze_worker_test_{name}_{}_{}",
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

#[tokio::test]
async fn test_worker_manifest_discovery_and_detection() {
    let temp = TempDirGuard::new("custom_manifest");
    let model_dir = temp.path.join("custom_worker");
    std::fs::create_dir_all(&model_dir).unwrap();

    let entry_file = model_dir.join("custom_entry.py");
    std::fs::write(
        &entry_file,
        b"class CustomModel:\n    def load(self, model_dir, device): pass\n    def predict(self, inputs): return inputs\n",
    )
    .unwrap();

    let manifest_file = model_dir.join("snooze_model.json");
    std::fs::write(
        &manifest_file,
        br#"{"entrypoint": "custom_entry.py", "class": "CustomModel"}"#,
    )
    .unwrap();

    // 1. is_worker_model recognizes directory with snooze_model.json
    assert!(
        is_worker_model(model_dir.to_str().unwrap(), None),
        "is_worker_model should return true for directory with snooze_model.json"
    );

    // 2. is_worker_model recognizes snooze_model.json directly
    assert!(
        is_worker_model(manifest_file.to_str().unwrap(), None),
        "is_worker_model should return true for snooze_model.json file"
    );

    // 3. ModelDirectoryConfig::discover_models_in_dir discovers it
    let discovered = ModelDirectoryConfig::discover_models_in_dir(&temp.path)
        .await
        .unwrap();
    assert_eq!(discovered.len(), 1);
    assert_eq!(discovered[0].id, "custom_worker");
    assert_eq!(discovered[0].absolute_path, model_dir);
}
