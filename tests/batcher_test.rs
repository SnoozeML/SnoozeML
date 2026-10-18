//! Integration and behavioral unit tests for the dynamic micro-batcher accumulator.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use ndarray::ArrayD;
use snooze_ml::batcher::{DynamicBatcher, DynamicBatcherConfig};
use snooze_ml::types::{InferenceError, InferenceJob, TensorPayload};
use std::time::Duration;
use tokio::sync::oneshot;

fn make_tensor(name: &str, shape: Vec<usize>, val: f32) -> TensorPayload {
    let total: usize = shape.iter().product();
    let data = vec![val; total];
    let array = ArrayD::from_shape_vec(ndarray::IxDyn(&shape), data).unwrap();
    TensorPayload {
        name: name.to_string(),
        shape,
        data: array,
    }
}

#[test]
fn test_collate_inputs_success() {
    let req1 = vec![make_tensor("input_a", vec![1, 3, 4], 1.0)];
    let req2 = vec![make_tensor("input_a", vec![2, 3, 4], 2.0)];

    let collated = DynamicBatcher::collate_inputs(&[req1, req2]).unwrap();
    assert_eq!(collated.len(), 1);
    assert_eq!(collated[0].shape, vec![3, 3, 4]);
}

#[test]
fn test_collate_inputs_shape_mismatch() {
    let req1 = vec![make_tensor("input_a", vec![1, 3, 4], 1.0)];
    let req2 = vec![make_tensor("input_a", vec![1, 3, 8], 2.0)];

    let err = DynamicBatcher::collate_inputs(&[req1, req2]).unwrap_err();
    assert!(matches!(err, InferenceError::ShapeMismatch(_)));
}

#[test]
fn test_split_outputs_exact() {
    let batched_out = vec![make_tensor("output", vec![4, 2], 5.0)];
    let slice_sizes = vec![1, 2, 1];

    let split = DynamicBatcher::split_outputs(&batched_out, &slice_sizes).unwrap();
    assert_eq!(split.len(), 3);
    assert_eq!(split[0][0].shape, vec![1, 2]);
    assert_eq!(split[1][0].shape, vec![2, 2]);
    assert_eq!(split[2][0].shape, vec![1, 2]);
}

#[test]
fn test_dynamic_batcher_config_default() {
    let config = DynamicBatcherConfig::default();
    assert_eq!(config.max_batch_size, 16);
    assert_eq!(config.max_delay, Duration::from_millis(10));
    assert_eq!(config.queue_capacity, 1024);
}

#[test]
fn test_collate_inputs_empty_jobs() {
    let err = DynamicBatcher::collate_inputs(&[]).unwrap_err();
    assert!(matches!(err, InferenceError::InvalidInput(_)));
}

#[test]
fn test_collate_inputs_zero_tensors_in_job() {
    let err = DynamicBatcher::collate_inputs(&[vec![]]).unwrap_err();
    assert!(matches!(err, InferenceError::InvalidInput(_)));
}

#[test]
fn test_collate_inputs_tensor_count_mismatch() {
    let req1 = vec![
        make_tensor("t1", vec![1, 2], 1.0),
        make_tensor("t2", vec![1, 2], 2.0),
    ];
    let req2 = vec![make_tensor("t1", vec![1, 2], 3.0)];

    let err = DynamicBatcher::collate_inputs(&[req1, req2]).unwrap_err();
    assert!(matches!(err, InferenceError::ShapeMismatch(_)));
}

#[test]
fn test_collate_inputs_tensor_name_mismatch() {
    let req1 = vec![make_tensor("input_a", vec![1, 2], 1.0)];
    let req2 = vec![make_tensor("input_b", vec![1, 2], 2.0)];

    let err = DynamicBatcher::collate_inputs(&[req1, req2]).unwrap_err();
    assert!(matches!(err, InferenceError::ShapeMismatch(_)));
}

#[test]
fn test_collate_inputs_multiple_tensors_success() {
    let req1 = vec![
        make_tensor("ids", vec![2, 4], 1.0),
        make_tensor("mask", vec![2, 4], 1.0),
    ];
    let req2 = vec![
        make_tensor("ids", vec![3, 4], 2.0),
        make_tensor("mask", vec![3, 4], 0.0),
    ];

    let collated = DynamicBatcher::collate_inputs(&[req1, req2]).unwrap();
    assert_eq!(collated.len(), 2);
    assert_eq!(collated[0].name, "ids");
    assert_eq!(collated[0].shape, vec![5, 4]);
    assert_eq!(collated[1].name, "mask");
    assert_eq!(collated[1].shape, vec![5, 4]);
}

#[test]
fn test_split_outputs_batch_mismatch() {
    let batched_out = vec![make_tensor("output", vec![4, 2], 1.0)];
    let slice_sizes = vec![2, 3]; // Sums to 5 != 4

    let err = DynamicBatcher::split_outputs(&batched_out, &slice_sizes).unwrap_err();
    assert!(matches!(err, InferenceError::Runtime(_)));
}

#[test]
fn test_split_outputs_multi_tensor() {
    let batched_out = vec![
        make_tensor("out_a", vec![3, 2], 1.0),
        make_tensor("out_b", vec![3, 5], 2.0),
    ];
    let slice_sizes = vec![1, 2];

    let split = DynamicBatcher::split_outputs(&batched_out, &slice_sizes).unwrap();
    assert_eq!(split.len(), 2);

    // Client 0 has batch size 1 for both outputs
    assert_eq!(split[0].len(), 2);
    assert_eq!(split[0][0].name, "out_a");
    assert_eq!(split[0][0].shape, vec![1, 2]);
    assert_eq!(split[0][1].name, "out_b");
    assert_eq!(split[0][1].shape, vec![1, 5]);

    // Client 1 has batch size 2 for both outputs
    assert_eq!(split[1].len(), 2);
    assert_eq!(split[1][0].name, "out_a");
    assert_eq!(split[1][0].shape, vec![2, 2]);
    assert_eq!(split[1][1].name, "out_b");
    assert_eq!(split[1][1].shape, vec![2, 5]);
}

#[tokio::test]
async fn test_dynamic_batcher_timeout_flush() {
    let config = DynamicBatcherConfig {
        max_batch_size: 10, // Not reachable with 1 request
        max_delay: Duration::from_millis(20),
        queue_capacity: 10,
    };

    let batcher = DynamicBatcher::spawn(config, |inputs, responders| {
        for (tx, _) in responders {
            let _ = tx.send(Ok(inputs.clone()));
        }
    });

    let (tx, rx) = oneshot::channel();
    let job = InferenceJob {
        inputs: vec![make_tensor("in", vec![1, 2], 42.0)],
        response_tx: tx,
        submitted_at: std::time::Instant::now(),
    };

    batcher.submit(job).await.unwrap();

    // Must flush via timeout
    let result = tokio::time::timeout(Duration::from_millis(200), rx)
        .await
        .expect("Batcher should have flushed before timeout")
        .unwrap()
        .unwrap();

    assert_eq!(result[0].shape, vec![1, 2]);
}

#[tokio::test]
async fn test_dynamic_batcher_collation_failure_notifies_all() {
    let config = DynamicBatcherConfig {
        max_batch_size: 2,
        max_delay: Duration::from_millis(50),
        queue_capacity: 10,
    };

    let batcher = DynamicBatcher::spawn(config, |_inputs, responders| {
        for (tx, _) in responders {
            let _ = tx.send(Ok(vec![]));
        }
    });

    let (tx1, rx1) = oneshot::channel();
    let (tx2, rx2) = oneshot::channel();

    // job1 and job2 have mismatched trailing shapes
    let job1 = InferenceJob {
        inputs: vec![make_tensor("in", vec![1, 2], 1.0)],
        response_tx: tx1,
        submitted_at: std::time::Instant::now(),
    };
    let job2 = InferenceJob {
        inputs: vec![make_tensor("in", vec![1, 4], 2.0)],
        response_tx: tx2,
        submitted_at: std::time::Instant::now(),
    };

    batcher.submit(job1).await.unwrap();
    batcher.submit(job2).await.unwrap();

    let res1 = rx1.await.unwrap();
    let res2 = rx2.await.unwrap();

    assert!(matches!(res1, Err(InferenceError::ShapeMismatch(_))));
    assert!(matches!(res2, Err(InferenceError::ShapeMismatch(_))));
}

#[test]
fn test_collate_inputs_rejects_0d_scalar_tensors() {
    let scalar_tensor = TensorPayload {
        name: "scalar_in".to_string(),
        shape: vec![],
        data: ArrayD::from_shape_vec(ndarray::IxDyn(&[]), vec![42.0]).unwrap(),
    };
    let normal_tensor = make_tensor("in", vec![1, 2], 1.0);

    // Single job with 0-D tensor
    let err_single = DynamicBatcher::collate_inputs(&[vec![scalar_tensor.clone()]]).unwrap_err();
    assert!(matches!(err_single, InferenceError::InvalidInput(_)));

    // Multi-job where one job has a 0-D tensor
    let err_multi =
        DynamicBatcher::collate_inputs(&[vec![normal_tensor], vec![scalar_tensor]]).unwrap_err();
    assert!(matches!(err_multi, InferenceError::InvalidInput(_)));
}

#[tokio::test]
async fn test_batcher_prunes_abandoned_requests_without_infer() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    let infer_call_count = Arc::new(AtomicUsize::new(0));
    let infer_count_clone = infer_call_count.clone();

    let config = DynamicBatcherConfig {
        max_batch_size: 2,
        max_delay: Duration::from_millis(20),
        queue_capacity: 10,
    };

    let batcher = DynamicBatcher::spawn(config, move |inputs, responders| {
        infer_count_clone.fetch_add(1, Ordering::SeqCst);
        for (tx, _) in responders {
            let _ = tx.send(Ok(inputs.clone()));
        }
    });

    // Case 1: Client sends request but immediately drops the receiver (abandoned/disconnected)
    let (tx1, rx1) = oneshot::channel();
    drop(rx1); // Client disconnected

    let job1 = InferenceJob {
        inputs: vec![make_tensor("in", vec![1, 2], 1.0)],
        response_tx: tx1,
        submitted_at: std::time::Instant::now(),
    };
    batcher.submit(job1).await.unwrap();

    // Wait for timeout to flush
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Backend forward pass must NOT have been called because the only request was abandoned
    assert_eq!(
        infer_call_count.load(Ordering::SeqCst),
        0,
        "Abandoned job must be pruned without invoking backend inference"
    );

    // Case 2: One client drops receiver, another stays active
    let (tx_dead, rx_dead) = oneshot::channel();
    drop(rx_dead);
    let (tx_alive, rx_alive) = oneshot::channel();

    let dead_job = InferenceJob {
        inputs: vec![make_tensor("in", vec![1, 2], 1.0)],
        response_tx: tx_dead,
        submitted_at: std::time::Instant::now(),
    };
    let alive_job = InferenceJob {
        inputs: vec![make_tensor("in", vec![1, 2], 2.0)],
        response_tx: tx_alive,
        submitted_at: std::time::Instant::now(),
    };

    batcher.submit(dead_job).await.unwrap();
    batcher.submit(alive_job).await.unwrap();

    let res = tokio::time::timeout(Duration::from_millis(100), rx_alive)
        .await
        .expect("Alive receiver must receive a response")
        .unwrap()
        .unwrap();

    assert_eq!(res.len(), 1);
    assert_eq!(
        infer_call_count.load(Ordering::SeqCst),
        1,
        "Surviving job must be dispatched cleanly"
    );
}

#[test]
fn test_split_outputs_mismatched_output_tensor_count() {
    let t1 = make_tensor("out1", vec![2, 4], 1.0);
    let t2 = make_tensor("out2", vec![2, 4], 2.0);

    // Empty output list rejected
    let err_empty = DynamicBatcher::split_outputs(&[], &[1, 1]).unwrap_err();
    assert!(matches!(err_empty, InferenceError::Runtime(_)));

    // Fewer output tensors than expected
    let err_fewer =
        DynamicBatcher::split_outputs_with_expected(std::slice::from_ref(&t1), &[1, 1], Some(2))
            .unwrap_err();
    assert!(matches!(err_fewer, InferenceError::Runtime(_)));

    // More output tensors than expected
    let err_more =
        DynamicBatcher::split_outputs_with_expected(&[t1.clone(), t2.clone()], &[1, 1], Some(1))
            .unwrap_err();
    assert!(matches!(err_more, InferenceError::Runtime(_)));

    // Matching tensor count succeeds
    let ok = DynamicBatcher::split_outputs_with_expected(&[t1, t2], &[1, 1], Some(2)).unwrap();
    assert_eq!(ok.len(), 2);
    assert_eq!(ok[0].len(), 2);
}

#[tokio::test]
async fn test_batcher_queue_saturation_backpressure_and_rejection() {
    let config = DynamicBatcherConfig {
        max_batch_size: 10,
        max_delay: Duration::from_secs(60), // Block flush so queue accumulates
        queue_capacity: 2,
    };

    let batcher = DynamicBatcher::spawn(config, |_inputs, _responders| {});

    let (tx1, _rx1) = oneshot::channel();
    let (tx2, _rx2) = oneshot::channel();
    let (tx3, _rx3) = oneshot::channel();

    let job1 = InferenceJob {
        inputs: vec![make_tensor("in", vec![1, 2], 1.0)],
        response_tx: tx1,
        submitted_at: std::time::Instant::now(),
    };
    let job2 = InferenceJob {
        inputs: vec![make_tensor("in", vec![1, 2], 2.0)],
        response_tx: tx2,
        submitted_at: std::time::Instant::now(),
    };
    let job3 = InferenceJob {
        inputs: vec![make_tensor("in", vec![1, 2], 3.0)],
        response_tx: tx3,
        submitted_at: std::time::Instant::now(),
    };

    // First two fill queue to capacity (2)
    assert!(batcher.try_submit(job1).is_ok());
    assert!(batcher.try_submit(job2).is_ok());

    // Third try_submit must fail fast with QueueError due to saturation
    let err = batcher.try_submit(job3).unwrap_err();
    assert!(matches!(err, InferenceError::QueueError(_)));
}
