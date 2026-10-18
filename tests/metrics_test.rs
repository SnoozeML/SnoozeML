//! Integration and unit tests for server Prometheus telemetry and metrics tracker.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use snooze_ml::server::metrics::ServerMetrics;

#[test]
fn test_metrics_collection_and_formatting() {
    let metrics = ServerMetrics::default();
    metrics.record_inference(true, 10_000);
    metrics.record_inference(false, 5_000);
    metrics.record_batch();

    let output = metrics.format_prometheus();
    assert!(output.contains("snoozeml_requests_total{status=\"success\"} 1"));
    assert!(output.contains("snoozeml_requests_total{status=\"error\"} 1"));
    assert!(output.contains("snoozeml_requests_total{status=\"all\"} 2"));
    assert!(output.contains("snoozeml_inference_latency_seconds_total 0.015000"));
    assert!(output.contains("snoozeml_batches_dispatched_total 1"));
}

#[test]
fn test_metrics_zero_defaults() {
    let metrics = ServerMetrics::default();
    let output = metrics.format_prometheus();
    assert!(output.contains("snoozeml_requests_total{status=\"success\"} 0"));
    assert!(output.contains("snoozeml_requests_total{status=\"error\"} 0"));
    assert!(output.contains("snoozeml_requests_total{status=\"all\"} 0"));
    assert!(output.contains("snoozeml_inference_latency_seconds_total 0.000000"));
    assert!(output.contains("snoozeml_batches_dispatched_total 0"));
}

#[test]
fn test_metrics_shared_concurrency() {
    let metrics = ServerMetrics::new_shared();
    let m1 = metrics.clone();
    let m2 = metrics.clone();

    m1.record_inference(true, 100_000);
    m2.record_inference(false, 50_000);
    m1.record_batch();
    m2.record_batch();

    let output = metrics.format_prometheus();
    assert!(output.contains("snoozeml_requests_total{status=\"success\"} 1"));
    assert!(output.contains("snoozeml_requests_total{status=\"error\"} 1"));
    assert!(output.contains("snoozeml_requests_total{status=\"all\"} 2"));
    assert!(output.contains("snoozeml_inference_latency_seconds_total 0.150000"));
    assert!(output.contains("snoozeml_batches_dispatched_total 2"));
}

#[test]
fn test_metrics_deallocations_and_retries() {
    let metrics = ServerMetrics::default();
    metrics.record_deallocation("oom");
    metrics.record_deallocation("oom");
    metrics.record_deallocation("idle");
    metrics.record_deallocation("capacity");
    metrics.record_load_retry();
    metrics.record_thrashing();

    let output = metrics.format_prometheus();
    assert!(output.contains("snoozeml_model_deallocations_total{reason=\"oom\"} 2"));
    assert!(output.contains("snoozeml_model_deallocations_total{reason=\"idle\"} 1"));
    assert!(output.contains("snoozeml_model_deallocations_total{reason=\"capacity\"} 1"));
    assert!(output.contains("snoozeml_model_load_retries_total 1"));
    assert!(output.contains("snoozeml_deallocation_thrashing_total 1"));
}
