//! Prometheus metrics tracking for request counts, execution durations, and batching behavior.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Operational metrics tracker for the SnoozeML inference daemon.
#[derive(Debug, Default)]
pub struct ServerMetrics {
    total_requests: AtomicU64,
    successful_requests: AtomicU64,
    failed_requests: AtomicU64,
    total_latency_us: AtomicU64,
    total_batches: AtomicU64,
    oom_deallocations: AtomicU64,
    idle_deallocations: AtomicU64,
    capacity_deallocations: AtomicU64,
    load_retries: AtomicU64,
    thrashing_events: AtomicU64,
    stream_messages_consumed: AtomicU64,
    stream_messages_published: AtomicU64,
    stream_errors: AtomicU64,
}

impl ServerMetrics {
    /// Creates a new metrics tracker wrapped in an `Arc`.
    #[must_use]
    pub fn new_shared() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Records an inference request completion.
    pub fn record_inference(&self, success: bool, duration_us: u64) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        if success {
            self.successful_requests.fetch_add(1, Ordering::Relaxed);
        } else {
            self.failed_requests.fetch_add(1, Ordering::Relaxed);
        }
        self.total_latency_us
            .fetch_add(duration_us, Ordering::Relaxed);
    }

    /// Records a micro-batch execution dispatch.
    pub fn record_batch(&self) {
        self.total_batches.fetch_add(1, Ordering::Relaxed);
    }

    /// Records a model deallocation event labeled with its cause (`"oom"`, `"idle"`, or `"capacity"`).
    pub fn record_deallocation(&self, reason: &str) {
        match reason {
            "oom" => self.oom_deallocations.fetch_add(1, Ordering::Relaxed),
            "capacity" => self.capacity_deallocations.fetch_add(1, Ordering::Relaxed),
            _ => self.idle_deallocations.fetch_add(1, Ordering::Relaxed),
        };
    }

    /// Records an allocation retry attempt after an OOM deallocation.
    pub fn record_load_retry(&self) {
        self.load_retries.fetch_add(1, Ordering::Relaxed);
    }

    /// Records an anti-thrashing protection trigger.
    pub fn record_thrashing(&self) {
        self.thrashing_events.fetch_add(1, Ordering::Relaxed);
    }

    /// Records an ingested message from a streaming broker.
    pub fn record_stream_consumed(&self, _broker: &str, _topic: &str) {
        self.stream_messages_consumed
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Records a published response to a streaming broker.
    pub fn record_stream_published(&self, _broker: &str, _topic: &str) {
        self.stream_messages_published
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Records an error encountered during stream processing or publishing.
    pub fn record_stream_error(&self, _broker: &str, _topic: &str, _reason: &str) {
        self.stream_errors.fetch_add(1, Ordering::Relaxed);
    }

    /// Formats current metrics in standard Prometheus exposition format.
    #[must_use]
    pub fn format_prometheus(&self) -> String {
        let total = self.total_requests.load(Ordering::Relaxed);
        let success = self.successful_requests.load(Ordering::Relaxed);
        let failed = self.failed_requests.load(Ordering::Relaxed);
        let latency_secs = (self.total_latency_us.load(Ordering::Relaxed) as f64) / 1_000_000.0;
        let batches = self.total_batches.load(Ordering::Relaxed);
        let oom_deallocations = self.oom_deallocations.load(Ordering::Relaxed);
        let idle_deallocations = self.idle_deallocations.load(Ordering::Relaxed);
        let capacity_deallocations = self.capacity_deallocations.load(Ordering::Relaxed);
        let load_retries = self.load_retries.load(Ordering::Relaxed);
        let thrashing = self.thrashing_events.load(Ordering::Relaxed);
        let stream_in = self.stream_messages_consumed.load(Ordering::Relaxed);
        let stream_out = self.stream_messages_published.load(Ordering::Relaxed);
        let stream_err = self.stream_errors.load(Ordering::Relaxed);

        format!(
            "# HELP snoozeml_requests_total Total number of inference requests received\n\
             # TYPE snoozeml_requests_total counter\n\
             snoozeml_requests_total{{status=\"success\"}} {success}\n\
             snoozeml_requests_total{{status=\"error\"}} {failed}\n\
             snoozeml_requests_total{{status=\"all\"}} {total}\n\
             # HELP snoozeml_inference_latency_seconds_total Total duration of inference execution in seconds\n\
             # TYPE snoozeml_inference_latency_seconds_total counter\n\
             snoozeml_inference_latency_seconds_total {latency_secs:.6}\n\
             # HELP snoozeml_batches_dispatched_total Total micro-batches processed by the execution engine\n\
             # TYPE snoozeml_batches_dispatched_total counter\n\
             snoozeml_batches_dispatched_total {batches}\n\
             # HELP snoozeml_model_deallocations_total Total number of model unloads by deallocation reason\n\
             # TYPE snoozeml_model_deallocations_total counter\n\
             snoozeml_model_deallocations_total{{reason=\"oom\"}} {oom_deallocations}\n\
             snoozeml_model_deallocations_total{{reason=\"idle\"}} {idle_deallocations}\n\
             snoozeml_model_deallocations_total{{reason=\"capacity\"}} {capacity_deallocations}\n\
             # HELP snoozeml_model_load_retries_total Total allocation retry attempts after OOM\n\
             # TYPE snoozeml_model_load_retries_total counter\n\
             snoozeml_model_load_retries_total {load_retries}\n\
             # HELP snoozeml_deallocation_thrashing_total Total rapid re-deallocation events detected\n\
             # TYPE snoozeml_deallocation_thrashing_total counter\n\
             snoozeml_deallocation_thrashing_total {thrashing}\n\
             # HELP snoozeml_stream_messages_consumed_total Total messages received from streaming message brokers\n\
             # TYPE snoozeml_stream_messages_consumed_total counter\n\
             snoozeml_stream_messages_consumed_total {stream_in}\n\
             # HELP snoozeml_stream_messages_published_total Total messages published to streaming message brokers\n\
             # TYPE snoozeml_stream_messages_published_total counter\n\
             snoozeml_stream_messages_published_total {stream_out}\n\
             # HELP snoozeml_stream_errors_total Total errors encountered during streaming ingestion or publishing\n\
             # TYPE snoozeml_stream_errors_total counter\n\
             snoozeml_stream_errors_total {stream_err}\n"
        )
    }
}
