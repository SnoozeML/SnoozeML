---
name: streaming-integration
description: >-
  Runbook for configuring, implementing, and verifying real-time event streaming
  pipelines in SnoozeML across Apache Kafka, RabbitMQ, Redis Streams, and Mock
  streams. Use this skill when modifying src/stream/, configuring models.toml
  [[streams]], or writing streaming integration tests in tests/stream_test.rs.
---

# Streaming Integration & Event Pipelines Runbook

SnoozeML includes a high-throughput event streaming engine (`src/stream/`) that connects model inference to external event brokers without writing custom glue code. This runbook details how to configure pipelines, implement drivers, and verify end-to-end event flows.

---

## 1. Streaming Pipeline Architecture

In SnoozeML, an event streaming pipeline links:
1. **Source / Ingest Broker**: Polls or receives messages from Kafka topics, RabbitMQ queues, or Redis streams.
2. **Transformer & Batcher**: Deserializes raw payloads (JSON or binary) into `TensorPayload` inputs and enqueues them into the target model's `DynamicBatcher`.
3. **Inference Execution**: Coalesces concurrent records along `Axis(0)` and runs model inference.
4. **Sink / Egress Broker**: Serializes output tensors and publishes them to an egress topic or queue.
5. **Offset Commit / Acknowledgment**: Only commits partition offsets or sends `ack` after egress publish completes (at-least-once delivery).

---

## 2. Configuring Streaming in `models.toml`

Stream pipelines are declared using `[[streams]]` tables in `models.toml`:

```toml
[[streams]]
stream_id = "sensor-telemetry-pipeline"
model_id = "sensor_anomaly_detector"
batch_size = 32
max_latency_ms = 10
input_tensor = "sensor_features"
output_tensor = "anomaly_score"

[streams.ingest]
broker_type = "kafka" # or "rabbitmq", "redis", "mock"
endpoints = ["localhost:9092"]
topic = "raw-sensor-readings"
group_id = "snooze-inference-workers"

[streams.egress]
broker_type = "kafka"
endpoints = ["localhost:9092"]
topic = "processed-sensor-scores"
```

For schema details on RabbitMQ and Redis configuration, see [references/streaming-config-reference.md](./references/streaming-config-reference.md).

---

## 3. Implementing or Extending a Stream Driver (`src/stream/`)

When adding a new broker driver or modifying existing drivers:

### Step 1: Implement `StreamDriver` Trait (`src/stream/driver.rs`)
Implement the async trait `StreamDriver`:
```rust
#[async_trait::async_trait]
pub trait StreamDriver: Send + Sync {
    /// Poll for the next batch of stream messages.
    async fn poll_batch(&mut self, max_messages: usize, timeout: Duration) -> Result<Vec<StreamMessage>, StreamError>;

    /// Acknowledge successful processing of a batch of messages.
    async fn commit(&mut self, messages: &[StreamMessage]) -> Result<(), StreamError>;

    /// Publish an inference result message to the egress destination.
    async fn publish(&mut self, topic: &str, message: &StreamMessage) -> Result<(), StreamError>;
}
```

### Step 2: Feature-Gating Hygiene
Ensure new broker dependencies are gated in `Cargo.toml` and guarded with `#[cfg(feature = "streams-<broker>")]` in `src/stream/mod.rs`. Provide clean error messages when feature flags are disabled.

### Step 3: Zero-Panic & Reconnection Invariants
- Handle network disconnects with exponential retry backoff. Never crash or `.unwrap()`.
- Respect `max_queue_size` by awaiting batcher capacity before polling more broker records.

---

## 4. Verification & Testing

Always verify streaming logic using `MockStreamDriver` in `tests/stream_test.rs`:

```powershell
# Run the dedicated streaming integration suite
cargo test --test stream_test --features streams-all

# Verify compilation with individual broker features
cargo check --no-default-features --features streams-kafka
cargo check --no-default-features --features streams-redis
cargo check --no-default-features --features streams-rabbitmq
```
