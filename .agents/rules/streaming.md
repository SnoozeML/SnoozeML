---
trigger: always_on
---

# Streaming Engine Invariants & Rules (`src/stream/`)

SnoozeML features a real-time event streaming subsystem (`src/stream/`) that connects model inference to event brokers (Apache Kafka via `rskafka`, RabbitMQ via `lapin`, Redis Streams via `redis`, and mock streams via `mock_stream.rs`). All streaming code must maintain the following architectural rules.

---

## 1. Concurrency & Actor Isolation
* **Independent Tokio Tasks:** Every streaming consumer defined under `[[streams]]` in `models.toml` must be spawned as an isolated Tokio background task (`tokio::spawn`). A network failure, deserialization failure, or broker timeout in one stream must never affect other streams or HTTP/gRPC listeners.
* **Graceful Task Cancellation:** Stream consumer loops must listen to a cancellation token or shutdown signal (`tokio::sync::watch` or `broadcast`) and finish pending inflight batches before exiting cleanly.
* **Bounded Batching & Queue Backpressure:**
  - Before pushing ingested records into `ModelController` or `DynamicBatcher`, check `controller.queue_len()`.
  - When the queue exceeds configured limits (`max_queue_size`), apply backpressure: pause broker polling or await queue capacity rather than dropping events or consuming unbounded messages into memory.

---

## 2. At-Least-Once Delivery & Commit-After-Infer
* **Commit Invariant:** Never acknowledge (`ack`) a message or commit a partition offset until:
  1. The input tensor is extracted from the message payload.
  2. The model forward pass completes successfully (`infer`).
  3. The resulting output tensors are serialized and dispatched to the configured `egress` broker/topic (if egress is configured).
* **Failure Handling & NACK:**
  - On transient broker errors or model load timeouts (`InferenceError::Loading`, `InferenceError::Timeout`), apply exponential retry backoff with jitter.
  - On non-retryable format errors (`InferenceError::InvalidInput`), reject/nack the message or route to a dead-letter queue (DLQ) if configured, preventing message loop deadlocks.

---

## 3. Zero-Panic Async Reactor Safety
* **No Unwraps in Consumer Loops:** As with all SnoozeML code, never call `.unwrap()`, `.expect()`, or `panic!()` in stream drivers. Network reconnections, broker disconnects, and malformed frames must be mapped into `StreamError` or handled with reconnect loops.
* **Exponential Backoff on Disconnect:** If a broker connection drops, the driver must sleep with exponential backoff (e.g. 50ms, 100ms, 200ms, ... up to a max cap like 5s) before retrying. Never spin tight loops on failed connection attempts.

---

## 4. Feature Flag Hygiene
* **Broker-Specific Feature Gating:** Every broker implementation must be conditionally compiled behind its respective feature flag:
  - `streams`: Base streaming infrastructure (`StreamDriver`, `StreamMessage`, `MockStreamDriver`).
  - `streams-kafka`: Apache Kafka driver using `rskafka`.
  - `streams-rabbitmq`: RabbitMQ driver using `lapin`.
  - `streams-redis`: Redis Streams driver using `redis`.
  - `streams-all`: Meta-feature enabling all brokers.
* **Clean Fallbacks:** Provide clear error messages when a user attempts to use a broker type not compiled into their binary:
  ```rust
  #[cfg(not(feature = "streams-kafka"))]
  return Err(StreamError::Configuration("Kafka streaming support not compiled; enable 'streams-kafka' feature".into()));
  ```

---

## 5. Verification & Testing
* **Mock Stream Testing:** Always test streaming behavior, message transformation, and offset acking using `MockStreamDriver` in `tests/stream_test.rs` without requiring external broker daemons.
* **Feature Build Checks:** Ensure all streaming feature combinations compile without warnings:
  ```powershell
  cargo check --no-default-features --features streams
  cargo check --no-default-features --features streams-kafka
  cargo check --no-default-features --features streams-redis
  cargo check --no-default-features --features streams-rabbitmq
  cargo test --test stream_test --features streams-all
  ```
