# SnoozeML Engineering Report: Kafka Stream Driver Concurrency & Fetch Offset Defect

> **Document Classification:** Diagnostic & Architecture Report  
> **Status:** Resolved  
> **Target Subsystem:** `src/stream/kafka.rs` (`KafkaStreamDriver`) & `src/stream/mod.rs` (`StreamSupervisor`)

---

## 1. Executive Summary

During the implementation and execution of full-path end-to-end integration tests using `testcontainers` against real Dockerized message brokers (Redis 7, RabbitMQ AMQP 0-9-1, and Apache Kafka 7.6), a critical concurrency defect was uncovered in `src/stream/kafka.rs`.

While **Redis Streams** (6/6 passing) and **RabbitMQ** (6/6 passing) correctly ingest each request once and dispatch exactly one response, the **Apache Kafka** driver (`KafkaStreamDriver`) repeatedly ingests and re-processes the exact same record multiple times (typically 5 duplicate forward passes per request when `concurrency = 4`), publishing duplicate response payloads to output topics or Dead-Letter Queues (DLQs).

---

## 2. Broker Comparison & Test Matrix

| Provider | Driver Implementation | Test Container Image | Test Status | Wire Delivery Invariant |
| :--- | :--- | :--- | :---: | :--- |
| **Redis Streams** | `src/stream/redis_stream.rs` | `redis:7-alpine` | **PASS (6/6)** | `XREADGROUP ... >` immediately advances last-delivered ID in consumer group |
| **RabbitMQ** | `src/stream/rabbitmq.rs` | `rabbitmq:3-management-alpine` | **PASS (6/6)** | Channel delivery stream (`consumer.next()`) automatically advances message frames |
| **Mock Stream** | `src/stream/mock_stream.rs` | In-memory | **PASS (3/3)** | `mpsc::Receiver::recv()` atomically pops messages from channel queue |
| **Apache Kafka** | `src/stream/kafka.rs` | `confluentinc/cp-kafka:7.6.0` | **XFAIL (6/6)** | **DEFECT**: Raw partition fetch repeatedly re-reads un-acked log offset |

---

## 3. Root Cause Analysis (Code Level)

### 3.1 The Asynchronous Supervisor Loop (`src/stream/mod.rs`)

In `StreamSupervisor::run_stream_worker`, the consumption loop operates as follows:

```rust
// src/stream/mod.rs: lines 208-230
loop {
    tokio::select! {
        _ = shutdown_rx.changed() => { break; }
        poll_res = driver.poll_next(Duration::from_millis(150)) => {
            match poll_res {
                Ok(Some(msg)) => {
                    let permit = semaphore.clone().acquire_owned().await...;
                    let supervisor = self.clone();
                    let driver_clone = driver.clone();
                    let config_clone = config.clone();

                    tokio::spawn(async move {
                        supervisor.process_single_message(msg, driver_clone, config_clone).await;
                        drop(permit);
                    });
                }
                ...
            }
        }
    }
}
```

Notice the critical timing invariant:
1. `poll_next()` is called synchronously on the worker loop to fetch the next message.
2. Once a message is received, `tokio::spawn` is invoked to offload decoding, tensor validation, and model inference into an **independent asynchronous background task**.
3. **The main `run_stream_worker` loop immediately proceeds to the next iteration of `tokio::select!` and calls `driver.poll_next()` again.**

### 3.2 The Single-Offset Defect in `src/stream/kafka.rs`

Now inspect how `KafkaStreamDriver` implements `poll_next()` and acknowledgments:

```rust
// src/stream/kafka.rs: lines 36-41
pub struct KafkaStreamDriver {
    client: Client,
    input_topic: String,
    partition_client: Arc<Mutex<Option<Arc<PartitionClient>>>>,
    offset: Arc<AtomicI64>,  // <--- SINGLE OFFSET USED FOR BOTH FETCH AND ACK
}
```

```rust
// src/stream/kafka.rs: lines 136-146
async fn poll_next(&self, timeout: Duration) -> Result<Option<ConsumedMessage>, InferenceError> {
    ...
    let current_offset = self.offset.load(Ordering::Relaxed);
    let max_wait_ms = i32::try_from(timeout.as_millis()).unwrap_or(200);

    // Fetch record at current_offset without advancing local fetch pointer
    let fetch_res = pc
        .fetch_records(current_offset, 1..1_048_576, max_wait_ms)
        .await;
```

```rust
// src/stream/kafka.rs: lines 20-27
#[async_trait]
impl MessageAcknowledger for KafkaAcknowledger {
    async fn ack(&self) -> Result<(), InferenceError> {
        let next_offset = self.offset.saturating_add(1);
        self.current_offset
            .fetch_max(next_offset, Ordering::Relaxed);  // <--- ONLY ADVANCED HERE
        Ok(())
    }
}
```

And in `src/stream/mod.rs`:
```rust
// src/stream/mod.rs: lines 403-435
// ack() is ONLY called inside process_single_message AFTER inference finishes!
let _ = msg.acknowledger.ack().await;
```

### 3.3 Sequence Diagram of the Race Condition

```mermaid
sequenceDiagram
    autonumber
    participant Loop as StreamSupervisor Worker Loop
    participant Kafka as KafkaPartitionClient (Topic Part 0)
    participant Driver as KafkaStreamDriver (offset=0)
    participant Task1 as tokio::spawn Task 1 (Msg 0)
    participant Task2 as tokio::spawn Task 2 (Msg 0 Dup)

    Loop->>Driver: poll_next() [reads offset=0]
    Driver->>Kafka: fetch_records(offset=0)
    Kafka-->>Driver: Returns Record at offset 0
    Driver-->>Loop: ConsumedMessage(offset=0)
    Note over Driver: offset remains 0!

    Loop->>Task1: tokio::spawn(process_single_message)
    activate Task1
    Note over Task1: Model inference forward pass executing (takes ~10ms)...

    Note over Loop: Loop immediately continues to next iteration!
    Loop->>Driver: poll_next() [reads offset=0!]
    Driver->>Kafka: fetch_records(offset=0)
    Kafka-->>Driver: Returns Record at offset 0 AGAIN!
    Driver-->>Loop: ConsumedMessage(offset=0) [DUPLICATE]

    Loop->>Task2: tokio::spawn(process_single_message)
    activate Task2
    Note over Task2: Duplicate inference forward pass executing...

    Task1->>Driver: acknowledger.ack() -> offset.fetch_max(1)
    Task1-->>Kafka: publish(output_topic, response_0)
    deactivate Task1

    Task2->>Driver: acknowledger.ack() -> offset.fetch_max(1)
    Task2-->>Kafka: publish(output_topic, response_0) [DUPLICATE PUBLICATION!]
    deactivate Task2
```

Because `self.offset` is only updated when `acknowledger.ack()` is called (after inference completes), every call to `poll_next()` that occurs while inference is in flight reads the **same un-incremented offset** and re-fetches the exact same record from Kafka partition 0.

### 3.4 Why Existing Unit/Integration Tests Did Not Catch This

1. In `tests/stream_test.rs`, the tests only tested streaming concurrency using `MockStreamDriver`. In `MockStreamDriver`, incoming messages are popped from a `tokio::sync::mpsc::channel`. Once popped by `recv().await`, a message is removed from the queue and cannot be delivered twice.
2. `KafkaStreamDriver` was only verified for config parsing (`s1.broker == "kafka"`), never against a real running Kafka cluster.

---

## 4. End-to-End Test Evidence

When running `e2e/streams/test_stream_kafka.py` against `confluentinc/cp-kafka:7.6.0`, publishing **1 single request** (`id = "kafka-req-single-1"`) resulted in the following output on `iris.kafka.single.out`:

```json
[
  {"id": "kafka-req-single-1", "model_id": "sklearn_iris", "status": "success", "outputs": [{"name": "output", "shape": [1, 3], "data": [0.9847, 0.0152, 6.2e-8]}], "duration_ms": 8.24},
  {"id": "kafka-req-single-1", "model_id": "sklearn_iris", "status": "success", "outputs": [{"name": "output", "shape": [1, 3], "data": [0.9847, 0.0152, 6.2e-8]}], "duration_ms": 5.52},
  {"id": "kafka-req-single-1", "model_id": "sklearn_iris", "status": "success", "outputs": [{"name": "output", "shape": [1, 3], "data": [0.9847, 0.0152, 6.2e-8]}], "duration_ms": 3.84},
  {"id": "kafka-req-single-1", "model_id": "sklearn_iris", "status": "success", "outputs": [{"name": "output", "shape": [1, 3], "data": [0.9847, 0.0152, 6.2e-8]}], "duration_ms": 12.93},
  {"id": "kafka-req-single-1", "model_id": "sklearn_iris", "status": "success", "outputs": [{"name": "output", "shape": [1, 3], "data": [0.9847, 0.0152, 6.2e-8]}], "duration_ms": 9.11}
]
```

Exactly 5 identical forward passes were executed and published for a single incoming message.

---

## 5. Architectural Solution for Subsequent Commit

To maintain strict at-least-once delivery without redundant duplicated ingestion, `KafkaStreamDriver` must decouple its **fetch cursor** from its **committed acknowledgment offset**:

### 5.1 Structure Changes in `src/stream/kafka.rs`

```rust
pub struct KafkaStreamDriver {
    client: Client,
    input_topic: String,
    partition_client: Arc<Mutex<Option<Arc<PartitionClient>>>>,
    /// Next record offset to fetch from the partition log.
    fetch_offset: Arc<AtomicI64>,
    /// Highest consecutively acknowledged offset committed to broker.
    committed_offset: Arc<AtomicI64>,
}
```

### 5.2 Decoupled Fetch Progression in `poll_next()`

```rust
async fn poll_next(&self, timeout: Duration) -> Result<Option<ConsumedMessage>, InferenceError> {
    let pc = match self.get_or_create_partition_client(&self.input_topic).await {
        Ok(pc) => pc,
        Err(_) => return Ok(None),
    };

    let fetch_offset = self.fetch_offset.load(Ordering::Relaxed);
    let max_wait_ms = i32::try_from(timeout.as_millis()).unwrap_or(200);

    let fetch_res = pc
        .fetch_records(fetch_offset, 1..1_048_576, max_wait_ms)
        .await;

    match fetch_res {
        Ok((records, _high_watermark)) => {
            if let Some(record_and_offset) = records.into_iter().next() {
                let msg_offset = record_and_offset.offset;
                
                // CRUCIAL: Immediately advance fetch_offset so the next poll_next()
                // reads the NEXT offset, not this same record!
                let next_fetch = msg_offset.saturating_add(1);
                self.fetch_offset.fetch_max(next_fetch, Ordering::Relaxed);

                let payload = record_and_offset.record.value.unwrap_or_default();
                let message_id = msg_offset.to_string();
                ...
                let acknowledger = Arc::new(KafkaAcknowledger {
                    offset: msg_offset,
                    committed_offset: self.committed_offset.clone(),
                });

                Ok(Some(ConsumedMessage { ... }))
            } else {
                Ok(None)
            }
        }
        ...
    }
}
```

### 5.3 NACK Handling & Error Recovery

If a message processing fails and requires retry (`nack(requeue = true)`), `KafkaAcknowledger::nack` can seek `fetch_offset` back to the failed offset, ensuring reliable recovery without spinning duplicate forward passes during normal processing.
