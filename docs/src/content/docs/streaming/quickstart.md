---
title: "Event Streaming Quickstart & Guide"
description: "SnoozeML provides first-class, high-throughput integration with distributed message brokers and event streaming platforms, including **Apache Kafka**, **Rabb..."
---

SnoozeML provides first-class, high-throughput integration with distributed message brokers and event streaming platforms, including **Apache Kafka**, **RabbitMQ**, and **Redis Streams**.

Like HTTP REST and gRPC, event streaming in SnoozeML is a **completely decoupled transport layer**. A stream worker is not locked to a single model—by default, **all models registered in SnoozeML can be invoked over any stream**.

---

## 1. Feature Highlights

- **Decoupled Multi-Model Transport**: A single stream (e.g. an incoming Kafka topic) can dynamically dispatch inference requests to any registered model in SnoozeML.
- **Asynchronous & Non-Blocking**: Ingestion, inference dispatch, and response publishing run in isolated Tokio background workers. Slow brokers or stream bursts **never** block HTTP REST or gRPC endpoints.
- **Adaptive Micro-Batching Integration**: Stream requests enter the central dynamic batcher, seamlessly coalescing with incoming HTTP and gRPC traffic along `Axis(0)` to saturate CPU, GPU, and tree accelerators.
- **Automatic Broker Provisioning (`auto_create_topics = true`)**: By default, missing Kafka topics, RabbitMQ queues, and Redis consumer groups are created automatically on startup.
- **Dynamic Reply-To Routing**: Requests can specify dynamic return destinations (`reply_to`) to route outputs to client-specific queues.
- **Dead-Letter Queue (DLQ) Resilience**: Corrupted JSON, malformed Protobuf, or invalid tensor dimensions route to a configured DLQ with diagnostics, acknowledging the broker to prevent poison-pill redelivery loops.
- **Dual Wire Formats**: Supports standard human-readable JSON and ultra-fast binary Protobuf with raw memory float buffers.
- **Built-in Backpressure**: Bounded concurrency semaphores throttle consumption when inference queues are full, preventing out-of-memory errors during traffic spikes.
- **Prometheus Observability**: Comprehensive counters track consumed messages, published responses, and stage-specific errors.

---

## 2. Quick Start (Minimal Configuration)

SnoozeML is designed for instant developer ergonomics. You do not need to pre-create topics or write complex routing rules.

### Step 1: Add a Minimal Stream to `models.toml`

Add a `[[streams]]` table to your `models.toml` file. The absolute minimum configuration requires only 4 lines:

```toml
[[streams]]
name = "my-stream"
broker = "redis"                # Or "kafka", "rabbitmq"
endpoint = "127.0.0.1:6379"     # Broker connection address
input_topic = "snooze.requests" # Input stream/queue to consume from
```

:::tip
**Why is this so simple?**
- `auto_create_topics` defaults to `true`: SnoozeML creates the queue/stream automatically.
- `output_topic` defaults to `"snooze.responses"`.
- `concurrency` defaults to `16` workers with backpressure.
- `format` defaults to `"json"`.
- `model_id` is omitted: **all models** in your model directory are immediately callable!
:::

### Step 2: Start SnoozeML

Launch the daemon pointing to your models directory:

```bash
snooze serve --models-dir ./models
```

You will see SnoozeML discover your models, initialize the broker connection, auto-create missing topics, and begin consuming:

```text
INFO Initializing stream worker stream="my-stream" broker="redis" input_topic="snooze.requests"
INFO Ensuring topics/queues exist on broker stream="my-stream"
INFO Stream worker successfully started; consuming messages stream="my-stream"
```

### Step 3: Publish Your First Inference Request

Publish a JSON payload to your input topic (`snooze.requests`). Notice that you specify the target `model_id` directly in the request:

```json
{
  "id": "req-001",
  "model_id": "demo-model",
  "inputs": [
    {
      "name": "input",
      "shape": [1, 4],
      "data": [1.0, 2.0, 3.0, 4.0]
    }
  ]
}
```

### Step 4: Read the Response

SnoozeML runs the request through dynamic micro-batching and scale-to-zero memory management, publishing the result to `snooze.responses`:

```json
{
  "id": "req-001",
  "model_id": "demo-model",
  "status": "success",
  "outputs": [
    {
      "name": "output",
      "shape": [1, 4],
      "data": [2.0, 4.0, 6.0, 8.0]
    }
  ],
  "duration_ms": 0.85
}
```

---

## 3. Configuration & Customization Reference

Every aspect of stream processing can be tailored to your production infrastructure.

### Complete Options Reference

| Parameter | Type | Default | Required? | Description |
| :--- | :--- | :--- | :--- | :--- |
| `name` | String | `"default-stream"` | No | Descriptive label for logs and metrics. |
| `broker` | String | `"mock"` | **Yes** | Broker driver: `"kafka"`, `"rabbitmq"`, `"redis"`, or `"mock"`. |
| `endpoint` | String | `"memory://"` | **Yes** | Broker connection address or bootstrap servers. |
| `input_topic` | String | `"snooze.requests"` | **Yes** | Source topic, queue, or stream key to consume from. |
| `output_topic`| Option\<String\> | `"snooze.responses"`| No | Default destination topic/queue for successful responses. |
| `error_topic` | Option\<String\> | `None` | No | Dead-Letter Queue (DLQ) topic for invalid payloads and errors. |
| `model_id` | Option\<String\> | `None` | No | Fallback default model. If omitted, stream serves **all models**. |
| `format` | String | `"json"` | No | Wire format: `"json"` or `"protobuf"`. |
| `auto_create_topics` | Boolean | `true` | No | Automatically declare missing topics, queues, or streams. |
| `concurrency` | Integer | `16` | No | Maximum in-flight concurrent requests (backpressure semaphore). |
| `partitions` | Integer | `1` | No | Partition count when auto-creating Kafka topics. |
| `consumer_group` | Option\<String\> | `None` | No | Consumer group ID for Kafka or Redis Streams. |
| `prefetch_count` | Option\<Integer\> | `None` | No | RabbitMQ basic QoS channel prefetch limit. |

---

## 4. Customizing Behavior by Use Case

### 4.1 Broker Selection

#### Apache Kafka (`broker = "kafka"`)
SnoozeML includes a native, pure-Rust Kafka driver powered by `rskafka`. It does not require `librdkafka`, CMake, or external C compilers:

```toml
[[streams]]
name = "kafka-pipeline"
broker = "kafka"
endpoint = "10.0.0.5:9092,10.0.0.6:9092"
input_topic = "ml.inference.in"
output_topic = "ml.inference.out"
error_topic = "ml.inference.dlq"
consumer_group = "snooze-cluster"
partitions = 6
```

#### RabbitMQ (`broker = "rabbitmq"`)
Uses AMQP 0-9-1 with durable queue declarations and consumer acknowledgments:

```toml
[[streams]]
name = "amqp-worker"
broker = "rabbitmq"
endpoint = "amqp://user:password@rabbitmq.internal:5672/%2f"
input_topic = "tasks.queue"
output_topic = "tasks.results"
prefetch_count = 32
```

#### Redis Streams (`broker = "redis"`)
Uses Redis consumer groups (`XREADGROUP`, `XACK`, `XADD`) with automatic `MKSTREAM` creation:

```toml
[[streams]]
name = "redis-cluster"
broker = "redis"
endpoint = "redis://127.0.0.1:6379"
input_topic = "events.stream"
output_topic = "events.completed"
consumer_group = "snooze-group"
```

---

### 4.2 Decoupled Multi-Model Serving vs. Dedicated Binding

#### Decoupled (Default - Recommended)
When `model_id` is omitted, the stream worker inspects each incoming message's `model_id` field and dispatches to that model:

```toml
[[streams]]
name = "universal-stream"
broker = "kafka"
endpoint = "127.0.0.1:9092"
input_topic = "all.models.in"
output_topic = "all.models.out"
# model_id is omitted: accepts requests for "bert", "resnet", "xgboost", etc.
```

#### Dedicated Model Binding
If you want a dedicated queue exclusively for a specific high-priority model, set `model_id`:

```toml
[[streams]]
name = "fraud-detector-stream"
broker = "kafka"
endpoint = "127.0.0.1:9092"
model_id = "fraud-model" # Incoming requests don't need to specify model_id
input_topic = "transactions.raw"
output_topic = "transactions.scored"
```

---

### 4.3 Dynamic Reply-To Routing

Instead of routing all responses to a single static `output_topic`, clients can specify a dynamic return topic in their request:

```json
{
  "id": "req-987",
  "model_id": "sentiment-model",
  "reply_to": "client-42.private.queue",
  "inputs": [ ... ]
}
```

SnoozeML publishes the result directly to `client-42.private.queue`. If `reply_to` is omitted, SnoozeML falls back to the configured `output_topic`.

---

### 4.4 Dead-Letter Queue (DLQ) & Poison Pill Isolation

In production streaming, corrupted payloads or shape mismatches must not stall worker threads or trigger infinite retry loops.

Configure `error_topic`:

```toml
[[streams]]
name = "production-stream"
broker = "kafka"
endpoint = "127.0.0.1:9092"
input_topic = "data.in"
output_topic = "data.out"
error_topic = "data.dlq"
```

When an error occurs (such as invalid JSON, unparseable Protobuf, unknown `model_id`, or mismatched tensor shapes):
1. The error message is formatted with diagnostic context and the original correlation ID.
2. The payload is published to `data.dlq`.
3. The original broker message is positively acknowledged (`ack`).
4. `snoozeml_stream_errors_total` metric is incremented.

---

### 4.5 High-Throughput Binary Protobuf Format

For heavy workloads (vision tensors, large embeddings), JSON float array parsing incurs CPU and memory overhead. Setting `format = "protobuf"` enables binary wire serialization:

```toml
[[streams]]
name = "fast-binary-stream"
broker = "kafka"
endpoint = "127.0.0.1:9092"
input_topic = "embeddings.in"
output_topic = "embeddings.out"
format = "protobuf"
```

In Protobuf mode, payloads use the SnoozeML native schema:
- **Input**: `snoozeml.v1.ModelInferRequest` with raw IEEE 754 float byte buffers (`raw_data`).
- **Output**: `snoozeml.v1.ModelInferResponse` with zero-copy float arrays.

---

### 4.6 Concurrency & Backpressure Tuning

The `concurrency` setting dictates the maximum number of in-flight inference tasks a stream worker may submit simultaneously:

```toml
[[streams]]
concurrency = 64
```

- **How it works**: SnoozeML acquires a permit from a `tokio::sync::Semaphore` before polling the next message.
- **Why it matters**: If your model is executing a heavy batch or waiting on GPU VRAM, consumption pauses automatically, communicating backpressure to Kafka or RabbitMQ and preventing out-of-memory crashes.

---

## 5. Prometheus Metrics Reference

Every stream worker exports real-time metrics on the Prometheus `/metrics` endpoint:

| Metric | Type | Labels | Description |
| :--- | :--- | :--- | :--- |
| `snoozeml_stream_messages_consumed_total` | Counter | `stream_id`, `driver` | Count of messages successfully pulled from broker. |
| `snoozeml_stream_messages_published_total` | Counter | `stream_id`, `topic` | Count of responses successfully published. |
| `snoozeml_stream_errors_total` | Counter | `stream_id`, `stage` | Count of errors (`"malformed_request"`, `"missing_model_id"`, `"invalid_tensors"`, `"inference_error"`, `"publish_failed"`). |