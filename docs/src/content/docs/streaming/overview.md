---
title: "Event Streaming Architecture & Drivers"
description: "SnoozeML provides first-class, high-throughput asynchronous integration with distributed event streaming platforms and message brokers."
---

SnoozeML provides first-class, high-throughput asynchronous integration with distributed event streaming platforms and message brokers.

---

## 1. Core Architectural Invariants

### 1.1 Decoupled Multi-Model Transport
Streams in SnoozeML are a completely decoupled transport layer, exactly like HTTP REST and gRPC:
- A stream worker is **not** bound to a single model. By default, **all models registered in SnoozeML are available over any stream**.
- Each incoming stream message specifies the target model via `model_id` (identical to the URL path `/v1/models/:model_id/infer` in HTTP or the `model_id` field in gRPC `ModelInferRequest`).
- Specifying `model_id` in a stream's configuration is entirely optional; when provided, it acts as a default/fallback if an incoming message omits `model_id`, or allows dedicating a private queue to a specific model.

### 1.2 Completely Non-Blocking
Traditional ML serving setups often block web server threads when processing queue messages or ingest Kafka events on synchronous loops. In SnoozeML:
- Message consumers, infer dispatchers, and output publishers run inside independent asynchronous Tokio worker loops supervised by `StreamSupervisor`.
- HTTP REST (`src/server/http.rs`) and native gRPC (`src/server/grpc/`) endpoints operate with zero latency impact or blocking interference from stream processing workloads.

### 1.2 Seamless Dynamic Micro-Batching Integration
Stream messages do not bypass or duplicate the batching runtime:
- Upon deserialization, each consumed message creates an asynchronous inference job submitted directly to `ModelController::infer`.
- The `DynamicBatcher` background task automatically collates tensors along `Axis(0)` across both HTTP, gRPC, and stream requests into optimal micro-batches within configured batch sizes (`max_batch_size`) and delay limits (`max_delay_ms`).
- Sliced outputs are then delivered back to the stream worker for non-blocking publishing to the target output topic.

### 1.3 Backpressure & Concurrency Control
Each stream binding features a bounded `tokio::sync::Semaphore` with `concurrency` permits (default: `16`):
- Before consuming and submitting a message to the inference engine, a permit is acquired.
- If inference queues are saturated, consumption throttles automatically, respecting broker backpressure mechanisms.

### 1.4 Automatic Topic & Queue Creation
By default (`auto_create_topics = true`), SnoozeML detects missing broker destinations and automatically creates them before starting consumer loops:
- **Kafka**: Issues metadata queries and `CreateTopics` requests with configured partition counts.
- **RabbitMQ**: Declares durable queues with `passive = false`.
- **Redis Streams**: Executes `XGROUP CREATE ... MKSTREAM` to create both the stream and consumer group if they do not already exist.

### 1.5 Dead-Letter Queue (DLQ) & Poison Pill Protection
When encountering malformed payloads, invalid JSON/Protobuf, or tensor shape mismatches:
- The error is captured and formatted into an error response envelope.
- If `dlq_topic` is configured, the error payload is published to the DLQ.
- The original broker message is positively acknowledged (`ack`) so corrupted payloads never cause infinite retry loops.

---

## 2. Supported Drivers

### Apache Kafka
- **Engine**: Pure-Rust [`rskafka`](https://github.com/influxdata/rskafka).
- **Toolchain**: 100% pure Rust with zero C `librdkafka` or CMake toolchain requirements.
- **Capabilities**: Partition-level consumption, auto topic creation, offset tracking, record publishing.

### RabbitMQ (AMQP 0-9-1)
- **Engine**: Asynchronous [`lapin`](https://github.com/CleverCloud/lapin).
- **Capabilities**: Basic QoS prefetch limits, durable queues, consumer acks and nacks.

### Redis Streams
- **Engine**: Asynchronous [`redis-rs`](https://github.com/redis-rs/redis-rs).
- **Capabilities**: Distributed consumer groups (`XGROUP`, `XREADGROUP`, `XACK`), auto-stream creation (`MKSTREAM`), and result publishing via `XADD`.

### In-Memory Mock (`mock:`)
- **Engine**: Zero-dependency broadcast channels.
- **Capabilities**: Used for deterministic local unit testing and automated CI pipelines.