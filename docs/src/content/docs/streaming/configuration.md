---
title: "Stream Configuration & DLQ"
description: "SnoozeML stream bindings can be defined directly in `models.toml` or inside an independent TOML configuration file passed via `--streams-config <PATH>`."
---

SnoozeML stream bindings can be defined directly in `models.toml` or inside an independent TOML configuration file passed via `--streams-config <PATH>`.

---

## 1. Stream Configuration Schema

Each binding is declared with a `[[streams]]` table:

```toml
[[streams]]
id = "orders-stream"
# model_id is omitted: this stream can serve ALL models in the registry!
driver = "kafka"
endpoint = "127.0.0.1:9092"
input_topic = "orders.unscored"
output_topic = "orders.scored"
dlq_topic = "orders.dlq"
format = "json"
auto_create_topics = true
concurrency = 32
partitions = 3
group_id = "snooze-inference-workers"
```

### Configuration Options

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `id` | string | *Required* | Unique name identifying this stream worker. |
| `model_id` | string | *Optional* (`None`) | Default model identifier. If omitted (`None`), the stream dynamically routes to the `model_id` specified in each incoming message, serving all registered models. |
| `driver` | string | *Required* | Stream driver (`"kafka"`, `"rabbitmq"`, `"redis"`, or `"mock"`). |
| `endpoint` | string | *Required* | Broker connection string (e.g. `"127.0.0.1:9092"`, `"amqp://user:pass@host:5672"`, `"redis://127.0.0.1:6379"`). |
| `input_topic` | string | *Required* | Source topic, queue, or stream key. |
| `output_topic` | string | *Optional* | Destination topic, queue, or stream key for inference outputs. |
| `dlq_topic` | string | *Optional* | Destination for invalid inputs or execution errors. |
| `format` | string | `"json"` | Message serialization format: `"json"` or `"protobuf"`. |
| `auto_create_topics` | boolean | `true` | When `true`, automatically creates missing broker topics/queues. |
| `concurrency` | integer | `16` | Maximum concurrent in-flight inference tasks per stream. |
| `partitions` | integer | `1` | Partition count when creating new Kafka topics. |
| `group_id` | string | *Optional* | Consumer group name (Kafka / Redis Streams). |
| `consumer_id` | string | *Optional* | Consumer member name (Redis Streams). |
| `prefetch_count` | integer | *Optional* | Channel prefetch QoS limit (RabbitMQ). |

---

## 2. Wire Formats

### 2.1 JSON Format (`format = "json"`)

#### Input Payload
```json
{
  "id": "order-req-1234",
  "model_id": "fraud-detection-model",
  "inputs": [
    {
      "name": "features",
      "shape": [1, 4],
      "data": [10.5, 2.0, 300.0, 1.0]
    }
  ]
}
```

#### Successful Output Payload
```json
{
  "id": "order-req-1234",
  "model_id": "fraud-detection-model",
  "status": "success",
  "outputs": [
    {
      "name": "probabilities",
      "shape": [1, 2],
      "data": [0.03, 0.97]
    }
  ],
  "duration_ms": 1.45
}
```

#### Dead-Letter Queue (DLQ) Error Payload
```json
{
  "id": "order-req-1234",
  "model_id": "fraud-detection-model",
  "status": "error",
  "error": "Tensor 'features' shape [1, 4] expects 4 elements, but received 3"
}
```

### 2.2 Protobuf Format (`format = "protobuf"`)
For maximum throughput and zero float string parsing overhead, set `format = "protobuf"`. Messages are serialized using the SnoozeML native Protobuf schema (`snoozeml.v1.ModelInferRequest` and `snoozeml.v1.ModelInferResponse`).

---

## 3. Observability & Telemetry

Stream processing metrics are exported automatically on the Prometheus `/metrics` endpoint:
- `snoozeml_stream_messages_consumed_total{stream_id="...", driver="..."}`: Total messages pulled from broker.
- `snoozeml_stream_messages_published_total{stream_id="...", topic="..."}`: Total outputs successfully published.
- `snoozeml_stream_errors_total{stream_id="...", stage="..."}`: Errors encountered during deserialization, inference, or publishing.