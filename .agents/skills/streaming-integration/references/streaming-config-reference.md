# Streaming Broker Configuration Reference

This reference details the configuration schema for Kafka, RabbitMQ, and Redis Streams in SnoozeML's `models.toml`.

---

## 1. Apache Kafka (`streams-kafka`)

```toml
[[streams]]
stream_id = "kafka-pipeline"
model_id = "text_classifier"
batch_size = 64
max_latency_ms = 15
input_tensor = "input_ids"
output_tensor = "logits"

[streams.ingest]
broker_type = "kafka"
endpoints = ["10.0.0.1:9092", "10.0.0.2:9092"]
topic = "incoming-text-events"
group_id = "snooze-inference-cg"
auto_offset_reset = "latest" # "earliest" or "latest"

[streams.egress]
broker_type = "kafka"
endpoints = ["10.0.0.1:9092"]
topic = "classification-results"
```

---

## 2. RabbitMQ (`streams-rabbitmq`)

```toml
[[streams]]
stream_id = "rabbitmq-pipeline"
model_id = "fraud_detector"
batch_size = 16
max_latency_ms = 5
input_tensor = "transaction_data"
output_tensor = "fraud_probability"

[streams.ingest]
broker_type = "rabbitmq"
endpoints = ["amqp://guest:guest@localhost:5672/%2f"]
queue = "incoming-transactions"
prefetch_count = 64

[streams.egress]
broker_type = "rabbitmq"
endpoints = ["amqp://guest:guest@localhost:5672/%2f"]
exchange = "fraud-alerts"
routing_key = "alerts.high"
```

---

## 3. Redis Streams (`streams-redis`)

```toml
[[streams]]
stream_id = "redis-pipeline"
model_id = "recommendation_model"
batch_size = 32
max_latency_ms = 10
input_tensor = "user_features"
output_tensor = "item_scores"

[streams.ingest]
broker_type = "redis"
endpoints = ["redis://127.0.0.1:6379"]
stream_key = "user-activity-stream"
group_name = "snooze-recommender"
consumer_name = "worker-1"

[streams.egress]
broker_type = "redis"
endpoints = ["redis://127.0.0.1:6379"]
stream_key = "user-recommendations-stream"
```

---

## 4. In-Memory Mock Stream (`mock`)

Used for fast deterministic integration tests without running external broker infrastructure:

```toml
[[streams]]
stream_id = "test-mock-pipeline"
model_id = "mock"
batch_size = 4
max_latency_ms = 2
input_tensor = "input"
output_tensor = "output"

[streams.ingest]
broker_type = "mock"
channel_capacity = 1024

[streams.egress]
broker_type = "mock"
channel_capacity = 1024
```
