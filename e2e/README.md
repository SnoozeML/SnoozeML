# SnoozeML End-to-End (E2E) System & Integration Tests

This directory (`e2e/`) contains comprehensive black-box system, wire-interoperability, and event-streaming integration tests that execute against live `snooze` binary processes and containerized message brokers.

Unlike `python/tests/` (which unit-tests the Python SDK in isolation) and `tests/` (which tests Rust internals and in-process routers), `e2e/` tests the entire compiled binary against real third-party protocols and real broker instances spun up via `testcontainers`.

---

## Directory Structure

```text
e2e/
├── README.md               # This guide and status report
├── requirements.txt        # Python dependencies (tritonclient, testcontainers, redis, pika, kafka-python-ng)
├── conftest.py             # Shared pytest fixtures, ephemeral server launcher, and reference ONNX models
├── triton/                 # Triton Inference Server protocol interoperability suite
│   ├── test_triton_grpc.py # Official tritonclient.grpc suite (unary, bidirectional streaming, RBAC)
│   └── test_triton_http.py # Official tritonclient.http suite (JSON, Triton binary extensions, RBAC)
├── docker/                 # Containerized SnoozeML suite (using testcontainers & Docker Compose)
│   ├── docker-compose.test.yml         # Compose service specification for testing the SnoozeML container
│   ├── test_containerized_daemon.py    # Full endpoint & real model validation against OCI container
│   └── test_container_python_client.py # Complete Python SDK client (sync & async) against CPU container
└── streams/                # Event streaming broker integration suite (using testcontainers)
    ├── test_stream_redis.py    # Redis Streams driver test suite (redis:7-alpine)
    ├── test_stream_rabbitmq.py # RabbitMQ AMQP 0-9-1 driver test suite (rabbitmq:3-management-alpine)
    ├── test_stream_kafka.py    # Apache Kafka driver test suite (confluentinc/cp-kafka:7.6.0)
    └── test_stream_mock.py     # Deterministic mock stream & SDK MockStreamClient test suite
```

---

## Prerequisites

1. **Build the `snooze` binary with streaming features**:
   ```bash
   cargo build --bin snooze --features streams-all
   ```
   *(The test fixtures automatically discover the newest binary in `target/release/` or `target/debug/`, or you can set `SNOOZE_BIN=target/debug/snooze`.)*

2. **Docker Engine**:
   Docker Desktop or local `dockerd` must be running for the `testcontainers` suites (`RedisContainer`, `RabbitMqContainer`, `KafkaContainer`).

3. **Install Python dependencies into your virtual environment**:
   ```bash
   pip install -r e2e/requirements.txt
   pip install -e ./python
   ```

---

## Running the Tests

### 1. Run the Entire E2E Test Suite
```bash
pytest e2e -v
```

### 2. Run Only Triton Interoperability Tests
```bash
# Run both HTTP and gRPC Triton tests
pytest e2e/triton -v

# Run only gRPC (unary + bidirectional stream + RBAC)
pytest e2e/triton/test_triton_grpc.py -v

# Run only HTTP (JSON + binary tensor framing + RBAC)
pytest e2e/triton/test_triton_http.py -v
```

### 3. Run Event Streaming Broker Tests
```bash
# Run all stream broker integration tests
pytest e2e/streams -v

# Run only Redis Streams tests (requires Docker)
pytest e2e/streams/test_stream_redis.py -v

# Run only RabbitMQ AMQP tests (requires Docker)
pytest e2e/streams/test_stream_rabbitmq.py -v

# Run only Kafka stream tests (requires Docker)
pytest e2e/streams/test_stream_kafka.py -v

# Run Mock stream and Python SDK MockStreamClient tests
pytest e2e/streams/test_stream_mock.py -v
```

### 4. Run Containerized Daemon Tests (Docker Compose & testcontainers)
```bash
# Build the local CPU Docker image first (zero CUDA)
docker build --target slim -t snooze:local .

# Run the full containerized test suite
pytest e2e/docker -v

# Run specifically the full Python SDK client suite against the container
pytest e2e/docker/test_container_python_client.py -v

# Optionally target a specific custom local image tag
SNOOZE_IMAGE=snooze:my-tag pytest e2e/docker -v
```

---

## Test Coverage Matrix

Every stream test runs against real fitted Scikit-Learn ONNX models (zero mock models) and exercises both happy paths and error/Dead-Letter Queue (DLQ) paths:

| Test Scenario | Redis Streams | RabbitMQ | Kafka | Mock Stream | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Single Inference Happy Path** | PASS | PASS | PASS | PASS | Validates output tensor shapes, probability sums ($\approx 1.0$), class argmax |
| **Concurrent Micro-Batching** | PASS | PASS | PASS | PASS | Validates Axis(0) batching, correlation IDs, and response fan-out |
| **Custom Dynamic `reply_to`** | PASS | PASS | PASS | PASS | Validates routing responses to per-request override topics/queues |
| **Malformed JSON Payload (DLQ)** | PASS | PASS | PASS | PASS | Poison-pill routing to DLQ with descriptive error diagnostics |
| **Shape/Dimension Mismatch (DLQ)** | PASS | PASS | PASS | PASS | Element count errors route to DLQ without crashing the worker task |
| **Unregistered Model ID (DLQ)** | PASS | PASS | PASS | PASS | Requests for unknown models route to DLQ with NotFound diagnostics |

---

## Detailed Findings & Diagnostic Report

### 1. Redis Streams (`src/stream/redis_stream.rs`) — Status: PASSING (6/6)
- The Redis Streams driver functions with `redis:7-alpine`.
- Consumer groups created with `$ MKSTREAM` successfully capture incoming inference requests via `XREADGROUP ... >`.
- Dynamic micro-batching collates multiple requests into single forward passes and publishes to output keys via `XADD`.
- Dead-Letter Queue (DLQ) error routing cleanly isolates poison pills and malformed JSON payloads.

### 2. RabbitMQ AMQP (`src/stream/rabbitmq.rs`) — Status: PASSING (6/6)
- The RabbitMQ driver functions with `rabbitmq:3-management-alpine`.
- Queue auto-creation declares `input_topic`, `output_topic`, and `error_topic` at supervisor startup.
- In AMQP 0-9-1, routing to exchange `""` with routing key `reply_to` requires destination queues to be bound/declared prior to consumption. With pre-declared custom reply queues, per-request dynamic routing operates as intended.

### 3. Mock Stream (`src/stream/mock_stream.rs`) — Status: PASSING (3/3)
- Tested via Python SDK `MockStreamClient` against real live `snooze serve` instances over HTTP/gRPC.
- Successfully validates client-side and server-side envelope framing, correlation ID matching, and error handling.

### 4. Apache Kafka (`src/stream/kafka.rs`) — Status: PASSING (6/6)
- The Apache Kafka driver functions with `confluentinc/cp-kafka:7.6.0` via `rskafka`.
- `KafkaStreamDriver` decouples `fetch_offset` from `committed_offset` and buffers multi-record partition batch fetches, ensuring each message is ingested once while maintaining commit-after-infer delivery guarantees.
- Supports concurrent micro-batching, dynamic `reply_to` topic overrides, and Dead-Letter Queue (DLQ) routing for malformed payloads, shape mismatches, and unknown models.
