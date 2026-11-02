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
│   ├── docker-compose.test.yml     # Compose service specification for testing the SnoozeML container
│   └── test_containerized_daemon.py # Full endpoint & real model validation against OCI container
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
# Run the full containerized daemon test suite against real models
pytest e2e/docker -v

# Target a specific SnoozeML container image flavor (e.g. CUDA or local build)
SNOOZE_IMAGE=ghcr.io/snoozeml/snoozeml:cuda pytest e2e/docker -v
```

---

## Test Coverage Matrix

Every stream test runs against real fitted Scikit-Learn ONNX models (zero mock models) and exercises both happy paths and error/Dead-Letter Queue (DLQ) paths:

| Test Scenario | Redis Streams | RabbitMQ | Kafka | Mock Stream | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Single Inference Happy Path** | PASS | PASS | XFAIL* | PASS | Validates output tensor shapes, probability sums ($\approx 1.0$), class argmax |
| **Concurrent Micro-Batching** | PASS | PASS | XFAIL* | PASS | Validates Axis(0) batching, correlation IDs, and response fan-out |
| **Custom Dynamic `reply_to`** | PASS | PASS | XFAIL* | PASS | Validates routing responses to per-request override topics/queues |
| **Malformed JSON Payload (DLQ)** | PASS | PASS | XFAIL* | PASS | Poison-pill routing to DLQ with descriptive error diagnostics |
| **Shape/Dimension Mismatch (DLQ)** | PASS | PASS | XFAIL* | PASS | Element count errors route to DLQ without crashing the worker task |
| **Unregistered Model ID (DLQ)** | PASS | PASS | XFAIL* | PASS | Requests for unknown models route to DLQ with NotFound diagnostics |

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

### 4. Apache Kafka (`src/stream/kafka.rs`) — Status: XFAIL (6/6) — Identified Issue in `src/`
- **Symptom:**
  When a single request is published to an input topic, the daemon publishes **multiple duplicate responses** (e.g. 5 identical responses) to the output topic or DLQ.
- **Root Cause Analysis in `src/stream/kafka.rs`:**
  1. `KafkaStreamDriver` maintains a single `offset: Arc<AtomicI64>`.
  2. In `poll_next()`, records are fetched using `pc.fetch_records(current_offset, ...)` where `current_offset = self.offset.load(Ordering::Relaxed)`.
  3. However, `self.offset` is **only updated inside `KafkaAcknowledger::ack()`**, which executes in an asynchronous background Tokio task **after** model inference completes.
  4. While that inference task is executing in Tokio, the worker loop in `src/stream/mod.rs` continues polling `driver.poll_next()`. Because `self.offset` has not yet been advanced, `poll_next()` repeatedly fetches the exact same record from partition 0, spawning redundant duplicate inference tasks.
- **Recommended Fix (for subsequent commit):**
  In `KafkaStreamDriver`, decouple the fetch offset from the commit offset. Introduce an atomic `fetch_offset` that is advanced immediately when records are read from the partition client during `poll_next()`, while retaining `ack_offset` for partition offset commits.
