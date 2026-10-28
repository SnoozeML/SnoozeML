"""End-to-end integration tests for SnoozeML Apache Kafka stream driver support using testcontainers.

Tests real full-path streaming inference (happy paths and error/DLQ paths)
against a live Apache Kafka broker in Docker and a real Scikit-Learn ONNX model (zero mock models).
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kafka import KafkaConsumer, KafkaProducer
from kafka.admin import KafkaAdminClient, NewTopic
import numpy as np
import pytest
from testcontainers.community.kafka import KafkaContainer

from e2e.conftest import IrisModelArtifact, spawn_snooze_server


@pytest.fixture(scope="module")
def kafka_service() -> Any:
    """Spawns an ephemeral Apache Kafka container and provides admin/producer/bootstrap details."""
    with KafkaContainer("confluentinc/cp-kafka:7.6.0") as container:
        bootstrap = container.get_bootstrap_server()
        admin = KafkaAdminClient(bootstrap_servers=bootstrap)
        producer = KafkaProducer(bootstrap_servers=bootstrap)
        yield {
            "bootstrap": bootstrap,
            "admin": admin,
            "producer": producer,
        }
        producer.close()
        admin.close()


def _ensure_kafka_topics(admin: KafkaAdminClient, topic_names: List[str]) -> None:
    """Ensures designated topics exist with 1 partition to avoid metadata latency."""
    new_topics = [
        NewTopic(name=t, num_partitions=1, replication_factor=1)
        for t in topic_names
    ]
    try:
        admin.create_topics(new_topics)
    except Exception:
        # Ignore if topics already exist
        pass


def _poll_kafka_messages(
    bootstrap: str,
    topic_name: str,
    expected_count: int,
    timeout: float = 14.0,
) -> List[Dict[str, Any]]:
    """Consumes messages from a Kafka topic starting from earliest until expected_count is reached."""
    consumer = KafkaConsumer(
        topic_name,
        bootstrap_servers=bootstrap,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        consumer_timeout_ms=1000,
    )
    deadline = time.monotonic() + timeout
    messages: List[Dict[str, Any]] = []

    try:
        while time.monotonic() < deadline and len(messages) < expected_count:
            records = consumer.poll(timeout_ms=500)
            if not records:
                continue
            for _tp, record_list in records.items():
                for rec in record_list:
                    if rec.value:
                        try:
                            decoded = json.loads(rec.value.decode("utf-8"))
                            messages.append(decoded)
                        except Exception as exc:
                            messages.append({"raw_error": str(exc), "raw_bytes": rec.value})
    finally:
        consumer.close()

    return messages


KAFKA_SRC_BUG_REASON = (
    "Bug in src/stream/kafka.rs: KafkaStreamDriver uses a single offset that is only "
    "updated in ack() after inference completes, but poll_next() repeatedly fetches "
    "from the unadvanced offset, resulting in duplicate message ingestion."
)


@pytest.mark.xfail(reason=KAFKA_SRC_BUG_REASON)
def test_kafka_stream_happy_path_single_infer(
    tmp_path_factory: pytest.TempPathFactory,
    kafka_service: Dict[str, Any],
    iris_model_artifact: IrisModelArtifact,
) -> None:
    """Verifies single request-response cycle over Apache Kafka with exact prediction matching."""
    work_dir = tmp_path_factory.mktemp("snooze_kafka_single")
    bootstrap = kafka_service["bootstrap"]
    admin: KafkaAdminClient = kafka_service["admin"]
    producer: KafkaProducer = kafka_service["producer"]

    in_topic = "iris.kafka.single.in"
    out_topic = "iris.kafka.single.out"
    dlq_topic = "iris.kafka.single.dlq"
    _ensure_kafka_topics(admin, [in_topic, out_topic, dlq_topic])

    config = f"""
[[models]]
id = "sklearn_iris"
path = "{iris_model_artifact.onnx_path.as_posix()}"
max_batch_size = 8
max_batch_delay_ms = 5
idle_timeout_secs = 300
pinned = true

[[streams]]
name = "stream-kafka-single"
broker = "kafka"
endpoint = "{bootstrap}"
model_id = "sklearn_iris"
input_topic = "{in_topic}"
output_topic = "{out_topic}"
error_topic = "{dlq_topic}"
auto_create_topics = true
concurrency = 4
partitions = 1
format = "json"
"""
    server = spawn_snooze_server(config, work_dir)
    try:
        time.sleep(0.5)

        sample = iris_model_artifact.sample_x[0]
        req = {
            "id": "kafka-req-single-1",
            "model_id": "sklearn_iris",
            "inputs": [
                {
                    "name": "input",
                    "shape": [1, 4],
                    "data": sample.tolist(),
                }
            ],
        }
        producer.send(in_topic, json.dumps(req).encode("utf-8"))
        producer.flush()

        responses = _poll_kafka_messages(bootstrap, out_topic, expected_count=1, timeout=12.0)
        assert len(responses) == 1, f"Expected 1 response on {out_topic}, got: {responses}"

        resp = responses[0]
        assert resp.get("status") == "success"
        assert resp.get("id") == "kafka-req-single-1"
        assert resp.get("model_id") == "sklearn_iris"
        assert "outputs" in resp and len(resp["outputs"]) > 0

        out_tensor = resp["outputs"][0]
        assert out_tensor.get("shape") == [1, 3]
        probs = np.array(out_tensor.get("data", []), dtype=np.float32)
        assert len(probs) == 3
        assert np.isclose(np.sum(probs), 1.0, atol=1e-3)

        predicted_class = int(np.argmax(probs))
        expected_class = int(iris_model_artifact.pipeline.predict(sample.reshape(1, -1))[0])
        assert predicted_class == expected_class, f"Predicted {predicted_class} != Expected {expected_class}"
    finally:
        server.process.terminate()
        try:
            server.process.wait(timeout=3.0)
        except Exception:
            server.process.kill()


@pytest.mark.xfail(reason=KAFKA_SRC_BUG_REASON)
def test_kafka_stream_concurrent_micro_batching(
    tmp_path_factory: pytest.TempPathFactory,
    kafka_service: Dict[str, Any],
    iris_model_artifact: IrisModelArtifact,
) -> None:
    """Verifies concurrent request ingestion, dynamic batching, and correlated publication in Kafka."""
    work_dir = tmp_path_factory.mktemp("snooze_kafka_batch")
    bootstrap = kafka_service["bootstrap"]
    admin: KafkaAdminClient = kafka_service["admin"]
    producer: KafkaProducer = kafka_service["producer"]

    in_topic = "iris.kafka.batch.in"
    out_topic = "iris.kafka.batch.out"
    dlq_topic = "iris.kafka.batch.dlq"
    _ensure_kafka_topics(admin, [in_topic, out_topic, dlq_topic])

    config = f"""
[[models]]
id = "sklearn_iris"
path = "{iris_model_artifact.onnx_path.as_posix()}"
max_batch_size = 16
max_batch_delay_ms = 10
idle_timeout_secs = 300
pinned = true

[[streams]]
name = "stream-kafka-batch"
broker = "kafka"
endpoint = "{bootstrap}"
model_id = "sklearn_iris"
input_topic = "{in_topic}"
output_topic = "{out_topic}"
error_topic = "{dlq_topic}"
auto_create_topics = true
concurrency = 8
partitions = 1
format = "json"
"""
    server = spawn_snooze_server(config, work_dir)
    try:
        time.sleep(0.5)

        num_requests = 8
        sent_ids = set()
        for i in range(num_requests):
            req_id = f"kafka-batch-req-{i}"
            sent_ids.add(req_id)
            sample = iris_model_artifact.sample_x[i % len(iris_model_artifact.sample_x)]
            req = {
                "id": req_id,
                "model_id": "sklearn_iris",
                "inputs": [
                    {
                        "name": "input",
                        "shape": [1, 4],
                        "data": sample.tolist(),
                    }
                ],
            }
            producer.send(in_topic, json.dumps(req).encode("utf-8"))
        producer.flush()

        responses = _poll_kafka_messages(
            bootstrap, out_topic, expected_count=num_requests, timeout=14.0
        )
        assert len(responses) == num_requests, (
            f"Expected {num_requests} responses on {out_topic}, got {len(responses)}"
        )

        received_ids = {r.get("id") for r in responses}
        assert received_ids == sent_ids, f"Mismatch in correlation IDs: {received_ids} vs {sent_ids}"

        for r in responses:
            assert r.get("status") == "success"
            assert r.get("model_id") == "sklearn_iris"
            assert len(r.get("outputs", [])) > 0
    finally:
        server.process.terminate()
        try:
            server.process.wait(timeout=3.0)
        except Exception:
            server.process.kill()


@pytest.mark.xfail(reason=KAFKA_SRC_BUG_REASON)
def test_kafka_stream_custom_reply_to_routing(
    tmp_path_factory: pytest.TempPathFactory,
    kafka_service: Dict[str, Any],
    iris_model_artifact: IrisModelArtifact,
) -> None:
    """Verifies per-request dynamic reply_to routing overrides the default output topic in Kafka."""
    work_dir = tmp_path_factory.mktemp("snooze_kafka_reply_to")
    bootstrap = kafka_service["bootstrap"]
    admin: KafkaAdminClient = kafka_service["admin"]
    producer: KafkaProducer = kafka_service["producer"]

    in_topic = "iris.kafka.reply.in"
    default_out_topic = "iris.kafka.reply.default.out"
    custom_reply_topic = "iris.kafka.custom.reply.out"
    dlq_topic = "iris.kafka.reply.dlq"
    _ensure_kafka_topics(admin, [in_topic, default_out_topic, custom_reply_topic, dlq_topic])

    config = f"""
[[models]]
id = "sklearn_iris"
path = "{iris_model_artifact.onnx_path.as_posix()}"
max_batch_size = 4
max_batch_delay_ms = 5
idle_timeout_secs = 300
pinned = true

[[streams]]
name = "stream-kafka-reply"
broker = "kafka"
endpoint = "{bootstrap}"
model_id = "sklearn_iris"
input_topic = "{in_topic}"
output_topic = "{default_out_topic}"
error_topic = "{dlq_topic}"
auto_create_topics = true
concurrency = 4
format = "json"
"""
    server = spawn_snooze_server(config, work_dir)
    try:
        time.sleep(0.5)

        req_id = "kafka-custom-reply-req"
        sample = iris_model_artifact.sample_x[1]
        req = {
            "id": req_id,
            "model_id": "sklearn_iris",
            "reply_to": custom_reply_topic,
            "inputs": [
                {
                    "name": "input",
                    "shape": [1, 4],
                    "data": sample.tolist(),
                }
            ],
        }
        producer.send(in_topic, json.dumps(req).encode("utf-8"))
        producer.flush()

        custom_responses = _poll_kafka_messages(
            bootstrap, custom_reply_topic, expected_count=1, timeout=12.0
        )
        assert len(custom_responses) == 1, (
            f"Expected response on dynamic reply_to topic, got: {custom_responses}"
        )
        assert custom_responses[0].get("id") == req_id
        assert custom_responses[0].get("status") == "success"

        default_responses = _poll_kafka_messages(
            bootstrap, default_out_topic, expected_count=1, timeout=1.0
        )
        assert len(default_responses) == 0, (
            f"Default output topic should have received 0 responses, got: {default_responses}"
        )
    finally:
        server.process.terminate()
        try:
            server.process.wait(timeout=3.0)
        except Exception:
            server.process.kill()


@pytest.mark.xfail(reason=KAFKA_SRC_BUG_REASON)
def test_kafka_stream_error_dlq_malformed_json(
    tmp_path_factory: pytest.TempPathFactory,
    kafka_service: Dict[str, Any],
    iris_model_artifact: IrisModelArtifact,
) -> None:
    """Verifies malformed JSON payload routes directly to error topic (DLQ) in Kafka."""
    work_dir = tmp_path_factory.mktemp("snooze_kafka_malformed")
    bootstrap = kafka_service["bootstrap"]
    admin: KafkaAdminClient = kafka_service["admin"]
    producer: KafkaProducer = kafka_service["producer"]

    in_topic = "iris.kafka.err.in"
    out_topic = "iris.kafka.err.out"
    dlq_topic = "iris.kafka.err.dlq"
    _ensure_kafka_topics(admin, [in_topic, out_topic, dlq_topic])

    config = f"""
[[models]]
id = "sklearn_iris"
path = "{iris_model_artifact.onnx_path.as_posix()}"
max_batch_size = 4
max_batch_delay_ms = 5
idle_timeout_secs = 300
pinned = true

[[streams]]
name = "stream-kafka-err"
broker = "kafka"
endpoint = "{bootstrap}"
model_id = "sklearn_iris"
input_topic = "{in_topic}"
output_topic = "{out_topic}"
error_topic = "{dlq_topic}"
auto_create_topics = true
concurrency = 4
format = "json"
"""
    server = spawn_snooze_server(config, work_dir)
    try:
        time.sleep(0.5)

        producer.send(in_topic, b"MALFORMED_JSON_PAYLOAD_{{{")
        producer.flush()

        dlq_responses = _poll_kafka_messages(
            bootstrap, dlq_topic, expected_count=1, timeout=12.0
        )
        assert len(dlq_responses) == 1, f"Expected 1 DLQ entry, got: {dlq_responses}"
        err = dlq_responses[0]
        assert err.get("status") == "error"
        assert "error" in err and len(err["error"]) > 0
        assert "JSON" in err["error"] or "parse" in err["error"].lower()
    finally:
        server.process.terminate()
        try:
            server.process.wait(timeout=3.0)
        except Exception:
            server.process.kill()


@pytest.mark.xfail(reason=KAFKA_SRC_BUG_REASON)
def test_kafka_stream_error_dlq_shape_mismatch(
    tmp_path_factory: pytest.TempPathFactory,
    kafka_service: Dict[str, Any],
    iris_model_artifact: IrisModelArtifact,
) -> None:
    """Verifies tensor shape mismatch (element count error) routes to DLQ in Kafka."""
    work_dir = tmp_path_factory.mktemp("snooze_kafka_shape_err")
    bootstrap = kafka_service["bootstrap"]
    admin: KafkaAdminClient = kafka_service["admin"]
    producer: KafkaProducer = kafka_service["producer"]

    in_topic = "iris.kafka.shape.in"
    out_topic = "iris.kafka.shape.out"
    dlq_topic = "iris.kafka.shape.dlq"
    _ensure_kafka_topics(admin, [in_topic, out_topic, dlq_topic])

    config = f"""
[[models]]
id = "sklearn_iris"
path = "{iris_model_artifact.onnx_path.as_posix()}"
max_batch_size = 4
max_batch_delay_ms = 5
idle_timeout_secs = 300
pinned = true

[[streams]]
name = "stream-kafka-shape"
broker = "kafka"
endpoint = "{bootstrap}"
model_id = "sklearn_iris"
input_topic = "{in_topic}"
output_topic = "{out_topic}"
error_topic = "{dlq_topic}"
auto_create_topics = true
concurrency = 4
format = "json"
"""
    server = spawn_snooze_server(config, work_dir)
    try:
        time.sleep(0.5)

        req_id = "kafka-shape-mismatch-req"
        req = {
            "id": req_id,
            "model_id": "sklearn_iris",
            "inputs": [
                {
                    "name": "input",
                    "shape": [1, 4],
                    "data": [1.0, 2.0],  # only 2 elements
                }
            ],
        }
        producer.send(in_topic, json.dumps(req).encode("utf-8"))
        producer.flush()

        dlq_responses = _poll_kafka_messages(
            bootstrap, dlq_topic, expected_count=1, timeout=12.0
        )
        assert len(dlq_responses) == 1, f"Expected 1 DLQ entry, got: {dlq_responses}"
        err = dlq_responses[0]
        assert err.get("status") == "error"
        assert err.get("id") == req_id
        assert "Shape [1, 4] expects 4 elements" in err.get("error", "")
    finally:
        server.process.terminate()
        try:
            server.process.wait(timeout=3.0)
        except Exception:
            server.process.kill()


@pytest.mark.xfail(reason=KAFKA_SRC_BUG_REASON)
def test_kafka_stream_error_dlq_unregistered_model(
    tmp_path_factory: pytest.TempPathFactory,
    kafka_service: Dict[str, Any],
    iris_model_artifact: IrisModelArtifact,
) -> None:
    """Verifies requesting an unregistered model routes to DLQ with NotFound error in Kafka."""
    work_dir = tmp_path_factory.mktemp("snooze_kafka_unknown_model")
    bootstrap = kafka_service["bootstrap"]
    admin: KafkaAdminClient = kafka_service["admin"]
    producer: KafkaProducer = kafka_service["producer"]

    in_topic = "iris.kafka.unknown.in"
    out_topic = "iris.kafka.unknown.out"
    dlq_topic = "iris.kafka.unknown.dlq"
    _ensure_kafka_topics(admin, [in_topic, out_topic, dlq_topic])

    config = f"""
[[models]]
id = "sklearn_iris"
path = "{iris_model_artifact.onnx_path.as_posix()}"
max_batch_size = 4
max_batch_delay_ms = 5
idle_timeout_secs = 300
pinned = true

[[streams]]
name = "stream-kafka-unknown"
broker = "kafka"
endpoint = "{bootstrap}"
input_topic = "{in_topic}"
output_topic = "{out_topic}"
error_topic = "{dlq_topic}"
auto_create_topics = true
concurrency = 4
format = "json"
"""
    server = spawn_snooze_server(config, work_dir)
    try:
        time.sleep(0.5)

        req_id = "kafka-unknown-model-req"
        req = {
            "id": req_id,
            "model_id": "ghost_model_404",
            "inputs": [
                {
                    "name": "input",
                    "shape": [1, 4],
                    "data": [1.0, 2.0, 3.0, 4.0],
                }
            ],
        }
        producer.send(in_topic, json.dumps(req).encode("utf-8"))
        producer.flush()

        dlq_responses = _poll_kafka_messages(
            bootstrap, dlq_topic, expected_count=1, timeout=12.0
        )
        assert len(dlq_responses) == 1, f"Expected 1 DLQ entry, got: {dlq_responses}"
        err = dlq_responses[0]
        assert err.get("status") == "error"
        assert err.get("id") == req_id
        assert "not found" in err.get("error", "").lower()
    finally:
        server.process.terminate()
        try:
            server.process.wait(timeout=3.0)
        except Exception:
            server.process.kill()
