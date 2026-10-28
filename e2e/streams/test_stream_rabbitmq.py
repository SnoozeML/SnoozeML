"""End-to-end integration tests for SnoozeML RabbitMQ AMQP broker support using testcontainers.

Tests real full-path streaming inference (happy paths and error/DLQ paths)
against a live RabbitMQ broker in Docker and a real Scikit-Learn ONNX model (zero mock models).
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

import numpy as np
import pika
import pytest
from testcontainers.community.rabbitmq import RabbitMqContainer

from e2e.conftest import IrisModelArtifact, spawn_snooze_server


@pytest.fixture(scope="module")
def rabbitmq_service() -> Any:
    """Spawns an ephemeral RabbitMQ container and provides AMQP connection parameters."""
    with RabbitMqContainer("rabbitmq:3-management-alpine") as container:
        host = container.get_container_host_ip()
        port = int(container.get_exposed_port(5672))
        endpoint = f"amqp://guest:guest@{host}:{port}/%2f"
        params = pika.ConnectionParameters(
            host=host,
            port=port,
            credentials=pika.PlainCredentials("guest", "guest"),
            heartbeat=30,
        )
        yield {
            "host": host,
            "port": port,
            "endpoint": endpoint,
            "params": params,
        }


def _publish_rmq_message(
    params: pika.ConnectionParameters,
    queue_name: str,
    payload_bytes: bytes,
    properties: pika.BasicProperties | None = None,
) -> None:
    """Publishes a raw byte message to a RabbitMQ queue via the default exchange."""
    conn = pika.BlockingConnection(params)
    try:
        ch = conn.channel()
        ch.queue_declare(queue=queue_name, durable=True)
        ch.basic_publish(
            exchange="",
            routing_key=queue_name,
            body=payload_bytes,
            properties=properties or pika.BasicProperties(delivery_mode=2),
        )
    finally:
        conn.close()


def _poll_rmq_messages(
    params: pika.ConnectionParameters,
    queue_name: str,
    expected_count: int,
    timeout: float = 12.0,
) -> List[Dict[str, Any]]:
    """Polls messages from a RabbitMQ queue until expected_count messages are gathered."""
    deadline = time.monotonic() + timeout
    messages: List[Dict[str, Any]] = []

    conn = pika.BlockingConnection(params)
    try:
        ch = conn.channel()
        ch.queue_declare(queue=queue_name, durable=True)

        while time.monotonic() < deadline and len(messages) < expected_count:
            method, _props, body = ch.basic_get(queue=queue_name, auto_ack=True)
            if method:
                try:
                    decoded = json.loads(body.decode("utf-8"))
                    messages.append(decoded)
                except Exception as exc:
                    messages.append({"raw_error": str(exc), "raw_bytes": body})
            else:
                time.sleep(0.1)
    finally:
        conn.close()

    return messages


def test_rabbitmq_stream_happy_path_single_infer(
    tmp_path_factory: pytest.TempPathFactory,
    rabbitmq_service: Dict[str, Any],
    iris_model_artifact: IrisModelArtifact,
) -> None:
    """Verifies single request-response cycle over RabbitMQ AMQP with exact prediction matching."""
    work_dir = tmp_path_factory.mktemp("snooze_rmq_single")
    endpoint = rabbitmq_service["endpoint"]
    params = rabbitmq_service["params"]

    config = f"""
[[models]]
id = "sklearn_iris"
path = "{iris_model_artifact.onnx_path.as_posix()}"
max_batch_size = 8
max_batch_delay_ms = 5
idle_timeout_secs = 300
pinned = true

[[streams]]
name = "stream-rmq-single"
broker = "rabbitmq"
endpoint = "{endpoint}"
model_id = "sklearn_iris"
input_topic = "iris.rmq.single.in"
output_topic = "iris.rmq.single.out"
error_topic = "iris.rmq.single.dlq"
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
            "id": "rmq-req-single-1",
            "model_id": "sklearn_iris",
            "inputs": [
                {
                    "name": "input",
                    "shape": [1, 4],
                    "data": sample.tolist(),
                }
            ],
        }
        _publish_rmq_message(
            params, "iris.rmq.single.in", json.dumps(req).encode("utf-8")
        )

        responses = _poll_rmq_messages(
            params, "iris.rmq.single.out", expected_count=1, timeout=12.0
        )
        assert len(responses) == 1, f"Expected 1 response on iris.rmq.single.out, got: {responses}"

        resp = responses[0]
        assert resp.get("status") == "success"
        assert resp.get("id") == "rmq-req-single-1"
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


def test_rabbitmq_stream_concurrent_micro_batching(
    tmp_path_factory: pytest.TempPathFactory,
    rabbitmq_service: Dict[str, Any],
    iris_model_artifact: IrisModelArtifact,
) -> None:
    """Verifies concurrent request ingestion, dynamic batching, and correlated publication in RabbitMQ."""
    work_dir = tmp_path_factory.mktemp("snooze_rmq_batch")
    endpoint = rabbitmq_service["endpoint"]
    params = rabbitmq_service["params"]

    config = f"""
[[models]]
id = "sklearn_iris"
path = "{iris_model_artifact.onnx_path.as_posix()}"
max_batch_size = 16
max_batch_delay_ms = 10
idle_timeout_secs = 300
pinned = true

[[streams]]
name = "stream-rmq-batch"
broker = "rabbitmq"
endpoint = "{endpoint}"
model_id = "sklearn_iris"
input_topic = "iris.rmq.batch.in"
output_topic = "iris.rmq.batch.out"
error_topic = "iris.rmq.batch.dlq"
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
            req_id = f"rmq-batch-req-{i}"
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
            _publish_rmq_message(params, "iris.rmq.batch.in", json.dumps(req).encode("utf-8"))

        responses = _poll_rmq_messages(
            params, "iris.rmq.batch.out", expected_count=num_requests, timeout=15.0
        )
        assert len(responses) == num_requests, (
            f"Expected {num_requests} responses on iris.rmq.batch.out, got {len(responses)}"
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


def test_rabbitmq_stream_custom_reply_to_routing(
    tmp_path_factory: pytest.TempPathFactory,
    rabbitmq_service: Dict[str, Any],
    iris_model_artifact: IrisModelArtifact,
) -> None:
    """Verifies per-request dynamic reply_to routing overrides the default output queue in RabbitMQ."""
    work_dir = tmp_path_factory.mktemp("snooze_rmq_reply_to")
    endpoint = rabbitmq_service["endpoint"]
    params = rabbitmq_service["params"]

    config = f"""
[[models]]
id = "sklearn_iris"
path = "{iris_model_artifact.onnx_path.as_posix()}"
max_batch_size = 4
max_batch_delay_ms = 5
idle_timeout_secs = 300
pinned = true

[[streams]]
name = "stream-rmq-reply"
broker = "rabbitmq"
endpoint = "{endpoint}"
model_id = "sklearn_iris"
input_topic = "iris.rmq.reply.in"
output_topic = "iris.rmq.reply.default.out"
error_topic = "iris.rmq.reply.dlq"
auto_create_topics = true
concurrency = 4
format = "json"
"""
    server = spawn_snooze_server(config, work_dir)
    try:
        time.sleep(0.5)

        # Pre-declare the custom reply queue on the broker so RabbitMQ default exchange
        # has a bound destination queue when SnoozeML routes the message.
        conn = pika.BlockingConnection(params)
        ch = conn.channel()
        ch.queue_declare(queue="iris.rmq.custom.reply.out", durable=True)
        conn.close()

        req_id = "rmq-custom-reply-req"
        sample = iris_model_artifact.sample_x[2]
        req = {
            "id": req_id,
            "model_id": "sklearn_iris",
            "reply_to": "iris.rmq.custom.reply.out",
            "inputs": [
                {
                    "name": "input",
                    "shape": [1, 4],
                    "data": sample.tolist(),
                }
            ],
        }
        _publish_rmq_message(params, "iris.rmq.reply.in", json.dumps(req).encode("utf-8"))

        custom_responses = _poll_rmq_messages(
            params, "iris.rmq.custom.reply.out", expected_count=1, timeout=12.0
        )
        assert len(custom_responses) == 1, (
            f"Expected response on dynamic reply_to queue, got: {custom_responses}"
        )
        assert custom_responses[0].get("id") == req_id
        assert custom_responses[0].get("status") == "success"

        default_responses = _poll_rmq_messages(
            params, "iris.rmq.reply.default.out", expected_count=1, timeout=1.0
        )
        assert len(default_responses) == 0, (
            f"Default output queue should have received 0 responses, got: {default_responses}"
        )
    finally:
        server.process.terminate()
        try:
            server.process.wait(timeout=3.0)
        except Exception:
            server.process.kill()


def test_rabbitmq_stream_error_dlq_malformed_json(
    tmp_path_factory: pytest.TempPathFactory,
    rabbitmq_service: Dict[str, Any],
    iris_model_artifact: IrisModelArtifact,
) -> None:
    """Verifies malformed JSON payload routes directly to error queue (DLQ) in RabbitMQ."""
    work_dir = tmp_path_factory.mktemp("snooze_rmq_malformed")
    endpoint = rabbitmq_service["endpoint"]
    params = rabbitmq_service["params"]

    config = f"""
[[models]]
id = "sklearn_iris"
path = "{iris_model_artifact.onnx_path.as_posix()}"
max_batch_size = 4
max_batch_delay_ms = 5
idle_timeout_secs = 300
pinned = true

[[streams]]
name = "stream-rmq-err"
broker = "rabbitmq"
endpoint = "{endpoint}"
model_id = "sklearn_iris"
input_topic = "iris.rmq.err.in"
output_topic = "iris.rmq.err.out"
error_topic = "iris.rmq.err.dlq"
auto_create_topics = true
concurrency = 4
format = "json"
"""
    server = spawn_snooze_server(config, work_dir)
    try:
        time.sleep(0.5)

        _publish_rmq_message(params, "iris.rmq.err.in", b"MALFORMED_JSON_PAYLOAD_{{{")

        dlq_responses = _poll_rmq_messages(
            params, "iris.rmq.err.dlq", expected_count=1, timeout=12.0
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


def test_rabbitmq_stream_error_dlq_shape_mismatch(
    tmp_path_factory: pytest.TempPathFactory,
    rabbitmq_service: Dict[str, Any],
    iris_model_artifact: IrisModelArtifact,
) -> None:
    """Verifies tensor shape mismatch (element count error) routes to DLQ in RabbitMQ."""
    work_dir = tmp_path_factory.mktemp("snooze_rmq_shape_err")
    endpoint = rabbitmq_service["endpoint"]
    params = rabbitmq_service["params"]

    config = f"""
[[models]]
id = "sklearn_iris"
path = "{iris_model_artifact.onnx_path.as_posix()}"
max_batch_size = 4
max_batch_delay_ms = 5
idle_timeout_secs = 300
pinned = true

[[streams]]
name = "stream-rmq-shape"
broker = "rabbitmq"
endpoint = "{endpoint}"
model_id = "sklearn_iris"
input_topic = "iris.rmq.shape.in"
output_topic = "iris.rmq.shape.out"
error_topic = "iris.rmq.shape.dlq"
auto_create_topics = true
concurrency = 4
format = "json"
"""
    server = spawn_snooze_server(config, work_dir)
    try:
        time.sleep(0.5)

        req_id = "rmq-shape-mismatch-req"
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
        _publish_rmq_message(params, "iris.rmq.shape.in", json.dumps(req).encode("utf-8"))

        dlq_responses = _poll_rmq_messages(
            params, "iris.rmq.shape.dlq", expected_count=1, timeout=12.0
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


def test_rabbitmq_stream_error_dlq_unregistered_model(
    tmp_path_factory: pytest.TempPathFactory,
    rabbitmq_service: Dict[str, Any],
    iris_model_artifact: IrisModelArtifact,
) -> None:
    """Verifies requesting an unregistered model routes to DLQ with NotFound error in RabbitMQ."""
    work_dir = tmp_path_factory.mktemp("snooze_rmq_unknown_model")
    endpoint = rabbitmq_service["endpoint"]
    params = rabbitmq_service["params"]

    config = f"""
[[models]]
id = "sklearn_iris"
path = "{iris_model_artifact.onnx_path.as_posix()}"
max_batch_size = 4
max_batch_delay_ms = 5
idle_timeout_secs = 300
pinned = true

[[streams]]
name = "stream-rmq-unknown"
broker = "rabbitmq"
endpoint = "{endpoint}"
input_topic = "iris.rmq.unknown.in"
output_topic = "iris.rmq.unknown.out"
error_topic = "iris.rmq.unknown.dlq"
auto_create_topics = true
concurrency = 4
format = "json"
"""
    server = spawn_snooze_server(config, work_dir)
    try:
        time.sleep(0.5)

        req_id = "rmq-unknown-model-req"
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
        _publish_rmq_message(params, "iris.rmq.unknown.in", json.dumps(req).encode("utf-8"))

        dlq_responses = _poll_rmq_messages(
            params, "iris.rmq.unknown.dlq", expected_count=1, timeout=12.0
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
