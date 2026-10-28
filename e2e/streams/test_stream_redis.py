"""End-to-end integration tests for SnoozeML Redis Streams broker support using testcontainers.

Tests real full-path streaming inference (happy paths and error/DLQ paths)
against a live Redis instance in Docker and a real Scikit-Learn ONNX model (zero mock models).
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
import pytest
import redis
from testcontainers.community.redis import RedisContainer

from e2e.conftest import IrisModelArtifact, spawn_snooze_server
from snoozeml.streams import StreamInferRequest, StreamInferResponse


@pytest.fixture(scope="module")
def redis_service() -> Any:
    """Spawns an ephemeral Redis 7 container."""
    with RedisContainer("redis:7-alpine") as container:
        host = container.get_container_host_ip()
        port = int(container.get_exposed_port(6379))
        client = redis.Redis(host=host, port=port, decode_responses=False)
        yield {
            "host": host,
            "port": port,
            "endpoint": f"redis://{host}:{port}",
            "client": client,
        }


def _poll_stream_entries(
    client: redis.Redis,
    stream_key: str,
    expected_count: int,
    timeout: float = 12.0,
) -> List[Dict[str, Any]]:
    """Polls entries from a Redis stream key until expected_count entries are gathered."""
    deadline = time.monotonic() + timeout
    entries_found: List[Dict[str, Any]] = []
    cursor = "0-0"

    while time.monotonic() < deadline and len(entries_found) < expected_count:
        remaining_ms = max(50, int((deadline - time.monotonic()) * 1000))
        block_ms = min(remaining_ms, 250)
        try:
            res = client.xread(streams={stream_key: cursor}, count=16, block=block_ms)
        except Exception:
            time.sleep(0.1)
            continue

        if not res:
            continue

        for _sname, msgs in res:
            for mid, fdict in msgs:
                cursor = mid.decode("utf-8") if isinstance(mid, bytes) else str(mid)
                raw_payload = fdict.get(b"payload") or fdict.get("payload") or fdict.get(b"data")
                if raw_payload is None and fdict:
                    raw_payload = next(iter(fdict.values()))
                if raw_payload:
                    try:
                        decoded = json.loads(raw_payload.decode("utf-8"))
                        entries_found.append(decoded)
                    except Exception as exc:
                        entries_found.append({"raw_error": str(exc), "raw_bytes": raw_payload})

    return entries_found


def test_redis_stream_happy_path_single_infer(
    tmp_path_factory: pytest.TempPathFactory,
    redis_service: Dict[str, Any],
    iris_model_artifact: IrisModelArtifact,
) -> None:
    """Verifies single request-response cycle over Redis Streams with exact prediction matching."""
    work_dir = tmp_path_factory.mktemp("snooze_redis_single")
    client: redis.Redis = redis_service["client"]
    endpoint = redis_service["endpoint"]

    config = f"""
[[models]]
id = "sklearn_iris"
path = "{iris_model_artifact.onnx_path.as_posix()}"
max_batch_size = 8
max_batch_delay_ms = 5
idle_timeout_secs = 300
pinned = true

[[streams]]
name = "stream-redis-single"
broker = "redis"
endpoint = "{endpoint}"
model_id = "sklearn_iris"
input_topic = "iris.single.in"
output_topic = "iris.single.out"
error_topic = "iris.single.dlq"
auto_create_topics = true
concurrency = 4
partitions = 1
format = "json"
"""
    server = spawn_snooze_server(config, work_dir)
    try:
        # Give stream supervisor a brief moment to initialize the consumer group
        time.sleep(0.3)

        sample = iris_model_artifact.sample_x[0]  # [5.1, 3.5, 1.4, 0.2]
        req = {
            "id": "redis-req-single-1",
            "model_id": "sklearn_iris",
            "inputs": [
                {
                    "name": "input",
                    "shape": [1, 4],
                    "data": sample.tolist(),
                }
            ],
        }
        client.xadd("iris.single.in", {"payload": json.dumps(req).encode("utf-8")})

        responses = _poll_stream_entries(client, "iris.single.out", expected_count=1, timeout=10.0)
        assert len(responses) == 1, f"Expected 1 response on iris.single.out, got: {responses}"

        resp = responses[0]
        assert resp.get("status") == "success"
        assert resp.get("id") == "redis-req-single-1"
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


def test_redis_stream_concurrent_micro_batching(
    tmp_path_factory: pytest.TempPathFactory,
    redis_service: Dict[str, Any],
    iris_model_artifact: IrisModelArtifact,
) -> None:
    """Verifies concurrent request ingestion, dynamic batching, and correlated output publication."""
    work_dir = tmp_path_factory.mktemp("snooze_redis_batch")
    client: redis.Redis = redis_service["client"]
    endpoint = redis_service["endpoint"]

    config = f"""
[[models]]
id = "sklearn_iris"
path = "{iris_model_artifact.onnx_path.as_posix()}"
max_batch_size = 16
max_batch_delay_ms = 10
idle_timeout_secs = 300
pinned = true

[[streams]]
name = "stream-redis-batch"
broker = "redis"
endpoint = "{endpoint}"
model_id = "sklearn_iris"
input_topic = "iris.batch.in"
output_topic = "iris.batch.out"
error_topic = "iris.batch.dlq"
auto_create_topics = true
concurrency = 8
partitions = 1
format = "json"
"""
    server = spawn_snooze_server(config, work_dir)
    try:
        time.sleep(0.3)

        num_requests = 8
        sent_ids = set()
        for i in range(num_requests):
            req_id = f"batch-req-{i}"
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
            client.xadd("iris.batch.in", {"payload": json.dumps(req).encode("utf-8")})

        responses = _poll_stream_entries(
            client, "iris.batch.out", expected_count=num_requests, timeout=12.0
        )
        assert len(responses) == num_requests, (
            f"Expected {num_requests} responses on iris.batch.out, got {len(responses)}"
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


def test_redis_stream_custom_reply_to_routing(
    tmp_path_factory: pytest.TempPathFactory,
    redis_service: Dict[str, Any],
    iris_model_artifact: IrisModelArtifact,
) -> None:
    """Verifies per-request dynamic reply_to routing overrides the default output_topic."""
    work_dir = tmp_path_factory.mktemp("snooze_redis_reply_to")
    client: redis.Redis = redis_service["client"]
    endpoint = redis_service["endpoint"]

    config = f"""
[[models]]
id = "sklearn_iris"
path = "{iris_model_artifact.onnx_path.as_posix()}"
max_batch_size = 4
max_batch_delay_ms = 5
idle_timeout_secs = 300
pinned = true

[[streams]]
name = "stream-redis-reply"
broker = "redis"
endpoint = "{endpoint}"
model_id = "sklearn_iris"
input_topic = "iris.reply.in"
output_topic = "iris.reply.default.out"
error_topic = "iris.reply.dlq"
auto_create_topics = true
concurrency = 4
format = "json"
"""
    server = spawn_snooze_server(config, work_dir)
    try:
        time.sleep(0.3)

        req_id = "custom-reply-req-42"
        sample = iris_model_artifact.sample_x[1]
        req = {
            "id": req_id,
            "model_id": "sklearn_iris",
            "reply_to": "iris.custom.reply.out",
            "inputs": [
                {
                    "name": "input",
                    "shape": [1, 4],
                    "data": sample.tolist(),
                }
            ],
        }
        client.xadd("iris.reply.in", {"payload": json.dumps(req).encode("utf-8")})

        # Must arrive on iris.custom.reply.out
        custom_responses = _poll_stream_entries(
            client, "iris.custom.reply.out", expected_count=1, timeout=10.0
        )
        assert len(custom_responses) == 1, (
            f"Expected response on dynamic reply_to stream, got: {custom_responses}"
        )
        assert custom_responses[0].get("id") == req_id
        assert custom_responses[0].get("status") == "success"

        # Default output stream must remain empty
        default_responses = _poll_stream_entries(
            client, "iris.reply.default.out", expected_count=1, timeout=1.0
        )
        assert len(default_responses) == 0, (
            f"Default output stream should have received 0 responses, got: {default_responses}"
        )
    finally:
        server.process.terminate()
        try:
            server.process.wait(timeout=3.0)
        except Exception:
            server.process.kill()


def test_redis_stream_error_dlq_malformed_json(
    tmp_path_factory: pytest.TempPathFactory,
    redis_service: Dict[str, Any],
    iris_model_artifact: IrisModelArtifact,
) -> None:
    """Verifies malformed JSON payload routes directly to error_topic (DLQ) with status='error'."""
    work_dir = tmp_path_factory.mktemp("snooze_redis_malformed")
    client: redis.Redis = redis_service["client"]
    endpoint = redis_service["endpoint"]

    config = f"""
[[models]]
id = "sklearn_iris"
path = "{iris_model_artifact.onnx_path.as_posix()}"
max_batch_size = 4
max_batch_delay_ms = 5
idle_timeout_secs = 300
pinned = true

[[streams]]
name = "stream-redis-err"
broker = "redis"
endpoint = "{endpoint}"
model_id = "sklearn_iris"
input_topic = "iris.err.in"
output_topic = "iris.err.out"
error_topic = "iris.err.dlq"
auto_create_topics = true
concurrency = 4
format = "json"
"""
    server = spawn_snooze_server(config, work_dir)
    try:
        time.sleep(0.3)

        client.xadd("iris.err.in", {"payload": b"MALFORMED_JSON_{{{{"})

        dlq_responses = _poll_stream_entries(
            client, "iris.err.dlq", expected_count=1, timeout=10.0
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


def test_redis_stream_error_dlq_shape_mismatch(
    tmp_path_factory: pytest.TempPathFactory,
    redis_service: Dict[str, Any],
    iris_model_artifact: IrisModelArtifact,
) -> None:
    """Verifies tensor shape mismatch (element count error) routes to error_topic."""
    work_dir = tmp_path_factory.mktemp("snooze_redis_shape_err")
    client: redis.Redis = redis_service["client"]
    endpoint = redis_service["endpoint"]

    config = f"""
[[models]]
id = "sklearn_iris"
path = "{iris_model_artifact.onnx_path.as_posix()}"
max_batch_size = 4
max_batch_delay_ms = 5
idle_timeout_secs = 300
pinned = true

[[streams]]
name = "stream-redis-shape"
broker = "redis"
endpoint = "{endpoint}"
model_id = "sklearn_iris"
input_topic = "iris.shape.in"
output_topic = "iris.shape.out"
error_topic = "iris.shape.dlq"
auto_create_topics = true
concurrency = 4
format = "json"
"""
    server = spawn_snooze_server(config, work_dir)
    try:
        time.sleep(0.3)

        req_id = "shape-mismatch-req"
        req = {
            "id": req_id,
            "model_id": "sklearn_iris",
            "inputs": [
                {
                    "name": "input",
                    "shape": [1, 4],
                    "data": [1.0, 2.0],  # only 2 elements instead of 4
                }
            ],
        }
        client.xadd("iris.shape.in", {"payload": json.dumps(req).encode("utf-8")})

        dlq_responses = _poll_stream_entries(
            client, "iris.shape.dlq", expected_count=1, timeout=10.0
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


def test_redis_stream_error_dlq_unregistered_model(
    tmp_path_factory: pytest.TempPathFactory,
    redis_service: Dict[str, Any],
    iris_model_artifact: IrisModelArtifact,
) -> None:
    """Verifies requesting an unregistered model routes to error_topic with NotFound error."""
    work_dir = tmp_path_factory.mktemp("snooze_redis_unknown_model")
    client: redis.Redis = redis_service["client"]
    endpoint = redis_service["endpoint"]

    config = f"""
[[models]]
id = "sklearn_iris"
path = "{iris_model_artifact.onnx_path.as_posix()}"
max_batch_size = 4
max_batch_delay_ms = 5
idle_timeout_secs = 300
pinned = true

[[streams]]
name = "stream-redis-unknown"
broker = "redis"
endpoint = "{endpoint}"
input_topic = "iris.unknown.in"
output_topic = "iris.unknown.out"
error_topic = "iris.unknown.dlq"
auto_create_topics = true
concurrency = 4
format = "json"
"""
    server = spawn_snooze_server(config, work_dir)
    try:
        time.sleep(0.3)

        req_id = "unknown-model-req"
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
        client.xadd("iris.unknown.in", {"payload": json.dumps(req).encode("utf-8")})

        dlq_responses = _poll_stream_entries(
            client, "iris.unknown.dlq", expected_count=1, timeout=10.0
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
