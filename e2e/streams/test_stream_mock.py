"""End-to-end integration tests for SnoozeML Mock stream driver and Python MockStreamClient.

Tests stream message framing, correlation ID matching, dynamic reply_to routing,
and DLQ error routing using the SDK's MockStreamClient wired to a real live SnoozeML daemon.
"""

from __future__ import annotations

from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest

from e2e.conftest import IrisModelArtifact, spawn_snooze_server
from snoozeml.client import SnoozeClient
from snoozeml.exceptions import RuntimeInferenceError
from snoozeml.streams import MockStreamClient, StreamInferRequest, StreamInferResponse, StreamWireFormat


def test_mock_stream_client_with_live_snooze_daemon(
    tmp_path_factory: pytest.TempPathFactory,
    iris_model_artifact: IrisModelArtifact,
) -> None:
    """Verifies MockStreamClient wired to a live SnoozeClient executes full roundtrip inference."""
    work_dir = tmp_path_factory.mktemp("snooze_mock_stream")
    config = f"""
[[models]]
id = "sklearn_iris"
path = "{iris_model_artifact.onnx_path.as_posix()}"
max_batch_size = 8
max_batch_delay_ms = 5
idle_timeout_secs = 300
pinned = true
"""
    server = spawn_snooze_server(config, work_dir)
    try:
        http_client = SnoozeClient(endpoint=f"http://{server.http_url}")
        stream_client = MockStreamClient(
            client=http_client,
            default_model_id="sklearn_iris",
            input_topic="mock.requests",
            output_topic="mock.responses",
            error_topic="mock.dlq",
        )

        sample = iris_model_artifact.sample_x[:1]  # 2D shape (1, 4)
        resp = stream_client.infer(sample)

        assert resp.model_id == "sklearn_iris"
        probs = resp.get_numpy()
        assert probs.shape == (1, 3)
        assert np.isclose(np.sum(probs), 1.0, atol=1e-3)

        predicted_class = int(np.argmax(probs))
        expected_class = int(iris_model_artifact.pipeline.predict(sample)[0])
        assert predicted_class == expected_class
    finally:
        server.process.terminate()
        try:
            server.process.wait(timeout=3.0)
        except Exception:
            server.process.kill()


def test_mock_stream_client_custom_reply_to_and_correlation_id(
    tmp_path_factory: pytest.TempPathFactory,
    iris_model_artifact: IrisModelArtifact,
) -> None:
    """Verifies correlation ID tracking and dynamic reply_to routing in MockStreamClient."""
    work_dir = tmp_path_factory.mktemp("snooze_mock_reply_to")
    config = f"""
[[models]]
id = "sklearn_iris"
path = "{iris_model_artifact.onnx_path.as_posix()}"
max_batch_size = 4
max_batch_delay_ms = 5
idle_timeout_secs = 300
pinned = true
"""
    server = spawn_snooze_server(config, work_dir)
    try:
        http_client = SnoozeClient(endpoint=f"http://{server.http_url}")
        stream_client = MockStreamClient(
            client=http_client,
            default_model_id="sklearn_iris",
            input_topic="mock.requests",
            output_topic="mock.default.out",
            error_topic="mock.dlq",
        )

        req_id = "corr-mock-stream-999"
        sample = iris_model_artifact.sample_x[:1]

        published_id = stream_client.publish_request(
            sample,
            request_id=req_id,
            reply_to="mock.custom.out",
        )
        assert published_id == req_id

        # Must be available on mock.custom.out
        stream_resp = stream_client.read_response(
            correlation_id=req_id,
            topic="mock.custom.out",
        )
        assert stream_resp is not None
        assert stream_resp.id == req_id
        assert stream_resp.status == "success"

        # Default output topic must have nothing
        default_resp = stream_client.read_response(topic="mock.default.out")
        assert default_resp is None
    finally:
        server.process.terminate()
        try:
            server.process.wait(timeout=3.0)
        except Exception:
            server.process.kill()


def test_mock_stream_client_dlq_error_routing_on_invalid_shape(
    tmp_path_factory: pytest.TempPathFactory,
    iris_model_artifact: IrisModelArtifact,
) -> None:
    """Verifies that an error during inference is cleanly captured and routed to error_topic."""
    work_dir = tmp_path_factory.mktemp("snooze_mock_err")
    config = f"""
[[models]]
id = "sklearn_iris"
path = "{iris_model_artifact.onnx_path.as_posix()}"
max_batch_size = 4
max_batch_delay_ms = 5
idle_timeout_secs = 300
pinned = true
"""
    server = spawn_snooze_server(config, work_dir)
    try:
        http_client = SnoozeClient(endpoint=f"http://{server.http_url}")
        stream_client = MockStreamClient(
            client=http_client,
            default_model_id="sklearn_iris",
            input_topic="mock.requests",
            output_topic="mock.responses",
            error_topic="mock.dlq",
        )

        # Send invalid input with mismatched feature count (shape [1, 2] instead of [1, 4])
        invalid_sample = np.array([[1.0, 2.0]], dtype=np.float32)
        stream_client.publish_request(invalid_sample, request_id="bad-shape-req")

        dlq_resp = stream_client.read_response(
            correlation_id="bad-shape-req",
            topic="mock.dlq",
        )
        assert dlq_resp is not None
        assert dlq_resp.status == "error"
        assert dlq_resp.error is not None
        assert "shape" in dlq_resp.error.lower() or "elements" in dlq_resp.error.lower() or "invalid" in dlq_resp.error.lower()
    finally:
        server.process.terminate()
        try:
            server.process.wait(timeout=3.0)
        except Exception:
            server.process.kill()
