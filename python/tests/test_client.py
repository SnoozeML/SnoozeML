"""Unit tests for SnoozeClient and AsyncSnoozeClient configuration and endpoint resolution."""

from __future__ import annotations

import asyncio
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import socket
import threading
from typing import Any, Dict, List
import warnings

import numpy as np
import pytest

from snoozeml import (
    MAX_SUPPORTED_SERVER_VERSION,
    MIN_SUPPORTED_SERVER_VERSION,
    AsyncClient,
    AsyncSnoozeClient,
    Client,
    InferenceResponse,
    InferenceTimeoutError,
    InvalidInputError,
    ModelNotFoundError,
    ModelStats,
    StatDuration,
    SnoozeClient,
    SnoozeConnectionError,
    __version__,
)


def test_client_aliases() -> None:
    assert Client is SnoozeClient
    assert AsyncClient is AsyncSnoozeClient
    assert __version__ == "0.1.2"


def test_endpoint_resolution_variants(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SNOOZE_HTTP_ENDPOINT", raising=False)
    monkeypatch.delenv("SNOOZE_GRPC_ENDPOINT", raising=False)

    # 1. Default endpoints
    c1 = SnoozeClient()
    assert c1.http_endpoint == "http://127.0.0.1:8080"
    assert c1.grpc_endpoint == "127.0.0.1:50051"
    assert c1.use_grpc is False
    c1.close()

    # 2. gRPC target endpoint (e.g. "localhost:50051" from design doc)
    c2 = SnoozeClient(endpoint="localhost:50051")
    assert c2.grpc_endpoint == "127.0.0.1:50051"
    assert c2.http_endpoint == "http://127.0.0.1:8080"
    assert c2.use_grpc is True
    c2.close()

    # 3. HTTP URL endpoint
    c3 = SnoozeClient(endpoint="http://localhost:9000")
    assert c3.http_endpoint == "http://127.0.0.1:9000"
    assert c3.grpc_endpoint == "127.0.0.1:50051"
    assert c3.use_grpc is False
    c3.close()

    # 4. Environment variable overrides
    monkeypatch.setenv("SNOOZE_HTTP_ENDPOINT", "http://10.0.0.5:8081")
    monkeypatch.setenv("SNOOZE_GRPC_ENDPOINT", "10.0.0.5:50052")
    monkeypatch.setenv("SNOOZE_API_KEY", "sk-env-key")
    c4 = SnoozeClient(use_grpc=True)
    assert c4.http_endpoint == "http://10.0.0.5:8081"
    assert c4.grpc_endpoint == "10.0.0.5:50052"
    assert c4.use_grpc is True
    assert "sk-env-key" not in repr(c4)
    c4.close()


def test_connection_error_on_unreachable_port() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        closed_port = int(s.getsockname()[1])

    with SnoozeClient(
        http_endpoint=f"http://127.0.0.1:{closed_port}",
        grpc_endpoint=f"127.0.0.1:{closed_port}",
        timeout=0.5,
    ) as client:
        with pytest.raises(SnoozeConnectionError):
            client.models.list()

        with pytest.raises((SnoozeConnectionError, InferenceTimeoutError)):
            client.infer("demo-model", {"input": np.ones((1, 2), dtype=np.float32)}, use_grpc=True)


def test_async_client_connection_error_and_context_manager() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        closed_port = int(s.getsockname()[1])

    async def _check() -> None:
        async with AsyncSnoozeClient(
            http_endpoint=f"http://127.0.0.1:{closed_port}",
            grpc_endpoint=f"127.0.0.1:{closed_port}",
            timeout=0.5,
        ) as aclient:
            with pytest.raises(SnoozeConnectionError):
                await aclient.models.list()

    asyncio.run(_check())


def test_stream_infer_validation() -> None:
    with SnoozeClient() as client:
        with pytest.raises(InvalidInputError, match="model_id cannot be empty"):
            list(client.stream_infer("", inputs=[]))

    async def _async_check() -> None:
        async with AsyncSnoozeClient() as aclient:
            with pytest.raises(InvalidInputError, match="model_id cannot be empty"):
                async for _ in aclient.stream_infer("   ", inputs=[]):
                    pass

    asyncio.run(_async_check())


def test_model_stats_client_methods(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_stats_resp = {
        "model_stats": [
            {
                "name": "bert",
                "version": "1",
                "last_inference": 1730000000500,
                "inference_count": 100,
                "execution_count": 25,
                "inference_stats": {
                    "success": {"count": 98, "ns": 196_000_000},
                    "fail": {"count": 2, "ns": 4_000_000},
                    "queue": {"count": 100, "ns": 50_000_000},
                    "compute_infer": {"count": 100, "ns": 150_000_000},
                },
            }
        ]
    }

    with SnoozeClient() as client:
        monkeypatch.setattr(client, "_request_json", lambda method, path, **kw: mock_stats_resp)

        stats = client.get_model_stats("bert")
        assert stats.name == "bert"
        assert stats.inference_count == 100
        assert stats.execution_count == 25
        assert stats.success.count == 98
        assert stats.fail.count == 2
        assert stats.total_requests == 100
        assert stats.error_rate == pytest.approx(0.02)

        all_stats = client.list_model_stats()
        assert len(all_stats) == 1
        assert all_stats[0].name == "bert"

        # Empty response raises ModelNotFoundError
        monkeypatch.setattr(client, "_request_json", lambda method, path, **kw: {"model_stats": []})
        with pytest.raises(ModelNotFoundError, match="No statistics available for model"):
            client.get_model_stats("bert")

    async def _async_stats_check() -> None:
        async with AsyncSnoozeClient() as aclient:
            monkeypatch.setattr(aclient, "_request_json", lambda method, path, **kw: asyncio.sleep(0, mock_stats_resp))
            stats_async = await aclient.get_model_stats("bert")
            assert stats_async.name == "bert"
            assert stats_async.inference_count == 100

            all_stats_async = await aclient.list_model_stats()
            assert len(all_stats_async) == 1
            assert all_stats_async[0].name == "bert"

    asyncio.run(_async_stats_check())


class _VersionTestHandler(BaseHTTPRequestHandler):
    """Ephemeral HTTP handler for testing server version compatibility checks."""

    state: Dict[str, Any] = {}

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def _send_json(self, status: int, payload: Any) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        calls: List[str] = self.state.setdefault("calls", [])
        calls.append(self.path)

        if self.path == "/health/ready":
            if self.state.get("omit_health_version", False):
                self._send_json(200, {"status": "ready"})
            else:
                self._send_json(
                    200,
                    {"status": "ready", "version": self.state.get("version", "0.1.2")},
                )
            return

        if self.path == "/health/live":
            self._send_json(
                200,
                {"status": "live", "version": self.state.get("version", "0.1.2")},
            )
            return

        if self.path == "/":
            self._send_json(
                200,
                {
                    "service": "snooze-ml",
                    "status": "ok",
                    "version": self.state.get("version", "0.1.2"),
                },
            )
            return

        if self.path == "/v1/models":
            self._send_json(200, [])
            return

        self._send_json(404, {"error": "not found"})


def test_server_version_check_on_first_connection() -> None:
    assert MIN_SUPPORTED_SERVER_VERSION == (0, 1, 2)
    assert MAX_SUPPORTED_SERVER_VERSION == (0, 1)

    _VersionTestHandler.state = {"version": "0.1.2", "calls": []}
    server = HTTPServer(("127.0.0.1", 0), _VersionTestHandler)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        url = f"http://127.0.0.1:{port}"

        # 1. Supported version (0.1.2) -> no warning, checked only once across multiple requests
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with SnoozeClient(url) as client:
                assert client.server_version is None
                assert client.models.list() == []
                assert client.server_version == "0.1.2"
                assert client.models.list() == []
            assert len(caught) == 0
            assert _VersionTestHandler.state["calls"].count("/health/ready") == 1

        # 2. Newer unsupported major/minor version (0.2.0) -> emits UserWarning once on first connection
        _VersionTestHandler.state = {"version": "0.2.0", "calls": []}
        with pytest.warns(UserWarning, match="higher than the maximum supported version"):
            with SnoozeClient(url) as client:
                assert client.models.list() == []
                assert client.server_version == "0.2.0"
                # Subsequent calls must not re-check or re-warn
                with warnings.catch_warnings(record=True) as second_caught:
                    warnings.simplefilter("always")
                    assert client.models.list() == []
                    assert len(second_caught) == 0
        assert _VersionTestHandler.state["calls"].count("/health/ready") == 1

        # 3. Older unsupported version (0.1.1) -> emits UserWarning once on first connection
        _VersionTestHandler.state = {"version": "0.1.1", "calls": []}
        with pytest.warns(UserWarning, match="older than the minimum supported version"):
            with SnoozeClient(url) as client:
                assert client.health_ready() is True
                assert client.server_version == "0.1.1"
        assert _VersionTestHandler.state["calls"] == ["/health/ready"]

        # 4. Fallback to GET / when /health/ready omits "version" (legacy server)
        _VersionTestHandler.state = {
            "version": "0.3.0",
            "omit_health_version": True,
            "calls": [],
        }
        with pytest.warns(UserWarning, match="higher than the maximum supported version"):
            with SnoozeClient(url) as client:
                assert client.models.list() == []
                assert client.server_version == "0.3.0"
        assert "/health/ready" in _VersionTestHandler.state["calls"]
        assert "/" in _VersionTestHandler.state["calls"]

        # 5. check_version=False skips version check completely
        _VersionTestHandler.state = {"version": "2.0.0", "calls": []}
        with warnings.catch_warnings(record=True) as caught_disabled:
            warnings.simplefilter("always")
            with SnoozeClient(url, check_version=False) as client:
                assert client.models.list() == []
                assert client.server_version is None
            assert len(caught_disabled) == 0
            assert "/health/ready" not in _VersionTestHandler.state["calls"]

        # 6. AsyncSnoozeClient emits warning on first connection
        _VersionTestHandler.state = {"version": "0.2.1", "calls": []}

        async def _async_version_check() -> None:
            with pytest.warns(UserWarning, match="higher than the maximum supported version"):
                async with AsyncSnoozeClient(url) as aclient:
                    assert await aclient.models.list() == []
                    assert aclient.server_version == "0.2.1"

        asyncio.run(_async_version_check())
    finally:
        server.shutdown()
        server.server_close()
