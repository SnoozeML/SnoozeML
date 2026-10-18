"""Unit tests for SnoozeClient and AsyncSnoozeClient configuration and endpoint resolution."""

from __future__ import annotations

import asyncio
import socket

import numpy as np
import pytest

from snoozeml import (
    AsyncClient,
    AsyncSnoozeClient,
    Client,
    InferenceTimeoutError,
    SnoozeClient,
    SnoozeConnectionError,
)


def test_client_aliases() -> None:
    assert Client is SnoozeClient
    assert AsyncClient is AsyncSnoozeClient


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
