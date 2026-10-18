"""Unit tests for snoozeml.exceptions and HTTP/gRPC error mapping parity."""

from __future__ import annotations

from typing import Any, List, Tuple

import grpc
import pytest

from snoozeml.exceptions import (
    AuthenticationError,
    InferenceTimeoutError,
    InvalidInputError,
    ModelLoadingError,
    ModelNotFoundError,
    OutOfMemoryError,
    PermissionDeniedError,
    ResourceExhaustedError,
    RuntimeInferenceError,
    ShapeMismatchError,
    SnoozeConnectionError,
    SnoozeError,
    UnsupportedArchitectureError,
    map_grpc_error,
    map_http_error,
)


class _DummyRpcError(grpc.RpcError):
    """Test helper simulating a grpc.RpcError with status code, details, and trailing metadata."""

    def __init__(
        self,
        code: grpc.StatusCode,
        details: str,
        metadata: List[Tuple[str, str]] | None = None,
    ) -> None:
        super().__init__()
        self._code = code
        self._details = details
        self._metadata = metadata or []

    def code(self) -> grpc.StatusCode:
        return self._code

    def details(self) -> str:
        return self._details

    def trailing_metadata(self) -> Any:
        return self._metadata


@pytest.mark.parametrize(
    ("status_code", "body", "headers", "expected_cls"),
    [
        (400, '{"error": "Invalid tensor input: dimension 0"}', None, InvalidInputError),
        (400, '{"error": "Batch collation shape mismatch: [1, 4] vs [1, 5]"}', None, ShapeMismatchError),
        (401, '{"error": "Missing or invalid API key"}', None, AuthenticationError),
        (403, '{"error": "Insufficient permissions"}', None, PermissionDeniedError),
        (404, '{"error": "Model not found: bert"}', None, ModelNotFoundError),
        (501, '{"error": "Unsupported Safetensors architecture: GPT2"}', None, UnsupportedArchitectureError),
        (503, '{"error": "Model is currently loading weights"}', None, ModelLoadingError),
        (503, '{"error": "Device/System out of memory: CUDA failure 2"}', None, OutOfMemoryError),
        (503, '{"error": "Resource limit reached"}', {"Retry-After": "5"}, ResourceExhaustedError),
        (504, '{"error": "Request processing deadline exceeded"}', None, InferenceTimeoutError),
        (500, '{"error": "Inference runtime failure: ORT error"}', None, RuntimeInferenceError),
        (418, "Plain text error", None, SnoozeError),
    ],
)
def test_map_http_error_parity(
    status_code: int,
    body: str,
    headers: dict[str, str] | None,
    expected_cls: type[SnoozeError],
) -> None:
    err = map_http_error(status_code, body, headers=headers, model_id="test-model")
    assert isinstance(err, expected_cls)
    assert err.status_code == status_code
    assert err.model_id == "test-model"
    if headers and "Retry-After" in headers:
        assert err.retry_after == 5


@pytest.mark.parametrize(
    ("grpc_code", "details", "metadata", "expected_cls"),
    [
        (grpc.StatusCode.NOT_FOUND, "Model not found: bert", None, ModelNotFoundError),
        (grpc.StatusCode.INVALID_ARGUMENT, "Invalid tensor input", None, InvalidInputError),
        (grpc.StatusCode.INVALID_ARGUMENT, "Batch collation shape mismatch", None, ShapeMismatchError),
        (grpc.StatusCode.DEADLINE_EXCEEDED, "Request processing deadline exceeded", None, InferenceTimeoutError),
        (grpc.StatusCode.UNAUTHENTICATED, "Invalid API key", None, AuthenticationError),
        (grpc.StatusCode.PERMISSION_DENIED, "Insufficient permissions", None, PermissionDeniedError),
        (grpc.StatusCode.UNIMPLEMENTED, "Unsupported Safetensors architecture", None, UnsupportedArchitectureError),
        (grpc.StatusCode.RESOURCE_EXHAUSTED, "CUDA failure 2: out of memory", None, OutOfMemoryError),
        (grpc.StatusCode.RESOURCE_EXHAUSTED, "Resource limit reached", [("retry-after", "5")], ResourceExhaustedError),
        (grpc.StatusCode.UNAVAILABLE, "Model is currently loading weights", None, ModelLoadingError),
        (grpc.StatusCode.UNAVAILABLE, "Socket closed", None, SnoozeConnectionError),
        (grpc.StatusCode.INTERNAL, "Runtime failure", None, RuntimeInferenceError),
    ],
)
def test_map_grpc_error_parity(
    grpc_code: grpc.StatusCode,
    details: str,
    metadata: List[Tuple[str, str]] | None,
    expected_cls: type[SnoozeError],
) -> None:
    rpc_err = _DummyRpcError(grpc_code, details, metadata)
    err = map_grpc_error(rpc_err, model_id="test-model")
    assert isinstance(err, expected_cls)
    assert err.grpc_code == grpc_code.name
    assert err.model_id == "test-model"
    if metadata:
        assert err.retry_after == 5
