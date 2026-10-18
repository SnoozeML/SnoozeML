"""Domain exception hierarchy and HTTP/gRPC error mapping for SnoozeML.

Maintains strict parity with SnoozeML's Rust ``InferenceError`` variants across both
HTTP REST (``src/server/http.rs``) and Tonic gRPC (``src/server/grpc/convert.rs``).
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional

try:
    import grpc
except ImportError:
    grpc = None  # type: ignore


class SnoozeError(Exception):
    """Base exception for all SnoozeML client, server, and conversion errors.

    Attributes:
        message: Human-readable explanation of the error.
        status_code: Optional HTTP status code (e.g. ``404``, ``503``).
        grpc_code: Optional gRPC status code name (e.g. ``"NOT_FOUND"``).
        retry_after: Optional server-suggested retry delay in seconds.
        model_id: Optional identifier of the target model associated with the failure.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        grpc_code: Optional[str] = None,
        retry_after: Optional[int] = None,
        model_id: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.grpc_code = grpc_code
        self.retry_after = retry_after
        self.model_id = model_id


class ModelNotFoundError(SnoozeError):
    """Raised when a requested ``model_id`` is not registered (HTTP 404 / gRPC ``NOT_FOUND``)."""


class InvalidInputError(SnoozeError):
    """Raised when input tensors have invalid shapes, dtypes, or element counts (HTTP 400 / gRPC ``INVALID_ARGUMENT``)."""


class ShapeMismatchError(InvalidInputError):
    """Raised when tensor dimensions are incompatible during batch collation or forward pass (HTTP 400 / gRPC ``INVALID_ARGUMENT``)."""


class InferenceTimeoutError(SnoozeError):
    """Raised when a request exceeds its processing or queue deadline (HTTP 504 / gRPC ``DEADLINE_EXCEEDED``)."""


class ModelLoadingError(SnoozeError):
    """Raised when model weights are currently being loaded (HTTP 503 / gRPC ``UNAVAILABLE``)."""


class ResourceExhaustedError(SnoozeError):
    """Raised when server memory or queue capacity is exhausted (HTTP 503 / gRPC ``RESOURCE_EXHAUSTED``)."""


class OutOfMemoryError(ResourceExhaustedError):
    """Raised when device VRAM or host RAM allocation fails (HTTP 503 / gRPC ``RESOURCE_EXHAUSTED``)."""


class UnsupportedArchitectureError(SnoozeError):
    """Raised when a Safetensors model architecture is unsupported natively (HTTP 501 / gRPC ``UNIMPLEMENTED``)."""


class AuthenticationError(SnoozeError):
    """Raised when authentication credentials are missing or invalid (HTTP 401 / gRPC ``UNAUTHENTICATED``)."""


class PermissionDeniedError(SnoozeError):
    """Raised when the authenticated identity lacks the required RBAC role (HTTP 403 / gRPC ``PERMISSION_DENIED``)."""


class SnoozeConnectionError(SnoozeError):
    """Raised when the SnoozeML daemon cannot be reached over HTTP or gRPC."""


class RuntimeInferenceError(SnoozeError):
    """Raised when an internal engine or runtime error occurs (HTTP 500 / gRPC ``INTERNAL``)."""


class ConversionError(SnoozeError):
    """Raised when exporting a scikit-learn, PyTorch, or XGBoost model fails."""


def _extract_error_message(body_text: str) -> str:
    """Extracts the ``error`` field from a JSON error response body if present."""
    if not body_text:
        return "Unknown server error"
    stripped = body_text.strip()
    if stripped.startswith("{"):
        try:
            parsed: Any = json.loads(stripped)
            if isinstance(parsed, dict) and "error" in parsed:
                return str(parsed["error"])
        except Exception:
            pass
    return stripped


def _parse_retry_after(headers: Optional[Mapping[str, str]]) -> Optional[int]:
    """Parses the ``Retry-After`` HTTP header or gRPC metadata key if present."""
    if not headers:
        return None
    for key, value in headers.items():
        if key.lower() == "retry-after":
            try:
                return int(str(value).strip())
            except ValueError:
                return None
    return None


def map_http_error(
    status_code: int,
    body_text: str,
    headers: Optional[Mapping[str, str]] = None,
    model_id: Optional[str] = None,
) -> SnoozeError:
    """Maps an HTTP error response from SnoozeML to the corresponding :class:`SnoozeError` subclass.

    Args:
        status_code: HTTP status code returned by the server.
        body_text: Raw response body text or JSON string.
        headers: Optional response headers dictionary.
        model_id: Optional model identifier involved in the request.

    Returns:
        A specific :class:`SnoozeError` subclass matching the failure mode.
    """
    msg = _extract_error_message(body_text)
    msg_lower = msg.lower()
    retry_after = _parse_retry_after(headers)

    kwargs = {
        "status_code": status_code,
        "retry_after": retry_after,
        "model_id": model_id,
    }

    if status_code == 400:
        if "shape mismatch" in msg_lower or "incompatible" in msg_lower:
            return ShapeMismatchError(msg, **kwargs)
        return InvalidInputError(msg, **kwargs)

    if status_code == 401:
        return AuthenticationError(msg, **kwargs)

    if status_code == 403:
        return PermissionDeniedError(msg, **kwargs)

    if status_code == 404:
        return ModelNotFoundError(msg, **kwargs)

    if status_code == 501:
        return UnsupportedArchitectureError(msg, **kwargs)

    if status_code == 503:
        if "currently loading" in msg_lower or "loading weights" in msg_lower:
            return ModelLoadingError(msg, **kwargs)
        if (
            "out of memory" in msg_lower
            or "bad_alloc" in msg_lower
            or "cuda failure 2" in msg_lower
        ):
            return OutOfMemoryError(msg, **kwargs)
        return ResourceExhaustedError(msg, **kwargs)

    if status_code == 504:
        return InferenceTimeoutError(msg, **kwargs)

    if status_code >= 500:
        return RuntimeInferenceError(msg, **kwargs)

    return SnoozeError(msg, **kwargs)


def map_grpc_error(
    rpc_error: Any,
    model_id: Optional[str] = None,
) -> SnoozeError:
    """Maps a :class:`grpc.RpcError` from SnoozeML to the canonical :class:`SnoozeError` subclass.

    Args:
        rpc_error: The gRPC exception raised by ``grpcio``.
        model_id: Optional model identifier involved in the RPC.

    Returns:
        A specific :class:`SnoozeError` subclass matching the gRPC status code.
    """
    if grpc is None:
        return RuntimeInferenceError(str(rpc_error), model_id=model_id)

    code = rpc_error.code() if hasattr(rpc_error, "code") else grpc.StatusCode.UNKNOWN
    details = (
        rpc_error.details()
        if hasattr(rpc_error, "details") and rpc_error.details()
        else str(rpc_error)
    )
    code_name = code.name if hasattr(code, "name") else str(code)
    details_lower = details.lower()

    retry_after: Optional[int] = None
    if hasattr(rpc_error, "trailing_metadata"):
        try:
            trailing = rpc_error.trailing_metadata()
            if trailing:
                meta_dict = {str(k).lower(): str(v) for k, v in trailing}
                retry_after = _parse_retry_after(meta_dict)
        except Exception:
            pass

    kwargs = {
        "grpc_code": code_name,
        "retry_after": retry_after,
        "model_id": model_id,
    }

    if code == grpc.StatusCode.NOT_FOUND:
        return ModelNotFoundError(details, status_code=404, **kwargs)

    if code == grpc.StatusCode.INVALID_ARGUMENT:
        if "shape mismatch" in details_lower or "incompatible" in details_lower:
            return ShapeMismatchError(details, status_code=400, **kwargs)
        return InvalidInputError(details, status_code=400, **kwargs)

    if code == grpc.StatusCode.DEADLINE_EXCEEDED:
        return InferenceTimeoutError(details, status_code=504, **kwargs)

    if code == grpc.StatusCode.UNAUTHENTICATED:
        return AuthenticationError(details, status_code=401, **kwargs)

    if code == grpc.StatusCode.PERMISSION_DENIED:
        return PermissionDeniedError(details, status_code=403, **kwargs)

    if code == grpc.StatusCode.UNIMPLEMENTED:
        return UnsupportedArchitectureError(details, status_code=501, **kwargs)

    if code == grpc.StatusCode.RESOURCE_EXHAUSTED:
        if (
            "out of memory" in details_lower
            or "bad_alloc" in details_lower
            or "cuda failure 2" in details_lower
        ):
            return OutOfMemoryError(details, status_code=503, **kwargs)
        return ResourceExhaustedError(details, status_code=503, **kwargs)

    if code == grpc.StatusCode.UNAVAILABLE:
        if "currently loading" in details_lower or "loading weights" in details_lower:
            return ModelLoadingError(details, status_code=503, **kwargs)
        return SnoozeConnectionError(details, status_code=503, **kwargs)

    if code == grpc.StatusCode.INTERNAL:
        return RuntimeInferenceError(details, status_code=500, **kwargs)

    return SnoozeError(details, **kwargs)
