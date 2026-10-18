"""High-level synchronous and asynchronous clients for the SnoozeML inference daemon."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
import urllib.parse

try:
    import grpc
    from grpc_health.v1 import health_pb2, health_pb2_grpc
    from ._proto import InferenceServiceStub
    _HAS_GRPC = True
    _GrpcRpcError = grpc.RpcError
except ImportError:
    grpc = None  # type: ignore
    health_pb2 = None  # type: ignore
    health_pb2_grpc = None  # type: ignore
    InferenceServiceStub = None  # type: ignore
    _HAS_GRPC = False

    class _GrpcRpcError(Exception):  # type: ignore
        pass

import requests
from .exceptions import (
    InferenceTimeoutError,
    InvalidInputError,
    SnoozeConnectionError,
    SnoozeError,
    map_grpc_error,
    map_http_error,
)
from .inference import (
    InferenceResponse,
    InputTypes,
    build_grpc_infer_request,
    normalize_inputs,
)
from .logging import get_logger, mask_secret, register_secret_for_redaction
from .models import AsyncModelsManager, ModelInfo, ModelsManager

logger = get_logger("client")

DEFAULT_HTTP_PORT = 8080
DEFAULT_GRPC_PORT = 50051
DEFAULT_MAX_MESSAGE_LENGTH = 64 * 1024 * 1024  # 64 MB


def _normalize_localhost_url(url: str) -> str:
    """Replaces ``localhost`` with ``127.0.0.1`` in HTTP URLs to avoid IPv6 fallback delays."""
    parsed = urllib.parse.urlparse(url)
    if parsed.hostname == "localhost":
        port_part = f":{parsed.port}" if parsed.port else ""
        netloc = f"127.0.0.1{port_part}"
        return urllib.parse.urlunparse(parsed._replace(netloc=netloc))
    return url.rstrip("/")


def _normalize_grpc_target(target: str) -> str:
    """Normalizes a gRPC ``host:port`` target string, mapping ``localhost`` to ``127.0.0.1``."""
    cleaned = target.strip()
    if cleaned.startswith("http://"):
        cleaned = cleaned[len("http://") :]
    elif cleaned.startswith("https://"):
        cleaned = cleaned[len("https://") :]
    cleaned = cleaned.rstrip("/")
    if cleaned.startswith("localhost:"):
        cleaned = "127.0.0.1:" + cleaned[len("localhost:") :]
    elif cleaned == "localhost":
        cleaned = f"127.0.0.1:{DEFAULT_GRPC_PORT}"
    elif ":" not in cleaned:
        cleaned = f"{cleaned}:{DEFAULT_GRPC_PORT}"
    return cleaned


def _resolve_endpoints(
    endpoint: Optional[str],
    http_endpoint: Optional[str],
    grpc_endpoint: Optional[str],
    use_grpc: Optional[bool],
) -> Tuple[str, str, bool]:
    """Resolves HTTP base URL, gRPC target, and default ``use_grpc`` flag from user parameters."""
    env_http = os.getenv("SNOOZE_HTTP_ENDPOINT")
    env_grpc = os.getenv("SNOOZE_GRPC_ENDPOINT")

    resolved_http = http_endpoint or env_http
    resolved_grpc = grpc_endpoint or env_grpc
    inferred_use_grpc = False

    if endpoint:
        ep = endpoint.strip()
        if ep.startswith("http://") or ep.startswith("https://"):
            if not resolved_http:
                resolved_http = ep
            parsed = urllib.parse.urlparse(ep)
            host = parsed.hostname or "127.0.0.1"
            if not resolved_grpc:
                resolved_grpc = f"{host}:{DEFAULT_GRPC_PORT}"
            inferred_use_grpc = False
        else:
            # Bare host or host:port
            host_part, _, port_part = ep.partition(":")
            host = host_part or "127.0.0.1"
            port = int(port_part) if port_part.isdigit() else None
            if use_grpc is True or port == DEFAULT_GRPC_PORT:
                if not resolved_grpc:
                    resolved_grpc = f"{host}:{port or DEFAULT_GRPC_PORT}"
                if not resolved_http:
                    resolved_http = f"http://{host}:{DEFAULT_HTTP_PORT}"
                inferred_use_grpc = True
            else:
                if not resolved_http:
                    resolved_http = f"http://{host}:{port or DEFAULT_HTTP_PORT}"
                if not resolved_grpc:
                    resolved_grpc = f"{host}:{DEFAULT_GRPC_PORT}"
                inferred_use_grpc = False

    final_http = _normalize_localhost_url(
        resolved_http or f"http://127.0.0.1:{DEFAULT_HTTP_PORT}"
    )
    final_grpc = _normalize_grpc_target(
        resolved_grpc or f"127.0.0.1:{DEFAULT_GRPC_PORT}"
    )
    final_use_grpc = use_grpc if use_grpc is not None else inferred_use_grpc
    if not _HAS_GRPC:
        if use_grpc is True:
            raise SnoozeError(
                "gRPC support requires 'grpcio' and 'protobuf'. Install with: pip install 'snoozeml[grpc]'"
            )
        final_use_grpc = False
    return final_http, final_grpc, final_use_grpc


def _read_pem_bytes(tls_ca_cert: Union[str, bytes, Path]) -> bytes:
    """Reads PEM certificate bytes from a file path, PEM string, or raw bytes."""
    if isinstance(tls_ca_cert, bytes):
        return tls_ca_cert
    if isinstance(tls_ca_cert, Path):
        return tls_ca_cert.read_bytes()
    if isinstance(tls_ca_cert, str):
        if "-----BEGIN CERTIFICATE-----" in tls_ca_cert:
            return tls_ca_cert.encode("utf-8")
        return Path(tls_ca_cert).read_bytes()
    raise TypeError(f"Unsupported tls_ca_cert type: {type(tls_ca_cert).__name__}")


class SnoozeClient:
    """Synchronous client for SnoozeML model management and tensor inference.

    Supports both HTTP REST (JSON) and high-throughput native gRPC (raw FP32 byte buffers)
    with unified API Key / JWT Bearer authentication and optional TLS.

    Example::

        import numpy as np
        from snoozeml import SnoozeClient

        with SnoozeClient(endpoint="localhost:50051", use_grpc=True) as client:
            models = client.models.list()
            input_data = np.random.randn(1, 10).astype(np.float32)
            resp = client.infer("demo-model", inputs={"input": input_data})
            output = resp.get_numpy("output")
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        *,
        http_endpoint: Optional[str] = None,
        grpc_endpoint: Optional[str] = None,
        use_grpc: Optional[bool] = None,
        api_key: Optional[str] = None,
        token: Optional[str] = None,
        timeout: float = 30.0,
        tls_ca_cert: Optional[Union[str, bytes, Path]] = None,
        verify_tls: bool = True,
        max_message_length: int = DEFAULT_MAX_MESSAGE_LENGTH,
    ) -> None:
        self.http_endpoint, self.grpc_endpoint, self.use_grpc = _resolve_endpoints(
            endpoint=endpoint,
            http_endpoint=http_endpoint,
            grpc_endpoint=grpc_endpoint,
            use_grpc=use_grpc,
        )
        self._api_key = api_key if api_key is not None else os.getenv("SNOOZE_API_KEY")
        self._token = token if token is not None else os.getenv("SNOOZE_JWT_TOKEN")
        register_secret_for_redaction(self._api_key)
        register_secret_for_redaction(self._token)

        self.timeout = float(timeout)
        self.tls_ca_cert = tls_ca_cert
        self.verify_tls = verify_tls
        self.max_message_length = int(max_message_length)

        self._session = requests.Session()
        if isinstance(tls_ca_cert, (str, Path)) and "-----BEGIN CERTIFICATE-----" not in str(tls_ca_cert):
            self._session.verify = str(tls_ca_cert)
        else:
            self._session.verify = verify_tls

        self._channel: Optional[grpc.Channel] = None
        self._infer_stub: Optional[InferenceServiceStub] = None
        self._health_stub: Optional[health_pb2_grpc.HealthStub] = None

        self.models = ModelsManager(self)

        logger.debug(
            "Initialized SnoozeClient",
            extra={
                "http_endpoint": self.http_endpoint,
                "grpc_endpoint": self.grpc_endpoint,
                "use_grpc": self.use_grpc,
                "auth_configured": bool(self._api_key or self._token),
            },
        )

    def __repr__(self) -> str:
        return (
            f"SnoozeClient(http_endpoint={self.http_endpoint!r}, "
            f"grpc_endpoint={self.grpc_endpoint!r}, "
            f"use_grpc={self.use_grpc!r}, "
            f"api_key={mask_secret(self._api_key)!r}, "
            f"token={mask_secret(self._token)!r})"
        )

    def __enter__(self) -> "SnoozeClient":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        """Closes underlying HTTP connection pools and gRPC channels."""
        self._session.close()
        if self._channel is not None:
            try:
                self._channel.close()
            except Exception:
                pass
            self._channel = None
            self._infer_stub = None
            self._health_stub = None
        logger.debug("Closed SnoozeClient connections")

    def _http_headers(self, json_body: bool = True) -> Dict[str, str]:
        """Constructs HTTP request headers including authentication credentials."""
        headers: Dict[str, str] = {}
        if json_body:
            headers["Content-Type"] = "application/json"
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        elif self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
            headers["X-API-Key"] = self._api_key
        return headers

    def _grpc_metadata(self) -> List[Tuple[str, str]]:
        """Constructs gRPC call metadata tuples including authentication credentials."""
        metadata: List[Tuple[str, str]] = []
        if self._token:
            metadata.append(("authorization", f"Bearer {self._token}"))
        elif self._api_key:
            metadata.append(("authorization", f"Bearer {self._api_key}"))
            metadata.append(("x-api-key", self._api_key))
        return metadata

    def _ensure_grpc_stubs(self) -> Tuple[Any, Any]:
        """Lazily initializes the gRPC channel and service stubs."""
        if not _HAS_GRPC or grpc is None or InferenceServiceStub is None or health_pb2_grpc is None:
            raise SnoozeError(
                "gRPC support requires 'grpcio' and 'protobuf'. Install with: pip install 'snoozeml[grpc]'"
            )
        if self._channel is None or self._infer_stub is None or self._health_stub is None:
            options = [
                ("grpc.max_receive_message_length", self.max_message_length),
                ("grpc.max_send_message_length", self.max_message_length),
                ("grpc.keepalive_time_ms", 10000),
                ("grpc.keepalive_timeout_ms", 5000),
            ]
            use_tls = self.tls_ca_cert is not None or self.http_endpoint.startswith("https://")
            if use_tls:
                root_certs = (
                    _read_pem_bytes(self.tls_ca_cert)
                    if self.tls_ca_cert is not None
                    else None
                )
                creds = grpc.ssl_channel_credentials(root_certificates=root_certs)
                self._channel = grpc.secure_channel(
                    self.grpc_endpoint, creds, options=options
                )
            else:
                self._channel = grpc.insecure_channel(
                    self.grpc_endpoint, options=options
                )
            self._infer_stub = InferenceServiceStub(self._channel)
            self._health_stub = health_pb2_grpc.HealthStub(self._channel)
        return self._infer_stub, self._health_stub

    def _request_raw(
        self,
        method: str,
        path: str,
        *,
        json_payload: Optional[Any] = None,
        timeout: Optional[float] = None,
        model_id: Optional[str] = None,
    ) -> requests.Response:
        """Executes an HTTP request against the SnoozeML REST endpoint and maps errors."""
        url = f"{self.http_endpoint}{path}"
        req_timeout = timeout if timeout is not None else self.timeout
        headers = self._http_headers(json_body=json_payload is not None)
        body_bytes = (
            json.dumps(json_payload, allow_nan=False).encode("utf-8")
            if json_payload is not None
            else None
        )

        t0 = time.perf_counter()
        try:
            resp = self._session.request(
                method=method,
                url=url,
                data=body_bytes,
                headers=headers,
                timeout=req_timeout,
            )
        except requests.exceptions.ConnectTimeout as exc:
            logger.error(
                "HTTP connection timed out",
                extra={"method": method, "url": url, "timeout_s": req_timeout},
            )
            raise SnoozeConnectionError(
                f"Failed to connect to SnoozeML HTTP server at {url} within {req_timeout}s: {exc}",
                model_id=model_id,
            ) from exc
        except requests.exceptions.Timeout as exc:
            logger.error(
                "HTTP request timed out",
                extra={"method": method, "url": url, "timeout_s": req_timeout},
            )
            raise InferenceTimeoutError(
                f"HTTP request to {url} timed out after {req_timeout}s",
                status_code=504,
                model_id=model_id,
            ) from exc
        except requests.exceptions.RequestException as exc:
            logger.error(
                "HTTP connection error",
                extra={"method": method, "url": url, "error": str(exc)},
            )
            raise SnoozeConnectionError(
                f"Failed to connect to SnoozeML HTTP server at {url}: {exc}",
                model_id=model_id,
            ) from exc

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if resp.status_code >= 400:
            mapped_err = map_http_error(
                status_code=resp.status_code,
                body_text=resp.text,
                headers=dict(resp.headers),
                model_id=model_id,
            )
            logger.warning(
                "HTTP request failed",
                extra={
                    "method": method,
                    "path": path,
                    "status_code": resp.status_code,
                    "duration_ms": round(elapsed_ms, 2),
                    "error": mapped_err.message,
                },
            )
            raise mapped_err

        return resp

    def _request_multipart(
        self,
        path: str,
        *,
        files: Mapping[str, Any],
        data: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
        model_id: Optional[str] = None,
    ) -> requests.Response:
        """Executes a multipart/form-data POST request against the SnoozeML REST endpoint."""
        url = f"{self.http_endpoint}{path}"
        req_timeout = timeout if timeout is not None else self.timeout
        # Omit application/json so requests calculates multipart boundary header automatically
        headers = self._http_headers(json_body=False)

        t0 = time.perf_counter()
        try:
            resp = self._session.post(
                url=url,
                files=files,
                data=data,
                headers=headers,
                timeout=req_timeout,
            )
        except requests.exceptions.ConnectTimeout as exc:
            logger.error(
                "HTTP connection timed out during multipart upload",
                extra={"url": url, "timeout_s": req_timeout},
            )
            raise SnoozeConnectionError(
                f"Failed to connect to SnoozeML HTTP server at {url} within {req_timeout}s: {exc}",
                model_id=model_id,
            ) from exc
        except requests.exceptions.Timeout as exc:
            logger.error(
                "HTTP request timed out during multipart upload",
                extra={"url": url, "timeout_s": req_timeout},
            )
            raise InferenceTimeoutError(
                f"HTTP request to {url} timed out after {req_timeout}s",
                status_code=504,
                model_id=model_id,
            ) from exc
        except requests.exceptions.RequestException as exc:
            logger.error(
                "HTTP connection error during multipart upload",
                extra={"url": url, "error": str(exc)},
            )
            raise SnoozeConnectionError(
                f"Failed to connect to SnoozeML HTTP server at {url}: {exc}",
                model_id=model_id,
            ) from exc

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if resp.status_code >= 400:
            mapped_err = map_http_error(
                status_code=resp.status_code,
                body_text=resp.text,
                headers=dict(resp.headers),
                model_id=model_id,
            )
            logger.warning(
                "HTTP multipart request failed",
                extra={
                    "path": path,
                    "status_code": resp.status_code,
                    "duration_ms": round(elapsed_ms, 2),
                    "error": mapped_err.message,
                },
            )
            raise mapped_err

        return resp

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        json_payload: Optional[Any] = None,
        timeout: Optional[float] = None,
        model_id: Optional[str] = None,
    ) -> Any:
        """Executes an HTTP request and parses the JSON response body."""
        resp = self._request_raw(
            method,
            path,
            json_payload=json_payload,
            timeout=timeout,
            model_id=model_id,
        )
        try:
            return resp.json()
        except ValueError as exc:
            raise SnoozeError(
                f"Invalid JSON response from {path}: {resp.text[:200]}",
                status_code=resp.status_code,
                model_id=model_id,
            ) from exc

    def infer(
        self,
        model_id: str,
        inputs: InputTypes,
        *,
        use_grpc: Optional[bool] = None,
        timeout: Optional[float] = None,
        client_timeout_ms: Optional[int] = None,
        parameters: Optional[Mapping[str, str]] = None,
    ) -> InferenceResponse:
        """Executes unary tensor inference against a registered model.

        Args:
            model_id: Target model identifier (e.g. ``"demo-model"``, ``"bert-base-uncased"``).
            inputs: Input tensors provided as a dictionary ``{name: np.ndarray}``, a list of
                :class:`~snoozeml.inference.TensorInput` or dicts, or a single ``np.ndarray``.
            use_grpc: Optional per-request transport override. If ``None``, uses :attr:`use_grpc`.
            timeout: Optional request timeout in seconds (defaults to :attr:`timeout`).
            client_timeout_ms: Optional gRPC deadline hint in milliseconds embedded in the request.
            parameters: Optional gRPC execution parameters / routing hints.

        Returns:
            :class:`~snoozeml.inference.InferenceResponse` containing output tensors and server latency.

        Raises:
            InvalidInputError: If ``model_id`` or ``inputs`` fail validation.
            ModelNotFoundError: If ``model_id`` is not registered.
            SnoozeError: On authentication, timeout, resource exhaustion, or runtime failure.
        """
        if not model_id or not str(model_id).strip():
            raise InvalidInputError("model_id cannot be empty")
        model_id = str(model_id).strip()

        tensor_inputs = normalize_inputs(inputs)
        dispatch_grpc = self.use_grpc if use_grpc is None else bool(use_grpc)

        if dispatch_grpc:
            return self._infer_grpc(
                model_id=model_id,
                tensors=tensor_inputs,
                timeout=timeout,
                client_timeout_ms=client_timeout_ms,
                parameters=parameters,
            )
        return self._infer_rest(
            model_id=model_id,
            tensors=tensor_inputs,
            timeout=timeout,
        )

    def _infer_rest(
        self,
        model_id: str,
        tensors: List[Any],
        timeout: Optional[float] = None,
    ) -> InferenceResponse:
        """Executes inference over HTTP REST (``POST /v1/models/{model_id}/infer``)."""
        payload = {"inputs": [t.to_rest_dict() for t in tensors]}
        t0 = time.perf_counter()
        data = self._request_json(
            "POST",
            f"/v1/models/{model_id}/infer",
            json_payload=payload,
            timeout=timeout,
            model_id=model_id,
        )
        client_ms = (time.perf_counter() - t0) * 1000.0
        response = InferenceResponse.from_rest_dict(data, default_model_id=model_id)
        logger.debug(
            "Completed REST inference",
            extra={
                "model_id": model_id,
                "transport": "http",
                "server_duration_ms": round(response.duration_ms, 3),
                "client_latency_ms": round(client_ms, 3),
                "outputs": len(response.outputs),
            },
        )
        return response

    def _infer_grpc(
        self,
        model_id: str,
        tensors: List[Any],
        timeout: Optional[float] = None,
        client_timeout_ms: Optional[int] = None,
        parameters: Optional[Mapping[str, str]] = None,
    ) -> InferenceResponse:
        """Executes inference over native gRPC (``snoozeml.v1.InferenceService/Infer``)."""
        infer_stub, _ = self._ensure_grpc_stubs()
        req = build_grpc_infer_request(
            model_id=model_id,
            tensors=tensors,
            client_timeout_ms=client_timeout_ms,
            parameters=parameters,
        )
        req_timeout = timeout if timeout is not None else self.timeout
        metadata = self._grpc_metadata()

        t0 = time.perf_counter()
        try:
            proto_resp = infer_stub.Infer(
                req,
                timeout=req_timeout,
                metadata=metadata,
            )
        except _GrpcRpcError as exc:
            mapped = map_grpc_error(exc, model_id=model_id)
            logger.warning(
                "gRPC inference failed",
                extra={
                    "model_id": model_id,
                    "transport": "grpc",
                    "grpc_code": mapped.grpc_code,
                    "error": mapped.message,
                },
            )
            raise mapped from exc

        client_ms = (time.perf_counter() - t0) * 1000.0
        response = InferenceResponse.from_proto(proto_resp)
        logger.debug(
            "Completed gRPC inference",
            extra={
                "model_id": model_id,
                "transport": "grpc",
                "server_duration_ms": round(response.duration_ms, 3),
                "client_latency_ms": round(client_ms, 3),
                "outputs": len(response.outputs),
            },
        )
        return response

    # ------------------------------------------------------------------
    # Convenience Model Management & Health Methods
    # ------------------------------------------------------------------

    def list_models(self) -> List[ModelInfo]:
        """Lists all registered models (shorthand for ``client.models.list()``)."""
        return self.models.list()

    def get_model(self, model_id: str) -> ModelInfo:
        """Retrieves metadata for a single model (shorthand for ``client.models.get(model_id)``)."""
        return self.models.get(model_id)

    def load_model(self, model_id: str, *, pin: bool = False) -> Dict[str, Any]:
        """Loads a model into memory (shorthand for ``client.models.load(model_id, pin=pin)``)."""
        return self.models.load(model_id, pin=pin)

    def unload_model(self, model_id: str) -> Dict[str, Any]:
        """Unloads a model from memory (shorthand for ``client.models.unload(model_id)``)."""
        return self.models.unload(model_id)

    def evict_model(self, model_id: str) -> Dict[str, Any]:
        """Evicts a model from memory (shorthand for ``client.models.evict(model_id)``)."""
        return self.models.evict(model_id)

    def pin_model(self, model_id: str) -> Dict[str, Any]:
        """Pins a model in memory (shorthand for ``client.models.pin(model_id)``)."""
        return self.models.pin(model_id)

    def unpin_model(self, model_id: str) -> Dict[str, Any]:
        """Unpins a model (shorthand for ``client.models.unpin(model_id)``)."""
        return self.models.unpin(model_id)

    def health_live(self) -> bool:
        """Checks the HTTP liveness probe (``GET /health/live``).

        Returns:
            ``True`` if the server responds with ``{"status": "live"}``.
        """
        data = self._request_json("GET", "/health/live")
        return isinstance(data, Mapping) and data.get("status") == "live"

    def health_ready(self) -> bool:
        """Checks the HTTP readiness probe (``GET /health/ready``).

        Returns:
            ``True`` if the server responds with ``{"status": "ready"}``.
        """
        data = self._request_json("GET", "/health/ready")
        return isinstance(data, Mapping) and data.get("status") == "ready"

    def grpc_health(self, service: str = "") -> bool:
        """Checks the gRPC standard health check service (``grpc.health.v1.Health/Check``).

        Args:
            service: Service name to query (``""`` for root server or
                ``"snoozeml.v1.InferenceService"``).

        Returns:
            ``True`` if the service reports ``SERVING``.
        """
        _, health_stub = self._ensure_grpc_stubs()
        req = health_pb2.HealthCheckRequest(service=service)
        try:
            res = health_stub.Check(req, timeout=self.timeout)
            return bool(res.status == health_pb2.HealthCheckResponse.SERVING)
        except _GrpcRpcError as exc:
            raise map_grpc_error(exc) from exc

    def service_info(self) -> Dict[str, Any]:
        """Retrieves daemon metadata and route discovery info from ``GET /``."""
        data = self._request_json("GET", "/")
        return dict(data) if isinstance(data, Mapping) else {}

    def metrics(self) -> str:
        """Retrieves Prometheus exposition metrics text from ``GET /metrics``."""
        resp = self._request_raw("GET", "/metrics")
        return resp.text

    def upload_model(
        self,
        file: Union[str, Path, bytes, Any],
        *,
        filename: Optional[str] = None,
        model_id: Optional[str] = None,
        description: Optional[str] = None,
        pinned: bool = False,
        max_batch_size: Optional[int] = None,
        max_batch_delay_ms: Optional[int] = None,
        idle_timeout_secs: Optional[int] = None,
        min_residency_secs: Optional[int] = None,
        autoscale_queue_threshold: Optional[int] = None,
        max_replicas: Optional[int] = None,
        routing_strategy: Optional[str] = None,
        device: Optional[str] = None,
    ) -> ModelInfo:
        """Uploads and dynamically registers a model via HTTP without mounting local volumes.

        Convenience alias for :meth:`ModelsManager.upload`.
        """
        return self.models.upload(
            file=file,
            filename=filename,
            model_id=model_id,
            description=description,
            pinned=pinned,
            max_batch_size=max_batch_size,
            max_batch_delay_ms=max_batch_delay_ms,
            idle_timeout_secs=idle_timeout_secs,
            min_residency_secs=min_residency_secs,
            autoscale_queue_threshold=autoscale_queue_threshold,
            max_replicas=max_replicas,
            routing_strategy=routing_strategy,
            device=device,
        )

    def refresh_models(self) -> List[ModelInfo]:
        """Triggers runtime discovery and refresh of models on the SnoozeML daemon via HTTP.

        Convenience alias for :meth:`ModelsManager.refresh`.
        """
        return self.models.refresh()


class AsyncSnoozeClient:
    """Asynchronous client for SnoozeML model management and tensor inference.

    Uses ``grpc.aio`` for non-blocking gRPC transport and asynchronous HTTP execution.

    Example::

        import numpy as np
        from snoozeml import AsyncSnoozeClient

        async with AsyncSnoozeClient(endpoint="localhost:50051", use_grpc=True) as client:
            models = await client.models.list()
            resp = await client.infer("demo-model", {"input": np.ones((1, 10), dtype=np.float32)})
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        *,
        http_endpoint: Optional[str] = None,
        grpc_endpoint: Optional[str] = None,
        use_grpc: Optional[bool] = None,
        api_key: Optional[str] = None,
        token: Optional[str] = None,
        timeout: float = 30.0,
        tls_ca_cert: Optional[Union[str, bytes, Path]] = None,
        verify_tls: bool = True,
        max_message_length: int = DEFAULT_MAX_MESSAGE_LENGTH,
    ) -> None:
        self._sync_client = SnoozeClient(
            endpoint=endpoint,
            http_endpoint=http_endpoint,
            grpc_endpoint=grpc_endpoint,
            use_grpc=use_grpc,
            api_key=api_key,
            token=token,
            timeout=timeout,
            tls_ca_cert=tls_ca_cert,
            verify_tls=verify_tls,
            max_message_length=max_message_length,
        )
        self.http_endpoint = self._sync_client.http_endpoint
        self.grpc_endpoint = self._sync_client.grpc_endpoint
        self.use_grpc = self._sync_client.use_grpc
        self.timeout = self._sync_client.timeout
        self.tls_ca_cert = tls_ca_cert
        self.verify_tls = verify_tls
        self.max_message_length = self._sync_client.max_message_length

        self._aio_channel: Optional[grpc.aio.Channel] = None
        self._aio_infer_stub: Optional[InferenceServiceStub] = None
        self._aio_health_stub: Optional[health_pb2_grpc.HealthStub] = None

        self.models = AsyncModelsManager(self)

    def __repr__(self) -> str:
        return (
            f"AsyncSnoozeClient(http_endpoint={self.http_endpoint!r}, "
            f"grpc_endpoint={self.grpc_endpoint!r}, "
            f"use_grpc={self.use_grpc!r}, "
            f"api_key={mask_secret(self._sync_client._api_key)!r}, "
            f"token={mask_secret(self._sync_client._token)!r})"
        )

    async def __aenter__(self) -> "AsyncSnoozeClient":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    async def close(self) -> None:
        """Asynchronously closes underlying HTTP sessions and ``grpc.aio`` channels."""
        self._sync_client.close()
        if self._aio_channel is not None:
            try:
                await self._aio_channel.close()
            except Exception:
                pass
            self._aio_channel = None
            self._aio_infer_stub = None
            self._aio_health_stub = None

    def _ensure_aio_grpc_stubs(self) -> Tuple[Any, Any]:
        """Lazily initializes the ``grpc.aio`` asynchronous channel and stubs."""
        if not _HAS_GRPC or grpc is None or InferenceServiceStub is None or health_pb2_grpc is None:
            raise SnoozeError(
                "gRPC support requires 'grpcio' and 'protobuf'. Install with: pip install 'snoozeml[grpc]'"
            )
        if (
            self._aio_channel is None
            or self._aio_infer_stub is None
            or self._aio_health_stub is None
        ):
            options = [
                ("grpc.max_receive_message_length", self.max_message_length),
                ("grpc.max_send_message_length", self.max_message_length),
                ("grpc.keepalive_time_ms", 10000),
                ("grpc.keepalive_timeout_ms", 5000),
            ]
            use_tls = self.tls_ca_cert is not None or self.http_endpoint.startswith("https://")
            if use_tls:
                root_certs = (
                    _read_pem_bytes(self.tls_ca_cert)
                    if self.tls_ca_cert is not None
                    else None
                )
                creds = grpc.ssl_channel_credentials(root_certificates=root_certs)
                self._aio_channel = grpc.aio.secure_channel(
                    self.grpc_endpoint, creds, options=options
                )
            else:
                self._aio_channel = grpc.aio.insecure_channel(
                    self.grpc_endpoint, options=options
                )
            self._aio_infer_stub = InferenceServiceStub(self._aio_channel)
            self._aio_health_stub = health_pb2_grpc.HealthStub(self._aio_channel)
        return self._aio_infer_stub, self._aio_health_stub

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        json_payload: Optional[Any] = None,
        timeout: Optional[float] = None,
        model_id: Optional[str] = None,
    ) -> Any:
        """Executes an HTTP JSON request asynchronously without blocking the event loop."""
        return await asyncio.to_thread(
            self._sync_client._request_json,
            method,
            path,
            json_payload=json_payload,
            timeout=timeout,
            model_id=model_id,
        )

    async def infer(
        self,
        model_id: str,
        inputs: InputTypes,
        *,
        use_grpc: Optional[bool] = None,
        timeout: Optional[float] = None,
        client_timeout_ms: Optional[int] = None,
        parameters: Optional[Mapping[str, str]] = None,
    ) -> InferenceResponse:
        """Asynchronously executes unary tensor inference against a registered model."""
        if not model_id or not str(model_id).strip():
            raise InvalidInputError("model_id cannot be empty")
        model_id = str(model_id).strip()

        tensor_inputs = normalize_inputs(inputs)
        dispatch_grpc = self.use_grpc if use_grpc is None else bool(use_grpc)

        if dispatch_grpc:
            infer_stub, _ = self._ensure_aio_grpc_stubs()
            req = build_grpc_infer_request(
                model_id=model_id,
                tensors=tensor_inputs,
                client_timeout_ms=client_timeout_ms,
                parameters=parameters,
            )
            req_timeout = timeout if timeout is not None else self.timeout
            metadata = self._sync_client._grpc_metadata()

            t0 = time.perf_counter()
            try:
                proto_resp = await infer_stub.Infer(
                    req,
                    timeout=req_timeout,
                    metadata=metadata,
                )
            except _GrpcRpcError as exc:
                mapped = map_grpc_error(exc, model_id=model_id)
                logger.warning(
                    "Async gRPC inference failed",
                    extra={
                        "model_id": model_id,
                        "transport": "grpc",
                        "grpc_code": mapped.grpc_code,
                        "error": mapped.message,
                    },
                )
                raise mapped from exc

            client_ms = (time.perf_counter() - t0) * 1000.0
            response = InferenceResponse.from_proto(proto_resp)
            logger.debug(
                "Completed async gRPC inference",
                extra={
                    "model_id": model_id,
                    "transport": "grpc",
                    "server_duration_ms": round(response.duration_ms, 3),
                    "client_latency_ms": round(client_ms, 3),
                },
            )
            return response

        return await asyncio.to_thread(
            self._sync_client._infer_rest,
            model_id,
            tensor_inputs,
            timeout,
        )

    async def list_models(self) -> List[ModelInfo]:
        """Asynchronously lists all registered models."""
        return await self.models.list()

    async def get_model(self, model_id: str) -> ModelInfo:
        """Asynchronously retrieves metadata for a specific model."""
        return await self.models.get(model_id)

    async def load_model(self, model_id: str, *, pin: bool = False) -> Dict[str, Any]:
        """Asynchronously loads a model into memory."""
        return await self.models.load(model_id, pin=pin)

    async def unload_model(self, model_id: str) -> Dict[str, Any]:
        """Asynchronously unloads a model from memory."""
        return await self.models.unload(model_id)

    async def evict_model(self, model_id: str) -> Dict[str, Any]:
        """Asynchronously evicts a model from memory."""
        return await self.models.evict(model_id)

    async def pin_model(self, model_id: str) -> Dict[str, Any]:
        """Asynchronously pins a model in memory."""
        return await self.models.pin(model_id)

    async def unpin_model(self, model_id: str) -> Dict[str, Any]:
        """Asynchronously unpins a model."""
        return await self.models.unpin(model_id)

    async def health_live(self) -> bool:
        """Asynchronously checks the HTTP liveness probe (``GET /health/live``)."""
        return await asyncio.to_thread(self._sync_client.health_live)

    async def health_ready(self) -> bool:
        """Asynchronously checks the HTTP readiness probe (``GET /health/ready``)."""
        return await asyncio.to_thread(self._sync_client.health_ready)

    async def grpc_health(self, service: str = "") -> bool:
        """Asynchronously checks the gRPC health check service."""
        _, health_stub = self._ensure_aio_grpc_stubs()
        req = health_pb2.HealthCheckRequest(service=service)
        try:
            res = await health_stub.Check(req, timeout=self.timeout)
            return bool(res.status == health_pb2.HealthCheckResponse.SERVING)
        except _GrpcRpcError as exc:
            raise map_grpc_error(exc) from exc

    async def service_info(self) -> Dict[str, Any]:
        """Asynchronously retrieves daemon metadata from ``GET /``."""
        return await asyncio.to_thread(self._sync_client.service_info)

    async def metrics(self) -> str:
        """Asynchronously retrieves Prometheus metrics from ``GET /metrics``."""
        return await asyncio.to_thread(self._sync_client.metrics)

    async def upload_model(
        self,
        file: Union[str, Path, bytes, Any],
        *,
        filename: Optional[str] = None,
        model_id: Optional[str] = None,
        description: Optional[str] = None,
        pinned: bool = False,
        max_batch_size: Optional[int] = None,
        max_batch_delay_ms: Optional[int] = None,
        idle_timeout_secs: Optional[int] = None,
        min_residency_secs: Optional[int] = None,
        autoscale_queue_threshold: Optional[int] = None,
        max_replicas: Optional[int] = None,
        routing_strategy: Optional[str] = None,
        device: Optional[str] = None,
    ) -> ModelInfo:
        """Asynchronously uploads and dynamically registers a model via HTTP.

        Convenience alias for :meth:`AsyncModelsManager.upload`.
        """
        return await self.models.upload(
            file=file,
            filename=filename,
            model_id=model_id,
            description=description,
            pinned=pinned,
            max_batch_size=max_batch_size,
            max_batch_delay_ms=max_batch_delay_ms,
            idle_timeout_secs=idle_timeout_secs,
            min_residency_secs=min_residency_secs,
            autoscale_queue_threshold=autoscale_queue_threshold,
            max_replicas=max_replicas,
            routing_strategy=routing_strategy,
            device=device,
        )

    async def refresh_models(self) -> List[ModelInfo]:
        """Asynchronously triggers runtime discovery and refresh of models on the daemon.

        Convenience alias for :meth:`AsyncModelsManager.refresh`.
        """
        return await self.models.refresh()
