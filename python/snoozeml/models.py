"""Model metadata, operational state representations, and lifecycle management for SnoozeML."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import tempfile
import time
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .exceptions import InferenceTimeoutError, ModelNotFoundError
from .logging import get_logger

if TYPE_CHECKING:
    from .client import AsyncSnoozeClient, SnoozeClient

logger = get_logger("models")


class ModelState(str, Enum):
    """Operational lifecycle state of a registered model within SnoozeML."""

    IDLE = "idle"
    """Configuration registered, but weights are deallocated (scaled to zero)."""

    LOADING = "loading"
    """Model weights are currently being loaded into host RAM or device VRAM."""

    READY = "ready"
    """Weights are resident in memory and ready for immediate inference."""

    BUSY = "busy"
    """Inference forward pass is actively executing."""

    @classmethod
    def from_str(cls, value: str) -> "ModelState":
        """Parses a case-insensitive state string into a :class:`ModelState`."""
        normalized = str(value).strip().lower()
        for member in cls:
            if member.value == normalized:
                return member
        return cls.IDLE


@dataclass
class ModelInfo:
    """Metadata and live operational state for a registered model in SnoozeML.

    Attributes:
        id: Unique model identifier.
        state: Operational status (:class:`ModelState`).
        onnx_path: Filesystem or URI path to model weights.
        idle_timeout_secs: Inactivity threshold in seconds before automatic unloading.
        max_batch_size: Maximum batch size for dynamic collation along ``Axis(0)``.
        max_batch_delay_ms: Maximum latency budget in milliseconds for batch accumulation.
        last_accessed_secs_ago: Seconds elapsed since the model last served an inference request.
        description: Optional human-readable description of the model.
        pinned: Whether the model is pinned in memory and exempt from automatic eviction.
        memory_bytes: Optional estimated memory footprint in bytes (0 if not reported).
    """

    id: str
    state: ModelState
    onnx_path: str
    idle_timeout_secs: int = 300
    max_batch_size: int = 16
    max_batch_delay_ms: int = 10
    last_accessed_secs_ago: int = 0
    description: Optional[str] = None
    pinned: bool = False
    memory_bytes: int = 0

    @property
    def is_loaded(self) -> bool:
        """Returns ``True`` if model weights are currently resident in memory (``ready`` or ``busy``)."""
        return self.state in (ModelState.READY, ModelState.BUSY)

    @property
    def path(self) -> str:
        """Alias for :attr:`onnx_path`."""
        return self.onnx_path

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelInfo":
        """Constructs a :class:`ModelInfo` from a JSON dictionary returned by ``GET /v1/models``.

        Args:
            data: Dictionary parsed from the SnoozeML REST API.

        Returns:
            Populated :class:`ModelInfo` instance.
        """
        raw_state = data.get("state", "idle")
        state = (
            raw_state
            if isinstance(raw_state, ModelState)
            else ModelState.from_str(str(raw_state))
        )
        return cls(
            id=str(data.get("id", "")),
            state=state,
            onnx_path=str(data.get("onnx_path", data.get("path", ""))),
            idle_timeout_secs=int(data.get("idle_timeout_secs", 300)),
            max_batch_size=int(data.get("max_batch_size", 16)),
            max_batch_delay_ms=int(data.get("max_batch_delay_ms", 10)),
            last_accessed_secs_ago=int(data.get("last_accessed_secs_ago", 0)),
            description=data.get("description"),
            pinned=bool(data.get("pinned", False)),
            memory_bytes=int(data.get("memory_bytes", 0)),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serializes this :class:`ModelInfo` back into a plain Python dictionary."""
        result: Dict[str, Any] = {
            "id": self.id,
            "state": self.state.value,
            "onnx_path": self.onnx_path,
            "idle_timeout_secs": self.idle_timeout_secs,
            "max_batch_size": self.max_batch_size,
            "max_batch_delay_ms": self.max_batch_delay_ms,
            "last_accessed_secs_ago": self.last_accessed_secs_ago,
            "pinned": self.pinned,
            "memory_bytes": self.memory_bytes,
        }
        if self.description is not None:
            result["description"] = self.description
        return result


class ModelsManager:
    """Synchronous manager for querying and controlling model lifecycles on a SnoozeML daemon.

    Accessed via :attr:`snoozeml.SnoozeClient.models`.
    """

    def __init__(self, client: "SnoozeClient") -> None:
        self._client = client

    def list(self) -> List[ModelInfo]:
        """Lists all registered models and their current lifecycle telemetry.

        Returns:
            List of :class:`ModelInfo` objects.

        Raises:
            SnoozeError: On authentication, permission, or connection errors.
        """
        logger.debug("Listing registered models")
        raw_list = self._client._request_json("GET", "/v1/models")
        if not isinstance(raw_list, list):
            return []
        models = [ModelInfo.from_dict(item) for item in raw_list if isinstance(item, Mapping)]
        logger.debug("Retrieved model list", extra={"count": len(models)})
        return models

    def get(self, model_id: str) -> ModelInfo:
        """Retrieves metadata and operational state for a specific model.

        Args:
            model_id: Unique identifier of the model to inspect.

        Returns:
            Matching :class:`ModelInfo` instance.

        Raises:
            ModelNotFoundError: If ``model_id`` is not registered on the server.
            SnoozeError: On authentication or transport failures.
        """
        if not model_id or not model_id.strip():
            raise ModelNotFoundError("Model ID cannot be empty", status_code=404, model_id=model_id)

        models = self.list()
        for info in models:
            if info.id == model_id:
                return info

        logger.warning("Model not found in registry", extra={"model_id": model_id})
        raise ModelNotFoundError(
            f"Model not found: {model_id}",
            status_code=404,
            model_id=model_id,
        )

    def load(self, model_id: str, *, pin: bool = False) -> Dict[str, Any]:
        """Eagerly loads a model's weights into memory ahead of traffic.

        Args:
            model_id: Target model identifier.
            pin: If ``True``, also pins the model in memory so it is exempt from
                automatic scale-to-zero and LRU eviction.

        Returns:
            Status dictionary returned by the server.
        """
        if pin:
            return self.pin(model_id)

        logger.info("Loading model into memory", extra={"model_id": model_id})
        resp = self._client._request_json(
            "POST", f"/v1/models/{model_id}/load", model_id=model_id
        )
        return dict(resp) if isinstance(resp, Mapping) else {"model_id": model_id, "status": "loaded"}

    def unload(self, model_id: str) -> Dict[str, Any]:
        """Explicitly unloads (deallocates) a model from memory (scale-to-zero).

        Args:
            model_id: Target model identifier.

        Returns:
            Status dictionary returned by the server.
        """
        logger.info("Unloading model from memory", extra={"model_id": model_id})
        resp = self._client._request_json(
            "POST", f"/v1/models/{model_id}/unload", model_id=model_id
        )
        return dict(resp) if isinstance(resp, Mapping) else {"model_id": model_id, "status": "unloaded"}

    def evict(self, model_id: str) -> Dict[str, Any]:
        """Alias for :meth:`unload` — explicitly evicts/snoozes a model from memory.

        Args:
            model_id: Target model identifier.

        Returns:
            Status dictionary returned by the server.
        """
        return self.unload(model_id)

    def pin(self, model_id: str) -> Dict[str, Any]:
        """Pins a model in memory so it is eagerly loaded and exempt from automatic eviction.

        Args:
            model_id: Target model identifier.

        Returns:
            Status dictionary returned by the server (e.g. ``{"model_id": ..., "status": "pinned", "pinned": True}``).
        """
        logger.info("Pinning model in memory", extra={"model_id": model_id})
        resp = self._client._request_json(
            "POST", f"/v1/models/{model_id}/pin", model_id=model_id
        )
        return (
            dict(resp)
            if isinstance(resp, Mapping)
            else {"model_id": model_id, "status": "pinned", "pinned": True}
        )

    def unpin(self, model_id: str) -> Dict[str, Any]:
        """Unpins a model so it returns to normal scale-to-zero and LRU lifecycle management.

        Args:
            model_id: Target model identifier.

        Returns:
            Status dictionary returned by the server (e.g. ``{"model_id": ..., "status": "unpinned", "pinned": False}``).
        """
        logger.info("Unpinning model", extra={"model_id": model_id})
        resp = self._client._request_json(
            "POST", f"/v1/models/{model_id}/unpin", model_id=model_id
        )
        return (
            dict(resp)
            if isinstance(resp, Mapping)
            else {"model_id": model_id, "status": "unpinned", "pinned": False}
        )

    def refresh(self) -> List[ModelInfo]:
        """Triggers runtime discovery and refresh of models on the SnoozeML daemon via HTTP.

        Scans the daemon's configured models directory for newly added or modified models.

        Returns:
            Updated list of :class:`ModelInfo` objects registered on the server.
        """
        logger.info("Triggering runtime refresh of models via HTTP")
        resp = self._client._request_raw("POST", "/v1/models/refresh")
        data = resp.json()
        raw_list = data.get("models", []) if isinstance(data, Mapping) else []
        models = [ModelInfo.from_dict(item) for item in raw_list if isinstance(item, Mapping)]
        logger.info("Models directory refreshed", extra={"count": len(models)})
        return models

    def upload(
        self,
        file: Union[str, Path, bytes, Any],
        *,
        filename: Optional[str] = None,
        model_id: Optional[str] = None,
        description: Optional[str] = None,
        pinned: bool = False,
        overwrite: bool = False,
        max_batch_size: Optional[int] = None,
        max_batch_delay_ms: Optional[int] = None,
        idle_timeout_secs: Optional[int] = None,
        min_residency_secs: Optional[int] = None,
        autoscale_queue_threshold: Optional[int] = None,
        max_replicas: Optional[int] = None,
        routing_strategy: Optional[str] = None,
        device: Optional[str] = None,
    ) -> ModelInfo:
        """Uploads a model file directly to the SnoozeML daemon via HTTP.

        Supports passing custom runtime configuration (pinned memory, replica limits,
        batch sizes, idle timeouts, overwrite) without needing to mount local filesystem volumes.

        Args:
            file: Filesystem path (str or Path), raw bytes, or readable binary stream.
            filename: Optional explicit filename (e.g. "iris_classifier.onnx").
            model_id: Optional unique identifier for the model (defaults to file stem).
            description: Optional custom human-readable description.
            pinned: If True, eagerly loads model into memory and exempts it from scale-to-zero.
            overwrite: If True, replaces an existing model weight file on the server if present.
            max_batch_size: Maximum batch size for dynamic Axis(0) micro-batching.
            max_batch_delay_ms: Maximum latency budget in ms before executing micro-batch.
            idle_timeout_secs: Inactivity threshold in seconds before scale-to-zero unload.
            min_residency_secs: Minimum duration in seconds model stays resident before LRU eviction.
            autoscale_queue_threshold: Queue backlog threshold to trigger replica autoscale.
            max_replicas: Upper limit on parallel replica instances.
            routing_strategy: Replica routing strategy ("least_queue", "round_robin", "random").
            device: Hardware execution provider ("cpu", "cuda", "auto").

        Returns:
            :class:`ModelInfo` representing the newly registered model.
        """
        if isinstance(file, (str, Path)):
            path = Path(file)
            actual_filename = filename or path.name
            with open(path, "rb") as f:
                content = f.read()
        elif isinstance(file, bytes):
            actual_filename = filename or (f"{model_id}.onnx" if model_id else "model.onnx")
            content = file
        elif hasattr(file, "read"):
            actual_filename = filename or (f"{model_id}.onnx" if model_id else "model.onnx")
            content = file.read()
        else:
            raise TypeError(f"Unsupported file type: {type(file).__name__}")

        files = {"file": (actual_filename, content, "application/octet-stream")}
        data: Dict[str, str] = {}
        if model_id is not None:
            data["model_id"] = str(model_id)
        if description is not None:
            data["description"] = str(description)
        if pinned:
            data["pinned"] = "true"
        if overwrite:
            data["overwrite"] = "true"
        if max_batch_size is not None:
            data["max_batch_size"] = str(max_batch_size)
        if max_batch_delay_ms is not None:
            data["max_batch_delay_ms"] = str(max_batch_delay_ms)
        if idle_timeout_secs is not None:
            data["idle_timeout_secs"] = str(idle_timeout_secs)
        if min_residency_secs is not None:
            data["min_residency_secs"] = str(min_residency_secs)
        if autoscale_queue_threshold is not None:
            data["autoscale_queue_threshold"] = str(autoscale_queue_threshold)
        if max_replicas is not None:
            data["max_replicas"] = str(max_replicas)
        if routing_strategy is not None:
            data["routing_strategy"] = str(routing_strategy)
        if device is not None:
            data["device"] = str(device)

        logger.info(
            "Uploading model to SnoozeML via HTTP",
            extra={
                "filename": actual_filename,
                "model_id": model_id,
                "size_bytes": len(content),
                "pinned": pinned,
                "overwrite": overwrite,
            },
        )
        resp = self._client._request_multipart("/v1/models/upload", files=files, data=data, model_id=model_id)
        model_meta = ModelInfo.from_dict(resp.json())
        logger.info("Successfully uploaded and registered model", extra={"model_id": model_meta.id})
        return model_meta

    def wait_for_ready(
        self,
        model_id: str,
        *,
        timeout: float = 30.0,
        poll_interval: float = 0.2,
        trigger_load: bool = False,
    ) -> ModelInfo:
        """Polls model metadata until the model reaches a loaded state (:attr:`ModelState.READY` or :attr:`ModelState.BUSY`).

        Args:
            model_id: Target model identifier.
            timeout: Maximum time in seconds to wait before raising :class:`InferenceTimeoutError`.
            poll_interval: Interval in seconds between status polls.
            trigger_load: If ``True`` and the model is currently ``idle``, triggers :meth:`load` first.

        Returns:
            Loaded :class:`ModelInfo` instance.
        """
        if trigger_load:
            info = self.get(model_id)
            if not info.is_loaded:
                self.load(model_id)

        deadline = time.monotonic() + float(timeout)
        while True:
            info = self.get(model_id)
            if info.is_loaded:
                return info
            if time.monotonic() >= deadline:
                raise InferenceTimeoutError(
                    f"Timed out waiting {timeout}s for model '{model_id}' to become ready (current state: {info.state.value})",
                    status_code=504,
                    model_id=model_id,
                )
            time.sleep(max(0.01, float(poll_interval)))

    def deploy_sklearn(
        self,
        model: Any,
        *,
        model_id: str,
        initial_types: Optional[Sequence[Tuple[str, Any]]] = None,
        sample_input: Optional[Any] = None,
        input_name: str = "float_input",
        output_name: str = "output",
        opset_version: int = 17,
        overwrite: bool = True,
        pinned: bool = False,
        description: Optional[str] = None,
        max_batch_size: Optional[int] = None,
        max_batch_delay_ms: Optional[int] = None,
        idle_timeout_secs: Optional[int] = None,
        min_residency_secs: Optional[int] = None,
        autoscale_queue_threshold: Optional[int] = None,
        max_replicas: Optional[int] = None,
        routing_strategy: Optional[str] = None,
        device: Optional[str] = None,
    ) -> ModelInfo:
        """Converts a fitted scikit-learn estimator or ``Pipeline`` to ONNX and deploys it to SnoozeML in one call."""
        from .converters.sklearn import export_sklearn

        with tempfile.TemporaryDirectory(prefix="snoozeml_sklearn_") as tmpdir:
            onnx_file = Path(tmpdir) / f"{model_id}.onnx"
            export_sklearn(
                model=model,
                initial_types=initial_types,
                output_path=onnx_file,
                sample_input=sample_input,
                input_name=input_name,
                output_name=output_name,
                opset_version=opset_version,
            )
            return self.upload(
                file=onnx_file,
                filename=f"{model_id}.onnx",
                model_id=model_id,
                description=description,
                pinned=pinned,
                overwrite=overwrite,
                max_batch_size=max_batch_size,
                max_batch_delay_ms=max_batch_delay_ms,
                idle_timeout_secs=idle_timeout_secs,
                min_residency_secs=min_residency_secs,
                autoscale_queue_threshold=autoscale_queue_threshold,
                max_replicas=max_replicas,
                routing_strategy=routing_strategy,
                device=device,
            )

    def deploy_torch(
        self,
        model: Any,
        sample_input: Any,
        *,
        model_id: str,
        input_names: Optional[Sequence[str]] = None,
        output_names: Optional[Sequence[str]] = None,
        dynamic_axes: Optional[Mapping[str, Mapping[int, str]]] = None,
        opset_version: int = 17,
        overwrite: bool = True,
        pinned: bool = False,
        description: Optional[str] = None,
        max_batch_size: Optional[int] = None,
        max_batch_delay_ms: Optional[int] = None,
        idle_timeout_secs: Optional[int] = None,
        min_residency_secs: Optional[int] = None,
        autoscale_queue_threshold: Optional[int] = None,
        max_replicas: Optional[int] = None,
        routing_strategy: Optional[str] = None,
        device: Optional[str] = None,
    ) -> ModelInfo:
        """Exports a PyTorch ``nn.Module`` to ONNX and deploys it to SnoozeML in one call."""
        from .converters.torch import export_torch

        with tempfile.TemporaryDirectory(prefix="snoozeml_torch_") as tmpdir:
            onnx_file = Path(tmpdir) / f"{model_id}.onnx"
            export_torch(
                model=model,
                sample_input=sample_input,
                output_path=onnx_file,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=opset_version,
            )
            return self.upload(
                file=onnx_file,
                filename=f"{model_id}.onnx",
                model_id=model_id,
                description=description,
                pinned=pinned,
                overwrite=overwrite,
                max_batch_size=max_batch_size,
                max_batch_delay_ms=max_batch_delay_ms,
                idle_timeout_secs=idle_timeout_secs,
                min_residency_secs=min_residency_secs,
                autoscale_queue_threshold=autoscale_queue_threshold,
                max_replicas=max_replicas,
                routing_strategy=routing_strategy,
                device=device,
            )

    def deploy_xgboost(
        self,
        model: Any,
        *,
        model_id: str,
        overwrite: bool = True,
        pinned: bool = False,
        description: Optional[str] = None,
        max_batch_size: Optional[int] = None,
        max_batch_delay_ms: Optional[int] = None,
        idle_timeout_secs: Optional[int] = None,
        min_residency_secs: Optional[int] = None,
        autoscale_queue_threshold: Optional[int] = None,
        max_replicas: Optional[int] = None,
        routing_strategy: Optional[str] = None,
        device: Optional[str] = None,
    ) -> ModelInfo:
        """Exports an XGBoost model to ``.xgb.json`` and deploys it to SnoozeML in one call."""
        from .converters.xgboost import export_xgboost

        with tempfile.TemporaryDirectory(prefix="snoozeml_xgb_") as tmpdir:
            xgb_file = Path(tmpdir) / f"{model_id}.xgb.json"
            export_xgboost(model=model, output_path=xgb_file)
            return self.upload(
                file=xgb_file,
                filename=f"{model_id}.xgb.json",
                model_id=model_id,
                description=description,
                pinned=pinned,
                overwrite=overwrite,
                max_batch_size=max_batch_size,
                max_batch_delay_ms=max_batch_delay_ms,
                idle_timeout_secs=idle_timeout_secs,
                min_residency_secs=min_residency_secs,
                autoscale_queue_threshold=autoscale_queue_threshold,
                max_replicas=max_replicas,
                routing_strategy=routing_strategy,
                device=device,
            )


class AsyncModelsManager:
    """Asynchronous manager for querying and controlling model lifecycles on a SnoozeML daemon.

    Accessed via :attr:`snoozeml.AsyncSnoozeClient.models`.
    """

    def __init__(self, client: "AsyncSnoozeClient") -> None:
        self._client = client

    async def list(self) -> List[ModelInfo]:
        """Asynchronously lists all registered models and their current telemetry."""
        logger.debug("Listing registered models (async)")
        raw_list = await self._client._request_json("GET", "/v1/models")
        if not isinstance(raw_list, list):
            return []
        models = [ModelInfo.from_dict(item) for item in raw_list if isinstance(item, Mapping)]
        logger.debug("Retrieved model list (async)", extra={"count": len(models)})
        return models

    async def get(self, model_id: str) -> ModelInfo:
        """Asynchronously retrieves metadata and operational state for a specific model."""
        if not model_id or not model_id.strip():
            raise ModelNotFoundError("Model ID cannot be empty", status_code=404, model_id=model_id)

        models = await self.list()
        for info in models:
            if info.id == model_id:
                return info

        logger.warning("Model not found in registry (async)", extra={"model_id": model_id})
        raise ModelNotFoundError(
            f"Model not found: {model_id}",
            status_code=404,
            model_id=model_id,
        )

    async def load(self, model_id: str, *, pin: bool = False) -> Dict[str, Any]:
        """Asynchronously loads a model's weights into memory."""
        if pin:
            return await self.pin(model_id)

        logger.info("Loading model into memory (async)", extra={"model_id": model_id})
        resp = await self._client._request_json(
            "POST", f"/v1/models/{model_id}/load", model_id=model_id
        )
        return dict(resp) if isinstance(resp, Mapping) else {"model_id": model_id, "status": "loaded"}

    async def unload(self, model_id: str) -> Dict[str, Any]:
        """Asynchronously unloads a model from memory."""
        logger.info("Unloading model from memory (async)", extra={"model_id": model_id})
        resp = await self._client._request_json(
            "POST", f"/v1/models/{model_id}/unload", model_id=model_id
        )
        return dict(resp) if isinstance(resp, Mapping) else {"model_id": model_id, "status": "unloaded"}

    async def evict(self, model_id: str) -> Dict[str, Any]:
        """Alias for :meth:`unload` — asynchronously evicts/snoozes a model from memory."""
        return await self.unload(model_id)

    async def pin(self, model_id: str) -> Dict[str, Any]:
        """Asynchronously pins a model in memory."""
        logger.info("Pinning model in memory (async)", extra={"model_id": model_id})
        resp = await self._client._request_json(
            "POST", f"/v1/models/{model_id}/pin", model_id=model_id
        )
        return (
            dict(resp)
            if isinstance(resp, Mapping)
            else {"model_id": model_id, "status": "pinned", "pinned": True}
        )

    async def unpin(self, model_id: str) -> Dict[str, Any]:
        """Asynchronously unpins a model."""
        logger.info("Unpinning model (async)", extra={"model_id": model_id})
        resp = await self._client._request_json(
            "POST", f"/v1/models/{model_id}/unpin", model_id=model_id
        )
        return (
            dict(resp)
            if isinstance(resp, Mapping)
            else {"model_id": model_id, "status": "unpinned", "pinned": False}
        )

    async def refresh(self) -> List[ModelInfo]:
        """Asynchronously triggers runtime discovery and refresh of models on the daemon.

        Scans the daemon's configured models directory for newly added or modified models.

        Returns:
            Updated list of :class:`ModelInfo` objects registered on the server.
        """
        return await asyncio.to_thread(self._client._sync_client.models.refresh)

    async def upload(
        self,
        file: Union[str, Path, bytes, Any],
        *,
        filename: Optional[str] = None,
        model_id: Optional[str] = None,
        description: Optional[str] = None,
        pinned: bool = False,
        overwrite: bool = False,
        max_batch_size: Optional[int] = None,
        max_batch_delay_ms: Optional[int] = None,
        idle_timeout_secs: Optional[int] = None,
        min_residency_secs: Optional[int] = None,
        autoscale_queue_threshold: Optional[int] = None,
        max_replicas: Optional[int] = None,
        routing_strategy: Optional[str] = None,
        device: Optional[str] = None,
    ) -> ModelInfo:
        """Asynchronously uploads a model file directly to the daemon via HTTP.

        Supports passing custom runtime configuration (pinned memory, replica limits,
        batch sizes, idle timeouts, overwrite) without needing to mount local filesystem volumes.

        Args:
            file: Filesystem path (str or Path), raw bytes, or readable binary stream.
            filename: Optional explicit filename (e.g. "iris_classifier.onnx").
            model_id: Optional unique identifier for the model (defaults to file stem).
            description: Optional custom human-readable description.
            pinned: If True, eagerly loads model into memory and exempts it from scale-to-zero.
            overwrite: If True, replaces an existing model weight file on the server if present.
            max_batch_size: Maximum batch size for dynamic Axis(0) micro-batching.
            max_batch_delay_ms: Maximum latency budget in ms before executing micro-batch.
            idle_timeout_secs: Inactivity threshold in seconds before scale-to-zero unload.
            min_residency_secs: Minimum duration in seconds model stays resident before LRU eviction.
            autoscale_queue_threshold: Queue backlog threshold to trigger replica autoscale.
            max_replicas: Upper limit on parallel replica instances.
            routing_strategy: Replica routing strategy ("least_queue", "round_robin", "random").
            device: Hardware execution provider ("cpu", "cuda", "auto").

        Returns:
            :class:`ModelInfo` representing the newly registered model.
        """
        return await asyncio.to_thread(
            self._client._sync_client.models.upload,
            file,
            filename=filename,
            model_id=model_id,
            description=description,
            pinned=pinned,
            overwrite=overwrite,
            max_batch_size=max_batch_size,
            max_batch_delay_ms=max_batch_delay_ms,
            idle_timeout_secs=idle_timeout_secs,
            min_residency_secs=min_residency_secs,
            autoscale_queue_threshold=autoscale_queue_threshold,
            max_replicas=max_replicas,
            routing_strategy=routing_strategy,
            device=device,
        )

    async def wait_for_ready(
        self,
        model_id: str,
        *,
        timeout: float = 30.0,
        poll_interval: float = 0.2,
        trigger_load: bool = False,
    ) -> ModelInfo:
        """Asynchronously polls model metadata until the model reaches a loaded state."""
        if trigger_load:
            info = await self.get(model_id)
            if not info.is_loaded:
                await self.load(model_id)

        deadline = time.monotonic() + float(timeout)
        while True:
            info = await self.get(model_id)
            if info.is_loaded:
                return info
            if time.monotonic() >= deadline:
                raise InferenceTimeoutError(
                    f"Timed out waiting {timeout}s for model '{model_id}' to become ready (current state: {info.state.value})",
                    status_code=504,
                    model_id=model_id,
                )
            await asyncio.sleep(max(0.01, float(poll_interval)))

    async def deploy_sklearn(
        self,
        model: Any,
        *,
        model_id: str,
        initial_types: Optional[Sequence[Tuple[str, Any]]] = None,
        sample_input: Optional[Any] = None,
        input_name: str = "float_input",
        output_name: str = "output",
        opset_version: int = 17,
        overwrite: bool = True,
        pinned: bool = False,
        description: Optional[str] = None,
        max_batch_size: Optional[int] = None,
        max_batch_delay_ms: Optional[int] = None,
        idle_timeout_secs: Optional[int] = None,
        min_residency_secs: Optional[int] = None,
        autoscale_queue_threshold: Optional[int] = None,
        max_replicas: Optional[int] = None,
        routing_strategy: Optional[str] = None,
        device: Optional[str] = None,
    ) -> ModelInfo:
        """Asynchronously converts a fitted scikit-learn estimator or ``Pipeline`` and deploys it."""
        return await asyncio.to_thread(
            self._client._sync_client.models.deploy_sklearn,
            model,
            model_id=model_id,
            initial_types=initial_types,
            sample_input=sample_input,
            input_name=input_name,
            output_name=output_name,
            opset_version=opset_version,
            overwrite=overwrite,
            pinned=pinned,
            description=description,
            max_batch_size=max_batch_size,
            max_batch_delay_ms=max_batch_delay_ms,
            idle_timeout_secs=idle_timeout_secs,
            min_residency_secs=min_residency_secs,
            autoscale_queue_threshold=autoscale_queue_threshold,
            max_replicas=max_replicas,
            routing_strategy=routing_strategy,
            device=device,
        )

    async def deploy_torch(
        self,
        model: Any,
        sample_input: Any,
        *,
        model_id: str,
        input_names: Optional[Sequence[str]] = None,
        output_names: Optional[Sequence[str]] = None,
        dynamic_axes: Optional[Mapping[str, Mapping[int, str]]] = None,
        opset_version: int = 17,
        overwrite: bool = True,
        pinned: bool = False,
        description: Optional[str] = None,
        max_batch_size: Optional[int] = None,
        max_batch_delay_ms: Optional[int] = None,
        idle_timeout_secs: Optional[int] = None,
        min_residency_secs: Optional[int] = None,
        autoscale_queue_threshold: Optional[int] = None,
        max_replicas: Optional[int] = None,
        routing_strategy: Optional[str] = None,
        device: Optional[str] = None,
    ) -> ModelInfo:
        """Asynchronously exports a PyTorch ``nn.Module`` to ONNX and deploys it."""
        return await asyncio.to_thread(
            self._client._sync_client.models.deploy_torch,
            model,
            sample_input,
            model_id=model_id,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            overwrite=overwrite,
            pinned=pinned,
            description=description,
            max_batch_size=max_batch_size,
            max_batch_delay_ms=max_batch_delay_ms,
            idle_timeout_secs=idle_timeout_secs,
            min_residency_secs=min_residency_secs,
            autoscale_queue_threshold=autoscale_queue_threshold,
            max_replicas=max_replicas,
            routing_strategy=routing_strategy,
            device=device,
        )

    async def deploy_xgboost(
        self,
        model: Any,
        *,
        model_id: str,
        overwrite: bool = True,
        pinned: bool = False,
        description: Optional[str] = None,
        max_batch_size: Optional[int] = None,
        max_batch_delay_ms: Optional[int] = None,
        idle_timeout_secs: Optional[int] = None,
        min_residency_secs: Optional[int] = None,
        autoscale_queue_threshold: Optional[int] = None,
        max_replicas: Optional[int] = None,
        routing_strategy: Optional[str] = None,
        device: Optional[str] = None,
    ) -> ModelInfo:
        """Asynchronously exports an XGBoost model to ``.xgb.json`` and deploys it."""
        return await asyncio.to_thread(
            self._client._sync_client.models.deploy_xgboost,
            model,
            model_id=model_id,
            overwrite=overwrite,
            pinned=pinned,
            description=description,
            max_batch_size=max_batch_size,
            max_batch_delay_ms=max_batch_delay_ms,
            idle_timeout_secs=idle_timeout_secs,
            min_residency_secs=min_residency_secs,
            autoscale_queue_threshold=autoscale_queue_threshold,
            max_replicas=max_replicas,
            routing_strategy=routing_strategy,
            device=device,
        )
