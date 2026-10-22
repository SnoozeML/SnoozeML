"""Event streaming message envelopes, wire serializers, and broker clients for SnoozeML.

Maintains strict binary and JSON wire parity with SnoozeML's Rust streaming engine
(``src/stream/message.rs``, ``src/stream/redis_stream.rs``, ``src/stream/kafka.rs``,
and ``src/stream/rabbitmq.rs``).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import json
import time
from typing import Any, Callable, Deque, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import uuid

try:
    import numpy as np  # type: ignore
    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    _HAS_NUMPY = False

try:
    from ._proto import DataType as ProtoDataType
    from ._proto import ModelInferRequest, ModelInferResponse, TensorData
    _HAS_PROTO = True
except ImportError:
    ProtoDataType = None  # type: ignore
    ModelInferRequest = None  # type: ignore
    ModelInferResponse = None  # type: ignore
    TensorData = None  # type: ignore
    _HAS_PROTO = False

from .exceptions import (
    InferenceTimeoutError,
    InvalidInputError,
    RuntimeInferenceError,
    SnoozeConnectionError,
    SnoozeError,
)
from .inference import (
    InferenceResponse,
    InputTypes,
    OutputList,
    TensorInput,
    TensorOutput,
    _from_f32_le_bytes,
    normalize_inputs,
)
from .logging import get_logger

logger = get_logger("streams")


class StreamWireFormat(str, Enum):
    """Supported payload serialization formats across SnoozeML streaming transports."""

    JSON = "json"
    """Standard JSON serialization matching ``StreamInferRequest`` and ``StreamInferResponse``."""

    PROTOBUF = "protobuf"
    """Binary Protobuf serialization matching ``snoozeml.v1.ModelInferRequest`` and ``ModelInferResponse``."""

    @classmethod
    def from_str(cls, value: Union["StreamWireFormat", str]) -> "StreamWireFormat":
        """Parses a case-insensitive wire format string into a :class:`StreamWireFormat`."""
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower()
        for member in cls:
            if member.value == normalized:
                return member
        raise InvalidInputError(
            f"Unsupported stream wire format '{value}'. Expected 'json' or 'protobuf'."
        )


@dataclass
class StreamInferRequest:
    """Outgoing streaming inference request envelope matching ``src/stream/message.rs``.

    Attributes:
        inputs: Validated list of :class:`~snoozeml.inference.TensorInput` tensors.
        model_id: Optional target model identifier (overrides the stream binding's default model).
        id: Correlation identifier used to match responses to requests (auto-generated UUID if omitted).
        reply_to: Optional dynamic destination topic, queue, or stream key for routing the response.
    """

    inputs: List[TensorInput]
    model_id: Optional[str] = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    reply_to: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.inputs:
            raise InvalidInputError("StreamInferRequest inputs cannot be empty")
        if not self.id or not str(self.id).strip():
            self.id = uuid.uuid4().hex
        else:
            self.id = str(self.id).strip()
        if self.model_id is not None:
            cleaned_model = str(self.model_id).strip()
            self.model_id = cleaned_model if cleaned_model else None
        if self.reply_to is not None:
            cleaned_reply = str(self.reply_to).strip()
            self.reply_to = cleaned_reply if cleaned_reply else None

    @classmethod
    def from_inputs(
        cls,
        inputs: InputTypes,
        *,
        model_id: Optional[str] = None,
        id: Optional[str] = None,
        reply_to: Optional[str] = None,
        default_input_name: str = "input",
    ) -> "StreamInferRequest":
        """Constructs a :class:`StreamInferRequest` from NumPy arrays, PyTorch tensors, or dicts.

        Args:
            inputs: Input tensors in any format accepted by :func:`~snoozeml.inference.normalize_inputs`.
            model_id: Optional target model identifier.
            id: Optional explicit correlation ID (auto-generated if ``None``).
            reply_to: Optional dynamic egress topic/queue override.
            default_input_name: Binding name when a bare array is provided.

        Returns:
            Validated :class:`StreamInferRequest`.
        """
        normalized = normalize_inputs(inputs, default_input_name=default_input_name)
        corr_id = str(id).strip() if id is not None and str(id).strip() else uuid.uuid4().hex
        return cls(
            inputs=normalized,
            model_id=model_id,
            id=corr_id,
            reply_to=reply_to,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the request envelope into a JSON-compatible dictionary."""
        result: Dict[str, Any] = {
            "id": self.id,
            "inputs": [t.to_rest_dict() for t in self.inputs],
        }
        if self.model_id is not None:
            result["model_id"] = self.model_id
        if self.reply_to is not None:
            result["reply_to"] = self.reply_to
        return result

    def to_bytes(
        self,
        format: Union[StreamWireFormat, str] = StreamWireFormat.JSON,
    ) -> bytes:
        """Encodes the request envelope into raw wire bytes for publishing to a broker.

        Args:
            format: Serialization format (:attr:`StreamWireFormat.JSON` or :attr:`StreamWireFormat.PROTOBUF`).

        Returns:
            Encoded byte payload ready for Kafka, RabbitMQ, or Redis Streams.
        """
        wire_fmt = StreamWireFormat.from_str(format)
        if wire_fmt == StreamWireFormat.JSON:
            return json.dumps(self.to_dict(), separators=(",", ":"), allow_nan=False).encode("utf-8")

        if not _HAS_PROTO or ModelInferRequest is None:
            raise InvalidInputError(
                "Protobuf stream serialization requires 'protobuf'. Install with: pip install 'snoozeml[grpc]'"
            )
        proto_req = ModelInferRequest(
            model_id=self.model_id or "",
            inputs=[t.to_proto() for t in self.inputs],
        )
        if self.id:
            proto_req.parameters["id"] = self.id
        if self.reply_to:
            proto_req.parameters["reply_to"] = self.reply_to
        return bytes(proto_req.SerializeToString())

    @classmethod
    def from_bytes(
        cls,
        payload: Union[bytes, bytearray, memoryview, str],
        format: Union[StreamWireFormat, str] = StreamWireFormat.JSON,
    ) -> "StreamInferRequest":
        """Deserializes a :class:`StreamInferRequest` from raw broker bytes.

        Args:
            payload: Raw bytes or UTF-8 string read from a stream broker.
            format: Wire format (:attr:`StreamWireFormat.JSON` or :attr:`StreamWireFormat.PROTOBUF`).

        Returns:
            Decoded :class:`StreamInferRequest`.
        """
        wire_fmt = StreamWireFormat.from_str(format)
        raw_bytes = payload.encode("utf-8") if isinstance(payload, str) else bytes(payload)

        if wire_fmt == StreamWireFormat.JSON:
            try:
                data = json.loads(raw_bytes.decode("utf-8"))
            except Exception as exc:
                raise InvalidInputError(f"Failed to parse stream JSON request: {exc}") from exc
            if not isinstance(data, Mapping):
                raise InvalidInputError("Stream JSON request payload must be an object")
            raw_inputs = data.get("inputs", [])
            if not isinstance(raw_inputs, Sequence) or not raw_inputs:
                raise InvalidInputError("Stream JSON request is missing non-empty 'inputs' list")
            tensors = [TensorInput.from_dict(item) for item in raw_inputs]
            return cls(
                inputs=tensors,
                model_id=data.get("model_id"),
                id=str(data.get("id") or uuid.uuid4().hex),
                reply_to=data.get("reply_to"),
            )

        if not _HAS_PROTO or ModelInferRequest is None:
            raise InvalidInputError(
                "Protobuf stream deserialization requires 'protobuf'. Install with: pip install 'snoozeml[grpc]'"
            )
        proto_req = ModelInferRequest()
        try:
            proto_req.ParseFromString(raw_bytes)
        except Exception as exc:
            raise InvalidInputError(f"Failed to decode protobuf stream request: {exc}") from exc

        tensors = []
        for pt in proto_req.inputs:
            shape = [int(d) for d in pt.shape]
            arr = _from_f32_le_bytes(pt.raw_data, shape)
            tensors.append(TensorInput(name=str(pt.name), shape=shape, data=arr))

        params = dict(proto_req.parameters)
        corr_id = params.get("id") or params.get("correlation_id") or uuid.uuid4().hex
        reply_to = params.get("reply_to")
        model_id = str(proto_req.model_id) if proto_req.model_id else None
        return cls(
            inputs=tensors,
            model_id=model_id,
            id=corr_id,
            reply_to=reply_to,
        )


@dataclass
class StreamInferResponse:
    """Incoming streaming inference response envelope published by SnoozeML to output/DLQ topics.

    Attributes:
        id: Correlation ID matching the original :class:`StreamInferRequest`.
        model_id: Identifier of the model that executed the request.
        status: Execution status (``"success"`` or ``"error"``).
        outputs: Output tensors wrapped in an :class:`~snoozeml.inference.OutputList`.
        error: Diagnostic error message when ``status == "error"``.
        duration_ms: Server-side execution duration in milliseconds.
    """

    id: str
    model_id: str
    status: str = "success"
    outputs: List[TensorOutput] = field(default_factory=list)
    error: Optional[str] = None
    duration_ms: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.outputs, OutputList):
            self.outputs = OutputList(self.outputs)

    @property
    def is_success(self) -> bool:
        """Returns ``True`` if the stream inference succeeded (``status == "success"``)."""
        return self.status.lower() == "success" and self.error is None

    def raise_for_status(self) -> None:
        """Raises a :class:`~snoozeml.exceptions.RuntimeInferenceError` if the response represents an error."""
        if not self.is_success:
            msg = self.error or f"Stream inference failed with status '{self.status}'"
            raise RuntimeInferenceError(msg, model_id=self.model_id)

    def to_inference_response(self) -> InferenceResponse:
        """Converts a successful stream response into a standard :class:`~snoozeml.inference.InferenceResponse`.

        Raises:
            RuntimeInferenceError: If :attr:`is_success` is ``False``.
        """
        self.raise_for_status()
        return InferenceResponse(
            model_id=self.model_id,
            outputs=list(self.outputs),
            duration_ms=self.duration_ms,
        )

    def get_tensor(self, name: Optional[str] = None) -> TensorOutput:
        """Retrieves an output :class:`~snoozeml.inference.TensorOutput` by name (or the first output)."""
        return self.to_inference_response().get_tensor(name)

    def get_numpy(self, name: Optional[str] = None) -> np.ndarray:
        """Returns the output tensor as a reshaped ``np.ndarray`` (``dtype=np.float32``)."""
        return self.to_inference_response().get_numpy(name)

    def get_torch(self, name: Optional[str] = None, device: Optional[str] = None) -> Any:
        """Returns the output tensor as a ``torch.Tensor``."""
        return self.to_inference_response().get_torch(name, device=device)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the response envelope into a JSON-compatible dictionary."""
        result: Dict[str, Any] = {
            "id": self.id,
            "model_id": self.model_id,
            "status": self.status,
            "duration_ms": self.duration_ms,
        }
        if self.outputs:
            result["outputs"] = [out.to_dict() for out in self.outputs]
        if self.error is not None:
            result["error"] = self.error
        return result

    def to_bytes(
        self,
        format: Union[StreamWireFormat, str] = StreamWireFormat.JSON,
    ) -> bytes:
        """Encodes this response envelope into raw bytes matching ``src/stream/message.rs``."""
        wire_fmt = StreamWireFormat.from_str(format)
        if wire_fmt == StreamWireFormat.JSON:
            return json.dumps(self.to_dict(), separators=(",", ":"), allow_nan=False).encode("utf-8")

        if not _HAS_PROTO or ModelInferResponse is None or TensorData is None or ProtoDataType is None:
            raise InvalidInputError(
                "Protobuf stream serialization requires 'protobuf'. Install with: pip install 'snoozeml[grpc]'"
            )
        proto_outputs = []
        for out in self.outputs:
            if _HAS_NUMPY and np is not None:
                le_bytes = np.ascontiguousarray(out.data, dtype="<f4").tobytes()
            else:
                le_bytes = out.data.tobytes()
            proto_outputs.append(
                TensorData(
                    name=out.name,
                    shape=list(out.shape),
                    data_type=ProtoDataType.DATA_TYPE_FP32,
                    raw_data=le_bytes,
                )
            )
        proto_resp = ModelInferResponse(
            model_id=self.model_id,
            outputs=proto_outputs,
            duration_ms=self.duration_ms,
        )
        return bytes(proto_resp.SerializeToString())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StreamInferResponse":
        """Constructs a :class:`StreamInferResponse` from a decoded JSON dictionary."""
        raw_outputs = data.get("outputs", [])
        outputs = [
            TensorOutput.from_rest_dict(item)
            for item in raw_outputs
            if isinstance(item, Mapping)
        ]
        return cls(
            id=str(data.get("id", "")),
            model_id=str(data.get("model_id", "")),
            status=str(data.get("status", "success")),
            outputs=outputs,
            error=data.get("error"),
            duration_ms=float(data.get("duration_ms", 0.0)),
        )

    @classmethod
    def from_bytes(
        cls,
        payload: Union[bytes, bytearray, memoryview, str],
        format: Union[StreamWireFormat, str] = StreamWireFormat.JSON,
        default_id: str = "",
    ) -> "StreamInferResponse":
        """Deserializes a :class:`StreamInferResponse` from raw broker bytes.

        Args:
            payload: Raw bytes or UTF-8 string read from an output or DLQ stream.
            format: Wire format (:attr:`StreamWireFormat.JSON` or :attr:`StreamWireFormat.PROTOBUF`).
            default_id: Fallback correlation ID when decoding Protobuf responses.

        Returns:
            Decoded :class:`StreamInferResponse`.
        """
        wire_fmt = StreamWireFormat.from_str(format)
        raw_bytes = payload.encode("utf-8") if isinstance(payload, str) else bytes(payload)

        if wire_fmt == StreamWireFormat.JSON:
            try:
                data = json.loads(raw_bytes.decode("utf-8"))
            except Exception as exc:
                raise InvalidInputError(f"Failed to parse stream JSON response: {exc}") from exc
            if not isinstance(data, Mapping):
                raise InvalidInputError("Stream JSON response payload must be an object")
            return cls.from_dict(data)

        if not _HAS_PROTO or ModelInferResponse is None:
            raise InvalidInputError(
                "Protobuf stream deserialization requires 'protobuf'. Install with: pip install 'snoozeml[grpc]'"
            )
        proto_resp = ModelInferResponse()
        try:
            proto_resp.ParseFromString(raw_bytes)
        except Exception as exc:
            raise InvalidInputError(f"Failed to decode protobuf stream response: {exc}") from exc

        outputs = [TensorOutput.from_proto(t) for t in proto_resp.outputs]
        return cls(
            id=default_id,
            model_id=str(proto_resp.model_id),
            status="success",
            outputs=outputs,
            error=None,
            duration_ms=float(proto_resp.duration_ms),
        )


class MockStreamClient:
    """Deterministic in-memory streaming client for testing stream pipelines in Python.

    Can be wired either to a live :class:`~snoozeml.SnoozeClient` or to a custom
    inference handler callback, simulating SnoozeML's ``src/stream/mock_stream.rs``
    broker with full JSON and Protobuf wire serialization round-trips.
    """

    def __init__(
        self,
        *,
        client: Optional[Any] = None,
        handler: Optional[Callable[[str, List[TensorInput]], List[TensorOutput]]] = None,
        default_model_id: Optional[str] = None,
        input_topic: str = "snooze.requests",
        output_topic: str = "snooze.responses",
        error_topic: str = "snooze.dlq",
        format: Union[StreamWireFormat, str] = StreamWireFormat.JSON,
    ) -> None:
        self._client = client
        self._handler = handler
        self.default_model_id = default_model_id
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.error_topic = error_topic
        self.format = StreamWireFormat.from_str(format)
        self.topics: Dict[str, Deque[bytes]] = {}

    def publish_request(
        self,
        inputs: InputTypes,
        *,
        model_id: Optional[str] = None,
        request_id: Optional[str] = None,
        reply_to: Optional[str] = None,
        topic: Optional[str] = None,
    ) -> str:
        """Serializes and publishes a :class:`StreamInferRequest` onto the input topic.

        If ``client`` or ``handler`` is attached, immediately processes the message and
        publishes the serialized :class:`StreamInferResponse` onto ``reply_to`` or ``output_topic``.

        Returns:
            Correlation ID of the published request.
        """
        req = StreamInferRequest.from_inputs(
            inputs,
            model_id=model_id or self.default_model_id,
            id=request_id,
            reply_to=reply_to,
        )
        target_in = topic or self.input_topic
        raw_req = req.to_bytes(self.format)
        self.topics.setdefault(target_in, deque()).append(raw_req)

        if self._client is not None or self._handler is not None:
            self.process_next(topic=target_in)

        return req.id

    def process_next(self, *, topic: Optional[str] = None) -> Optional[StreamInferResponse]:
        """Pops one raw request frame from ``topic``, executes inference, and writes the response frame."""
        source_topic = topic or self.input_topic
        queue = self.topics.get(source_topic)
        if not queue:
            return None

        raw_bytes = queue.popleft()
        t0 = time.perf_counter()
        try:
            req = StreamInferRequest.from_bytes(raw_bytes, format=self.format)
        except Exception as exc:
            err_resp = StreamInferResponse(
                id="unknown",
                model_id=self.default_model_id or "unknown",
                status="error",
                error=str(exc),
                duration_ms=(time.perf_counter() - t0) * 1000.0,
            )
            self.topics.setdefault(self.error_topic, deque()).append(
                err_resp.to_bytes(StreamWireFormat.JSON)
            )
            return err_resp

        target_model = req.model_id or self.default_model_id
        dest_topic = req.reply_to or self.output_topic
        if not target_model:
            err_resp = StreamInferResponse(
                id=req.id,
                model_id="unrouted",
                status="error",
                error="Stream message missing 'model_id' and stream has no default model_id",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
            )
            self.topics.setdefault(self.error_topic, deque()).append(
                err_resp.to_bytes(StreamWireFormat.JSON)
            )
            return err_resp

        try:
            if self._handler is not None:
                outputs = self._handler(target_model, req.inputs)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                resp = StreamInferResponse(
                    id=req.id,
                    model_id=target_model,
                    status="success",
                    outputs=outputs,
                    duration_ms=elapsed_ms,
                )
            elif self._client is not None:
                infer_res = self._client.infer(target_model, req.inputs)
                resp = StreamInferResponse(
                    id=req.id,
                    model_id=infer_res.model_id,
                    status="success",
                    outputs=list(infer_res.outputs),
                    duration_ms=infer_res.duration_ms,
                )
            else:
                return None

            self.topics.setdefault(dest_topic, deque()).append(resp.to_bytes(self.format))
            return resp
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            err_resp = StreamInferResponse(
                id=req.id,
                model_id=target_model,
                status="error",
                error=str(exc),
                duration_ms=elapsed_ms,
            )
            self.topics.setdefault(self.error_topic, deque()).append(
                err_resp.to_bytes(StreamWireFormat.JSON)
            )
            return err_resp

    def read_response(
        self,
        *,
        correlation_id: Optional[str] = None,
        topic: Optional[str] = None,
    ) -> Optional[StreamInferResponse]:
        """Reads and deserializes the next matching :class:`StreamInferResponse` from ``topic``."""
        target_out = topic or self.output_topic
        queue = self.topics.get(target_out)
        if not queue:
            return None

        if correlation_id is None:
            raw = queue.popleft()
            return StreamInferResponse.from_bytes(raw, format=self.format)

        retained: Deque[bytes] = deque()
        matched: Optional[StreamInferResponse] = None
        while queue:
            raw = queue.popleft()
            decoded = StreamInferResponse.from_bytes(
                raw, format=self.format, default_id=correlation_id
            )
            if matched is None and (decoded.id == correlation_id or not decoded.id):
                if not decoded.id:
                    decoded.id = correlation_id
                matched = decoded
            else:
                retained.append(raw)
        self.topics[target_out] = retained
        return matched

    def infer(
        self,
        inputs: InputTypes,
        *,
        model_id: Optional[str] = None,
        reply_to: Optional[str] = None,
    ) -> InferenceResponse:
        """Publishes a request and returns the resulting :class:`~snoozeml.inference.InferenceResponse`."""
        corr_id = self.publish_request(inputs, model_id=model_id, reply_to=reply_to)
        out_topic = reply_to or self.output_topic
        resp = self.read_response(correlation_id=corr_id, topic=out_topic)
        if resp is None:
            dlq_resp = self.read_response(correlation_id=corr_id, topic=self.error_topic)
            if dlq_resp is not None:
                dlq_resp.raise_for_status()
            raise InferenceTimeoutError(
                f"No stream response received for correlation_id='{corr_id}' on topic '{out_topic}'"
            )
        return resp.to_inference_response()


class RedisStreamClient:
    """Redis Streams producer/consumer client compatible with SnoozeML's ``RedisStreamDriver``.

    Publishes requests via ``XADD <input_topic> * payload <bytes>`` and reads responses
    from ``<output_topic>`` (or per-request ``reply_to``) via ``XREAD``.

    Args:
        endpoint: Redis connection URL (e.g. ``"redis://127.0.0.1:6379"``). Ignored if ``redis_client`` is passed.
        input_topic: Redis stream key where SnoozeML consumes requests (default ``"snooze.requests"``).
        output_topic: Redis stream key where SnoozeML publishes responses (default ``"snooze.responses"``).
        default_model_id: Optional default ``model_id`` attached to outgoing requests.
        format: Wire serialization format (``"json"`` or ``"protobuf"``).
        redis_client: Optional pre-configured ``redis.Redis`` client instance.
    """

    def __init__(
        self,
        endpoint: str = "redis://127.0.0.1:6379",
        *,
        input_topic: str = "snooze.requests",
        output_topic: str = "snooze.responses",
        default_model_id: Optional[str] = None,
        format: Union[StreamWireFormat, str] = StreamWireFormat.JSON,
        redis_client: Optional[Any] = None,
    ) -> None:
        self.endpoint = endpoint
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.default_model_id = default_model_id
        self.format = StreamWireFormat.from_str(format)

        if redis_client is not None:
            self._redis = redis_client
        else:
            try:
                import redis  # type: ignore
            except ImportError as exc:
                raise SnoozeError(
                    "RedisStreamClient requires the 'redis' package. Install with: pip install redis"
                ) from exc
            self._redis = redis.Redis.from_url(endpoint, decode_responses=False)

    def publish_request(
        self,
        inputs: InputTypes,
        *,
        model_id: Optional[str] = None,
        request_id: Optional[str] = None,
        reply_to: Optional[str] = None,
        topic: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Publishes a :class:`StreamInferRequest` to Redis Streams via ``XADD``.

        Returns:
            Tuple of ``(correlation_id, redis_stream_entry_id)``.
        """
        req = StreamInferRequest.from_inputs(
            inputs,
            model_id=model_id or self.default_model_id,
            id=request_id,
            reply_to=reply_to,
        )
        target_stream = topic or self.input_topic
        payload_bytes = req.to_bytes(self.format)
        try:
            entry_id = self._redis.xadd(target_stream, {"payload": payload_bytes})
        except Exception as exc:
            raise SnoozeConnectionError(
                f"Failed to XADD stream request to Redis stream '{target_stream}': {exc}"
            ) from exc

        entry_id_str = (
            entry_id.decode("utf-8") if isinstance(entry_id, (bytes, bytearray)) else str(entry_id)
        )
        return req.id, entry_id_str

    def wait_for_response(
        self,
        correlation_id: str,
        *,
        topic: Optional[str] = None,
        last_id: str = "0-0",
        timeout: float = 10.0,
    ) -> StreamInferResponse:
        """Polls the Redis output stream via ``XREAD`` until the matching ``correlation_id`` arrives.

        Args:
            correlation_id: Request correlation ID to match.
            topic: Output stream key (defaults to :attr:`output_topic`).
            last_id: Redis stream cursor ID to start reading after (default ``"0-0"``).
            timeout: Maximum wait time in seconds before raising :class:`InferenceTimeoutError`.

        Returns:
            Matching :class:`StreamInferResponse`.
        """
        target_stream = topic or self.output_topic
        cursor = last_id
        deadline = time.monotonic() + float(timeout)

        while time.monotonic() < deadline:
            remaining_ms = max(10, int((deadline - time.monotonic()) * 1000))
            block_ms = min(remaining_ms, 250)
            try:
                entries = self._redis.xread(
                    streams={target_stream: cursor}, count=16, block=block_ms
                )
            except Exception as exc:
                raise SnoozeConnectionError(
                    f"Failed to XREAD from Redis stream '{target_stream}': {exc}"
                ) from exc

            if not entries:
                continue

            for _stream_name, messages in entries:
                for msg_id, fields in messages:
                    cursor = (
                        msg_id.decode("utf-8")
                        if isinstance(msg_id, (bytes, bytearray))
                        else str(msg_id)
                    )
                    raw_payload = (
                        fields.get(b"payload")
                        or fields.get("payload")
                        or fields.get(b"data")
                        or fields.get("data")
                    )
                    if raw_payload is None and fields:
                        raw_payload = next(iter(fields.values()))
                    if raw_payload is None:
                        continue

                    resp = StreamInferResponse.from_bytes(
                        raw_payload, format=self.format, default_id=correlation_id
                    )
                    if resp.id == correlation_id or self.format == StreamWireFormat.PROTOBUF:
                        return resp

        raise InferenceTimeoutError(
            f"Timed out waiting {timeout}s for response '{correlation_id}' on Redis stream '{target_stream}'"
        )

    def infer(
        self,
        inputs: InputTypes,
        *,
        model_id: Optional[str] = None,
        reply_to: Optional[str] = None,
        timeout: float = 10.0,
    ) -> InferenceResponse:
        """Publishes a request to Redis Streams and waits for the correlated :class:`InferenceResponse`."""
        corr_id, _ = self.publish_request(inputs, model_id=model_id, reply_to=reply_to)
        stream_resp = self.wait_for_response(
            corr_id,
            topic=reply_to or self.output_topic,
            timeout=timeout,
        )
        return stream_resp.to_inference_response()
