"""Tensor payload validation, wire serialization, and inference response types for SnoozeML."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np

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
from .exceptions import InvalidInputError


class TensorDataType(str, Enum):
    """Supported tensor element data types."""

    FP32 = "fp32"


def _checked_shape_product(shape: Sequence[int], name: str) -> int:
    """Computes the product of tensor dimensions while validating positive non-empty bounds."""
    if not shape:
        raise InvalidInputError(f"Tensor '{name}' shape cannot be empty")

    total = 1
    for dim in shape:
        if not isinstance(dim, (int, np.integer)) or isinstance(dim, bool):
            raise InvalidInputError(
                f"Tensor '{name}' shape dimension must be an integer, got {type(dim).__name__}"
            )
        dim_int = int(dim)
        if dim_int <= 0:
            raise InvalidInputError(
                f"Tensor '{name}' dimension must be positive, got {dim_int}"
            )
        total *= dim_int
    return total


@dataclass
class TensorInput:
    """Validated multi-dimensional input tensor ready for REST or gRPC serialization.

    Attributes:
        name: Binding identifier of the input tensor node (e.g. ``"input"``, ``"input_ids"``).
        shape: List of positive integer dimensions (e.g. ``[1, 3, 224, 224]``).
        data: Contiguous ``float32`` NumPy array matching :attr:`shape`.
    """

    name: str
    shape: List[int]
    data: np.ndarray

    def __post_init__(self) -> None:
        if not self.name or not str(self.name).strip():
            raise InvalidInputError("Tensor name cannot be empty")
        self.name = str(self.name)
        self.shape = [int(d) for d in self.shape]
        expected_elements = _checked_shape_product(self.shape, self.name)

        arr = np.ascontiguousarray(self.data, dtype=np.float32)
        if arr.size != expected_elements:
            raise InvalidInputError(
                f"Tensor '{self.name}' shape {self.shape} expects {expected_elements} elements, "
                f"but provided buffer has {arr.size} elements"
            )
        self.data = arr.reshape(self.shape)

    @classmethod
    def from_array(
        cls,
        name: str,
        array: Union[np.ndarray, Sequence[Any]],
        shape: Optional[Sequence[int]] = None,
    ) -> "TensorInput":
        """Constructs a :class:`TensorInput` from a NumPy array or nested Python sequence.

        Args:
            name: Binding name of the tensor node.
            array: NumPy array or nested sequence of numeric values.
            shape: Optional explicit shape override. If omitted, inferred from ``array``.

        Returns:
            Validated :class:`TensorInput`.

        Raises:
            InvalidInputError: If the shape is empty, contains non-positive dimensions,
                or does not match the element count.
        """
        try:
            raw_arr = np.asarray(array, dtype=np.float32)
            if shape is None and raw_arr.ndim == 0:
                raise InvalidInputError(
                    f"Tensor '{name}' cannot be a 0-dimensional scalar; expected at least 1D shape (e.g. [1, features])"
                )
            arr = np.ascontiguousarray(raw_arr, dtype=np.float32)
        except InvalidInputError:
            raise
        except (ValueError, TypeError) as exc:
            raise InvalidInputError(
                f"Tensor '{name}' could not be converted to float32 array: {exc}"
            ) from exc

        inferred_shape = list(shape) if shape is not None else list(arr.shape)
        if not inferred_shape:
            raise InvalidInputError(
                f"Tensor '{name}' cannot be a 0-dimensional scalar; expected at least 1D shape (e.g. [1, features])"
            )
        return cls(name=name, shape=inferred_shape, data=arr)

    @classmethod
    def from_dict(cls, item: Mapping[str, Any]) -> "TensorInput":
        """Constructs a :class:`TensorInput` from a dictionary ``{"name": ..., "shape": ..., "data": ...}``."""
        if "name" not in item:
            raise InvalidInputError("Input tensor dictionary is missing required 'name' field")
        name = str(item["name"])
        if "data" not in item:
            raise InvalidInputError(f"Tensor '{name}' dictionary is missing required 'data' field")
        raw_data = item["data"]
        shape = item.get("shape")
        if shape is not None and not isinstance(shape, (list, tuple)):
            raise InvalidInputError(f"Tensor '{name}' shape must be a list or tuple of integers")
        return cls.from_array(name=name, array=raw_data, shape=shape)

    def to_rest_dict(self) -> Dict[str, Any]:
        """Serializes the tensor into the JSON dictionary format expected by ``POST /v1/models/:id/infer``."""
        flat = self.data.ravel()
        # Check for NaN or Inf which are invalid in standard JSON
        if not np.all(np.isfinite(flat)):
            for val in flat:
                if math.isnan(float(val)) or math.isinf(float(val)):
                    raise InvalidInputError(
                        f"Tensor '{self.name}' contains non-finite float values (NaN or Inf)"
                    )
        return {
            "name": self.name,
            "shape": list(self.shape),
            "data": flat.tolist(),
        }

    def to_proto(self) -> Any:
        """Serializes the tensor into a Protobuf ``TensorData`` message with raw little-endian FP32 bytes."""
        if not _HAS_PROTO or TensorData is None or ProtoDataType is None:
            raise InvalidInputError(
                "gRPC serialization requires 'protobuf'. Install with: pip install 'snoozeml[grpc]'"
            )
        le_bytes = np.ascontiguousarray(self.data, dtype="<f4").tobytes()
        return TensorData(
            name=self.name,
            shape=list(self.shape),
            data_type=ProtoDataType.DATA_TYPE_FP32,
            raw_data=le_bytes,
        )


@dataclass
class TensorOutput:
    """Multi-dimensional output tensor returned from model inference.

    Attributes:
        name: Binding identifier of the output tensor node (e.g. ``"output"``, ``"logits"``).
        shape: Dimensions of the output tensor.
        data: NumPy ``float32`` array reshaped to :attr:`shape`.
    """

    name: str
    shape: List[int]
    data: np.ndarray

    def to_numpy(self) -> np.ndarray:
        """Returns the output tensor as a reshaped ``np.ndarray`` of dtype ``float32``."""
        return self.data.reshape(self.shape)

    def as_numpy(self) -> np.ndarray:
        """Alias for :meth:`to_numpy`."""
        return self.to_numpy()

    def to_list(self) -> List[float]:
        """Returns the flattened output tensor elements as a Python ``list[float]``."""
        return [float(x) for x in self.data.ravel().tolist()]

    def to_dict(self) -> Dict[str, Any]:
        """Returns a dictionary ``{"name": ..., "shape": ..., "data": ...}`` matching the REST format."""
        return {
            "name": self.name,
            "shape": list(self.shape),
            "data": self.to_list(),
        }

    @classmethod
    def from_rest_dict(cls, item: Mapping[str, Any]) -> "TensorOutput":
        """Constructs a :class:`TensorOutput` from a REST JSON output tensor dictionary."""
        name = str(item.get("name", "output"))
        shape = [int(d) for d in item.get("shape", [])]
        raw_data = item.get("data", [])
        arr = np.asarray(raw_data, dtype=np.float32)
        if shape:
            arr = arr.reshape(shape)
        return cls(name=name, shape=shape, data=arr)

    @classmethod
    def from_proto(cls, tensor_proto: Any) -> "TensorOutput":
        """Constructs a :class:`TensorOutput` from a Protobuf ``TensorData`` message."""
        name = str(tensor_proto.name)
        shape = [int(d) for d in tensor_proto.shape]
        arr = np.frombuffer(tensor_proto.raw_data, dtype="<f4").copy()
        if shape:
            arr = arr.reshape(shape)
        return cls(name=name, shape=shape, data=arr)


class OutputList(List[TensorOutput]):
    """Enhanced list of :class:`TensorOutput` supporting both index and name lookup.

    Allows accessing outputs by index (``outputs[0]``), by tensor name (``outputs["logits"]``),
    and checking membership by name (``"logits" in outputs``).
    """

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            for out in self:
                if out.name == key:
                    return out
            available = [out.name for out in self]
            raise KeyError(
                f"Output tensor '{key}' not found. Available outputs: {available}"
            )
        return super().__getitem__(key)

    def __contains__(self, key: Any) -> bool:
        if isinstance(key, str):
            return any(out.name == key for out in self)
        return super().__contains__(key)

    def get(self, name: str, default: Optional[TensorOutput] = None) -> Optional[TensorOutput]:
        """Retrieves a tensor by name, or returns ``default`` if not found."""
        for out in self:
            if out.name == name:
                return out
        return default


@dataclass
class InferenceResponse:
    """Structured result returned by :meth:`snoozeml.SnoozeClient.infer`.

    Attributes:
        model_id: Identifier of the executed model.
        outputs: Ordered list of :class:`TensorOutput` objects (supports name indexing).
        duration_ms: Server-side execution duration in milliseconds.
    """

    model_id: str
    outputs: List[TensorOutput]
    duration_ms: float

    def __post_init__(self) -> None:
        if not isinstance(self.outputs, OutputList):
            self.outputs = OutputList(self.outputs)

    def get_tensor(self, name: Optional[str] = None) -> TensorOutput:
        """Retrieves a :class:`TensorOutput` by binding name (or the first output if ``name`` is ``None``).

        Args:
            name: Optional name of the output tensor node (e.g. ``"logits"``, ``"output"``).
                If ``None``, returns the first output tensor.

        Returns:
            Matching :class:`TensorOutput`.

        Raises:
            KeyError: If no output tensors exist or ``name`` is not found.
        """
        if not self.outputs:
            raise KeyError(f"Model '{self.model_id}' response contains no output tensors")

        if name is None:
            return self.outputs[0]

        for out in self.outputs:
            if out.name == name:
                return out

        available = [out.name for out in self.outputs]
        raise KeyError(
            f"Output tensor '{name}' not found in response from '{self.model_id}'. "
            f"Available outputs: {available}"
        )

    def get_numpy(self, name: Optional[str] = None) -> np.ndarray:
        """Returns the output tensor as a reshaped ``np.ndarray`` (``dtype=np.float32``).

        Args:
            name: Optional name of the output tensor node (e.g. ``"logits"``).
                If ``None``, returns the first output tensor's NumPy array.

        Returns:
            Multi-dimensional ``np.ndarray`` matching the output tensor's shape.

        Raises:
            KeyError: If ``name`` is not present in the response outputs.
        """
        return self.get_tensor(name).to_numpy()

    def to_dict(self) -> Dict[str, np.ndarray]:
        """Returns a dictionary mapping each output tensor name to its ``np.ndarray``."""
        return {out.name: out.to_numpy() for out in self.outputs}

    def __getitem__(self, key: Union[str, int]) -> np.ndarray:
        """Allows indexing by tensor name (``resp["output"]``) or positional index (``resp[0]``)."""
        if isinstance(key, int):
            return self.outputs[key].to_numpy()
        return self.get_numpy(key)

    @classmethod
    def from_rest_dict(cls, data: Mapping[str, Any], default_model_id: str = "") -> "InferenceResponse":
        """Constructs an :class:`InferenceResponse` from a REST JSON payload."""
        model_id = str(data.get("model_id", default_model_id))
        raw_outputs = data.get("outputs", [])
        outputs = [
            TensorOutput.from_rest_dict(item)
            for item in raw_outputs
            if isinstance(item, Mapping)
        ]
        duration_ms = float(data.get("duration_ms", 0.0))
        return cls(model_id=model_id, outputs=outputs, duration_ms=duration_ms)

    @classmethod
    def from_proto(cls, proto_resp: ModelInferResponse) -> "InferenceResponse":
        """Constructs an :class:`InferenceResponse` from a Protobuf ``ModelInferResponse``."""
        outputs = [TensorOutput.from_proto(t) for t in proto_resp.outputs]
        return cls(
            model_id= str(proto_resp.model_id),
            outputs=outputs,
            duration_ms=float(proto_resp.duration_ms),
        )


InputTypes = Union[
    Mapping[str, Union[np.ndarray, Sequence[Any], TensorInput]],
    Sequence[Union[TensorInput, Mapping[str, Any]]],
    TensorInput,
    np.ndarray,
]


def normalize_inputs(
    inputs: InputTypes,
    default_input_name: str = "input",
) -> List[TensorInput]:
    """Normalizes flexible user input representations into a validated ``List[TensorInput]``.

    Supported formats for ``inputs``:
      1. ``Dict[str, np.ndarray | Sequence | TensorInput]`` — e.g. ``{"input_ids": arr}``
      2. ``List[TensorInput | Dict[str, Any]]`` — e.g. ``[{"name": "input", "shape": [1, 3], "data": [1, 2, 3]}]``
      3. Single :class:`TensorInput`
      4. Single ``np.ndarray`` (assigned binding name ``default_input_name``, default ``"input"``)

    Args:
        inputs: Input tensors in any supported format.
        default_input_name: Binding name used when a single bare ``np.ndarray`` is passed.

    Returns:
        Non-empty list of validated :class:`TensorInput` objects.

    Raises:
        InvalidInputError: If ``inputs`` is empty or any tensor fails validation.
    """
    if inputs is None:
        raise InvalidInputError("Inference inputs cannot be None")

    if isinstance(inputs, TensorInput):
        return [inputs]

    if isinstance(inputs, np.ndarray):
        return [TensorInput.from_array(default_input_name, inputs)]

    if isinstance(inputs, Mapping):
        if not inputs:
            raise InvalidInputError("Inference inputs dictionary cannot be empty")
        normalized: List[TensorInput] = []
        for name, val in inputs.items():
            if isinstance(val, TensorInput):
                normalized.append(val)
            elif isinstance(val, Mapping) and "data" in val:
                d = dict(val)
                d.setdefault("name", name)
                normalized.append(TensorInput.from_dict(d))
            else:
                normalized.append(TensorInput.from_array(str(name), val))
        return normalized

    if isinstance(inputs, (list, tuple)):
        if not inputs:
            raise InvalidInputError("Inference inputs list cannot be empty")
        result: List[TensorInput] = []
        for item in inputs:
            if isinstance(item, TensorInput):
                result.append(item)
            elif isinstance(item, Mapping):
                result.append(TensorInput.from_dict(item))
            else:
                raise InvalidInputError(
                    f"Unsupported element in inputs list: expected TensorInput or dict, got {type(item).__name__}"
                )
        return result

    raise InvalidInputError(
        f"Unsupported inputs type '{type(inputs).__name__}'; expected dict, list, TensorInput, or numpy.ndarray"
    )


def build_grpc_infer_request(
    model_id: str,
    tensors: Sequence[TensorInput],
    client_timeout_ms: Optional[int] = None,
    parameters: Optional[Mapping[str, str]] = None,
) -> Any:
    """Constructs a Protobuf ``ModelInferRequest`` from validated :class:`TensorInput` objects."""
    if not _HAS_PROTO or ModelInferRequest is None:
        raise InvalidInputError(
            "gRPC inference requires 'protobuf'. Install with: pip install 'snoozeml[grpc]'"
        )
    req = ModelInferRequest(
        model_id=model_id,
        inputs=[t.to_proto() for t in tensors],
    )
    if client_timeout_ms is not None:
        if client_timeout_ms <= 0:
            raise InvalidInputError("client_timeout_ms must be positive")
        req.client_timeout_ms = int(client_timeout_ms)
    if parameters:
        for k, v in parameters.items():
            req.parameters[str(k)] = str(v)
    return req
