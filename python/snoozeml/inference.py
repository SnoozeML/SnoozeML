"""Tensor payload validation, wire serialization, and inference response types for SnoozeML."""

from __future__ import annotations

from array import array as _stdlib_array
from dataclasses import dataclass
from enum import Enum
import math
import struct
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

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
from .exceptions import InvalidInputError


class TensorDataType(str, Enum):
    """Supported tensor element data types."""

    FP32 = "fp32"


class _Float32Array:
    """Lightweight stdlib float32 multi-dimensional array used when ``numpy`` is not installed."""

    def __init__(self, flat_values: Sequence[float], shape: Sequence[int]) -> None:
        self._buf = _stdlib_array("f", (float(x) for x in flat_values))
        self.shape: Tuple[int, ...] = tuple(int(d) for d in shape)

    @property
    def size(self) -> int:
        return len(self._buf)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def ravel(self) -> "_Float32Array":
        return _Float32Array(self._buf, (len(self._buf),))

    def reshape(self, new_shape: Sequence[int]) -> "_Float32Array":
        shape_list = [int(d) for d in new_shape]
        return _Float32Array(self._buf, shape_list)

    def tolist(self) -> Any:
        if len(self.shape) <= 1:
            return list(self._buf)

        def _build(offset: int, dims: Sequence[int]) -> Tuple[List[Any], int]:
            if len(dims) == 1:
                end = offset + dims[0]
                return list(self._buf[offset:end]), end
            items: List[Any] = []
            cur = offset
            for _ in range(dims[0]):
                sub, cur = _build(cur, dims[1:])
                items.append(sub)
            return items, cur

        nested, _ = _build(0, self.shape)
        return nested

    def tobytes(self) -> bytes:
        return struct.pack(f"<{len(self._buf)}f", *self._buf)

    def __iter__(self) -> Iterator[float]:
        return iter(self._buf)

    def __len__(self) -> int:
        return self.shape[0] if self.shape else 0

    def __getitem__(self, idx: Any) -> Any:
        if isinstance(idx, slice) and self.shape:
            stride = 1
            for d in self.shape[1:]:
                stride *= d
            start, stop, step = idx.indices(self.shape[0])
            if step != 1:
                raise ValueError("Only contiguous unit-step slicing is supported")
            sliced_vals = self._buf[start * stride : stop * stride]
            new_shape = (max(0, stop - start),) + self.shape[1:]
            return _Float32Array(sliced_vals, new_shape)
        if isinstance(idx, int):
            if len(self.shape) <= 1:
                return float(self._buf[idx])
            stride = 1
            for d in self.shape[1:]:
                stride *= d
            if idx < 0:
                idx += self.shape[0]
            start = idx * stride
            return _Float32Array(self._buf[start : start + stride], self.shape[1:])
        if isinstance(idx, tuple):
            cur: Any = self
            for sub_idx in idx:
                cur = cur[sub_idx]
            return cur
        raise TypeError(f"Unsupported index type: {type(idx).__name__}")


def _infer_nested_shape_and_flat(value: Any) -> Tuple[List[int], List[float]]:
    """Infers shape and flattened float list from nested Python sequences."""
    if isinstance(value, _Float32Array):
        return list(value.shape), list(value._buf)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [], [float(value)]
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return [0], []
        first_shape, _ = _infer_nested_shape_and_flat(value[0])
        flat: List[float] = []
        for item in value:
            sub_shape, sub_flat = _infer_nested_shape_and_flat(item)
            if sub_shape != first_shape:
                raise ValueError("Inhomogeneous sequence dimensions")
            flat.extend(sub_flat)
        return [len(value)] + first_shape, flat
    raise TypeError(f"Cannot convert {type(value).__name__} to float32 tensor")


def _to_float32_array(value: Any, shape: Optional[Sequence[int]] = None) -> Any:
    """Converts ``value`` into a contiguous ``np.ndarray`` (if NumPy is installed) or ``_Float32Array``."""
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    if _HAS_NUMPY and np is not None:
        raw_arr = np.asarray(value, dtype=np.float32)
        arr = np.ascontiguousarray(raw_arr, dtype=np.float32)
        if shape is not None:
            return arr.reshape([int(d) for d in shape])
        return arr

    inferred_shape, flat = _infer_nested_shape_and_flat(value)
    final_shape = [int(d) for d in shape] if shape is not None else inferred_shape
    return _Float32Array(flat, final_shape)


def _from_f32_le_bytes(raw_bytes: bytes, shape: Sequence[int]) -> Any:
    """Decodes raw little-endian float32 bytes into an array."""
    if _HAS_NUMPY and np is not None:
        arr = np.frombuffer(raw_bytes, dtype="<f4").copy()
        return arr.reshape(list(shape)) if shape else arr
    count = len(raw_bytes) // 4
    vals = struct.unpack(f"<{count}f", raw_bytes)
    return _Float32Array(vals, list(shape) if shape else [count])


def _is_int_like(val: Any) -> bool:
    if isinstance(val, bool):
        return False
    if isinstance(val, int):
        return True
    if _HAS_NUMPY and np is not None and isinstance(val, np.integer):
        return True
    return False


def _checked_shape_product(shape: Sequence[int], name: str) -> int:
    """Computes the product of tensor dimensions while validating positive non-empty bounds."""
    if not shape:
        raise InvalidInputError(f"Tensor '{name}' shape cannot be empty")

    total = 1
    for dim in shape:
        if not _is_int_like(dim):
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
        data: Contiguous ``float32`` NumPy array (or fallback array) matching :attr:`shape`.
    """

    name: str
    shape: List[int]
    data: Any

    def __post_init__(self) -> None:
        if not self.name or not str(self.name).strip():
            raise InvalidInputError("Tensor name cannot be empty")
        self.name = str(self.name)
        self.shape = [int(d) for d in self.shape]
        expected_elements = _checked_shape_product(self.shape, self.name)

        arr = _to_float32_array(self.data)
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
        array: Any,
        shape: Optional[Sequence[int]] = None,
    ) -> "TensorInput":
        """Constructs a :class:`TensorInput` from a NumPy array, PyTorch ``Tensor``, or nested sequence.

        Args:
            name: Binding name of the tensor node.
            array: NumPy array, PyTorch tensor (auto-detached to CPU), or nested sequence of numeric values.
            shape: Optional explicit shape override. If omitted, inferred from ``array``.

        Returns:
            Validated :class:`TensorInput`.

        Raises:
            InvalidInputError: If the shape is empty, contains non-positive dimensions,
                or does not match the element count.
        """
        try:
            arr = _to_float32_array(array)
            if shape is None and arr.ndim == 0:
                raise InvalidInputError(
                    f"Tensor '{name}' cannot be a 0-dimensional scalar; expected at least 1D shape (e.g. [1, features])"
                )
        except InvalidInputError:
            raise
        except (ValueError, TypeError, RuntimeError) as exc:
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
        if _HAS_NUMPY and np is not None:
            is_valid = bool(np.all(np.isfinite(flat)))
        else:
            is_valid = all(not (math.isnan(float(v)) or math.isinf(float(v))) for v in flat)
        if not is_valid:
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
        if _HAS_NUMPY and np is not None:
            le_bytes = np.ascontiguousarray(self.data, dtype="<f4").tobytes()
        else:
            le_bytes = self.data.tobytes()
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
        data: NumPy ``float32`` array (or fallback array) reshaped to :attr:`shape`.
    """

    name: str
    shape: List[int]
    data: Any

    def to_numpy(self) -> Any:
        """Returns the output tensor as a reshaped ``np.ndarray`` of dtype ``float32``."""
        return self.data.reshape(self.shape)

    def as_numpy(self) -> Any:
        """Alias for :meth:`to_numpy`."""
        return self.to_numpy()

    def to_torch(self, device: Optional[str] = None) -> Any:
        """Returns the output tensor as a PyTorch ``torch.Tensor`` (``dtype=torch.float32``).

        Args:
            device: Optional PyTorch device string (e.g. ``"cpu"``, ``"cuda"``, ``"mps"``).

        Returns:
            ``torch.Tensor`` matching :attr:`shape`.

        Raises:
            InvalidInputError: If ``torch`` is not installed.
        """
        try:
            import torch
        except ImportError as exc:
            raise InvalidInputError(
                "PyTorch tensor conversion requires 'torch'. Install with: pip install torch"
            ) from exc

        if _HAS_NUMPY and np is not None:
            tensor = torch.from_numpy(np.ascontiguousarray(self.to_numpy(), dtype=np.float32))
        else:
            tensor = torch.tensor(self.to_list(), dtype=torch.float32).reshape(self.shape)
        if device is not None:
            tensor = tensor.to(device)
        return tensor

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
        arr = _to_float32_array(raw_data, shape=shape if shape else None)
        return cls(name=name, shape=shape, data=arr)

    @classmethod
    def from_proto(cls, tensor_proto: Any) -> "TensorOutput":
        """Constructs a :class:`TensorOutput` from a Protobuf ``TensorData`` message."""
        name = str(tensor_proto.name)
        shape = [int(d) for d in tensor_proto.shape]
        arr = _from_f32_le_bytes(tensor_proto.raw_data, shape)
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

    def get_torch(self, name: Optional[str] = None, device: Optional[str] = None) -> Any:
        """Returns the output tensor as a reshaped PyTorch ``torch.Tensor`` (``dtype=torch.float32``).

        Args:
            name: Optional name of the output tensor node (if ``None``, uses first output).
            device: Optional PyTorch device string (e.g. ``"cpu"``, ``"cuda"``).

        Returns:
            Multi-dimensional ``torch.Tensor`` matching the output tensor's shape.
        """
        return self.get_tensor(name).to_torch(device=device)

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
    Mapping[str, Union[Any, Sequence[Any], TensorInput]],
    Sequence[Union[TensorInput, Mapping[str, Any]]],
    TensorInput,
    Any,
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

    if (
        (_HAS_NUMPY and np is not None and isinstance(inputs, np.ndarray))
        or isinstance(inputs, _Float32Array)
        or (hasattr(inputs, "detach") and hasattr(inputs, "cpu"))
    ):
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


def slice_inputs_batch(
    inputs: InputTypes,
    batch_size: int,
    default_input_name: str = "input",
) -> List[List[TensorInput]]:
    """Splits a dataset of input tensors along ``Axis(0)`` into chunks of ``batch_size``.

    Args:
        inputs: Input tensors in any supported format.
        batch_size: Maximum batch size along ``Axis(0)`` per chunk.
        default_input_name: Binding name when a bare array is passed.

    Returns:
        List of ``List[TensorInput]`` chunks ready for sequential or concurrent ``infer()`` calls.

    Raises:
        InvalidInputError: If ``batch_size <= 0`` or input tensors have mismatched ``Axis(0)`` lengths.
    """
    if batch_size <= 0:
        raise InvalidInputError(f"batch_size must be positive, got {batch_size}")

    tensors = normalize_inputs(inputs, default_input_name=default_input_name)
    total_rows = int(tensors[0].shape[0])
    for t in tensors[1:]:
        if int(t.shape[0]) != total_rows:
            raise InvalidInputError(
                f"All input tensors must share the same leading batch dimension: "
                f"'{tensors[0].name}' has {total_rows}, but '{t.name}' has {t.shape[0]}"
            )

    chunks: List[List[TensorInput]] = []
    for start in range(0, total_rows, batch_size):
        end = min(start + batch_size, total_rows)
        chunk_tensors: List[TensorInput] = []
        for t in tensors:
            if _HAS_NUMPY and np is not None:
                sliced_arr = np.ascontiguousarray(t.data[start:end], dtype=np.float32)
            else:
                sliced_arr = t.data[start:end]
            chunk_tensors.append(
                TensorInput(
                    name=t.name,
                    shape=list(sliced_arr.shape),
                    data=sliced_arr,
                )
            )
        chunks.append(chunk_tensors)
    return chunks


def concat_inference_responses(
    responses: Sequence[InferenceResponse],
) -> InferenceResponse:
    """Concatenates a sequence of chunked :class:`InferenceResponse` objects along ``Axis(0)``."""
    if not responses:
        raise InvalidInputError("Cannot concatenate an empty sequence of InferenceResponse objects")
    if len(responses) == 1:
        return responses[0]

    first = responses[0]
    total_duration_ms = sum(r.duration_ms for r in responses)
    merged_outputs: List[TensorOutput] = []

    for out_idx, first_out in enumerate(first.outputs):
        matched_outs: List[TensorOutput] = [first_out]
        for resp in responses[1:]:
            if first_out.name in resp.outputs:
                matched_outs.append(resp.get_tensor(first_out.name))
            elif out_idx < len(resp.outputs):
                matched_outs.append(resp.outputs[out_idx])
            else:
                raise InvalidInputError(
                    f"Output tensor '{first_out.name}' missing from chunk response"
                )
        if _HAS_NUMPY and np is not None:
            arrays = [o.to_numpy() for o in matched_outs]
            concatenated = np.ascontiguousarray(np.concatenate(arrays, axis=0), dtype=np.float32)
            merged_shape = list(concatenated.shape)
        else:
            flat_merged: List[float] = []
            total_axis0 = 0
            trailing_dims = list(first_out.shape[1:])
            for o in matched_outs:
                total_axis0 += int(o.shape[0]) if o.shape else o.data.size
                flat_merged.extend(o.to_list())
            merged_shape = [total_axis0] + trailing_dims
            concatenated = _Float32Array(flat_merged, merged_shape)

        merged_outputs.append(
            TensorOutput(
                name=first_out.name,
                shape=merged_shape,
                data=concatenated,
            )
        )

    return InferenceResponse(
        model_id=first.model_id,
        outputs=merged_outputs,
        duration_ms=total_duration_ms,
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
