"""Lightweight, dependency-free ONNX ModelProto serializer using standard Protobuf wire encoding.

Constructs spec-compliant ONNX IR v8 (``opset 17`` / ``ai.onnx.ml opset 3``) binary ``.onnx``
files directly from NumPy arrays and computational graph definitions, allowing SnoozeML's
converters to export scikit-learn and PyTorch models even when optional C++ packages
like ``skl2onnx`` or ``onnx`` are not installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import struct
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# ONNX TensorProto.DataType constants
ONNX_FLOAT = 1
ONNX_INT32 = 6
ONNX_INT64 = 7

# ONNX AttributeProto.AttributeType constants
ATTR_FLOAT = 1
ATTR_INT = 2
ATTR_STRING = 3
ATTR_TENSOR = 4
ATTR_FLOATS = 6
ATTR_INTS = 7
ATTR_STRINGS = 8


def _encode_varint(value: int) -> bytes:
    """Encodes a signed or unsigned 64-bit integer as a Protobuf base-128 varint."""
    if value < 0:
        value += 1 << 64
    out = bytearray()
    while value > 0x7F:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value & 0x7F)
    return bytes(out)


def _encode_tag(field_number: int, wire_type: int) -> bytes:
    """Encodes a Protobuf field tag ``(field_number << 3) | wire_type``."""
    return _encode_varint((field_number << 3) | wire_type)


def _encode_bytes_field(field_number: int, payload: bytes) -> bytes:
    """Encodes a length-delimited (wire type 2) bytes or sub-message field."""
    return _encode_tag(field_number, 2) + _encode_varint(len(payload)) + payload


def _encode_string_field(field_number: int, text: str) -> bytes:
    """Encodes a UTF-8 string field (wire type 2)."""
    return _encode_bytes_field(field_number, text.encode("utf-8"))


def _encode_varint_field(field_number: int, value: int) -> bytes:
    """Encodes a varint field (wire type 0)."""
    return _encode_tag(field_number, 0) + _encode_varint(value)


def _encode_float_field(field_number: int, value: float) -> bytes:
    """Encodes a 32-bit float field (wire type 5)."""
    return _encode_tag(field_number, 5) + struct.pack("<f", float(value))


def encode_attribute(
    name: str,
    *,
    i: Optional[int] = None,
    f: Optional[float] = None,
    s: Optional[Union[str, bytes]] = None,
    ints: Optional[Sequence[int]] = None,
    floats: Optional[Sequence[float]] = None,
    strings: Optional[Sequence[Union[str, bytes]]] = None,
    tensor: Optional[bytes] = None,
) -> bytes:
    """Encodes an ONNX ``AttributeProto`` message."""
    parts: List[bytes] = [_encode_string_field(1, name)]
    if i is not None:
        parts.append(_encode_varint_field(3, int(i)))
        parts.append(_encode_varint_field(20, ATTR_INT))
    elif f is not None:
        parts.append(_encode_float_field(2, float(f)))
        parts.append(_encode_varint_field(20, ATTR_FLOAT))
    elif s is not None:
        raw_s = s.encode("utf-8") if isinstance(s, str) else bytes(s)
        parts.append(_encode_bytes_field(4, raw_s))
        parts.append(_encode_varint_field(20, ATTR_STRING))
    elif ints is not None:
        for val in ints:
            parts.append(_encode_varint_field(8, int(val)))
        parts.append(_encode_varint_field(20, ATTR_INTS))
    elif floats is not None:
        for val in floats:
            parts.append(_encode_float_field(7, float(val)))
        parts.append(_encode_varint_field(20, ATTR_FLOATS))
    elif strings is not None:
        for val in strings:
            raw_s = val.encode("utf-8") if isinstance(val, str) else bytes(val)
            parts.append(_encode_bytes_field(9, raw_s))
        parts.append(_encode_varint_field(20, ATTR_STRINGS))
    elif tensor is not None:
        parts.append(_encode_bytes_field(5, tensor))
        parts.append(_encode_varint_field(20, ATTR_TENSOR))
    return b"".join(parts)


def encode_tensor_proto(name: str, array: np.ndarray) -> bytes:
    """Encodes a NumPy array as an ONNX ``TensorProto`` initializer."""
    if array.dtype in (np.int64, np.int32):
        arr = np.ascontiguousarray(array, dtype=np.int64)
        dtype_code = ONNX_INT64
        raw_bytes = arr.astype("<i8", copy=False).tobytes()
    else:
        arr = np.ascontiguousarray(array, dtype=np.float32)
        dtype_code = ONNX_FLOAT
        raw_bytes = arr.astype("<f4", copy=False).tobytes()

    parts: List[bytes] = []
    for dim in arr.shape:
        parts.append(_encode_varint_field(1, int(dim)))
    parts.append(_encode_varint_field(2, dtype_code))
    if name:
        parts.append(_encode_string_field(8, name))
    parts.append(_encode_bytes_field(9, raw_bytes))
    return b"".join(parts)


def encode_value_info(
    name: str,
    dims: Sequence[Union[int, str, None]],
    elem_type: int = ONNX_FLOAT,
) -> bytes:
    """Encodes an ONNX ``ValueInfoProto`` with symbolic or fixed dimensions."""
    dim_messages: List[bytes] = []
    for d in dims:
        if d is None or isinstance(d, str):
            sym = d if isinstance(d, str) and d else "batch_size"
            dim_msg = _encode_string_field(2, sym)
        else:
            dim_msg = _encode_varint_field(1, int(d))
        dim_messages.append(_encode_bytes_field(1, dim_msg))

    shape_proto = b"".join(dim_messages)
    tensor_type_proto = _encode_varint_field(1, elem_type) + _encode_bytes_field(
        2, shape_proto
    )
    type_proto = _encode_bytes_field(1, tensor_type_proto)
    return _encode_string_field(1, name) + _encode_bytes_field(2, type_proto)


def encode_node(
    op_type: str,
    inputs: Sequence[str],
    outputs: Sequence[str],
    *,
    name: str = "",
    domain: str = "",
    attributes: Optional[Sequence[bytes]] = None,
) -> bytes:
    """Encodes an ONNX ``NodeProto``."""
    parts: List[bytes] = []
    for inp in inputs:
        parts.append(_encode_string_field(1, inp))
    for out in outputs:
        parts.append(_encode_string_field(2, out))
    if name:
        parts.append(_encode_string_field(3, name))
    parts.append(_encode_string_field(4, op_type))
    if attributes:
        for attr in attributes:
            parts.append(_encode_bytes_field(5, attr))
    if domain:
        parts.append(_encode_string_field(7, domain))
    return b"".join(parts)


@dataclass
class OnnxGraphBuilder:
    """Helper for assembling an ONNX ``ModelProto`` computation graph."""

    name: str = "snoozeml_graph"
    nodes: List[bytes] = field(default_factory=list)
    initializers: List[bytes] = field(default_factory=list)
    inputs: List[bytes] = field(default_factory=list)
    outputs: List[bytes] = field(default_factory=list)
    _counter: int = 0

    def unique_name(self, prefix: str) -> str:
        """Generates a unique tensor or node name within the graph."""
        self._counter += 1
        return f"{prefix}_{self._counter}"

    def add_input(
        self,
        name: str,
        dims: Sequence[Union[int, str, None]],
        elem_type: int = ONNX_FLOAT,
    ) -> None:
        """Registers a graph input ``ValueInfoProto``."""
        self.inputs.append(encode_value_info(name, dims, elem_type))

    def add_output(
        self,
        name: str,
        dims: Sequence[Union[int, str, None]],
        elem_type: int = ONNX_FLOAT,
    ) -> None:
        """Registers a graph output ``ValueInfoProto``."""
        self.outputs.append(encode_value_info(name, dims, elem_type))

    def add_initializer(self, name: str, array: np.ndarray) -> str:
        """Adds a constant weight tensor ``TensorProto`` to the graph initializers."""
        self.initializers.append(encode_tensor_proto(name, array))
        return name

    def add_node(
        self,
        op_type: str,
        inputs: Sequence[str],
        outputs: Sequence[str],
        *,
        name: Optional[str] = None,
        domain: str = "",
        attributes: Optional[Sequence[bytes]] = None,
    ) -> None:
        """Appends an operator ``NodeProto`` to the graph."""
        node_name = name or self.unique_name(op_type.lower())
        self.nodes.append(
            encode_node(
                op_type=op_type,
                inputs=inputs,
                outputs=outputs,
                name=node_name,
                domain=domain,
                attributes=attributes,
            )
        )

    def build_model(
        self,
        *,
        opset_version: int = 17,
        extra_opsets: Optional[Dict[str, int]] = None,
        ir_version: int = 8,
        producer_name: str = "snoozeml",
        producer_version: str = "0.1.0",
        doc_string: str = "",
    ) -> bytes:
        """Serializes the complete ONNX ``ModelProto`` into bytes."""
        graph_parts: List[bytes] = []
        for node in self.nodes:
            graph_parts.append(_encode_bytes_field(1, node))
        graph_parts.append(_encode_string_field(2, self.name))
        for init in self.initializers:
            graph_parts.append(_encode_bytes_field(5, init))
        for inp in self.inputs:
            graph_parts.append(_encode_bytes_field(11, inp))
        for out in self.outputs:
            graph_parts.append(_encode_bytes_field(12, out))
        graph_proto = b"".join(graph_parts)

        model_parts: List[bytes] = [
            _encode_varint_field(1, int(ir_version)),
            _encode_string_field(2, producer_name),
            _encode_string_field(3, producer_version),
            _encode_varint_field(5, 1),  # model_version = 1
        ]
        if doc_string:
            model_parts.append(_encode_string_field(6, doc_string))
        model_parts.append(_encode_bytes_field(7, graph_proto))

        # Default ai.onnx domain ("")
        default_opset = _encode_string_field(1, "") + _encode_varint_field(
            2, int(opset_version)
        )
        model_parts.append(_encode_bytes_field(8, default_opset))

        if extra_opsets:
            for dom, ver in extra_opsets.items():
                opset_msg = _encode_string_field(1, dom) + _encode_varint_field(
                    2, int(ver)
                )
                model_parts.append(_encode_bytes_field(8, opset_msg))

        return b"".join(model_parts)
