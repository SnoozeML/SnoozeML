from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class DataType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    DATA_TYPE_UNSPECIFIED: _ClassVar[DataType]
    DATA_TYPE_FP32: _ClassVar[DataType]
    DATA_TYPE_FP16: _ClassVar[DataType]
    DATA_TYPE_BF16: _ClassVar[DataType]
    DATA_TYPE_INT32: _ClassVar[DataType]
    DATA_TYPE_INT64: _ClassVar[DataType]
    DATA_TYPE_UINT8: _ClassVar[DataType]
    DATA_TYPE_BOOL: _ClassVar[DataType]
DATA_TYPE_UNSPECIFIED: DataType
DATA_TYPE_FP32: DataType
DATA_TYPE_FP16: DataType
DATA_TYPE_BF16: DataType
DATA_TYPE_INT32: DataType
DATA_TYPE_INT64: DataType
DATA_TYPE_UINT8: DataType
DATA_TYPE_BOOL: DataType

class TensorData(_message.Message):
    __slots__ = ("name", "shape", "data_type", "raw_data")
    NAME_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    DATA_TYPE_FIELD_NUMBER: _ClassVar[int]
    RAW_DATA_FIELD_NUMBER: _ClassVar[int]
    name: str
    shape: _containers.RepeatedScalarFieldContainer[int]
    data_type: DataType
    raw_data: bytes
    def __init__(self, name: _Optional[str] = ..., shape: _Optional[_Iterable[int]] = ..., data_type: _Optional[_Union[DataType, str]] = ..., raw_data: _Optional[bytes] = ...) -> None: ...

class ModelInferRequest(_message.Message):
    __slots__ = ("model_id", "inputs", "client_timeout_ms", "parameters")
    class ParametersEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    INPUTS_FIELD_NUMBER: _ClassVar[int]
    CLIENT_TIMEOUT_MS_FIELD_NUMBER: _ClassVar[int]
    PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    inputs: _containers.RepeatedCompositeFieldContainer[TensorData]
    client_timeout_ms: int
    parameters: _containers.ScalarMap[str, str]
    def __init__(self, model_id: _Optional[str] = ..., inputs: _Optional[_Iterable[_Union[TensorData, _Mapping]]] = ..., client_timeout_ms: _Optional[int] = ..., parameters: _Optional[_Mapping[str, str]] = ...) -> None: ...

class ModelInferResponse(_message.Message):
    __slots__ = ("model_id", "outputs", "duration_ms")
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    DURATION_MS_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    outputs: _containers.RepeatedCompositeFieldContainer[TensorData]
    duration_ms: float
    def __init__(self, model_id: _Optional[str] = ..., outputs: _Optional[_Iterable[_Union[TensorData, _Mapping]]] = ..., duration_ms: _Optional[float] = ...) -> None: ...
