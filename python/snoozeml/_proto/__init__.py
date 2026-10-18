"""Internal generated Protocol Buffer and gRPC stubs for SnoozeML."""

from .snoozeml.v1.inference_pb2 import (
    DataType,
    ModelInferRequest,
    ModelInferResponse,
    TensorData,
)
from .snoozeml.v1.inference_pb2_grpc import (
    InferenceService,
    InferenceServiceServicer,
    InferenceServiceStub,
    add_InferenceServiceServicer_to_server,
)

__all__ = [
    "DataType",
    "ModelInferRequest",
    "ModelInferResponse",
    "TensorData",
    "InferenceService",
    "InferenceServiceServicer",
    "InferenceServiceStub",
    "add_InferenceServiceServicer_to_server",
]
