"""Official Python SDK and ML conversion toolkit for the SnoozeML inference daemon."""

from .exceptions import (
    AuthenticationError,
    ConversionError,
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
)
from .inference import (
    InferenceResponse,
    OutputList,
    TensorDataType,
    TensorInput,
    TensorOutput,
    normalize_inputs,
)
from .logging import (
    RedactingFilter,
    StructuredFormatter,
    configure_logging,
    get_logger,
    mask_secret,
)
from .models import AsyncModelsManager, ModelInfo, ModelsManager, ModelState
from .worker import RawTensor, SnoozeModel, WorkerError
from .client import AsyncSnoozeClient, SnoozeClient

# Alias for convenience and compatibility with design documentation
Client = SnoozeClient
AsyncClient = AsyncSnoozeClient

try:
    from .converters import export_sklearn, export_torch, export_xgboost
except ImportError:
    pass

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "SnoozeClient",
    "AsyncSnoozeClient",
    "Client",
    "AsyncClient",
    "ModelInfo",
    "ModelState",
    "ModelsManager",
    "AsyncModelsManager",
    "InferenceResponse",
    "OutputList",
    "TensorInput",
    "TensorOutput",
    "TensorDataType",
    "normalize_inputs",
    "SnoozeError",
    "ModelNotFoundError",
    "InvalidInputError",
    "ShapeMismatchError",
    "InferenceTimeoutError",
    "ModelLoadingError",
    "ResourceExhaustedError",
    "OutOfMemoryError",
    "UnsupportedArchitectureError",
    "AuthenticationError",
    "PermissionDeniedError",
    "SnoozeConnectionError",
    "RuntimeInferenceError",
    "ConversionError",
    "export_sklearn",
    "export_torch",
    "export_xgboost",
    "configure_logging",
    "get_logger",
    "mask_secret",
    "RedactingFilter",
    "StructuredFormatter",
]
