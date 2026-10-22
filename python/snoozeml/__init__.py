"""Official Python SDK and ML conversion toolkit for the SnoozeML inference daemon."""

from .auth import (
    JwtAlgorithm,
    Role,
    create_jwt_token,
    decode_jwt_token,
    mint_jwt_token,
)
from .config import (
    ApiKeyEntry,
    AuthSettings,
    GlobalDefaultsConfig,
    JwtSettings,
    ModelConfigEntry,
    SnoozeConfig,
    SnoozeDaemon,
    StreamBinding,
    find_snooze_binary,
    scaffold_worker,
)
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
    concat_inference_responses,
    normalize_inputs,
    slice_inputs_batch,
)
from .logging import (
    RedactingFilter,
    StructuredFormatter,
    configure_logging,
    get_logger,
    mask_secret,
)
from .metrics import DaemonMetrics, parse_prometheus_metrics
from .models import AsyncModelsManager, ModelInfo, ModelsManager, ModelState
from .streams import (
    MockStreamClient,
    RedisStreamClient,
    StreamInferRequest,
    StreamInferResponse,
    StreamWireFormat,
)
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
    "Role",
    "JwtAlgorithm",
    "create_jwt_token",
    "mint_jwt_token",
    "decode_jwt_token",
    "SnoozeConfig",
    "GlobalDefaultsConfig",
    "ModelConfigEntry",
    "StreamBinding",
    "ApiKeyEntry",
    "JwtSettings",
    "AuthSettings",
    "SnoozeDaemon",
    "find_snooze_binary",
    "scaffold_worker",
    "DaemonMetrics",
    "parse_prometheus_metrics",
    "StreamWireFormat",
    "StreamInferRequest",
    "StreamInferResponse",
    "MockStreamClient",
    "RedisStreamClient",
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
    "slice_inputs_batch",
    "concat_inference_responses",
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
