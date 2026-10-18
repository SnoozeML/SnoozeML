Structured Logging & Error Handling
===================================

Structured Logging & Secret Redaction
-------------------------------------

The SDK logs under the ``"snoozeml"`` logger namespace with a default :class:`logging.NullHandler` attached. Call :func:`~snoozeml.configure_logging` to enable structured key-value or JSON logging:

.. code-block:: python

   from snoozeml import configure_logging

   # Human-readable structured logs:
   # 2026-10-16T23:30:00.000Z [DEBUG] snoozeml.client: Completed gRPC inference model_id=demo-model transport=grpc
   configure_logging(level="DEBUG", format_json=False)

   # Or single-line JSON logs for log aggregators (Datadog, Loki, ELK):
   configure_logging(level="INFO", format_json=True)

All loggers under ``"snoozeml"`` attach :class:`~snoozeml.logging.RedactingFilter`, which automatically scrubs ``Bearer`` tokens, ``x-api-key`` headers, and any API keys or JWT tokens passed to :class:`~snoozeml.SnoozeClient`.

Error Mapping Parity (HTTP & gRPC)
----------------------------------

Both HTTP REST and gRPC transports map server errors to the same :class:`~snoozeml.exceptions.SnoozeError` exception hierarchy:

.. list-table:: Canonical Error Mapping
   :header-rows: 1
   :widths: 32 18 22 28

   * - Python Exception
     - HTTP Status
     - gRPC Code
     - Description
   * - :class:`~snoozeml.exceptions.ModelNotFoundError`
     - ``404``
     - ``NOT_FOUND``
     - Unregistered ``model_id``
   * - :class:`~snoozeml.exceptions.InvalidInputError`
     - ``400``
     - ``INVALID_ARGUMENT``
     - Malformed tensor shape, dtype, or element count
   * - :class:`~snoozeml.exceptions.ShapeMismatchError`
     - ``400``
     - ``INVALID_ARGUMENT``
     - Incompatible tensor shapes during collation
   * - :class:`~snoozeml.exceptions.AuthenticationError`
     - ``401``
     - ``UNAUTHENTICATED``
     - Missing or invalid API key / JWT token
   * - :class:`~snoozeml.exceptions.PermissionDeniedError`
     - ``403``
     - ``PERMISSION_DENIED``
     - Caller role lacks permission for the operation
   * - :class:`~snoozeml.exceptions.InferenceTimeoutError`
     - ``504``
     - ``DEADLINE_EXCEEDED``
     - Request exceeded processing or queue deadline
   * - :class:`~snoozeml.exceptions.ModelLoadingError`
     - ``503``
     - ``UNAVAILABLE``
     - Model weights are currently loading
   * - :class:`~snoozeml.exceptions.ResourceExhaustedError`
     - ``503``
     - ``RESOURCE_EXHAUSTED``
     - Memory or queue capacity exhausted (includes ``retry_after``)
   * - :class:`~snoozeml.exceptions.OutOfMemoryError`
     - ``503``
     - ``RESOURCE_EXHAUSTED``
     - Device VRAM or host RAM allocation failure
   * - :class:`~snoozeml.exceptions.UnsupportedArchitectureError`
     - ``501``
     - ``UNIMPLEMENTED``
     - Unsupported Safetensors architecture
   * - :class:`~snoozeml.exceptions.RuntimeInferenceError`
     - ``500``
     - ``INTERNAL``
     - Internal inference engine error
