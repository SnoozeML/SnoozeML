Structured Logging & Error Handling
===================================

Structured Logging & Secret Redaction
-------------------------------------

The SDK logs under the ``"snoozeml"`` logger namespace with a default :class:`logging.NullHandler` attached. Call :func:`~snoozeml.configure_logging` to enable structured key-value or JSON logging:

.. code-block:: python

   from snoozeml import configure_logging

   # Human-readable structured logs:
   # 2026-10-22T21:30:00.000Z [DEBUG] snoozeml.client: Completed gRPC inference model_id=demo-model transport=grpc
   configure_logging(level="DEBUG", format_json=False)

   # Or single-line JSON logs for log aggregators (Datadog, Loki, ELK):
   configure_logging(level="INFO", format_json=True)

All loggers under ``"snoozeml"`` attach :class:`~snoozeml.logging.RedactingFilter`, which automatically scrubs ``Bearer`` tokens, ``x-api-key`` headers, and any API keys or JWT tokens passed to :class:`~snoozeml.SnoozeClient`.

Scale-to-Zero Error Handling & Automatic Retries
------------------------------------------------

Under SnoozeML's scale-to-zero model, idle models are evicted from VRAM/RAM. When an incoming inference request targets an unloaded model:

1. The server transitions the model to the ``Loading`` state and returns HTTP ``503 Service Unavailable`` with a ``Retry-After: 5`` header (or gRPC code ``Unavailable``).
2. If GPU memory is saturated, the server performs LRU deallocation and returns ``ResourceExhausted`` (HTTP 503).

:class:`~snoozeml.SnoozeClient` handles these transient states automatically via ``max_retries`` and ``retry_backoff``:

.. code-block:: python

   from snoozeml import SnoozeClient

   # The client automatically parses 'Retry-After' headers and retries with exponential backoff
   client = SnoozeClient(
       "http://localhost:8080",
       max_retries=3,
       retry_backoff=0.25,
   )

   # Or block explicitly until the model is loaded in memory
   client.models.wait_for_ready("my-model", timeout=30.0, trigger_load=True)

Defensive Error Handling Pattern
--------------------------------

For fine-grained control, catch domain exceptions from :mod:`snoozeml.exceptions`:

.. code-block:: python

   import time
   from snoozeml import SnoozeClient
   from snoozeml.exceptions import (
       ModelLoadingError,
       ResourceExhaustedError,
       InvalidInputError,
       SnoozeError,
   )

   client = SnoozeClient("http://localhost:8080")

   try:
       response = client.infer("churn_model", {"features": [[1.0, 0.0, 24.5]]})
   except ModelLoadingError:
       print("Model is currently loading weights into VRAM.")
   except ResourceExhaustedError as e:
       print(f"Server VRAM saturated. Suggested retry delay: {e.retry_after} seconds.")
   except InvalidInputError as e:
       print(f"Tensor shape or dtype mismatch: {e}")
   except SnoozeError as e:
       print(f"General SnoozeML error: {e}")

Canonical Error Mapping Parity (HTTP & gRPC)
--------------------------------------------

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
