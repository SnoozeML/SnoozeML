Quickstart & Client Usage
=========================

Installation
------------

Install the ``snoozeml`` package from the ``python/`` directory of the repository:

.. code-block:: bash

   # Core HTTP + gRPC client & native ONNX/XGBoost converters
   pip install ./python

   # Include optional ML converter dependencies (scikit-learn, torch, skl2onnx, xgboost)
   pip install "./python[converters]"

Connecting to a SnoozeML Daemon
-------------------------------

:class:`~snoozeml.SnoozeClient` (synchronous) and :class:`~snoozeml.AsyncSnoozeClient` (asynchronous) support both HTTP REST (default port ``8080``) and native gRPC (default port ``50051``):

.. code-block:: python

   from snoozeml import SnoozeClient

   # Connect with gRPC enabled for inference and HTTP for lifecycle management
   client = SnoozeClient(
       http_endpoint="http://localhost:8080",
       grpc_endpoint="localhost:50051",
       use_grpc=True,
       api_key="sk-admin-key",  # or set SNOOZE_API_KEY env var
   )

Model Lifecycle & Scale-to-Zero Management
------------------------------------------

Use ``client.models`` (:class:`~snoozeml.models.ModelsManager`) to inspect model states, pin critical models in memory, or trigger manual scale-to-zero evictions:

.. code-block:: python

   # List all registered models
   for model in client.models.list():
       print(f"{model.id}: state={model.state.value}, loaded={model.is_loaded}, pinned={model.pinned}")

   # Inspect a specific model
   info = client.models.get("demo-model")
   print(f"Path: {info.onnx_path}, Idle Timeout: {info.idle_timeout_secs}s")

   # Pin model in memory (exempts from scale-to-zero sweeping and LRU eviction)
   client.models.pin("demo-model")

   # Unpin and explicitly evict/unload weights from RAM/VRAM
   client.models.unpin("demo-model")
   client.models.evict("demo-model")

Uploading Models via HTTP (Files & In-Memory Bytes)
---------------------------------------------------

Deploy models directly via HTTP without volume mounts or daemon restarts. Supports local file paths or raw in-memory bytes:

.. code-block:: python

   # Option A: Upload from local file path with custom runtime configuration
   client.upload_model(
       file="models/iris_classifier.onnx",
       model_id="iris_classifier",
       max_batch_size=32,
       idle_timeout_secs=120,
       pinned=True,
   )

   # Option B: Upload raw in-memory bytes without saving to disk
   raw_bytes = booster.save_raw(raw_format="json")
   client.upload_model(
       file=raw_bytes,
       filename="cancer_xgboost.xgb.json",  # Required for format detection when uploading bytes
       model_id="cancer_xgboost",
   )

   # Option C: Refresh models directory on server at runtime
   refreshed = client.refresh_models()

Running Tensor Inference
------------------------


Pass a dictionary mapping input binding names to NumPy arrays (or a single ``np.ndarray``) to :meth:`~snoozeml.SnoozeClient.infer`:

.. code-block:: python

   import numpy as np

   input_tensor = np.random.randn(4, 10).astype(np.float32)

   response = client.infer(
       model_id="demo-model",
       inputs={"input": input_tensor},
       use_grpc=True,
   )

   # Extract reshaped float32 NumPy array by output binding name
   output = response.get_numpy("output")
   print(f"Shape: {output.shape}, Server Duration: {response.duration_ms:.2f} ms")

Asynchronous Inference & Micro-Batching
---------------------------------------

Use :class:`~snoozeml.AsyncSnoozeClient` with ``asyncio.gather`` to dispatch concurrent requests that SnoozeML automatically coalesces into dynamic micro-batches along ``Axis(0)``:

.. code-block:: python

   import asyncio
   import numpy as np
   from snoozeml import AsyncSnoozeClient

   async def run_batch():
       async with AsyncSnoozeClient(endpoint="localhost:50051", use_grpc=True) as client:
           requests = [
               client.infer("demo-model", {"input": np.ones((1, 10), dtype=np.float32)})
               for _ in range(16)
           ]
           responses = await asyncio.gather(*requests)
           return [r.get_numpy("output") for r in responses]

   outputs = asyncio.run(run_batch())
