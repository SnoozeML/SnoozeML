Quickstart & Client Usage
=========================

The SnoozeML Python SDK provides synchronous (:class:`~snoozeml.SnoozeClient`) and asynchronous (:class:`~snoozeml.AsyncSnoozeClient`) clients that connect your machine learning code to the SnoozeML daemon.

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

:class:`~snoozeml.SnoozeClient` handles both HTTP REST (for management, model upload, and lifecycle operations) and native gRPC (for high-throughput binary tensor inference):

.. code-block:: python

   from snoozeml import SnoozeClient

   # Connect with gRPC enabled for inference and HTTP for lifecycle management
   client = SnoozeClient(
       http_endpoint="http://localhost:8080",
       grpc_endpoint="localhost:50051",
       use_grpc=True,              # Route infer() calls over binary gRPC
       api_key="sk-admin-key",     # Authenticate via x-api-key / Bearer
       max_retries=3,              # Automatically retry transient 503 / Loading states
       retry_backoff=0.25,         # Exponential backoff base delay in seconds
   )

   # Context manager automatically handles connection pooling and cleanup
   with client:
       # Wait until daemon passes health probes
       client.wait_until_ready(timeout=10.0)

Running Tensor Inference
------------------------

Pass a dictionary mapping input binding names to NumPy arrays, PyTorch tensors, or Python nested lists:

.. code-block:: python

   import numpy as np
   from snoozeml import SnoozeClient

   with SnoozeClient("http://localhost:8080", use_grpc=True) as client:
       input_tensor = np.random.randn(2, 4).astype(np.float32)

       response = client.infer(
           model_id="iris_classifier",
           inputs={"input": input_tensor},
       )

       # Read output as NumPy ndarray
       predictions = response.get_numpy("output")
       print(f"Predictions shape: {predictions.shape}")
       print(f"Server latency: {response.duration_ms:.2f} ms")

Direct PyTorch Tensors Support
------------------------------

PyTorch tensors are automatically detached from autograd graphs and transferred to CPU memory before serialization:

.. code-block:: python

   import torch
   from snoozeml import SnoozeClient

   client = SnoozeClient("http://localhost:8080")

   # Tensors with requires_grad=True or on GPU are safely handled
   x = torch.randn(4, 16, requires_grad=True)
   resp = client.infer("mlp_net", {"input": x})

   # Convert output directly back to a PyTorch tensor on your target device
   out_torch = resp.get_torch("output", device="cuda:0" if torch.cuda.is_available() else "cpu")

Dataset Batch Chunking (``infer_dataset``)
------------------------------------------

When scoring large offline datasets, sending tens of thousands of rows at once can trigger network limits or GPU out-of-memory errors. The :meth:`~snoozeml.SnoozeClient.infer_dataset` method slices inputs along ``Axis(0)`` into batches and concatenates outputs automatically:

.. code-block:: python

   import numpy as np

   # 10,000 samples with 32 features each
   large_features = np.random.randn(10000, 32).astype(np.float32)

   # Slices into chunks of 128 and returns unified InferenceResponse
   full_response = client.infer_dataset(
       model_id="fraud_detector",
       inputs={"features": large_features},
       batch_size=128,
   )

   print(f"Total scored samples: {full_response.get_numpy('output').shape[0]}")

Model Lifecycle & Scale-to-Zero Management
------------------------------------------

Use ``client.models`` (:class:`~snoozeml.models.ModelsManager`) to inspect model states, pin critical models in memory, or trigger manual scale-to-zero evictions:

.. code-block:: python

   # List all registered models and inspection states
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

   # Wait for a model to finish loading its weights into memory
   client.models.wait_for_ready("demo-model", timeout=30.0, trigger_load=True)

One-Step Model Deployment & Upload
----------------------------------

Deploy trained models directly from Python memory or local files:

.. code-block:: python

   from sklearn.datasets import load_iris
   from sklearn.ensemble import RandomForestClassifier
   import numpy as np

   X, y = load_iris(return_X_y=True)
   clf = RandomForestClassifier(n_estimators=10).fit(X.astype(np.float32), y)

   # Deploy directly from memory: export -> upload -> register -> load
   model_info = client.deploy_sklearn(
       model=clf,
       sample_input=X[:5].astype(np.float32),
       model_id="iris_rf",
       overwrite=True,
       load=True,
       pinned=True,
   )
   print(f"Model deployed with state: {model_info.state}")

Asynchronous Concurrent Micro-Batching
--------------------------------------

Use :class:`~snoozeml.AsyncSnoozeClient` with ``asyncio.gather`` to dispatch concurrent requests that SnoozeML automatically coalesces into dynamic micro-batches along ``Axis(0)``:

.. code-block:: python

   import asyncio
   import numpy as np
   from snoozeml import AsyncSnoozeClient

   async def run_batch():
       async with AsyncSnoozeClient(http_endpoint="http://localhost:8080", grpc_endpoint="localhost:50051", use_grpc=True) as client:
           requests = [
               client.infer("demo-model", {"input": np.ones((1, 10), dtype=np.float32)})
               for _ in range(16)
           ]
           responses = await asyncio.gather(*requests)
           return [r.get_numpy("output") for r in responses]

   outputs = asyncio.run(run_batch())
