Model Converters & One-Step Deployment
=======================================

``snoozeml.converters`` provides one-function export pipelines that convert trained Python models into ``.onnx`` or ``.xgb.json`` artifacts ready for SnoozeML's dynamic micro-batching engine.

In addition, :class:`~snoozeml.SnoozeClient` provides one-step deployment methods (:meth:`~snoozeml.SnoozeClient.deploy_sklearn`, :meth:`~snoozeml.SnoozeClient.deploy_torch`, :meth:`~snoozeml.SnoozeClient.deploy_xgboost`) that convert, upload, and register models on the server without touching the filesystem.

One-Step In-Memory Deployment
-----------------------------

Deploy trained models directly to a running SnoozeML instance without creating local files:

.. code-block:: python

   from snoozeml import SnoozeClient
   from sklearn.ensemble import RandomForestClassifier
   import numpy as np

   client = SnoozeClient("http://localhost:8080", api_key="admin-key")

   # 1. Scikit-learn one-step deployment
   clf = RandomForestClassifier().fit(X_train, y_train)
   client.deploy_sklearn(
       model=clf,
       sample_input=X_train[:5],
       model_id="iris_rf",
       max_batch_size=32,
       idle_timeout_secs=60,
       load=True,
       overwrite=True,
   )

   # 2. PyTorch one-step deployment
   client.deploy_torch(
       model=torch_model,
       sample_input=sample_tensor,
       model_id="torch_mlp",
       max_batch_size=64,
       overwrite=True,
   )

   # 3. XGBoost one-step deployment (pure-Rust tree format)
   client.deploy_xgboost(
       model=xgb_booster,
       model_id="fraud_tree",
       max_batch_size=128,
       overwrite=True,
   )

1. Standalone scikit-learn Export (``export_sklearn``)
------------------------------------------------------

:func:`~snoozeml.converters.export_sklearn` exports fitted scikit-learn estimators (``RandomForestClassifier``, ``GradientBoostingClassifier``, ``LogisticRegression``, ``LinearRegression``, ``MLPClassifier``, etc.) and ``Pipeline`` objects to ONNX with dynamic ``batch_size`` along ``Axis(0)``:

.. code-block:: python

   import numpy as np
   from sklearn.datasets import load_iris
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.pipeline import make_pipeline
   from sklearn.preprocessing import StandardScaler
   from snoozeml.converters import export_sklearn

   iris = load_iris()
   X, y = iris.data.astype(np.float32), iris.target

   pipe = make_pipeline(
       StandardScaler(),
       RandomForestClassifier(n_estimators=20, max_depth=4, random_state=42),
   )
   pipe.fit(X, y)

   onnx_path = export_sklearn(
       model=pipe,
       initial_types=[("float_input", (None, 4))],
       output_path="models/iris_rf_pipeline.onnx",
   )

2. Standalone PyTorch Export (``export_torch``)
-----------------------------------------------

:func:`~snoozeml.converters.export_torch` exports PyTorch ``nn.Module`` models to ONNX while preserving the original ``training`` mode and automatically configuring dynamic batch axes on dimension ``0``:

.. code-block:: python

   import torch
   import torch.nn as nn
   from snoozeml.converters import export_torch

   model = nn.Sequential(
       nn.Linear(16, 32),
       nn.ReLU(),
       nn.Linear(32, 4),
   ).eval()

   dummy_input = torch.randn(1, 16)
   onnx_path = export_torch(
       model=model,
       sample_input=dummy_input,
       dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
       output_path="models/torch_mlp.onnx",
   )

.. note::

   Exporting PyTorch models only requires the lightweight CPU wheel of PyTorch. To avoid downloading ~2.5 GB of unused CUDA packages on Linux, install PyTorch from its official CPU index:

   .. code-block:: bash

      pip install torch --index-url https://download.pytorch.org/whl/cpu

3. Standalone XGBoost Export (``export_xgboost``)
-------------------------------------------------

:func:`~snoozeml.converters.export_xgboost` exports ``xgboost.Booster``, ``XGBClassifier``, and ``XGBRegressor`` models to ``.xgb.json`` for zero-FFI execution on SnoozeML's pure-Rust ``hessboost`` backend:

.. code-block:: python

   import xgboost as xgb
   from snoozeml.converters import export_xgboost

   clf = xgb.XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.1)
   clf.fit(X, y)

   xgb_path = export_xgboost(clf, output_path="models/iris_tree.xgb.json")
