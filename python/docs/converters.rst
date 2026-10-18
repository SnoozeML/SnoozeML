Model Converters (scikit-learn, PyTorch & XGBoost)
==================================================

``snoozeml.converters`` provides one-function export pipelines that convert trained Python models into ``.onnx`` or ``.xgb.json`` artifacts ready for SnoozeML's dynamic micro-batching engine.

1. Exporting scikit-learn Models & Pipelines
--------------------------------------------

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

2. Exporting PyTorch ``nn.Module`` Models
-----------------------------------------

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

3. Exporting XGBoost Models (Pure-Rust Tree Backend)
----------------------------------------------------

:func:`~snoozeml.converters.export_xgboost` exports ``xgboost.Booster``, ``XGBClassifier``, and ``XGBRegressor`` models to ``.xgb.json`` for zero-FFI execution on SnoozeML's pure-Rust ``hessboost`` backend:

.. code-block:: python

   import xgboost as xgb
   from snoozeml.converters import export_xgboost

   clf = xgb.XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.1)
   clf.fit(X, y)

   xgb_path = export_xgboost(clf, output_path="models/iris_tree.xgb.json")
