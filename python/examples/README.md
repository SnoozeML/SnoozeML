# SnoozeML Python Examples

Interactive Jupyter notebook examples demonstrate training on standard datasets, exporting to SnoozeML formats, deploying via Docker, and querying with `snoozeml`:

👉 **The complete interactive notebooks and guide are located at the repository root [`examples/`](../../examples):**

1. [**`01_sklearn_onnx.ipynb`**](../../examples/01_sklearn_onnx.ipynb): Scikit-learn `RandomForestClassifier` on Iris to ONNX.
2. [**`02_xgboost_tree.ipynb`**](../../examples/02_xgboost_tree.ipynb): XGBoost `XGBClassifier` on Breast Cancer to native pure-Rust tree format (`.xgb.json`).
3. [**`03_pipeline_regression_onnx.ipynb`**](../../examples/03_pipeline_regression_onnx.ipynb): Scikit-learn `Pipeline` (`StandardScaler` + `Ridge`) on Diabetes to ONNX.

See [**`examples/README.md`**](../../examples/README.md) for full setup instructions.
