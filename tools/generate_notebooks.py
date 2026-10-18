#!/usr/bin/env python3
"""Script to generate SnoozeML example Jupyter notebooks."""

import json
from pathlib import Path

def create_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

def md_cell(source):
    lines = [f"{line}\n" for line in source.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines
    }

def code_cell(source):
    lines = [f"{line}\n" for line in source.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines
    }

def generate_notebooks(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Scikit-Learn ONNX Notebook
    nb1_cells = [
        md_cell("""# 🌸 Scikit-learn ONNX Inference with SnoozeML

This notebook demonstrates the end-to-end workflow:
1. **Train** a `RandomForestClassifier` on the standard Iris dataset.
2. **Export** the model to ONNX using `snoozeml.converters.export_sklearn`.
3. **Upload / Deploy** the model to SnoozeML (via Docker volume mount `./models:/models`).
4. **Infer** on the test set via the `SnoozeClient` and verify predictions."""),
        code_cell("""from pathlib import Path
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from snoozeml import SnoozeClient
from snoozeml.converters import export_sklearn"""),
        md_cell("## 1. Load Data & Train Model"),
        code_cell("""# Load the classic Iris dataset (4 features, 3 classes) and split 80/20
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit a simple Random Forest classifier
clf = RandomForestClassifier(n_estimators=20, random_state=42).fit(X_train, y_train)
print(f"Training accuracy: {clf.score(X_train, y_train):.4f}")"""),
        md_cell("""## 2. Export & Upload to SnoozeML

In Docker (`docker compose up`), the repository's `./models` folder is mounted into `/models`.
Writing the `.onnx` model here makes it immediately accessible to the SnoozeML daemon."""),
        code_cell("""# Resolve the models directory (relative to this notebook)
models_dir = Path("../models") if Path("../models").exists() else Path("./models")
onnx_path = models_dir / "iris_classifier.onnx"

# Export scikit-learn model to single-tensor FP32 ONNX
export_sklearn(clf, sample_input=X_train, input_name="input", output_path=onnx_path)
print(f"Exported model to {onnx_path} ({onnx_path.stat().st_size} bytes)")"""),
        md_cell("""> **Note on Docker:** If SnoozeML was already running before exporting this model, restart the container once so it auto-discovers the new model on startup:
> ```bash
> docker compose restart snoozeml
> ```"""),
        md_cell("## 3. Run Inference via SnoozeClient"),
        code_cell("""# Connect to SnoozeML daemon (defaults to http://localhost:8080)
client = SnoozeClient(endpoint="http://localhost:8080")

# Run inference on the test set (batch of 30 samples)
response = client.infer(model_id="iris_classifier", inputs={"input": X_test})
probs = response.get_numpy() # shape: (30, 3)
preds = probs.argmax(axis=1)

# Evaluate results
print(f"SnoozeML Latency:        {response.duration_ms:.2f} ms")
print(f"SnoozeML Test Accuracy:  {accuracy_score(y_test, preds):.4f}")
print(f"Predicted classes:       {preds[:5]}")
print(f"Ground truth classes:    {y_test[:5]}")""")
    ]

    nb1_path = out_dir / "01_sklearn_onnx.ipynb"
    with open(nb1_path, "w", encoding="utf-8") as f:
        json.dump(create_notebook(nb1_cells), f, indent=1)
    print(f"Generated {nb1_path}")

    # 2. XGBoost Tree Notebook
    nb2_cells = [
        md_cell("""# 🌲 XGBoost Pure-Rust Tree Inference with SnoozeML

This notebook demonstrates:
1. **Train** an XGBoost gradient boosted tree on the standard Breast Cancer dataset.
2. **Export** directly to JSON via `snoozeml.converters.export_xgboost`.
3. **Upload / Deploy** to SnoozeML's native pure-Rust `hessboost` tree runtime (zero C++ dependencies).
4. **Infer** on test data using `SnoozeClient` and evaluate ROC-AUC."""),
        code_cell("""from pathlib import Path
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import xgboost as xgb

from snoozeml import SnoozeClient
from snoozeml.converters import export_xgboost"""),
        md_cell("## 1. Load Data & Train XGBoost"),
        code_cell("""# Load Breast Cancer dataset (30 features, binary classification)
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit an XGBoost classifier
model = xgb.XGBClassifier(n_estimators=25, max_depth=3, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)
print(f"Local train accuracy: {model.score(X_train, y_train):.4f}")"""),
        md_cell("""## 2. Export & Upload to SnoozeML

SnoozeML discovers any `.xgb.json` file in `./models` and serves it via its pure-Rust `hessboost` engine."""),
        code_cell("""# Save directly into the mounted SnoozeML models directory
models_dir = Path("../models") if Path("../models").exists() else Path("./models")
model_path = models_dir / "cancer_xgboost.xgb.json"

export_xgboost(model, output_path=model_path)
print(f"Exported tree model to {model_path} ({model_path.stat().st_size} bytes)")"""),
        md_cell("""> **Note on Docker:** If SnoozeML was already running before exporting this model, restart the container:
> ```bash
> docker compose restart snoozeml
> ```"""),
        md_cell("## 3. Run Inference via SnoozeClient"),
        code_cell("""# Connect to SnoozeML daemon
client = SnoozeClient(endpoint="http://localhost:8080")

# Run inference on test set
response = client.infer(model_id="cancer_xgboost", inputs={"features": X_test})
probs = response.get_numpy().flatten()
preds = (probs >= 0.5).astype(int)

# Evaluate results
print(f"SnoozeML Latency:  {response.duration_ms:.2f} ms")
print(f"Test ROC-AUC:      {roc_auc_score(y_test, probs):.4f}")
print(f"Test Accuracy:     {accuracy_score(y_test, preds):.4f}")
print(f"Predicted labels:  {preds[:5]}")
print(f"Ground truth:      {y_test[:5]}")""")
    ]

    nb2_path = out_dir / "02_xgboost_tree.ipynb"
    with open(nb2_path, "w", encoding="utf-8") as f:
        json.dump(create_notebook(nb2_cells), f, indent=1)
    print(f"Generated {nb2_path}")

    # 3. Scikit-Learn Regression Pipeline Notebook
    nb3_cells = [
        md_cell("""# 📈 Scikit-learn Regression Pipeline ONNX with SnoozeML

This notebook demonstrates:
1. **Train** a scikit-learn `Pipeline` (`StandardScaler` + `Ridge` regression) on the standard Diabetes dataset.
2. **Export** the multi-step preprocessing and regressor pipeline to ONNX.
3. **Upload / Deploy** to SnoozeML.
4. **Infer** continuous targets on unseen test samples and evaluate $R^2$ and MSE."""),
        code_cell("""from pathlib import Path
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error

from snoozeml import SnoozeClient
from snoozeml.converters import export_sklearn"""),
        md_cell("## 1. Load Data & Fit Pipeline"),
        code_cell("""# Load Diabetes regression dataset (10 features)
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an end-to-end Pipeline with scaling and ridge regression
pipe = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X_train, y_train)
print(f"Training R² score: {pipe.score(X_train, y_train):.4f}")"""),
        md_cell("""## 2. Export & Upload Pipeline to SnoozeML

SnoozeML compiles the scaler normalization and ridge linear regression into a fused ONNX graph."""),
        code_cell("""# Export pipeline directly to the SnoozeML models directory
models_dir = Path("../models") if Path("../models").exists() else Path("./models")
model_path = models_dir / "diabetes_pipeline.onnx"

export_sklearn(pipe, sample_input=X_train, input_name="input", output_path=model_path)
print(f"Pipeline exported to {model_path} ({model_path.stat().st_size} bytes)")"""),
        md_cell("""> **Note on Docker:** If SnoozeML was already running, restart the container:
> ```bash
> docker compose restart snoozeml
> ```"""),
        md_cell("## 3. Run Inference via SnoozeClient"),
        code_cell("""# Connect to SnoozeML daemon
client = SnoozeClient(endpoint="http://localhost:8080")

# Run inference on test partition
response = client.infer(model_id="diabetes_pipeline", inputs={"input": X_test})
preds = response.get_numpy().flatten()

# Evaluate regression metrics
print(f"SnoozeML Latency:    {response.duration_ms:.2f} ms")
print(f"SnoozeML Test R²:    {r2_score(y_test, preds):.4f}")
print(f"SnoozeML Test MSE:   {mean_squared_error(y_test, preds):.2f}")
print(f"Predicted targets:   {preds[:5].round(1)}")
print(f"Ground truth:        {y_test[:5].round(1)}")""")
    ]

    nb3_path = out_dir / "03_pipeline_regression_onnx.ipynb"
    with open(nb3_path, "w", encoding="utf-8") as f:
        json.dump(create_notebook(nb3_cells), f, indent=1)
    print(f"Generated {nb3_path}")

if __name__ == "__main__":
    out_dir = Path("examples")
    generate_notebooks(out_dir)
