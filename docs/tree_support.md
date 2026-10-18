# Pure-Rust Tree Backend (`hessboost` / XGBoost)

SnoozeML provides a high-performance, native **Tree** execution backend supporting Gradient Boosted Decision Trees (GBDT) and ensemble models exported from **XGBoost**, with zero C/C++ runtime dependencies and zero FFI, powered by [`hessboost`](https://crates.io/crates/hessboost) ([GitHub: brndnmtthws/hessboost](https://github.com/brndnmtthws/hessboost)).

---

## 1. Overview & Key Advantages

Traditional deployment of XGBoost and tree models in production often requires linking against libxgboost dynamic libraries, installing CMake and C++ toolchains, or spinning up Python Flask/FastAPI microservices with high memory footprints and Python GIL bottlenecks.

SnoozeML's `tree` backend solves this:
- **Zero C/C++ or FFI Dependencies**: 100% safe, pure-Rust tree traversal and matrix operations.
- **Full XGBoost 3.x Schema Compatibility**: Directly loads JSON models saved via `booster.save_model("model.json")`.
- **Micro-Batching & Multi-Threading**: Parallel evaluation accelerated across host CPU cores using Rayon.
- **Scale-to-Zero & Cold Starts**: Models sleep when inactive, freeing memory until an inference request arrives.
- **Zero-Panic Policy**: Production-grade bounds checking and error handling on every prediction path.

---

## 2. Supported Formats & File Extensions

The Tree backend automatically discovers and loads models from the following file formats:

| Format / Extension | Origin | Description |
| :--- | :--- | :--- |
| `*.xgb.json` / `*.tree.json` | Python / XGBoost | Standard XGBoost JSON model format (`booster.save_model("model.xgb.json")`). |
| `*.json` | Python / XGBoost | Generic JSON model automatically inspected for GBDT/XGBoost structures. |
| `*.xgb` / `*.tree` | Python / XGBoost / SnoozeML | Explicit tree model binaries or JSON payloads. |
| `*.bin` | Pure Rust (`hessboost`) | Native compact binary serialization format (`model.save_binary(...)`). |

### Auto-Detection
When discovering models in your `models/` directory:
- Files with extensions `.xgb.json`, `.tree.json`, `.xgb`, or `.tree` are automatically recognized as tree models.
- Standard `.json` files (excluding configuration files such as `config.json` or `tokenizer.json`) are checked for XGBoost structure tags (`"gradient_booster"`, `"gbtree"`, `"trees"`).
- In `models.toml`, you can explicitly set `device = "tree"` or `provider = "tree"`.

---

## 3. Supported Objectives & Tasks

The backend supports standard XGBoost objectives:

- **Regression**: `reg:squarederror`, `reg:squaredlogerror`, `reg:absoluteerror`.
- **Binary Classification**: `binary:logistic` (returns sigmoid probabilities $\in [0, 1]$), `binary:logitraw` (returns raw margins).
- **Multiclass Classification**: `multi:softprob` (returns an $N \times K$ probability matrix), `multi:softmax` (returns discrete class indices).
- **Count & Ranking**: `count:poisson`, `rank:pairwise`, `rank:ndcg`, `rank:map`.

---

## 4. Easy Model Training & Export

SnoozeML includes a dedicated training utility at `tools/train_tree.py`:

```bash
# 1. Train on built-in Iris dataset (multiclass classification, 3 classes, 4 features):
python tools/train_tree.py --dataset iris --name iris_model

# 2. Train on California Housing dataset (regression, 8 features):
python tools/train_tree.py --dataset california --name housing_model

# 3. Train on Breast Cancer dataset (binary classification, 30 features):
python tools/train_tree.py --dataset breast_cancer --name cancer_model

# 4. Train from your own custom CSV dataset:
python tools/train_tree.py --csv dataset.csv --target price --name custom_pricing
```

### Python Code Example (Standard XGBoost)

You can also train and save models directly with standard Python code:

```python
import xgboost as xgb
import numpy as np

# Prepare training data
X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
y = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
dtrain = xgb.DMatrix(X, label=y)

# Train model
params = {
    "max_depth": 3,
    "eta": 0.1,
    "objective": "binary:logistic",
}
booster = xgb.train(params, dtrain, num_boost_round=50)

# Save directly into your SnoozeML models directory:
booster.save_model("models/my_classifier.xgb.json")
```

---

## 5. Configuration (`models.toml`)

Add your tree model to `models.toml` (or `snooze.toml`):

```toml
[[models]]
id = "my-classifier"
path = "my_classifier.xgb.json"
description = "Production XGBoost Binary Classifier"
device = "tree"
max_batch_size = 64
max_batch_delay_ms = 5
idle_timeout_secs = 300
```

---

## 6. HTTP Inference REST API

### Request
Send a POST request to `/v1/models/{model_id}/infer`:

```bash
curl -X POST http://127.0.0.1:8080/v1/models/my-classifier/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "name": "features",
        "shape": [2, 2],
        "data": [1.5, 2.5, 6.5, 7.5]
      }
    ]
  }'
```

### Response
```json
{
  "model_id": "my-classifier",
  "outputs": [
    {
      "name": "predictions",
      "shape": [2, 1],
      "data": [0.124589, 0.897412]
    }
  ],
  "duration_ms": 0.42
}
```

For multiclass models with $K$ classes, the output shape is `[batch_size, K]`, returning the full probability distribution across all classes per sample.

---

## 7. Verification & End-to-End Parity Testing

To verify numerical parity between Python XGBoost and SnoozeML:

```bash
python tools/train_tree.py --dataset iris --name iris_model --verify
```

This trains the model, exports the JSON artifact, triggers inference on the running SnoozeML instance, and prints a side-by-side comparison confirming exact matching outputs.
