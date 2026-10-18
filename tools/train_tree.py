#!/usr/bin/env python3
"""SnoozeML Tree Model Training & Export Tool.

Trains Gradient Boosted Decision Tree (XGBoost) models, exports them to the standard
XGBoost JSON format supported by SnoozeML's 'tree' backend, generates sample inference
payloads, and optionally tests end-to-end serving against a running SnoozeML instance.

Requirements:
    pip install xgboost scikit-learn numpy pandas
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import urllib.error
import urllib.request

try:
    import numpy as np
except ImportError:
    print("[ERROR] numpy is required. Install via: pip install numpy", file=sys.stderr)
    sys.exit(1)

try:
    import xgboost as xgb
except ImportError:
    print("[ERROR] xgboost is required. Install via: pip install xgboost", file=sys.stderr)
    sys.exit(1)

try:
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
except ImportError:
    print("[ERROR] scikit-learn is required. Install via: pip install scikit-learn", file=sys.stderr)
    sys.exit(1)


def load_dataset(
    dataset_name: str,
    csv_path: Optional[str] = None,
    target_col: Optional[str] = None,
    n_samples: int = 500,
    n_features: int = 10,
) -> Tuple[np.ndarray, np.ndarray, List[str], str, Optional[int]]:
    """Loads dataset and returns (X, y, feature_names, default_objective, num_class)."""
    dataset_lower = dataset_name.lower().strip()

    if csv_path:
        try:
            import pandas as pd
        except ImportError:
            print("[ERROR] pandas is required to load CSV files: pip install pandas", file=sys.stderr)
            sys.exit(1)

        if not os.path.isfile(csv_path):
            print(f"[ERROR] CSV file not found: {csv_path}", file=sys.stderr)
            sys.exit(1)

        df = pd.read_csv(csv_path)
        if target_col not in df.columns:
            print(f"[ERROR] Target column '{target_col}' not found in CSV columns: {list(df.columns)}", file=sys.stderr)
            sys.exit(1)

        y_raw = df[target_col].values
        X_df = df.drop(columns=[target_col])
        X = X_df.select_dtypes(include=[np.number]).values.astype(np.float32)
        feature_names = list(X_df.select_dtypes(include=[np.number]).columns)

        # Detect classification vs regression
        unique_vals = np.unique(y_raw)
        if len(unique_vals) == 2:
            y = (y_raw == unique_vals[1]).astype(np.float32)
            return X, y, feature_names, "binary:logistic", None
        elif len(unique_vals) <= 20 and np.issubdtype(y_raw.dtype, np.integer):
            y = y_raw.astype(np.float32)
            return X, y, feature_names, "multi:softprob", len(unique_vals)
        else:
            y = y_raw.astype(np.float32)
            return X, y, feature_names, "reg:squarederror", None

    if dataset_lower == "iris":
        data = datasets.load_iris()
        X = data.data.astype(np.float32)
        y = data.target.astype(np.float32)
        return X, y, list(data.feature_names), "multi:softprob", 3

    elif dataset_lower in ("breast_cancer", "cancer"):
        data = datasets.load_breast_cancer()
        X = data.data.astype(np.float32)
        y = data.target.astype(np.float32)
        return X, y, list(data.feature_names), "binary:logistic", None

    elif dataset_lower in ("california", "housing"):
        data = datasets.fetch_california_housing()
        X = data.data.astype(np.float32)
        y = data.target.astype(np.float32)
        return X, y, list(data.feature_names), "reg:squarederror", None

    elif dataset_lower == "diabetes":
        data = datasets.load_diabetes()
        X = data.data.astype(np.float32)
        y = data.target.astype(np.float32)
        return X, y, list(data.feature_names), "reg:squarederror", None

    elif dataset_lower == "synthetic":
        X, y = datasets.make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=min(n_features, 6),
            n_classes=2,
            random_state=42,
        )
        feature_names = [f"feat_{i}" for i in range(n_features)]
        return X.astype(np.float32), y.astype(np.float32), feature_names, "binary:logistic", None

    else:
        print(f"[ERROR] Unknown dataset: '{dataset_name}'. Available: iris, breast_cancer, california, diabetes, synthetic", file=sys.stderr)
        sys.exit(1)


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    objective: str,
    num_class: Optional[int],
    n_estimators: int = 50,
    max_depth: int = 4,
    learning_rate: float = 0.1,
    random_state: int = 42,
) -> Tuple[xgb.Booster, Dict[str, Any]]:
    """Trains an XGBoost booster with the specified hyperparameters."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params: Dict[str, Any] = {
        "max_depth": max_depth,
        "eta": learning_rate,
        "objective": objective,
        "eval_metric": "rmse" if "reg" in objective else ("mlogloss" if num_class else "logloss"),
        "seed": random_state,
    }
    if num_class and num_class > 2:
        params["num_class"] = num_class

    evals = [(dtrain, "train"), (dtest, "eval")]
    evals_result: Dict[str, Any] = {}

    print(f"[*] Training XGBoost Booster:")
    print(f"    - Samples: Train = {X_train.shape[0]}, Test = {X_test.shape[0]}")
    print(f"    - Features: {X.shape[1]}")
    print(f"    - Objective: {objective}")
    print(f"    - Max Depth: {max_depth}, Eta: {learning_rate}, Rounds: {n_estimators}")

    start_time = time.time()
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=n_estimators,
        evals=evals,
        evals_result=evals_result,
        verbose_eval=False,
    )
    elapsed = time.time() - start_time
    print(f"[+] Training completed in {elapsed * 1000:.1f}ms")

    # Metrics evaluation
    preds = booster.predict(dtest)
    if "reg" in objective:
        mse = float(np.mean((preds - y_test) ** 2))
        mae = float(np.mean(np.abs(preds - y_test)))
        print(f"    - Test MSE: {mse:.4f}, MAE: {mae:.4f}")
    elif num_class:
        pred_labels = np.argmax(preds, axis=1)
        acc = float(np.mean(pred_labels == y_test))
        print(f"    - Test Accuracy: {acc * 100:.2f}%")
    else:
        pred_labels = (preds > 0.5).astype(np.float32)
        acc = float(np.mean(pred_labels == y_test))
        print(f"    - Test Accuracy: {acc * 100:.2f}%")

    metrics = {
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "n_features": int(X.shape[1]),
        "objective": objective,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "test_predictions_sample": [float(p) for p in preds.flatten()[:5]],
    }
    return booster, metrics


def save_model_artifacts(
    booster: xgb.Booster,
    model_name: str,
    output_dir: str,
    feature_names: List[str],
    X_sample: np.ndarray,
) -> Tuple[str, str]:
    """Exports model to XGBoost JSON format and creates a test payload."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Model JSON file (supported directly by SnoozeML tree-backend)
    model_path = os.path.join(output_dir, f"{model_name}.xgb.json")
    booster.save_model(model_path)
    file_size_kb = os.path.getsize(model_path) / 1024.0
    print(f"[+] Exported XGBoost model: {model_path} ({file_size_kb:.1f} KB)")

    # 2. Sample Inference Payload for REST API
    sample_path = os.path.join(output_dir, f"{model_name}_sample.json")
    # Take first 2 rows for batch demonstration
    batch_sample = X_sample[:2]
    payload = {
        "inputs": [
            {
                "name": "features",
                "shape": list(batch_sample.shape),
                "data": [float(x) for x in batch_sample.flatten()],
            }
        ]
    }
    with open(sample_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[+] Saved sample test payload: {sample_path}")

    return model_path, sample_path


def generate_config_snippet(
    model_name: str,
    relative_model_path: str,
    feature_count: int,
    objective: str,
) -> str:
    """Generates a snippet for models.toml or snooze.toml."""
    return f"""# Generated by SnoozeML tools/train_tree.py
[[models]]
id = "{model_name}"
path = "{relative_model_path}"
description = "Trained XGBoost tree model ({objective}, {feature_count} features)"
device = "tree"
max_batch_size = 64
max_batch_delay_ms = 5
idle_timeout_secs = 300
"""


def verify_with_snooze(
    endpoint: str,
    model_id: str,
    payload_file: str,
    booster: xgb.Booster,
) -> bool:
    """Submits a verification request to a running SnoozeML daemon."""
    url = f"{endpoint.rstrip('/')}/v1/models/{model_id}/infer"
    print(f"\n[*] Verifying inference against SnoozeML: {url}")

    with open(payload_file, "r", encoding="utf-8") as f:
        req_data = json.load(f)

    # 1. Compute local Python XGBoost predictions
    input_item = req_data["inputs"][0]
    shape = input_item["shape"]
    raw_data = input_item["data"]
    arr = np.array(raw_data, dtype=np.float32).reshape(shape)
    dmat = xgb.DMatrix(arr)
    local_preds = booster.predict(dmat).flatten()

    # 2. Submit HTTP POST to SnoozeML
    req_bytes = json.dumps(req_data).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=req_bytes,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        start_t = time.time()
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            resp_bytes = resp.read()
            server_duration = time.time() - start_t
            resp_json = json.loads(resp_bytes.decode("utf-8"))
    except urllib.error.URLError as e:
        print(f"[-] Could not reach SnoozeML server at {url}: {e}")
        print("    Ensure the daemon is running (`snooze serve` or `docker compose up`).")
        return False

    server_outputs = resp_json.get("outputs", [])
    if not server_outputs:
        print(f"[-] Invalid server response (no outputs): {resp_json}")
        return False

    server_data = server_outputs[0].get("data", [])
    server_preds = np.array(server_data, dtype=np.float32).flatten()

    print(f"[+] SnoozeML response received in {server_duration * 1000:.2f}ms:")
    print(f"    - Returned Output Node: {server_outputs[0].get('name')}")
    print(f"    - Returned Shape: {server_outputs[0].get('shape')}")
    print(f"    - Server Latency: {resp_json.get('duration_ms', 0):.2f}ms")

    # 3. Compare parity
    print("\n    Side-by-Side Verification:")
    print("    Index | Python XGBoost | SnoozeML Rust | Diff")
    print("    ------|----------------|---------------|-------------")
    max_diff = 0.0
    for idx, (p_local, p_server) in enumerate(zip(local_preds, server_preds)):
        diff = abs(p_local - p_server)
        max_diff = max(max_diff, diff)
        print(f"    {idx:5d} | {p_local:14.6f} | {p_server:13.6f} | {diff:11.2e}")

    if max_diff < 1e-4:
        print(f"\n[SUCCESS] Exact parity confirmed! Max absolute difference: {max_diff:.2e}")
        return True
    else:
        print(f"\n[WARNING] Discrepancy detected! Max difference: {max_diff:.4f}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="SnoozeML Tree Model Training & Export Tool (XGBoost JSON)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # 1. Train on Iris dataset (3 classes, 4 features) and export to models/:
  python tools/train_tree.py --dataset iris --name iris_model

  # 2. Train on California Housing dataset (regression):
  python tools/train_tree.py --dataset california --name housing_model

  # 3. Train on Breast Cancer dataset (binary classification):
  python tools/train_tree.py --dataset breast_cancer --name cancer_model

  # 4. Train from a custom CSV file:
  python tools/train_tree.py --csv my_data.csv --target price --name custom_model

  # 5. Train and immediately verify against a running SnoozeML server:
  python tools/train_tree.py --dataset iris --name iris_model --verify
""",
    )

    parser.add_argument(
        "--dataset",
        "-d",
        default="iris",
        help="Built-in dataset to train on: 'iris', 'breast_cancer', 'california', 'diabetes', 'synthetic' (default: iris)",
    )
    parser.add_argument(
        "--csv",
        help="Path to custom CSV dataset file",
    )
    parser.add_argument(
        "--target",
        help="Name of target column in CSV file (required if --csv is provided)",
    )
    parser.add_argument(
        "--name",
        "-n",
        default=None,
        help="Model identifier name (default: based on dataset name)",
    )
    parser.add_argument(
        "--models-dir",
        default="./models",
        help="Output models directory (default: ./models)",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=50,
        help="Number of boosting rounds / trees (default: 50)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=4,
        help="Maximum tree depth (default: 4)",
    )
    parser.add_argument(
        "--learning-rate",
        "--eta",
        type=float,
        default=0.1,
        help="Boosting learning rate / shrinkage (default: 0.1)",
    )
    parser.add_argument(
        "--objective",
        default=None,
        help="XGBoost objective (e.g. 'binary:logistic', 'multi:softprob', 'reg:squarederror'). Defaults to auto-detection.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Submit a verification test inference request to running SnoozeML daemon",
    )
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:8080",
        help="SnoozeML server base URL for verification (default: http://127.0.0.1:8080)",
    )

    args = parser.parse_args()

    model_name = args.name or (os.path.splitext(os.path.basename(args.csv))[0] if args.csv else f"{args.dataset}_tree")

    # 1. Load data
    print(f"[*] Loading dataset '{args.dataset}'...")
    X, y, feature_names, default_obj, num_class = load_dataset(
        args.dataset,
        csv_path=args.csv,
        target_col=args.target,
    )
    objective = args.objective or default_obj

    # 2. Train model
    booster, metrics = train_model(
        X,
        y,
        objective=objective,
        num_class=num_class,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
    )

    # 3. Export artifacts
    model_path, sample_path = save_model_artifacts(
        booster=booster,
        model_name=model_name,
        output_dir=args.models_dir,
        feature_names=feature_names,
        X_sample=X,
    )

    # 4. Print configuration instructions
    rel_model_path = os.path.relpath(model_path, args.models_dir)
    config_snippet = generate_config_snippet(
        model_name=model_name,
        relative_model_path=rel_model_path,
        feature_count=X.shape[1],
        objective=objective,
    )

    print("\n" + "=" * 60)
    print("SnoozeML Configuration Snippet (add to models.toml or snooze.toml):")
    print("=" * 60)
    print(config_snippet.strip())
    print("=" * 60)

    # 5. Optional verification
    if args.verify:
        verify_with_snooze(
            endpoint=args.endpoint,
            model_id=model_name,
            payload_file=sample_path,
            booster=booster,
        )


if __name__ == "__main__":
    main()
