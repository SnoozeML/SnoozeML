"""Unit tests for snoozeml.converters (scikit-learn, PyTorch, and XGBoost exporters)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

import numpy as np
import pytest
from sklearn.datasets import load_iris, make_classification, make_regression
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
xgb = pytest.importorskip("xgboost")

from snoozeml.converters import export_sklearn, export_torch, export_xgboost
from snoozeml.exceptions import ConversionError


def test_export_sklearn_linear_and_logistic(tmp_path: Path) -> None:
    X_reg, y_reg = make_regression(n_samples=40, n_features=4, noise=0.1, random_state=42)
    X_reg = X_reg.astype(np.float32)
    y_reg = y_reg.astype(np.float32)

    lin = LinearRegression().fit(X_reg, y_reg)
    lin_path = export_sklearn(
        lin,
        initial_types=[("float_input", (None, 4))],
        output_path=tmp_path / "lin_reg.onnx",
    )
    assert lin_path.is_file()
    assert lin_path.stat().st_size > 50

    ridge = Ridge(alpha=1.0).fit(X_reg, y_reg)
    ridge_path = export_sklearn(ridge, sample_input=X_reg[:2], output_path=tmp_path / "ridge.onnx")
    assert ridge_path.is_file()

    # Binary LogisticRegression
    X_bin, y_bin = make_classification(
        n_samples=50, n_features=4, n_informative=3, n_redundant=0, n_classes=2, random_state=42
    )
    X_bin = X_bin.astype(np.float32)
    clf_bin = LogisticRegression().fit(X_bin, y_bin)
    bin_path = export_sklearn(clf_bin, output_path=tmp_path / "logreg_bin.onnx")
    assert bin_path.is_file()

    # Multiclass LogisticRegression
    iris = load_iris()
    X_iris = iris.data.astype(np.float32)
    y_iris = iris.target
    clf_multi = LogisticRegression(max_iter=300).fit(X_iris, y_iris)
    multi_path = export_sklearn(clf_multi, output_path=tmp_path / "logreg_multi.onnx")
    assert multi_path.is_file()


def test_export_sklearn_mlp_trees_and_pipelines(tmp_path: Path) -> None:
    iris = load_iris()
    X = iris.data.astype(np.float32)
    y = iris.target

    # 1. MLPClassifier & MLPRegressor
    mlp_clf = MLPClassifier(hidden_layer_sizes=(8,), solver="lbfgs", max_iter=500, random_state=42).fit(X, y)
    mlp_clf_path = export_sklearn(mlp_clf, output_path=tmp_path / "mlp_clf.onnx")
    assert mlp_clf_path.is_file()

    mlp_reg = MLPRegressor(hidden_layer_sizes=(6,), solver="lbfgs", max_iter=500, random_state=42).fit(
        X, X[:, 0]
    )
    mlp_reg_path = export_sklearn(mlp_reg, output_path=tmp_path / "mlp_reg.onnx")
    assert mlp_reg_path.is_file()

    # 2. DecisionTreeClassifier & DecisionTreeRegressor
    dt_clf = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X, y)
    assert export_sklearn(dt_clf, output_path=tmp_path / "dt_clf.onnx").is_file()

    dt_reg = DecisionTreeRegressor(max_depth=3, random_state=42).fit(X, X[:, 0])
    assert export_sklearn(dt_reg, output_path=tmp_path / "dt_reg.onnx").is_file()

    # 3. RandomForestClassifier & RandomForestRegressor
    rf_clf = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42).fit(X, y)
    assert export_sklearn(rf_clf, output_path=tmp_path / "rf_clf.onnx").is_file()

    rf_reg = RandomForestRegressor(n_estimators=5, max_depth=3, random_state=42).fit(X, X[:, 0])
    assert export_sklearn(rf_reg, output_path=tmp_path / "rf_reg.onnx").is_file()

    # 4. GradientBoostingClassifier & GradientBoostingRegressor
    gb_clf = GradientBoostingClassifier(n_estimators=5, max_depth=2, random_state=42).fit(X, y)
    assert export_sklearn(gb_clf, output_path=tmp_path / "gb_clf.onnx").is_file()

    gb_reg = GradientBoostingRegressor(n_estimators=5, max_depth=2, random_state=42).fit(X, X[:, 0])
    assert export_sklearn(gb_reg, output_path=tmp_path / "gb_reg.onnx").is_file()

    # 5. Pipelines with StandardScaler and MinMaxScaler
    pipe1 = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(n_estimators=4, max_depth=3, random_state=42),
    ).fit(X, y)
    assert export_sklearn(pipe1, output_path=tmp_path / "pipe_std_rf.onnx").is_file()

    pipe2 = make_pipeline(
        MinMaxScaler(),
        LogisticRegression(max_iter=200),
    ).fit(X, y)
    assert export_sklearn(pipe2, output_path=tmp_path / "pipe_minmax_lr.onnx").is_file()


def test_export_sklearn_validation_errors(tmp_path: Path) -> None:
    with pytest.raises(ConversionError, match="Cannot export None model"):
        export_sklearn(None, output_path=tmp_path / "fail.onnx")

    unfitted = RandomForestClassifier(n_estimators=5)
    with pytest.raises(ConversionError, match="not fitted"):
        export_sklearn(unfitted, initial_types=[("input", (None, 4))], output_path=tmp_path / "fail.onnx")


class _MockLinear:
    def __init__(self, in_features: int, out_features: int) -> None:
        rng = np.random.default_rng(42)
        self.weight = rng.standard_normal((out_features, in_features)).astype(np.float32)
        self.bias = rng.standard_normal((out_features,)).astype(np.float32)

    def children(self) -> List[Any]:
        return []


class _MockReLU:
    def children(self) -> List[Any]:
        return []


class _MockSequential:
    def __init__(self, layers: List[Any]) -> None:
        self._layers = layers
        self.training = True

    def children(self) -> List[Any]:
        return self._layers

    def eval(self) -> "_MockSequential":
        self.training = False
        return self

    def train(self, mode: bool = True) -> "_MockSequential":
        self.training = mode
        return self


# Rename classes to match PyTorch layer names ("Linear", "ReLU") for native graph builder
_MockLinear.__name__ = "Linear"
_MockReLU.__name__ = "ReLU"


def test_export_torch_module_and_state_preservation(tmp_path: Path) -> None:
    net = _MockSequential([_MockLinear(4, 8), _MockReLU(), _MockLinear(8, 3)])
    assert net.training is True

    sample = np.ones((2, 4), dtype=np.float32)
    out_path = export_torch(
        model=net,
        sample_input=sample,
        output_path=tmp_path / "torch_mlp.onnx",
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    assert out_path.is_file()
    assert out_path.stat().st_size > 100
    # Verify original training state was restored after export
    assert net.training is True


def test_export_torch_validation_errors(tmp_path: Path) -> None:
    with pytest.raises(ConversionError, match="Cannot export None"):
        export_torch(None, np.ones((1, 4)), output_path=tmp_path / "fail.onnx")

    net = _MockSequential([_MockLinear(4, 2)])
    with pytest.raises(ConversionError, match="sample_input is required"):
        export_torch(net, None, output_path=tmp_path / "fail.onnx")

    with pytest.raises(ConversionError, match="at least 2D sample_input"):
        export_torch(net, np.ones((4,), dtype=np.float32), output_path=tmp_path / "fail.onnx")


def test_export_xgboost_booster_and_sklearn_wrappers(tmp_path: Path) -> None:
    iris = load_iris()
    X = iris.data.astype(np.float32)
    y = iris.target.astype(np.int32)

    # 1. XGBClassifier
    clf = xgb.XGBClassifier(n_estimators=5, max_depth=3, random_state=42)
    clf.fit(X, y)
    clf_path = export_xgboost(clf, output_path=tmp_path / "iris_clf.xgb.json")
    assert clf_path.is_file()
    assert clf_path.stat().st_size > 200

    # 2. Raw Booster
    dtrain = xgb.DMatrix(X, label=y)
    booster = xgb.train({"max_depth": 3, "objective": "multi:softprob", "num_class": 3}, dtrain, num_boost_round=4)
    booster_path = export_xgboost(booster, output_path=tmp_path / "iris_booster.xgb.json")
    assert booster_path.is_file()

    # 3. Validation errors
    with pytest.raises(ConversionError, match="Cannot export None"):
        export_xgboost(None, output_path=tmp_path / "fail.xgb.json")

    unfitted_xgb = xgb.XGBClassifier(n_estimators=3)
    with pytest.raises(ConversionError, match="not fitted"):
        export_xgboost(unfitted_xgb, output_path=tmp_path / "fail.xgb.json")

    with pytest.raises(ConversionError, match="Unsupported XGBoost model type"):
        export_xgboost({"not": "a_model"}, output_path=tmp_path / "fail.xgb.json")
