"""Scikit-learn estimator and Pipeline to ONNX converter for SnoozeML."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..exceptions import ConversionError
from ..logging import get_logger
from ._onnx_builder import OnnxGraphBuilder, encode_attribute

logger = get_logger("converters.sklearn")

InitialTypeSpec = Sequence[Tuple[str, Any]]


def _infer_input_spec(
    model: Any,
    initial_types: Optional[InitialTypeSpec],
    sample_input: Optional[Union[np.ndarray, Sequence[Any]]],
    default_input_name: str,
) -> Tuple[str, int]:
    """Resolves the input tensor binding name and feature count ``n_features``."""
    input_name = default_input_name
    n_features: Optional[int] = None

    if initial_types:
        first = initial_types[0]
        if isinstance(first, (tuple, list)) and len(first) >= 2:
            input_name = str(first[0])
            type_desc = first[1]
            # Support tuple shapes like (None, 4) or [None, 4]
            if isinstance(type_desc, (tuple, list)) and len(type_desc) >= 2:
                dim = type_desc[1]
                if isinstance(dim, (int, np.integer)) and int(dim) > 0:
                    n_features = int(dim)
            # Support skl2onnx FloatTensorType([None, 4])
            elif hasattr(type_desc, "shape") and len(type_desc.shape) >= 2:
                dim = type_desc.shape[1]
                if isinstance(dim, (int, np.integer)) and int(dim) > 0:
                    n_features = int(dim)

    if n_features is None and sample_input is not None:
        arr = np.asarray(sample_input)
        if arr.ndim == 1:
            n_features = int(arr.shape[0])
        elif arr.ndim >= 2:
            n_features = int(arr.shape[1])

    if n_features is None and hasattr(model, "n_features_in_"):
        n_features = int(model.n_features_in_)

    if n_features is None or n_features <= 0:
        raise ConversionError(
            "Could not determine input feature count 'n_features'. "
            "Pass initial_types=[('input', (None, n_features))] or sample_input=X_sample."
        )

    return input_name, n_features


def _check_fitted(model: Any) -> None:
    """Verifies that a scikit-learn estimator or Pipeline has been fitted."""
    try:
        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(model)
    except ImportError:
        pass
    except Exception as exc:
        raise ConversionError(f"Scikit-learn model is not fitted: {exc}") from exc


def _compile_tree_structure(
    builder: OnnxGraphBuilder,
    tree_obj: Any,
    input_tensor: str,
    is_classifier: bool,
) -> Tuple[str, int]:
    """Compiles a fitted ``sklearn.tree._tree.Tree`` into standard ONNX tensor operators."""
    children_left = tree_obj.children_left
    children_right = tree_obj.children_right
    feature = tree_obj.feature
    threshold = tree_obj.threshold
    values = tree_obj.value  # shape: (n_nodes, n_outputs, max_n_classes)

    if is_classifier:
        raw_root = np.asarray(values[0][0], dtype=np.float32)
        n_out = int(raw_root.size)
    else:
        raw_root = np.asarray(values[0][:, 0], dtype=np.float32)
        n_out = int(raw_root.size)

    def build_node(node_id: int) -> str:
        left_id = int(children_left[node_id])
        right_id = int(children_right[node_id])
        if left_id == right_id:  # Leaf node
            if is_classifier:
                counts = np.asarray(values[node_id][0], dtype=np.float32)
                total = float(np.sum(counts))
                probs = counts / total if total > 0 else counts
                leaf_arr = probs.reshape(1, n_out).astype(np.float32)
            else:
                reg_val = np.asarray(values[node_id][:, 0], dtype=np.float32)
                leaf_arr = reg_val.reshape(1, n_out).astype(np.float32)
            return builder.add_initializer(
                builder.unique_name("tree_leaf"), leaf_arr
            )

        feat_idx = int(feature[node_id])
        thresh_val = float(threshold[node_id])

        idx_init = builder.add_initializer(
            builder.unique_name("feat_idx"),
            np.array([feat_idx], dtype=np.int64),
        )
        gathered = builder.unique_name("feat_col")
        builder.add_node(
            "Gather",
            inputs=[input_tensor, idx_init],
            outputs=[gathered],
            attributes=[encode_attribute("axis", i=1)],
        )

        thresh_init = builder.add_initializer(
            builder.unique_name("thresh"),
            np.array([[thresh_val]], dtype=np.float32),
        )
        cond_out = builder.unique_name("cond")
        builder.add_node(
            "LessOrEqual",
            inputs=[gathered, thresh_init],
            outputs=[cond_out],
        )

        left_tensor = build_node(left_id)
        right_tensor = build_node(right_id)

        where_out = builder.unique_name("tree_branch")
        builder.add_node(
            "Where",
            inputs=[cond_out, left_tensor, right_tensor],
            outputs=[where_out],
        )
        return where_out

    return build_node(0), n_out


def _compile_activation(
    builder: OnnxGraphBuilder,
    input_tensor: str,
    activation: str,
) -> str:
    """Appends an ONNX activation node matching scikit-learn MLP activation names."""
    act = activation.lower().strip()
    if act in ("identity", "linear", "none"):
        return input_tensor
    out_tensor = builder.unique_name(f"act_{act}")
    if act == "relu":
        builder.add_node("Relu", [input_tensor], [out_tensor])
    elif act == "tanh":
        builder.add_node("Tanh", [input_tensor], [out_tensor])
    elif act == "logistic" or act == "sigmoid":
        builder.add_node("Sigmoid", [input_tensor], [out_tensor])
    elif act == "softmax":
        builder.add_node(
            "Softmax",
            [input_tensor],
            [out_tensor],
            attributes=[encode_attribute("axis", i=-1)],
        )
    else:
        raise ConversionError(f"Unsupported activation function: '{activation}'")
    return out_tensor


def _compile_step(
    builder: OnnxGraphBuilder,
    step: Any,
    current_tensor: str,
) -> Tuple[str, int]:
    """Compiles a single scikit-learn transformer or estimator step into ONNX graph nodes."""
    cls_name = type(step).__name__

    # 1. Preprocessing scalers
    if cls_name == "StandardScaler":
        n_features = int(step.n_features_in_)
        if getattr(step, "with_mean", True) and getattr(step, "mean_", None) is not None:
            mean_arr = np.asarray(step.mean_, dtype=np.float32).reshape(1, n_features)
            mean_init = builder.add_initializer(builder.unique_name("scaler_mean"), mean_arr)
            sub_out = builder.unique_name("scaler_centered")
            builder.add_node("Sub", [current_tensor, mean_init], [sub_out])
            current_tensor = sub_out
        if getattr(step, "with_std", True) and getattr(step, "scale_", None) is not None:
            scale_arr = np.asarray(step.scale_, dtype=np.float32).reshape(1, n_features)
            scale_arr = np.where(scale_arr == 0.0, 1.0, scale_arr)
            scale_init = builder.add_initializer(builder.unique_name("scaler_scale"), scale_arr)
            div_out = builder.unique_name("scaler_scaled")
            builder.add_node("Div", [current_tensor, scale_init], [div_out])
            current_tensor = div_out
        return current_tensor, n_features

    if cls_name == "MinMaxScaler":
        n_features = int(step.n_features_in_)
        scale_arr = np.asarray(step.scale_, dtype=np.float32).reshape(1, n_features)
        min_arr = np.asarray(step.min_, dtype=np.float32).reshape(1, n_features)
        scale_init = builder.add_initializer(builder.unique_name("minmax_scale"), scale_arr)
        min_init = builder.add_initializer(builder.unique_name("minmax_min"), min_arr)
        mul_out = builder.unique_name("minmax_mul")
        add_out = builder.unique_name("minmax_out")
        builder.add_node("Mul", [current_tensor, scale_init], [mul_out])
        builder.add_node("Add", [mul_out, min_init], [add_out])
        return add_out, n_features

    # 2. Multi-Layer Perceptrons (MLPRegressor / MLPClassifier)
    if hasattr(step, "coefs_") and hasattr(step, "intercepts_"):
        coefs = step.coefs_
        intercepts = step.intercepts_
        hidden_act = getattr(step, "activation", "relu")
        out_act = getattr(step, "out_activation_", "identity")
        n_layers = len(coefs)
        n_out = 1
        for idx, (w_mat, b_vec) in enumerate(zip(coefs, intercepts)):
            w_arr = np.asarray(w_mat, dtype=np.float32)
            b_arr = np.asarray(b_vec, dtype=np.float32).reshape(1, -1)
            n_out = int(b_arr.shape[1])
            w_init = builder.add_initializer(builder.unique_name(f"mlp_w{idx}"), w_arr)
            b_init = builder.add_initializer(builder.unique_name(f"mlp_b{idx}"), b_arr)
            mm_out = builder.unique_name(f"mlp_mm{idx}")
            lin_out = builder.unique_name(f"mlp_lin{idx}")
            builder.add_node("MatMul", [current_tensor, w_init], [mm_out])
            builder.add_node("Add", [mm_out, b_init], [lin_out])
            act_to_apply = hidden_act if idx < n_layers - 1 else out_act
            current_tensor = _compile_activation(builder, lin_out, act_to_apply)
        return current_tensor, n_out

    # 3. Linear Models (LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression, SGD*)
    if hasattr(step, "coef_") and hasattr(step, "intercept_"):
        coef = np.asarray(step.coef_, dtype=np.float32)
        intercept = np.asarray(step.intercept_, dtype=np.float32)
        if coef.ndim == 1:
            w_arr = coef.reshape(-1, 1)
        else:
            # sklearn stores coef_ as (n_targets, n_features); transpose to (n_features, n_targets)
            w_arr = coef.T
        n_out = int(w_arr.shape[1])
        b_arr = intercept.reshape(1, n_out)

        w_init = builder.add_initializer(builder.unique_name("lin_w"), w_arr)
        b_init = builder.add_initializer(builder.unique_name("lin_b"), b_arr)
        mm_out = builder.unique_name("lin_mm")
        lin_out = builder.unique_name("lin_add")
        builder.add_node("MatMul", [current_tensor, w_init], [mm_out])
        builder.add_node("Add", [mm_out, b_init], [lin_out])
        current_tensor = lin_out

        is_classifier = hasattr(step, "classes_")
        if is_classifier and cls_name in ("LogisticRegression", "SGDClassifier"):
            if n_out == 1:
                sig_out = builder.unique_name("prob_pos")
                builder.add_node("Sigmoid", [current_tensor], [sig_out])
                one_init = builder.add_initializer(
                    builder.unique_name("one"), np.array([[1.0]], dtype=np.float32)
                )
                prob_neg = builder.unique_name("prob_neg")
                builder.add_node("Sub", [one_init, sig_out], [prob_neg])
                concat_out = builder.unique_name("probs")
                builder.add_node(
                    "Concat",
                    [prob_neg, sig_out],
                    [concat_out],
                    attributes=[encode_attribute("axis", i=1)],
                )
                return concat_out, 2
            current_tensor = _compile_activation(builder, current_tensor, "softmax")
        return current_tensor, n_out

    # 4. Single Decision Tree (DecisionTreeClassifier / DecisionTreeRegressor)
    if hasattr(step, "tree_"):
        is_classifier = hasattr(step, "classes_")
        return _compile_tree_structure(builder, step.tree_, current_tensor, is_classifier)

    # 5. Forest Ensembles (RandomForest*, ExtraTrees*)
    if hasattr(step, "estimators_") and isinstance(step.estimators_, list) and step.estimators_:
        first_est = step.estimators_[0]
        if hasattr(first_est, "tree_"):
            is_classifier = hasattr(step, "classes_")
            acc_tensor: Optional[str] = None
            n_out = 1
            for tree_est in step.estimators_:
                t_out, n_out = _compile_tree_structure(
                    builder, tree_est.tree_, current_tensor, is_classifier
                )
                if acc_tensor is None:
                    acc_tensor = t_out
                else:
                    sum_name = builder.unique_name("rf_sum")
                    builder.add_node("Add", [acc_tensor, t_out], [sum_name])
                    acc_tensor = sum_name
            assert acc_tensor is not None
            n_trees = float(len(step.estimators_))
            div_init = builder.add_initializer(
                builder.unique_name("rf_count"),
                np.array([[n_trees]], dtype=np.float32),
            )
            avg_out = builder.unique_name("rf_avg")
            builder.add_node("Div", [acc_tensor, div_init], [avg_out])
            return avg_out, n_out

        # GradientBoostingRegressor / GradientBoostingClassifier (estimators_ is 2D ndarray)
    if (
        hasattr(step, "estimators_")
        and isinstance(step.estimators_, np.ndarray)
        and step.estimators_.size > 0
    ):
        lr = float(getattr(step, "learning_rate", 0.1))
        n_cols = int(step.estimators_.shape[1])
        # Compute initial baseline prediction if available
        init_vals = np.zeros((1, n_cols), dtype=np.float32)
        if hasattr(step, "init_") and hasattr(step, "n_features_in_"):
            try:
                dummy = np.zeros((1, int(step.n_features_in_)), dtype=np.float32)
                raw_init = step._raw_predict_init(dummy)
                init_vals = np.asarray(raw_init, dtype=np.float32).reshape(1, n_cols)
            except Exception:
                pass
        acc_tensor = builder.add_initializer(builder.unique_name("gb_init"), init_vals)
        lr_init = builder.add_initializer(
            builder.unique_name("gb_lr"), np.array([[lr]], dtype=np.float32)
        )
        for row in step.estimators_:
            col_tensors: List[str] = []
            for tree_est in row:
                t_out, _ = _compile_tree_structure(
                    builder, tree_est.tree_, current_tensor, is_classifier=False
                )
                col_tensors.append(t_out)
            if len(col_tensors) == 1:
                stage_out = col_tensors[0]
            else:
                stage_out = builder.unique_name("gb_stage_concat")
                builder.add_node(
                    "Concat",
                    col_tensors,
                    [stage_out],
                    attributes=[encode_attribute("axis", i=1)],
                )
            scaled_stage = builder.unique_name("gb_scaled")
            builder.add_node("Mul", [stage_out, lr_init], [scaled_stage])
            next_acc = builder.unique_name("gb_acc")
            builder.add_node("Add", [acc_tensor, scaled_stage], [next_acc])
            acc_tensor = next_acc

        if hasattr(step, "classes_"):
            if n_cols == 1:
                sig_out = builder.unique_name("gb_prob_pos")
                builder.add_node("Sigmoid", [acc_tensor], [sig_out])
                one_init = builder.add_initializer(
                    builder.unique_name("gb_one"), np.array([[1.0]], dtype=np.float32)
                )
                prob_neg = builder.unique_name("gb_prob_neg")
                builder.add_node("Sub", [one_init, sig_out], [prob_neg])
                concat_out = builder.unique_name("gb_probs")
                builder.add_node(
                    "Concat",
                    [prob_neg, sig_out],
                    [concat_out],
                    attributes=[encode_attribute("axis", i=1)],
                )
                return concat_out, 2
            return _compile_activation(builder, acc_tensor, "softmax"), n_cols
        return acc_tensor, n_cols

    raise ConversionError(
        f"Estimator '{cls_name}' is not natively supported without 'skl2onnx'. "
        f"Install 'snoozeml[converters]' (pip install skl2onnx onnx) for extended estimator support."
    )


def _compile_sklearn_native(
    model: Any,
    input_name: str,
    output_name: str,
    n_features: int,
    opset_version: int,
) -> bytes:
    """Compiles a fitted scikit-learn estimator or Pipeline into an ONNX ``ModelProto`` byte buffer."""
    builder = OnnxGraphBuilder(name=f"sklearn_{type(model).__name__}")
    builder.add_input(input_name, ["batch_size", n_features])

    steps: List[Any] = []
    if hasattr(model, "steps") and isinstance(model.steps, list):
        for _, step_obj in model.steps:
            if step_obj is not None and step_obj != "passthrough":
                steps.append(step_obj)
    else:
        steps.append(model)

    current = input_name
    n_out = 1
    for step in steps:
        current, n_out = _compile_step(builder, step, current)

    # Rename final output tensor to output_name via Identity node
    builder.add_node("Identity", [current], [output_name], name="final_output")
    builder.add_output(output_name, ["batch_size", n_out])
    return builder.build_model(
        opset_version=opset_version,
        doc_string=f"Exported from scikit-learn {type(model).__name__} by SnoozeML",
    )


def export_sklearn(
    model: Any,
    initial_types: Optional[InitialTypeSpec] = None,
    output_path: Union[str, Path] = "model.onnx",
    *,
    sample_input: Optional[Union[np.ndarray, Sequence[Any]]] = None,
    input_name: str = "float_input",
    output_name: str = "output",
    opset_version: int = 17,
    prefer_skl2onnx: bool = False,
    options: Optional[Dict[Any, Any]] = None,
) -> Path:
    """Exports a fitted scikit-learn estimator or ``Pipeline`` to an ONNX file for SnoozeML.

    Produces a standard ONNX model with ``float32`` tensor inputs and outputs that can be
    served directly by SnoozeML's ``ort`` backend with dynamic micro-batching along ``Axis(0)``.

    Args:
        model: Fitted scikit-learn estimator (e.g. ``RandomForestClassifier``,
            ``LogisticRegression``, ``LinearRegression``, ``MLPClassifier``) or ``Pipeline``.
        initial_types: Optional list of ``(name, shape_or_type)`` tuples such as
            ``[("float_input", (None, 4))]``. If omitted, inferred from ``sample_input``
            or ``model.n_features_in_``.
        output_path: Destination filesystem path for the exported ``.onnx`` file.
        sample_input: Optional sample feature array ``(batch, n_features)`` used to infer
            input dimensions when ``initial_types`` is omitted.
        input_name: Default input tensor binding name when ``initial_types`` is omitted.
        output_name: Output tensor binding name (default ``"output"``).
        opset_version: Target ONNX opset version (default ``17``).
        prefer_skl2onnx: If ``True`` and ``skl2onnx`` is installed, delegates conversion
            to ``skl2onnx.convert_sklearn``. Defaults to ``False`` (uses SnoozeML's native
            single-tensor FP32 ONNX compiler with automatic fallback to ``skl2onnx``).
        options: Optional dictionary of converter options forwarded to ``skl2onnx``.

    Returns:
        Resolved :class:`pathlib.Path` to the written ``.onnx`` model file.

    Raises:
        ConversionError: If the model is not fitted, input dimensions cannot be determined,
            or conversion fails.
    """
    if model is None:
        raise ConversionError("Cannot export None model")

    _check_fitted(model)
    resolved_input_name, n_features = _infer_input_spec(
        model=model,
        initial_types=initial_types,
        sample_input=sample_input,
        default_input_name=input_name,
    )

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Exporting scikit-learn model to ONNX",
        extra={
            "estimator": type(model).__name__,
            "input_name": resolved_input_name,
            "n_features": n_features,
            "opset_version": opset_version,
            "output_path": str(out_file),
        },
    )

    model_bytes: Optional[bytes] = None

    if prefer_skl2onnx:
        model_bytes = _try_skl2onnx_convert(
            model, resolved_input_name, n_features, opset_version, options
        )

    if model_bytes is None:
        try:
            model_bytes = _compile_sklearn_native(
                model=model,
                input_name=resolved_input_name,
                output_name=output_name,
                n_features=n_features,
                opset_version=opset_version,
            )
        except ConversionError:
            # Fallback to skl2onnx if installed
            model_bytes = _try_skl2onnx_convert(
                model, resolved_input_name, n_features, opset_version, options
            )
            if model_bytes is None:
                raise

    out_file.write_bytes(model_bytes)
    size_bytes = out_file.stat().st_size
    logger.info(
        "Successfully exported scikit-learn model to ONNX",
        extra={"output_path": str(out_file), "size_bytes": size_bytes},
    )
    return out_file


def _try_skl2onnx_convert(
    model: Any,
    input_name: str,
    n_features: int,
    opset_version: int,
    options: Optional[Dict[Any, Any]],
) -> Optional[bytes]:
    """Attempts conversion via ``skl2onnx`` if the package is installed."""
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError:
        return None

    try:
        skl_initial_types = [(input_name, FloatTensorType([None, n_features]))]
        skl_options = options
        if skl_options is None and hasattr(model, "classes_"):
            skl_options = {id(model): {"zipmap": False}}
        onnx_model = convert_sklearn(
            model,
            initial_types=skl_initial_types,
            target_opset=opset_version,
            options=skl_options,
        )
        return bytes(onnx_model.SerializeToString())
    except Exception as exc:
        raise ConversionError(f"skl2onnx conversion failed: {exc}") from exc
