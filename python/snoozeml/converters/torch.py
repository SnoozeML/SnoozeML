"""PyTorch ``nn.Module`` to ONNX converter for SnoozeML."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from ..exceptions import ConversionError
from ..logging import get_logger
from ._onnx_builder import OnnxGraphBuilder, encode_attribute

logger = get_logger("converters.torch")


def _extract_tensor_numpy(tensor_or_arr: Any) -> np.ndarray:
    """Converts a PyTorch ``Tensor`` or NumPy-compatible array to a ``float32`` ``np.ndarray``."""
    if hasattr(tensor_or_arr, "detach"):
        return np.ascontiguousarray(tensor_or_arr.detach().cpu().numpy(), dtype=np.float32)
    return np.ascontiguousarray(tensor_or_arr, dtype=np.float32)


def _compile_torch_module_native(
    model: Any,
    sample_arr: np.ndarray,
    input_name: str,
    output_name: str,
    opset_version: int,
) -> bytes:
    """Compiles a PyTorch ``nn.Module`` (Sequential, Linear, MLP, Activations, Normalization) into ONNX."""
    builder = OnnxGraphBuilder(name=f"torch_{type(model).__name__}")
    in_dims: List[Union[int, str, None]] = ["batch_size"] + [
        int(d) for d in sample_arr.shape[1:]
    ]
    builder.add_input(input_name, in_dims)

    # Collect leaf modules in forward execution order
    leaf_modules: List[Any] = []

    def _collect_leaves(mod: Any) -> None:
        children = list(mod.children()) if hasattr(mod, "children") else []
        if not children:
            leaf_modules.append(mod)
        else:
            for child in children:
                _collect_leaves(child)

    _collect_leaves(model)

    current = input_name
    curr_shape = list(sample_arr.shape)

    for idx, mod in enumerate(leaf_modules):
        cls_name = type(mod).__name__

        if cls_name in ("Identity", "Dropout", "Dropout1d", "Dropout2d", "AlphaDropout"):
            continue

        if cls_name == "Flatten":
            flat_out = builder.unique_name(f"flatten_{idx}")
            start_dim = int(getattr(mod, "start_dim", 1))
            builder.add_node(
                "Flatten",
                [current],
                [flat_out],
                attributes=[encode_attribute("axis", i=start_dim)],
            )
            flat_features = int(np.prod(curr_shape[start_dim:]))
            curr_shape = curr_shape[:start_dim] + [flat_features]
            current = flat_out
            continue

        if cls_name == "Linear":
            weight = _extract_tensor_numpy(mod.weight)  # (out_features, in_features)
            w_t = np.ascontiguousarray(weight.T, dtype=np.float32)  # (in_features, out_features)
            out_features = int(w_t.shape[1])
            w_init = builder.add_initializer(builder.unique_name(f"linear_w_{idx}"), w_t)
            mm_out = builder.unique_name(f"linear_mm_{idx}")
            builder.add_node("MatMul", [current, w_init], [mm_out])
            current = mm_out

            if getattr(mod, "bias", None) is not None:
                bias = _extract_tensor_numpy(mod.bias).reshape(1, out_features)
                b_init = builder.add_initializer(builder.unique_name(f"linear_b_{idx}"), bias)
                add_out = builder.unique_name(f"linear_add_{idx}")
                builder.add_node("Add", [current, b_init], [add_out])
                current = add_out

            curr_shape = curr_shape[:-1] + [out_features]
            continue

        if cls_name in ("BatchNorm1d", "LayerNorm"):
            n_feat = int(curr_shape[-1])
            eps = float(getattr(mod, "eps", 1e-5))
            if cls_name == "BatchNorm1d" and getattr(mod, "running_mean", None) is not None:
                mean = _extract_tensor_numpy(mod.running_mean).reshape(1, n_feat)
                var = _extract_tensor_numpy(mod.running_var).reshape(1, n_feat)
                scale = (
                    _extract_tensor_numpy(mod.weight).reshape(1, n_feat)
                    if getattr(mod, "weight", None) is not None
                    else np.ones((1, n_feat), dtype=np.float32)
                )
                bias = (
                    _extract_tensor_numpy(mod.bias).reshape(1, n_feat)
                    if getattr(mod, "bias", None) is not None
                    else np.zeros((1, n_feat), dtype=np.float32)
                )
                inv_std = scale / np.sqrt(var + eps)
                effective_bias = bias - mean * inv_std
                s_init = builder.add_initializer(builder.unique_name(f"bn_scale_{idx}"), inv_std.astype(np.float32))
                b_init = builder.add_initializer(builder.unique_name(f"bn_bias_{idx}"), effective_bias.astype(np.float32))
                mul_out = builder.unique_name(f"bn_mul_{idx}")
                add_out = builder.unique_name(f"bn_add_{idx}")
                builder.add_node("Mul", [current, s_init], [mul_out])
                builder.add_node("Add", [mul_out, b_init], [add_out])
                current = add_out
                continue

        if cls_name == "ReLU":
            out_t = builder.unique_name(f"relu_{idx}")
            builder.add_node("Relu", [current], [out_t])
            current = out_t
            continue

        if cls_name == "Sigmoid":
            out_t = builder.unique_name(f"sigmoid_{idx}")
            builder.add_node("Sigmoid", [current], [out_t])
            current = out_t
            continue

        if cls_name == "Tanh":
            out_t = builder.unique_name(f"tanh_{idx}")
            builder.add_node("Tanh", [current], [out_t])
            current = out_t
            continue

        if cls_name == "Softmax":
            out_t = builder.unique_name(f"softmax_{idx}")
            dim = int(getattr(mod, "dim", -1) or -1)
            builder.add_node(
                "Softmax",
                [current],
                [out_t],
                attributes=[encode_attribute("axis", i=dim)],
            )
            current = out_t
            continue

        if cls_name == "LeakyReLU":
            out_t = builder.unique_name(f"leakyrelu_{idx}")
            slope = float(getattr(mod, "negative_slope", 0.01))
            builder.add_node(
                "LeakyRelu",
                [current],
                [out_t],
                attributes=[encode_attribute("alpha", f=slope)],
            )
            current = out_t
            continue

        raise ConversionError(
            f"Layer '{cls_name}' requires 'onnx' and 'onnxscript' packages for torch.onnx.export. "
            f"Install via: pip install onnx onnxscript"
        )

    builder.add_node("Identity", [current], [output_name], name="final_output")
    out_dims: List[Union[int, str, None]] = ["batch_size"] + [
        int(d) for d in curr_shape[1:]
    ]
    builder.add_output(output_name, out_dims)
    return builder.build_model(
        opset_version=opset_version,
        doc_string=f"Exported from PyTorch {type(model).__name__} by SnoozeML",
    )


def export_torch(
    model: Any,
    sample_input: Any,
    output_path: Union[str, Path] = "model.onnx",
    *,
    input_names: Optional[Sequence[str]] = None,
    output_names: Optional[Sequence[str]] = None,
    dynamic_axes: Optional[Mapping[str, Mapping[int, str]]] = None,
    opset_version: int = 17,
    prefer_torch_onnx: bool = True,
    **kwargs: Any,
) -> Path:
    """Exports a PyTorch ``nn.Module`` to an ONNX file ready for serving with SnoozeML.

    Automatically configures ``dynamic_axes`` along ``Axis(0)`` (``batch_size``) when not
    explicitly provided so that SnoozeML's dynamic micro-batching engine can coalesce
    concurrent requests of varying batch sizes.

    Args:
        model: PyTorch ``torch.nn.Module`` instance to export.
        sample_input: Representative input ``torch.Tensor`` (or ``np.ndarray`` / tuple of tensors)
            used to trace or infer input dimensions.
        output_path: Destination filesystem path for the ``.onnx`` model file.
        input_names: Optional list of input binding names (defaults to ``["input"]``).
        output_names: Optional list of output binding names (defaults to ``["output"]``).
        dynamic_axes: Optional dynamic axes mapping. If ``None``, defaults to dynamic
            ``batch_size`` on dimension ``0`` for all inputs and outputs.
        opset_version: Target ONNX opset version (default ``17``).
        prefer_torch_onnx: If ``True``, attempts ``torch.onnx.export`` first and falls back
            to SnoozeML's native ONNX graph builder when ``onnx``/``onnxscript`` are not installed.
        **kwargs: Additional keyword arguments forwarded to ``torch.onnx.export``.

    Returns:
        Resolved :class:`pathlib.Path` to the written ``.onnx`` file.

    Raises:
        ConversionError: If ``model`` or ``sample_input`` is invalid or export fails.
    """
    if model is None:
        raise ConversionError("Cannot export None PyTorch model")
    if sample_input is None:
        raise ConversionError("sample_input is required to export a PyTorch model")

    resolved_inputs = list(input_names) if input_names else ["input"]
    resolved_outputs = list(output_names) if output_names else ["output"]

    if dynamic_axes is None:
        dyn: Dict[str, Dict[int, str]] = {}
        for name in resolved_inputs + resolved_outputs:
            dyn[name] = {0: "batch_size"}
    else:
        dyn = {k: dict(v) for k, v in dynamic_axes.items()}

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Exporting PyTorch model to ONNX",
        extra={
            "module": type(model).__name__,
            "input_names": resolved_inputs,
            "output_names": resolved_outputs,
            "opset_version": opset_version,
            "output_path": str(out_file),
        },
    )

    was_training = bool(getattr(model, "training", False))
    if hasattr(model, "eval"):
        model.eval()

    try:
        exported = False
        if prefer_torch_onnx:
            exported = _try_torch_onnx_export(
                model=model,
                sample_input=sample_input,
                out_file=out_file,
                input_names=resolved_inputs,
                output_names=resolved_outputs,
                dynamic_axes=dyn,
                opset_version=opset_version,
                **kwargs,
            )

        if not exported:
            if isinstance(sample_input, (tuple, list)):
                first_sample = sample_input[0]
            else:
                first_sample = sample_input
            sample_arr = _extract_tensor_numpy(first_sample)
            if sample_arr.ndim < 2:
                raise ConversionError(
                    f"Expected at least 2D sample_input [batch_size, ...], got shape {list(sample_arr.shape)}"
                )
            model_bytes = _compile_torch_module_native(
                model=model,
                sample_arr=sample_arr,
                input_name=resolved_inputs[0],
                output_name=resolved_outputs[0],
                opset_version=opset_version,
            )
            out_file.write_bytes(model_bytes)
    finally:
        if was_training and hasattr(model, "train"):
            model.train(True)

    size_bytes = out_file.stat().st_size
    logger.info(
        "Successfully exported PyTorch model to ONNX",
        extra={"output_path": str(out_file), "size_bytes": size_bytes},
    )
    return out_file


def _try_torch_onnx_export(
    model: Any,
    sample_input: Any,
    out_file: Path,
    input_names: List[str],
    output_names: List[str],
    dynamic_axes: Mapping[str, Mapping[int, str]],
    opset_version: int,
    **kwargs: Any,
) -> bool:
    """Attempts ``torch.onnx.export`` if PyTorch and its ONNX exporter dependencies are available."""
    try:
        import torch
    except ImportError:
        return False

    if not isinstance(model, torch.nn.Module):
        return False

    if isinstance(sample_input, np.ndarray):
        torch_input: Any = torch.from_numpy(np.ascontiguousarray(sample_input, dtype=np.float32))
    else:
        torch_input = sample_input

    export_kwargs = dict(kwargs)
    # PyTorch 2.6+ defaults dynamo=True which requires onnxscript; try legacy exporter first if onnx is present
    try:
        import inspect

        sig = inspect.signature(torch.onnx.export)
        if "dynamo" in sig.parameters and "dynamo" not in export_kwargs:
            export_kwargs["dynamo"] = False
    except Exception:
        pass

    try:
        with torch.no_grad():
            torch.onnx.export(
                model,
                torch_input,
                str(out_file),
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=opset_version,
                **export_kwargs,
            )
        return out_file.is_file() and out_file.stat().st_size > 0
    except Exception as exc:
        logger.debug(
            "torch.onnx.export unavailable or failed; falling back to native ONNX builder",
            extra={"reason": str(exc)},
        )
        return False
