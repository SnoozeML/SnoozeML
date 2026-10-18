"""XGBoost Booster and scikit-learn wrapper exporter for SnoozeML's native pure-Rust tree backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

from ..exceptions import ConversionError
from ..logging import get_logger

logger = get_logger("converters.xgboost")


def export_xgboost(
    model: Any,
    output_path: Union[str, Path] = "model.xgb.json",
) -> Path:
    """Exports an XGBoost ``Booster`` or scikit-learn wrapper (``XGBClassifier``, ``XGBRegressor``)
    to SnoozeML's native ``.xgb.json`` format.

    SnoozeML serves ``.xgb.json`` files directly through its pure-Rust ``hessboost`` tree backend
    without requiring ONNX Runtime or C++ dependencies.

    Args:
        model: Trained ``xgboost.Booster``, ``xgboost.XGBClassifier``, ``xgboost.XGBRegressor``,
            or ``xgboost.XGBRanker``.
        output_path: Destination filesystem path (recommended extension: ``.xgb.json``).

    Returns:
        Resolved :class:`pathlib.Path` to the exported JSON model file.

    Raises:
        ConversionError: If ``model`` is invalid or has not been fitted.
    """
    if model is None:
        raise ConversionError("Cannot export None XGBoost model")

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    booster: Any = None
    if hasattr(model, "get_booster") and callable(model.get_booster):
        try:
            booster = model.get_booster()
        except Exception as exc:
            raise ConversionError(f"XGBoost model is not fitted: {exc}") from exc
    elif hasattr(model, "save_model") and callable(model.save_model):
        booster = model
    else:
        raise ConversionError(
            f"Unsupported XGBoost model type '{type(model).__name__}'; "
            "expected xgboost.Booster, XGBClassifier, or XGBRegressor"
        )

    logger.info(
        "Exporting XGBoost model to JSON",
        extra={"model_type": type(model).__name__, "output_path": str(out_file)},
    )

    try:
        booster.save_model(str(out_file))
    except Exception as exc:
        raise ConversionError(f"Failed to save XGBoost model to {out_file}: {exc}") from exc

    size_bytes = out_file.stat().st_size
    logger.info(
        "Successfully exported XGBoost model",
        extra={"output_path": str(out_file), "size_bytes": size_bytes},
    )
    return out_file
