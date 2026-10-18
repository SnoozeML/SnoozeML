"""Chronos-Bolt Time Series Foundation Model Worker for SnoozeML.

This module provides a duck-typed SnoozeML custom model for Amazon Science's
Chronos-Bolt zero-shot probabilistic time series forecasting models
(e.g., ``amazon/chronos-bolt-mini`` or ``amazon/chronos-bolt-small``).

It loads the Chronos-Bolt pipeline using the ``chronos-forecasting`` library,
handles device placement, and computes multi-step quantile forecasts across
dynamically batched time series inputs along Axis(0).
"""

from __future__ import annotations

import os
from typing import Any, Dict


class Model:
    """SnoozeML worker model class for Chronos-Bolt forecasting."""

    def __init__(self) -> None:
        self.pipeline: Any = None
        self.device: str = "cpu"
        self.torch_dtype: Any = None

    def load(self, model_dir: str, device: str) -> None:
        """Loads the Chronos-Bolt pipeline from local directory or Hugging Face Hub.

        Args:
            model_dir: Path to the model directory (may contain pre-downloaded
                weights or config files).
            device: Target device string passed from SnoozeML (e.g. 'cuda',
                'cuda:0', 'mps', or 'cpu'). SnoozeML automatically resolves
                'auto' before invoking load.
        """
        try:
            import torch
            from chronos import ChronosBoltPipeline
        except ImportError as exc:
            raise RuntimeError(
                "Chronos-Bolt requires 'chronos-forecasting' and 'torch'. "
                "Install them via: pip install chronos-forecasting torch"
            ) from exc

        # Resolve device to valid PyTorch device string
        if device in ("cuda", "mps", "cpu") or device.startswith("cuda:"):
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Select efficient precision on GPU
        if self.device.startswith("cuda") and torch.cuda.is_bf16_supported():
            self.torch_dtype = torch.bfloat16
        else:
            self.torch_dtype = torch.float32

        # Check if model files are stored locally in model_dir, else fetch pretrained weights
        local_config = os.path.join(model_dir, "config.json")
        model_source = model_dir if os.path.isfile(local_config) else "amazon/chronos-bolt-mini"

        self.pipeline = ChronosBoltPipeline.from_pretrained(
            model_source,
            device_map=self.device,
            torch_dtype=self.torch_dtype,
        )

    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Runs zero-shot quantile forecasting on batched historical context.

        Args:
            inputs: Dictionary containing input tensors batched along Axis(0):
                - ``context``: FP32 array/tensor of shape ``(batch_size, context_length)``.
                  Also accepts ``input`` as a fallback key name.

        Returns:
            Dictionary containing:
                - ``forecast``: Multi-step quantile forecasts of shape
                  ``(batch_size, num_quantiles, prediction_length)``.
                  Default 9 deciles: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9].
                - ``median``: 50th-percentile point forecast of shape
                  ``(batch_size, prediction_length)``.
        """
        import torch

        # Accept "context" or standard "input" tensor key
        context = inputs.get("context")
        if context is None:
            context = inputs.get("input")
        if context is None:
            raise ValueError("Input tensor 'context' (or 'input') is required for forecasting")

        # Convert to torch tensor if needed
        if not isinstance(context, torch.Tensor):
            context = torch.as_tensor(context, dtype=torch.float32)

        # Standard prediction horizon (can be customized or parameterized)
        prediction_length = 12

        # ChronosBoltPipeline uses positional 'inputs' (or 'inputs=...').
        # Passing context as the first positional argument avoids keyword name differences.
        try:
            forecast = self.pipeline.predict(context, prediction_length=prediction_length)
        except TypeError:
            try:
                forecast = self.pipeline.predict(inputs=context, prediction_length=prediction_length)
            except TypeError:
                forecast = self.pipeline.predict(context=context, prediction_length=prediction_length)

        # In Chronos-Bolt's default 9 deciles [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        # index 4 corresponds to the 0.5 (median) forecast.
        # Handle layout dynamically: (batch, 9, horizon) vs (batch, horizon, 9)
        if forecast.ndim == 3:
            if forecast.shape[1] == 9:
                median = forecast[:, 4, :]
            elif forecast.shape[2] == 9:
                median = forecast[:, :, 4]
            else:
                median = forecast[:, forecast.shape[1] // 2, :]
        else:
            median = forecast

        # SnoozeML automatically detaches and converts PyTorch tensors to FP32 NumPy arrays
        return {
            "forecast": forecast.to(torch.float32),
            "median": median.to(torch.float32),
        }
