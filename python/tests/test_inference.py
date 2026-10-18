"""Unit tests for snoozeml.inference tensor validation, serialization, and response handling."""

from __future__ import annotations

import numpy as np
import pytest

from snoozeml.exceptions import InvalidInputError
from snoozeml.inference import (
    InferenceResponse,
    TensorInput,
    TensorOutput,
    build_grpc_infer_request,
    normalize_inputs,
)


def test_tensor_input_validation_and_conversions() -> None:
    arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    tensor = TensorInput.from_array("features", arr)
    assert tensor.name == "features"
    assert tensor.shape == [2, 3]
    assert tensor.data.dtype == np.float32

    rest_dict = tensor.to_rest_dict()
    assert rest_dict["name"] == "features"
    assert rest_dict["shape"] == [2, 3]
    assert rest_dict["data"] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    proto_msg = tensor.to_proto()
    assert proto_msg.name == "features"
    assert list(proto_msg.shape) == [2, 3]
    assert len(proto_msg.raw_data) == 2 * 3 * 4

    # Round-trip through TensorOutput.from_proto
    out = TensorOutput.from_proto(proto_msg)
    np.testing.assert_allclose(out.to_numpy(), arr.astype(np.float32))


def test_tensor_input_rejects_invalid_shapes_and_nan() -> None:
    with pytest.raises(InvalidInputError, match="Tensor name cannot be empty"):
        TensorInput(name="", shape=[1, 2], data=np.zeros((1, 2), dtype=np.float32))

    with pytest.raises(InvalidInputError, match="shape cannot be empty"):
        TensorInput(name="x", shape=[], data=np.array(1.0, dtype=np.float32))

    with pytest.raises(InvalidInputError, match="dimension must be positive"):
        TensorInput(name="x", shape=[0, 4], data=np.zeros((0, 4), dtype=np.float32))

    with pytest.raises(InvalidInputError, match="dimension must be positive"):
        TensorInput(name="x", shape=[-1, 4], data=np.zeros((1, 4), dtype=np.float32))

    with pytest.raises(InvalidInputError, match="expects 6 elements, but provided buffer has 4"):
        TensorInput(name="x", shape=[2, 3], data=np.zeros(4, dtype=np.float32))

    with pytest.raises(InvalidInputError, match="cannot be a 0-dimensional scalar"):
        TensorInput.from_array("scalar", np.array(42.0))

    nan_tensor = TensorInput.from_array("nan_in", np.array([[1.0, float("nan")]]))
    with pytest.raises(InvalidInputError, match="non-finite float values"):
        nan_tensor.to_rest_dict()


def test_normalize_inputs_formats() -> None:
    arr = np.ones((1, 4), dtype=np.float32)

    # 1. Dict mapping name -> ndarray
    res1 = normalize_inputs({"input_ids": arr})
    assert len(res1) == 1
    assert res1[0].name == "input_ids"
    assert res1[0].shape == [1, 4]

    # 2. List of dicts
    res2 = normalize_inputs([{"name": "input", "shape": [1, 4], "data": [1, 2, 3, 4]}])
    assert len(res2) == 1
    assert res2[0].shape == [1, 4]

    # 3. Single TensorInput
    res3 = normalize_inputs(res2[0])
    assert len(res3) == 1

    # 4. Bare ndarray
    res4 = normalize_inputs(arr, default_input_name="features")
    assert len(res4) == 1
    assert res4[0].name == "features"

    with pytest.raises(InvalidInputError):
        normalize_inputs({})

    with pytest.raises(InvalidInputError):
        normalize_inputs([])


def test_inference_response_accessors() -> None:
    resp = InferenceResponse.from_rest_dict(
        {
            "model_id": "demo-model",
            "duration_ms": 1.25,
            "outputs": [
                {"name": "logits", "shape": [1, 3], "data": [0.1, 0.2, 0.7]},
                {"name": "hidden", "shape": [1, 2], "data": [1.5, -1.5]},
            ],
        }
    )
    assert resp.model_id == "demo-model"
    assert resp.duration_ms == 1.25

    # Default first output when name is None
    first = resp.get_numpy()
    assert first.shape == (1, 3)

    # Named lookup & __getitem__
    logits = resp.get_numpy("logits")
    np.testing.assert_allclose(logits, np.array([[0.1, 0.2, 0.7]], dtype=np.float32))
    np.testing.assert_allclose(resp["hidden"], np.array([[1.5, -1.5]], dtype=np.float32))
    np.testing.assert_allclose(resp[0], logits)

    as_dict = resp.to_dict()
    assert set(as_dict.keys()) == {"logits", "hidden"}

    # Enhanced outputs indexing by name and index
    assert "logits" in resp.outputs
    assert "missing" not in resp.outputs
    assert resp.outputs[0].name == "logits"
    assert resp.outputs["logits"].name == "logits"
    np.testing.assert_allclose(resp.outputs["logits"].as_numpy(), logits)
    np.testing.assert_allclose(resp.outputs["hidden"].to_numpy(), resp["hidden"])
    assert resp.outputs.get("logits") is not None
    assert resp.outputs.get("missing") is None

    with pytest.raises(KeyError, match="not found"):
        _ = resp.outputs["missing"]

    with pytest.raises(KeyError, match="not found in response"):
        resp.get_numpy("nonexistent")


def test_build_grpc_infer_request() -> None:
    tensors = normalize_inputs({"input": np.array([[1.0, 2.0]], dtype=np.float32)})
    req = build_grpc_infer_request(
        "demo-model",
        tensors,
        client_timeout_ms=500,
        parameters={"route": "fast"},
    )
    assert req.model_id == "demo-model"
    assert req.client_timeout_ms == 500
    assert req.parameters["route"] == "fast"
    assert len(req.inputs) == 1
