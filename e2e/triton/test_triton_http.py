"""End-to-end interoperability tests using NVIDIA's official ``tritonclient.http`` SDK.

All tests execute against real fitted Scikit-Learn ONNX models (``sklearn_iris``,
``sklearn_rf``, ``secure_iris``) and XGBoost tree models (``xgb_iris``) — zero mocks.

Exercises:
  - Server health & metadata: ``is_server_live``, ``is_server_ready``, ``get_server_metadata``
  - Repository index & scale-to-zero lifecycle: ``get_model_repository_index``,
    ``load_model``, ``is_model_ready``, ``unload_model`` on real ONNX models
  - Model introspection: ``get_model_metadata``, ``get_model_config``, ``get_inference_statistics``
  - HTTP REST inference: ``infer`` with both JSON mode (``binary_data=False``) and
    Triton binary HTTP extension (``binary_data=True``) across ``FP32``, ``FP64``, ``INT32``, ``INT64``
    against real Scikit-Learn ONNX and XGBoost ``predict_proba`` ground truth
  - Out-of-scope HTTP endpoints returning ``501 Not Implemented``
  - 3-tier RBAC authentication via HTTP headers
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest

triton_http = pytest.importorskip("tritonclient.http")
from tritonclient.utils import InferenceServerException

from e2e.conftest import TritonDaemonEndpoints


def test_triton_http_server_health_and_metadata(
    triton_daemon: TritonDaemonEndpoints,
) -> None:
    """Verifies /v2/health/live, /v2/health/ready, and /v2 metadata via tritonclient.http."""
    with triton_http.InferenceServerClient(url=triton_daemon.http_url) as client:
        assert client.is_server_live() is True
        assert client.is_server_ready() is True

        meta = client.get_server_metadata()
        assert meta["name"] == "snooze-ml"
        assert "model_repository" in meta["extensions"]
        assert "binary_tensor_data" in meta["extensions"]


def test_triton_http_repository_and_model_lifecycle(
    triton_daemon: TritonDaemonEndpoints,
) -> None:
    """Verifies /v2/repository/index, load, ready, and unload on real ONNX models."""
    with triton_http.InferenceServerClient(url=triton_daemon.http_url) as client:
        client.unload_model("sklearn_rf")
        assert client.is_model_ready("sklearn_rf") is False
        assert client.is_model_ready("sklearn_iris") is True

        repo_all = client.get_model_repository_index()
        states = {m["name"]: m["state"] for m in repo_all}
        assert states.get("sklearn_rf") == "UNAVAILABLE"
        assert states.get("sklearn_iris") == "READY"

        client.load_model("sklearn_rf")
        assert client.is_model_ready("sklearn_rf") is True

        # Run real inference on loaded sklearn_rf
        x = triton_daemon.sample_x
        inp = triton_http.InferInput("input", list(x.shape), "FP32")
        inp.set_data_from_numpy(x, binary_data=True)
        out_req = triton_http.InferRequestedOutput("output", binary_data=True)
        res = client.infer("sklearn_rf", inputs=[inp], outputs=[out_req])
        expected_rf = triton_daemon.sklearn_rf.predict_proba(x).astype(np.float32)
        np.testing.assert_allclose(res.as_numpy("output"), expected_rf, atol=1e-4)

        client.unload_model("sklearn_rf")
        assert client.is_model_ready("sklearn_rf") is False


def test_triton_http_model_metadata_config_and_statistics(
    triton_daemon: TritonDaemonEndpoints,
) -> None:
    """Verifies /v2/models/{name}, /config, and /stats via tritonclient.http."""
    with triton_http.InferenceServerClient(url=triton_daemon.http_url) as client:
        meta = client.get_model_metadata("sklearn_iris")
        assert meta["name"] == "sklearn_iris"
        assert meta["platform"] == "onnxruntime_onnx"

        cfg = client.get_model_config("sklearn_iris")
        assert cfg["name"] == "sklearn_iris"
        assert cfg["max_batch_size"] == 16

        x = triton_daemon.sample_x[:2]
        inp = triton_http.InferInput("input", list(x.shape), "FP32")
        inp.set_data_from_numpy(x, binary_data=False)
        out_req = triton_http.InferRequestedOutput("output", binary_data=False)
        res = client.infer("sklearn_iris", inputs=[inp], outputs=[out_req])
        expected = triton_daemon.sklearn_pipe.predict_proba(x).astype(np.float32)
        np.testing.assert_allclose(res.as_numpy("output"), expected, atol=1e-4)

        stats = client.get_inference_statistics(model_name="sklearn_iris")
        assert len(stats["model_stats"]) == 1
        assert stats["model_stats"][0]["name"] == "sklearn_iris"
        assert stats["model_stats"][0]["inference_stats"]["success"]["count"] >= 1


@pytest.mark.parametrize("binary_data", [False, True])
@pytest.mark.parametrize(
    ("np_dtype", "triton_dtype"),
    [
        (np.float32, "FP32"),
        (np.float64, "FP64"),
        (np.int32, "INT32"),
        (np.int64, "INT64"),
    ],
)
def test_triton_http_infer_json_and_binary_modes(
    triton_daemon: TritonDaemonEndpoints,
    binary_data: bool,
    np_dtype: Any,
    triton_dtype: str,
) -> None:
    """Verifies HTTP inference in both pure OIP v2 JSON mode and Triton binary extension mode on real models."""
    with triton_http.InferenceServerClient(url=triton_daemon.http_url) as client:
        arr = triton_daemon.sample_x.astype(np_dtype)
        arr_f32 = arr.astype(np.float32)
        expected_sklearn = triton_daemon.sklearn_pipe.predict_proba(arr_f32).astype(np.float32)

        inp = triton_http.InferInput("input", list(arr.shape), triton_dtype)
        inp.set_data_from_numpy(arr, binary_data=binary_data)
        out_req = triton_http.InferRequestedOutput("output", binary_data=binary_data)

        res = client.infer(
            "sklearn_iris",
            inputs=[inp],
            outputs=[out_req],
            request_id=f"http-{triton_dtype}-{binary_data}",
        )
        out_arr = res.as_numpy("output")
        assert out_arr is not None
        assert out_arr.dtype == np.float32
        np.testing.assert_allclose(out_arr, expected_sklearn, atol=1e-4)

        # Also test real XGBoost tree model
        if triton_daemon.xgb_clf is not None:
            expected_xgb = triton_daemon.xgb_clf.predict_proba(arr_f32).astype(np.float32)
            xgb_inp = triton_http.InferInput("input", list(arr.shape), triton_dtype)
            xgb_inp.set_data_from_numpy(arr, binary_data=binary_data)
            xgb_out = triton_http.InferRequestedOutput("predictions", binary_data=binary_data)
            xgb_res = client.infer("xgb_iris", inputs=[xgb_inp], outputs=[xgb_out])
            xgb_probs = xgb_res.as_numpy("predictions")
            assert xgb_probs is not None
            np.testing.assert_allclose(xgb_probs, expected_xgb, atol=1e-4)


def test_triton_http_unimplemented_endpoints(
    triton_daemon: TritonDaemonEndpoints,
) -> None:
    """Verifies out-of-scope HTTP endpoints raise InferenceServerException (501)."""
    with triton_http.InferenceServerClient(url=triton_daemon.http_url) as client:
        for fn in [
            client.get_system_shared_memory_status,
            client.get_cuda_shared_memory_status,
            client.get_trace_settings,
            client.get_log_settings,
        ]:
            with pytest.raises(InferenceServerException):
                fn()


def test_triton_http_rbac_authentication(
    triton_auth_daemon: TritonDaemonEndpoints,
) -> None:
    """Verifies 3-tier RBAC enforcement over Triton HTTP headers on a real ONNX model."""
    assert triton_auth_daemon.api_keys is not None
    keys = triton_auth_daemon.api_keys

    with triton_http.InferenceServerClient(url=triton_auth_daemon.http_url) as client:
        assert client.is_server_live() is True
        assert client.is_server_ready() is True

        # Unauthenticated metadata fails
        with pytest.raises(InferenceServerException):
            client.get_model_metadata("secure_iris")

        # ReadOnly can read metadata, cannot infer
        ro_headers = {"x-api-key": keys["readonly"]}
        meta = client.get_model_metadata("secure_iris", headers=ro_headers)
        assert meta["name"] == "secure_iris"

        x = triton_auth_daemon.sample_x
        inp = triton_http.InferInput("input", list(x.shape), "FP32")
        inp.set_data_from_numpy(x, binary_data=False)
        out_req = triton_http.InferRequestedOutput("output", binary_data=False)

        with pytest.raises(InferenceServerException):
            client.infer("secure_iris", inputs=[inp], outputs=[out_req], headers=ro_headers)

        # Inference role can infer, cannot unload
        inf_headers = {"x-api-key": keys["inference"]}
        res = client.infer("secure_iris", inputs=[inp], outputs=[out_req], headers=inf_headers)
        expected = triton_auth_daemon.sklearn_pipe.predict_proba(x).astype(np.float32)
        np.testing.assert_allclose(res.as_numpy("output"), expected, atol=1e-4)

        with pytest.raises(InferenceServerException):
            client.unload_model("secure_iris", headers=inf_headers)

        # Admin role can unload and load
        adm_headers = {"x-api-key": keys["admin"]}
        client.unload_model("secure_iris", headers=adm_headers)
        assert client.is_model_ready("secure_iris", headers=adm_headers) is False
        client.load_model("secure_iris", headers=adm_headers)
        assert client.is_model_ready("secure_iris", headers=adm_headers) is True


def test_triton_http_explicit_cuda_device_handling(
    triton_daemon: TritonDaemonEndpoints,
) -> None:
    """Verifies that loading a model with provider='cuda' fails explicitly on CPU runners (no silent fallback) and executes on GPU."""
    with triton_http.InferenceServerClient(url=triton_daemon.http_url) as client:
        require_gpu = bool(os.environ.get("REQUIRE_GPU"))

        try:
            client.load_model("cuda_rf")
            loaded = True
        except InferenceServerException as exc:
            loaded = False
            err_msg = str(exc)

        if loaded:
            assert client.is_model_ready("cuda_rf") is True
            x = triton_daemon.sample_x
            inp = triton_http.InferInput("input", list(x.shape), "FP32")
            inp.set_data_from_numpy(x, binary_data=True)
            res = client.infer("cuda_rf", inputs=[inp])
            expected = triton_daemon.sklearn_rf.predict_proba(x).astype(np.float32)
            np.testing.assert_allclose(res.as_numpy("output"), expected, atol=1e-4)
            client.unload_model("cuda_rf")
        else:
            if require_gpu:
                pytest.fail(f"REQUIRE_GPU is set but loading cuda_rf failed: {err_msg}")
            # Ensure error explicitly notes failure to load CUDA runtime without silently running as CPU
            assert "cuda" in err_msg.lower() or "500" in err_msg or "503" in err_msg
            assert client.is_model_ready("cuda_rf") is False
