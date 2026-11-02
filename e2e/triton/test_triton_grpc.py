"""End-to-end interoperability tests using NVIDIA's official ``tritonclient.grpc`` SDK.

All tests execute against real fitted Scikit-Learn ONNX models (``sklearn_iris``,
``sklearn_rf``, ``secure_iris``) and XGBoost tree models (``xgb_iris``) — zero mocks.

Exercises:
  - Server health & metadata: ``is_server_live``, ``is_server_ready``, ``get_server_metadata``
  - Repository index & scale-to-zero lifecycle: ``get_model_repository_index``,
    ``load_model``, ``is_model_ready``, ``unload_model`` on real ONNX models
  - Model introspection: ``get_model_metadata``, ``get_model_config``, ``get_inference_statistics``
  - Unary inference: ``infer`` across ``FP32``, ``FP64``, ``INT32``, ``INT64`` against real
    Scikit-Learn ONNX and XGBoost ``predict_proba`` ground truth
  - Bidirectional gRPC streaming: ``start_stream`` + ``async_stream_infer`` + ``stop_stream``
    verifying real ONNX model predictions for every streamed batch
  - Out-of-scope RPCs returning ``UNIMPLEMENTED``
  - 3-tier RBAC authentication via gRPC request headers
"""

from __future__ import annotations

import os
from functools import partial
from pathlib import Path
import queue
import sys
from typing import Any, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest

triton_grpc = pytest.importorskip("tritonclient.grpc")
from tritonclient.utils import InferenceServerException

from e2e.conftest import TritonDaemonEndpoints


def test_triton_grpc_server_health_and_metadata(
    triton_daemon: TritonDaemonEndpoints,
) -> None:
    """Verifies ServerLive, ServerReady, and ServerMetadata via tritonclient.grpc."""
    with triton_grpc.InferenceServerClient(url=triton_daemon.grpc_url) as client:
        assert client.is_server_live() is True
        assert client.is_server_ready() is True

        meta = client.get_server_metadata()
        assert meta.name == "snooze-ml"
        assert len(meta.version) > 0
        assert "model_repository" in meta.extensions
        assert "binary_tensor_data" in meta.extensions


def test_triton_grpc_repository_and_model_lifecycle(
    triton_daemon: TritonDaemonEndpoints,
) -> None:
    """Verifies RepositoryIndex, ModelReady, RepositoryModelLoad, and RepositoryModelUnload on real ONNX models."""
    with triton_grpc.InferenceServerClient(url=triton_daemon.grpc_url) as client:
        # 1. Ensure unpinned sklearn_rf ONNX model is unloaded while pinned sklearn_iris is ready
        client.unload_model("sklearn_rf")
        assert client.is_model_ready("sklearn_rf") is False
        assert client.is_model_ready("sklearn_iris") is True

        # 2. Repository index reflects unloaded vs loaded state
        repo_all = client.get_model_repository_index()
        names_all = {m.name: m.state for m in repo_all.models}
        assert names_all.get("sklearn_rf") == "UNAVAILABLE"
        assert names_all.get("sklearn_iris") == "READY"

        # 3. Load sklearn_rf into ONNX Runtime and verify READY
        client.load_model("sklearn_rf")
        assert client.is_model_ready("sklearn_rf") is True
        repo_after_load = client.get_model_repository_index()
        states_after_load = {m.name: m.state for m in repo_after_load.models}
        assert states_after_load.get("sklearn_rf") == "READY"

        # 4. Run real inference on loaded sklearn_rf and verify against RandomForest.predict_proba
        x = triton_daemon.sample_x
        inp = triton_grpc.InferInput("input", list(x.shape), "FP32")
        inp.set_data_from_numpy(x)
        res = client.infer("sklearn_rf", inputs=[inp])
        expected_rf = triton_daemon.sklearn_rf.predict_proba(x).astype(np.float32)
        np.testing.assert_allclose(res.as_numpy("output"), expected_rf, atol=1e-4)

        # 5. Unload sklearn_rf and verify UNAVAILABLE again
        client.unload_model("sklearn_rf")
        assert client.is_model_ready("sklearn_rf") is False


def test_triton_grpc_model_metadata_config_and_statistics(
    triton_daemon: TritonDaemonEndpoints,
) -> None:
    """Verifies ModelMetadata, ModelConfig, and ModelStatistics via tritonclient.grpc."""
    with triton_grpc.InferenceServerClient(url=triton_daemon.grpc_url) as client:
        meta = client.get_model_metadata("sklearn_iris")
        assert meta.name == "sklearn_iris"
        assert meta.platform == "onnxruntime_onnx"
        assert "1" in meta.versions

        cfg_resp = client.get_model_config("sklearn_iris")
        assert cfg_resp.config.name == "sklearn_iris"
        assert cfg_resp.config.max_batch_size == 16

        # Perform real ONNX inference on sklearn_iris and verify stats increment
        x = triton_daemon.sample_x[:2]
        inp = triton_grpc.InferInput("input", list(x.shape), "FP32")
        inp.set_data_from_numpy(x)
        out_req = triton_grpc.InferRequestedOutput("output")
        res = client.infer("sklearn_iris", inputs=[inp], outputs=[out_req])
        expected = triton_daemon.sklearn_pipe.predict_proba(x).astype(np.float32)
        np.testing.assert_allclose(res.as_numpy("output"), expected, atol=1e-4)

        stats = client.get_inference_statistics(model_name="sklearn_iris")
        assert len(stats.model_stats) == 1
        model_stat = stats.model_stats[0]
        assert model_stat.name == "sklearn_iris"
        assert model_stat.inference_stats.success.count >= 1
        assert model_stat.execution_count >= 1


@pytest.mark.parametrize(
    ("np_dtype", "triton_dtype"),
    [
        (np.float32, "FP32"),
        (np.float64, "FP64"),
        (np.int32, "INT32"),
        (np.int64, "INT64"),
    ],
)
def test_triton_grpc_infer_dtypes_and_real_models(
    triton_daemon: TritonDaemonEndpoints,
    np_dtype: Any,
    triton_dtype: str,
) -> None:
    """Verifies unary gRPC inference across all supported input dtypes against real ONNX and XGBoost models."""
    with triton_grpc.InferenceServerClient(url=triton_daemon.grpc_url) as client:
        # Cast input to target dtype (FP32, FP64, INT32, INT64) and compute exact sklearn ground truth
        arr = triton_daemon.sample_x.astype(np_dtype)
        arr_f32 = arr.astype(np.float32)
        expected_sklearn = triton_daemon.sklearn_pipe.predict_proba(arr_f32).astype(np.float32)

        inp = triton_grpc.InferInput("input", list(arr.shape), triton_dtype)
        inp.set_data_from_numpy(arr)
        res = client.infer("sklearn_iris", inputs=[inp], request_id=f"req-{triton_dtype}")
        out_arr = res.as_numpy("output")
        assert out_arr is not None
        assert out_arr.dtype == np.float32
        np.testing.assert_allclose(out_arr, expected_sklearn, atol=1e-4)

        # Also verify real XGBoost tree model ground truth
        if triton_daemon.xgb_clf is not None:
            expected_xgb = triton_daemon.xgb_clf.predict_proba(arr_f32).astype(np.float32)
            xgb_inp = triton_grpc.InferInput("input", list(arr.shape), triton_dtype)
            xgb_inp.set_data_from_numpy(arr)
            xgb_res = client.infer("xgb_iris", inputs=[xgb_inp])
            xgb_probs = xgb_res.as_numpy("predictions")
            assert xgb_probs is not None
            np.testing.assert_allclose(xgb_probs, expected_xgb, atol=1e-4)


def test_triton_grpc_bidirectional_stream_infer(
    triton_daemon: TritonDaemonEndpoints,
) -> None:
    """Verifies bidirectional gRPC streaming (ModelStreamInfer) on a real ONNX model."""
    results_queue: "queue.Queue[Tuple[Optional[Any], Optional[InferenceServerException]]]" = (
        queue.Queue()
    )

    def _stream_callback(
        user_data: "queue.Queue[Tuple[Optional[Any], Optional[InferenceServerException]]]",
        result: Optional[Any],
        error: Optional[InferenceServerException],
    ) -> None:
        user_data.put((result, error))

    with triton_grpc.InferenceServerClient(url=triton_daemon.grpc_url) as client:
        client.start_stream(callback=partial(_stream_callback, results_queue))
        try:
            x = triton_daemon.sample_x
            batches: List[np.ndarray] = [
                x[0:1],
                x[1:4],
                x[4:6],
            ]
            for idx, batch in enumerate(batches):
                inp = triton_grpc.InferInput("input", list(batch.shape), "FP32")
                inp.set_data_from_numpy(batch)
                client.async_stream_infer(
                    model_name="sklearn_iris",
                    inputs=[inp],
                    request_id=f"stream-{idx}",
                )

            received = {}
            for _ in range(len(batches)):
                result, error = results_queue.get(timeout=10.0)
                assert error is None, f"Stream inference returned error: {error}"
                assert result is not None
                resp_proto = result.get_response()
                received[resp_proto.id] = result.as_numpy("output")

            for idx, batch in enumerate(batches):
                req_id = f"stream-{idx}"
                assert req_id in received
                expected = triton_daemon.sklearn_pipe.predict_proba(batch).astype(np.float32)
                np.testing.assert_allclose(received[req_id], expected, atol=1e-4)
        finally:
            client.stop_stream()


def test_triton_grpc_unimplemented_endpoints(
    triton_daemon: TritonDaemonEndpoints,
) -> None:
    """Verifies out-of-scope hardware/shared-memory/trace/log RPCs return UNIMPLEMENTED."""
    with triton_grpc.InferenceServerClient(url=triton_daemon.grpc_url) as client:
        for fn in [
            client.get_system_shared_memory_status,
            client.get_cuda_shared_memory_status,
            client.get_trace_settings,
            client.get_log_settings,
        ]:
            with pytest.raises(InferenceServerException) as exc_info:
                fn()
            assert "UNIMPLEMENTED" in str(exc_info.value.status())


def test_triton_grpc_rbac_authentication(
    triton_auth_daemon: TritonDaemonEndpoints,
) -> None:
    """Verifies 3-tier RBAC enforcement over Triton gRPC headers on a real ONNX model."""
    assert triton_auth_daemon.api_keys is not None
    keys = triton_auth_daemon.api_keys

    with triton_grpc.InferenceServerClient(url=triton_auth_daemon.grpc_url) as client:
        # 1. ServerLive / ServerReady remain public
        assert client.is_server_live() is True
        assert client.is_server_ready() is True

        # 2. Unauthenticated ModelMetadata fails
        with pytest.raises(InferenceServerException):
            client.get_model_metadata("secure_iris")

        # 3. ReadOnly key can inspect metadata, but cannot infer or load/unload
        ro_headers = {"x-api-key": keys["readonly"]}
        meta = client.get_model_metadata("secure_iris", headers=ro_headers)
        assert meta.name == "secure_iris"

        x = triton_auth_daemon.sample_x
        inp = triton_grpc.InferInput("input", list(x.shape), "FP32")
        inp.set_data_from_numpy(x)

        with pytest.raises(InferenceServerException):
            client.infer("secure_iris", inputs=[inp], headers=ro_headers)

        # 4. Inference key can infer, but cannot load/unload
        inf_headers = {"x-api-key": keys["inference"]}
        res = client.infer("secure_iris", inputs=[inp], headers=inf_headers)
        expected = triton_auth_daemon.sklearn_pipe.predict_proba(x).astype(np.float32)
        np.testing.assert_allclose(res.as_numpy("output"), expected, atol=1e-4)

        with pytest.raises(InferenceServerException):
            client.unload_model("secure_iris", headers=inf_headers)

        # 5. Admin key can unload and reload
        adm_headers = {"x-api-key": keys["admin"]}
        client.unload_model("secure_iris", headers=adm_headers)
        assert client.is_model_ready("secure_iris", headers=adm_headers) is False
        client.load_model("secure_iris", headers=adm_headers)
        assert client.is_model_ready("secure_iris", headers=adm_headers) is True


def test_triton_grpc_explicit_cuda_device_handling(
    triton_daemon: TritonDaemonEndpoints,
) -> None:
    """Verifies that gRPC model load for provider='cuda' fails explicitly on CPU runners and executes on GPU."""
    with triton_grpc.InferenceServerClient(url=triton_daemon.grpc_url) as client:
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
            inp = triton_grpc.InferInput("input", list(x.shape), "FP32")
            inp.set_data_from_numpy(x)
            res = client.infer("cuda_rf", inputs=[inp])
            expected = triton_daemon.sklearn_rf.predict_proba(x).astype(np.float32)
            np.testing.assert_allclose(res.as_numpy("output"), expected, atol=1e-4)
            client.unload_model("cuda_rf")
        else:
            if require_gpu:
                pytest.fail(f"REQUIRE_GPU is set but loading cuda_rf failed via gRPC: {err_msg}")
            assert (
                "cuda" in err_msg.lower()
                or "internal" in err_msg.lower()
                or "unavailable" in err_msg.lower()
            )
            assert client.is_model_ready("cuda_rf") is False
