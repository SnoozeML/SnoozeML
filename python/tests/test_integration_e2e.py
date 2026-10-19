"""End-to-end integration tests for the snoozeml Python SDK against a live SnoozeML Rust daemon.

Spawns real ``snooze serve`` processes on ephemeral ports to verify:
  1. Model management lifecycle (list, get, load, pin, unpin, evict/unload).
  2. Synchronous and asynchronous inference over both HTTP REST and native gRPC.
  3. End-to-end numerical equivalence for models exported via ``export_sklearn``,
     ``export_torch``, and ``export_xgboost``.
  4. 3-tier RBAC authentication (Admin, Inference, ReadOnly API keys) and error parity.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import socket
import subprocess
import time
from typing import Generator, Tuple

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
xgb = pytest.importorskip("xgboost")

from snoozeml import (
    AsyncSnoozeClient,
    AuthenticationError,
    InvalidInputError,
    ModelNotFoundError,
    ModelState,
    PermissionDeniedError,
    SnoozeClient,
    export_sklearn,
    export_torch,
    export_xgboost,
)
from tests.test_converters import _MockLinear, _MockReLU, _MockSequential

REPO_ROOT = Path(__file__).resolve().parents[2]


def _find_snooze_binary() -> Path:
    """Locates the newest compiled ``snooze`` binary in ``target/release`` or ``target/debug``."""
    exe_name = "snooze.exe" if os.name == "nt" else "snooze"
    candidates = [
        REPO_ROOT / "target" / "release" / exe_name,
        REPO_ROOT / "target" / "debug" / exe_name,
    ]
    existing = [p for p in candidates if p.is_file()]
    if existing:
        existing.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return existing[0]
    pytest.skip("Compiled snooze binary not found in target/release or target/debug")
    raise RuntimeError("Unreachable")


def _get_free_port() -> int:
    """Allocates an available ephemeral TCP port on 127.0.0.1."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_for_server(proc: subprocess.Popen, http_port: int, timeout: float = 15.0) -> None:
    """Polls ``/health/live`` until the SnoozeML daemon is accepting connections."""
    deadline = time.monotonic() + timeout
    client = SnoozeClient(http_endpoint=f"http://127.0.0.1:{http_port}", timeout=0.5)
    try:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                stderr_out = ""
                if proc.stderr is not None:
                    stderr_out = proc.stderr.read().decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"SnoozeML process exited early with code {proc.returncode}: {stderr_out}"
                )
            try:
                if client.health_live():
                    return
            except Exception:
                pass
            time.sleep(0.1)
    finally:
        client.close()
    raise RuntimeError(f"SnoozeML server did not become ready on port {http_port} within {timeout}s")


@pytest.fixture(scope="module")
def live_snooze_daemon(tmp_path_factory: pytest.TempPathFactory) -> Generator[Tuple[str, str, dict], None, None]:
    """Prepares exported ONNX and XGBoost models and starts a live SnoozeML Rust daemon."""
    snooze_bin = _find_snooze_binary()
    work_dir = tmp_path_factory.mktemp("snooze_e2e")
    models_dir = work_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    iris = load_iris()
    X_iris = iris.data.astype(np.float32)
    y_iris = iris.target.astype(np.int32)

    # 1. Export scikit-learn Pipeline (StandardScaler + LogisticRegression) -> ONNX
    pipe_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, random_state=42))
    pipe_model.fit(X_iris, y_iris)
    export_sklearn(
        pipe_model,
        initial_types=[("float_input", (None, 4))],
        output_path=models_dir / "sklearn_pipe.onnx",
    )

    # 2. Export scikit-learn RandomForestClassifier -> ONNX
    rf_model = RandomForestClassifier(n_estimators=6, max_depth=3, random_state=42)
    rf_model.fit(X_iris, y_iris)
    export_sklearn(
        rf_model,
        initial_types=[("float_input", (None, 4))],
        output_path=models_dir / "sklearn_rf.onnx",
    )

    # 3. Export PyTorch-compatible MLP -> ONNX
    l1 = _MockLinear(4, 6)
    l2 = _MockLinear(6, 3)
    torch_mlp = _MockSequential([l1, _MockReLU(), l2])
    export_torch(
        torch_mlp,
        sample_input=X_iris[:2],
        output_path=models_dir / "torch_mlp.onnx",
    )

    # 4. Export XGBoost Classifier -> .xgb.json (pure-Rust tree backend)
    xgb_clf = xgb.XGBClassifier(
        n_estimators=8,
        max_depth=3,
        learning_rate=0.2,
        objective="multi:softprob",
        random_state=42,
    )
    xgb_clf.fit(X_iris, y_iris)
    export_xgboost(xgb_clf, output_path=models_dir / "xgb_iris.xgb.json")

    http_port = _get_free_port()
    grpc_port = _get_free_port()

    cmd = [
        str(snooze_bin),
        "serve",
        "--host",
        "127.0.0.1",
        "--port",
        str(http_port),
        "--grpc-port",
        str(grpc_port),
        "--models-dir",
        str(models_dir),
        "--demo",
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    try:
        _wait_for_server(proc, http_port)
        http_ep = f"http://127.0.0.1:{http_port}"
        grpc_ep = f"127.0.0.1:{grpc_port}"
        artifacts = {
            "X_sample": X_iris[:5],
            "pipe_model": pipe_model,
            "rf_model": rf_model,
            "l1": l1,
            "l2": l2,
            "xgb_clf": xgb_clf,
        }
        yield http_ep, grpc_ep, artifacts
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_e2e_health_service_info_and_metrics(
    live_snooze_daemon: Tuple[str, str, dict],
) -> None:
    http_ep, grpc_ep, _ = live_snooze_daemon
    with SnoozeClient(http_endpoint=http_ep, grpc_endpoint=grpc_ep) as client:
        assert client.health_live() is True
        assert client.health_ready() is True
        assert client.grpc_health("") is True
        assert client.grpc_health("snoozeml.v1.InferenceService") is True

        info = client.service_info()
        assert info.get("service") == "snooze-ml"
        assert info.get("status") == "ok"

        metrics_text = client.metrics()
        assert "snoozeml_requests_total" in metrics_text


def test_e2e_model_lifecycle_management(
    live_snooze_daemon: Tuple[str, str, dict],
) -> None:
    http_ep, grpc_ep, _ = live_snooze_daemon
    with SnoozeClient(http_endpoint=http_ep, grpc_endpoint=grpc_ep) as client:
        models = client.models.list()
        model_ids = {m.id for m in models}
        assert "demo-model" in model_ids
        assert "sklearn_pipe" in model_ids
        assert "sklearn_rf" in model_ids
        assert "torch_mlp" in model_ids
        assert "xgb_iris" in model_ids

        # Pin demo-model
        pin_res = client.models.pin("demo-model")
        assert pin_res["pinned"] is True
        info = client.models.get("demo-model")
        assert info.pinned is True
        assert info.state == ModelState.READY
        assert info.is_loaded is True

        # Unpin demo-model
        unpin_res = client.models.unpin("demo-model")
        assert unpin_res["pinned"] is False
        assert client.models.get("demo-model").pinned is False

        # Evict / unload demo-model
        evict_res = client.models.evict("demo-model")
        assert evict_res["status"] == "unloaded"
        info_after = client.models.get("demo-model")
        assert info_after.state == ModelState.IDLE
        assert info_after.is_loaded is False

        # Explicit load
        load_res = client.models.load("demo-model")
        assert load_res["status"] == "loaded"
        assert client.models.get("demo-model").is_loaded is True

        # Dynamic HTTP upload with custom runtime settings
        upload_info = client.upload_model(
            file=b"fake_mock_bytes",
            filename="dynamic_uploaded_model.onnx",
            model_id="dynamic_uploaded_model",
            pinned=True,
            max_batch_size=32,
            max_batch_delay_ms=25,
            idle_timeout_secs=120,
            description="Dynamically uploaded via HTTP",
        )
        assert upload_info.id == "dynamic_uploaded_model"
        assert upload_info.pinned is True
        assert upload_info.max_batch_size == 32
        assert upload_info.max_batch_delay_ms == 25
        assert upload_info.idle_timeout_secs == 120
        assert upload_info.description == "Dynamically uploaded via HTTP"

        # Runtime refresh via HTTP
        refreshed_models = client.refresh_models()
        refreshed_ids = {m.id for m in refreshed_models}
        assert "dynamic_uploaded_model" in refreshed_ids


def test_e2e_http_and_grpc_inference_equivalence(
    live_snooze_daemon: Tuple[str, str, dict],
) -> None:
    http_ep, grpc_ep, artifacts = live_snooze_daemon
    X_sample: np.ndarray = artifacts["X_sample"]

    with SnoozeClient(http_endpoint=http_ep, grpc_endpoint=grpc_ep) as client:
        # 1. Demo mock model: HTTP vs gRPC
        x_demo = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)
        res_http = client.infer("demo-model", {"input": x_demo}, use_grpc=False)
        res_grpc = client.infer("demo-model", {"input": x_demo}, use_grpc=True)
        np.testing.assert_allclose(res_http.get_numpy("output"), res_grpc.get_numpy("output"), atol=1e-5)
        assert res_http.get_numpy().shape == (2, 4)

        # 2. Exported scikit-learn Pipeline (StandardScaler + LogisticRegression)
        expected_pipe = artifacts["pipe_model"].predict_proba(X_sample).astype(np.float32)
        pipe_http = client.infer("sklearn_pipe", {"float_input": X_sample}, use_grpc=False)
        pipe_grpc = client.infer("sklearn_pipe", {"float_input": X_sample}, use_grpc=True)
        np.testing.assert_allclose(pipe_http.get_numpy(), expected_pipe, atol=1e-4)
        np.testing.assert_allclose(pipe_grpc.get_numpy(), expected_pipe, atol=1e-4)

        # 3. Exported scikit-learn RandomForestClassifier
        expected_rf = artifacts["rf_model"].predict_proba(X_sample).astype(np.float32)
        rf_http = client.infer("sklearn_rf", {"float_input": X_sample}, use_grpc=False)
        rf_grpc = client.infer("sklearn_rf", {"float_input": X_sample}, use_grpc=True)
        np.testing.assert_allclose(rf_http.get_numpy(), expected_rf, atol=1e-4)
        np.testing.assert_allclose(rf_grpc.get_numpy(), expected_rf, atol=1e-4)

        # 4. Exported PyTorch-compatible MLP (Linear -> ReLU -> Linear)
        l1, l2 = artifacts["l1"], artifacts["l2"]
        hidden = np.maximum(0.0, X_sample @ l1.weight.T + l1.bias)
        expected_mlp = (hidden @ l2.weight.T + l2.bias).astype(np.float32)
        mlp_http = client.infer("torch_mlp", {"input": X_sample}, use_grpc=False)
        mlp_grpc = client.infer("torch_mlp", {"input": X_sample}, use_grpc=True)
        np.testing.assert_allclose(mlp_http.get_numpy(), expected_mlp, atol=1e-4)
        np.testing.assert_allclose(mlp_grpc.get_numpy(), expected_mlp, atol=1e-4)

        # 5. Exported XGBoost model (pure-Rust hessboost backend)
        expected_xgb = artifacts["xgb_clf"].predict_proba(X_sample).astype(np.float32)
        xgb_http = client.infer("xgb_iris", {"features": X_sample}, use_grpc=False)
        xgb_grpc = client.infer("xgb_iris", {"features": X_sample}, use_grpc=True)
        np.testing.assert_allclose(xgb_http.get_numpy().ravel(), expected_xgb.ravel(), atol=1e-4)
        np.testing.assert_allclose(xgb_grpc.get_numpy().ravel(), expected_xgb.ravel(), atol=1e-4)


def test_e2e_async_client_and_concurrent_batching(
    live_snooze_daemon: Tuple[str, str, dict],
) -> None:
    http_ep, grpc_ep, artifacts = live_snooze_daemon
    X_sample: np.ndarray = artifacts["X_sample"]

    async def _run_async_checks() -> None:
        async with AsyncSnoozeClient(
            http_endpoint=http_ep,
            grpc_endpoint=grpc_ep,
            use_grpc=True,
        ) as aclient:
            assert await aclient.health_live() is True
            assert await aclient.grpc_health("snoozeml.v1.InferenceService") is True

            models = await aclient.models.list()
            assert any(m.id == "demo-model" for m in models)

            # Dispatch 8 concurrent single-row inference requests to exercise dynamic batching
            tasks = [
                aclient.infer("sklearn_pipe", {"float_input": X_sample[i : i + 1]})
                for i in range(len(X_sample))
            ]
            responses = await asyncio.gather(*tasks)
            expected = artifacts["pipe_model"].predict_proba(X_sample).astype(np.float32)
            for idx, resp in enumerate(responses):
                np.testing.assert_allclose(resp.get_numpy(), expected[idx : idx + 1], atol=1e-4)

    asyncio.run(_run_async_checks())


def test_e2e_error_handling_parity(
    live_snooze_daemon: Tuple[str, str, dict],
) -> None:
    http_ep, grpc_ep, _ = live_snooze_daemon
    x = np.ones((1, 4), dtype=np.float32)

    with SnoozeClient(http_endpoint=http_ep, grpc_endpoint=grpc_ep) as client:
        with pytest.raises(ModelNotFoundError):
            client.models.get("nonexistent-model-id")

        with pytest.raises(ModelNotFoundError):
            client.infer("nonexistent-model-id", {"input": x}, use_grpc=False)

        with pytest.raises(ModelNotFoundError):
            client.infer("nonexistent-model-id", {"input": x}, use_grpc=True)

        with pytest.raises(InvalidInputError):
            client.infer("", {"input": x})


def test_e2e_authenticated_rbac_daemon() -> None:
    """Spawns a SnoozeML daemon with 3-tier RBAC enabled and verifies role enforcement."""
    snooze_bin = _find_snooze_binary()
    http_port = _get_free_port()
    grpc_port = _get_free_port()

    cmd = [
        str(snooze_bin),
        "serve",
        "--host",
        "127.0.0.1",
        "--port",
        str(http_port),
        "--grpc-port",
        str(grpc_port),
        "--demo",
        "--auth-enabled",
        "--protect-metrics",
        "--api-key",
        "readonly:sk-readonly-key-111",
        "--api-key",
        "infer:sk-infer-key-222",
        "--api-key",
        "admin:sk-admin-key-333",
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    try:
        _wait_for_server(proc, http_port)
        http_ep = f"http://127.0.0.1:{http_port}"
        grpc_ep = f"127.0.0.1:{grpc_port}"
        x = np.ones((1, 4), dtype=np.float32)

        # 1. Unauthenticated client -> AuthenticationError (401 / UNAUTHENTICATED)
        with SnoozeClient(http_endpoint=http_ep, grpc_endpoint=grpc_ep) as anon:
            assert anon.health_live() is True  # Public probe
            with pytest.raises(AuthenticationError):
                anon.models.list()
            with pytest.raises(AuthenticationError):
                anon.infer("demo-model", {"input": x}, use_grpc=False)
            with pytest.raises(AuthenticationError):
                anon.infer("demo-model", {"input": x}, use_grpc=True)

        # 2. ReadOnly client -> can list/get, cannot infer or pin
        with SnoozeClient(
            http_endpoint=http_ep,
            grpc_endpoint=grpc_ep,
            api_key="sk-readonly-key-111",
        ) as ro_client:
            assert len(ro_client.models.list()) == 1
            with pytest.raises(PermissionDeniedError):
                ro_client.infer("demo-model", {"input": x}, use_grpc=False)
            with pytest.raises(PermissionDeniedError):
                ro_client.infer("demo-model", {"input": x}, use_grpc=True)
            with pytest.raises(PermissionDeniedError):
                ro_client.models.pin("demo-model")

        # 3. Inference client -> can list and infer (HTTP & gRPC), cannot pin/unload/metrics
        with SnoozeClient(
            http_endpoint=http_ep,
            grpc_endpoint=grpc_ep,
            api_key="sk-infer-key-222",
        ) as inf_client:
            res_h = inf_client.infer("demo-model", {"input": x}, use_grpc=False)
            res_g = inf_client.infer("demo-model", {"input": x}, use_grpc=True)
            np.testing.assert_allclose(res_h.get_numpy(), res_g.get_numpy())
            with pytest.raises(PermissionDeniedError):
                inf_client.models.pin("demo-model")
            with pytest.raises(PermissionDeniedError):
                inf_client.metrics()

        # 4. Admin client -> full access
        with SnoozeClient(
            http_endpoint=http_ep,
            grpc_endpoint=grpc_ep,
            api_key="sk-admin-key-333",
        ) as admin_client:
            assert admin_client.models.pin("demo-model")["pinned"] is True
            assert "snoozeml_requests_total" in admin_client.metrics()
            assert admin_client.models.unpin("demo-model")["pinned"] is False
            assert admin_client.models.evict("demo-model")["status"] == "unloaded"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
