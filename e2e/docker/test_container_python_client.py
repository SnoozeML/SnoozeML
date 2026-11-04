"""End-to-end integration tests running the SnoozeML Python SDK against a containerized Rust daemon.

Spawns a local CPU-only SnoozeML container (no CUDA) via testcontainers or Docker Compose
to verify that ALL endpoints and supported model architectures operate seamlessly across
container network boundaries:

1. Endpoints Covered:
   - Health & Diagnostics: GET /health/live, GET /health/ready, GET /, GET /metrics, grpc.health.v1.Health/Check
   - Model Lifecycle Management: GET /v1/models, POST /v1/models/{id}/load, POST /v1/models/{id}/unload,
     POST /v1/models/{id}/pin, POST /v1/models/{id}/unpin, POST /v1/models/refresh, POST /v1/models/upload
   - Open Inference Protocol (OIP v2) / NVIDIA Triton HTTP REST:
     GET /v2, GET /v2/health/live, GET /v2/health/ready, GET /v2/models/{name}, GET /v2/models/{name}/ready,
     GET /v2/models/{name}/config, GET /v2/models/stats, GET /v2/models/{name}/stats, POST /v2/repository/index,
     POST /v2/repository/models/{name}/load, POST /v2/repository/models/{name}/unload, POST /v2/models/{name}/infer
   - Inference Transports:
     POST /v1/models/{id}/infer (REST), snoozeml.v1.InferenceService/Infer (gRPC),
     snoozeml.v1.InferenceService/StreamInfer (bi-directional streaming), infer_dataset chunking.

2. Model Architectures Trained Locally & Deployed:
   - Pure-Rust Tree / GBDT (`provider: tree` / `hessboost`): trained with XGBoost on Iris (.xgb.json).
   - Scikit-Learn Logistic Regression ONNX Pipeline (`provider: ort`): trained with scikit-learn (.onnx).
   - Scikit-Learn Random Forest ONNX (`provider: ort`): trained with scikit-learn ensemble (.onnx).
   - Candle Safetensors Sequential MLP (`provider: candle`): 2-layer ReLU MLP (.safetensors).
   - Deterministic Mock Engine (`mock:cpu`): synthetic scale-to-zero and error testing.

3. Authentication, Security & RBAC:
   - Dual header authentication: `x-api-key: <key>` and `Authorization: Bearer <key>`.
   - 3-tier Role-Based Access Control: Admin, Inference, ReadOnly permissions.
   - Public endpoint reachability without credentials.
   - Typed error mapping across REST and gRPC (401 Auth, 403 Forbidden, 404 NotFound, 400 InvalidInput).

4. Async Client Parity:
   - Full AsyncSnoozeClient test suite covering health, lifecycle, uploads, concurrent batching, and streaming.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional
import urllib.error
import urllib.request

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PYTHON_SDK = REPO_ROOT / "python"
if str(PYTHON_SDK) not in sys.path:
    sys.path.insert(0, str(PYTHON_SDK))

import numpy as np
import pytest
import safetensors.numpy
from skl2onnx import to_onnx
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from snoozeml import (
    AsyncSnoozeClient,
    AuthenticationError,
    DaemonMetrics,
    InferenceResponse,
    InvalidInputError,
    ModelInfo,
    ModelNotFoundError,
    ModelState,
    ModelStats,
    PermissionDeniedError,
    SnoozeClient,
    SnoozeError,
    export_sklearn,
    export_xgboost,
)


def _is_docker_available() -> bool:
    """Returns True if the local Docker daemon is reachable and responding."""
    try:
        res = subprocess.run(
            ["docker", "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=3,
        )
        return res.returncode == 0
    except Exception:
        return False


def _find_snooze_image() -> Optional[str]:
    """Resolves local CPU container image candidate (strictly local, zero remote pulls)."""
    if env_image := os.environ.get("SNOOZE_IMAGE"):
        return env_image

    candidates = [
        "snooze:local",
        "snoozeml:local",
        "snooze:slim",
        "snooze:cpu",
        "snooze:latest",
        "snoozeml:latest",
    ]
    for candidate in candidates:
        try:
            res = subprocess.run(
                ["docker", "image", "inspect", candidate],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2,
            )
            if res.returncode == 0:
                return candidate
        except Exception:
            pass
    return None


@dataclass(frozen=True)
class ContainerDaemonContext:
    """Connection parameters, ground-truth models, and credentials for containerized SnoozeML."""

    http_url: str  # host:port (e.g. "127.0.0.1:8080")
    grpc_url: str  # host:port (e.g. "127.0.0.1:50051")
    sample_x: np.ndarray
    expected_xgb_prob: np.ndarray
    expected_onnx_prob: np.ndarray
    xgb_clf: Any
    sklearn_pipe: Any
    admin_api_key: str
    infer_api_key: str
    readonly_api_key: str
    models_dir: Path


@pytest.fixture(scope="module")
def container_client_daemon(
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[ContainerDaemonContext, None, None]:
    """Spins up a CPU SnoozeML daemon container configured with real models and RBAC."""
    if not _is_docker_available():
        pytest.skip("Docker daemon is not running or reachable; skipping container client test suite")

    image_name = _find_snooze_image()
    if image_name is None:
        pytest.skip(
            "Local Docker image 'snooze:local' not found in Docker engine. "
            "Build with 'docker build --target slim -t snooze:local .' before running container tests."
        )

    # 1. Prepare Iris dataset & ground truth
    iris = load_iris()
    x_iris = iris.data.astype(np.float32)
    y_iris = iris.target.astype(np.int32)
    sample_x = np.ascontiguousarray(x_iris[:8], dtype=np.float32)

    work_dir = tmp_path_factory.mktemp("snooze_container_client_models")
    models_dir = work_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # 2. Fit & export real XGBoost classifier (.xgb.json)
    xgb_clf = xgb.XGBClassifier(
        n_estimators=10,
        max_depth=3,
        learning_rate=0.2,
        objective="multi:softprob",
        random_state=42,
    )
    xgb_clf.fit(x_iris, y_iris)
    xgb_path = models_dir / "xgb_iris.xgb.json"
    export_xgboost(xgb_clf, output_path=xgb_path)
    expected_xgb_prob = xgb_clf.predict_proba(sample_x).astype(np.float32)

    # 3. Fit & export real Scikit-Learn ONNX pipeline (.onnx)
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, random_state=42))
    pipe.fit(x_iris, y_iris)
    onnx_path = models_dir / "sklearn_pipe.onnx"
    export_sklearn(pipe, initial_types=[("float_input", (None, 4))], output_path=onnx_path)
    expected_onnx_prob = pipe.predict_proba(sample_x).astype(np.float32)

    admin_key = "admin-secret-key-12345"
    infer_key = "infer-secret-key-12345"
    readonly_key = "read-secret-key-12345"

    # 4. Write models.toml with 3-tier RBAC and multiple initial model configurations
    config_toml = models_dir / "models.toml"
    config_toml.write_text(
        f"""
[auth]
enabled = true
protect_metrics = false

[[auth.api_keys]]
key = "{admin_key}"
role = "admin"
name = "admin-token"

[[auth.api_keys]]
key = "{infer_key}"
role = "inference"
name = "inference-token"

[[auth.api_keys]]
key = "{readonly_key}"
role = "readonly"
name = "readonly-token"

[[models]]
id = "xgb_iris"
path = "/models/xgb_iris.xgb.json"
pinned = true
max_batch_size = 16
max_batch_delay_ms = 5
idle_timeout_secs = 300
provider = "tree"
description = "Pure-Rust XGBoost Iris Classifier"

[[models]]
id = "sklearn_pipe"
path = "/models/sklearn_pipe.onnx"
pinned = false
max_batch_size = 16
max_batch_delay_ms = 5
idle_timeout_secs = 300
description = "Scikit-Learn Logistic Regression ONNX Pipeline"

[[models]]
id = "mock_cpu"
path = "mock:cpu"
pinned = false
max_batch_size = 16
max_batch_delay_ms = 5
idle_timeout_secs = 300
description = "Deterministic Mock Engine"
""".strip(),
        encoding="utf-8",
    )

    compose_models_dir = None
    compose_instance = None
    container_instance = None

    try:
        try:
            from testcontainers.compose import DockerCompose

            compose_dir = REPO_ROOT / "e2e" / "docker"
            compose_models_dir = compose_dir / "models"
            if compose_models_dir.exists():
                shutil.rmtree(str(compose_models_dir), ignore_errors=True)
            shutil.copytree(str(models_dir), str(compose_models_dir))

            os.environ["SNOOZE_IMAGE"] = image_name
            compose = DockerCompose(
                str(compose_dir),
                compose_file_name="docker-compose.test.yml",
                pull=False,
            )
            compose.start()
            compose_instance = compose
            host = compose.get_service_host("snooze", 8080)
            http_port = compose.get_service_port("snooze", 8080)
            grpc_port = compose.get_service_port("snooze", 50051)
        except Exception:
            # Clean up half-started compose service before fallback
            if compose_instance is not None:
                try:
                    compose_instance.stop()
                except Exception:
                    pass
                compose_instance = None

            from testcontainers.core.container import DockerContainer

            container = (
                DockerContainer(image_name)
                .with_bind_ports(8080, None)
                .with_bind_ports(50051, None)
                .with_volume_mapping(str(models_dir), "/models", mode="rw")
                .with_command(
                    [
                        "serve",
                        "--host",
                        "0.0.0.0",
                        "--port",
                        "8080",
                        "--grpc-port",
                        "50051",
                        "--config",
                        "/models/models.toml",
                    ]
                )
            )
            container.start()
            container_instance = container
            host = container.get_container_host_ip()
            http_port = int(container.get_exposed_port(8080))
            grpc_port = int(container.get_exposed_port(50051))

        http_url = f"{host}:{http_port}"
        grpc_url = f"{host}:{grpc_port}"

        # Poll until server is ready
        deadline = time.monotonic() + 25.0
        live_url = f"http://{http_url}/health/live"
        live = False
        while time.monotonic() < deadline:
            try:
                req = urllib.request.Request(live_url, method="GET")
                with urllib.request.urlopen(req, timeout=1.0) as resp:
                    if resp.status == 200:
                        live = True
                        break
            except Exception:
                time.sleep(0.5)

        if not live:
            pytest.fail(f"Containerized SnoozeML did not become live at {live_url} within 25s")

        yield ContainerDaemonContext(
            http_url=http_url,
            grpc_url=grpc_url,
            sample_x=sample_x,
            expected_xgb_prob=expected_xgb_prob,
            expected_onnx_prob=expected_onnx_prob,
            xgb_clf=xgb_clf,
            sklearn_pipe=pipe,
            admin_api_key=admin_key,
            infer_api_key=infer_key,
            readonly_api_key=readonly_key,
            models_dir=models_dir,
        )
    finally:
        if compose_instance is not None:
            try:
                compose_instance.stop()
            except Exception:
                pass
        if container_instance is not None:
            try:
                container_instance.stop()
            except Exception:
                pass
        if compose_models_dir is not None and compose_models_dir.exists():
            try:
                shutil.rmtree(str(compose_models_dir), ignore_errors=True)
            except Exception:
                pass


# ==============================================================================
# 1. SERVICE HEALTH, PROBING & PROMETHEUS METRICS VIA PYTHON SDK
# ==============================================================================


def test_client_health_service_info_and_metrics(
    container_client_daemon: ContainerDaemonContext,
) -> None:
    """Verifies SnoozeClient health checks, gRPC health stubs, discovery info, and Prometheus metrics."""
    ctx = container_client_daemon
    with SnoozeClient(
        http_endpoint=f"http://{ctx.http_url}",
        grpc_endpoint=ctx.grpc_url,
        api_key=ctx.admin_api_key,
    ) as client:
        # 1. HTTP Liveness and Readiness
        assert client.health_live() is True
        assert client.health_ready() is True
        assert client.wait_until_ready(timeout=5.0) is True

        # 2. Native gRPC Standard Health Check
        assert client.grpc_health("") is True
        assert client.grpc_health("snoozeml.v1.InferenceService") is True

        # 3. Discovery Metadata (GET /)
        info = client.service_info()
        assert info.get("service") == "snooze-ml"
        assert info.get("status") == "ok"
        assert "models" in info.get("endpoints", {})

        # 4. Prometheus Metrics Parsing
        raw_metrics = client.metrics()
        assert "snoozeml_requests_total" in raw_metrics

        daemon_metrics = client.get_metrics()
        assert isinstance(daemon_metrics, DaemonMetrics)
        assert daemon_metrics.requests_total >= 0

        # 5. OIP v2 / Triton Health Probes
        v2_live = client._request_raw("GET", "/v2/health/live")
        assert v2_live.status_code == 200

        v2_ready = client._request_raw("GET", "/v2/health/ready")
        assert v2_ready.status_code == 200

        v2_meta = client._request_json("GET", "/v2")
        assert v2_meta.get("name") == "snooze-ml"
        assert "extensions" in v2_meta


# ==============================================================================
# 2. MODEL LIFECYCLE MANAGEMENT (LIST, GET, PIN, UNPIN, UNLOAD, LOAD, EVICT, STATS)
# ==============================================================================


def test_client_model_lifecycle_scale_to_zero(
    container_client_daemon: ContainerDaemonContext,
) -> None:
    """Verifies that SnoozeClient controls dynamic scale-to-zero, pinning, eviction, and stats."""
    ctx = container_client_daemon
    with SnoozeClient(
        http_endpoint=f"http://{ctx.http_url}",
        grpc_endpoint=ctx.grpc_url,
        api_key=ctx.admin_api_key,
    ) as client:
        # 1. models.list() & list_models() shorthand
        models_list = client.models.list()
        shorthand_list = client.list_models()
        assert len(models_list) == len(shorthand_list)

        catalog = {m.id: m for m in models_list}
        assert "xgb_iris" in catalog
        assert "sklearn_pipe" in catalog
        assert "mock_cpu" in catalog

        # xgb_iris is pinned=true -> starts in READY state
        assert catalog["xgb_iris"].state == ModelState.READY
        assert catalog["xgb_iris"].pinned is True
        assert catalog["xgb_iris"].is_loaded is True

        # sklearn_pipe is pinned=false -> starts in IDLE state (scale-to-zero)
        assert catalog["sklearn_pipe"].state == ModelState.IDLE
        assert catalog["sklearn_pipe"].pinned is False
        assert catalog["sklearn_pipe"].is_loaded is False

        # 2. models.get(model_id) & get_model(model_id)
        pipe_info: ModelInfo = client.models.get("sklearn_pipe")
        assert pipe_info.id == "sklearn_pipe"
        assert pipe_info.is_loaded is False

        # 3. models.load(model_id) -> transitions IDLE to READY
        load_res = client.models.load("sklearn_pipe")
        assert load_res["status"] == "loaded"
        assert client.models.get("sklearn_pipe").state == ModelState.READY
        assert client.models.get("sklearn_pipe").is_loaded is True

        # 4. models.pin(model_id) & unpin(model_id)
        pin_res = client.models.pin("sklearn_pipe")
        assert pin_res["pinned"] is True
        assert client.models.get("sklearn_pipe").pinned is True

        unpin_res = client.models.unpin("sklearn_pipe")
        assert unpin_res["pinned"] is False
        assert client.models.get("sklearn_pipe").pinned is False

        # 5. models.unload(model_id) -> transitions back to IDLE
        unload_res = client.models.unload("sklearn_pipe")
        assert unload_res["status"] == "unloaded"
        assert client.models.get("sklearn_pipe").state == ModelState.IDLE
        assert client.models.get("sklearn_pipe").is_loaded is False

        # 6. models.evict(model_id) -> alias for unload
        client.models.load("sklearn_pipe")
        evict_res = client.models.evict("sklearn_pipe")
        assert evict_res["status"] == "unloaded"
        assert client.models.get("sklearn_pipe").is_loaded is False

        # 7. models.stats() & models.get_stats(model_id)
        all_stats = client.models.stats()
        assert isinstance(all_stats, list)
        xgb_stats = client.models.get_stats("xgb_iris")
        assert isinstance(xgb_stats, ModelStats)
        assert xgb_stats.model_id == "xgb_iris"

        # 8. models.refresh() -> returns List[ModelInfo]
        refreshed_models = client.models.refresh()
        assert isinstance(refreshed_models, list)
        refreshed_ids = {m.id for m in refreshed_models}
        assert "xgb_iris" in refreshed_ids


# ==============================================================================
# 3. TRITON / OPEN INFERENCE PROTOCOL (OIP v2) REST ENDPOINTS
# ==============================================================================


def test_client_triton_oip_v2_endpoints(
    container_client_daemon: ContainerDaemonContext,
) -> None:
    """Verifies all NVIDIA Triton / OIP v2 REST endpoints (metadata, ready, config, stats, index, infer)."""
    ctx = container_client_daemon
    with SnoozeClient(
        http_endpoint=f"http://{ctx.http_url}",
        grpc_endpoint=ctx.grpc_url,
        api_key=ctx.admin_api_key,
    ) as client:
        # 1. GET /v2/models/{name} (model metadata)
        meta = client._request_json("GET", "/v2/models/xgb_iris")
        assert meta.get("name") == "xgb_iris"
        assert "inputs" in meta
        assert "outputs" in meta

        # 2. GET /v2/models/{name}/ready
        ready_resp = client._request_raw("GET", "/v2/models/xgb_iris/ready")
        assert ready_resp.status_code == 200

        # 3. GET /v2/models/{name}/config
        config_data = client._request_json("GET", "/v2/models/xgb_iris/config")
        assert config_data.get("name") == "xgb_iris"
        assert "dynamic_batching" in config_data

        # 4. POST /v2/repository/index
        index_entries = client._request_json("POST", "/v2/repository/index")
        assert isinstance(index_entries, list)
        names = {item["name"] for item in index_entries}
        assert "xgb_iris" in names

        # 5. POST /v2/models/{name}/infer (OIP v2 inference)
        x_flat = ctx.sample_x[:2].flatten().tolist()
        v2_payload = {
            "inputs": [
                {
                    "name": "input",
                    "shape": [2, 4],
                    "datatype": "FP32",
                    "data": x_flat,
                }
            ]
        }
        infer_data = client._request_json("POST", "/v2/models/xgb_iris/infer", json_payload=v2_payload)
        assert infer_data.get("model_name") == "xgb_iris"
        outputs = infer_data.get("outputs", [])
        assert len(outputs) > 0
        assert outputs[0]["shape"] == [2, 3]

        # 6. POST /v2/repository/models/{name}/unload & load
        unload_resp = client._request_raw("POST", "/v2/repository/models/xgb_iris/unload")
        assert unload_resp.status_code == 200

        load_resp = client._request_raw("POST", "/v2/repository/models/xgb_iris/load")
        assert load_resp.status_code == 200


# ==============================================================================
# 4. LOCAL TRAINING & DYNAMIC DEPLOYMENT: PURE-RUST TREE (XGBOOST GBDT)
# ==============================================================================


def test_training_and_deploy_pure_rust_tree_xgboost(
    container_client_daemon: ContainerDaemonContext,
) -> None:
    """Trains a distinct XGBoost classifier locally, deploys it via deploy_xgboost, and tests REST/gRPC parity."""
    ctx = container_client_daemon
    iris = load_iris()
    x_iris = iris.data.astype(np.float32)
    y_iris = iris.target.astype(np.int32)
    test_x = np.ascontiguousarray(x_iris[:4], dtype=np.float32)

    # Train a new local model
    custom_xgb = xgb.XGBClassifier(
        n_estimators=6,
        max_depth=2,
        learning_rate=0.3,
        objective="multi:softprob",
        random_state=123,
    )
    custom_xgb.fit(x_iris, y_iris)
    expected_prob = custom_xgb.predict_proba(test_x).astype(np.float32)

    with SnoozeClient(
        http_endpoint=f"http://{ctx.http_url}",
        grpc_endpoint=ctx.grpc_url,
        api_key=ctx.admin_api_key,
    ) as client:
        # Deploy using client.deploy_xgboost (converts and uploads in one step)
        info = client.deploy_xgboost(
            custom_xgb,
            model_id="custom_xgb_dynamic",
            description="Dynamically trained XGBoost tree model",
            pinned=False,
            max_batch_size=8,
        )
        assert info.id == "custom_xgb_dynamic"

        # REST Inference
        resp_rest = client.infer("custom_xgb_dynamic", {"input": test_x}, use_grpc=False)
        assert resp_rest.model_id == "custom_xgb_dynamic"
        prob_rest = resp_rest.get_numpy()
        np.testing.assert_allclose(prob_rest, expected_prob, atol=1e-4)

        # gRPC Inference
        resp_grpc = client.infer("custom_xgb_dynamic", {"input": test_x}, use_grpc=True)
        assert resp_grpc.model_id == "custom_xgb_dynamic"
        prob_grpc = resp_grpc.get_numpy()
        np.testing.assert_allclose(prob_grpc, expected_prob, atol=1e-4)

        # Strict REST vs gRPC Equivalence
        np.testing.assert_allclose(prob_rest, prob_grpc, atol=1e-5)


# ==============================================================================
# 5. LOCAL TRAINING & DYNAMIC DEPLOYMENT: SCIKIT-LEARN ONNX PIPELINE
# ==============================================================================


def test_training_and_deploy_sklearn_logistic_regression_onnx(
    container_client_daemon: ContainerDaemonContext,
) -> None:
    """Trains a Scikit-Learn Logistic Regression pipeline locally, deploys via deploy_sklearn, and tests REST/gRPC parity."""
    ctx = container_client_daemon
    iris = load_iris()
    x_iris = iris.data.astype(np.float32)
    y_iris = iris.target.astype(np.int32)
    test_x = np.ascontiguousarray(x_iris[10:14], dtype=np.float32)

    # Train a new local model
    custom_pipe = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=200, random_state=888))
    custom_pipe.fit(x_iris, y_iris)
    expected_prob = custom_pipe.predict_proba(test_x).astype(np.float32)

    with SnoozeClient(
        http_endpoint=f"http://{ctx.http_url}",
        grpc_endpoint=ctx.grpc_url,
        api_key=ctx.admin_api_key,
    ) as client:
        # Deploy using client.deploy_sklearn
        info = client.deploy_sklearn(
            custom_pipe,
            model_id="custom_sklearn_pipe",
            input_name="float_input",
            description="Dynamically trained Scikit-Learn ONNX Pipeline",
            pinned=False,
        )
        assert info.id == "custom_sklearn_pipe"

        # REST Inference
        resp_rest = client.infer("custom_sklearn_pipe", {"float_input": test_x}, use_grpc=False)
        assert resp_rest.model_id == "custom_sklearn_pipe"
        prob_rest = resp_rest.get_numpy()
        np.testing.assert_allclose(prob_rest, expected_prob, atol=1e-4)

        # gRPC Inference
        resp_grpc = client.infer("custom_sklearn_pipe", {"float_input": test_x}, use_grpc=True)
        prob_grpc = resp_grpc.get_numpy()
        np.testing.assert_allclose(prob_grpc, expected_prob, atol=1e-4)

        # Parity
        np.testing.assert_allclose(prob_rest, prob_grpc, atol=1e-5)


# ==============================================================================
# 6. LOCAL TRAINING & DYNAMIC DEPLOYMENT: SCIKIT-LEARN RANDOM FOREST (ONNX)
# ==============================================================================


def test_training_and_deploy_sklearn_random_forest_onnx(
    container_client_daemon: ContainerDaemonContext,
) -> None:
    """Trains a Random Forest ensemble locally, exports to ONNX, uploads via upload_model, and verifies parity."""
    ctx = container_client_daemon
    iris = load_iris()
    # Predict sepal length from remaining features (regression produces clean float32 tensor)
    x_features = iris.data[:, 1:].astype(np.float32)
    y_target = iris.data[:, 0].astype(np.float32)
    test_x = np.ascontiguousarray(x_features[20:25], dtype=np.float32)

    rf = RandomForestRegressor(n_estimators=4, max_depth=3, random_state=42)
    rf.fit(x_features, y_target)
    expected_pred = rf.predict(test_x).astype(np.float32)
    if expected_pred.ndim == 1:
        expected_pred = expected_pred[:, np.newaxis]

    # Convert to ONNX with skl2onnx
    onnx_proto = to_onnx(rf, test_x[:1])
    onnx_bytes = onnx_proto.SerializeToString()

    with SnoozeClient(
        http_endpoint=f"http://{ctx.http_url}",
        grpc_endpoint=ctx.grpc_url,
        api_key=ctx.admin_api_key,
    ) as client:
        # Upload via SDK upload_model
        info = client.upload_model(
            file=onnx_bytes,
            filename="dynamic_rf.onnx",
            model_id="dynamic_rf",
            description="Dynamically exported Random Forest Regressor ONNX",
            pinned=False,
        )
        assert info.id == "dynamic_rf"

        input_node_name = onnx_proto.graph.input[0].name

        # REST Inference
        resp_rest = client.infer("dynamic_rf", {input_node_name: test_x}, use_grpc=False)
        pred_rest = resp_rest.get_numpy()
        np.testing.assert_allclose(pred_rest, expected_pred, atol=1e-4)

        # gRPC Inference
        resp_grpc = client.infer("dynamic_rf", {input_node_name: test_x}, use_grpc=True)
        pred_grpc = resp_grpc.get_numpy()
        np.testing.assert_allclose(pred_grpc, expected_pred, atol=1e-4)

        # Strict Parity
        np.testing.assert_allclose(pred_rest, pred_grpc, atol=1e-5)


# ==============================================================================
# 7. LOCAL TRAINING & DYNAMIC DEPLOYMENT: CANDLE SAFETENSORS (SEQUENTIAL MLP)
# ==============================================================================


def test_training_and_deploy_candle_safetensors_mlp(
    container_client_daemon: ContainerDaemonContext,
) -> None:
    """Builds a 2-layer sequential ReLU MLP in Safetensors, uploads via upload_model, and verifies against NumPy math."""
    ctx = container_client_daemon
    np.random.seed(999)

    # Layer 1: 4 -> 8 Linear + ReLU
    w1 = np.random.randn(8, 4).astype(np.float32) * 0.5
    b1 = np.random.randn(8).astype(np.float32) * 0.1
    # Layer 2: 8 -> 3 Linear
    w2 = np.random.randn(3, 8).astype(np.float32) * 0.5
    b2 = np.random.randn(3).astype(np.float32) * 0.1

    tensors = {
        "fc1.weight": w1,
        "fc1.bias": b1,
        "fc2.weight": w2,
        "fc2.bias": b2,
    }
    safetensors_bytes = safetensors.numpy.save(tensors)

    # Compute ground truth math: y = (relu(x @ W1.T + b1)) @ W2.T + b2
    test_x = ctx.sample_x[:5]
    h1 = np.maximum(0.0, test_x @ w1.T + b1)
    expected_output = h1 @ w2.T + b2

    with SnoozeClient(
        http_endpoint=f"http://{ctx.http_url}",
        grpc_endpoint=ctx.grpc_url,
        api_key=ctx.admin_api_key,
    ) as client:
        # Upload Safetensors model
        info = client.upload_model(
            file=safetensors_bytes,
            filename="candle_mlp.safetensors",
            model_id="candle_mlp",
            description="Pure Safetensors 2-layer MLP executed by Candle backend",
            pinned=False,
            provider="candle",
        )
        assert info.id == "candle_mlp"

        # REST Inference
        resp_rest = client.infer("candle_mlp", {"input": test_x}, use_grpc=False)
        assert resp_rest.model_id == "candle_mlp"
        actual_rest = resp_rest.get_numpy()
        np.testing.assert_allclose(actual_rest, expected_output, atol=1e-4)

        # gRPC Inference
        resp_grpc = client.infer("candle_mlp", {"input": test_x}, use_grpc=True)
        actual_grpc = resp_grpc.get_numpy()
        np.testing.assert_allclose(actual_grpc, expected_output, atol=1e-4)

        # Parity
        np.testing.assert_allclose(actual_rest, actual_grpc, atol=1e-5)


# ==============================================================================
# 8. DETERMINISTIC MOCK ENGINE EXECUTION & SCALE-TO-ZERO
# ==============================================================================


def test_deterministic_mock_model_execution_and_reclamation(
    container_client_daemon: ContainerDaemonContext,
) -> None:
    """Verifies that the built-in deterministic mock backend executes cleanly across REST and gRPC."""
    ctx = container_client_daemon
    test_x = ctx.sample_x[:3]

    with SnoozeClient(
        http_endpoint=f"http://{ctx.http_url}",
        grpc_endpoint=ctx.grpc_url,
        api_key=ctx.admin_api_key,
    ) as client:
        # REST Inference on mock_cpu
        resp_rest = client.infer("mock_cpu", {"input": test_x}, use_grpc=False)
        assert resp_rest.model_id == "mock_cpu"
        out_rest = resp_rest.get_numpy()
        assert out_rest.shape == test_x.shape

        # gRPC Inference on mock_cpu
        resp_grpc = client.infer("mock_cpu", {"input": test_x}, use_grpc=True)
        assert resp_grpc.model_id == "mock_cpu"
        out_grpc = resp_grpc.get_numpy()
        np.testing.assert_allclose(out_rest, out_grpc, atol=1e-5)

        # Unload mock_cpu to verify deallocation
        unload_res = client.models.unload("mock_cpu")
        assert unload_res["status"] == "unloaded"
        assert client.models.get("mock_cpu").is_loaded is False


# ==============================================================================
# 9. DATASET CHUNKING & BI-DIRECTIONAL gRPC STREAMING
# ==============================================================================


def test_client_infer_dataset_and_streaming(
    container_client_daemon: ContainerDaemonContext,
) -> None:
    """Verifies client.infer_dataset chunk concatenation and client.stream_infer over persistent gRPC."""
    ctx = container_client_daemon
    big_x = np.repeat(ctx.sample_x, 2, axis=0)  # 16 rows
    expected_big_prob = ctx.xgb_clf.predict_proba(big_x).astype(np.float32)

    with SnoozeClient(
        http_endpoint=f"http://{ctx.http_url}",
        grpc_endpoint=ctx.grpc_url,
        api_key=ctx.infer_api_key,
    ) as client:
        # 1. infer_dataset over REST (chunks of batch_size=4)
        ds_rest = client.infer_dataset(
            "xgb_iris", {"input": big_x}, batch_size=4, use_grpc=False
        )
        assert ds_rest.model_id == "xgb_iris"
        prob_ds_rest = ds_rest.get_numpy()
        assert prob_ds_rest.shape == expected_big_prob.shape
        np.testing.assert_allclose(prob_ds_rest, expected_big_prob, atol=1e-4)

        # 2. infer_dataset over gRPC (chunks of batch_size=4)
        ds_grpc = client.infer_dataset(
            "xgb_iris", {"input": big_x}, batch_size=4, use_grpc=True
        )
        prob_ds_grpc = ds_grpc.get_numpy()
        assert prob_ds_grpc.shape == expected_big_prob.shape
        np.testing.assert_allclose(prob_ds_grpc, expected_big_prob, atol=1e-4)

        # 3. Bi-directional gRPC streaming: stream_infer
        stream_chunks = [
            {"input": big_x[0:3]},
            {"input": big_x[3:7]},
            {"input": big_x[7:16]},
        ]
        stream_results = list(client.stream_infer("xgb_iris", stream_chunks))
        assert len(stream_results) == len(stream_chunks)

        for chunk_inp, res in zip(stream_chunks, stream_results):
            sub_x = chunk_inp["input"]
            expected_sub = ctx.xgb_clf.predict_proba(sub_x).astype(np.float32)
            actual_sub = res.get_numpy()
            np.testing.assert_allclose(actual_sub, expected_sub, atol=1e-4)


# ==============================================================================
# 10. ASYNC CLIENT: LIFECYCLE, CONCURRENCY, DYNAMIC BATCHING & STREAMING
# ==============================================================================


@pytest.mark.asyncio
async def test_async_client_lifecycle_and_concurrent_batching(
    container_client_daemon: ContainerDaemonContext,
) -> None:
    """Verifies that AsyncSnoozeClient executes concurrent requests and triggers container micro-batching."""
    ctx = container_client_daemon
    x = ctx.sample_x

    async with AsyncSnoozeClient(
        http_endpoint=f"http://{ctx.http_url}",
        grpc_endpoint=ctx.grpc_url,
        api_key=ctx.admin_api_key,
    ) as async_client:
        # 1. Async health and info
        assert await async_client.health_live() is True
        assert await async_client.health_ready() is True
        assert await async_client.grpc_health("") is True

        info = await async_client.service_info()
        assert info.get("service") == "snooze-ml"

        metrics_text = await async_client.metrics()
        assert "snoozeml_requests_total" in metrics_text

        parsed_metrics = await async_client.get_metrics()
        assert parsed_metrics.requests_total >= 0

        # 2. Async model management
        models = await async_client.models.list()
        assert len(models) >= 2

        pipe_info = await async_client.models.get("sklearn_pipe")
        assert pipe_info.id == "sklearn_pipe"

        stats = await async_client.models.stats()
        assert isinstance(stats, list)

        # 3. Concurrent async inference gathering across varying slice sizes
        slices = [
            x[0:1],
            x[1:3],
            x[3:6],
            x[6:8],
            x[0:2],
            x[2:5],
        ]

        tasks = [
            async_client.infer("xgb_iris", {"input": sl}, use_grpc=False)
            for sl in slices
        ]
        results = await asyncio.gather(*tasks)

        assert len(results) == len(slices)
        for sl, res in zip(slices, results):
            expected = ctx.xgb_clf.predict_proba(sl).astype(np.float32)
            actual = res.get_numpy()
            np.testing.assert_allclose(actual, expected, atol=1e-4)

        # 4. Async infer_dataset
        big_x = np.repeat(x, 2, axis=0)
        expected_big = ctx.xgb_clf.predict_proba(big_x).astype(np.float32)
        ds_res = await async_client.infer_dataset(
            "xgb_iris", {"input": big_x}, batch_size=4, concurrency=2, use_grpc=True
        )
        np.testing.assert_allclose(ds_res.get_numpy(), expected_big, atol=1e-4)

        # 5. Async stream_infer
        stream_chunks = [
            {"input": x[0:2]},
            {"input": x[2:6]},
        ]
        received = []
        async for frame in async_client.stream_infer("xgb_iris", stream_chunks):
            received.append(frame)

        assert len(received) == len(stream_chunks)
        for chunk_inp, frame in zip(stream_chunks, received):
            exp = ctx.xgb_clf.predict_proba(chunk_inp["input"]).astype(np.float32)
            np.testing.assert_allclose(frame.get_numpy(), exp, atol=1e-4)


# ==============================================================================
# 11. 3-TIER ROLE-BASED ACCESS CONTROL (RBAC) & DUAL HEADER AUTHENTICATION
# ==============================================================================


def test_client_rbac_permissions_and_unauthorized_access(
    container_client_daemon: ContainerDaemonContext,
) -> None:
    """Verifies that Admin, Inference, and ReadOnly credentials enforce RBAC rules across REST and gRPC."""
    ctx = container_client_daemon
    x = ctx.sample_x[:2]

    # 1. Unauthenticated client on protected endpoints raises AuthenticationError
    with SnoozeClient(
        http_endpoint=f"http://{ctx.http_url}",
        grpc_endpoint=ctx.grpc_url,
        api_key=None,
    ) as anon_client:
        # Public health is allowed without credentials
        assert anon_client.health_live() is True
        assert anon_client.health_ready() is True
        assert anon_client.service_info().get("status") == "ok"
        assert anon_client.grpc_health("") is True

        # Protected REST infer -> 401 AuthenticationError
        with pytest.raises(AuthenticationError):
            anon_client.infer("xgb_iris", {"input": x}, use_grpc=False)

        # Protected gRPC infer -> UNAUTHENTICATED
        with pytest.raises(AuthenticationError):
            anon_client.infer("xgb_iris", {"input": x}, use_grpc=True)

        # Protected REST catalog -> 401
        with pytest.raises(AuthenticationError):
            anon_client.models.list()

    # 2. Header parity: verify Bearer token syntax vs X-API-Key syntax
    with SnoozeClient(
        http_endpoint=f"http://{ctx.http_url}",
        grpc_endpoint=ctx.grpc_url,
        token=ctx.infer_api_key,  # Passed as Authorization: Bearer <token>
    ) as bearer_client:
        res = bearer_client.infer("xgb_iris", {"input": x}, use_grpc=False)
        assert res.model_id == "xgb_iris"

    # 3. ReadOnly client can inspect catalog and stats, but cannot infer or execute admin commands
    with SnoozeClient(
        http_endpoint=f"http://{ctx.http_url}",
        grpc_endpoint=ctx.grpc_url,
        api_key=ctx.readonly_api_key,
    ) as read_client:
        # Allowed: list models
        models = read_client.models.list()
        assert len(models) >= 2

        # Allowed: get stats
        stats = read_client.models.stats()
        assert isinstance(stats, list)

        # Forbidden: infer on model -> 403 PermissionDeniedError
        with pytest.raises(PermissionDeniedError):
            read_client.infer("xgb_iris", {"input": x}, use_grpc=False)

        # Forbidden: gRPC infer -> 403 / PERMISSION_DENIED
        with pytest.raises(PermissionDeniedError):
            read_client.infer("xgb_iris", {"input": x}, use_grpc=True)

        # Forbidden: pin model -> 403 PermissionDeniedError
        with pytest.raises(PermissionDeniedError):
            read_client.models.pin("xgb_iris")

        # Forbidden: unload model -> 403 PermissionDeniedError
        with pytest.raises(PermissionDeniedError):
            read_client.models.unload("xgb_iris")

    # 4. Inference client can run infer, but cannot execute admin operations (pin, unload, upload)
    with SnoozeClient(
        http_endpoint=f"http://{ctx.http_url}",
        grpc_endpoint=ctx.grpc_url,
        api_key=ctx.infer_api_key,
    ) as infer_client:
        # Allowed: infer over REST & gRPC
        res_rest = infer_client.infer("xgb_iris", {"input": x}, use_grpc=False)
        assert res_rest.model_id == "xgb_iris"

        res_grpc = infer_client.infer("xgb_iris", {"input": x}, use_grpc=True)
        assert res_grpc.model_id == "xgb_iris"

        # Forbidden: unload model -> 403 PermissionDeniedError
        with pytest.raises(PermissionDeniedError):
            infer_client.models.unload("xgb_iris")

        # Forbidden: pin model -> 403 PermissionDeniedError
        with pytest.raises(PermissionDeniedError):
            infer_client.models.pin("xgb_iris")

        # Forbidden: refresh models -> 403 PermissionDeniedError
        with pytest.raises(PermissionDeniedError):
            infer_client.models.refresh()


# ==============================================================================
# 12. DEFENSIVE ERROR HANDLING PARITY & TYPED EXCEPTIONS
# ==============================================================================


def test_client_typed_errors_and_negative_invariants(
    container_client_daemon: ContainerDaemonContext,
) -> None:
    """Verifies that non-existent models, shape mismatches, and empty inputs raise typed exceptions."""
    ctx = container_client_daemon

    with SnoozeClient(
        http_endpoint=f"http://{ctx.http_url}",
        grpc_endpoint=ctx.grpc_url,
        api_key=ctx.admin_api_key,
    ) as client:
        # 1. Non-existent model ID raises ModelNotFoundError
        with pytest.raises(ModelNotFoundError):
            client.infer("ghost_model_non_existent", {"input": ctx.sample_x})

        with pytest.raises(ModelNotFoundError):
            client.models.get("ghost_model_non_existent")

        # 2. Empty or whitespace model ID raises InvalidInputError before wire dispatch
        with pytest.raises(InvalidInputError):
            client.infer("", {"input": ctx.sample_x})

        with pytest.raises(InvalidInputError):
            client.infer("   ", {"input": ctx.sample_x})

        # 3. Dimension mismatch (10 features sent to 4-feature Iris model) raises InvalidInputError
        invalid_shape_x = np.ones((2, 10), dtype=np.float32)
        with pytest.raises(InvalidInputError):
            client.infer("xgb_iris", {"input": invalid_shape_x}, use_grpc=False)

        with pytest.raises(InvalidInputError):
            client.infer("xgb_iris", {"input": invalid_shape_x}, use_grpc=True)
