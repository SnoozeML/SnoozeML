"""End-to-end integration tests running the SnoozeML Python SDK against a containerized Rust daemon.

Spawns a local CPU-only SnoozeML container (no CUDA) via testcontainers or Docker Compose
to verify that all public methods on both synchronous `SnoozeClient` and asynchronous
`AsyncSnoozeClient` operate seamlessly across container network boundaries:
  1. Service health, readiness, discovery info, and Prometheus metrics.
  2. Complete model lifecycle management (list, get, load, unload, evict, pin, unpin, stats, refresh).
  3. Runtime model upload via the Python SDK client.
  4. Numerical inference equivalence for pure-Rust tree (XGBoost) and ONNX (Scikit-Learn)
     models across both HTTP REST and native gRPC transports.
  5. Automatic dataset micro-batch chunking along Axis(0) (`infer_dataset`).
  6. Bi-directional gRPC streaming (`stream_infer`).
  7. High-concurrency async inference and Dynamic Micro-Batching (`AsyncSnoozeClient`).
  8. 3-tier Role-Based Access Control (Admin, Inference, ReadOnly) and error handling parity.
  9. Typed exception mapping for negative invariants (404 Not Found, 400 Bad Request, Auth errors).
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
from sklearn.datasets import load_iris
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

    # 4. Write models.toml with 3-tier RBAC configuration
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
# 3. INFERENCE ACCURACY & DUAL TRANSPORT PARITY (REST vs NATIVE gRPC)
# ==============================================================================


def test_client_inference_rest_and_grpc_parity(
    container_client_daemon: ContainerDaemonContext,
) -> None:
    """Verifies numerical equivalence between REST and gRPC against XGBoost and ONNX ground truth."""
    ctx = container_client_daemon
    x = ctx.sample_x

    with SnoozeClient(
        http_endpoint=f"http://{ctx.http_url}",
        grpc_endpoint=ctx.grpc_url,
        api_key=ctx.infer_api_key,
    ) as client:
        # --- A. Pure-Rust XGBoost Tree Model ---
        # REST
        resp_xgb_rest: InferenceResponse = client.infer(
            "xgb_iris", {"input": x}, use_grpc=False
        )
        assert resp_xgb_rest.model_id == "xgb_iris"
        assert resp_xgb_rest.duration_ms >= 0.0
        prob_xgb_rest = resp_xgb_rest.get_numpy()
        np.testing.assert_allclose(prob_xgb_rest, ctx.expected_xgb_prob, atol=1e-4)

        # gRPC
        resp_xgb_grpc: InferenceResponse = client.infer(
            "xgb_iris", {"input": x}, use_grpc=True
        )
        assert resp_xgb_grpc.model_id == "xgb_iris"
        prob_xgb_grpc = resp_xgb_grpc.get_numpy()
        np.testing.assert_allclose(prob_xgb_grpc, ctx.expected_xgb_prob, atol=1e-4)

        # Strict REST vs gRPC Equivalence
        np.testing.assert_allclose(prob_xgb_rest, prob_xgb_grpc, atol=1e-5)

        # --- B. Scikit-Learn ONNX Pipeline Model (Auto-loaded on demand) ---
        # REST (tests scale-from-zero lazy load)
        resp_onnx_rest: InferenceResponse = client.infer(
            "sklearn_pipe", {"float_input": x}, use_grpc=False
        )
        assert resp_onnx_rest.model_id == "sklearn_pipe"
        prob_onnx_rest = resp_onnx_rest.get_numpy()
        np.testing.assert_allclose(prob_onnx_rest, ctx.expected_onnx_prob, atol=1e-4)

        # gRPC
        resp_onnx_grpc: InferenceResponse = client.infer(
            "sklearn_pipe", {"float_input": x}, use_grpc=True
        )
        assert resp_onnx_grpc.model_id == "sklearn_pipe"
        prob_onnx_grpc = resp_onnx_grpc.get_numpy()
        np.testing.assert_allclose(prob_onnx_grpc, ctx.expected_onnx_prob, atol=1e-4)

        # Strict REST vs gRPC Equivalence
        np.testing.assert_allclose(prob_onnx_rest, prob_onnx_grpc, atol=1e-5)


# ==============================================================================
# 4. DATASET CHUNKING & BI-DIRECTIONAL gRPC STREAMING
# ==============================================================================


def test_client_infer_dataset_and_streaming(
    container_client_daemon: ContainerDaemonContext,
) -> None:
    """Verifies client.infer_dataset chunk concatenation and client.stream_infer over persistent gRPC."""
    ctx = container_client_daemon
    # Duplicate sample_x to create a multi-batch dataset (16 rows)
    big_x = np.repeat(ctx.sample_x, 2, axis=0)
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
# 5. ASYNC CLIENT: CONCURRENCY, DYNAMIC BATCHING & STREAMING
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
# 6. DYNAMIC RUNTIME MODEL UPLOAD VIA SDK
# ==============================================================================


def test_client_dynamic_model_upload_and_infer(
    container_client_daemon: ContainerDaemonContext,
) -> None:
    """Verifies that the Python SDK upload_model method uploads a model and immediately runs inference."""
    ctx = container_client_daemon
    iris = load_iris()
    x_iris = iris.data.astype(np.float32)
    y_iris = iris.target.astype(np.int32)
    test_x = np.ascontiguousarray(x_iris[:3], dtype=np.float32)

    # Train a new distinct model
    new_clf = xgb.XGBClassifier(n_estimators=4, max_depth=2, random_state=777)
    new_clf.fit(x_iris, y_iris)
    expected_prob = new_clf.predict_proba(test_x).astype(np.float32)

    temp_path = ctx.models_dir / "dynamic_sdk_tree.xgb.json"
    export_xgboost(new_clf, output_path=temp_path)
    file_bytes = temp_path.read_bytes()

    with SnoozeClient(
        http_endpoint=f"http://{ctx.http_url}",
        grpc_endpoint=ctx.grpc_url,
        api_key=ctx.admin_api_key,
    ) as client:
        # Upload via SDK upload_model (provider inferred automatically by backend from .xgb.json)
        uploaded_info = client.upload_model(
            file=file_bytes,
            filename="dynamic_sdk_tree.xgb.json",
            model_id="dynamic_sdk_tree",
            description="Dynamically uploaded model via SnoozeClient",
            pinned=False,
        )
        assert uploaded_info.id == "dynamic_sdk_tree"
        assert uploaded_info.state in (ModelState.IDLE, ModelState.READY)

        # Verify model appears in catalog
        info = client.models.get("dynamic_sdk_tree")
        assert info.id == "dynamic_sdk_tree"

        # Execute inference on the newly uploaded model
        infer_resp = client.infer("dynamic_sdk_tree", {"input": test_x})
        assert infer_resp.model_id == "dynamic_sdk_tree"
        actual_prob = infer_resp.get_numpy()
        np.testing.assert_allclose(actual_prob, expected_prob, atol=1e-4)

    # Clean up local artifact
    try:
        temp_path.unlink()
    except Exception:
        pass


# ==============================================================================
# 7. 3-TIER ROLE-BASED ACCESS CONTROL (RBAC) ENFORCEMENT ACROSS TRANSPORTS
# ==============================================================================


def test_client_rbac_permissions_and_unauthorized_access(
    container_client_daemon: ContainerDaemonContext,
) -> None:
    """Verifies that Admin, Inference, and ReadOnly API keys enforce RBAC rules across REST and gRPC."""
    ctx = container_client_daemon
    x = ctx.sample_x[:2]

    # 1. Unauthenticated client on protected endpoints raises AuthenticationError
    with SnoozeClient(
        http_endpoint=f"http://{ctx.http_url}",
        grpc_endpoint=ctx.grpc_url,
        api_key=None,
    ) as anon_client:
        # Protected REST infer -> 401
        with pytest.raises(AuthenticationError):
            anon_client.infer("xgb_iris", {"input": x}, use_grpc=False)

        # Protected gRPC infer -> UNAUTHENTICATED
        with pytest.raises(AuthenticationError):
            anon_client.infer("xgb_iris", {"input": x}, use_grpc=True)

        # Protected REST catalog -> 401
        with pytest.raises(AuthenticationError):
            anon_client.models.list()

    # 2. ReadOnly client can inspect catalog, but cannot infer or execute admin commands
    with SnoozeClient(
        http_endpoint=f"http://{ctx.http_url}",
        grpc_endpoint=ctx.grpc_url,
        api_key=ctx.readonly_api_key,
    ) as read_client:
        # Allowed: list models
        models = read_client.models.list()
        assert len(models) >= 2

        # Forbidden: infer on model -> 403 PermissionDeniedError
        with pytest.raises(PermissionDeniedError):
            read_client.infer("xgb_iris", {"input": x}, use_grpc=False)

        # Forbidden: gRPC infer -> 403 / PERMISSION_DENIED
        with pytest.raises(PermissionDeniedError):
            read_client.infer("xgb_iris", {"input": x}, use_grpc=True)

        # Forbidden: pin model -> 403 PermissionDeniedError
        with pytest.raises(PermissionDeniedError):
            read_client.models.pin("xgb_iris")

    # 3. Inference client can run infer, but cannot execute admin operations (pin, unload)
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


# ==============================================================================
# 8. DEFENSIVE ERROR HANDLING PARITY & TYPED EXCEPTIONS
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
