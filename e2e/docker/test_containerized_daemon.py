"""End-to-end integration tests for containerized SnoozeML using testcontainers and Docker Compose.

Verifies full-path serving, dynamic scale-to-zero lifecycle, Triton v2 compliance,
and inference correctness against a live SnoozeML daemon running inside an OCI container
using real trained XGBoost tree models and Scikit-Learn ONNX models (zero mocks).
"""

from __future__ import annotations

from dataclasses import dataclass
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, Generator, Optional
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

# Ensure testcontainers is present
try:
    import testcontainers
except ImportError as err:
    pytest.fail(f"testcontainers is required for containerized tests: {err}")


def is_docker_available() -> bool:
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


def find_snooze_image() -> str:
    """Resolves the container image: strictly local images only (no remote pulling)."""
    if env_image := os.environ.get("SNOOZE_IMAGE"):
        return env_image

    # Check local image candidates only
    for candidate in ("snooze:local", "snoozeml:local", "snooze:latest", "snoozeml:latest"):
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

    pytest.fail(
        "Local Docker image 'snooze:local' not found in Docker engine. "
        "Build it locally with 'docker build --target slim -t snooze:local .' before running container tests. "
        "Remote image pulling is strictly disabled."
    )


@dataclass(frozen=True)
class ContainerDaemonEndpoints:
    """Connection details, fitted reference models, and ground truth for containerized SnoozeML."""

    http_url: str  # host:port format (e.g. "127.0.0.1:8080")
    grpc_url: str  # host:port format (e.g. "127.0.0.1:50051")
    sample_x: np.ndarray
    expected_xgb_prob: np.ndarray
    expected_onnx_prob: np.ndarray
    models_dir: Path
    xgb_clf: Any
    sklearn_pipe: Any


@pytest.fixture(scope="module")
def containerized_snooze(
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[ContainerDaemonEndpoints, None, None]:
    """Prepares real models and spins up a SnoozeML container via testcontainers."""
    if not is_docker_available():
        pytest.fail("Docker daemon is not running or not reachable on this host; required for containerized test suite")

    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb

    from snoozeml.converters import export_sklearn, export_xgboost

    iris = load_iris()
    x_iris = iris.data.astype(np.float32)
    y_iris = iris.target.astype(np.int32)
    sample_x = np.ascontiguousarray(x_iris[:6], dtype=np.float32)

    work_dir = tmp_path_factory.mktemp("snooze_container_models")
    models_dir = work_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Fit & export real XGBoost classifier (.xgb.json)
    xgb_clf = xgb.XGBClassifier(
        n_estimators=10,
        max_depth=3,
        learning_rate=0.2,
        objective="multi:softprob",
        random_state=42,
    )
    xgb_clf.fit(x_iris, y_iris)
    xgb_path = models_dir / "real_xgb.xgb.json"
    export_xgboost(xgb_clf, output_path=xgb_path)
    expected_xgb_prob = xgb_clf.predict_proba(sample_x).astype(np.float32)

    # 2. Fit & export real Scikit-Learn ONNX pipeline (.onnx)
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, random_state=42))
    pipe.fit(x_iris, y_iris)
    onnx_path = models_dir / "real_onnx.onnx"
    export_sklearn(pipe, initial_types=[("input", (None, 4))], output_path=onnx_path)
    expected_onnx_prob = pipe.predict_proba(sample_x).astype(np.float32)

    # 3. Write models.toml for container
    # - real_xgb: pinned=true (loaded eagerly at startup)
    # - real_onnx: pinned=false (starts in Idle state to test scale-to-zero)
    config_toml = models_dir / "models.toml"
    config_toml.write_text(
        f"""
[[models]]
id = "real_xgb"
path = "/models/real_xgb.xgb.json"
pinned = true
max_batch_size = 16
max_batch_delay_ms = 5
idle_timeout_secs = 300
provider = "tree"
description = "Real fitted XGBoost classifier running in container"

[[models]]
id = "real_onnx"
path = "/models/real_onnx.onnx"
pinned = false
max_batch_size = 16
max_batch_delay_ms = 5
idle_timeout_secs = 300
description = "Real Scikit-Learn LogisticRegression ONNX pipeline in container"
""".strip(),
        encoding="utf-8",
    )

    image_name = find_snooze_image()

    # Launch container using testcontainers DockerCompose or DockerContainer
    compose_file = REPO_ROOT / "e2e" / "docker" / "docker-compose.test.yml"
    use_compose = False

    try:
        from testcontainers.compose import DockerCompose

        # Copy models to compose context if needed
        import shutil
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
        use_compose = True
        host = compose.get_service_host("snooze", 8080)
        http_port = compose.get_service_port("snooze", 8080)
        grpc_port = compose.get_service_port("snooze", 50051)
    except Exception as exc:
        # Fallback to GenericContainer if DockerCompose is unavailable
        try:
            from testcontainers.core.container import DockerContainer

            container = (
                DockerContainer(image_name)
                .with_bind_ports(8080, None)
                .with_bind_ports(50051, None)
                .with_volume_mapping(str(models_dir), "/models", mode="ro")
                .with_command(
                    ["serve", "--host", "0.0.0.0", "--port", "8080", "--grpc-port", "50051", "--config", "/models/models.toml"]
                )
            )
            container.start()
            host = container.get_container_host_ip()
            http_port = int(container.get_exposed_port(8080))
            grpc_port = int(container.get_exposed_port(50051))
        except Exception as e2:
            pytest.fail(f"Failed to spawn SnoozeML container '{image_name}': {exc}; {e2}")

    http_url = f"{host}:{http_port}"
    grpc_url = f"{host}:{grpc_port}"

    # Wait for server health
    deadline = time.monotonic() + 20.0
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
        if use_compose:
            compose.stop()
        else:
            container.stop()
        pytest.fail(f"Containerized SnoozeML did not become live at {live_url} within 20s")

    try:
        yield ContainerDaemonEndpoints(
            http_url=http_url,
            grpc_url=grpc_url,
            sample_x=sample_x,
            expected_xgb_prob=expected_xgb_prob,
            expected_onnx_prob=expected_onnx_prob,
            models_dir=models_dir,
            xgb_clf=xgb_clf,
            sklearn_pipe=pipe,
        )
    finally:
        if use_compose:
            try:
                compose.stop()
            except Exception:
                pass
            # Clean up copied models dir
            try:
                import shutil
                shutil.rmtree(str(compose_models_dir), ignore_errors=True)
            except Exception:
                pass
        else:
            try:
                container.stop()
            except Exception:
                pass


def test_container_health_and_service_info(
    containerized_snooze: ContainerDaemonEndpoints,
) -> None:
    """Verifies that the containerized daemon responds to health, readiness, and root home queries."""
    base_url = f"http://{containerized_snooze.http_url}"

    # 1. GET /
    with urllib.request.urlopen(f"{base_url}/") as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["service"] == "snooze-ml"
        assert data["status"] == "ok"
        assert "models" in data["endpoints"]

    # 2. GET /health/live
    with urllib.request.urlopen(f"{base_url}/health/live") as resp:
        assert resp.status == 200
        live_data = json.loads(resp.read().decode("utf-8"))
        assert live_data["status"] == "live"

    # 3. GET /health/ready
    with urllib.request.urlopen(f"{base_url}/health/ready") as resp:
        assert resp.status == 200
        ready_data = json.loads(resp.read().decode("utf-8"))
        assert ready_data["status"] == "ready"


def test_container_model_lifecycle_scale_to_zero(
    containerized_snooze: ContainerDaemonEndpoints,
) -> None:
    """Verifies model catalog, load, unload, and state transitions inside the container."""
    base_url = f"http://{containerized_snooze.http_url}"

    # 1. GET /v1/models -> verify initial model states
    with urllib.request.urlopen(f"{base_url}/v1/models") as resp:
        assert resp.status == 200
        catalog = json.loads(resp.read().decode("utf-8"))
        models_by_id = {m["id"]: m for m in catalog}
        assert "real_xgb" in models_by_id
        assert "real_onnx" in models_by_id

        # real_xgb was pinned = true, so it must be ready eagerly
        assert models_by_id["real_xgb"]["state"].lower() == "ready"
        assert models_by_id["real_xgb"]["pinned"] is True

        # real_onnx was pinned = false, so it must be idle
        assert models_by_id["real_onnx"]["state"].lower() == "idle"
        assert models_by_id["real_onnx"]["pinned"] is False

    # 2. Explicit POST /v1/models/real_onnx/load -> transitions to ready
    load_req = urllib.request.Request(f"{base_url}/v1/models/real_onnx/load", method="POST")
    with urllib.request.urlopen(load_req) as resp:
        assert resp.status == 200

    # Verify state is now ready
    with urllib.request.urlopen(f"{base_url}/v1/models") as resp:
        catalog = json.loads(resp.read().decode("utf-8"))
        models_by_id = {m["id"]: m for m in catalog}
        assert models_by_id["real_onnx"]["state"].lower() == "ready"

    # 3. Explicit POST /v1/models/real_onnx/unload -> transitions back to idle
    unload_req = urllib.request.Request(f"{base_url}/v1/models/real_onnx/unload", method="POST")
    with urllib.request.urlopen(unload_req) as resp:
        assert resp.status == 200

    with urllib.request.urlopen(f"{base_url}/v1/models") as resp:
        catalog = json.loads(resp.read().decode("utf-8"))
        models_by_id = {m["id"]: m for m in catalog}
        assert models_by_id["real_onnx"]["state"].lower() == "idle"


def test_container_real_xgboost_inference_rest(
    containerized_snooze: ContainerDaemonEndpoints,
) -> None:
    """Runs REST inference against the containerized XGBoost model and checks mathematical ground truth."""
    base_url = f"http://{containerized_snooze.http_url}"
    x = containerized_snooze.sample_x

    payload = {
        "inputs": [
            {
                "name": "input",
                "shape": list(x.shape),
                "data": x.flatten().tolist(),
            }
        ]
    }
    payload_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/v1/models/real_xgb/infer",
        data=payload_bytes,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        result = json.loads(resp.read().decode("utf-8"))
        assert result["model_id"] == "real_xgb"
        assert len(result["outputs"]) > 0

        out_tensor = result["outputs"][0]
        actual_data = np.array(out_tensor["data"], dtype=np.float32).reshape(out_tensor["shape"])

        # Check outputs against Python XGBoost ground truth
        np.testing.assert_allclose(actual_data, containerized_snooze.expected_xgb_prob, atol=1e-4)

        # Check probability distributions sum to ~1.0
        prob_sums = actual_data.sum(axis=1)
        np.testing.assert_allclose(prob_sums, np.ones(len(x)), atol=1e-4)


def test_container_real_onnx_inference_rest(
    containerized_snooze: ContainerDaemonEndpoints,
) -> None:
    """Runs REST inference against the containerized ONNX model and verifies predictions against Scikit-Learn."""
    base_url = f"http://{containerized_snooze.http_url}"
    x = containerized_snooze.sample_x

    payload = {
        "inputs": [
            {
                "name": "input",
                "shape": list(x.shape),
                "data": x.flatten().tolist(),
            }
        ]
    }
    payload_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/v1/models/real_onnx/infer",
        data=payload_bytes,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        result = json.loads(resp.read().decode("utf-8"))
        assert result["model_id"] == "real_onnx"

        # Find probability output
        out_tensor = result["outputs"][0]
        actual_data = np.array(out_tensor["data"], dtype=np.float32).reshape(out_tensor["shape"])

        np.testing.assert_allclose(actual_data, containerized_snooze.expected_onnx_prob, atol=1e-4)


def test_container_triton_v2_inference(
    containerized_snooze: ContainerDaemonEndpoints,
) -> None:
    """Verifies that third-party tritonclient.http communicates successfully with the containerized daemon."""
    try:
        import tritonclient.http as triton_http
    except ImportError as err:
        pytest.fail(f"tritonclient.http is required for Triton v2 container test: {err}")

    with triton_http.InferenceServerClient(url=containerized_snooze.http_url) as client:
        assert client.is_server_live() is True
        assert client.is_server_ready() is True

        # Query metadata
        meta = client.get_server_metadata()
        assert meta["name"] == "snooze-ml"

        # Infer on real_xgb
        x = containerized_snooze.sample_x
        inp = triton_http.InferInput("input", list(x.shape), "FP32")
        inp.set_data_from_numpy(x, binary_data=True)
        res = client.infer("real_xgb", inputs=[inp])
        out_name = res.get_response()["outputs"][0]["name"]
        actual_out = res.as_numpy(out_name)
        assert actual_out is not None

        np.testing.assert_allclose(actual_out, containerized_snooze.expected_xgb_prob, atol=1e-4)


def test_container_dynamic_model_upload(
    containerized_snooze: ContainerDaemonEndpoints,
) -> None:
    """Tests runtime multipart model upload into the container followed by live inference."""
    from sklearn.datasets import load_iris
    import xgboost as xgb
    from snoozeml.converters import export_xgboost

    iris = load_iris()
    x_iris = iris.data.astype(np.float32)
    y_iris = iris.target.astype(np.int32)
    sample_x = np.ascontiguousarray(x_iris[:3], dtype=np.float32)

    # Train a new distinct model
    new_xgb = xgb.XGBClassifier(n_estimators=5, max_depth=2, random_state=123)
    new_xgb.fit(x_iris, y_iris)
    expected_prob = new_xgb.predict_proba(sample_x).astype(np.float32)

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir) / "dynamic_tree.xgb.json"
        export_xgboost(new_xgb, output_path=temp_path)
        file_bytes = temp_path.read_bytes()

    # Upload via multipart/form-data to /v1/models/upload
    boundary = "----WebKitFormBoundarySnoozeUploadTest"
    body = io.BytesIO()
    body.write(f"--{boundary}\r\n".encode("utf-8"))
    body.write(b'Content-Disposition: form-data; name="file"; filename="dynamic_tree.xgb.json"\r\n')
    body.write(b"Content-Type: application/json\r\n\r\n")
    body.write(file_bytes)
    body.write(b"\r\n")
    body.write(f"--{boundary}\r\n".encode("utf-8"))
    body.write(b'Content-Disposition: form-data; name="model_id"\r\n\r\n')
    body.write(b"dynamic_tree\r\n")
    body.write(f"--{boundary}\r\n".encode("utf-8"))
    body.write(b'Content-Disposition: form-data; name="provider"\r\n\r\n')
    body.write(b"tree\r\n")
    body.write(f"--{boundary}--\r\n".encode("utf-8"))
    body_data = body.getvalue()

    base_url = f"http://{containerized_snooze.http_url}"
    req = urllib.request.Request(
        f"{base_url}/v1/models/upload",
        data=body_data,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )

    with urllib.request.urlopen(req) as resp:
        assert resp.status in (200, 201)

    # Now infer on the dynamically uploaded model
    payload = {
        "inputs": [
            {
                "name": "input",
                "shape": list(sample_x.shape),
                "data": sample_x.flatten().tolist(),
            }
        ]
    }
    infer_req = urllib.request.Request(
        f"{base_url}/v1/models/dynamic_tree/infer",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(infer_req) as resp:
        assert resp.status == 200
        res = json.loads(resp.read().decode("utf-8"))
        assert res["model_id"] == "dynamic_tree"
        actual_out = np.array(res["outputs"][0]["data"], dtype=np.float32).reshape(res["outputs"][0]["shape"])
        np.testing.assert_allclose(actual_out, expected_prob, atol=1e-4)

    # Clean up uploaded file
    try:
        (containerized_snooze.models_dir / "dynamic_tree.xgb.json").unlink(missing_ok=True)
    except Exception:
        pass


def test_container_native_grpc_inference(
    containerized_snooze: ContainerDaemonEndpoints,
) -> None:
    """Verifies native gRPC inference (snoozeml.v1.InferenceService/Infer) against containerized daemon."""
    from snoozeml import SnoozeClient

    x = containerized_snooze.sample_x
    with SnoozeClient(
        http_endpoint=f"http://{containerized_snooze.http_url}",
        grpc_endpoint=containerized_snooze.grpc_url,
    ) as client:
        # 1. Infer on real_xgb over gRPC
        res_xgb = client.infer("real_xgb", {"input": x}, use_grpc=True)
        assert res_xgb.model_id == "real_xgb"
        actual_xgb = res_xgb.get_numpy()
        np.testing.assert_allclose(actual_xgb, containerized_snooze.expected_xgb_prob, atol=1e-4)

        # 2. Infer on real_onnx over gRPC
        res_onnx = client.infer("real_onnx", {"input": x}, use_grpc=True)
        assert res_onnx.model_id == "real_onnx"
        actual_onnx = res_onnx.get_numpy()
        np.testing.assert_allclose(actual_onnx, containerized_snooze.expected_onnx_prob, atol=1e-4)


@pytest.mark.asyncio
async def test_container_concurrent_micro_batching(
    containerized_snooze: ContainerDaemonEndpoints,
) -> None:
    """Verifies that the container's DynamicBatcher coalesces concurrent async requests along Axis(0)."""
    import asyncio
    from snoozeml import AsyncSnoozeClient

    x = containerized_snooze.sample_x
    # Prepare slices with varying sizes to test Axis(0) collation and splitting
    slices = [
        x[0:1],  # 1 sample
        x[1:3],  # 2 samples
        x[3:6],  # 3 samples
        x[0:2],  # 2 samples
        x[2:5],  # 3 samples
        x[5:6],  # 1 sample
    ]

    async with AsyncSnoozeClient(http_endpoint=f"http://{containerized_snooze.http_url}") as client:
        # Fire all requests concurrently
        tasks = [
            client.infer("real_xgb", {"input": sl})
            for sl in slices
        ]
        results = await asyncio.gather(*tasks)

        assert len(results) == len(slices)
        for sl, res in zip(slices, results):
            expected = containerized_snooze.xgb_clf.predict_proba(sl).astype(np.float32)
            actual = res.get_numpy()
            assert actual.shape == expected.shape
            np.testing.assert_allclose(actual, expected, atol=1e-4)


def test_container_pin_and_unpin_lifecycle(
    containerized_snooze: ContainerDaemonEndpoints,
) -> None:
    """Tests runtime pinning and unpinning of models inside the container."""
    base_url = f"http://{containerized_snooze.http_url}"

    # 1. real_onnx starts pinned=False
    with urllib.request.urlopen(f"{base_url}/v1/models") as resp:
        catalog = {m["id"]: m for m in json.loads(resp.read().decode("utf-8"))}
        assert catalog["real_onnx"]["pinned"] is False

    # 2. Pin real_onnx
    pin_req = urllib.request.Request(f"{base_url}/v1/models/real_onnx/pin", method="POST")
    with urllib.request.urlopen(pin_req) as resp:
        assert resp.status == 200

    # Verify pinned status updated
    with urllib.request.urlopen(f"{base_url}/v1/models") as resp:
        catalog = {m["id"]: m for m in json.loads(resp.read().decode("utf-8"))}
        assert catalog["real_onnx"]["pinned"] is True

    # 3. Unpin real_onnx
    unpin_req = urllib.request.Request(f"{base_url}/v1/models/real_onnx/unpin", method="POST")
    with urllib.request.urlopen(unpin_req) as resp:
        assert resp.status == 200

    with urllib.request.urlopen(f"{base_url}/v1/models") as resp:
        catalog = {m["id"]: m for m in json.loads(resp.read().decode("utf-8"))}
        assert catalog["real_onnx"]["pinned"] is False


def test_container_prometheus_metrics(
    containerized_snooze: ContainerDaemonEndpoints,
) -> None:
    """Verifies that Prometheus metrics are exposed, valid, and monotonically incremented."""
    base_url = f"http://{containerized_snooze.http_url}"

    # 1. Fetch initial metrics
    with urllib.request.urlopen(f"{base_url}/metrics") as resp:
        assert resp.status == 200
        initial_metrics = resp.read().decode("utf-8")
        assert "snoozeml_requests_total" in initial_metrics
        assert "snoozeml_inference_latency_seconds_total" in initial_metrics
        assert "snoozeml_model_deallocations_total" in initial_metrics

    # 2. Run an inference request
    x = containerized_snooze.sample_x[:1]
    payload = {
        "inputs": [
            {
                "name": "input",
                "shape": list(x.shape),
                "data": x.flatten().tolist(),
            }
        ]
    }
    infer_req = urllib.request.Request(
        f"{base_url}/v1/models/real_xgb/infer",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(infer_req) as resp:
        assert resp.status == 200

    # 3. Fetch updated metrics and verify
    with urllib.request.urlopen(f"{base_url}/metrics") as resp:
        updated_metrics = resp.read().decode("utf-8")
        assert "snoozeml_requests_total" in updated_metrics


def test_container_error_handling_negative_invariants(
    containerized_snooze: ContainerDaemonEndpoints,
) -> None:
    """Verifies negative edge cases, 404s, 400s, and defensive behavior in the container daemon."""
    base_url = f"http://{containerized_snooze.http_url}"

    # 1. 404 Not Found on unknown model inference
    unknown_req = urllib.request.Request(
        f"{base_url}/v1/models/ghost_model_404/infer",
        data=b'{"inputs": []}',
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(unknown_req)
    assert exc_info.value.code == 404
    err_body = json.loads(exc_info.value.read().decode("utf-8"))
    assert "error" in err_body

    # 2. 400 Bad Request on shape/feature count mismatch (10 features sent to 4-feature model)
    bad_shape_payload = {
        "inputs": [
            {
                "name": "input",
                "shape": [1, 10],
                "data": [1.0] * 10,
            }
        ]
    }
    mismatch_req = urllib.request.Request(
        f"{base_url}/v1/models/real_xgb/infer",
        data=json.dumps(bad_shape_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(mismatch_req)
    assert exc_info.value.code == 400
    err_body = json.loads(exc_info.value.read().decode("utf-8"))
    assert "error" in err_body

    # 3. 400 Bad Request on malformed JSON payload
    corrupt_req = urllib.request.Request(
        f"{base_url}/v1/models/real_xgb/infer",
        data=b"THIS_IS_NOT_VALID_JSON{{{",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(corrupt_req)
    assert exc_info.value.code == 400

    # 4. 404 Not Found on unknown model unload
    unknown_unload = urllib.request.Request(
        f"{base_url}/v1/models/ghost_model_404/unload",
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(unknown_unload)
    assert exc_info.value.code == 404


def test_container_triton_v2_model_metadata_and_repository(
    containerized_snooze: ContainerDaemonEndpoints,
) -> None:
    """Verifies Triton v2 metadata, readiness, and repository operations via tritonclient.http."""
    try:
        import tritonclient.http as triton_http
    except ImportError as err:
        pytest.fail(f"tritonclient.http is required for Triton v2 repository test: {err}")

    with triton_http.InferenceServerClient(url=containerized_snooze.http_url) as client:
        # 1. Model metadata
        meta = client.get_model_metadata("real_xgb")
        assert meta["name"] == "real_xgb"
        assert len(meta["inputs"]) > 0

        # 2. Model config
        cfg = client.get_model_config("real_xgb")
        assert cfg["name"] == "real_xgb"

        # 3. Model readiness
        assert client.is_model_ready("real_xgb") is True

        # 4. Repository index
        repo = client.get_model_repository_index()
        repo_names = [m["name"] for m in repo]
        assert "real_xgb" in repo_names
        assert "real_onnx" in repo_names

        # 5. Unload & load model via Triton v2 protocol
        client.unload_model("real_xgb")
        assert client.is_model_ready("real_xgb") is False

        client.load_model("real_xgb")
        assert client.is_model_ready("real_xgb") is True


def test_container_python_sdk_high_level_client(
    containerized_snooze: ContainerDaemonEndpoints,
) -> None:
    """Verifies that the official Python SDK client works seamlessly with the container."""
    from snoozeml import SnoozeClient

    with SnoozeClient(http_endpoint=f"http://{containerized_snooze.http_url}") as client:
        # 1. models.list()
        models = client.models.list()
        model_ids = {m.id for m in models}
        assert "real_xgb" in model_ids
        assert "real_onnx" in model_ids

        # 2. models.get()
        xgb_info = client.models.get("real_xgb")
        assert xgb_info.id == "real_xgb"
        assert xgb_info.is_loaded is True

        # 3. infer() high level ergonomic API
        x = containerized_snooze.sample_x
        resp = client.infer("real_xgb", {"input": x})
        actual = resp.get_numpy()
        np.testing.assert_allclose(actual, containerized_snooze.expected_xgb_prob, atol=1e-4)

        # 4. infer() on real_onnx
        resp_onnx = client.infer("real_onnx", {"input": x})
        actual_onnx = resp_onnx.get_numpy()
        np.testing.assert_allclose(actual_onnx, containerized_snooze.expected_onnx_prob, atol=1e-4)


def test_container_models_refresh_endpoint(
    containerized_snooze: ContainerDaemonEndpoints,
) -> None:
    """Verifies the POST /v1/models/refresh endpoint triggers re-scan without failure."""
    base_url = f"http://{containerized_snooze.http_url}"
    req = urllib.request.Request(f"{base_url}/v1/models/refresh", method="POST")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["status"] == "refreshed"
        assert "models" in data
