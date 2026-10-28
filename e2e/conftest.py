"""Shared pytest fixtures for SnoozeML end-to-end (E2E) system and interoperability tests.

Spawns real ``snooze serve`` child processes on ephemeral localhost ports backed by
real fitted Scikit-Learn ONNX models (LogisticRegression Pipeline & RandomForest) and
XGBoost tree models (``.xgb.json``) — zero mock backends.

You can also target an externally running SnoozeML daemon or container by setting:
  - ``SNOOZE_HTTP_URL`` (e.g. ``localhost:8080``)
  - ``SNOOZE_GRPC_URL`` (e.g. ``localhost:50051``)
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any, Dict, Generator, Optional
import urllib.request

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_SDK_DIR = REPO_ROOT / "python"
if str(PYTHON_SDK_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_SDK_DIR))


@dataclass(frozen=True)
class TritonDaemonEndpoints:
    """Connection details, fitted reference models, and test data for a live SnoozeML daemon."""

    http_url: str  # host:port format expected by tritonclient.http (e.g. "127.0.0.1:18080")
    grpc_url: str  # host:port format expected by tritonclient.grpc (e.g. "127.0.0.1:50051")
    sample_x: np.ndarray
    sklearn_pipe: Any
    sklearn_rf: Any
    xgb_clf: Optional[Any] = None
    api_keys: Optional[Dict[str, str]] = None


def _find_snooze_binary() -> Path:
    """Locates the compiled ``snooze`` binary via ``SNOOZE_BIN`` or ``target/{release,debug}``."""
    env_bin = os.environ.get("SNOOZE_BIN")
    if env_bin:
        bin_path = Path(env_bin)
        if bin_path.is_file():
            return bin_path
        pytest.fail(f"SNOOZE_BIN points to non-existent file: {env_bin}")

    exe_name = "snooze.exe" if os.name == "nt" else "snooze"
    candidates = [
        REPO_ROOT / "target" / "release" / exe_name,
        REPO_ROOT / "target" / "debug" / exe_name,
    ]
    existing = [p for p in candidates if p.is_file()]
    if existing:
        existing.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return existing[0]

    pytest.skip(
        "Compiled snooze binary not found in target/release or target/debug. "
        "Run `cargo build --bin snooze` first."
    )
    raise RuntimeError("Unreachable")


def _get_free_port() -> int:
    """Allocates an available ephemeral TCP port on 127.0.0.1."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_server_live(proc: subprocess.Popen, http_url: str, timeout: float = 15.0) -> None:
    """Polls ``GET /health/live`` until the SnoozeML daemon is accepting requests."""
    deadline = time.monotonic() + timeout
    url = f"http://{http_url}/health/live"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            stderr_out = ""
            if proc.stderr is not None:
                try:
                    stderr_out = proc.stderr.read().decode("utf-8", errors="replace")
                except Exception:
                    pass
            raise RuntimeError(
                f"SnoozeML process exited early with code {proc.returncode}: {stderr_out}"
            )
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=0.5) as resp:
                if resp.status == 200:
                    return
        except Exception:
            pass
        time.sleep(0.1)
    raise RuntimeError(f"SnoozeML server did not become live at {url} within {timeout}s")


def _wait_for_triton_live(proc: subprocess.Popen, http_url: str, timeout: float = 15.0) -> None:
    """Polls ``GET /v2/health/live`` until the SnoozeML daemon is accepting requests."""
    _wait_for_server_live(proc, http_url, timeout)


@dataclass
class SnoozeServerInstance:
    """Context and management handle for an ephemeral SnoozeML daemon."""

    process: subprocess.Popen
    http_port: int
    grpc_port: int
    http_url: str
    grpc_url: str
    work_dir: Path
    config_path: Path


def spawn_snooze_server(
    config_content: str,
    tmp_path: Path,
    extra_args: Optional[list[str]] = None,
    timeout: float = 15.0,
) -> SnoozeServerInstance:
    """Writes config and launches a real ``snooze serve`` daemon on ephemeral ports."""
    snooze_bin = _find_snooze_binary()
    config_path = tmp_path / "models.toml"
    config_path.write_text(config_content.strip(), encoding="utf-8")

    http_port = _get_free_port()
    grpc_port = _get_free_port()
    http_url = f"127.0.0.1:{http_port}"
    grpc_url = f"127.0.0.1:{grpc_port}"

    cmd = [
        str(snooze_bin),
        "serve",
        "--host",
        "127.0.0.1",
        "--port",
        str(http_port),
        "--grpc-port",
        str(grpc_port),
        "--config",
        str(config_path),
    ]
    if extra_args:
        cmd.extend(extra_args)

    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    try:
        _wait_for_server_live(proc, http_url, timeout=timeout)
    except Exception:
        proc.terminate()
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            proc.kill()
        raise

    return SnoozeServerInstance(
        process=proc,
        http_port=http_port,
        grpc_port=grpc_port,
        http_url=http_url,
        grpc_url=grpc_url,
        work_dir=tmp_path,
        config_path=config_path,
    )



@dataclass(frozen=True)
class IrisModelArtifact:
    """Pre-exported Scikit-Learn Iris ONNX model artifact and reference pipeline."""

    onnx_path: Path
    pipeline: Any
    sample_x: np.ndarray


@pytest.fixture(scope="session")
def iris_model_artifact(tmp_path_factory: pytest.TempPathFactory) -> IrisModelArtifact:
    """Session-scoped fixture providing a fitted Scikit-Learn LogisticRegression ONNX model."""
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from snoozeml.converters import export_sklearn

    iris = load_iris()
    x_iris = iris.data.astype(np.float32)
    y_iris = iris.target.astype(np.int32)
    sample_x = np.ascontiguousarray(x_iris[:6], dtype=np.float32)

    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, random_state=42))
    pipe.fit(x_iris, y_iris)

    work_dir = tmp_path_factory.mktemp("snooze_iris_onnx")
    onnx_path = work_dir / "sklearn_iris.onnx"
    export_sklearn(
        pipe,
        initial_types=[("input", (None, 4))],
        output_path=onnx_path,
    )

    return IrisModelArtifact(
        onnx_path=onnx_path,
        pipeline=pipe,
        sample_x=sample_x,
    )


@pytest.fixture(scope="session")
def triton_daemon(
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[TritonDaemonEndpoints, None, None]:
    """Spawns a live SnoozeML daemon serving real ONNX and XGBoost models (zero mocks)."""
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from snoozeml.converters import export_sklearn, export_xgboost

    iris = load_iris()
    x_iris = iris.data.astype(np.float32)
    y_iris = iris.target.astype(np.int32)
    sample_x = np.ascontiguousarray(x_iris[:6], dtype=np.float32)

    # 1. Fit real Scikit-Learn Pipeline (StandardScaler + LogisticRegression)
    sklearn_pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, random_state=42))
    sklearn_pipe.fit(x_iris, y_iris)

    # 2. Fit real Scikit-Learn RandomForestClassifier
    sklearn_rf = RandomForestClassifier(n_estimators=8, max_depth=3, random_state=42)
    sklearn_rf.fit(x_iris, y_iris)

    ext_http = os.environ.get("SNOOZE_HTTP_URL")
    ext_grpc = os.environ.get("SNOOZE_GRPC_URL")
    if ext_http and ext_grpc:
        http_clean = ext_http.replace("http://", "").replace("https://", "").rstrip("/")
        yield TritonDaemonEndpoints(
            http_url=http_clean,
            grpc_url=ext_grpc,
            sample_x=sample_x,
            sklearn_pipe=sklearn_pipe,
            sklearn_rf=sklearn_rf,
        )
        return

    snooze_bin = _find_snooze_binary()
    work_dir = tmp_path_factory.mktemp("snooze_triton_e2e")
    models_dir = work_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    sklearn_iris_path = models_dir / "sklearn_iris.onnx"
    export_sklearn(
        sklearn_pipe,
        initial_types=[("input", (None, 4))],
        output_path=sklearn_iris_path,
    )

    sklearn_rf_path = models_dir / "sklearn_rf.onnx"
    export_sklearn(
        sklearn_rf,
        initial_types=[("input", (None, 4))],
        output_path=sklearn_rf_path,
    )

    # 3. Fit and export real XGBoost model if xgboost is installed
    xgb_clf: Optional[Any] = None
    xgb_toml_block = ""
    try:
        import xgboost as xgb

        xgb_clf = xgb.XGBClassifier(
            n_estimators=8,
            max_depth=3,
            learning_rate=0.2,
            objective="multi:softprob",
            random_state=42,
        )
        xgb_clf.fit(x_iris, y_iris)
        xgb_path = models_dir / "xgb_iris.xgb.json"
        export_xgboost(xgb_clf, output_path=xgb_path)
        xgb_toml_block = f"""
[[models]]
id = "xgb_iris"
path = "{xgb_path.as_posix()}"
max_batch_size = 16
max_batch_delay_ms = 5
idle_timeout_secs = 300
pinned = true
description = "XGBoost Iris classifier (.xgb.json)"
"""
    except ImportError:
        pass

    # 4. Write models.toml:
    # - sklearn_iris: pinned = true (eagerly loaded at startup)
    # - sklearn_rf: pinned = false (starts unloaded in Idle state to test scale-to-zero & load/unload)
    # - xgb_iris: pinned = true
    config_toml = work_dir / "models.toml"
    config_toml.write_text(
        f"""
[[models]]
id = "sklearn_iris"
path = "{sklearn_iris_path.as_posix()}"
max_batch_size = 16
max_batch_delay_ms = 5
idle_timeout_secs = 300
pinned = true
description = "Scikit-Learn Iris LogisticRegression Pipeline (ONNX)"

[[models]]
id = "sklearn_rf"
path = "{sklearn_rf_path.as_posix()}"
max_batch_size = 16
max_batch_delay_ms = 5
idle_timeout_secs = 300
pinned = false
description = "Scikit-Learn Iris RandomForest (ONNX, unpinned)"
{xgb_toml_block}
""".strip(),
        encoding="utf-8",
    )

    http_port = _get_free_port()
    grpc_port = _get_free_port()
    http_url = f"127.0.0.1:{http_port}"
    grpc_url = f"127.0.0.1:{grpc_port}"

    cmd = [
        str(snooze_bin),
        "serve",
        "--host",
        "127.0.0.1",
        "--port",
        str(http_port),
        "--grpc-port",
        str(grpc_port),
        "--config",
        str(config_toml),
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    try:
        _wait_for_triton_live(proc, http_url)
        yield TritonDaemonEndpoints(
            http_url=http_url,
            grpc_url=grpc_url,
            sample_x=sample_x,
            sklearn_pipe=sklearn_pipe,
            sklearn_rf=sklearn_rf,
            xgb_clf=xgb_clf,
        )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture(scope="session")
def triton_auth_daemon(
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[TritonDaemonEndpoints, None, None]:
    """Spawns a live SnoozeML daemon with 3-tier RBAC authentication serving a real ONNX model."""
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from snoozeml.converters import export_sklearn

    snooze_bin = _find_snooze_binary()
    work_dir = tmp_path_factory.mktemp("snooze_triton_auth_e2e")
    models_dir = work_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    iris = load_iris()
    x_iris = iris.data.astype(np.float32)
    y_iris = iris.target.astype(np.int32)
    sample_x = np.ascontiguousarray(x_iris[:3], dtype=np.float32)

    sklearn_pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, random_state=42))
    sklearn_pipe.fit(x_iris, y_iris)
    onnx_path = models_dir / "secure_iris.onnx"
    export_sklearn(
        sklearn_pipe,
        initial_types=[("input", (None, 4))],
        output_path=onnx_path,
    )

    config_toml = work_dir / "models_auth.toml"
    api_keys = {
        "admin": "sk-triton-admin-secret",
        "inference": "sk-triton-infer-secret",
        "readonly": "sk-triton-readonly-secret",
    }

    config_toml.write_text(
        f"""
[auth]
enabled = true

[[auth.api_keys]]
name = "admin-key"
key = "{api_keys['admin']}"
role = "admin"

[[auth.api_keys]]
name = "infer-key"
key = "{api_keys['inference']}"
role = "inference"

[[auth.api_keys]]
name = "readonly-key"
key = "{api_keys['readonly']}"
role = "readonly"

[[models]]
id = "secure_iris"
path = "{onnx_path.as_posix()}"
max_batch_size = 8
max_batch_delay_ms = 5
idle_timeout_secs = 300
pinned = true
""".strip(),
        encoding="utf-8",
    )

    http_port = _get_free_port()
    grpc_port = _get_free_port()
    http_url = f"127.0.0.1:{http_port}"
    grpc_url = f"127.0.0.1:{grpc_port}"

    cmd = [
        str(snooze_bin),
        "serve",
        "--host",
        "127.0.0.1",
        "--port",
        str(http_port),
        "--grpc-port",
        str(grpc_port),
        "--config",
        str(config_toml),
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    try:
        _wait_for_triton_live(proc, http_url)
        yield TritonDaemonEndpoints(
            http_url=http_url,
            grpc_url=grpc_url,
            sample_x=sample_x,
            sklearn_pipe=sklearn_pipe,
            sklearn_rf=sklearn_pipe,
            api_keys=api_keys,
        )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
