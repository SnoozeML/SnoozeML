"""Programmatic configuration builder, custom worker scaffolder, and local daemon runner for SnoozeML.

Generates validated ``models.toml``, ``streams.toml``, and ``models.json`` configuration
files matching SnoozeML's Rust configuration schema (``src/config.rs``), scaffolds
process-isolated Python worker directories (``snooze_model.py``), and manages local
``snooze serve`` child processes from Python.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import textwrap
import time
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Union

from .auth import JwtAlgorithm, Role
from .exceptions import InvalidInputError, SnoozeConnectionError, SnoozeError
from .logging import get_logger
from .streams import StreamWireFormat
from .worker import run_check

if TYPE_CHECKING:
    from .client import AsyncSnoozeClient, SnoozeClient

logger = get_logger("config")


def _toml_val(val: Any) -> str:
    """Formats a primitive Python value as a TOML literal."""
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, int):
        return str(val)
    if isinstance(val, float):
        return repr(val)
    return json.dumps(str(val))


@dataclass
class GlobalDefaultsConfig:
    """Global fallback model settings matching ``[defaults]`` in ``models.toml``."""

    idle_timeout_secs: Optional[int] = 300
    max_batch_size: Optional[int] = 16
    max_batch_delay_ms: Optional[int] = 10
    device: Optional[str] = "auto"
    max_loaded_models: Optional[int] = None
    min_residency_secs: Optional[int] = 10
    autoscale_queue_threshold: Optional[int] = None
    max_replicas: Optional[int] = None
    routing_strategy: Optional[str] = None
    pinned: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serializes non-None global default fields into a dictionary."""
        result: Dict[str, Any] = {}
        for key in (
            "idle_timeout_secs",
            "max_batch_size",
            "max_batch_delay_ms",
            "device",
            "max_loaded_models",
            "min_residency_secs",
            "autoscale_queue_threshold",
            "max_replicas",
            "routing_strategy",
            "pinned",
        ):
            val = getattr(self, key)
            if val is not None:
                result[key] = val
        return result


@dataclass
class ModelConfigEntry:
    """Per-model configuration entry matching ``[[models]]`` in ``models.toml``."""

    id: str
    path: str
    description: Optional[str] = None
    device: Optional[str] = None
    max_batch_size: Optional[int] = None
    max_batch_delay_ms: Optional[int] = None
    idle_timeout_secs: Optional[int] = None
    min_residency_secs: Optional[int] = None
    autoscale_queue_threshold: Optional[int] = None
    max_replicas: Optional[int] = None
    routing_strategy: Optional[str] = None
    pinned: Optional[bool] = None

    def __post_init__(self) -> None:
        if not self.id or not str(self.id).strip():
            raise InvalidInputError("ModelConfigEntry 'id' cannot be empty")
        if not self.path or not str(self.path).strip():
            raise InvalidInputError(f"ModelConfigEntry '{self.id}' 'path' cannot be empty")
        self.id = str(self.id).strip()
        self.path = str(self.path).strip()

    def to_dict(self) -> Dict[str, Any]:
        """Serializes this model entry into a dictionary."""
        result: Dict[str, Any] = {"id": self.id, "path": self.path}
        for key in (
            "description",
            "device",
            "max_batch_size",
            "max_batch_delay_ms",
            "idle_timeout_secs",
            "min_residency_secs",
            "autoscale_queue_threshold",
            "max_replicas",
            "routing_strategy",
            "pinned",
        ):
            val = getattr(self, key)
            if val is not None:
                result[key] = val
        return result


@dataclass
class StreamBinding:
    """Event stream binding configuration matching ``[[streams]]`` in ``models.toml``.

    Attributes:
        name: Unique human-readable identifier for this stream worker binding.
        broker: Broker type (``"kafka"``, ``"rabbitmq"``, ``"redis"``, or ``"mock"``).
        endpoint: Broker connection URL or bootstrap servers (e.g. ``"redis://127.0.0.1:6379"``).
        input_topic: Topic, queue name, or Redis stream key to consume requests from.
        model_id: Optional target model ID (if ``None``, dynamically routes per message ``model_id``).
        output_topic: Optional destination topic/queue/stream for responses (default ``"snooze.responses"``).
        error_topic: Optional Dead-Letter Queue (DLQ) topic for failed messages (default ``"snooze.dlq"``).
        consumer_group: Optional consumer group identifier (default ``"snoozeml-group"``).
        auto_create_topics: Whether to auto-create topics/queues on startup (default ``True``).
        concurrency: Max in-flight inference requests for backpressure control (default ``16``).
        partitions: Partition count when auto-creating topics (default ``1``).
        format: Wire serialization format (``"json"`` or ``"protobuf"``).
    """

    name: str = "default-stream"
    broker: str = "redis"
    endpoint: str = "redis://127.0.0.1:6379"
    input_topic: str = "snooze.requests"
    model_id: Optional[str] = None
    output_topic: Optional[str] = "snooze.responses"
    error_topic: Optional[str] = "snooze.dlq"
    consumer_group: Optional[str] = "snoozeml-group"
    auto_create_topics: bool = True
    concurrency: int = 16
    partitions: int = 1
    format: Union[StreamWireFormat, str] = StreamWireFormat.JSON

    def __post_init__(self) -> None:
        if not self.name or not str(self.name).strip():
            raise InvalidInputError("StreamBinding 'name' cannot be empty")
        self.name = str(self.name).strip()
        self.broker = str(self.broker).strip().lower()
        if self.broker not in ("kafka", "rabbitmq", "redis", "mock"):
            raise InvalidInputError(
                f"Unsupported stream broker '{self.broker}'. Expected 'kafka', 'rabbitmq', 'redis', or 'mock'."
            )
        if not self.endpoint or not str(self.endpoint).strip():
            raise InvalidInputError("StreamBinding 'endpoint' cannot be empty")
        if not self.input_topic or not str(self.input_topic).strip():
            raise InvalidInputError("StreamBinding 'input_topic' cannot be empty")
        if self.concurrency <= 0:
            raise InvalidInputError("StreamBinding 'concurrency' must be positive")
        if self.partitions <= 0:
            raise InvalidInputError("StreamBinding 'partitions' must be positive")
        self.format = StreamWireFormat.from_str(self.format)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes this stream binding into a dictionary."""
        fmt_val = (
            self.format.value
            if isinstance(self.format, StreamWireFormat)
            else str(self.format)
        )
        result: Dict[str, Any] = {
            "name": self.name,
            "broker": self.broker,
            "endpoint": self.endpoint,
            "input_topic": self.input_topic,
            "auto_create_topics": bool(self.auto_create_topics),
            "concurrency": int(self.concurrency),
            "partitions": int(self.partitions),
            "format": fmt_val,
        }
        if self.model_id is not None:
            result["model_id"] = self.model_id
        if self.output_topic is not None:
            result["output_topic"] = self.output_topic
        if self.error_topic is not None:
            result["error_topic"] = self.error_topic
        if self.consumer_group is not None:
            result["consumer_group"] = self.consumer_group
        return result


@dataclass
class ApiKeyEntry:
    """Static API key configuration entry matching ``[[auth.api_keys]]``."""

    role: Union[Role, str] = Role.INFERENCE
    key: Optional[str] = None
    key_env: Optional[str] = None
    name: Optional[str] = None

    def __post_init__(self) -> None:
        self.role = Role.from_str(self.role)
        if not self.key and not self.key_env:
            raise InvalidInputError("ApiKeyEntry requires either 'key' or 'key_env'")

    def to_dict(self) -> Dict[str, Any]:
        """Serializes this API key entry into a dictionary."""
        role_val = self.role.value if isinstance(self.role, Role) else str(self.role)
        result: Dict[str, Any] = {"role": role_val}
        if self.key is not None:
            result["key"] = self.key
        if self.key_env is not None:
            result["key_env"] = self.key_env
        if self.name is not None:
            result["name"] = self.name
        return result


@dataclass
class JwtSettings:
    """JWT authentication configuration matching ``[auth.jwt]``."""

    secret: Optional[str] = None
    secret_env: Optional[str] = None
    algorithm: Union[JwtAlgorithm, str] = JwtAlgorithm.HS256
    issuer: Optional[str] = None
    audience: Optional[str] = None
    leeway_secs: int = 60

    def __post_init__(self) -> None:
        self.algorithm = JwtAlgorithm.from_str(self.algorithm)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes this JWT configuration into a dictionary."""
        algo_val = (
            self.algorithm.value
            if isinstance(self.algorithm, JwtAlgorithm)
            else str(self.algorithm)
        )
        result: Dict[str, Any] = {
            "algorithm": algo_val,
            "leeway_secs": int(self.leeway_secs),
        }
        if self.secret is not None:
            result["secret"] = self.secret
        if self.secret_env is not None:
            result["secret_env"] = self.secret_env
        if self.issuer is not None:
            result["issuer"] = self.issuer
        if self.audience is not None:
            result["audience"] = self.audience
        return result


@dataclass
class AuthSettings:
    """Authentication and RBAC settings matching ``[auth]`` in ``models.toml``."""

    enabled: bool = True
    protect_metrics: bool = False
    api_keys: List[ApiKeyEntry] = field(default_factory=list)
    jwt: Optional[JwtSettings] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serializes this authentication configuration into a dictionary."""
        result: Dict[str, Any] = {
            "enabled": bool(self.enabled),
            "protect_metrics": bool(self.protect_metrics),
            "api_keys": [k.to_dict() for k in self.api_keys],
        }
        if self.jwt is not None:
            result["jwt"] = self.jwt.to_dict()
        return result


@dataclass
class SnoozeConfig:
    """Complete programmatic configuration for a SnoozeML model directory and streaming pipelines.

    Allows Python users to configure models, Kafka/RabbitMQ/Redis streams, and RBAC authentication
    purely in Python and export them to ``models.toml`` or ``models.json`` for the ``snooze`` binary.

    Example::

        from snoozeml import SnoozeConfig

        cfg = (
            SnoozeConfig()
            .add_model("iris-rf", path="iris_rf.onnx", pinned=True, max_batch_size=32)
            .add_stream(
                name="redis-pipeline",
                broker="redis",
                endpoint="redis://127.0.0.1:6379",
                input_topic="snooze.requests",
                output_topic="snooze.responses",
                model_id="iris-rf",
            )
            .add_api_key(role="admin", key_env="SNOOZE_ADMIN_KEY", name="admin-cli")
        )
        cfg.save("models/models.toml")
    """

    defaults: GlobalDefaultsConfig = field(default_factory=GlobalDefaultsConfig)
    models: List[ModelConfigEntry] = field(default_factory=list)
    streams: List[StreamBinding] = field(default_factory=list)
    auth: Optional[AuthSettings] = None

    def add_model(
        self,
        id: str,
        path: str,
        *,
        description: Optional[str] = None,
        device: Optional[str] = None,
        max_batch_size: Optional[int] = None,
        max_batch_delay_ms: Optional[int] = None,
        idle_timeout_secs: Optional[int] = None,
        min_residency_secs: Optional[int] = None,
        autoscale_queue_threshold: Optional[int] = None,
        max_replicas: Optional[int] = None,
        routing_strategy: Optional[str] = None,
        pinned: Optional[bool] = None,
    ) -> "SnoozeConfig":
        """Appends a model configuration entry and returns ``self`` for fluent chaining."""
        self.models.append(
            ModelConfigEntry(
                id=id,
                path=path,
                description=description,
                device=device,
                max_batch_size=max_batch_size,
                max_batch_delay_ms=max_batch_delay_ms,
                idle_timeout_secs=idle_timeout_secs,
                min_residency_secs=min_residency_secs,
                autoscale_queue_threshold=autoscale_queue_threshold,
                max_replicas=max_replicas,
                routing_strategy=routing_strategy,
                pinned=pinned,
            )
        )
        return self

    def add_stream(
        self,
        name: str = "default-stream",
        broker: str = "redis",
        endpoint: str = "redis://127.0.0.1:6379",
        input_topic: str = "snooze.requests",
        *,
        model_id: Optional[str] = None,
        output_topic: Optional[str] = "snooze.responses",
        error_topic: Optional[str] = "snooze.dlq",
        consumer_group: Optional[str] = "snoozeml-group",
        auto_create_topics: bool = True,
        concurrency: int = 16,
        partitions: int = 1,
        format: Union[StreamWireFormat, str] = StreamWireFormat.JSON,
    ) -> "SnoozeConfig":
        """Appends a message stream binding (Kafka, RabbitMQ, Redis, or Mock) and returns ``self``."""
        self.streams.append(
            StreamBinding(
                name=name,
                broker=broker,
                endpoint=endpoint,
                input_topic=input_topic,
                model_id=model_id,
                output_topic=output_topic,
                error_topic=error_topic,
                consumer_group=consumer_group,
                auto_create_topics=auto_create_topics,
                concurrency=concurrency,
                partitions=partitions,
                format=format,
            )
        )
        return self

    def add_api_key(
        self,
        role: Union[Role, str] = Role.INFERENCE,
        *,
        key: Optional[str] = None,
        key_env: Optional[str] = None,
        name: Optional[str] = None,
        protect_metrics: Optional[bool] = None,
    ) -> "SnoozeConfig":
        """Adds an API key entry to ``[auth]`` (enabling authentication) and returns ``self``."""
        if self.auth is None:
            self.auth = AuthSettings(enabled=True)
        if protect_metrics is not None:
            self.auth.protect_metrics = bool(protect_metrics)
        self.auth.api_keys.append(
            ApiKeyEntry(role=role, key=key, key_env=key_env, name=name)
        )
        return self

    def set_jwt(
        self,
        *,
        secret: Optional[str] = None,
        secret_env: Optional[str] = None,
        algorithm: Union[JwtAlgorithm, str] = JwtAlgorithm.HS256,
        issuer: Optional[str] = None,
        audience: Optional[str] = None,
        leeway_secs: int = 60,
        protect_metrics: Optional[bool] = None,
    ) -> "SnoozeConfig":
        """Configures ``[auth.jwt]`` verification settings and returns ``self``."""
        if self.auth is None:
            self.auth = AuthSettings(enabled=True)
        if protect_metrics is not None:
            self.auth.protect_metrics = bool(protect_metrics)
        self.auth.jwt = JwtSettings(
            secret=secret,
            secret_env=secret_env,
            algorithm=algorithm,
            issuer=issuer,
            audience=audience,
            leeway_secs=leeway_secs,
        )
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the entire configuration into a JSON-compatible dictionary."""
        result: Dict[str, Any] = {
            "defaults": self.defaults.to_dict(),
            "models": [m.to_dict() for m in self.models],
            "streams": [s.to_dict() for s in self.streams],
        }
        if self.auth is not None:
            result["auth"] = self.auth.to_dict()
        return result

    def to_json(self, indent: int = 2) -> str:
        """Serializes the configuration into a formatted JSON string (``models.json``)."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_toml(self) -> str:
        """Serializes the configuration into a TOML string compatible with ``src/config.rs``."""
        lines: List[str] = ["# Generated by snoozeml.SnoozeConfig", ""]

        defaults_dict = self.defaults.to_dict()
        if defaults_dict:
            lines.append("[defaults]")
            for k, v in defaults_dict.items():
                lines.append(f"{k} = {_toml_val(v)}")
            lines.append("")

        if self.auth is not None:
            lines.append("[auth]")
            lines.append(f"enabled = {_toml_val(self.auth.enabled)}")
            lines.append(f"protect_metrics = {_toml_val(self.auth.protect_metrics)}")
            lines.append("")

            if self.auth.jwt is not None:
                lines.append("[auth.jwt]")
                for k, v in self.auth.jwt.to_dict().items():
                    lines.append(f"{k} = {_toml_val(v)}")
                lines.append("")

            for api_key in self.auth.api_keys:
                lines.append("[[auth.api_keys]]")
                for k, v in api_key.to_dict().items():
                    lines.append(f"{k} = {_toml_val(v)}")
                lines.append("")

        for model in self.models:
            lines.append("[[models]]")
            for k, v in model.to_dict().items():
                lines.append(f"{k} = {_toml_val(v)}")
            lines.append("")

        for stream in self.streams:
            lines.append("[[streams]]")
            for k, v in stream.to_dict().items():
                lines.append(f"{k} = {_toml_val(v)}")
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    def save(self, path: Union[str, Path]) -> Path:
        """Writes the configuration to disk as TOML (``.toml``) or JSON (``.json``).

        Args:
            path: Target filesystem path (e.g. ``"models/models.toml"`` or ``"models/models.json"``).

        Returns:
            Resolved :class:`pathlib.Path` of the written configuration file.
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix.lower() == ".json":
            out_path.write_text(self.to_json(), encoding="utf-8")
        else:
            out_path.write_text(self.to_toml(), encoding="utf-8")
        logger.info("Saved SnoozeML configuration file", extra={"path": str(out_path)})
        return out_path

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SnoozeConfig":
        """Constructs a :class:`SnoozeConfig` from a parsed dictionary."""
        raw_defaults = data.get("defaults", {})
        defaults = (
            GlobalDefaultsConfig(**raw_defaults)
            if isinstance(raw_defaults, Mapping)
            else GlobalDefaultsConfig()
        )
        models = [
            ModelConfigEntry(**m)
            for m in data.get("models", [])
            if isinstance(m, Mapping)
        ]
        streams = [
            StreamBinding(**s)
            for s in data.get("streams", [])
            if isinstance(s, Mapping)
        ]
        auth: Optional[AuthSettings] = None
        raw_auth = data.get("auth")
        if isinstance(raw_auth, Mapping):
            api_keys = [
                ApiKeyEntry(**k)
                for k in raw_auth.get("api_keys", [])
                if isinstance(k, Mapping)
            ]
            raw_jwt = raw_auth.get("jwt")
            jwt_cfg = JwtSettings(**raw_jwt) if isinstance(raw_jwt, Mapping) else None
            auth = AuthSettings(
                enabled=bool(raw_auth.get("enabled", True)),
                protect_metrics=bool(raw_auth.get("protect_metrics", False)),
                api_keys=api_keys,
                jwt=jwt_cfg,
            )
        return cls(defaults=defaults, models=models, streams=streams, auth=auth)


def scaffold_worker(
    output_dir: Union[str, Path],
    *,
    model_code: Optional[str] = None,
    entrypoint: str = "snooze_model.py",
    class_name: str = "Model",
    tensor_format: str = "numpy",
    python: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
    requirements: Optional[Sequence[str]] = None,
    verify: bool = True,
    verify_inputs: Optional[Sequence[str]] = None,
    verify_device: str = "cpu",
) -> Path:
    """Scaffolds a process-isolated custom Python worker model directory (`snooze_model.py`).

    Creates ``snooze_model.py``, ``snooze_model.json``, and an optional ``requirements.txt``
    inside ``output_dir``, and optionally runs ``snoozeml.worker.run_check`` to verify that
    the worker loads and predicts cleanly.

    Args:
        output_dir: Target model directory (e.g. ``"models/my-custom-worker"``).
        model_code: Python source code defining the worker model class. If ``None``,
            generates a ready-to-customize template class.
        entrypoint: Entrypoint script filename (default ``"snooze_model.py"``).
        class_name: Model class name inside ``entrypoint`` (default ``"Model"``).
        tensor_format: Input tensor format passed to ``predict()`` (``"numpy"`` or ``"raw"``).
        python: Optional custom Python interpreter path written to ``snooze_model.json``.
        env: Optional environment variables mapping written to ``snooze_model.json``.
        requirements: Optional list of pip requirement specifiers written to ``requirements.txt``.
        verify: If ``True``, smoke-tests the generated worker immediately via :func:`~snoozeml.worker.run_check`.
        verify_inputs: Input specification strings for ``run_check`` (default ``["input:2x4"]``).
        verify_device: Device string passed to ``run_check`` (default ``"cpu"``).

    Returns:
        Resolved :class:`pathlib.Path` to the scaffolded worker directory.
    """
    if tensor_format not in ("numpy", "raw"):
        raise InvalidInputError("tensor_format must be 'numpy' or 'raw'")

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    if model_code is None:
        if tensor_format == "raw":
            default_code = f"""\
            from typing import Any, Dict
            from snoozeml.worker import RawTensor


            class {class_name}:
                def load(self, model_dir: str, device: str) -> None:
                    self.device = device

                def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
                    first = next(iter(inputs.values()))
                    return {{"output": RawTensor(first.shape, [x * 2.0 for x in first.data])}}
            """
        else:
            default_code = f"""\
            from typing import Any, Dict
            import numpy as np


            class {class_name}:
                def load(self, model_dir: str, device: str) -> None:
                    # Import heavy frameworks (e.g. torch, transformers) inside load()
                    self.device = device

                def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
                    first = next(iter(inputs.values()))
                    arr = np.ascontiguousarray(first, dtype=np.float32)
                    return {{"output": arr}}
            """
        script_content = textwrap.dedent(default_code)
    else:
        script_content = textwrap.dedent(model_code)

    script_path = target_dir / entrypoint
    script_path.write_text(script_content, encoding="utf-8")

    manifest: Dict[str, Any] = {
        "entrypoint": entrypoint,
        "class": class_name,
        "tensor_format": tensor_format,
    }
    if python is not None:
        manifest["python"] = python
    if env is not None:
        manifest["env"] = dict(env)

    manifest_path = target_dir / "snooze_model.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    req_path = target_dir / "requirements.txt"
    if requirements:
        req_path.write_text("\n".join(requirements).rstrip() + "\n", encoding="utf-8")
    elif not req_path.exists():
        req_path.write_text("# Add worker Python dependencies here\n", encoding="utf-8")

    if verify:
        import argparse

        specs = list(verify_inputs) if verify_inputs is not None else ["input:2x4"]
        ns = argparse.Namespace(
            check=str(target_dir),
            input=specs,
            device=verify_device,
            entrypoint=entrypoint,
            class_name=class_name,
            tensor_format=tensor_format,
        )
        rc = run_check(ns)
        if rc != 0:
            raise SnoozeError(f"Worker smoke check failed for '{target_dir}' with exit code {rc}")

    return target_dir


def find_snooze_binary(explicit_path: Optional[Union[str, Path]] = None) -> str:
    """Locates the ``snooze`` daemon executable on the system.

    Checks ``explicit_path``, ``SNOOZE_BIN`` environment variable, system ``PATH``,
    and local Cargo build directories (``target/release/snooze``, ``target/debug/snooze``).
    """
    if explicit_path is not None:
        p = Path(explicit_path)
        if p.is_file():
            return str(p.resolve())
        raise SnoozeError(f"Explicit snooze binary path does not exist: {explicit_path}")

    env_bin = os.getenv("SNOOZE_BIN")
    if env_bin and Path(env_bin).is_file():
        return str(Path(env_bin).resolve())

    which_bin = shutil.which("snooze")
    if which_bin:
        return which_bin

    # Check common workspace locations relative to cwd
    exe_name = "snooze.exe" if os.name == "nt" else "snooze"
    for candidate in (
        Path.cwd() / "target" / "release" / exe_name,
        Path.cwd() / "target" / "debug" / exe_name,
    ):
        if candidate.is_file():
            return str(candidate.resolve())

    raise SnoozeError(
        "Could not locate the 'snooze' binary. Add it to PATH, set SNOOZE_BIN, "
        "or pass binary_path=... to SnoozeDaemon."
    )


class SnoozeDaemon:
    """Context-managed local ``snooze serve`` process controller for Python workflows.

    Allows launching a configured SnoozeML daemon (including ``[[streams]]``, ``[[models]]``,
    and ``[auth]``) directly from Python scripts, tests, or Jupyter notebooks without modifying
    the Rust binary.
    """

    def __init__(
        self,
        *,
        config: Optional[SnoozeConfig] = None,
        models_dir: Optional[Union[str, Path]] = None,
        host: str = "127.0.0.1",
        port: int = 8080,
        grpc_port: int = 50051,
        grpc: bool = True,
        demo: bool = False,
        api_key: Optional[str] = None,
        jwt_secret: Optional[str] = None,
        binary_path: Optional[Union[str, Path]] = None,
        startup_timeout: float = 15.0,
    ) -> None:
        self.config = config
        self.models_dir = Path(models_dir) if models_dir is not None else None
        self.host = host
        self.port = int(port)
        self.grpc_port = int(grpc_port)
        self.grpc = bool(grpc)
        self.demo = bool(demo)
        self.api_key = api_key
        self.jwt_secret = jwt_secret
        self.binary_path = binary_path
        self.startup_timeout = float(startup_timeout)

        self._temp_dir: Optional[tempfile.TemporaryDirectory[str]] = None
        self._process: Optional[subprocess.Popen[bytes]] = None

    @property
    def http_endpoint(self) -> str:
        """Returns the HTTP base URL for this daemon instance."""
        return f"http://{self.host}:{self.port}"

    @property
    def grpc_endpoint(self) -> str:
        """Returns the gRPC ``host:port`` target for this daemon instance."""
        return f"{self.host}:{self.grpc_port}"

    def build_command(self, resolved_models_dir: Path, config_path: Optional[Path]) -> List[str]:
        """Constructs the ``snooze serve`` CLI command list."""
        binary = find_snooze_binary(self.binary_path)
        cmd = [
            binary,
            "serve",
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--models-dir",
            str(resolved_models_dir),
            "--grpc-port",
            str(self.grpc_port),
        ]
        if not self.grpc:
            cmd.append("--no-grpc")
        if self.demo:
            cmd.append("--demo")
        if config_path is not None:
            cmd.extend(["--config", str(config_path)])
        if self.api_key:
            cmd.extend(["--api-key", f"admin:{self.api_key}"])
        if self.jwt_secret:
            cmd.extend(["--jwt-secret", self.jwt_secret])
        return cmd

    def start(self) -> "SnoozeDaemon":
        """Starts the ``snooze serve`` child process and waits until ``/health/ready`` responds."""
        if self._process is not None and self._process.poll() is None:
            return self

        if self.models_dir is not None:
            resolved_dir = self.models_dir
            resolved_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._temp_dir = tempfile.TemporaryDirectory(prefix="snoozeml_daemon_")
            resolved_dir = Path(self._temp_dir.name)

        config_path: Optional[Path] = None
        if self.config is not None:
            config_path = self.config.save(resolved_dir / "models.toml")

        cmd = self.build_command(resolved_dir, config_path)
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        client = self.client()
        try:
            deadline = time.monotonic() + self.startup_timeout
            while time.monotonic() < deadline:
                if self._process.poll() is not None:
                    stderr_bytes = (
                        self._process.stderr.read() if self._process.stderr else b""
                    )
                    stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()
                    raise SnoozeConnectionError(
                        f"snooze daemon exited prematurely with code {self._process.returncode}: {stderr_text}"
                    )
                try:
                    if client.health_ready():
                        return self
                except Exception:
                    pass
                time.sleep(0.1)
            self.stop()
            raise SnoozeConnectionError(
                f"Timed out waiting {self.startup_timeout}s for snooze daemon at {self.http_endpoint}"
            )
        finally:
            client.close()

    def stop(self) -> None:
        """Stops the ``snooze serve`` child process and cleans up temporary directories."""
        if self._process is not None:
            if self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=2.0)
            self._process = None

        if self._temp_dir is not None:
            try:
                self._temp_dir.cleanup()
            except Exception:
                pass
            self._temp_dir = None

    def client(self, *, use_grpc: bool = False) -> "SnoozeClient":
        """Returns a configured :class:`~snoozeml.SnoozeClient` connected to this daemon."""
        from .client import SnoozeClient

        return SnoozeClient(
            http_endpoint=self.http_endpoint,
            grpc_endpoint=self.grpc_endpoint,
            use_grpc=use_grpc,
            api_key=self.api_key,
        )

    def async_client(self, *, use_grpc: bool = False) -> "AsyncSnoozeClient":
        """Returns a configured :class:`~snoozeml.AsyncSnoozeClient` connected to this daemon."""
        from .client import AsyncSnoozeClient

        return AsyncSnoozeClient(
            http_endpoint=self.http_endpoint,
            grpc_endpoint=self.grpc_endpoint,
            use_grpc=use_grpc,
            api_key=self.api_key,
        )

    def __enter__(self) -> "SnoozeDaemon":
        return self.start()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()
