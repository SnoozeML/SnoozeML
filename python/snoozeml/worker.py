"""SnoozeML process-isolated model worker (protocol v1).

The ``snooze`` daemon embeds this file and runs it with ``python -u -c`` for every
replica of a *worker model*: a directory containing ``snooze_model.py``. The worker
imports the user's model class, calls ``load(model_dir, device)`` once, then answers
batched ``predict(inputs)`` calls framed over stdin/stdout until it gets ``SHUTDOWN``
or stdin reaches EOF.

It depends only on the standard library. ``numpy`` is needed only for the default
``tensor_format="numpy"``; ``tensor_format="raw"`` passes :class:`RawTensor` objects instead.

Local smoke test (no daemon required)::

    python -m snoozeml.worker --check models/my-model --input input:4x3

Wire frame::

    [u32 LE frame_len][u8 msg_type][u32 LE header_len][header JSON][raw LE f32 bytes]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import struct
import sys
import traceback
from array import array
from typing import Any, Dict, List, Optional, Protocol, Tuple

PROTOCOL_VERSION = 1
MSG_READY, MSG_INFER, MSG_RESULT, MSG_ERROR, MSG_SHUTDOWN = 1, 2, 3, 4, 5
MAX_FRAME_BYTES = 1 << 30
_U32 = struct.Struct("<I")
_FRAME_PREFIX = struct.Struct("<IBI")
_LITTLE_ENDIAN = sys.byteorder == "little"


class SnoozeModel(Protocol):
    """Interface a user model class must provide (duck-typed; subclassing is optional)."""

    def load(self, model_dir: str, device: str) -> None:
        """Load weights from ``model_dir`` onto ``device`` (``cpu``, ``cuda``, ``cuda:N``, ``mps``)."""

    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Map input tensors (batch on axis 0) to output tensors with the same batch size."""


class RawTensor:
    """Dependency-free float32 tensor used when ``tensor_format="raw"``.

    ``data`` is a flat row-major ``array('f')`` whose length equals the product of ``shape``.
    """

    __slots__ = ("shape", "data")

    def __init__(self, shape, data) -> None:
        self.shape = tuple(int(d) for d in shape)
        self.data = data if isinstance(data, array) and data.typecode == "f" else array("f", data)
        numel = 1
        for dim in self.shape:
            numel *= dim
        if numel != len(self.data):
            raise ValueError(f"RawTensor shape {self.shape} expects {numel} elements, got {len(self.data)}")

    def __repr__(self) -> str:
        return f"RawTensor(shape={self.shape})"


class WorkerError(Exception):
    """Error carrying an explicit protocol ``kind`` (``oom``, ``invalid_input``, ``load``, ``runtime``)."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


# --------------------------------------------------------------------------- framing


def _read_exact(stream, size: int) -> Optional[bytearray]:
    """Read exactly ``size`` bytes. Returns ``None`` on clean EOF before the first byte."""
    buf = bytearray(size)
    view = memoryview(buf)
    got = 0
    while got < size:
        n = stream.readinto(view[got:])
        if not n:
            if got == 0:
                return None
            raise WorkerError("runtime", f"unexpected EOF: read {got} of {size} bytes")
        got += n
    return buf


def read_frame(stream) -> Optional[Tuple[int, Dict[str, Any], memoryview]]:
    """Read one frame as ``(msg_type, header, payload)``, or ``None`` on EOF."""
    raw_len = _read_exact(stream, 4)
    if raw_len is None:
        return None
    (frame_len,) = _U32.unpack(raw_len)
    if frame_len < 5 or frame_len > MAX_FRAME_BYTES:
        raise WorkerError("runtime", f"invalid frame length {frame_len}")
    body = _read_exact(stream, frame_len)
    if body is None:
        raise WorkerError("runtime", "unexpected EOF inside frame")
    msg_type = body[0]
    (header_len,) = _U32.unpack_from(body, 1)
    if 5 + header_len > frame_len:
        raise WorkerError("runtime", f"header length {header_len} exceeds frame length {frame_len}")
    header = json.loads(body[5 : 5 + header_len]) if header_len else {}
    return msg_type, header, memoryview(body)[5 + header_len :]


def write_frame(stream, msg_type: int, header: Dict[str, Any], chunks: Optional[List[Any]] = None) -> None:
    """Write one frame made of a JSON header and raw byte ``chunks``."""
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    views = [memoryview(c).cast("B") for c in (chunks or [])]
    frame_len = 1 + 4 + len(header_bytes) + sum(v.nbytes for v in views)
    if frame_len > MAX_FRAME_BYTES:
        raise WorkerError("runtime", f"output frame of {frame_len} bytes exceeds limit {MAX_FRAME_BYTES}")
    stream.write(_FRAME_PREFIX.pack(frame_len, msg_type, len(header_bytes)) + header_bytes)
    for view in views:
        stream.write(view)
    stream.flush()


# --------------------------------------------------------------------------- tensors


def decode_tensors(header: Dict[str, Any], payload: memoryview, tensor_format: str) -> Dict[str, Any]:
    """Decode the tensors of an ``INFER`` frame into ``numpy`` arrays or :class:`RawTensor`."""
    np = _numpy() if tensor_format == "numpy" else None
    tensors: Dict[str, Any] = {}
    offset = 0
    for meta in header.get("tensors", []):
        name, shape = str(meta["name"]), [int(d) for d in meta["shape"]]
        if meta.get("dtype", "f32") != "f32":
            raise WorkerError("invalid_input", f"tensor '{name}': unsupported dtype {meta.get('dtype')}")
        numel = 1
        for dim in shape:
            numel *= dim
        end = offset + numel * 4
        if end > len(payload):
            raise WorkerError("invalid_input", f"tensor '{name}' exceeds payload size")
        if np is not None:
            tensors[name] = np.frombuffer(payload, dtype="<f4", count=numel, offset=offset).reshape(shape)
        else:
            data = array("f")
            data.frombytes(payload[offset:end])
            if not _LITTLE_ENDIAN:
                data.byteswap()
            tensors[name] = RawTensor(shape, data)
        offset = end
    if offset != len(payload):
        raise WorkerError("invalid_input", f"payload has {len(payload) - offset} trailing bytes")
    return tensors


def encode_tensors(outputs: Any, batch_size: Optional[int]) -> Tuple[Dict[str, Any], List[Any]]:
    """Validate ``predict`` outputs and encode them as ``(header, chunks)``."""
    if not isinstance(outputs, dict) or not outputs:
        raise WorkerError("runtime", f"predict() must return a non-empty dict of tensors, got {type(outputs).__name__}")
    metas, chunks = [], []
    for name, value in outputs.items():
        if not isinstance(name, str):
            raise WorkerError("runtime", f"output names must be str, got {type(name).__name__}")
        if isinstance(value, RawTensor):
            shape, data = list(value.shape), value.data
            if not _LITTLE_ENDIAN:
                data = array("f", data)
                data.byteswap()
            chunk = data
        else:
            try:
                if hasattr(value, "detach") and hasattr(value, "cpu"):
                    value = value.detach().cpu().numpy()
                np = _numpy()
                arr = np.ascontiguousarray(value, dtype="<f4")
            except WorkerError:
                raise
            except Exception as exc:  # e.g. CUDA tensors that were not moved with .cpu()
                raise WorkerError("runtime", f"output '{name}' is not convertible to float32 ({exc}); "
                                  "return numpy arrays, CPU tensors or RawTensor") from exc
            shape, chunk = list(arr.shape), arr
        if batch_size is not None and (not shape or shape[0] != batch_size):
            raise WorkerError("runtime", f"output '{name}' has shape {shape}; axis 0 must equal the "
                              f"input batch size {batch_size}")
        metas.append({"name": name, "shape": shape, "dtype": "f32"})
        chunks.append(chunk)
    return {"tensors": metas}, chunks


def _numpy():
    try:
        import numpy  # noqa: PLC0415 - optional dependency
    except ImportError as exc:
        raise WorkerError("load", "numpy is required for tensor_format='numpy' (pip install numpy), "
                          "or set \"tensor_format\": \"raw\" in snooze_model.json") from exc
    return numpy


# --------------------------------------------------------------------------- model lifecycle


def classify_error(exc: BaseException) -> str:
    """Map an exception raised by user code to a protocol error kind."""
    if isinstance(exc, WorkerError):
        return exc.kind
    if isinstance(exc, MemoryError) or type(exc).__name__ == "OutOfMemoryError":
        return "oom"
    if "out of memory" in str(exc).lower():
        return "oom"
    if isinstance(exc, (ValueError, KeyError)):
        return "invalid_input"
    return "runtime"


def resolve_device(device: str) -> str:
    """Resolve ``auto`` using torch if the user module imported it, else return ``device`` unchanged."""
    if device != "auto":
        return device
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.cuda.is_available():
                return "cuda"
            mps = getattr(torch.backends, "mps", None)
            if mps is not None and mps.is_available():
                return "mps"
        except Exception:  # noqa: BLE001 - device probing must never break loading
            pass
    return "cpu"


def load_user_model(model_dir: str, entrypoint: str, class_name: str, device: str):
    """Import ``entrypoint`` from ``model_dir``, instantiate ``class_name`` and call ``load``."""
    model_dir = os.path.abspath(model_dir)
    path = os.path.join(model_dir, entrypoint)
    if not os.path.isfile(path):
        raise WorkerError("load", f"entrypoint '{path}' not found")
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
    spec = importlib.util.spec_from_file_location("snooze_user_model", path)
    if spec is None or spec.loader is None:
        raise WorkerError("load", f"cannot import '{path}'")
    module = importlib.util.module_from_spec(spec)
    sys.modules["snooze_user_model"] = module
    spec.loader.exec_module(module)
    cls = getattr(module, class_name, None)
    if cls is None:
        raise WorkerError("load", f"class '{class_name}' not found in '{entrypoint}'")
    model = cls()
    for method in ("load", "predict"):
        if not callable(getattr(model, method, None)):
            raise WorkerError("load", f"class '{class_name}' must define a '{method}' method")
    resolved = resolve_device(device)
    model.load(model_dir, resolved)
    return model, resolved


def handle_infer(model, header: Dict[str, Any], payload: memoryview, tensor_format: str):
    """Run one batched ``predict`` call and return ``(header, chunks)`` for a ``RESULT`` frame."""
    inputs = decode_tensors(header, payload, tensor_format)
    batch_sizes = {int(m["shape"][0]) for m in header.get("tensors", []) if m.get("shape")}
    batch_size = batch_sizes.pop() if len(batch_sizes) == 1 else None
    return encode_tensors(model.predict(inputs), batch_size)


def serve(model, tensor_format: str, rx, tx) -> int:
    """Answer ``INFER`` frames until ``SHUTDOWN`` or EOF. Returns the process exit code."""
    while True:
        frame = read_frame(rx)
        if frame is None:
            return 0
        msg_type, header, payload = frame
        if msg_type == MSG_SHUTDOWN:
            return 0
        if msg_type != MSG_INFER:
            write_frame(tx, MSG_ERROR, {"kind": "runtime", "message": f"unexpected message type {msg_type}"})
            continue
        try:
            out_header, chunks = handle_infer(model, header, payload, tensor_format)
        except Exception as exc:  # noqa: BLE001 - every user failure becomes an ERROR frame
            traceback.print_exc(file=sys.stderr)
            write_frame(tx, MSG_ERROR, {"kind": classify_error(exc), "message": f"{type(exc).__name__}: {exc}"})
            continue
        write_frame(tx, MSG_RESULT, out_header, chunks)


def _isolate_stdio():
    """Take private copies of stdin/stdout for the protocol and point fds 0/1 elsewhere.

    User ``print()`` calls and C extensions writing to fd 1 then go to stderr instead
    of corrupting the binary stream, and nothing in user code can read protocol bytes.
    """
    rx = os.fdopen(os.dup(0), "rb", buffering=1 << 16)
    tx = os.fdopen(os.dup(1), "wb", buffering=1 << 16)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.close(devnull)
    os.dup2(2, 1)
    sys.stdin = open(os.devnull, "r")  # noqa: SIM115 - lives for the whole process
    sys.stdout = sys.stderr
    return rx, tx


def run_worker(args) -> int:
    """Entry point used by the daemon."""
    entrypoint = args.entrypoint or "snooze_model.py"
    class_name = args.class_name or "Model"
    tensor_format = args.tensor_format or "numpy"
    rx, tx = _isolate_stdio()
    try:
        if tensor_format == "numpy":
            _numpy()
        model, device = load_user_model(args.model_dir, entrypoint, class_name, args.device)
    except BaseException as exc:  # noqa: BLE001 - report load failures, including SystemExit
        traceback.print_exc(file=sys.stderr)
        kind = classify_error(exc)
        write_frame(tx, MSG_ERROR, {"kind": "oom" if kind == "oom" else "load",
                                    "message": f"{type(exc).__name__}: {exc}"})
        return 1
    print(f"snooze-worker: model ready (device={device}, pid={os.getpid()})", file=sys.stderr, flush=True)
    write_frame(tx, MSG_READY, {"protocol": PROTOCOL_VERSION,
                                "info": {"python": sys.version.split()[0], "device": device}})
    return serve(model, tensor_format, rx, tx)


def run_check(args) -> int:
    """Load the model and run a single ``predict`` on deterministic inputs, printing a summary."""
    model_dir = args.check
    manifest_path = os.path.join(model_dir, "snooze_model.json")
    manifest: Dict[str, Any] = {}
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception as exc:
            print(f"Warning: failed to read manifest {manifest_path}: {exc}", file=sys.stderr)

    entrypoint = getattr(args, "entrypoint", None) or manifest.get("entrypoint") or "snooze_model.py"
    class_name = getattr(args, "class_name", None) or manifest.get("class") or "Model"
    tensor_format = getattr(args, "tensor_format", None) or manifest.get("tensor_format") or "numpy"

    model, device = load_user_model(model_dir, entrypoint, class_name, args.device)
    metas, chunks = [], []
    for spec in args.input or ["input:1x1"]:
        name, _, dims = spec.partition(":")
        shape = [int(d) for d in dims.lower().split("x") if d]
        numel = 1
        for dim in shape:
            numel *= dim
        data = array("f", (float(i % 7) / 7.0 for i in range(numel)))
        if not _LITTLE_ENDIAN:
            data.byteswap()
        metas.append({"name": name, "shape": shape, "dtype": "f32"})
        chunks.append(data.tobytes())
    payload = memoryview(bytearray(b"".join(chunks)))
    header, out_chunks = handle_infer(model, {"tensors": metas}, payload, tensor_format)
    print(f"OK: model loaded on '{device}'")
    for meta, chunk in zip(header["tensors"], out_chunks):
        values = memoryview(chunk).cast("B").cast("f").tolist()[:6]
        print(f"  output '{meta['name']}': shape={meta['shape']} first={[round(v, 4) for v in values]}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Parse arguments and run either the daemon worker loop or ``--check``."""
    parser = argparse.ArgumentParser(prog="snoozeml.worker", description="SnoozeML model worker")
    parser.add_argument("--model-dir", help="model directory (daemon mode)")
    parser.add_argument("--check", metavar="MODEL_DIR", help="smoke-test a model directory and exit")
    parser.add_argument("--input", action="append", metavar="NAME:DxD", help="input spec for --check")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--entrypoint", default=None)
    parser.add_argument("--class", dest="class_name", default=None)
    parser.add_argument("--tensor-format", choices=("numpy", "raw"), default=None)
    args = parser.parse_args(argv)
    if args.check:
        return run_check(args)
    if not args.model_dir:
        parser.error("--model-dir or --check is required")
    return run_worker(args)


if __name__ == "__main__":
    sys.exit(main())
