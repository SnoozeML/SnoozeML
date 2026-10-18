"""Tests for snoozeml.worker protocol framing, tensor encoding, and model lifecycle."""

import io
import os
import sys
import tempfile
from array import array
import numpy as np
import pytest

from snoozeml.worker import (
    MSG_ERROR,
    MSG_INFER,
    MSG_READY,
    MSG_RESULT,
    MSG_SHUTDOWN,
    RawTensor,
    WorkerError,
    _isolate_stdio,
    _read_exact,
    classify_error,
    decode_tensors,
    encode_tensors,
    handle_infer,
    load_user_model,
    read_frame,
    run_check,
    write_frame,
)


def test_raw_tensor():
    t = RawTensor([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert t.shape == (2, 3)
    assert len(t.data) == 6
    assert "RawTensor(shape=(2, 3))" in repr(t)

    with pytest.raises(ValueError, match="expects 6 elements"):
        RawTensor([2, 3], [1.0, 2.0])


def test_frame_roundtrip():
    stream = io.BytesIO()
    header = {"protocol": 1, "test": "val"}
    payload_chunk = b"\x00\x00\x80?\x00\x00\x00@"  # 1.0, 2.0 as LE f32

    write_frame(stream, MSG_READY, header, [payload_chunk])
    stream.seek(0)

    msg_type, read_hdr, read_payload = read_frame(stream)
    assert msg_type == MSG_READY
    assert read_hdr == header
    assert bytes(read_payload) == payload_chunk


def test_frame_eof():
    stream = io.BytesIO(b"")
    assert read_frame(stream) is None


def test_read_exact_partial_eof():
    stream = io.BytesIO(b"abc")
    with pytest.raises(WorkerError, match="unexpected EOF"):
        _read_exact(stream, 10)


def test_classify_error():
    assert classify_error(MemoryError("out of host memory")) == "oom"
    assert classify_error(RuntimeError("CUDA error: out of memory")) == "oom"
    assert classify_error(ValueError("bad shape")) == "invalid_input"
    assert classify_error(KeyError("missing_tensor")) == "invalid_input"
    assert classify_error(WorkerError("custom_kind", "test")) == "custom_kind"
    assert classify_error(ZeroDivisionError("division by zero")) == "runtime"


def test_tensor_encode_decode_numpy():
    arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="<f4")
    header, chunks = encode_tensors({"x": arr}, batch_size=2)
    assert header["tensors"][0]["name"] == "x"
    assert header["tensors"][0]["shape"] == [2, 3]

    payload = memoryview(b"".join(bytes(memoryview(c).cast("B")) for c in chunks))
    decoded = decode_tensors(header, payload, tensor_format="numpy")
    np.testing.assert_allclose(decoded["x"], arr)


def test_tensor_encode_decode_raw():
    raw = RawTensor([2, 2], [1.0, 2.0, 3.0, 4.0])
    header, chunks = encode_tensors({"r": raw}, batch_size=2)
    payload = memoryview(b"".join(bytes(memoryview(c).cast("B")) for c in chunks))
    decoded = decode_tensors(header, payload, tensor_format="raw")
    assert isinstance(decoded["r"], RawTensor)
    assert decoded["r"].shape == (2, 2)
    assert list(decoded["r"].data) == [1.0, 2.0, 3.0, 4.0]


def test_encode_tensors_batch_mismatch():
    arr = np.array([[1.0, 2.0]], dtype="<f4")
    with pytest.raises(WorkerError, match="axis 0 must equal the input batch size"):
        encode_tensors({"x": arr}, batch_size=4)


def test_load_and_infer_lifecycle():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_py = os.path.join(tmpdir, "snooze_model.py")
        with open(model_py, "w", encoding="utf-8") as f:
            f.write(
                "import numpy as np\n"
                "class Model:\n"
                "    def load(self, model_dir, device):\n"
                "        self.device = device\n"
                "    def predict(self, inputs):\n"
                "        return {'out': inputs['in'] * 2.0}\n"
            )

        model, device = load_user_model(tmpdir, "snooze_model.py", "Model", "cpu")
        assert device == "cpu"

        input_arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype="<f4")
        header, chunks = encode_tensors({"in": input_arr}, batch_size=2)
        payload = memoryview(b"".join(bytes(memoryview(c).cast("B")) for c in chunks))

        out_header, out_chunks = handle_infer(model, header, payload, "numpy")
        out_payload = memoryview(b"".join(bytes(memoryview(c).cast("B")) for c in out_chunks))
        outputs = decode_tensors(out_header, out_payload, "numpy")

        np.testing.assert_allclose(outputs["out"], input_arr * 2.0)


def test_run_check_cli():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_py = os.path.join(tmpdir, "snooze_model.py")
        with open(model_py, "w", encoding="utf-8") as f:
            f.write(
                "class Model:\n"
                "    def load(self, model_dir, device): pass\n"
                "    def predict(self, inputs):\n"
                "        return {'out': inputs['input']}\n"
            )

        class Args:
            check = tmpdir
            entrypoint = "snooze_model.py"
            class_name = "Model"
            device = "cpu"
            input = ["input:2x2"]
            tensor_format = "numpy"

        code = run_check(Args())
        assert code == 0


def test_encode_tensors_fake_torch_tensor():
    class FakeTorchTensor:
        def __init__(self, data):
            self._data = data

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.array(self._data, dtype="<f4")

    fake = FakeTorchTensor([[1.0, 2.0], [3.0, 4.0]])
    header, chunks = encode_tensors({"t": fake}, batch_size=2)
    assert header["tensors"][0]["name"] == "t"
    assert header["tensors"][0]["shape"] == [2, 2]
    payload = memoryview(b"".join(bytes(memoryview(c).cast("B")) for c in chunks))
    decoded = decode_tensors(header, payload, tensor_format="numpy")
    np.testing.assert_allclose(decoded["t"], [[1.0, 2.0], [3.0, 4.0]])


def test_run_check_honors_snooze_model_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        custom_py = os.path.join(tmpdir, "custom_entry.py")
        with open(custom_py, "w", encoding="utf-8") as f:
            f.write(
                "from snoozeml.worker import RawTensor\n"
                "class CustomModel:\n"
                "    def load(self, model_dir, device): pass\n"
                "    def predict(self, inputs):\n"
                "        inp = inputs['input']\n"
                "        return {'out': RawTensor(inp.shape, inp.data)}\n"
            )
        manifest_json = os.path.join(tmpdir, "snooze_model.json")
        with open(manifest_json, "w", encoding="utf-8") as f:
            f.write(
                '{\n'
                '  "entrypoint": "custom_entry.py",\n'
                '  "class": "CustomModel",\n'
                '  "tensor_format": "raw"\n'
                '}\n'
            )

        class Args:
            check = tmpdir
            entrypoint = None
            class_name = None
            device = "cpu"
            input = ["input:2x2"]
            tensor_format = None

        code = run_check(Args())
        assert code == 0

