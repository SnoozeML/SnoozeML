"""Unit tests for snoozeml.models (ModelState and ModelInfo)."""

from __future__ import annotations

from snoozeml.models import ModelInfo, ModelState


def test_model_state_parsing() -> None:
    assert ModelState.from_str("idle") == ModelState.IDLE
    assert ModelState.from_str("LOADING") == ModelState.LOADING
    assert ModelState.from_str(" ready ") == ModelState.READY
    assert ModelState.from_str("busy") == ModelState.BUSY
    assert ModelState.from_str("unknown_state") == ModelState.IDLE


def test_model_info_serialization_and_properties() -> None:
    raw = {
        "id": "bert-base-uncased",
        "state": "ready",
        "onnx_path": "models/bert.onnx",
        "idle_timeout_secs": 120,
        "max_batch_size": 32,
        "max_batch_delay_ms": 5,
        "last_accessed_secs_ago": 3,
        "description": "BERT embedding model",
        "pinned": True,
        "memory_bytes": 440_000_000,
    }
    info = ModelInfo.from_dict(raw)
    assert info.id == "bert-base-uncased"
    assert info.state == ModelState.READY
    assert info.is_loaded is True
    assert info.path == "models/bert.onnx"
    assert info.pinned is True
    assert info.memory_bytes == 440_000_000

    serialized = info.to_dict()
    assert serialized["id"] == "bert-base-uncased"
    assert serialized["state"] == "ready"
    assert serialized["pinned"] is True
    assert serialized["description"] == "BERT embedding model"

    idle_info = ModelInfo.from_dict({"id": "m2", "state": "idle", "onnx_path": "mock:test"})
    assert idle_info.is_loaded is False
