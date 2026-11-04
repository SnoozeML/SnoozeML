"""Unit tests for snoozeml.models (ModelState and ModelInfo)."""

from __future__ import annotations

import pytest

from snoozeml.models import ModelInfo, ModelState, ModelStats, StatDuration


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


def test_stat_duration_and_model_stats_properties() -> None:
    raw_stats = {
        "name": "demo-model",
        "version": "1",
        "last_inference": 1730000000123,
        "inference_count": 42,
        "execution_count": 10,
        "inference_stats": {
            "success": {"count": 40, "ns": 80_000_000},
            "fail": {"count": 2, "ns": 4_000_000},
            "queue": {"count": 42, "ns": 21_000_000},
            "compute_infer": {"count": 42, "ns": 60_000_000},
        },
    }

    stats = ModelStats.from_dict(raw_stats)
    assert stats.name == "demo-model"
    assert stats.version == "1"
    assert stats.last_inference_ms == 1730000000123
    assert stats.inference_count == 42
    assert stats.execution_count == 10

    # StatDuration properties
    assert stats.success.count == 40
    assert stats.success.ns == 80_000_000
    assert stats.success.ms == pytest.approx(80.0)
    assert stats.success.secs == pytest.approx(0.08)
    assert stats.success.avg_ms == pytest.approx(2.0)

    assert stats.fail.count == 2
    assert stats.fail.ns == 4_000_000
    assert stats.fail.avg_ms == pytest.approx(2.0)

    assert stats.queue.count == 42
    assert stats.queue.ms == pytest.approx(21.0)
    assert stats.compute_infer.ms == pytest.approx(60.0)

    # Aggregated properties
    assert stats.total_requests == 42
    assert stats.error_rate == pytest.approx(2 / 42)

    # Serialization roundtrip
    serialized = stats.to_dict()
    assert serialized["name"] == "demo-model"
    assert serialized["inference_stats"]["success"]["count"] == 40
    assert serialized["inference_stats"]["fail"]["count"] == 2

    # Empty stats defaults
    empty_stats = ModelStats.from_dict({})
    assert empty_stats.name == ""
    assert empty_stats.total_requests == 0
    assert empty_stats.error_rate == 0.0
    assert empty_stats.success.avg_ms == 0.0
