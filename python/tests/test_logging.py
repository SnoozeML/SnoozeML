"""Unit tests for snoozeml.logging and credential sanitization."""

from __future__ import annotations

import io
import json
import logging

from snoozeml import AsyncSnoozeClient, SnoozeClient, configure_logging, get_logger, mask_secret
from snoozeml.logging import sanitize_mapping, sanitize_text


def test_mask_secret() -> None:
    assert mask_secret(None) == "None"
    assert mask_secret("sk-super-secret-key-123") == "***REDACTED***"


def test_sanitize_text_redacts_bearer_and_api_keys() -> None:
    raw = "Header Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload.sig and api_key='sk-live-999'"
    cleaned = sanitize_text(raw)
    assert "eyJhbGciOiJIUzI1NiJ9" not in cleaned
    assert "sk-live-999" not in cleaned
    assert "***REDACTED***" in cleaned


def test_sanitize_mapping_redacts_sensitive_keys() -> None:
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": "sk-secret-key",
        "Authorization": "Bearer jwt-token-value",
        "nested": {"token": "my-secret-token", "model_id": "demo-model"},
    }
    sanitized = sanitize_mapping(headers)
    assert sanitized["Content-Type"] == "application/json"
    assert sanitized["X-API-Key"] == "***REDACTED***"
    assert sanitized["Authorization"] == "***REDACTED***"
    assert sanitized["nested"]["token"] == "***REDACTED***"
    assert sanitized["nested"]["model_id"] == "demo-model"


def test_client_repr_never_exposes_credentials() -> None:
    client = SnoozeClient(
        endpoint="http://localhost:8080",
        api_key="sk-ultra-secret-admin-key",
        token="eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.secret",
    )
    repr_str = repr(client)
    assert "sk-ultra-secret-admin-key" not in repr_str
    assert "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9" not in repr_str
    assert "***REDACTED***" in repr_str
    client.close()

    async_client = AsyncSnoozeClient(
        endpoint="localhost:50051",
        api_key="sk-async-secret-key",
    )
    assert "sk-async-secret-key" not in repr(async_client)


def test_configure_logging_human_and_json_formats() -> None:
    buf = io.StringIO()
    configure_logging(level="DEBUG", format_json=False, stream=buf)
    log = get_logger("test_human")

    # Log with registered secret and extra fields
    client = SnoozeClient(api_key="sk-registered-secret-value")
    log.info(
        "Sending request with sk-registered-secret-value",
        extra={"model_id": "bert", "api_key": "sk-registered-secret-value"},
    )
    output = buf.getvalue()
    assert "Sending request with ***REDACTED***" in output
    assert "sk-registered-secret-value" not in output
    assert "model_id=bert" in output
    assert "api_key=***REDACTED***" in output
    client.close()

    # Now test JSON structured output
    json_buf = io.StringIO()
    configure_logging(level=logging.INFO, format_json=True, stream=json_buf)
    log.info("Inference finished", extra={"model_id": "demo-model", "duration_ms": 1.5})
    json_line = json_buf.getvalue().strip().splitlines()[-1]
    parsed = json.loads(json_line)
    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "snoozeml.test_human"
    assert parsed["message"] == "Inference finished"
    assert parsed["model_id"] == "demo-model"
    assert parsed["duration_ms"] == 1.5
