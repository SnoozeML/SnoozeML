"""Structured logging and credential sanitization utilities for the SnoozeML Python SDK.

Provides a hierarchical logger rooted at ``"snoozeml"``, automatic redaction of
API keys and JWT bearer tokens from log records, and a helper to configure
human-readable or JSON-formatted structured output.
"""

from __future__ import annotations

import datetime
import json
import logging
import re
import sys
from typing import Any, Dict, MutableMapping, Optional, Set, TextIO, Union

LOGGER_NAME = "snoozeml"

# Regex patterns matching common credential formats in headers or strings
_BEARER_PATTERN = re.compile(r"(?i)(bearer\s+)([A-Za-z0-9\-._~+/]+=*)")
_API_KEY_KV_PATTERN = re.compile(
    r"(?i)((?:x-api-key|api_key|apikey|jwt_secret|secret|token|authorization)\s*[:=]\s*['\"]?)([^\s,'\"};]+)"
)
_SENSITIVE_FIELD_NAMES = frozenset(
    {
        "api_key",
        "apikey",
        "x-api-key",
        "authorization",
        "token",
        "jwt",
        "secret",
        "password",
        "hf_token",
    }
)


def mask_secret(secret: Optional[str]) -> str:
    """Masks a sensitive credential string so raw secrets are never exposed.

    Args:
        secret: The raw credential (API key, JWT token, etc.) or ``None``.

    Returns:
        ``"None"`` if ``secret`` is ``None``, ``"***REDACTED***"`` otherwise.
    """
    if secret is None:
        return "None"
    return "***REDACTED***"


def sanitize_text(text: str, known_secrets: Optional[Set[str]] = None) -> str:
    """Scrubs Bearer tokens, API key key-value pairs, and registered secrets from text.

    Args:
        text: Input string potentially containing credentials.
        known_secrets: Optional set of exact secret strings to replace.

    Returns:
        Sanitized string with credentials replaced by ``***REDACTED***``.
    """
    if not text:
        return text

    sanitized = _BEARER_PATTERN.sub(r"\1***REDACTED***", text)
    sanitized = _API_KEY_KV_PATTERN.sub(r"\1***REDACTED***", sanitized)

    if known_secrets:
        for secret in known_secrets:
            if secret and secret in sanitized:
                sanitized = sanitized.replace(secret, "***REDACTED***")

    return sanitized


def sanitize_mapping(
    mapping: MappingOrDict, known_secrets: Optional[Set[str]] = None
) -> Dict[str, Any]:
    """Returns a shallow copy of a dictionary with sensitive keys redacted.

    Args:
        mapping: Dictionary or mapping (such as HTTP headers or gRPC metadata).
        known_secrets: Optional set of exact secret strings to replace.

    Returns:
        A sanitized dictionary safe for logging.
    """
    result: Dict[str, Any] = {}
    for key, value in mapping.items():
        key_lower = str(key).lower()
        if key_lower in _SENSITIVE_FIELD_NAMES:
            result[str(key)] = "***REDACTED***"
        elif isinstance(value, str):
            result[str(key)] = sanitize_text(value, known_secrets)
        elif isinstance(value, dict):
            result[str(key)] = sanitize_mapping(value, known_secrets)
        else:
            result[str(key)] = value
    return result


MappingOrDict = Union[Dict[str, Any], MutableMapping[str, Any]]


class RedactingFilter(logging.Filter):
    """Logging filter that strips API keys, JWTs, and registered secrets from log records."""

    def __init__(self, name: str = "") -> None:
        super().__init__(name)
        self._known_secrets: Set[str] = set()

    def register_secret(self, secret: Optional[str]) -> None:
        """Registers a secret string to be unconditionally scrubbed from log messages.

        Args:
            secret: Raw API key or token string.
        """
        if secret and len(secret) >= 4:
            self._known_secrets.add(secret)

    def filter(self, record: logging.LogRecord) -> bool:
        """Sanitizes ``record.msg``, ``record.args``, and structured extra fields in-place."""
        if isinstance(record.msg, str):
            record.msg = sanitize_text(record.msg, self._known_secrets)

        if record.args:
            if isinstance(record.args, dict):
                record.args = sanitize_mapping(record.args, self._known_secrets)
            elif isinstance(record.args, tuple):
                record.args = tuple(
                    sanitize_text(arg, self._known_secrets)
                    if isinstance(arg, str)
                    else (
                        sanitize_mapping(arg, self._known_secrets)
                        if isinstance(arg, dict)
                        else arg
                    )
                    for arg in record.args
                )

        # Sanitize structured extra attributes if present
        for attr in list(vars(record).keys()):
            if attr.lower() in _SENSITIVE_FIELD_NAMES:
                setattr(record, attr, "***REDACTED***")
            else:
                val = getattr(record, attr)
                if isinstance(val, str) and attr not in (
                    "name",
                    "levelname",
                    "pathname",
                    "filename",
                    "module",
                    "funcName",
                ):
                    setattr(record, attr, sanitize_text(val, self._known_secrets))

        return True


_GLOBAL_REDACTING_FILTER = RedactingFilter()

_STANDARD_LOG_RECORD_ATTRS = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
        "taskName",
    }
)


class StructuredFormatter(logging.Formatter):
    """Formats log records as structured key-value lines or JSON objects."""

    def __init__(self, *, format_json: bool = False) -> None:
        super().__init__()
        self.format_json = format_json

    def format(self, record: logging.LogRecord) -> str:
        """Formats the log record including any structured ``extra`` fields."""
        _GLOBAL_REDACTING_FILTER.filter(record)
        message = record.getMessage()
        timestamp = datetime.datetime.fromtimestamp(
            record.created, tz=datetime.timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

        extras: Dict[str, Any] = {}
        for key, val in vars(record).items():
            if key not in _STANDARD_LOG_RECORD_ATTRS and not key.startswith("_"):
                extras[key] = val

        if self.format_json:
            payload: Dict[str, Any] = {
                "timestamp": timestamp,
                "level": record.levelname,
                "logger": record.name,
                "message": message,
            }
            payload.update(extras)
            if record.exc_info:
                payload["exception"] = self.formatException(record.exc_info)
            return json.dumps(payload, default=str)

        extra_str = ""
        if extras:
            kv_pairs = [f"{k}={v}" for k, v in extras.items()]
            extra_str = " " + " ".join(kv_pairs)

        formatted = f"{timestamp} [{record.levelname}] {record.name}: {message}{extra_str}"
        if record.exc_info:
            formatted += "\n" + self.formatException(record.exc_info)
        return formatted


def register_secret_for_redaction(secret: Optional[str]) -> None:
    """Registers a credential with the global SDK redacting filter.

    Args:
        secret: Raw API key or JWT token string.
    """
    _GLOBAL_REDACTING_FILTER.register_secret(secret)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Returns a child logger under the ``"snoozeml"`` hierarchy with secret redaction enabled.

    Args:
        name: Sub-module name (e.g. ``"client"``, ``"converters"``). If ``None``,
            returns the root ``"snoozeml"`` logger.

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    full_name = (
        LOGGER_NAME
        if not name or name == LOGGER_NAME
        else (name if name.startswith(f"{LOGGER_NAME}.") else f"{LOGGER_NAME}.{name}")
    )
    logger = logging.getLogger(full_name)
    if _GLOBAL_REDACTING_FILTER not in logger.filters:
        logger.addFilter(_GLOBAL_REDACTING_FILTER)
    return logger


def configure_logging(
    level: Union[int, str] = logging.INFO,
    *,
    format_json: bool = False,
    stream: Optional[TextIO] = None,
) -> logging.Logger:
    """Configures structured console or JSON logging for the ``snoozeml`` package.

    Args:
        level: Logging level (e.g. ``logging.DEBUG``, ``"INFO"``, ``"WARNING"``).
        format_json: If ``True``, emit single-line JSON records; otherwise emit
            human-readable structured key-value lines.
        stream: Output stream (defaults to :data:`sys.stderr`).

    Returns:
        The configured root ``"snoozeml"`` :class:`logging.Logger`.
    """
    root_logger = get_logger()
    if isinstance(level, str):
        resolved_level = getattr(logging, level.upper(), logging.INFO)
    else:
        resolved_level = int(level)

    root_logger.setLevel(resolved_level)

    # Remove existing StreamHandlers added by previous configure_logging calls
    for existing in list(root_logger.handlers):
        if not isinstance(existing, logging.NullHandler):
            root_logger.removeHandler(existing)

    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setLevel(resolved_level)
    handler.setFormatter(StructuredFormatter(format_json=format_json))
    handler.addFilter(_GLOBAL_REDACTING_FILTER)
    root_logger.addHandler(handler)
    return root_logger


# Attach a default NullHandler and RedactingFilter to the package root logger per PEP 282
_root = logging.getLogger(LOGGER_NAME)
_root.addHandler(logging.NullHandler())
_root.addFilter(_GLOBAL_REDACTING_FILTER)
