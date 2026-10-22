"""Authentication helpers, RBAC role definitions, and HMAC JWT minting/verification for SnoozeML.

Provides zero-dependency RFC 7519 JSON Web Token (JWT) creation and verification
compatible with SnoozeML's Rust authentication service (``src/auth.rs``).
"""

from __future__ import annotations

import base64
from enum import Enum
import hashlib
import hmac
import json
import os
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

from .exceptions import AuthenticationError, InvalidInputError
from .logging import register_secret_for_redaction


class Role(str, Enum):
    """Role-Based Access Control (RBAC) tier matching SnoozeML's ``src/auth.rs`` hierarchy.

    Hierarchy: ``READ_ONLY (1) < INFERENCE (2) < ADMIN (3)``.
    """

    READ_ONLY = "read_only"
    """Read-only access to ``GET /v1/models``."""

    INFERENCE = "inference"
    """Permission to execute model inference (REST & gRPC) and read model metadata."""

    ADMIN = "admin"
    """Full administrative access (load, unload, pin, unpin, upload, refresh, protected metrics)."""

    @classmethod
    def from_str(cls, value: Union["Role", str]) -> "Role":
        """Normalizes a role string or :class:`Role` enum into a canonical :class:`Role`.

        Accepts aliases supported by the Rust daemon (``"admin"``, ``"administrator"``,
        ``"infer"``, ``"inference"``, ``"user"``, ``"read"``, ``"readonly"``, ``"read_only"``, ``"ro"``).
        """
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower()
        if normalized in ("admin", "administrator"):
            return cls.ADMIN
        if normalized in ("read", "readonly", "read_only", "ro"):
            return cls.READ_ONLY
        if normalized in ("infer", "inference", "user"):
            return cls.INFERENCE
        raise InvalidInputError(
            f"Unsupported role '{value}'. Expected one of: 'read_only', 'inference', 'admin'."
        )


class JwtAlgorithm(str, Enum):
    """Supported HMAC signature algorithms for SnoozeML JWT authentication."""

    HS256 = "HS256"
    HS384 = "HS384"
    HS512 = "HS512"

    @classmethod
    def from_str(cls, value: Union["JwtAlgorithm", str]) -> "JwtAlgorithm":
        """Parses a case-insensitive algorithm identifier into a :class:`JwtAlgorithm`."""
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().upper()
        for member in cls:
            if member.value == normalized:
                return member
        raise InvalidInputError(
            f"Unsupported JWT algorithm '{value}'. Supported algorithms: HS256, HS384, HS512."
        )


_HASH_ALGORITHMS: Dict[JwtAlgorithm, Callable[[], Any]] = {
    JwtAlgorithm.HS256: hashlib.sha256,
    JwtAlgorithm.HS384: hashlib.sha384,
    JwtAlgorithm.HS512: hashlib.sha512,
}


def _b64url_encode(data: bytes) -> str:
    """Encodes raw bytes into unpadded URL-safe Base64 (RFC 7515)."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(segment: str) -> bytes:
    """Decodes an unpadded URL-safe Base64 string into raw bytes."""
    pad_len = (-len(segment)) % 4
    padded = segment + ("=" * pad_len)
    try:
        return base64.urlsafe_b64decode(padded.encode("ascii"))
    except Exception as exc:
        raise AuthenticationError(f"Malformed Base64URL segment in JWT: {exc}") from exc


def create_jwt_token(
    secret: Optional[str] = None,
    *,
    subject: str = "snoozeml-client",
    role: Optional[Union[Role, str]] = Role.INFERENCE,
    roles: Optional[Sequence[Union[Role, str]]] = None,
    expires_in_secs: int = 3600,
    exp: Optional[int] = None,
    iat: Optional[int] = None,
    nbf: Optional[int] = None,
    issuer: Optional[str] = None,
    audience: Optional[str] = None,
    algorithm: Union[JwtAlgorithm, str] = JwtAlgorithm.HS256,
    extra_claims: Optional[Mapping[str, Any]] = None,
) -> str:
    """Mints a signed HMAC JSON Web Token (JWT) compatible with the SnoozeML daemon.

    Both the shared ``secret`` and the generated JWT are automatically registered with
    SnoozeML's log redaction filter so they never leak in client telemetry.

    Args:
        secret: Shared HMAC secret key. If ``None``, reads ``SNOOZE_JWT_SECRET`` from the environment.
        subject: Subject identifier claim (``sub``).
        role: Primary RBAC role claim (``"read_only"``, ``"inference"``, or ``"admin"``).
        roles: Optional list of RBAC roles assigned to the token (``roles`` claim).
        expires_in_secs: Token lifetime in seconds from ``now`` when ``exp`` is not explicitly provided.
        exp: Explicit expiration epoch timestamp in seconds. Overrides ``expires_in_secs``.
        iat: Optional issued-at epoch timestamp in seconds (defaults to current UTC epoch time).
        nbf: Optional not-before epoch timestamp in seconds.
        issuer: Optional issuer claim (``iss``).
        audience: Optional audience claim (``aud``).
        algorithm: HMAC algorithm (:attr:`JwtAlgorithm.HS256`, ``HS384``, or ``HS512``).
        extra_claims: Optional additional custom JSON claims to include in the payload.

    Returns:
        Compact ``header.payload.signature`` JWT string ready for ``SnoozeClient(token=...)``.

    Raises:
        InvalidInputError: If the secret, subject, role, or expiration parameters are invalid.
    """
    resolved_secret = secret if secret is not None else os.getenv("SNOOZE_JWT_SECRET")
    if not resolved_secret or not str(resolved_secret).strip():
        raise InvalidInputError(
            "JWT secret is required; pass 'secret' or set the SNOOZE_JWT_SECRET environment variable."
        )
    resolved_secret = str(resolved_secret).strip()
    register_secret_for_redaction(resolved_secret)

    if not subject or not str(subject).strip():
        raise InvalidInputError("JWT 'subject' (sub) claim cannot be empty")

    algo = JwtAlgorithm.from_str(algorithm)
    now_epoch = int(time.time())
    issued_at = int(iat) if iat is not None else now_epoch

    if exp is not None:
        expiration = int(exp)
    else:
        if expires_in_secs <= 0:
            raise InvalidInputError("expires_in_secs must be positive")
        expiration = issued_at + int(expires_in_secs)

    claims: Dict[str, Any] = {}
    if extra_claims:
        claims.update(extra_claims)

    claims["sub"] = str(subject).strip()
    claims["exp"] = expiration
    claims["iat"] = issued_at

    if nbf is not None:
        claims["nbf"] = int(nbf)
    if issuer is not None:
        claims["iss"] = str(issuer)
    if audience is not None:
        claims["aud"] = str(audience)
    if role is not None:
        claims["role"] = Role.from_str(role).value
    if roles is not None:
        claims["roles"] = [Role.from_str(r).value for r in roles]

    header = {"alg": algo.value, "typ": "JWT"}
    header_b64 = _b64url_encode(
        json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    payload_b64 = _b64url_encode(
        json.dumps(claims, separators=(",", ":")).encode("utf-8")
    )

    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    digest_fn = _HASH_ALGORITHMS[algo]
    signature = hmac.new(
        resolved_secret.encode("utf-8"), signing_input, digest_fn
    ).digest()
    token = f"{header_b64}.{payload_b64}.{_b64url_encode(signature)}"
    register_secret_for_redaction(token)
    return token


# Convenience alias
mint_jwt_token = create_jwt_token


def decode_jwt_token(
    token: str,
    secret: Optional[str] = None,
    *,
    verify: bool = True,
    algorithm: Union[JwtAlgorithm, str] = JwtAlgorithm.HS256,
    leeway_secs: int = 60,
    issuer: Optional[str] = None,
    audience: Optional[str] = None,
) -> Dict[str, Any]:
    """Decodes and optionally verifies a SnoozeML HMAC JSON Web Token.

    Args:
        token: Compact ``header.payload.signature`` JWT string.
        secret: Shared HMAC secret key (required when ``verify=True``; falls back to ``SNOOZE_JWT_SECRET``).
        verify: Whether to verify the HMAC signature and temporal/issuer/audience claims.
        algorithm: Expected HMAC algorithm (default ``HS256``).
        leeway_secs: Clock skew tolerance in seconds for ``exp`` and ``nbf`` checks (default ``60``).
        issuer: Expected ``iss`` claim value if issuer validation is desired.
        audience: Expected ``aud`` claim value if audience validation is desired.

    Returns:
        Decoded JWT claims dictionary.

    Raises:
        AuthenticationError: If the token format, signature, or claims validation fails.
        InvalidInputError: If ``verify=True`` and no secret is provided.
    """
    if not token or not isinstance(token, str):
        raise AuthenticationError("JWT token must be a non-empty string")

    cleaned = token.strip()
    if cleaned.lower().startswith("bearer "):
        cleaned = cleaned[7:].strip()

    parts = cleaned.split(".")
    if len(parts) != 3:
        raise AuthenticationError("Invalid JWT structure: expected 3 dot-separated segments")

    header_b64, payload_b64, signature_b64 = parts
    try:
        header = json.loads(_b64url_decode(header_b64).decode("utf-8"))
        claims = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
    except AuthenticationError:
        raise
    except Exception as exc:
        raise AuthenticationError(f"Failed to decode JWT JSON segments: {exc}") from exc

    if not isinstance(header, dict) or not isinstance(claims, dict):
        raise AuthenticationError("JWT header and payload must be JSON objects")

    if verify:
        resolved_secret = secret if secret is not None else os.getenv("SNOOZE_JWT_SECRET")
        if not resolved_secret or not str(resolved_secret).strip():
            raise InvalidInputError(
                "JWT secret is required to verify token; pass 'secret' or set SNOOZE_JWT_SECRET."
            )
        resolved_secret = str(resolved_secret).strip()
        register_secret_for_redaction(resolved_secret)

        expected_algo = JwtAlgorithm.from_str(algorithm)
        token_alg = str(header.get("alg", ""))
        if token_alg != expected_algo.value:
            raise AuthenticationError(
                f"JWT algorithm mismatch: token uses '{token_alg}', expected '{expected_algo.value}'"
            )

        signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
        expected_sig = hmac.new(
            resolved_secret.encode("utf-8"),
            signing_input,
            _HASH_ALGORITHMS[expected_algo],
        ).digest()
        actual_sig = _b64url_decode(signature_b64)

        if not hmac.compare_digest(expected_sig, actual_sig):
            raise AuthenticationError("Invalid JSON Web Token: signature verification failed")

        now_epoch = int(time.time())
        leeway = max(0, int(leeway_secs))

        if "exp" not in claims:
            raise AuthenticationError("Invalid JSON Web Token: missing required 'exp' claim")
        try:
            exp_val = int(claims["exp"])
        except (ValueError, TypeError) as exc:
            raise AuthenticationError("Invalid 'exp' claim in JWT") from exc

        if now_epoch > exp_val + leeway:
            raise AuthenticationError("JSON Web Token has expired")

        if "nbf" in claims and claims["nbf"] is not None:
            try:
                nbf_val = int(claims["nbf"])
            except (ValueError, TypeError) as exc:
                raise AuthenticationError("Invalid 'nbf' claim in JWT") from exc
            if now_epoch + leeway < nbf_val:
                raise AuthenticationError("JSON Web Token is not yet valid (nbf)")

        if issuer is not None and claims.get("iss") != issuer:
            raise AuthenticationError(
                f"Invalid JWT issuer: expected '{issuer}', got '{claims.get('iss')}'"
            )

        if audience is not None and claims.get("aud") != audience:
            raise AuthenticationError(
                f"Invalid JWT audience: expected '{audience}', got '{claims.get('aud')}'"
            )

    return dict(claims)
