---
title: "Configuring JWT Authentication"
description: "For multi-tenant environments, OAuth2 / OIDC integrations, and browser-facing microservices, SnoozeML supports HMAC-signed **JSON Web Tokens (JWT)**."
---

For multi-tenant environments, OAuth2 / OIDC integrations, and browser-facing microservices, SnoozeML supports HMAC-signed **JSON Web Tokens (JWT)**.

---

## 1. Configuring the JWT Verifier

Configure the JWT settings in `models.toml` or via environment variables:

> [!IMPORTANT]
> **Avoid storing plaintext secrets in `models.toml`**. Storing secret keys in version-controlled configuration files poses security risks. SnoozeML supports resolving the HMAC secret dynamically from environment variables.

### Option A: Via Environment Variable (Recommended)
Simply set `[auth] enabled = true` in `models.toml`, and export `SNOOZE_JWT_SECRET`:

```bash
export SNOOZE_JWT_SECRET="your-cryptographically-secure-shared-secret-key-32-chars-long"
snooze serve --models-dir ./models
```

### Option B: Referencing Environment Variables in `models.toml`
When you need to configure claims validation (`issuer`, `audience`) or custom algorithms alongside environment-injected secrets:

```toml
[auth]
enabled = true

[auth.jwt]
# Dynamically resolve HMAC secret from environment variable (recommended)
secret_env = "SNOOZE_JWT_SECRET"

# Supported algorithms: "HS256" (default), "HS384", "HS512"
algorithm = "HS256"

# Optional: Validate the "iss" (issuer) claim
issuer = "auth.mycompany.com"

# Optional: Validate the "aud" (audience) claim
audience = "snoozeml"

# Clock drift allowance in seconds (default: 60)
leeway_secs = 60
```

> [!TIP]
> You can also use `$VAR` expansion: `secret = "$SNOOZE_JWT_SECRET"`. Hardcoding raw plaintext secrets directly into `secret = "..."` is supported for backward compatibility but triggers a security warning on startup.


---

## 2. JWT Claims Schema

SnoozeML extracts role permissions from either a singular `role` field or a `roles` array in the token payload:

### Example: Single Role
```json
{
  "sub": "user-uuid-12345",
  "name": "Jane Doe",
  "role": "inference",
  "iss": "auth.mycompany.com",
  "aud": "snoozeml",
  "exp": 1790680000
}
```

### Example: Multiple Roles
```json
{
  "sub": "service-account-etl",
  "roles": ["read_only", "inference"],
  "exp": 1790680000
}
```

SnoozeML resolves the **highest privilege** present among the parsed roles.

---

## 3. Generating Tokens for Clients

### In Python (using `PyJWT`)
```python
import time
import jwt

SECRET = "your-cryptographically-secure-shared-secret-key-32-chars-long"

payload = {
    "sub": "api-client-01",
    "role": "inference",
    "iss": "auth.mycompany.com",
    "aud": "snoozeml",
    "exp": int(time.time()) + 3600,  # Valid for 1 hour
}

token = jwt.encode(payload, SECRET, algorithm="HS256")
print("Bearer Token:", token)
```

### In Node.js / JavaScript (using `jsonwebtoken`)
```javascript
const jwt = require("jsonwebtoken");

const secret = "your-cryptographically-secure-shared-secret-key-32-chars-long";
const token = jwt.sign(
  {
    sub: "frontend-client",
    role: "inference",
    iss: "auth.mycompany.com",
    aud: "snoozeml",
  },
  secret,
  { expiresIn: "1h", algorithm: "HS256" }
);

console.log("Bearer Token:", token);
```

---

## 4. Sending JWTs in Requests

### HTTP REST Header
Transmit the token in the standard HTTP `Authorization` header:

```bash
curl -X POST http://127.0.0.1:8080/v1/models/demo-model/infer \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..." \
  -H "Content-Type: application/json" \
  -d '{"inputs": [{"name": "input", "shape": [1, 2], "data": [1.0, 2.0]}]}'
```

### gRPC Metadata Header (`authorization`)
In gRPC, transmit the token in the `authorization` metadata header:

```bash
grpcurl -plaintext \
  -proto proto/snoozeml/v1/inference.proto \
  -H "authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..." \
  -d '{"model_id": "demo-model", "inputs": [{"name": "input", "shape": [1, 2], "raw_data": "AADAQAAAQEAA"}]}' \
  127.0.0.1:50051 snoozeml.v1.InferenceService/Infer
```

In Python:
```python
metadata = [("authorization", f"Bearer {token}")]
response = client.Infer(request, metadata=metadata)
```

---

## 5. Token Error Handling

| Scenario | HTTP Status | gRPC Status | Response Header / Message |
| :--- | :--- | :--- | :--- |
| **Token Expired** | `401 Unauthorized` | `Status::unauthenticated` | `JSON Web Token has expired` |
| **Signature Invalid** | `401 Unauthorized` | `Status::unauthenticated` | `Invalid token signature` |
| **Issuer / Audience Mismatch** | `401 Unauthorized` | `Status::unauthenticated` | `Token claim mismatch` |
| **Insufficient Role** | `403 Forbidden` | `Status::permission_denied` | `Insufficient permissions: requires at least inference role` |
