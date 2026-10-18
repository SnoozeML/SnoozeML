# SnoozeML Authentication & Role-Based Access Control (RBAC) Specification

SnoozeML features a high-performance, asynchronous authentication and role-based access control (RBAC) layer implemented as Axum / Tower middleware.

---

## 1. Core Architecture

The authentication middleware protects SnoozeML's REST endpoints while allowing zero-friction unauthenticated access for health checks and local development.

```mermaid
flowchart TD
    Req[Incoming HTTP Request] --> PubCheck{Is Route Public?\n/health/* or /}
    PubCheck -- Yes --> Handler[Execute Route Handler]
    PubCheck -- No --> AuthEnabled{Is Auth Enabled\nin Config / CLI?}
    AuthEnabled -- No --> InjectAnon[Inject Anonymous Admin] --> Handler
    AuthEnabled -- Yes --> ExtractCreds[Extract Credential\nX-API-Key or Authorization]
    
    ExtractCreds -- Missing --> Err401[401 Unauthorized\nMissing Credentials]
    ExtractCreds -- Found --> MatchKey{Matches Configured\nAPI Key? Constant-Time ct_eq}
    
    MatchKey -- Yes --> IdentityKey[Resolve Identity\nRole from ApiKeyConfig]
    MatchKey -- No --> IsJwt{Valid JWT Signature\n& Unexpired?}
    
    IsJwt -- Yes --> IdentityJwt[Resolve Identity\nRole from Claims]
    IsJwt -- No --> Err401Invalid[401 Unauthorized\nInvalid or Expired Token]
    
    IdentityKey --> RoleCheck{Caller Role >=\nEndpoint Minimum Role?}
    IdentityJwt --> RoleCheck
    
    RoleCheck -- No --> Err403[403 Forbidden\nInsufficient Role Permissions]
    RoleCheck -- Yes --> InjectIdentity[Inject AuthIdentity into Request Extensions]
    InjectIdentity --> Handler
```

### Key Security Properties
- **Zero-Panic Invariants**: All credential parsing, token decoding, and header validations use safe pattern matching and checked conversions (`Result` / `Option`), strictly complying with SnoozeML's zero-panic policy.
- **Timing-Safe API Key Matching**: API keys are evaluated using constant-time digest equality checks (`subtle::ConstantTimeEq` on 32-byte SHA-256 hashes), preventing timing side-channel attacks.
- **Dual Bearer Resolution**: Supports both explicit `X-API-Key` headers and standard `Authorization: Bearer <token>` headers (checking for configured API keys before falling back to JWT validation, enabling compatibility with standard ML client SDKs).
- **Scale-to-Zero Compatibility**: Public health endpoints (`/health/live`, `/health/ready`) are never blocked by authentication, allowing orchestrators (Kubernetes, Nomad) to monitor container lifecycle without managing API secrets.

---

## 2. Role-Based Access Control (RBAC) Matrix

SnoozeML organizes endpoint permissions into a strict 3-tier role hierarchy:

$$\text{Admin} > \text{Inference} > \text{ReadOnly}$$

| Endpoint | Method | Required Role | Description |
| :--- | :--- | :--- | :--- |
| `/` | `GET` | **Public** | Service metadata and version |
| `/health/live` | `GET` | **Public** | Process liveness probe |
| `/health/ready` | `GET` | **Public** | Model server readiness probe |
| `/metrics` | `GET` | **Public** / `Admin` | Prometheus telemetry (configurable via `protect_metrics`) |
| `/v1/models` | `GET` | `ReadOnly` | List registered models and runtime telemetry |
| `/v1/models/:id/infer` | `POST` | `Inference` | Execute dynamic batch inference |
| `/v1/models/:id/load` | `POST` | `Admin` | Force model weights into memory |
| `/v1/models/:id/unload` | `POST` | `Admin` | Scale-to-zero memory reclamation |

*Note: Roles are cumulative. An `Admin` caller is authorized for all `Inference` and `ReadOnly` routes.*

---

## 3. Configuration

Authentication can be configured via `models.toml`, CLI arguments on `snooze serve`, or environment variables.

### 3.1 Via `models.toml` Configuration File

```toml
[auth]
enabled = true
protect_metrics = false # Set true to require Admin role on /metrics

# Optional HMAC-based JSON Web Token validation
[auth.jwt]
secret = "your-secure-jwt-hmac-shared-secret-at-least-32-chars"
algorithm = "HS256"      # HS256 (default), HS384, HS512
issuer = "snoozeml-auth" # Optional: validate iss claim
audience = "snoozeml"    # Optional: validate aud claim
leeway_secs = 60         # Clock drift allowance in seconds

# Static API Keys
[[auth.api_keys]]
key = "sk-admin-secret-987"
role = "admin"
name = "cluster-admin"

[[auth.api_keys]]
key = "sk-infer-client-654"
role = "inference"
name = "web-client"

[[auth.api_keys]]
key = "sk-ro-monitor-321"
role = "read_only"
name = "dashboard-crawler"
```

### 3.2 Via CLI Flags on `snooze serve`

```bash
# Start server with static API keys and a JWT secret
snooze serve \
  --demo \
  --api-key admin:sk-admin-secret-987 \
  --api-key infer:sk-infer-client-654 \
  --jwt-secret "your-secure-jwt-hmac-shared-secret-at-least-32-chars" \
  --protect-metrics
```

### 3.3 Via Environment Variables

```bash
export SNOOZE_JWT_SECRET="your-secure-jwt-hmac-shared-secret-at-least-32-chars"
snooze serve --demo --api-key infer:sk-infer-client-654
```

---

## 4. Client Request Formats & Examples

### 4.1 Using API Keys

#### Header: `X-API-Key`
```bash
curl -X POST http://localhost:8080/v1/models/demo-model/infer \
  -H "X-API-Key: sk-infer-client-654" \
  -H "Content-Type: application/json" \
  -d '{"inputs": [{"name": "input", "shape": [1, 2], "data": [1.0, 2.0]}]}'
```

#### Header: `Authorization: Bearer <key>`
```bash
curl -X POST http://localhost:8080/v1/models/demo-model/infer \
  -H "Authorization: Bearer sk-infer-client-654" \
  -H "Content-Type: application/json" \
  -d '{"inputs": [{"name": "input", "shape": [1, 2], "data": [1.0, 2.0]}]}'
```

### 4.2 Using JSON Web Tokens (JWT)

Clients can generate tokens containing standard claims and either a `role` string or `roles` array:

#### Claims Payload Format
```json
{
  "sub": "user-42",
  "exp": 1790680000,
  "role": "inference",
  "iss": "snoozeml-auth",
  "aud": "snoozeml"
}
```
Or with multiple roles:
```json
{
  "sub": "user-42",
  "exp": 1790680000,
  "roles": ["read_only", "inference"]
}
```

#### Header Submission
```bash
curl -X POST http://localhost:8080/v1/models/demo-model/infer \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..." \
  -H "Content-Type: application/json" \
  -d '{"inputs": [{"name": "input", "shape": [1, 2], "data": [1.0, 2.0]}]}'
```

---

## 5. Error Responses

### Missing Credentials (`401 Unauthorized`)
```http
HTTP/1.1 401 Unauthorized
WWW-Authenticate: Bearer realm="snoozeml"
Content-Type: application/json

{
  "error": "Authentication credentials are missing. Provide 'X-API-Key' or 'Authorization: Bearer <token>' header."
}
```

### Expired JWT (`401 Unauthorized`)
```http
HTTP/1.1 401 Unauthorized
WWW-Authenticate: Bearer realm="snoozeml"
Content-Type: application/json

{
  "error": "JSON Web Token has expired"
}
```

### Insufficient Role (`403 Forbidden`)
```http
HTTP/1.1 403 Forbidden
Content-Type: application/json

{
  "error": "Forbidden: requires Admin role, caller has Inference"
}
```
