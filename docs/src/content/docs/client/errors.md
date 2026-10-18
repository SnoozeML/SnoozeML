---
title: "Error Handling & Status Codes"
description: "SnoozeML uses standard HTTP response codes and canonical gRPC status codes to provide actionable failure telemetry. This page details each error condition ac..."
---

SnoozeML uses standard HTTP response codes and canonical gRPC status codes to provide actionable failure telemetry. This page details each error condition across both protocols, its root cause, and the recommended client retry strategy.

---

## 1. Summary of HTTP & gRPC Status Codes

| HTTP Status | gRPC Status (`tonic::Code`) | Reason | Retriable? | Cause / Action |
| :--- | :--- | :--- | :--- | :--- |
| `200 OK` | `OK` (`0`) | Success | N/A | Request completed successfully. |
| `400 Bad Request` | `INVALID_ARGUMENT` (`3`) | Client Error | **No** | Tensor shape mismatch, non-positive dimension, or invalid buffer length. |
| `401 Unauthorized` | `UNAUTHENTICATED` (`16`) | Auth Error | **No** (until fixed) | Missing, invalid, or expired credentials. |
| `403 Forbidden` | `PERMISSION_DENIED` (`7`) | Auth Error | **No** (until fixed) | Caller lacks required role for the target operation. |
| `404 Not Found` | `NOT_FOUND` (`5`) | Client Error | **No** | Target `model_id` does not exist in registry. |
| `501 Not Implemented`| `UNIMPLEMENTED` (`12`) | Format Error | **No** | Model architecture is unsupported by the backend. |
| `503 Service Unavailable`| `RESOURCE_EXHAUSTED` (`8`) / `UNAVAILABLE` (`14`)| Saturation / Loading | **Yes** (with backoff) | Memory is full and all active models are busy (`retry-after` header), or model is loading. |
| `504 Gateway Timeout` | `DEADLINE_EXCEEDED` (`4`) | Timeout | **Yes** (with backoff) | Batching or execution deadline exceeded. |
| `500 Internal Error` | `INTERNAL` (`13`) | Server Error | **No** | Unexpected backend runtime or calculation error. |

---

## 2. Detailed Error Scenarios

### `400 Bad Request` (Shape / Dimension Mismatch)
Occurs when the number of elements in `data` does not match the product of `shape`:
```http
HTTP/1.1 400 Bad Request
Content-Type: application/json

{
  "error": "Tensor 'input' has shape [1, 4] (requires 4 elements), but received 3 data points."
}
```
**Resolution**: Verify that the client serialized all dimensions and did not truncate the array.

---

### `401 Unauthorized` (Missing or Invalid Credentials)
Occurs when authentication is enabled and the request lacks valid credentials:
```http
HTTP/1.1 401 Unauthorized
WWW-Authenticate: Bearer realm="snoozeml"
Content-Type: application/json

{
  "error": "Authentication credentials are missing. Provide 'X-API-Key' or 'Authorization: Bearer <token>' header."
}
```
**Resolution**: Include an authorized API key in `X-API-Key` or a valid token in `Authorization: Bearer`.

---

### `403 Forbidden` (Insufficient Role)
Occurs when valid credentials are provided, but the assigned role is insufficient for the route:
```http
HTTP/1.1 403 Forbidden
Content-Type: application/json

{
  "error": "Forbidden: requires Admin role, caller has Inference"
}
```
**Resolution**: Use a credential with `Admin` privileges to execute administrative lifecycle operations.

---

### `404 Not Found` (Unknown Model)
Occurs when attempting to query a model that has not been registered in the daemon:
```http
HTTP/1.1 404 Not Found
Content-Type: application/json

{
  "error": "Model not found: 'non-existent-model'"
}
```
**Resolution**: Query `GET /v1/models` to inspect the list of available model IDs.

---

### `503 Service Unavailable` (Resource Saturation)
Occurs when GPU or host memory is completely exhausted and all currently resident models are actively executing batches (meaning none can be safely evicted):
```http
HTTP/1.1 503 Service Unavailable
Retry-After: 5
Content-Type: application/json

{
  "error": "Resource limit reached: cannot allocate model even after deallocating idle sessions: Out of memory loading 'transformer-xxl'. All active models are currently busy."
}
```
**Resolution**: The client should wait for the duration specified in the `Retry-After` header (typically 5 seconds) before retrying.

---

## 3. Recommended Client Retry Implementation

### HTTP REST Client (Python `requests`)

```python
import time
import random
import requests

def infer_with_retry(endpoint: str, payload: dict, max_retries: int = 3):
    headers = {"Content-Type": "application/json", "X-API-Key": "sk-my-key"}
    
    for attempt in range(max_retries):
        resp = requests.post(endpoint, json=payload, headers=headers)
        
        # Success: return results
        if resp.status_code == 200:
            return resp.json()
            
        # Retriable server overload (503 Service Unavailable)
        if resp.status_code == 503:
            retry_after = int(resp.headers.get("Retry-After", 2))
            jitter = random.uniform(0.1, 0.5)
            wait_time = retry_after + jitter
            print(f"[503] Server saturated. Backing off for {wait_time:.2f}s...")
            time.sleep(wait_time)
            continue
            
        # Non-retriable client errors: fail immediately
        resp.raise_for_status()
        
    raise RuntimeError(f"Exceeded max retries ({max_retries}) querying {endpoint}")
```

### gRPC Client (Python `grpcio`)

```python
import time
import random
import grpc

def grpc_infer_with_retry(client, request, metadata=None, max_retries=3):
    for attempt in range(max_retries):
        try:
            return client.Infer(request, metadata=metadata)
        except grpc.RpcError as e:
            # Check for retriable capacity or loading errors
            if e.code() in (grpc.StatusCode.RESOURCE_EXHAUSTED, grpc.StatusCode.UNAVAILABLE):
                # Inspect trailing metadata for retry-after hint
                retry_after = 2.0
                for key, val in e.trailing_metadata():
                    if key.lower() == "retry-after":
                        try:
                            retry_after = float(val)
                        except ValueError:
                            pass
                jitter = random.uniform(0.1, 0.5)
                wait_time = retry_after + jitter
                print(f"[{e.code()}] Server busy. Retrying in {wait_time:.2f}s...")
                time.sleep(wait_time)
                continue
                
            # Non-retriable errors (InvalidArgument, Unauthenticated, NotFound): fail immediately
            raise
            
    raise RuntimeError(f"Exceeded max retries ({max_retries})")
```