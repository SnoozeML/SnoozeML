---
title: "Health Probes & Orchestration"
description: "SnoozeML implements standard Kubernetes and container-compatible health probes. These endpoints are designed to integrate directly with cloud orchestrators (..."
---

SnoozeML implements standard Kubernetes and container-compatible health probes. These endpoints are designed to integrate directly with cloud orchestrators (such as Kubernetes, Nomad, Docker Compose, and AWS ECS).

---

## 1. Probe Endpoints

### 1. Liveness Probe (`GET /health/live`)
Evaluates whether the SnoozeML daemon process and its internal asynchronous event loops are healthy:
```bash
curl -s http://127.0.0.1:8080/health/live
```
Response (`200 OK`):
```json
{
  "status": "live"
}
```
If the process deadlocks or crashes, the HTTP server stops responding, signaling the orchestrator to restart the container.

---

### 2. Readiness Probe (`GET /health/ready`)
Evaluates whether the daemon has initialized its model registry, listeners, and sweeper tasks and is prepared to receive network traffic:
```bash
curl -s http://127.0.0.1:8080/health/ready
```
Response (`200 OK`):
```json
{
  "status": "ready"
}
```
Load balancers will not route customer traffic to this instance until `/health/ready` returns `200 OK`.

---

### 3. gRPC Health Checking Probe (`grpc.health.v1.Health/Check`)
Evaluates the health of the gRPC serving subsystem using the official standard gRPC health checking protocol:
```bash
grpcurl -plaintext 127.0.0.1:50051 grpc.health.v1.Health/Check
```
Response:
```json
{
  "status": "SERVING"
}
```

You can also check service-specific health:
```bash
grpcurl -plaintext -d '{"service": "snoozeml.v1.InferenceService"}' 127.0.0.1:50051 grpc.health.v1.Health/Check
```

Or using [`grpc-health-probe`](https://github.com/grpc-ecosystem/grpc-health-probe):
```bash
grpc-health-probe -addr=127.0.0.1:50051 -service=snoozeml.v1.InferenceService
# status: SERVING
```

---

## 2. Kubernetes Pod Configuration

Kubernetes (v1.24+) supports **native gRPC probes** alongside standard HTTP probes. Here is a production-grade Kubernetes `Deployment` snippet incorporating both HTTP and gRPC ports and probes:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: snoozeml-daemon
  labels:
    app: snoozeml
spec:
  replicas: 2
  selector:
    matchLabels:
      app: snoozeml
  template:
    metadata:
      labels:
        app: snoozeml
    spec:
      containers:
        - name: snoozeml
          image: myregistry.azurecr.io/snooze-ml:latest
          command: ["snooze", "serve", "--models-dir", "/models", "--port", "8080", "--grpc-port", "50051"]
          ports:
            - containerPort: 8080
              name: http
            - containerPort: 50051
              name: grpc
          # Option A: Native gRPC liveness probe (Kubernetes 1.24+)
          livenessProbe:
            grpc:
              port: 50051
              service: ""
            initialDelaySeconds: 5
            periodSeconds: 10
            timeoutSeconds: 2
            failureThreshold: 3
          # Option B: HTTP readiness probe
          readinessProbe:
            httpGet:
              path: /health/ready
              port: 8080
            initialDelaySeconds: 2
            periodSeconds: 5
            timeoutSeconds: 2
            failureThreshold: 2
          resources:
            requests:
              cpu: "1000m"
              memory: "2Gi"
            limits:
              cpu: "4000m"
              memory: "8Gi"
```

---

## 3. Docker Compose Configuration

For Docker Compose deployments, expose both HTTP and gRPC ports:

```yaml
version: "3.8"

services:
  snoozeml:
    image: ghcr.io/snoozeml/snoozeml:latest
    ports:
      - "8080:8080"   # HTTP REST & Prometheus metrics
      - "50051:50051" # Binary gRPC inference & health checking
    volumes:
      - ./models:/models:ro
    command: ["snooze", "serve", "--models-dir", "/models"]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health/live"]
      interval: 10s
      timeout: 3s
      retries: 3
      start_period: 5s
```