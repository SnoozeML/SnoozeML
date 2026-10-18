---
title: "Quickstart Guide"
description: "This walkthrough will get you running your first inference request with SnoozeML in under **2 minutes**, without needing to download external model weights."
---

This walkthrough will get you running your first inference request with SnoozeML in under **2 minutes**, without needing to download external model weights.

---

## Step 1: Start the Server with the Demo Model

SnoozeML includes a built-in mock model (`demo-model`) that exercises the entire asynchronous routing, dynamic micro-batching, and scale-to-zero lifecycle without requiring large weight downloads.

Run:

```bash
snooze serve --demo
```

Output:
```text
2026-09-27T12:00:00.120Z  INFO snooze_ml::server: Registered demo mock model id=demo-model
2026-09-27T12:00:00.125Z  INFO snooze_ml::server: Scale-to-zero background sweeper started
2026-09-27T12:00:00.130Z  INFO snooze_ml::server: SnoozeML HTTP REST Server listening for connections address=0.0.0.0:8080
2026-09-27T12:00:00.132Z  INFO snooze_ml::server: SnoozeML gRPC Server listening for connections address=0.0.0.0:50051
```

By default, the server binds to `0.0.0.0:8080` for **HTTP REST** and `0.0.0.0:50051` for **gRPC**. Both services run concurrently within the same process.

---

## Step 2: Verify Health & Service Info

Open a second terminal window and check the daemon's service information:

```bash
curl -s http://127.0.0.1:8080/ | jq .
```

Response:
```json
{
  "service": "snooze-ml",
  "version": "0.1.0",
  "status": "ok",
  "message": "SnoozeML High-Performance ONNX Inference Daemon",
  "endpoints": {
    "home": "/",
    "models": "/v1/models",
    "health_live": "/health/live",
    "health_ready": "/health/ready",
    "metrics": "/metrics"
  }
}
```

Check the HTTP health probes:
```bash
curl -s http://127.0.0.1:8080/health/live
# {"status":"live"}

curl -s http://127.0.0.1:8080/health/ready
# {"status":"ready"}
```

Verify gRPC health checking via [`grpcurl`](https://github.com/fullstorydev/grpcurl):
```bash
grpcurl -plaintext 127.0.0.1:50051 grpc.health.v1.Health/Check
# { "status": "SERVING" }
```

---

## Step 3: Inspect Registered Models

Query the `/v1/models` endpoint to see all registered models and their current residency states:

```bash
curl -s http://127.0.0.1:8080/v1/models | jq .
```

Response:
```json
[
  {
    "id": "demo-model",
    "state": "idle",
    "description": "Mock model for internal verification and integration testing",
    "device": "auto",
    "idle_timeout_secs": 300,
    "max_batch_size": 16,
    "max_batch_delay_ms": 10,
    "last_accessed_secs_ago": 12
  }
]
```

Notice that `demo-model` starts in the `idle` state. Its weights are **not loaded into memory**, preserving 0 MB of RAM/VRAM!

---

## Step 4: Send an Inference Request

Send a test tensor input to `/v1/models/demo-model/infer`:

```bash
curl -X POST http://127.0.0.1:8080/v1/models/demo-model/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "name": "input",
        "shape": [1, 3],
        "data": [1.0, 2.0, 3.0]
      }
    ]
  }' | jq .
```

Response:
```json
{
  "model_id": "demo-model",
  "outputs": [
    {
      "name": "output",
      "shape": [1, 4],
      "data": [1.0, 2.0, 3.0, 0.0]
    }
  ],
  "duration_ms": 1.15
}
```

### What Just Happened?
1. SnoozeML received the request and noticed `demo-model` was `idle`.
2. It lazily initialized the model into memory.
3. The dynamic batching engine waited for up to `10 ms` to coalesce concurrent requests.
4. The forward pass was executed, and the sliced output was returned to the client.
5. If you query `/v1/models` again, its state is now `ready`.

:::tip
You can also send this inference request over **gRPC** to port `50051`. For complete Python, Rust, TypeScript, and `grpcurl` examples, see the [gRPC Inference Guide](/SnoozeML/client/grpc-inference/).
:::

---

## Step 5: Check Prometheus Metrics

SnoozeML exports real-time Prometheus metrics at `/metrics`:

```bash
curl -s http://127.0.0.1:8080/metrics
```

Output:
```text
# HELP snoozeml_requests_total Total number of inference requests received
# TYPE snoozeml_requests_total counter
snoozeml_requests_total{status="success"} 1
snoozeml_requests_total{status="error"} 0
snoozeml_requests_total{status="all"} 1
# HELP snoozeml_inference_latency_seconds_total Total duration of inference execution in seconds
# TYPE snoozeml_inference_latency_seconds_total counter
snoozeml_inference_latency_seconds_total 0.001150
# HELP snoozeml_batches_dispatched_total Total micro-batches processed by the execution engine
# TYPE snoozeml_batches_dispatched_total counter
snoozeml_batches_dispatched_total 1
```

---

## Step 6: Launch the Terminal Monitor

Open a separate terminal window and launch the live TUI dashboard:

```bash
snooze monitor --endpoint http://127.0.0.1:8080
```

You will see a live interactive dashboard displaying active models, states, request throughput, and queue depth.

---

## Step 7: Test Scale-to-Zero

Wait `300` seconds (the default `idle_timeout`), or immediately trigger manual deallocation:

```bash
# Using the CLI:
snooze unload --model-id demo-model

# Or via the HTTP REST API:
curl -X POST http://127.0.0.1:8080/v1/models/demo-model/unload
```

Check `/v1/models` again: the model has reverted to `idle` and released all memory back to the operating system.