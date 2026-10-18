---
title: "Prometheus Metrics"
description: "SnoozeML natively exposes standardized Prometheus exposition metrics at `GET /metrics`. These metrics provide real-time visibility into inference throughput,..."
---

SnoozeML natively exposes standardized Prometheus exposition metrics at `GET /metrics`. These metrics provide real-time visibility into inference throughput, micro-batching efficiency, and error rates.

---

## 1. Exposed Metrics Reference

Query the endpoint directly:

```bash
curl -s http://127.0.0.1:8080/metrics
```

Output:
```text
# HELP snoozeml_requests_total Total number of inference requests received
# TYPE snoozeml_requests_total counter
snoozeml_requests_total{status="success"} 1520
snoozeml_requests_total{status="error"} 2
snoozeml_requests_total{status="all"} 1522

# HELP snoozeml_inference_latency_seconds_total Total duration of inference execution in seconds
# TYPE snoozeml_inference_latency_seconds_total counter
snoozeml_inference_latency_seconds_total 3.845000

# HELP snoozeml_batches_dispatched_total Total micro-batches processed by the execution engine
# TYPE snoozeml_batches_dispatched_total counter
snoozeml_batches_dispatched_total 124
```

### Metrics Definitions

| Metric Name | Type | Description |
| :--- | :--- | :--- |
| `snoozeml_requests_total` | Counter | Total number of individual client inference requests received, partitioned by `status="success"`, `status="error"`, and `status="all"`. |
| `snoozeml_inference_latency_seconds_total` | Counter | Cumulative wall-clock time spent executing model forward passes in seconds. |
| `snoozeml_batches_dispatched_total` | Counter | Total number of assembled micro-batches dispatched by the batching engine to worker threads. |

---

## 2. PromQL Formulas for Grafana & Alerting

### 1. Request Throughput (Requests Per Second)
Calculate the real-time inference request rate:
```promql
rate(snoozeml_requests_total{status="success"}[1m])
```

### 2. Average Forward Pass Latency
Calculate average compute latency per request:
```promql
rate(snoozeml_inference_latency_seconds_total[1m])
  /
rate(snoozeml_requests_total{status="success"}[1m])
```

### 3. Dynamic Micro-Batching Factor (Batch Efficiency)
Measures the average number of client requests consolidated into each forward pass:
```promql
rate(snoozeml_requests_total{status="all"}[1m])
  /
rate(snoozeml_batches_dispatched_total[1m])
```
*A higher number indicates higher hardware efficiency and greater GPU saturation.*

### 4. Error Percentage
```promql
(
  rate(snoozeml_requests_total{status="error"}[1m])
    /
  rate(snoozeml_requests_total{status="all"}[1m])
) * 100
```

---

## 3. Prometheus Scrape Configuration

Add SnoozeML to your `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: "snoozeml"
    scrape_interval: 10s
    metrics_path: /metrics
    static_configs:
      - targets: ["snooze-service:8080"]
```