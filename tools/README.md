# SnoozeML Performance & Capacity Testing Tool

A standalone development tool for load testing, capacity estimation, and Docker resource monitoring against a running SnoozeML inference daemon.

Zero external dependencies required (uses standard Python 3.8+ library).

---

## Prerequisites

- **Python 3.8+** (core load testing uses standard library with zero pip installs)
- **Docker** (optional; tracks container CPU %, Memory RSS / Limit, Network I/O, and health)
- **Visual Reporting (Optional)**: Install Plotly for rich interactive HTML dashboards:
  ```powershell
  pip install -r tools/requirements.txt
  ```

Start the SnoozeML container first:

```powershell
docker compose up -d
```

---

## Quick Start

Run the capacity hunter to automatically ramp concurrency and find the maximum requests/sec:

```powershell
python tools/perf_test.py
```

---

## Common Recipes

### 1. Global Full Suite & Complete Paged Dashboard (Recommended)
Runs all test stages in sequence (Capacity Ramp-up $\rightarrow$ Sustained 30s Load at peak capacity $\rightarrow$ Scale-to-Zero Reclamation) and outputs a complete, multi-page tabbed visual report:

```powershell
# From repo root:
python tools/perf_test.py --mode all --html-report report.html

# Or from within tools/ directory:
python perf_test.py --mode all --html-report report.html
```

### 2. Find Max Capacity & Breaking Point (Ramp-Up Mode)
Gradually increases concurrency to determine the maximum sustainable throughput (RPS), saturation point, and knee of the latency curve:

```powershell
python tools/perf_test.py --mode ramp --concurrency-steps 1,2,4,8,16,32,64 --step-duration 5
```

### 2. Sustained Fixed-Load Benchmark
Tests steady-state performance and verifies that memory does not leak under continuous traffic:

```powershell
python tools/perf_test.py --mode fixed --concurrency 16 --duration 30
```

### 3. Verify Scale-to-Zero & Memory Reclamation
Tests cold load -> active inference -> unload -> memory cooldown to verify that weights and queue buffers are reclaimed in Docker:

```powershell
python tools/perf_test.py --mode scale-to-zero --model all-MiniLM-L6-v2
```

### 4. Generate Interactive HTML Visual Report (Plotly)
Run the benchmark and immediately render a dashboard:

```powershell
python tools/perf_test.py --mode ramp --html-report report.html
```

Or generate a report from a previously saved JSON file:

```powershell
python tools/report_generator.py perf_results.json --output report.html
```

### 5. Run Full Suite and Export JSON (CI/CD)
Runs ramp-up, sustained load, and scale-to-zero in sequence, writing raw telemetry to JSON:

```powershell
python tools/perf_test.py --mode all --output-json perf_results.json
```

### 6. Target a Custom Model or Custom Payload
Benchmark a specific model with a custom payload file:

```powershell
python tools/perf_test.py --model my-model --payload-file ./custom_payload.json
```

### 7. Non-Docker or Remote Host
Benchmark a remote server without monitoring local Docker containers:

```powershell
python tools/perf_test.py --endpoint http://10.0.0.5:8080 --no-docker
```

---

## Command-Line Options Reference

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--mode` | `ramp` | Test mode: `ramp`, `fixed`, `scale-to-zero`, or `all` |
| `--endpoint` | `http://localhost:8080` | SnoozeML base HTTP endpoint |
| `--container` | `snoozeml` | Name of the Docker container to monitor |
| `--no-docker` | `false` | Disable Docker stats collection (HTTP-only benchmarking) |
| `--model` | *(auto-detected)* | Model ID to benchmark (queries `/v1/models` by default) |
| `--api-key` | *(none)* | Optional Bearer API key for authenticated servers |
| `--concurrency-steps` | `1,2,4,8,16,32,64` | Concurrency levels for `ramp` mode |
| `--step-duration` | `5.0` | Test duration in seconds per step in `ramp` mode |
| `--concurrency` | `16` | Number of concurrent worker threads for `fixed` mode |
| `--duration` | `15.0` | Test duration in seconds for `fixed` mode |
| `--requests` | *(none)* | Maximum total requests to send before stopping |
| `--error-threshold` | `5.0` | Error rate percentage that triggers breaking point detection |
| `--latency-threshold-ms`| `2000.0` | Average latency (ms) threshold to flag system saturation |
| `--payload-file` | *(none)* | Path to custom JSON payload file |
| `--seq-len` | `5` | Sequence length for auto-generated BERT tensor inputs |
| `--output-json` | *(none)* | File path to export structured benchmark results to JSON |
| `--html-report` | *(none)* | File path to export interactive Plotly HTML visual dashboard |

---

## Interpreting Output Metrics

- **Req/s (RPS)**: Overall inference throughput achieved by SnoozeML.
- **P50 / P95 / P99 (ms)**: Latency percentiles measured via high-precision monotonic clock.
- **Docker CPU %**: Percentage of total host CPU utilized (e.g. `200%` means 2 full CPU cores saturated across Rayon worker threads).
- **Docker RAM**: Container resident memory usage (RSS). Used to observe weight allocation and scale-to-zero reclamation.
- **Breaking Point**: Concurrency level where error rate or latency exceeds thresholds, indicating request queue saturation.

---

# SnoozeML Tree Model Training Tool (`tools/train_tree.py`)

A training and export tool for building Gradient Boosted Decision Tree (XGBoost) models that can be served directly by SnoozeML's `tree` backend with zero C/C++ or FFI dependencies.

### Prerequisites

```powershell
pip install xgboost scikit-learn numpy pandas
```

### Quick Recipes

```powershell
# 1. Train on Iris dataset (multiclass classification, 3 classes, 4 features):
python tools/train_tree.py --dataset iris --name iris_model

# 2. Train on California Housing dataset (regression, 8 features):
python tools/train_tree.py --dataset california --name housing_model

# 3. Train on Breast Cancer dataset (binary classification, 30 features):
python tools/train_tree.py --dataset breast_cancer --name cancer_model

# 4. Train from your own custom CSV dataset:
python tools/train_tree.py --csv my_data.csv --target price --name custom_pricing

# 5. Train and immediately verify against a running SnoozeML server:
python tools/train_tree.py --dataset iris --name iris_model --verify
```

### Generated Artifacts

- `models/<model_name>.xgb.json`: The trained XGBoost JSON model ready to be served.
- `models/<model_name>_sample.json`: A valid REST API inference payload matching the exact input feature format.
- A ready-to-paste `models.toml` configuration snippet.

---

# SnoozeML gRPC Endpoint Test & HTTP Comparison Tool (`tools/test_grpc.py`)

A comprehensive test and comparison suite for verifying SnoozeML's gRPC endpoints and validating numerical equivalence and latency differences against HTTP REST endpoints.

Uses dynamic Protocol Buffer loading at runtime with zero codegen or modifications required in the `proto/` directory.

### Prerequisites

```powershell
pip install -r tools/requirements.txt
```

### Quick Recipes

```powershell
# 1. Run all health, inference, equivalence, and error tests against default endpoints (http://localhost:8080, localhost:50051):
python tools/test_grpc.py

# 2. Run test suite using pytest:
pytest tools/test_grpc.py

# 3. Side-by-side numerical equivalence and latency comparison:
python tools/test_grpc.py --compare

# 4. Performance benchmark comparing HTTP vs gRPC throughput (RPS) and latency percentiles (P50, P90, P95, P99):
python tools/test_grpc.py --benchmark --iterations 50

# 5. Target a specific model and custom ports:
python tools/test_grpc.py --http-endpoint http://127.0.0.1:8080 --grpc-endpoint 127.0.0.1:50051 --model all-MiniLM-L6-v2

# 6. Test with Bearer API Key authorization:
python tools/test_grpc.py --api-key sk-admin123
```

### What It Verifies

1. **Standard gRPC Health Checking**: Queries `grpc.health.v1.Health.Check` for the root server (`""`) and `snoozeml.v1.InferenceService`, asserting `SERVING` status.
2. **Identical Inputs & Models**: Generates exact matching tensor inputs for BERT/MiniLM/Nomic (`input_ids` + `attention_mask`), Chronos (`context`), and generic models (`input`), matching `tools/perf_test.py`.
3. **Numerical Output Equivalence**: Sends the identical tensor payloads to HTTP `/v1/models/{model_id}/infer` and gRPC `InferenceService.Infer`, checking:
   - Output tensor count match
   - Output node name match
   - Tensor shape dimension match
   - Float32 element-wise tolerance check (`atol=1e-4`, `rtol=1e-4`)
4. **Negative Error Handling**: Confirms that non-existent models return `NOT_FOUND` (gRPC status 5) and malformed shapes or empty model IDs return `INVALID_ARGUMENT` (gRPC status 3).
5. **Latency & Wire Efficiency**: Compares JSON text payload size vs Protobuf binary size and reports client latency speedup.

