#!/usr/bin/env python3
"""SnoozeML Concurrent Multi-Model Benchmark Tool.

Sends simultaneous, asynchronous inference requests across multiple heterogeneous
models (e.g., Hugging Face Safetensors BGE embeddings, Pure-Rust XGBoost GBDT,
and Process-Isolated Python Worker Chronos-Bolt) against a running SnoozeML instance.

Demonstrates:
  - Multi-model concurrency & zero GIL contention
  - Dynamic micro-batching along Axis(0) per model
  - Real-time GPU compute and VRAM monitoring via `nvtop`

Requirements:
  Python 3.8+ (Zero external dependencies, standard library only).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple


def build_payload_for_model(model_id: str, sample_dir: str = "models") -> bytes:
    """Constructs valid JSON inference payload for a known model type."""
    mid = model_id.lower()

    # 1. Check for a dedicated sample file
    sample_file = os.path.join(sample_dir, f"{model_id}_sample.json")
    if os.path.isfile(sample_file):
        try:
            with open(sample_file, "r", encoding="utf-8") as f:
                return f.read().encode("utf-8")
        except Exception:
            pass

    # 2. Heuristics based on model identifier
    if "bge" in mid or "bert" in mid or "minilm" in mid or "embed" in mid:
        # Standard BERT / BGE transformer input
        tokens = [101.0, 7592.0, 2088.0, 999.0, 102.0]
        payload = {
            "inputs": [
                {
                    "name": "input_ids",
                    "shape": [1, len(tokens)],
                    "data": tokens,
                },
                {
                    "name": "attention_mask",
                    "shape": [1, len(tokens)],
                    "data": [1.0] * len(tokens),
                },
            ]
        }
    elif "chronos" in mid:
        # Chronos-Bolt time series context
        data = [
            112.0, 118.0, 132.0, 129.0, 121.0, 135.0, 148.0, 148.0,
            136.0, 119.0, 104.0, 118.0, 115.0, 126.0, 141.0, 135.0,
            125.0, 149.0, 170.0, 170.0, 158.0, 133.0, 114.0, 140.0,
            145.0, 150.0, 178.0, 163.0, 172.0, 178.0, 199.0, 199.0,
        ]
        payload = {
            "inputs": [
                {
                    "name": "context",
                    "shape": [1, len(data)],
                    "data": data,
                }
            ]
        }
    elif "tree" in mid or "cancer" in mid or "xgb" in mid:
        # XGBoost Breast Cancer (30 features)
        payload = {
            "inputs": [
                {
                    "name": "features",
                    "shape": [1, 30],
                    "data": [
                        17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471,
                        0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904,
                        0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0,
                        0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189,
                    ],
                }
            ]
        }
    else:
        # Generic fallback
        payload = {
            "inputs": [
                {
                    "name": "input",
                    "shape": [1, 10],
                    "data": [1.0] * 10,
                }
            ]
        }
    return json.dumps(payload).encode("utf-8")


def percentile(data: List[float], p: float) -> float:
    """Calculates percentile from sorted float array."""
    if not data:
        return 0.0
    k = (len(data) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return data[int(k)]
    return data[int(f)] * (c - k) + data[int(c)] * (k - f)


def send_inference_request(
    endpoint: str,
    model_id: str,
    payload_bytes: bytes,
    timeout: float = 30.0,
) -> Tuple[bool, float, Optional[str]]:
    """Sends a single HTTP POST inference request and returns (success, latency_ms, error_msg)."""
    url = f"{endpoint.rstrip('/')}/v1/models/{model_id}/infer"
    req = urllib.request.Request(
        url,
        data=payload_bytes,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return (resp.status == 200, elapsed_ms, None)
    except urllib.error.HTTPError as e:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return (False, elapsed_ms, f"HTTP {e.code}")
    except Exception as e:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return (False, elapsed_ms, str(e))


def worker_loop(
    endpoint: str,
    model_id: str,
    payload_bytes: bytes,
    duration: float,
    stop_event: Dict[str, bool],
    results: List[Tuple[bool, float, Optional[str]]],
) -> None:
    """Continuously fires requests until duration expires."""
    deadline = time.time() + duration
    while time.time() < deadline and not stop_event.get("stop", False):
        res = send_inference_request(endpoint, model_id, payload_bytes)
        results.append(res)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SnoozeML Concurrent Multi-Model Benchmark Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="http://localhost:8082",
        help="SnoozeML HTTP base URL (e.g. http://localhost:8082 for docker-compose cuda)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="bge_small,cancer_xgboost,chronos_bolt",
        help="Comma-separated model IDs to benchmark concurrently",
    )
    parser.add_argument(
        "--concurrency-per-model",
        type=int,
        default=8,
        help="Number of concurrent client workers per model",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Benchmark duration in seconds",
    )
    parser.add_argument(
        "--sample-dir",
        type=str,
        default="models",
        help="Directory to check for sample JSON payload files",
    )
    args = parser.parse_args()

    endpoint = args.endpoint.rstrip("/")
    model_ids = [m.strip() for m in args.models.split(",") if m.strip()]

    print("=" * 72)
    print("  SnoozeML Concurrent Multi-Model Benchmark")
    print("=" * 72)
    print(f"  Endpoint:              {endpoint}")
    print(f"  Target Models:         {', '.join(model_ids)}")
    print(f"  Concurrency / Model:   {args.concurrency_per_model}")
    print(f"  Total Concurrency:     {len(model_ids) * args.concurrency_per_model}")
    print(f"  Test Duration:         {args.duration:.1f}s")
    print("=" * 72)

    # 1. Health check & model verification
    try:
        with urllib.request.urlopen(f"{endpoint}/health/live", timeout=5.0) as resp:
            if resp.status != 200:
                print(f"[ERROR] Endpoint {endpoint} returned status {resp.status}", file=sys.stderr)
                sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to reach SnoozeML at {endpoint}: {e}", file=sys.stderr)
        print("  Ensure the container is running: docker compose --profile cuda up -d", file=sys.stderr)
        sys.exit(1)

    # Fetch registered models from server
    try:
        with urllib.request.urlopen(f"{endpoint}/v1/models", timeout=5.0) as resp:
            models_data = json.loads(resp.read().decode("utf-8"))
            registered = {m.get("id"): m for m in models_data}
    except Exception as e:
        print(f"[WARN] Failed to query /v1/models ({e}); proceeding with requested list", file=sys.stderr)
        registered = {}

    for mid in model_ids:
        if registered and mid not in registered:
            print(f"[WARN] Model '{mid}' not listed in /v1/models (registered: {list(registered.keys())})")
        else:
            state = registered.get(mid, {}).get("state", "registered")
            desc = registered.get(mid, {}).get("description", "")
            print(f"  [+] Ready: {mid:<16} (state: {state}) {desc}")

    # Build payloads
    payloads = {mid: build_payload_for_model(mid, args.sample_dir) for mid in model_ids}

    # Warm-up / cold-start request for each model
    print("\n[*] Sending initial warm-up requests to trigger lazy weight allocation...")
    for mid in model_ids:
        ok, lat, err = send_inference_request(endpoint, mid, payloads[mid], timeout=60.0)
        status = "OK" if ok else f"FAILED: {err}"
        print(f"    - {mid:<16}: {status} ({lat:.1f}ms)")

    print("\n" + "#" * 72)
    print("  >>> BENCHMARK COMMENCING NOW <<<")
    print("  TIP: Switch to your terminal running `nvtop` to observe real-time")
    print("       GPU utilization, VRAM usage, and multi-model micro-batching!")
    print("#" * 72 + "\n")

    stop_event: Dict[str, bool] = {"stop": False}
    model_results: Dict[str, List[Tuple[bool, float, Optional[str]]]] = {mid: [] for mid in model_ids}

    total_threads = len(model_ids) * args.concurrency_per_model
    start_time = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=total_threads) as executor:
        futures = []
        for mid in model_ids:
            for _ in range(args.concurrency_per_model):
                f = executor.submit(
                    worker_loop,
                    endpoint,
                    mid,
                    payloads[mid],
                    args.duration,
                    stop_event,
                    model_results[mid],
                )
                futures.append(f)

        # Progress ticker
        deadline = start_time + args.duration
        try:
            while time.perf_counter() < deadline:
                elapsed = time.perf_counter() - start_time
                total_reqs = sum(len(res) for res in model_results.values())
                current_rps = total_reqs / max(elapsed, 0.001)
                sys.stdout.write(
                    f"\r  Running: {elapsed:.1f}s / {args.duration:.1f}s | "
                    f"Total Requests: {total_reqs} | Aggregate Throughput: {current_rps:.1f} req/s"
                )
                sys.stdout.flush()
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\n[!] Benchmark interrupted by user; collecting stats...")
            stop_event["stop"] = True

        concurrent.futures.wait(futures)

    actual_duration = time.perf_counter() - start_time
    print(f"\n\n{'=' * 72}")
    print(f"  BENCHMARK RESULTS (Elapsed: {actual_duration:.2f}s)")
    print(f"{'=' * 72}\n")

    header = f"{'Model':<16} | {'Reqs':>6} | {'Req/s':>8} | {'Errors':>6} | {'P50 (ms)':>9} | {'P95 (ms)':>9} | {'P99 (ms)':>9}"
    print(header)
    print("-" * len(header))

    total_all_reqs = 0
    total_all_errors = 0

    for mid in model_ids:
        res_list = model_results[mid]
        total = len(res_list)
        errors = sum(1 for ok, _, _ in res_list if not ok)
        latencies = sorted(lat for ok, lat, _ in res_list if ok)

        rps = total / max(actual_duration, 0.001)
        p50 = percentile(latencies, 50.0) if latencies else 0.0
        p95 = percentile(latencies, 95.0) if latencies else 0.0
        p99 = percentile(latencies, 99.0) if latencies else 0.0

        total_all_reqs += total
        total_all_errors += errors

        print(f"{mid:<16} | {total:>6} | {rps:>8.1f} | {errors:>6} | {p50:>9.2f} | {p95:>9.2f} | {p99:>9.2f}")

    print("-" * len(header))
    overall_rps = total_all_reqs / max(actual_duration, 0.001)
    print(f"{'OVERALL':<16} | {total_all_reqs:>6} | {overall_rps:>8.1f} | {total_all_errors:>6} | {'-':>9} | {'-':>9} | {'-':>9}")
    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
