#!/usr/bin/env python3
"""
SnoozeML Docker Performance & Capacity Testing Tool
===================================================

A standalone, zero-dependency development tool for SnoozeML.
Automates load testing, capacity hunter (concurrency ramp-up to find saturation point),
real-time Docker container resource tracking (CPU %, Memory RSS/Limit, Net/Block I/O),
and scale-to-zero / memory reclamation verification.

Usage Examples:
---------------
  # 1. Run full suite (Ramp-up + Fixed load + Scale-to-zero)
  python tools/perf_test.py --mode all

  # 2. Concurrency ramp-up to find maximum requests/sec and breaking point
  python tools/perf_test.py --mode ramp --concurrency-steps 1,2,4,8,16,32,64 --step-duration 5

  # 3. Fixed sustained load benchmark
  python tools/perf_test.py --mode fixed --concurrency 16 --duration 20

  # 4. Scale-to-zero & memory deallocation verification
  python tools/perf_test.py --mode scale-to-zero

  # 5. Export structured metrics to JSON for CI/CD comparisons
  python tools/perf_test.py --mode ramp --output-json results.json
"""

import argparse
import concurrent.futures
import http.client
import json
import os
import re
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.parse
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ==============================================================================
# Helper Utilities
# ==============================================================================

def parse_bytes(s: str) -> float:
    """Parses human-readable byte strings (e.g. '113.7MiB', '7.6GiB', '4.2kB') into bytes."""
    s = s.strip()
    if not s:
        return 0.0
    units = {
        "b": 1.0,
        "k": 1000.0,
        "kb": 1000.0,
        "kib": 1024.0,
        "m": 1000.0**2,
        "mb": 1000.0**2,
        "mib": 1024.0**2,
        "g": 1000.0**3,
        "gb": 1000.0**3,
        "gib": 1024.0**3,
        "t": 1000.0**4,
        "tb": 1000.0**4,
        "tib": 1024.0**4,
    }
    s_lower = s.lower()
    for unit, multiplier in sorted(units.items(), key=lambda x: -len(x[0])):
        if s_lower.endswith(unit):
            try:
                num = float(s[:-len(unit)].strip())
                return num * multiplier
            except ValueError:
                return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def format_bytes(bytes_val: float) -> str:
    """Formats raw byte count into human-friendly representation."""
    abs_val = abs(bytes_val)
    if abs_val >= 1024.0**3:
        return f"{bytes_val / (1024.0**3):.2f} GiB"
    elif abs_val >= 1024.0**2:
        return f"{bytes_val / (1024.0**2):.2f} MiB"
    elif abs_val >= 1024.0:
        return f"{bytes_val / 1024.0:.2f} KiB"
    else:
        return f"{bytes_val:.0f} B"


def parse_cpu_perc(s: str) -> float:
    """Parses percentage string (e.g. '12.34%') into float."""
    s = s.strip().rstrip("%")
    try:
        return float(s)
    except ValueError:
        return 0.0


# ==============================================================================
# Docker Container Resource Monitor
# ==============================================================================

@dataclass
class DockerStatsSample:
    timestamp: float
    cpu_perc: float
    mem_used_bytes: float
    mem_limit_bytes: float
    mem_perc: float
    net_rx_bytes: float
    net_tx_bytes: float
    pids: int


class DockerMonitor:
    """Asynchronous background sampler for Docker container resource statistics."""

    def __init__(self, container_name: str, sample_interval: float = 0.5):
        self.container_name = container_name
        self.sample_interval = sample_interval
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.samples: List[DockerStatsSample] = []
        self.is_available = False
        self._lock = threading.Lock()

    def check_container_running(self) -> Tuple[bool, str]:
        """Checks whether the specified Docker container exists and is running."""
        try:
            res = subprocess.run(
                ["docker", "inspect", "--format", "{{json .State}}", self.container_name],
                capture_output=True,
                text=True,
                check=False,
            )
            if res.returncode != 0:
                return False, f"Container '{self.container_name}' not found. Exit code {res.returncode}."
            state = json.loads(res.stdout)
            running = state.get("Running", False)
            status = state.get("Status", "unknown")
            if not running:
                return False, f"Container '{self.container_name}' is not running (status: {status})."
            self.is_available = True
            return True, f"Container '{self.container_name}' is running."
        except FileNotFoundError:
            return False, "Docker CLI executable not found in PATH."
        except Exception as e:
            return False, f"Failed to inspect Docker container: {e}"

    def start(self):
        """Starts background sampling thread."""
        if not self.is_available:
            return
        self._stop_event.clear()
        self.samples = []
        self._thread = threading.Thread(target=self._sampling_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stops background sampling thread."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def sample_now(self) -> Optional[DockerStatsSample]:
        """Performs a single immediate sample of docker stats."""
        if not self.is_available:
            return None
        try:
            res = subprocess.run(
                ["docker", "stats", "--no-stream", "--format", "{{json .}}", self.container_name],
                capture_output=True,
                text=True,
                check=False,
                timeout=3.0,
            )
            if res.returncode != 0:
                return None
            data = json.loads(res.stdout)
            cpu = parse_cpu_perc(data.get("CPUPerc", "0%"))
            mem_raw = data.get("MemUsage", "")
            mem_used, mem_limit = 0.0, 0.0
            if "/" in mem_raw:
                parts = mem_raw.split("/")
                if len(parts) == 2:
                    mem_used = parse_bytes(parts[0])
                    mem_limit = parse_bytes(parts[1])

            mem_perc = parse_cpu_perc(data.get("MemPerc", "0%"))
            net_raw = data.get("NetIO", "")
            net_rx, net_tx = 0.0, 0.0
            if "/" in net_raw:
                parts = net_raw.split("/")
                if len(parts) == 2:
                    net_rx = parse_bytes(parts[0])
                    net_tx = parse_bytes(parts[1])

            try:
                pids = int(data.get("PIDs", "0"))
            except ValueError:
                pids = 0

            return DockerStatsSample(
                timestamp=time.time(),
                cpu_perc=cpu,
                mem_used_bytes=mem_used,
                mem_limit_bytes=mem_limit,
                mem_perc=mem_perc,
                net_rx_bytes=net_rx,
                net_tx_bytes=net_tx,
                pids=pids,
            )
        except Exception:
            return None

    def _sampling_loop(self):
        while not self._stop_event.is_set():
            sample = self.sample_now()
            if sample:
                with self._lock:
                    self.samples.append(sample)
            self._stop_event.wait(self.sample_interval)

    def get_summary(self) -> Dict[str, Any]:
        """Calculates aggregate stats across collected samples."""
        with self._lock:
            if not self.samples:
                return {
                    "peak_cpu_perc": 0.0,
                    "avg_cpu_perc": 0.0,
                    "baseline_mem_bytes": 0.0,
                    "peak_mem_bytes": 0.0,
                    "final_mem_bytes": 0.0,
                    "mem_delta_bytes": 0.0,
                    "net_rx_bytes": 0.0,
                    "net_tx_bytes": 0.0,
                    "sample_count": 0,
                }
            cpu_vals = [s.cpu_perc for s in self.samples]
            mem_vals = [s.mem_used_bytes for s in self.samples]
            first_mem = mem_vals[0] if mem_vals else 0.0
            last_mem = mem_vals[-1] if mem_vals else 0.0
            peak_mem = max(mem_vals) if mem_vals else 0.0
            peak_cpu = max(cpu_vals) if cpu_vals else 0.0
            avg_cpu = statistics.mean(cpu_vals) if cpu_vals else 0.0
            first_net_rx = self.samples[0].net_rx_bytes
            last_net_rx = self.samples[-1].net_rx_bytes
            first_net_tx = self.samples[0].net_tx_bytes
            last_net_tx = self.samples[-1].net_tx_bytes

            return {
                "peak_cpu_perc": round(peak_cpu, 2),
                "avg_cpu_perc": round(avg_cpu, 2),
                "baseline_mem_bytes": first_mem,
                "peak_mem_bytes": peak_mem,
                "final_mem_bytes": last_mem,
                "mem_delta_bytes": peak_mem - first_mem,
                "net_rx_bytes": max(0.0, last_net_rx - first_net_rx),
                "net_tx_bytes": max(0.0, last_net_tx - first_net_tx),
                "sample_count": len(self.samples),
            }


# ==============================================================================
# Model Discovery & Input Payload Generator
# ==============================================================================

class PayloadManager:
    """Prepares valid inference payloads tailored to registered models."""

    @staticmethod
    def discover_models(base_url: str, api_key: Optional[str] = None) -> List[Dict[str, Any]]:
        """Queries /v1/models on the target daemon."""
        parsed = urllib.parse.urlparse(base_url)
        conn = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=5)
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        try:
            conn.request("GET", "/v1/models", headers=headers)
            res = conn.getresponse()
            if res.status != 200:
                return []
            body = res.read().decode("utf-8")
            return json.loads(body)
        except Exception:
            return []
        finally:
            conn.close()

    @staticmethod
    def generate_payload(model_id: str, custom_file: Optional[str] = None, seq_len: int = 5) -> bytes:
        """Generates JSON payload bytes for inference."""
        if custom_file:
            with open(custom_file, "r", encoding="utf-8") as f:
                content = f.read()
                # Verify valid json
                json.loads(content)
                return content.encode("utf-8")

        # Heuristic based on model name
        mid = model_id.lower()
        if "minilm" in mid or "bert" in mid or "nomic" in mid or "embed" in mid:
            # Transformer text embedding input (input_ids + attention_mask)
            # Default tokens: [CLS] hello world ! [SEP]
            tokens = [101.0, 7592.0, 2088.0, 999.0, 102.0]
            if seq_len != 5:
                tokens = tokens + [100.0] * (seq_len - len(tokens)) if seq_len > len(tokens) else tokens[:seq_len]
            actual_len = len(tokens)
            payload = {
                "inputs": [
                    {
                        "name": "input_ids",
                        "shape": [1, actual_len],
                        "data": tokens,
                    },
                    {
                        "name": "attention_mask",
                        "shape": [1, actual_len],
                        "data": [1.0] * actual_len,
                    },
                ]
            }
        elif "chronos" in mid:
            # Time-series context input
            payload = {
                "inputs": [
                    {
                        "name": "context",
                        "shape": [1, 32],
                        "data": [float(i) * 0.1 for i in range(32)],
                    }
                ]
            }
        else:
            # Generic float vector input (works with mock and default onnx models)
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


# ==============================================================================
# High-Throughput Load Testing Engine
# ==============================================================================

@dataclass
class RequestResult:
    success: bool
    status_code: int
    duration_ms: float
    error_msg: str = ""


@dataclass
class StepBenchmarkResult:
    concurrency: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    error_rate_perc: float
    elapsed_secs: float
    requests_per_sec: float
    latency_min_ms: float
    latency_avg_ms: float
    latency_p50_ms: float
    latency_p90_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_max_ms: float
    status_distribution: Dict[int, int]
    docker_peak_cpu: float = 0.0
    docker_peak_mem_bytes: float = 0.0
    docker_mem_delta_bytes: float = 0.0


class LoadTester:
    """Executes concurrent requests with persistent HTTP keep-alive connection reuse."""

    def __init__(
        self,
        base_url: str,
        path: str,
        payload_bytes: bytes,
        api_key: Optional[str] = None,
        timeout_secs: float = 10.0,
    ):
        self.parsed = urllib.parse.urlparse(base_url)
        self.host = self.parsed.hostname or "127.0.0.1"
        self.port = self.parsed.port or (443 if self.parsed.scheme == "https" else 80)
        self.path = path
        self.payload_bytes = payload_bytes
        self.api_key = api_key
        self.timeout_secs = timeout_secs

    def _worker_loop(
        self,
        stop_event: threading.Event,
        max_requests: Optional[int],
        req_counter: List[int],
        counter_lock: threading.Lock,
        results_list: List[RequestResult],
        list_lock: threading.Lock,
    ):
        conn: Optional[http.client.HTTPConnection] = None
        headers = {
            "Content-Type": "application/json",
            "Connection": "keep-alive",
            "Content-Length": str(len(self.payload_bytes)),
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        local_results: List[RequestResult] = []

        try:
            while not stop_event.is_set():
                if max_requests is not None:
                    with counter_lock:
                        if req_counter[0] >= max_requests:
                            break
                        req_counter[0] += 1

                if conn is None:
                    conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout_secs)

                t0 = time.perf_counter_ns()
                try:
                    conn.request("POST", self.path, self.payload_bytes, headers)
                    resp = conn.getresponse()
                    _ = resp.read()
                    t1 = time.perf_counter_ns()
                    dur_ms = (t1 - t0) / 1_000_000.0
                    local_results.append(
                        RequestResult(
                            success=(200 <= resp.status < 300),
                            status_code=resp.status,
                            duration_ms=dur_ms,
                            error_msg="" if (200 <= resp.status < 300) else f"HTTP {resp.status}",
                        )
                    )
                except Exception as e:
                    t1 = time.perf_counter_ns()
                    dur_ms = (t1 - t0) / 1_000_000.0
                    local_results.append(
                        RequestResult(
                            success=False,
                            status_code=0,
                            duration_ms=dur_ms,
                            error_msg=str(e),
                        )
                    )
                    try:
                        conn.close()
                    except Exception:
                        pass
                    conn = None
        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass
            with list_lock:
                results_list.extend(local_results)

    def run_benchmark(
        self,
        concurrency: int,
        duration_secs: float,
        max_requests: Optional[int] = None,
        docker_monitor: Optional[DockerMonitor] = None,
    ) -> StepBenchmarkResult:
        """Runs a fixed-concurrency benchmark run for a given duration or max requests."""
        stop_event = threading.Event()
        req_counter = [0]
        counter_lock = threading.Lock()
        results: List[RequestResult] = []
        list_lock = threading.Lock()

        # Start docker monitoring for this step
        if docker_monitor:
            docker_monitor.start()

        start_time = time.perf_counter()
        threads: List[threading.Thread] = []

        for _ in range(concurrency):
            t = threading.Thread(
                target=self._worker_loop,
                args=(stop_event, max_requests, req_counter, counter_lock, results, list_lock),
                daemon=True,
            )
            threads.append(t)
            t.start()

        # Wait duration or completion
        if max_requests is None:
            time.sleep(duration_secs)
            stop_event.set()
        else:
            deadline = start_time + duration_secs
            while time.perf_counter() < deadline:
                with counter_lock:
                    if req_counter[0] >= max_requests:
                        break
                time.sleep(0.05)
            stop_event.set()

        for t in threads:
            t.join(timeout=3.0)

        elapsed = time.perf_counter() - start_time

        docker_stats = {}
        if docker_monitor:
            docker_monitor.stop()
            docker_stats = docker_monitor.get_summary()

        # Calculate statistics
        total = len(results)
        successful = sum(1 for r in results if r.success)
        failed = total - successful
        err_rate = (failed / total * 100.0) if total > 0 else 0.0
        rps = (total / elapsed) if elapsed > 0 else 0.0

        latencies = sorted(r.duration_ms for r in results) if results else [0.0]
        status_dist: Dict[int, int] = {}
        for r in results:
            status_dist[r.status_code] = status_dist.get(r.status_code, 0) + 1

        def percentile(p: float) -> float:
            if not latencies:
                return 0.0
            idx = int(round(p * (len(latencies) - 1)))
            return latencies[min(idx, len(latencies) - 1)]

        return StepBenchmarkResult(
            concurrency=concurrency,
            total_requests=total,
            successful_requests=successful,
            failed_requests=failed,
            error_rate_perc=round(err_rate, 2),
            elapsed_secs=round(elapsed, 3),
            requests_per_sec=round(rps, 2),
            latency_min_ms=round(latencies[0], 2),
            latency_avg_ms=round(statistics.mean(latencies), 2),
            latency_p50_ms=round(percentile(0.50), 2),
            latency_p90_ms=round(percentile(0.90), 2),
            latency_p95_ms=round(percentile(0.95), 2),
            latency_p99_ms=round(percentile(0.99), 2),
            latency_max_ms=round(latencies[-1], 2),
            status_distribution=status_dist,
            docker_peak_cpu=docker_stats.get("peak_cpu_perc", 0.0),
            docker_peak_mem_bytes=docker_stats.get("peak_mem_bytes", 0.0),
            docker_mem_delta_bytes=docker_stats.get("mem_delta_bytes", 0.0),
        )


# ==============================================================================
# High-Level Test Runners
# ==============================================================================

def print_banner(title: str):
    width = 78
    print("\n" + "=" * width)
    print(f" {title.upper()}")
    print("=" * width)


def print_step_table_header():
    print("-" * 100)
    print(
        f"{'Concurrency':>11} | {'Requests':>8} | {'Req/s (RPS)':>11} | {'Avg (ms)':>9} | "
        f"{'P50 (ms)':>9} | {'P95 (ms)':>9} | {'P99 (ms)':>9} | {'Error %':>7} | {'Docker CPU%':>11} | {'Docker RAM':>11}"
    )
    print("-" * 100)


def print_step_row(step: StepBenchmarkResult):
    ram_str = format_bytes(step.docker_peak_mem_bytes) if step.docker_peak_mem_bytes > 0 else "N/A"
    cpu_str = f"{step.docker_peak_cpu:.1f}%" if step.docker_peak_cpu > 0 else "N/A"
    print(
        f"{step.concurrency:>11} | {step.total_requests:>8} | {step.requests_per_sec:>11.1f} | "
        f"{step.latency_avg_ms:>9.1f} | {step.latency_p50_ms:>9.1f} | {step.latency_p95_ms:>9.1f} | "
        f"{step.latency_p99_ms:>9.1f} | {step.error_rate_perc:>6.1f}% | {cpu_str:>11} | {ram_str:>11}"
    )


def run_ramp_test(
    tester: LoadTester,
    docker_monitor: Optional[DockerMonitor],
    concurrency_steps: List[int],
    step_duration: float,
    error_threshold: float,
    latency_threshold_ms: float,
) -> Tuple[List[StepBenchmarkResult], Dict[str, Any]]:
    """Runs concurrency ramp-up steps to detect the saturation point / breaking point."""
    print_banner(f"Running Capacity Hunter / Concurrency Ramp-Up ({len(concurrency_steps)} Steps)")
    print(f"Target steps: {concurrency_steps}")
    print(f"Step duration: {step_duration}s per concurrency level")
    print(f"Breaking criteria: Error rate > {error_threshold}% or Latency > {latency_threshold_ms}ms\n")

    print_step_table_header()

    step_results: List[StepBenchmarkResult] = []
    breaking_point: Optional[Dict[str, Any]] = None
    peak_rps = 0.0
    optimal_concurrency = concurrency_steps[0] if concurrency_steps else 1

    for c in concurrency_steps:
        res = tester.run_benchmark(
            concurrency=c,
            duration_secs=step_duration,
            docker_monitor=docker_monitor,
        )
        step_results.append(res)
        print_step_row(res)

        if res.requests_per_sec > peak_rps and res.error_rate_perc < error_threshold:
            peak_rps = res.requests_per_sec
            optimal_concurrency = c

        # Saturation check
        if res.error_rate_perc >= error_threshold:
            breaking_point = {
                "concurrency": c,
                "reason": f"Error rate ({res.error_rate_perc:.1f}%) exceeded threshold ({error_threshold}%)",
                "step_result": asdict(res),
            }
            print(f"\n[!] BREAKING POINT DETECTED at Concurrency {c}: {breaking_point['reason']}")
            break

        if res.latency_avg_ms >= latency_threshold_ms:
            breaking_point = {
                "concurrency": c,
                "reason": f"Average latency ({res.latency_avg_ms:.1f}ms) exceeded threshold ({latency_threshold_ms}ms)",
                "step_result": asdict(res),
            }
            print(f"\n[!] SATURATION POINT DETECTED at Concurrency {c}: {breaking_point['reason']}")
            break

    print("-" * 100)

    findings = {
        "peak_requests_per_sec": peak_rps,
        "optimal_concurrency": optimal_concurrency,
        "breaking_point": breaking_point,
    }

    print("\nCapacity Summary:")
    print(f"  * Peak Sustainable Throughput : {peak_rps:.1f} requests/sec")
    print(f"  * Optimal Concurrency Level   : {optimal_concurrency} workers")
    if breaking_point:
        print(f"  * System Saturation / Limit   : Reached at {breaking_point['concurrency']} concurrent requests")
        print(f"  * Saturation Reason           : {breaking_point['reason']}")
    else:
        print(f"  * System Saturation / Limit   : All tested concurrency levels ({concurrency_steps}) passed within thresholds!")

    return step_results, findings


def run_fixed_benchmark(
    tester: LoadTester,
    docker_monitor: Optional[DockerMonitor],
    concurrency: int,
    duration: float,
    max_requests: Optional[int],
) -> StepBenchmarkResult:
    """Runs a sustained fixed-concurrency benchmark."""
    print_banner(f"Running Sustained Fixed-Load Benchmark (Concurrency={concurrency}, Duration={duration}s)")

    print_step_table_header()
    res = tester.run_benchmark(
        concurrency=concurrency,
        duration_secs=duration,
        max_requests=max_requests,
        docker_monitor=docker_monitor,
    )
    print_step_row(res)
    print("-" * 100)

    print("\nDetailed Latency & Status Breakdown:")
    print(f"  * Total Requests  : {res.total_requests} (Success: {res.successful_requests}, Failed: {res.failed_requests})")
    print(f"  * Throughput      : {res.requests_per_sec:.2f} RPS")
    print(f"  * Min Latency     : {res.latency_min_ms:.2f} ms")
    print(f"  * Avg Latency     : {res.latency_avg_ms:.2f} ms")
    print(f"  * Median (P50)    : {res.latency_p50_ms:.2f} ms")
    print(f"  * 90th Pct (P90)  : {res.latency_p90_ms:.2f} ms")
    print(f"  * 95th Pct (P95)  : {res.latency_p95_ms:.2f} ms")
    print(f"  * 99th Pct (P99)  : {res.latency_p99_ms:.2f} ms")
    print(f"  * Max Latency     : {res.latency_max_ms:.2f} ms")
    print(f"  * HTTP Statuses   : {res.status_distribution}")
    if res.docker_peak_mem_bytes > 0:
        print(f"  * Peak Docker RAM : {format_bytes(res.docker_peak_mem_bytes)} (Peak CPU: {res.docker_peak_cpu:.1f}%)")

    return res


def run_scale_to_zero_test(
    base_url: str,
    model_id: str,
    tester: LoadTester,
    docker_monitor: Optional[DockerMonitor],
    api_key: Optional[str] = None,
    cooldown_secs: float = 3.0,
) -> Dict[str, Any]:
    """Tests cold loading, active inference, explicit unloading, and memory reclamation in Docker."""
    print_banner(f"Running Scale-To-Zero & Memory Reclamation Verification for '{model_id}'")

    if not docker_monitor or not docker_monitor.is_available:
        print("[!] Docker monitor unavailable; cannot measure memory reclamation directly.")
        return {"status": "skipped", "reason": "Docker unavailable"}

    # 1. Measure initial idle baseline memory
    print("[1/5] Unloading model to ensure clean starting state...")
    parsed = urllib.parse.urlparse(base_url)
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=5)
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        conn.request("POST", f"/v1/models/{model_id}/unload", headers=headers)
        conn.getresponse().read()
    except Exception:
        pass
    finally:
        conn.close()

    time.sleep(cooldown_secs)
    baseline_sample = docker_monitor.sample_now()
    baseline_mem = baseline_sample.mem_used_bytes if baseline_sample else 0.0
    print(f"      Baseline Idle Memory : {format_bytes(baseline_mem)}")

    # 2. Trigger cold load and inference
    print(f"[2/5] Sending inference requests to trigger model cold-load...")
    _ = tester.run_benchmark(concurrency=2, duration_secs=4.0, max_requests=10)
    time.sleep(1.0)

    loaded_sample = docker_monitor.sample_now()
    loaded_mem = loaded_sample.mem_used_bytes if loaded_sample else baseline_mem
    mem_allocated = loaded_mem - baseline_mem
    print(f"      Loaded Model Memory  : {format_bytes(loaded_mem)} (+{format_bytes(mem_allocated)} allocated)")

    # 3. Explicitly trigger unload endpoint
    print(f"[3/5] Triggering scale-to-zero deallocation via POST /v1/models/{model_id}/unload...")
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=5)
    try:
        conn.request("POST", f"/v1/models/{model_id}/unload", headers=headers)
        res = conn.getresponse()
        resp_data = res.read().decode("utf-8")
        print(f"      Unload HTTP Status   : {res.status} ({resp_data.strip()})")
    except Exception as e:
        print(f"      Unload failed: {e}")
    finally:
        conn.close()

    # 4. Wait cooldown and sample reclaimed memory
    print(f"[4/5] Monitoring memory reclamation cooldown ({cooldown_secs}s)...")
    time.sleep(cooldown_secs)
    final_sample = docker_monitor.sample_now()
    final_mem = final_sample.mem_used_bytes if final_sample else loaded_mem
    reclaimed = loaded_mem - final_mem
    reclaimed_perc = (reclaimed / mem_allocated * 100.0) if mem_allocated > 0 else 0.0

    print(f"      Final Idle Memory    : {format_bytes(final_mem)}")
    print(f"      Reclaimed Memory     : {format_bytes(reclaimed)} ({reclaimed_perc:.1f}% of allocated weights/queues)")

    # 5. Evaluate result
    passed = reclaimed > 0 or final_mem <= baseline_mem * 1.15
    print("\n[5/5] Verification Result:")
    if passed:
        print(f"      [PASS] Scale-to-zero memory reclamation confirmed successfully! Reclaimed {format_bytes(reclaimed)}.")
    else:
        print(f"      [WARN] Memory after unload ({format_bytes(final_mem)}) did not drop as expected.")

    return {
        "status": "passed" if passed else "warning",
        "model_id": model_id,
        "baseline_mem_bytes": baseline_mem,
        "loaded_mem_bytes": loaded_mem,
        "final_mem_bytes": final_mem,
        "reclaimed_bytes": reclaimed,
        "reclaimed_percentage": round(reclaimed_perc, 2),
    }


# ==============================================================================
# Main CLI Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SnoozeML Performance & Capacity Load Testing Tool with Docker Resource Monitoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--endpoint",
        default="http://localhost:8080",
        help="SnoozeML HTTP server base endpoint (default: http://localhost:8080)",
    )
    parser.add_argument(
        "--container",
        default="snoozeml",
        help="Docker container name to monitor (default: snoozeml)",
    )
    parser.add_argument(
        "--no-docker",
        action="store_true",
        help="Disable Docker container statistics tracking",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model ID to benchmark (default: auto-detected from /v1/models)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional API key for authorization header (Bearer <key>)",
    )
    parser.add_argument(
        "--mode",
        choices=["ramp", "fixed", "scale-to-zero", "all"],
        default="ramp",
        help="Benchmark mode: 'ramp' (concurrency ramp-up to find max capacity), 'fixed' (sustained load), 'scale-to-zero' (memory reclamation), or 'all' (all tests)",
    )
    parser.add_argument(
        "--concurrency-steps",
        default="1,2,4,8,16,32,64",
        help="Comma-separated list of concurrency levels for ramp test (default: 1,2,4,8,16,32,64)",
    )
    parser.add_argument(
        "--step-duration",
        type=float,
        default=5.0,
        help="Duration in seconds per concurrency step in ramp mode (default: 5.0)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=16,
        help="Concurrency level for fixed load mode (default: 16)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=15.0,
        help="Duration in seconds for fixed load mode (default: 15.0)",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=None,
        help="Max requests for fixed load mode (optional; stops when reached)",
    )
    parser.add_argument(
        "--error-threshold",
        type=float,
        default=5.0,
        help="Error rate percentage to mark breaking point in ramp mode (default: 5.0)",
    )
    parser.add_argument(
        "--latency-threshold-ms",
        type=float,
        default=2000.0,
        help="Average latency threshold in ms to flag saturation (default: 2000.0)",
    )
    parser.add_argument(
        "--payload-file",
        default=None,
        help="Optional path to custom JSON file containing InferRequest payload",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=5,
        help="Sequence length for auto-generated BERT tensor inputs (default: 5)",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to output complete benchmark results as JSON",
    )
    parser.add_argument(
        "--html-report",
        "--report",
        dest="html_report",
        default=None,
        help="Optional path to output interactive HTML visual dashboard (e.g. report.html)",
    )

    args = parser.parse_args()

    print_banner("SnoozeML Performance & Capacity Test Suite")
    print(f"Target Endpoint : {args.endpoint}")
    print(f"Docker Container: {args.container if not args.no_docker else 'Disabled (--no-docker)'}")

    # 1. Check Docker container
    docker_monitor = None
    if not args.no_docker:
        docker_monitor = DockerMonitor(args.container)
        ok, msg = docker_monitor.check_container_running()
        if ok:
            print(f"Docker Status   : [OK] {msg}")
        else:
            print(f"Docker Status   : [WARN] {msg}")
            print("                  Continuing in HTTP-only performance mode without container stats.")
            docker_monitor.is_available = False

    # 2. Discover models if not specified
    model_id = args.model
    if not model_id:
        print("\nDiscovering registered models from server...")
        models = PayloadManager.discover_models(args.endpoint, args.api_key)
        if models:
            print(f"Found {len(models)} registered models:")
            for m in models:
                desc = m.get("description") or "auto"
                print(f"  * {m['id']:<24} (state: {m.get('state', 'unknown'):<8}, path: {m.get('onnx_path', '')})")
            # Prefer all-MiniLM-L6-v2 or the first available model
            model_id = "all-MiniLM-L6-v2" if any(m['id'] == "all-MiniLM-L6-v2" for m in models) else models[0]['id']
            print(f"\nAuto-selected target model: '{model_id}'")
        else:
            print("[WARN] Could not retrieve models from /v1/models (daemon may be starting or unauthenticated).")
            model_id = "all-MiniLM-L6-v2"
            print(f"Defaulting to target model: '{model_id}'")

    # 3. Prepare payload
    try:
        payload_bytes = PayloadManager.generate_payload(
            model_id=model_id,
            custom_file=args.payload_file,
            seq_len=args.seq_len,
        )
    except Exception as e:
        print(f"[ERROR] Failed to prepare request payload: {e}")
        sys.exit(1)

    infer_path = f"/v1/models/{model_id}/infer"
    tester = LoadTester(
        base_url=args.endpoint,
        path=infer_path,
        payload_bytes=payload_bytes,
        api_key=args.api_key,
    )

    report_data: Dict[str, Any] = {
        "timestamp": time.time(),
        "endpoint": args.endpoint,
        "container": args.container if (docker_monitor and docker_monitor.is_available) else None,
        "model_id": model_id,
        "mode": args.mode,
    }

    # 4. Execute selected modes
    try:
        if args.mode in ("ramp", "all"):
            try:
                c_steps = [int(x.strip()) for x in args.concurrency_steps.split(",") if x.strip()]
            except ValueError:
                c_steps = [1, 2, 4, 8, 16, 32, 64]
            ramp_results, findings = run_ramp_test(
                tester=tester,
                docker_monitor=docker_monitor,
                concurrency_steps=c_steps,
                step_duration=args.step_duration,
                error_threshold=args.error_threshold,
                latency_threshold_ms=args.latency_threshold_ms,
            )
            report_data["ramp_test"] = {
                "steps": [asdict(r) for r in ramp_results],
                "findings": findings,
            }

        if args.mode in ("fixed", "all"):
            fixed_c = args.concurrency
            if args.mode == "all" and "ramp_test" in report_data and args.concurrency == 16:
                opt_c = report_data["ramp_test"]["findings"].get("optimal_concurrency")
                if opt_c:
                    fixed_c = opt_c
                    print(f"\n[*] 'all' mode: automatically testing sustained load at discovered optimal concurrency ({fixed_c} workers)...")
            fixed_res = run_fixed_benchmark(
                tester=tester,
                docker_monitor=docker_monitor,
                concurrency=fixed_c,
                duration=args.duration,
                max_requests=args.requests,
            )
            report_data["fixed_test"] = asdict(fixed_res)

        if args.mode in ("scale-to-zero", "all"):
            s2z_res = run_scale_to_zero_test(
                base_url=args.endpoint,
                model_id=model_id,
                tester=tester,
                docker_monitor=docker_monitor,
                api_key=args.api_key,
            )
            report_data["scale_to_zero_test"] = s2z_res

    except KeyboardInterrupt:
        print("\n[!] Benchmark interrupted by user.")

    # 5. Export JSON if requested
    if args.output_json:
        try:
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=2)
            print(f"\n[+] Full performance results saved to: {args.output_json}")
        except Exception as e:
            print(f"\n[!] Failed to save JSON report: {e}")

    # 6. Export HTML report if requested
    if args.html_report:
        try:
            tools_dir = os.path.dirname(os.path.abspath(__file__))
            if tools_dir not in sys.path:
                sys.path.insert(0, tools_dir)
            from report_generator import generate_html_report
            generate_html_report(report_data, args.html_report)
        except ImportError:
            print("\n[!] 'plotly' is required to generate HTML visual reports.")
            print("    Please install dependencies: pip install -r tools/requirements.txt")
        except Exception as e:
            print(f"\n[!] Failed to generate HTML report: {e}")

    print("\nBenchmark completed successfully.")


if __name__ == "__main__":
    main()
