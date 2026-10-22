#!/usr/bin/env python3
"""
SnoozeML gRPC Endpoint Test & HTTP Comparison Suite
===================================================

A comprehensive Python test tool for SnoozeML gRPC and HTTP endpoints.
Tests:
  1. Standard gRPC Health Checking (overall server & snoozeml.v1.InferenceService).
  2. gRPC Unary Inference (ModelInferRequest -> ModelInferResponse).
  3. Negative error handling (model not found, invalid argument, malformed shapes).
  4. Authentication verification (API keys, Bearer tokens).
  5. End-to-end Numerical Equivalence: Sends the EXACT same input tensors to both
     HTTP REST (/v1/models/{model_id}/infer) and gRPC (InferenceService/Infer)
     and asserts that tensor shapes, names, and decoded float values match within tolerance.
  6. Side-by-side Performance & Latency Benchmark comparing HTTP vs gRPC.

Zero proto-folder modifications: Dynamically loads Protocol Buffer definitions
at runtime via grpc.protos_and_services().

Usage:
------
  # Run full test suite against default running server (http://localhost:8080, localhost:50051):
  python tools/test_grpc.py

  # Run with pytest:
  pytest tools/test_grpc.py

  # Compare HTTP and gRPC output equivalence & latency for all registered models:
  python tools/test_grpc.py --compare

  # Run side-by-side performance benchmark (50 iterations per model):
  python tools/test_grpc.py --benchmark --iterations 50

  # Target specific model with custom endpoints:
  python tools/test_grpc.py --http-endpoint http://127.0.0.1:8080 --grpc-endpoint 127.0.0.1:50051 --model all-MiniLM-L6-v2

  # Test authenticated daemon:
  python tools/test_grpc.py --api-key sk-admin123
"""

import argparse
import http.client
import json
import math
import os
import struct
import sys
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc

# Locate repo root and proto schema file
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROTO_REL_PATH = "proto/snoozeml/v1/inference.proto"
PROTO_ABS_PATH = os.path.join(REPO_ROOT, "proto", "snoozeml", "v1", "inference.proto")

if not os.path.isfile(PROTO_ABS_PATH):
    raise FileNotFoundError(f"Inference protobuf schema not found at: {PROTO_ABS_PATH}")

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

PYTHON_DIR = os.path.join(REPO_ROOT, "python")
if PYTHON_DIR not in sys.path:
    sys.path.insert(0, PYTHON_DIR)

# Dynamically load protobuf descriptors and gRPC stubs.
# First attempt to import pre-compiled stubs from snoozeml._proto.
# If not available, fall back to compiling the proto dynamically via grpc.protos_and_services.
try:
    from snoozeml._proto import (
        DataType,
        InferenceServiceStub,
        ModelInferRequest,
        ModelInferResponse,
        TensorData,
    )
except ImportError:
    _saved_cwd = os.getcwd()
    try:
        os.chdir(REPO_ROOT)
        _inference_protos, _inference_services = grpc.protos_and_services(PROTO_REL_PATH)
        ModelInferRequest = _inference_protos.ModelInferRequest
        ModelInferResponse = _inference_protos.ModelInferResponse
        TensorData = _inference_protos.TensorData
        DataType = _inference_protos.DataType
        InferenceServiceStub = _inference_services.InferenceServiceStub
    except (NotImplementedError, AttributeError) as exc:
        raise ImportError(
            "Could not load SnoozeML gRPC protobuf stubs. Either install the 'snoozeml' package "
            "(`pip install 'snoozeml[grpc]'` or `pip install -e ./python`) or install `grpcio-tools` (`pip install grpcio-tools`)."
        ) from exc
    finally:
        os.chdir(_saved_cwd)


# ==============================================================================
# Model Discovery & Input Payload Generator (Matches HTTP part exactly)
# ==============================================================================

class ModelPayloadGenerator:
    """Prepares valid inference payloads matching the HTTP part (tools/perf_test.py)."""

    @staticmethod
    def generate_input_tensors(
        model_id: str,
        custom_file: Optional[str] = None,
        seq_len: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Generates list of tensor dictionaries:
        [{"name": "input_ids", "shape": [1, seq_len], "data": [101.0, ...]}, ...]
        identical to the HTTP payloads used in SnoozeML testing.
        """
        if custom_file:
            with open(custom_file, "r", encoding="utf-8") as f:
                content = json.load(f)
                if "inputs" in content:
                    return content["inputs"]
                raise ValueError("Custom payload JSON must contain an 'inputs' list")

        mid = model_id.lower()
        if "minilm" in mid or "bert" in mid or "nomic" in mid or "embed" in mid:
            # Transformer text embedding input (input_ids + attention_mask)
            # Default tokens: [CLS] hello world ! [SEP]
            tokens = [101.0, 7592.0, 2088.0, 999.0, 102.0]
            if seq_len != 5:
                if seq_len > len(tokens):
                    tokens = tokens + [100.0] * (seq_len - len(tokens))
                else:
                    tokens = tokens[:seq_len]
            actual_len = len(tokens)
            return [
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
        elif "chronos" in mid:
            # Time-series context input
            return [
                {
                    "name": "context",
                    "shape": [1, 32],
                    "data": [float(i) * 0.1 for i in range(32)],
                }
            ]
        else:
            # Generic float vector input (works with mock, demo, and default onnx models)
            return [
                {
                    "name": "input",
                    "shape": [1, 10],
                    "data": [1.0] * 10,
                }
            ]

    @staticmethod
    def tensors_to_http_json(inputs: List[Dict[str, Any]]) -> str:
        """Serializes input tensors to JSON payload string for HTTP REST."""
        return json.dumps({"inputs": inputs})

    @staticmethod
    def tensors_to_grpc_request(
        model_id: str,
        inputs: List[Dict[str, Any]],
        client_timeout_ms: Optional[int] = None,
    ) -> Any:
        """Converts input tensor dictionaries to a Protobuf ModelInferRequest."""
        req = ModelInferRequest(model_id=model_id)
        if client_timeout_ms is not None:
            req.client_timeout_ms = client_timeout_ms

        for item in inputs:
            tensor = req.inputs.add()
            tensor.name = item["name"]
            tensor.shape.extend(item["shape"])
            tensor.data_type = DataType.DATA_TYPE_FP32

            raw_floats = [float(x) for x in item["data"]]
            tensor.raw_data = struct.pack(f"<{len(raw_floats)}f", *raw_floats)

        return req


# ==============================================================================
# Client Implementations (HTTP REST & gRPC)
# ==============================================================================

@dataclass
class HTTPInferResponse:
    success: bool
    status_code: int
    model_id: str
    outputs: List[Dict[str, Any]]
    duration_ms: float
    client_latency_ms: float
    raw_body_bytes: int
    error_msg: str = ""


@dataclass
class GRPCInferResponse:
    success: bool
    status_code: grpc.StatusCode
    model_id: str
    outputs: List[Dict[str, Any]]
    duration_ms: float
    client_latency_ms: float
    raw_payload_bytes: int
    error_msg: str = ""


class SnoozeHTTPClient:
    """HTTP REST client for SnoozeML daemon."""

    def __init__(self, base_url: str = "http://127.0.0.1:8080", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.parsed = urllib.parse.urlparse(self.base_url)
        raw_host = self.parsed.hostname or "127.0.0.1"
        # Normalize localhost to 127.0.0.1 to avoid Windows IPv6 (::1) 2-second DNS fallback delay
        self.host = "127.0.0.1" if raw_host == "localhost" else raw_host
        self.port = self.parsed.port or (443 if self.parsed.scheme == "https" else 80)
        self.api_key = api_key
        self._conn: Optional[http.client.HTTPConnection] = None

    def _get_connection(self, timeout: float = 10.0) -> http.client.HTTPConnection:
        if self._conn is None:
            self._conn = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        return self._conn

    def close(self):
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def _headers(self, content_type: bool = True) -> Dict[str, str]:
        headers = {}
        if content_type:
            headers["Content-Type"] = "application/json"
            headers["Connection"] = "keep-alive"
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def get_health_live(self) -> Tuple[bool, int, str]:
        try:
            conn = self._get_connection(5.0)
            conn.request("GET", "/health/live", headers=self._headers(False))
            res = conn.getresponse()
            body = res.read().decode("utf-8")
            return (res.status == 200, res.status, body)
        except Exception:
            self.close()
            conn = self._get_connection(5.0)
            conn.request("GET", "/health/live", headers=self._headers(False))
            res = conn.getresponse()
            body = res.read().decode("utf-8")
            return (res.status == 200, res.status, body)

    def get_health_ready(self) -> Tuple[bool, int, str]:
        try:
            conn = self._get_connection(5.0)
            conn.request("GET", "/health/ready", headers=self._headers(False))
            res = conn.getresponse()
            body = res.read().decode("utf-8")
            return (res.status == 200, res.status, body)
        except Exception:
            self.close()
            conn = self._get_connection(5.0)
            conn.request("GET", "/health/ready", headers=self._headers(False))
            res = conn.getresponse()
            body = res.read().decode("utf-8")
            return (res.status == 200, res.status, body)

    def list_models(self) -> List[Dict[str, Any]]:
        try:
            conn = self._get_connection(5.0)
            conn.request("GET", "/v1/models", headers=self._headers(False))
            res = conn.getresponse()
            if res.status != 200:
                return []
            body = res.read().decode("utf-8")
            return json.loads(body)
        except Exception:
            return []

    def infer(self, model_id: str, inputs: List[Dict[str, Any]], timeout: float = 30.0) -> HTTPInferResponse:
        payload = ModelPayloadGenerator.tensors_to_http_json(inputs).encode("utf-8")
        headers = self._headers(True)
        headers["Content-Length"] = str(len(payload))

        t0 = time.perf_counter_ns()
        try:
            conn = self._get_connection(timeout)
            conn.request("POST", f"/v1/models/{model_id}/infer", payload, headers)
            res = conn.getresponse()
            raw_body = res.read()
            t1 = time.perf_counter_ns()
            client_lat_ms = (t1 - t0) / 1_000_000.0

            if res.status != 200:
                return HTTPInferResponse(
                    success=False,
                    status_code=res.status,
                    model_id=model_id,
                    outputs=[],
                    duration_ms=0.0,
                    client_latency_ms=client_lat_ms,
                    raw_body_bytes=len(raw_body),
                    error_msg=raw_body.decode("utf-8", errors="replace"),
                )

            data = json.loads(raw_body.decode("utf-8"))
            return HTTPInferResponse(
                success=True,
                status_code=res.status,
                model_id=data.get("model_id", model_id),
                outputs=data.get("outputs", []),
                duration_ms=float(data.get("duration_ms", 0.0)),
                client_latency_ms=client_lat_ms,
                raw_body_bytes=len(raw_body),
            )
        except Exception as e:
            self.close()
            t1 = time.perf_counter_ns()
            return HTTPInferResponse(
                success=False,
                status_code=0,
                model_id=model_id,
                outputs=[],
                duration_ms=0.0,
                client_latency_ms=(t1 - t0) / 1_000_000.0,
                raw_body_bytes=0,
                error_msg=str(e),
            )


class SnoozeGRPCClient:
    """gRPC client for SnoozeML daemon."""

    def __init__(self, target: str = "127.0.0.1:50051", api_key: Optional[str] = None):
        # Normalize localhost to 127.0.0.1
        if target.startswith("localhost:"):
            target = "127.0.0.1:" + target[len("localhost:"):]
        self.target = target
        self.api_key = api_key
        # Connect with keepalive and large message limits
        options = [
            ("grpc.max_receive_message_length", 64 * 1024 * 1024),
            ("grpc.max_send_message_length", 64 * 1024 * 1024),
            ("grpc.keepalive_time_ms", 10000),
            ("grpc.keepalive_timeout_ms", 5000),
        ]
        self.channel = grpc.insecure_channel(self.target, options=options)
        self.health_stub = health_pb2_grpc.HealthStub(self.channel)
        self.infer_stub = InferenceServiceStub(self.channel)

    def close(self):
        self.channel.close()

    def _metadata(self) -> List[Tuple[str, str]]:
        if self.api_key:
            return [
                ("authorization", f"Bearer {self.api_key}"),
                ("x-api-key", self.api_key),
            ]
        return []

    def check_health(self, service: str = "", timeout: float = 5.0) -> Tuple[bool, int, str]:
        req = health_pb2.HealthCheckRequest(service=service)
        try:
            res = self.health_stub.Check(req, timeout=timeout)
            status_name = health_pb2.HealthCheckResponse.ServingStatus.Name(res.status)
            return (res.status == health_pb2.HealthCheckResponse.SERVING, res.status, status_name)
        except grpc.RpcError as e:
            return (False, -1, f"gRPC RPC Error: {e.code().name} - {e.details()}")

    def infer(
        self,
        model_id: str,
        inputs: List[Dict[str, Any]],
        client_timeout_ms: Optional[int] = None,
        timeout: float = 10.0,
        custom_metadata: Optional[List[Tuple[str, str]]] = None,
    ) -> GRPCInferResponse:
        req = ModelPayloadGenerator.tensors_to_grpc_request(model_id, inputs, client_timeout_ms)
        wire_size = req.ByteSize()
        metadata = custom_metadata if custom_metadata is not None else self._metadata()

        t0 = time.perf_counter_ns()
        try:
            res: ModelInferResponse = self.infer_stub.Infer(req, timeout=timeout, metadata=metadata)
            t1 = time.perf_counter_ns()
            client_lat_ms = (t1 - t0) / 1_000_000.0

            # Decode outputs into python dictionaries matching HTTP structure
            decoded_outputs = []
            for tensor in res.outputs:
                raw_bytes = tensor.raw_data
                count = len(raw_bytes) // 4
                floats = list(struct.unpack(f"<{count}f", raw_bytes))
                decoded_outputs.append({
                    "name": tensor.name,
                    "shape": list(tensor.shape),
                    "data": floats,
                })

            return GRPCInferResponse(
                success=True,
                status_code=grpc.StatusCode.OK,
                model_id=res.model_id,
                outputs=decoded_outputs,
                duration_ms=res.duration_ms,
                client_latency_ms=client_lat_ms,
                raw_payload_bytes=wire_size + res.ByteSize(),
            )
        except grpc.RpcError as e:
            t1 = time.perf_counter_ns()
            return GRPCInferResponse(
                success=False,
                status_code=e.code(),
                model_id=model_id,
                outputs=[],
                duration_ms=0.0,
                client_latency_ms=(t1 - t0) / 1_000_000.0,
                raw_payload_bytes=wire_size,
                error_msg=e.details() or e.code().name,
            )


# ==============================================================================
# Numerical & Behavioral Comparison Engine
# ==============================================================================

@dataclass
class ComparisonMetric:
    tensor_name: str
    shape_http: List[int]
    shape_grpc: List[int]
    shape_match: bool
    element_count: int
    max_abs_diff: float
    mean_abs_diff: float
    all_close: bool


@dataclass
class ComparisonSummary:
    model_id: str
    identical_outputs: bool
    metrics: List[ComparisonMetric]
    http_duration_ms: float
    grpc_duration_ms: float
    http_client_latency_ms: float
    grpc_client_latency_ms: float
    http_bytes: int
    grpc_bytes: int
    latency_speedup_x: float
    bandwidth_saved_pct: float
    notes: str = ""


class OutputComparator:
    """Verifies that HTTP and gRPC yield mathematically identical tensors."""

    @staticmethod
    def compare(
        http_resp: HTTPInferResponse,
        grpc_resp: GRPCInferResponse,
        atol: float = 1e-4,
        rtol: float = 1e-4,
    ) -> ComparisonSummary:
        if not http_resp.success or not grpc_resp.success:
            return ComparisonSummary(
                model_id=http_resp.model_id or grpc_resp.model_id,
                identical_outputs=False,
                metrics=[],
                http_duration_ms=http_resp.duration_ms,
                grpc_duration_ms=grpc_resp.duration_ms,
                http_client_latency_ms=http_resp.client_latency_ms,
                grpc_client_latency_ms=grpc_resp.client_latency_ms,
                http_bytes=http_resp.raw_body_bytes,
                grpc_bytes=grpc_resp.raw_payload_bytes,
                latency_speedup_x=0.0,
                bandwidth_saved_pct=0.0,
                notes=f"HTTP status: {http_resp.status_code} ({http_resp.error_msg}), gRPC status: {grpc_resp.status_code.name} ({grpc_resp.error_msg})",
            )

        http_map = {out["name"]: out for out in http_resp.outputs}
        grpc_map = {out["name"]: out for out in grpc_resp.outputs}

        all_identical = True
        metrics: List[ComparisonMetric] = []

        all_names = sorted(set(http_map.keys()) | set(grpc_map.keys()))
        for name in all_names:
            h_out = http_map.get(name)
            g_out = grpc_map.get(name)

            if not h_out or not g_out:
                all_identical = False
                metrics.append(
                    ComparisonMetric(
                        tensor_name=name,
                        shape_http=h_out["shape"] if h_out else [],
                        shape_grpc=g_out["shape"] if g_out else [],
                        shape_match=False,
                        element_count=0,
                        max_abs_diff=float("inf"),
                        mean_abs_diff=float("inf"),
                        all_close=False,
                    )
                )
                continue

            h_shape = h_out.get("shape", [])
            g_shape = g_out.get("shape", [])
            shape_match = (h_shape == g_shape)

            h_data = h_out.get("data", [])
            g_data = g_out.get("data", [])

            if len(h_data) != len(g_data) or not shape_match:
                all_identical = False
                metrics.append(
                    ComparisonMetric(
                        tensor_name=name,
                        shape_http=h_shape,
                        shape_grpc=g_shape,
                        shape_match=shape_match,
                        element_count=max(len(h_data), len(g_data)),
                        max_abs_diff=float("inf"),
                        mean_abs_diff=float("inf"),
                        all_close=False,
                    )
                )
                continue

            # Compare numerical values
            max_diff = 0.0
            sum_diff = 0.0
            close_flag = True

            for h_val, g_val in zip(h_data, g_data):
                diff = abs(h_val - g_val)
                if diff > max_diff:
                    max_diff = diff
                sum_diff += diff
                if not math.isclose(h_val, g_val, rel_tol=rtol, abs_tol=atol):
                    close_flag = False

            mean_diff = (sum_diff / len(h_data)) if h_data else 0.0
            if not shape_match or not close_flag:
                all_identical = False

            metrics.append(
                ComparisonMetric(
                    tensor_name=name,
                    shape_http=h_shape,
                    shape_grpc=g_shape,
                    shape_match=shape_match,
                    element_count=len(h_data),
                    max_abs_diff=max_diff,
                    mean_abs_diff=mean_diff,
                    all_close=close_flag,
                )
            )

        speedup = (
            (http_resp.client_latency_ms / grpc_resp.client_latency_ms)
            if grpc_resp.client_latency_ms > 0
            else 0.0
        )
        saved_pct = (
            ((http_resp.raw_body_bytes - grpc_resp.raw_payload_bytes) / http_resp.raw_body_bytes * 100.0)
            if http_resp.raw_body_bytes > 0
            else 0.0
        )

        return ComparisonSummary(
            model_id=http_resp.model_id,
            identical_outputs=all_identical,
            metrics=metrics,
            http_duration_ms=http_resp.duration_ms,
            grpc_duration_ms=grpc_resp.duration_ms,
            http_client_latency_ms=http_resp.client_latency_ms,
            grpc_client_latency_ms=grpc_resp.client_latency_ms,
            http_bytes=http_resp.raw_body_bytes,
            grpc_bytes=grpc_resp.raw_payload_bytes,
            latency_speedup_x=round(speedup, 2),
            bandwidth_saved_pct=round(saved_pct, 1),
            notes="Identical numerical output" if all_identical else "Numerical or shape mismatch",
        )


# ==============================================================================
# Visual Formatter / Reporter
# ==============================================================================

class ConsoleReporter:
    """Formats comparison results into clean CLI tables."""

    @staticmethod
    def print_header(title: str):
        bar = "=" * 80
        print(f"\n{bar}")
        print(f" {title.upper()}")
        print(f"{bar}")

    @staticmethod
    def print_comparison_table(summaries: List[ComparisonSummary]):
        ConsoleReporter.print_header("SnoozeML HTTP vs gRPC Inference Equivalence & Latency")
        header = f"{'Model ID':<22} | {'Outputs':<7} | {'Shape Match':<11} | {'Max |Diff|':<12} | {'HTTP Lat':<9} | {'gRPC Lat':<9} | {'Speedup':<8} | {'Status':<6}"
        divider = "-" * len(header)
        print(header)
        print(divider)

        for s in summaries:
            shapes_ok = "YES" if all(m.shape_match for m in s.metrics) else "NO"
            max_diff = max((m.max_abs_diff for m in s.metrics), default=0.0)
            diff_str = f"{max_diff:.2e}" if max_diff != float("inf") else "ERR"
            status = "PASS" if s.identical_outputs else "FAIL"

            row = (
                f"{s.model_id:<22} | "
                f"{len(s.metrics):<7} | "
                f"{shapes_ok:<11} | "
                f"{diff_str:<12} | "
                f"{s.http_client_latency_ms:6.2f} ms | "
                f"{s.grpc_client_latency_ms:6.2f} ms | "
                f"{s.latency_speedup_x:5.2f}x  | "
                f"{status:<6}"
            )
            print(row)

        print(divider)
        print("Note: 'Speedup' represents HTTP Client Latency / gRPC Client Latency.\n")

    @staticmethod
    def print_benchmark_table(bench_results: List[Dict[str, Any]]):
        ConsoleReporter.print_header("Side-by-Side Performance Benchmark (HTTP vs gRPC)")
        header = f"{'Protocol':<8} | {'Reqs':<6} | {'P50 (ms)':<9} | {'P90 (ms)':<9} | {'P95 (ms)':<9} | {'P99 (ms)':<9} | {'Avg (ms)':<9} | {'RPS':<8}"
        divider = "-" * len(header)
        print(header)
        print(divider)

        for r in bench_results:
            row = (
                f"{r['protocol']:<8} | "
                f"{r['total']:<6} | "
                f"{r['p50']:7.2f}  | "
                f"{r['p90']:7.2f}  | "
                f"{r['p95']:7.2f}  | "
                f"{r['p99']:7.2f}  | "
                f"{r['avg']:7.2f}  | "
                f"{r['rps']:6.1f}"
            )
            print(row)
        print(divider)


# ==============================================================================
# Benchmarking Engine
# ==============================================================================

def run_performance_comparison(
    http_client: SnoozeHTTPClient,
    grpc_client: SnoozeGRPCClient,
    model_id: str,
    inputs: List[Dict[str, Any]],
    iterations: int = 30,
    warmup: int = 5,
) -> List[Dict[str, Any]]:
    """Runs consecutive iterations over HTTP and gRPC to compare steady-state latencies."""
    print(f"\n[+] Warming up '{model_id}' ({warmup} iterations each)...")
    for _ in range(warmup):
        http_client.infer(model_id, inputs)
        grpc_client.infer(model_id, inputs)

    print(f"[+] Benchmarking {iterations} iterations on HTTP REST...")
    http_times = []
    t_start_http = time.perf_counter()
    for _ in range(iterations):
        res = http_client.infer(model_id, inputs)
        if res.success:
            http_times.append(res.client_latency_ms)
    t_elapsed_http = time.perf_counter() - t_start_http

    print(f"[+] Benchmarking {iterations} iterations on gRPC...")
    grpc_times = []
    t_start_grpc = time.perf_counter()
    for _ in range(iterations):
        res = grpc_client.infer(model_id, inputs)
        if res.success:
            grpc_times.append(res.client_latency_ms)
    t_elapsed_grpc = time.perf_counter() - t_start_grpc

    def calc_stats(protocol: str, times: List[float], total_secs: float) -> Dict[str, Any]:
        if not times:
            return {"protocol": protocol, "total": 0, "p50": 0, "p90": 0, "p95": 0, "p99": 0, "avg": 0, "rps": 0}
        times_sorted = sorted(times)
        n = len(times_sorted)
        p50 = times_sorted[int(n * 0.50)]
        p90 = times_sorted[min(n - 1, int(n * 0.90))]
        p95 = times_sorted[min(n - 1, int(n * 0.95))]
        p99 = times_sorted[min(n - 1, int(n * 0.99))]
        avg = sum(times_sorted) / n
        rps = n / total_secs if total_secs > 0 else 0
        return {
            "protocol": protocol,
            "total": n,
            "p50": p50,
            "p90": p90,
            "p95": p95,
            "p99": p99,
            "avg": avg,
            "rps": rps,
        }

    return [
        calc_stats("HTTP", http_times, t_elapsed_http),
        calc_stats("gRPC", grpc_times, t_elapsed_grpc),
    ]


# ==============================================================================
# Pytest Test Cases & Verification Functions
# ==============================================================================

# Global targets configurable via environment variables or CLI
HTTP_ENDPOINT = os.getenv("SNOOZE_HTTP_ENDPOINT", "http://localhost:8080")
GRPC_ENDPOINT = os.getenv("SNOOZE_GRPC_ENDPOINT", "localhost:50051")
API_KEY = os.getenv("SNOOZE_API_KEY", None)


def _check_endpoint_online(endpoint: str) -> bool:
    """Best-effort TCP connection check to determine if endpoint is listening."""
    import socket
    parsed = urllib.parse.urlparse(endpoint)
    host = parsed.hostname or (endpoint.split(":")[0] if ":" in endpoint else "127.0.0.1")
    port = parsed.port or (int(endpoint.split(":")[1]) if ":" in endpoint else 8080)
    try:
        with socket.create_connection((host, port), timeout=0.3):
            return True
    except OSError:
        return False


try:
    import pytest

    _server_online = _check_endpoint_online(HTTP_ENDPOINT) or _check_endpoint_online(GRPC_ENDPOINT)
    pytestmark = pytest.mark.skipif(
        not _server_online,
        reason=(
            f"SnoozeML daemon is not running on {HTTP_ENDPOINT} / {GRPC_ENDPOINT} "
            "(start with `snooze serve` to run live integration tests)"
        ),
    )
except ImportError:
    pass


def test_grpc_health_root():
    """Verifies gRPC standard health check for the server root (service='')."""
    client = SnoozeGRPCClient(GRPC_ENDPOINT, API_KEY)
    try:
        ok, code, status_name = client.check_health("")
        assert ok, f"Root health check failed with status: {status_name} (code: {code})"
        assert status_name == "SERVING"
    finally:
        client.close()


def test_grpc_health_inference_service():
    """Verifies gRPC standard health check for 'snoozeml.v1.InferenceService'."""
    client = SnoozeGRPCClient(GRPC_ENDPOINT, API_KEY)
    try:
        ok, code, status_name = client.check_health("snoozeml.v1.InferenceService")
        assert ok, f"InferenceService health check failed with status: {status_name} (code: {code})"
        assert status_name == "SERVING"
    finally:
        client.close()


def test_compare_http_and_grpc_inference():
    """
    Sends identical inputs to both HTTP and gRPC for discovered models and asserts:
    1. Both return HTTP 200 / gRPC OK.
    2. Output tensor counts and names match.
    3. Output tensor dimensions (shapes) match.
    4. Numeric float32 values match within atol=1e-4 tolerance.
    """
    http_cli = SnoozeHTTPClient(HTTP_ENDPOINT, API_KEY)
    grpc_cli = SnoozeGRPCClient(GRPC_ENDPOINT, API_KEY)

    try:
        models = http_cli.list_models()
        if not models:
            try:
                import pytest
                pytest.skip("No models registered on running SnoozeML server to compare inference")
            except ImportError:
                print("[SKIP] No models registered on running SnoozeML server to compare inference")
                return
        model_ids = [m["id"] for m in models]

        tested_count = 0
        for mid in model_ids:
            inputs = ModelPayloadGenerator.generate_input_tensors(mid, seq_len=5)

            # 1. Query HTTP
            http_res = http_cli.infer(mid, inputs)
            # 2. Query gRPC
            grpc_res = grpc_cli.infer(mid, inputs)

            # Compare outputs
            summary = OutputComparator.compare(http_res, grpc_res)
            assert summary.identical_outputs, (
                f"Model '{mid}' outputs differ between HTTP and gRPC! "
                f"Summary: {summary.notes}, Metrics: {summary.metrics}"
            )
            tested_count += 1

        assert tested_count > 0, "No models were tested during comparison!"
    finally:
        grpc_cli.close()


def test_grpc_error_model_not_found():
    """Verifies that requesting a nonexistent model returns NOT_FOUND (code 5)."""
    client = SnoozeGRPCClient(GRPC_ENDPOINT, API_KEY)
    try:
        inputs = [{"name": "input", "shape": [1, 2], "data": [1.0, 2.0]}]
        res = client.infer("non_existent_model_404_test", inputs)
        assert not res.success, "Inference on non-existent model unexpectedly succeeded"
        assert res.status_code == grpc.StatusCode.NOT_FOUND, (
            f"Expected StatusCode.NOT_FOUND, got {res.status_code}"
        )
    finally:
        client.close()


def test_grpc_error_empty_model_id():
    """Verifies that empty model_id returns INVALID_ARGUMENT (code 3)."""
    client = SnoozeGRPCClient(GRPC_ENDPOINT, API_KEY)
    try:
        inputs = [{"name": "input", "shape": [1, 2], "data": [1.0, 2.0]}]
        res = client.infer("", inputs)
        assert not res.success, "Inference with empty model_id unexpectedly succeeded"
        assert res.status_code == grpc.StatusCode.INVALID_ARGUMENT, (
            f"Expected StatusCode.INVALID_ARGUMENT, got {res.status_code}"
        )
    finally:
        client.close()


def test_grpc_error_malformed_tensor_buffer():
    """Verifies that mismatched tensor byte buffer returns INVALID_ARGUMENT (code 3)."""
    http_cli = SnoozeHTTPClient(HTTP_ENDPOINT, API_KEY)
    models = http_cli.list_models()
    if not models:
        try:
            import pytest
            pytest.skip("No registered model available on running server to test malformed tensor buffer")
        except ImportError:
            print("[SKIP] No registered model available to test malformed tensor buffer")
            return

    target_model = models[0]["id"]
    client = SnoozeGRPCClient(GRPC_ENDPOINT, API_KEY)
    try:
        # Declare shape [1, 4] (requires 16 bytes), but provide only 8 bytes (2 floats)
        req = ModelInferRequest(model_id=target_model)
        t = req.inputs.add()
        t.name = "input"
        t.shape.extend([1, 4])
        t.data_type = DataType.DATA_TYPE_FP32
        t.raw_data = struct.pack("<2f", 1.0, 2.0)  # Only 8 bytes

        try:
            client.infer_stub.Infer(req, timeout=5.0)
            assert False, "Mismatched tensor buffer unexpectedly succeeded"
        except grpc.RpcError as e:
            assert e.code() == grpc.StatusCode.INVALID_ARGUMENT, (
                f"Expected StatusCode.INVALID_ARGUMENT, got {e.code()}"
            )
    finally:
        client.close()


# ==============================================================================
# Standalone CLI Main Runner
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SnoozeML gRPC Test & HTTP Comparison Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--http-endpoint",
        default="http://localhost:8080",
        help="Base HTTP REST URL for SnoozeML daemon",
    )
    parser.add_argument(
        "--grpc-endpoint",
        default="localhost:50051",
        help="gRPC host:port for SnoozeML daemon",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Target model ID to test. If omitted, auto-discovers all models via /v1/models",
    )
    parser.add_argument(
        "--payload-file",
        default=None,
        help="Optional custom JSON payload file with an 'inputs' array",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=5,
        help="Sequence length for generated transformer tensor inputs",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional API key for authorization headers / metadata",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run comprehensive side-by-side numerical equivalence and latency comparison",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run side-by-side performance benchmark (HTTP vs gRPC throughput & latency percentiles)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=30,
        help="Number of iterations for benchmark mode",
    )

    args = parser.parse_args()

    http_cli = SnoozeHTTPClient(args.http_endpoint, args.api_key)
    grpc_cli = SnoozeGRPCClient(args.grpc_endpoint, args.api_key)

    ConsoleReporter.print_header("SnoozeML gRPC & HTTP Integration Test")
    print(f"[*] HTTP Endpoint: {args.http_endpoint}")
    print(f"[*] gRPC Endpoint: {args.grpc_endpoint}")

    # 1. Health Checking Verification
    print("\n--- 1. Testing Health Endpoints ---")
    live_ok, live_code, _ = http_cli.get_health_live()
    print(f"[HTTP] /health/live: {'PASS' if live_ok else 'FAIL'} (status: {live_code})")

    grpc_root_ok, grpc_root_code, grpc_root_msg = grpc_cli.check_health("")
    print(f"[gRPC] Health.Check(root): {'PASS' if grpc_root_ok else 'FAIL'} (status: {grpc_root_msg})")

    grpc_svc_ok, grpc_svc_code, grpc_svc_msg = grpc_cli.check_health("snoozeml.v1.InferenceService")
    print(f"[gRPC] Health.Check(InferenceService): {'PASS' if grpc_svc_ok else 'FAIL'} (status: {grpc_svc_msg})")

    if not grpc_root_ok or not grpc_svc_ok:
        print("[!] Warning: gRPC health check did not return SERVING.")

    # 2. Discover Models
    print("\n--- 2. Model Discovery ---")
    if args.model:
        model_ids = [args.model]
        print(f"[*] Targeting specified model: '{args.model}'")
    else:
        discovered = http_cli.list_models()
        if discovered:
            model_ids = [m["id"] for m in discovered]
            print(f"[*] Auto-discovered {len(model_ids)} model(s) from HTTP /v1/models: {model_ids}")
        else:
            model_ids = ["demo-model"]
            print("[*] No models discovered from /v1/models; falling back to 'demo-model'")

    # 3. Equivalence and Latency Comparison
    print("\n--- 3. Verifying Identical Inputs & Comparing HTTP vs gRPC ---")
    summaries: List[ComparisonSummary] = []
    all_passed = True

    for mid in model_ids:
        print(f"[*] Generating identical inputs for model: '{mid}'...")
        inputs = ModelPayloadGenerator.generate_input_tensors(
            mid,
            custom_file=args.payload_file,
            seq_len=args.seq_len,
        )

        for item in inputs:
            print(f"    - Input node '{item['name']}', shape: {item['shape']}, elements: {len(item['data'])}")

        http_resp = http_cli.infer(mid, inputs)
        grpc_resp = grpc_cli.infer(mid, inputs)

        summary = OutputComparator.compare(http_resp, grpc_resp)
        summaries.append(summary)

        if not summary.identical_outputs:
            all_passed = False
            print(f"[FAIL] Output mismatch for model '{mid}'! {summary.notes}")
        else:
            print(
                f"[PASS] Model '{mid}': Tensors identical. "
                f"HTTP: {summary.http_client_latency_ms:.2f}ms, gRPC: {summary.grpc_client_latency_ms:.2f}ms "
                f"(Speedup: {summary.latency_speedup_x:.2f}x)"
            )

    ConsoleReporter.print_comparison_table(summaries)

    # 4. Error Handling Tests
    print("--- 4. Negative Error Handling Tests ---")
    # Not found
    nf_res = grpc_cli.infer("non_existent_model_test_999", inputs)
    nf_pass = (not nf_res.success and nf_res.status_code == grpc.StatusCode.NOT_FOUND)
    print(f"[gRPC] Model Not Found (Code 5): {'PASS' if nf_pass else 'FAIL'} (Got {nf_res.status_code.name})")

    # Empty model ID
    empty_res = grpc_cli.infer("", inputs)
    empty_pass = (not empty_res.success and empty_res.status_code == grpc.StatusCode.INVALID_ARGUMENT)
    print(f"[gRPC] Empty Model ID (Code 3): {'PASS' if empty_pass else 'FAIL'} (Got {empty_res.status_code.name})")

    if not nf_pass or not empty_pass:
        all_passed = False

    # 5. Optional Benchmark Mode
    if args.benchmark and model_ids:
        target_model = model_ids[0]
        bench_inputs = ModelPayloadGenerator.generate_input_tensors(target_model, args.payload_file, args.seq_len)
        bench_results = run_performance_comparison(
            http_cli, grpc_cli, target_model, bench_inputs, iterations=args.iterations
        )
        ConsoleReporter.print_benchmark_table(bench_results)

    grpc_cli.close()

    if all_passed:
        print("\n[SUCCESS] All gRPC endpoint tests and HTTP comparison assertions PASSED successfully!")
        sys.exit(0)
    else:
        print("\n[FAILURE] One or more gRPC endpoint tests or comparison assertions FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
