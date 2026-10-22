"""Integration and behavioral tests for SnoozeML Python SDK extensions.

Exercises JWT minting/verification, Prometheus metrics parsing, event streaming
envelopes and broker clients, programmatic models.toml configuration & worker
scaffolding, dataset batch chunking, transient 503 retries, wait_until_ready,
and model upload overwrite against an ephemeral local HTTP server.
"""

from __future__ import annotations

import asyncio
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import tempfile
import threading
from typing import Any, Dict, List, Tuple
import unittest

import snoozeml
from snoozeml import (
    AsyncSnoozeClient,
    AuthenticationError,
    DaemonMetrics,
    GlobalDefaultsConfig,
    InvalidInputError,
    JwtAlgorithm,
    MockStreamClient,
    ModelNotFoundError,
    RedisStreamClient,
    Role,
    RuntimeInferenceError,
    SnoozeClient,
    SnoozeConfig,
    SnoozeError,
    StreamInferRequest,
    StreamInferResponse,
    StreamWireFormat,
    TensorInput,
    TensorOutput,
    concat_inference_responses,
    create_jwt_token,
    decode_jwt_token,
    mint_jwt_token,
    parse_prometheus_metrics,
    scaffold_worker,
    slice_inputs_batch,
)


class _EphemeralSnoozeHandler(BaseHTTPRequestHandler):
    """Stateful HTTP handler implementing SnoozeML REST endpoints for testing."""

    server_state: Dict[str, Any] = {}

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def _send_json(self, status: int, payload: Any, headers: Dict[str, str] | None = None) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        if headers:
            for k, v in headers.items():
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        state = self.server_state
        if self.path == "/health/ready":
            ready_calls = state.get("ready_calls", 0) + 1
            state["ready_calls"] = ready_calls
            if ready_calls < state.get("ready_after", 1):
                self._send_json(503, {"status": "starting"})
            else:
                self._send_json(200, {"status": "ready"})
            return

        if self.path == "/v1/models":
            models = list(state.get("models", {}).values())
            self._send_json(200, models)
            return

        if self.path == "/metrics":
            prom_text = (
                "# HELP snoozeml_requests_total Total inference requests processed\n"
                "# TYPE snoozeml_requests_total counter\n"
                'snoozeml_requests_total{status="all"} 42\n'
                'snoozeml_requests_total{status="success"} 40\n'
                'snoozeml_requests_total{status="error"} 2\n'
                "snoozeml_inference_latency_seconds_total 0.084\n"
                "snoozeml_batches_dispatched_total 20\n"
                'snoozeml_model_deallocations_total{reason="idle"} 2\n'
                'snoozeml_model_deallocations_total{reason="capacity"} 1\n'
                'snoozeml_model_deallocations_total{reason="oom"} 0\n'
                "snoozeml_model_load_retries_total 1\n"
                "snoozeml_deallocation_thrashing_total 0\n"
                "snoozeml_stream_messages_consumed_total 15\n"
                "snoozeml_stream_messages_published_total 14\n"
                "snoozeml_stream_errors_total 1\n"
            )
            raw = prom_text.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return

        self._send_json(404, {"error": f"Unknown path {self.path}"})

    def do_POST(self) -> None:
        state = self.server_state
        content_len = int(self.headers.get("Content-Length", "0"))
        body_bytes = self.rfile.read(content_len) if content_len > 0 else b""

        if self.path == "/v1/models/upload":
            body_text = body_bytes.decode("latin-1", errors="ignore")
            model_id = "uploaded_model"
            if 'name="model_id"' in body_text:
                after = body_text.split('name="model_id"', 1)[1]
                lines = [line.strip() for line in after.splitlines() if line.strip()]
                if lines:
                    model_id = lines[0]

            overwrite = False
            if 'name="overwrite"' in body_text:
                after_ow = body_text.split('name="overwrite"', 1)[1]
                lines_ow = [line.strip() for line in after_ow.splitlines() if line.strip()]
                if lines_ow and lines_ow[0].lower() in ("true", "1", "yes"):
                    overwrite = True

            models = state.setdefault("models", {})
            if model_id in models and not overwrite:
                self._send_json(409, {"error": f"Model '{model_id}' already exists"})
                return

            models[model_id] = {
                "id": model_id,
                "path": f"./models/{model_id}/model.onnx",
                "state": "Idle",
                "backend": "onnx",
            }
            self._send_json(
                201,
                {
                    "id": model_id,
                    "model_id": model_id,
                    "path": f"./models/{model_id}/model.onnx",
                    "state": "Idle",
                    "backend": "onnx",
                },
            )
            return

        if self.path.startswith("/v1/models/") and self.path.endswith("/load"):
            model_id = self.path.split("/")[3]
            models = state.setdefault("models", {})
            if model_id not in models:
                self._send_json(404, {"error": f"Model '{model_id}' not found"})
                return
            models[model_id]["state"] = "Ready"
            self._send_json(200, {"status": "loaded", "model_id": model_id, "state": "Ready"})
            return

        if self.path.startswith("/v1/models/") and self.path.endswith("/infer"):
            model_id = self.path.split("/")[3]
            transient_left = state.get("transient_503_remaining", 0)
            if transient_left > 0:
                state["transient_503_remaining"] = transient_left - 1
                self._send_json(
                    503,
                    {"error": f"Model '{model_id}' is currently loading"},
                    headers={"Retry-After": "0.01"},
                )
                return

            payload = json.loads(body_bytes.decode("utf-8"))
            inputs = payload.get("inputs", [])
            outputs = []
            for inp in inputs:
                shape = inp["shape"]
                doubled = [float(x) * 2.0 for x in inp["data"]]
                outputs.append(
                    {
                        "name": "output" if inp["name"] == "input" else f"{inp['name']}_out",
                        "shape": shape,
                        "data": doubled,
                    }
                )
            self._send_json(
                200,
                {
                    "model_id": model_id,
                    "outputs": outputs,
                    "duration_ms": 1.25,
                },
            )
            return

        self._send_json(404, {"error": f"Unknown POST path {self.path}"})

    def do_DELETE(self) -> None:
        state = self.server_state
        if self.path.startswith("/v1/models/"):
            model_id = self.path.split("/")[3]
            models = state.setdefault("models", {})
            if model_id not in models:
                self._send_json(404, {"error": f"Model '{model_id}' not found"})
                return
            del models[model_id]
            self._send_json(200, {"status": "deleted", "model_id": model_id})
            return
        self._send_json(404, {"error": "Not found"})


class _FakeRedisClient:
    """In-memory Redis Streams store supporting XADD and XREAD for RedisStreamClient testing."""

    def __init__(self) -> None:
        self.streams: Dict[str, List[Tuple[bytes, Dict[bytes, bytes]]]] = {}
        self._seq = 0

    def xadd(self, stream: str, fields: Dict[str, bytes]) -> bytes:
        self._seq += 1
        entry_id = f"{self._seq}-0".encode("utf-8")
        encoded_fields = {
            (k.encode("utf-8") if isinstance(k, str) else k): (
                v.encode("utf-8") if isinstance(v, str) else bytes(v)
            )
            for k, v in fields.items()
        }
        self.streams.setdefault(stream, []).append((entry_id, encoded_fields))
        return entry_id

    def xread(
        self,
        streams: Dict[str, str],
        count: int = 16,
        block: int = 100,
    ) -> List[Tuple[bytes, List[Tuple[bytes, Dict[bytes, bytes]]]]]:
        result: List[Tuple[bytes, List[Tuple[bytes, Dict[bytes, bytes]]]]] = []
        for stream_name, last_id in streams.items():
            entries = self.streams.get(stream_name, [])
            last_seq = int(str(last_id).split("-")[0]) if last_id and "-" in str(last_id) else 0
            matched = []
            for entry_id, fields in entries:
                seq = int(entry_id.decode("utf-8").split("-")[0])
                if seq > last_seq:
                    matched.append((entry_id, fields))
            if matched:
                result.append((stream_name.encode("utf-8"), matched[:count]))
        return result


class TestSnoozeSdkExtensions(unittest.TestCase):
    """Tests for JWT minting, metrics, streams, config/worker builder, and client workflows."""

    def test_jwt_minting_and_verification(self) -> None:
        secret = "top-secret-hmac-signing-key-2026"
        token = create_jwt_token(
            secret=secret,
            role=Role.ADMIN,
            subject="ops-pipeline",
            expires_in_secs=300,
            issuer="snooze-auth",
            audience="snooze-cluster",
            algorithm=JwtAlgorithm.HS256,
            extra_claims={"team": "ml-infra"},
        )
        self.assertIsInstance(token, str)
        self.assertEqual(token.count("."), 2)

        claims = decode_jwt_token(
            token,
            secret=secret,
            algorithm="HS256",
            issuer="snooze-auth",
            audience="snooze-cluster",
        )
        self.assertEqual(claims["sub"], "ops-pipeline")
        self.assertEqual(claims["role"], "admin")
        self.assertEqual(claims["iss"], "snooze-auth")
        self.assertEqual(claims["aud"], "snooze-cluster")
        self.assertEqual(claims["team"], "ml-infra")

        # Alias mint_jwt_token with HS384 and HS512
        token_512 = mint_jwt_token(secret=secret, role="inference", algorithm="HS512")
        claims_512 = decode_jwt_token(token_512, secret=secret, algorithm="HS512")
        self.assertEqual(claims_512["role"], "inference")

        # Invalid signature must raise AuthenticationError
        with self.assertRaises(AuthenticationError):
            decode_jwt_token(token, secret="wrong-secret")

        # Expired token must raise AuthenticationError
        expired_tok = create_jwt_token(
            secret=secret, role="read_only", iat=1000, exp=1010
        )
        with self.assertRaises(AuthenticationError):
            decode_jwt_token(expired_tok, secret=secret, leeway_secs=0)

    def test_prometheus_metrics_parser(self) -> None:
        raw_text = (
            "# HELP snoozeml_requests_total Total inference requests\n"
            "# TYPE snoozeml_requests_total counter\n"
            'snoozeml_requests_total{status="all"} 100\n'
            'snoozeml_requests_total{status="success"} 95\n'
            'snoozeml_requests_total{status="error"} 5\n'
            "snoozeml_inference_latency_seconds_total 0.25\n"
            "snoozeml_batches_dispatched_total 40\n"
            'snoozeml_model_deallocations_total{reason="idle"} 4\n'
            'snoozeml_model_deallocations_total{reason="capacity"} 2\n'
            'snoozeml_model_deallocations_total{reason="oom"} 1\n'
            "snoozeml_model_load_retries_total 1\n"
            "snoozeml_deallocation_thrashing_total 0\n"
            "snoozeml_stream_messages_consumed_total 50\n"
            "snoozeml_stream_messages_published_total 49\n"
            "snoozeml_stream_errors_total 1\n"
        )
        metrics = parse_prometheus_metrics(raw_text)
        self.assertEqual(metrics.requests_total, 100)
        self.assertEqual(metrics.requests_success, 95)
        self.assertEqual(metrics.requests_error, 5)
        self.assertAlmostEqual(metrics.average_latency_ms, 2.5)
        self.assertAlmostEqual(metrics.success_rate, 0.95)
        self.assertEqual(metrics.total_deallocations, 7)
        self.assertEqual(metrics.stream_messages_consumed_total, 50)

    def test_stream_envelopes_and_mock_stream_client(self) -> None:
        req = StreamInferRequest.from_inputs(
            {"input": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]},
            model_id="fraud_xgb",
            id="corr-123",
            reply_to="custom.out",
        )
        raw_json = req.to_bytes(StreamWireFormat.JSON)
        decoded_req = StreamInferRequest.from_bytes(raw_json, StreamWireFormat.JSON)
        self.assertEqual(decoded_req.id, "corr-123")
        self.assertEqual(decoded_req.model_id, "fraud_xgb")
        self.assertEqual(decoded_req.reply_to, "custom.out")
        self.assertEqual(decoded_req.inputs[0].shape, [2, 3])

        def _double_handler(model_id: str, inputs: List[TensorInput]) -> List[TensorOutput]:
            out_list = []
            for t in inputs:
                doubled = [[float(v) * 2.0 for v in row] for row in t.data.tolist()]
                out_list.append(
                    TensorOutput.from_rest_dict(
                        {"name": "output", "shape": t.shape, "data": [x for r in doubled for x in r]}
                    )
                )
            return out_list

        stream_client = MockStreamClient(
            handler=_double_handler,
            default_model_id="fraud_xgb",
        )
        infer_resp = stream_client.infer({"input": [[1.5, 2.5], [3.5, 4.5]]})
        self.assertEqual(infer_resp.model_id, "fraud_xgb")
        self.assertEqual(infer_resp.get_tensor("output").shape, [2, 2])
        self.assertEqual(infer_resp.get_tensor("output").to_list(), [3.0, 5.0, 7.0, 9.0])

    def test_redis_stream_client_roundtrip(self) -> None:
        fake_redis = _FakeRedisClient()
        redis_stream = RedisStreamClient(
            input_topic="snooze.in",
            output_topic="snooze.out",
            default_model_id="redis_model",
            redis_client=fake_redis,
        )
        corr_id, entry_id = redis_stream.publish_request({"input": [[10.0, 20.0]]})
        self.assertTrue(entry_id.endswith("-0"))

        # Simulate SnoozeML daemon consuming snooze.in and writing to snooze.out
        in_entries = fake_redis.streams["snooze.in"]
        self.assertEqual(len(in_entries), 1)
        raw_req = in_entries[0][1][b"payload"]
        parsed_req = StreamInferRequest.from_bytes(raw_req)
        self.assertEqual(parsed_req.id, corr_id)

        out_env = StreamInferResponse(
            id=corr_id,
            model_id="redis_model",
            status="success",
            outputs=[
                TensorOutput.from_rest_dict(
                    {"name": "output", "shape": [1, 2], "data": [100.0, 200.0]}
                )
            ],
            duration_ms=0.8,
        )
        fake_redis.xadd("snooze.out", {"payload": out_env.to_bytes()})

        resp = redis_stream.wait_for_response(corr_id, timeout=1.0)
        self.assertTrue(resp.is_success)
        self.assertEqual(resp.get_tensor("output").to_list(), [100.0, 200.0])

    def test_config_builder_and_worker_scaffolding(self) -> None:
        cfg = (
            SnoozeConfig(defaults=GlobalDefaultsConfig(max_loaded_models=8))
            .add_model(
                "bge_small",
                "./models/bge/model.onnx",
                device="cpu",
                max_batch_size=32,
                max_batch_delay_ms=5,
                idle_timeout_secs=120,
                pinned=True,
            )
            .add_stream(
                name="redis-pipeline",
                broker="redis",
                endpoint="redis://127.0.0.1:6379",
                input_topic="snooze.in",
                model_id="bge_small",
                output_topic="snooze.out",
                error_topic="snooze.dlq",
                consumer_group="workers",
                format="json",
            )
            .add_api_key(role="admin", key="admin-key-12345", name="ci-admin")
            .set_jwt(secret="jwt-secret-xyz", algorithm="HS256", issuer="snooze")
        )

        toml_str = cfg.to_toml()
        self.assertIn("max_loaded_models = 8", toml_str)
        self.assertIn('id = "bge_small"', toml_str)
        self.assertIn("[[streams]]", toml_str)
        self.assertIn('broker = "redis"', toml_str)
        self.assertIn("[[auth.api_keys]]", toml_str)
        self.assertIn("[auth.jwt]", toml_str)

        roundtrip = SnoozeConfig.from_dict(json.loads(cfg.to_json()))
        self.assertEqual(roundtrip.defaults.max_loaded_models, 8)
        self.assertEqual(len(roundtrip.models), 1)
        self.assertEqual(roundtrip.models[0].id, "bge_small")
        self.assertEqual(len(roundtrip.streams), 1)
        self.assertEqual(roundtrip.streams[0].broker, "redis")

        with tempfile.TemporaryDirectory() as tmpdir:
            worker_dir = scaffold_worker(
                Path(tmpdir) / "echo_worker",
                tensor_format="raw",
                verify=True,
            )
            self.assertTrue((worker_dir / "snooze_model.py").is_file())
            self.assertTrue((worker_dir / "snooze_model.json").is_file())
            self.assertTrue((worker_dir / "requirements.txt").is_file())

    def test_batch_slicing_and_client_features(self) -> None:
        # 1. Test slice_inputs_batch and concat_inference_responses
        dataset = {"input": [[float(i), float(i + 1)] for i in range(5)]}
        chunks = slice_inputs_batch(dataset, batch_size=2)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0][0].shape, [2, 2])
        self.assertEqual(chunks[1][0].shape, [2, 2])
        self.assertEqual(chunks[2][0].shape, [1, 2])

        # 2. Spin up ephemeral HTTP server for SnoozeClient & AsyncSnoozeClient tests
        _EphemeralSnoozeHandler.server_state = {
            "ready_calls": 0,
            "ready_after": 2,
            "transient_503_remaining": 2,
            "models": {},
        }
        server = HTTPServer(("127.0.0.1", 0), _EphemeralSnoozeHandler)
        port = server.server_port
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        try:
            url = f"http://127.0.0.1:{port}"
            client = SnoozeClient(url, max_retries=3, retry_backoff=0.01)

            # wait_until_ready polls until ready_after (2nd call) returns 200
            self.assertTrue(client.wait_until_ready(timeout=2.0, poll_interval=0.01))

            # get_metrics parses Prometheus counters
            metrics = client.get_metrics()
            self.assertIsInstance(metrics, DaemonMetrics)
            self.assertEqual(metrics.requests_total, 42)
            self.assertAlmostEqual(metrics.average_latency_ms, 2.0)

            # upload_model with overwrite=False then overwrite=True
            with tempfile.TemporaryDirectory() as tmpdir:
                dummy_onnx = Path(tmpdir) / "demo.onnx"
                dummy_onnx.write_bytes(b"ONNX_DUMMY_WEIGHTS")

                res1 = client.upload_model(dummy_onnx, model_id="demo_net")
                self.assertEqual(res1.id, "demo_net")

                # Second upload without overwrite raises SnoozeError (409)
                with self.assertRaises(SnoozeError) as ctx:
                    client.upload_model(dummy_onnx, model_id="demo_net", overwrite=False)
                self.assertEqual(ctx.exception.status_code, 409)

                # Second upload with overwrite=True deletes & re-uploads cleanly
                res2 = client.upload_model(dummy_onnx, model_id="demo_net", overwrite=True)
                self.assertEqual(res2.id, "demo_net")

            # wait_for_ready with trigger_load=True transitions model to Ready
            info = client.models.wait_for_ready("demo_net", timeout=2.0, trigger_load=True)
            self.assertTrue(info.is_loaded)

            # infer automatically retries transient 503 responses
            resp = client.infer("demo_net", {"input": [[1.0, 2.0, 3.0]]})
            self.assertEqual(resp.get_tensor("output").to_list(), [2.0, 4.0, 6.0])

            # infer_dataset chunks 5 rows into batches of 2 and concatenates outputs
            full_resp = client.infer_dataset(
                "demo_net",
                {"input": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]},
                batch_size=2,
            )
            self.assertEqual(full_resp.get_tensor("output").shape, [5, 2])
            self.assertEqual(
                full_resp.get_tensor("output").to_list(),
                [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
            )

            # Test AsyncSnoozeClient infer_dataset and get_metrics
            async def _run_async_checks() -> None:
                async with AsyncSnoozeClient(url, max_retries=2, retry_backoff=0.01) as aclient:
                    ametrics = await aclient.get_metrics()
                    self.assertEqual(ametrics.requests_total, 42)
                    aresp = await aclient.infer_dataset(
                        "demo_net",
                        {"input": [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]},
                        batch_size=2,
                        concurrency=2,
                    )
                    self.assertEqual(aresp.get_tensor("output").shape, [3, 2])
                    self.assertEqual(
                        aresp.get_tensor("output").to_list(),
                        [2.0, 2.0, 4.0, 4.0, 6.0, 6.0],
                    )

            asyncio.run(_run_async_checks())
            client.close()
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
