"""Structured Prometheus metrics parser for the SnoozeML inference daemon.

Parses the text exposition output of ``GET /metrics`` (``src/server/metrics.rs``)
into a typed :class:`DaemonMetrics` object for programmatic monitoring, alerting,
and notebook telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict


_METRIC_LINE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<labels>[^}]*)\})?\s+"
    r"(?P<value>[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)$"
)
_LABEL_PAIR_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*"([^"]*)"')


@dataclass
class DaemonMetrics:
    """Structured snapshot of SnoozeML operational counters parsed from ``GET /metrics``.

    Attributes:
        requests_total: Total inference requests received across all statuses.
        requests_success: Total successful inference completions.
        requests_error: Total failed inference requests.
        inference_latency_seconds_total: Cumulative server-side inference latency in seconds.
        batches_dispatched_total: Total dynamic micro-batches executed by the batcher.
        oom_deallocations_total: Total reactive model unloads triggered by OOM recovery.
        idle_deallocations_total: Total scale-to-zero unloads triggered by the idle sweeper.
        capacity_deallocations_total: Total LRU unloads triggered by ``max_loaded_models`` caps.
        model_load_retries_total: Total allocation retry attempts after OOM eviction.
        deallocation_thrashing_total: Total rapid re-deallocation thrashing events detected.
        stream_messages_consumed_total: Total messages ingested from streaming brokers.
        stream_messages_published_total: Total messages published to streaming brokers.
        stream_errors_total: Total errors encountered during streaming ingestion or egress.
        raw_metrics: Dictionary of all parsed metric keys (with normalized labels) to float values.
    """

    requests_total: int = 0
    requests_success: int = 0
    requests_error: int = 0
    inference_latency_seconds_total: float = 0.0
    batches_dispatched_total: int = 0
    oom_deallocations_total: int = 0
    idle_deallocations_total: int = 0
    capacity_deallocations_total: int = 0
    model_load_retries_total: int = 0
    deallocation_thrashing_total: int = 0
    stream_messages_consumed_total: int = 0
    stream_messages_published_total: int = 0
    stream_errors_total: int = 0
    raw_metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def average_latency_ms(self) -> float:
        """Average server-side inference latency in milliseconds (``0.0`` if no requests recorded)."""
        if self.requests_total <= 0:
            return 0.0
        return (self.inference_latency_seconds_total * 1000.0) / float(self.requests_total)

    @property
    def success_rate(self) -> float:
        """Fraction of successful inference requests in ``[0.0, 1.0]`` (``1.0`` if no requests recorded)."""
        if self.requests_total <= 0:
            return 1.0
        return float(self.requests_success) / float(self.requests_total)

    @property
    def total_deallocations(self) -> int:
        """Sum of idle, capacity, and OOM model deallocations."""
        return (
            self.oom_deallocations_total
            + self.idle_deallocations_total
            + self.capacity_deallocations_total
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serializes this :class:`DaemonMetrics` snapshot into a dictionary."""
        return {
            "requests_total": self.requests_total,
            "requests_success": self.requests_success,
            "requests_error": self.requests_error,
            "inference_latency_seconds_total": self.inference_latency_seconds_total,
            "average_latency_ms": self.average_latency_ms,
            "success_rate": self.success_rate,
            "batches_dispatched_total": self.batches_dispatched_total,
            "oom_deallocations_total": self.oom_deallocations_total,
            "idle_deallocations_total": self.idle_deallocations_total,
            "capacity_deallocations_total": self.capacity_deallocations_total,
            "total_deallocations": self.total_deallocations,
            "model_load_retries_total": self.model_load_retries_total,
            "deallocation_thrashing_total": self.deallocation_thrashing_total,
            "stream_messages_consumed_total": self.stream_messages_consumed_total,
            "stream_messages_published_total": self.stream_messages_published_total,
            "stream_errors_total": self.stream_errors_total,
        }

    @classmethod
    def from_prometheus(cls, text: str) -> "DaemonMetrics":
        """Parses Prometheus exposition text from ``GET /metrics`` into a :class:`DaemonMetrics` instance.

        Args:
            text: Raw Prometheus exposition string returned by the SnoozeML server.

        Returns:
            Populated :class:`DaemonMetrics` snapshot.
        """
        metrics = cls()
        if not text:
            return metrics

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            match = _METRIC_LINE_RE.match(line)
            if not match:
                continue

            name = match.group("name")
            labels_raw = match.group("labels") or ""
            val_str = match.group("value")

            try:
                val = float(val_str)
            except ValueError:
                continue

            labels: Dict[str, str] = {}
            if labels_raw:
                for k, v in _LABEL_PAIR_RE.findall(labels_raw):
                    labels[k] = v
                canonical_labels = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
                full_key = f"{name}{{{canonical_labels}}}"
            else:
                full_key = name

            metrics.raw_metrics[full_key] = val

            if name == "snoozeml_requests_total":
                status = labels.get("status", "")
                if status == "success":
                    metrics.requests_success = int(val)
                elif status == "error":
                    metrics.requests_error = int(val)
                elif status == "all":
                    metrics.requests_total = int(val)
            elif name == "snoozeml_inference_latency_seconds_total":
                metrics.inference_latency_seconds_total = float(val)
            elif name == "snoozeml_batches_dispatched_total":
                metrics.batches_dispatched_total = int(val)
            elif name == "snoozeml_model_deallocations_total":
                reason = labels.get("reason", "")
                if reason == "oom":
                    metrics.oom_deallocations_total = int(val)
                elif reason == "idle":
                    metrics.idle_deallocations_total = int(val)
                elif reason == "capacity":
                    metrics.capacity_deallocations_total = int(val)
            elif name == "snoozeml_model_load_retries_total":
                metrics.model_load_retries_total = int(val)
            elif name == "snoozeml_deallocation_thrashing_total":
                metrics.deallocation_thrashing_total = int(val)
            elif name == "snoozeml_stream_messages_consumed_total":
                metrics.stream_messages_consumed_total = int(val)
            elif name == "snoozeml_stream_messages_published_total":
                metrics.stream_messages_published_total = int(val)
            elif name == "snoozeml_stream_errors_total":
                metrics.stream_errors_total = int(val)

        return metrics


def parse_prometheus_metrics(text: str) -> DaemonMetrics:
    """Convenience function that parses Prometheus exposition text into :class:`DaemonMetrics`."""
    return DaemonMetrics.from_prometheus(text)
