Telemetry & Prometheus Metrics
==============================

SnoozeML exposes Prometheus metrics at ``GET /metrics`` (protected by ``Role.ADMIN`` when ``protect_metrics = true``). The :mod:`snoozeml.metrics` module parses Prometheus exposition text into a structured :class:`~snoozeml.metrics.DaemonMetrics` dataclass.

Fetching Metrics via ``SnoozeClient``
-------------------------------------

Call :meth:`~snoozeml.SnoozeClient.get_metrics` to scrape and parse metrics in one step:

.. code-block:: python

   from snoozeml import SnoozeClient

   client = SnoozeClient("http://localhost:8080", api_key="admin-key")

   metrics = client.get_metrics()
   print(f"Total Requests: {metrics.inference_requests_total}")
   print(f"Success Count: {metrics.inference_success_total}")
   print(f"Error Count: {metrics.inference_errors_total}")
   print(f"Success Rate: {metrics.success_rate:.2%}")
   print(f"Average Server Latency: {metrics.average_latency_ms:.2f} ms")
   print(f"Currently Loaded Models: {metrics.models_loaded_count}")
   print(f"Scale-to-Zero Evictions: {metrics.model_unloads_total}")

Parsing Raw Metrics Exposition Text
-----------------------------------

You can also parse raw Prometheus text directly using :func:`~snoozeml.metrics.parse_prometheus_metrics`:

.. code-block:: python

   from snoozeml.metrics import parse_prometheus_metrics

   raw_prometheus_text = requests.get("http://localhost:8080/metrics").text
   metrics = parse_prometheus_metrics(raw_prometheus_text)

Parsed Metrics Fields
---------------------

.. list-table:: DaemonMetrics Fields
   :header-rows: 1
   :widths: 35 15 50

   * - Field
     - Type
     - Description
   * - ``inference_requests_total``
     - ``int``
     - Cumulative inference requests received.
   * - ``inference_success_total``
     - ``int``
     - Cumulative successful inference requests.
   * - ``inference_errors_total``
     - ``int``
     - Cumulative failed inference requests.
   * - ``inference_duration_seconds_total``
     - ``float``
     - Total time spent in forward inference passes across all requests.
   * - ``model_loads_total``
     - ``int``
     - Total number of model weight load events from disk.
   * - ``model_unloads_total``
     - ``int``
     - Total number of scale-to-zero and LRU eviction events.
   * - ``models_loaded_count``
     - ``int``
     - Instantaneous count of models resident in host RAM / GPU VRAM.
   * - ``batcher_batches_total``
     - ``int``
     - Total micro-batches processed by the dynamic batcher.
   * - ``batcher_items_total``
     - ``int``
     - Total individual request items coalesced across all batches.
   * - ``queue_depth``
     - ``int``
     - Instantaneous count of requests waiting in the batcher queue.
   * - ``success_rate`` (property)
     - ``float``
     - Ratio of successful inferences to total requests (0.0 to 1.0).
   * - ``average_latency_ms`` (property)
     - ``float``
     - Average forward inference latency in milliseconds.
