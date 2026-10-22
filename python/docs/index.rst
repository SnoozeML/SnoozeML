SnoozeML Python SDK Documentation
=================================

Welcome to the official Python SDK documentation for **SnoozeML**, the bare-metal, scale-to-zero deep learning and machine learning inference daemon written in Rust.

.. note::
   Looking for server deployment, ``models.toml`` configuration, or Rust architecture guides?
   Visit the `← SnoozeML Core Service Documentation <../index.html>`_.

Key Capabilities
----------------

* **Dual-Transport Client** (:class:`~snoozeml.SnoozeClient` & :class:`~snoozeml.AsyncSnoozeClient`):
  Communicate with SnoozeML over HTTP REST (JSON) or high-throughput native gRPC (raw little-endian ``FP32`` byte buffers) with automatic NumPy and PyTorch tensor serialization, dataset batch chunking (:meth:`~snoozeml.SnoozeClient.infer_dataset`), and transient 503 retry resilience.
* **Model Lifecycle & Scale-to-Zero Control** (:class:`~snoozeml.models.ModelsManager`):
  List registered models, inspect live states (``idle``, ``loading``, ``ready``, ``busy``), preload weights (``load``), pin critical models in memory (``pin`` / ``unpin``), explicitly evict models (``evict`` / ``unload``), and block until ready (:meth:`~snoozeml.models.ModelsManager.wait_for_ready`).
* **One-Step Deployment & ML Converters** (:mod:`snoozeml.converters`):
  Deploy trained models directly from Python memory or export fitted ``scikit-learn`` estimators/pipelines (:func:`~snoozeml.converters.export_sklearn`), PyTorch ``nn.Module`` networks (:func:`~snoozeml.converters.export_torch`), and ``XGBoost`` models (:func:`~snoozeml.converters.export_xgboost`) directly into SnoozeML-ready artifacts.
* **Authentication, RBAC & JWT Minting** (:mod:`snoozeml.auth`):
  Mint and verify RFC 7519 HMAC JSON Web Tokens (``HS256``, ``HS384``, ``HS512``) matching SnoozeML's 3-tier RBAC hierarchy (``ReadOnly``, ``Inference``, ``Admin``).
* **Event Streaming & Real-Time Ingestion** (:mod:`snoozeml.streams`):
  Serialize streaming inference envelopes (:class:`~snoozeml.streams.StreamInferRequest`, :class:`~snoozeml.streams.StreamInferResponse`), test with :class:`~snoozeml.streams.MockStreamClient`, and connect to live Redis Streams (:class:`~snoozeml.streams.RedisStreamClient`).
* **Programmatic Config & Local Test Runner** (:mod:`snoozeml.config`):
  Build ``models.toml`` declaratively with :class:`~snoozeml.config.SnoozeConfig`, scaffold process-isolated custom Python workers (:func:`~snoozeml.config.scaffold_worker`), and run an ephemeral daemon in Python with :class:`~snoozeml.config.SnoozeDaemon`.
* **Telemetry & Prometheus Parsing** (:mod:`snoozeml.metrics`):
  Scrape and parse exposition counters from ``GET /metrics`` with :func:`~snoozeml.metrics.parse_prometheus_metrics` for monitoring success rates, latencies, and scale-to-zero memory events.
* **Structured Logging & Credential Sanitization** (:mod:`snoozeml.logging`):
  Built-in :class:`~snoozeml.logging.RedactingFilter` and :class:`~snoozeml.logging.StructuredFormatter` guarantee that API keys and JWT Bearer tokens are never leaked in logs or ``repr()`` strings.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   quickstart
   converters
   auth
   streams
   config_and_daemon
   metrics
   logging_and_errors

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index
