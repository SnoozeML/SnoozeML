SnoozeML Python SDK Documentation
=================================

Welcome to the official Python SDK documentation for **SnoozeML**, the bare-metal, scale-to-zero deep learning and machine learning inference daemon written in Rust.

.. note::
   Looking for server deployment, ``models.toml`` configuration, or Rust architecture guides?
   Visit the `← SnoozeML Core Service Documentation <../index.html>`_.

Key Capabilities
----------------

* **Dual-Transport Client** (:class:`~snoozeml.SnoozeClient` & :class:`~snoozeml.AsyncSnoozeClient`):
  Communicate with SnoozeML over HTTP REST (JSON) or high-throughput native gRPC (raw little-endian ``FP32`` byte buffers) with automatic NumPy ``ndarray`` serialization.
* **Model Lifecycle & Scale-to-Zero Control** (:class:`~snoozeml.models.ModelsManager`):
  List registered models, inspect live states (``idle``, ``loading``, ``ready``, ``busy``), preload weights (``load``), pin critical models in memory (``pin`` / ``unpin``), and explicitly evict models (``evict`` / ``unload``).
* **Automatic ONNX & XGBoost Conversion** (:mod:`snoozeml.converters`):
  Export fitted ``scikit-learn`` estimators/pipelines (:func:`~snoozeml.converters.export_sklearn`), PyTorch ``nn.Module`` networks (:func:`~snoozeml.converters.export_torch`), and ``XGBoost`` models (:func:`~snoozeml.converters.export_xgboost`) directly into SnoozeML-ready artifacts.
* **Structured Logging & Credential Sanitization** (:mod:`snoozeml.logging`):
  Built-in :class:`~snoozeml.logging.RedactingFilter` and :class:`~snoozeml.logging.StructuredFormatter` guarantee that API keys and JWT Bearer tokens are never leaked in logs or ``repr()`` strings.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   quickstart
   converters
   logging_and_errors

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index
