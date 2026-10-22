Configuration, Daemon Runner & Worker Scaffolding
===================================================

The :mod:`snoozeml.config` module enables declarative programmatic generation of ``models.toml`` and ``models.json`` files, provides an automated local daemon runner for testing, and scaffolds process-isolated custom Python workers.

Programmatic Configuration Builder (``SnoozeConfig``)
-----------------------------------------------------

Generate configuration files programmatically using a fluent builder:

.. code-block:: python

   from snoozeml import SnoozeConfig, GlobalDefaultsConfig

   config = (
       SnoozeConfig(defaults=GlobalDefaultsConfig(max_loaded_models=8, idle_timeout_secs=60))
       # Register an ONNX model
       .add_model("bert_embeddings", "bert.onnx", pinned=True, max_batch_size=32)
       # Register an XGBoost tree model
       .add_model("fraud_detector", "fraud.xgb.json", idle_timeout_secs=120)
       # Configure RBAC API keys
       .add_api_key(role="admin", key_env="SNOOZE_ADMIN_KEY", name="cluster-admin")
       .add_api_key(role="inference", key_env="SNOOZE_INFER_KEY", name="scoring-pod")
       # Configure HMAC JWT authentication
       .set_jwt(secret_env="SNOOZE_JWT_SECRET", algorithm="HS256")
       # Bind an event streaming pipeline
       .add_stream(
           name="fraud-stream",
           broker="redis",
           endpoint="redis://127.0.0.1:6379",
           input_topic="transactions.in",
           output_topic="transactions.out",
           error_topic="transactions.dlq",
           model_id="fraud_detector",
       )
   )

   # Save configuration to disk
   config.save("models/models.toml")

Automated Local Daemon Context Manager (``SnoozeDaemon``)
---------------------------------------------------------

Launch an ephemeral SnoozeML daemon subprocess directly from Python. When the context block exits, the daemon is cleanly terminated:

.. code-block:: python

   from snoozeml import SnoozeDaemon

   # Automatically finds 'snooze' binary in target/release, target/debug, or PATH
   with SnoozeDaemon(models_dir="models", port=8080) as client:
       # client is a fully initialized SnoozeClient connected to the ephemeral server!
       client.wait_until_ready(timeout=10.0)

       response = client.infer("iris_classifier", {"input": [[5.1, 3.5, 1.4, 0.2]]})
       print(f"Result: {response.get_numpy('output')}")

Custom Python Worker Scaffolding (``scaffold_worker``)
------------------------------------------------------

Process-isolated Python workers allow running proprietary models or frameworks (such as PyTorch generative networks or Amazon Chronos-Bolt) inside an isolated child process with IPC framing:

.. code-block:: python

   from snoozeml import scaffold_worker

   # Scaffolds models/custom_worker/snooze_model.py and snooze_model.json
   # and runs an isolated smoke test to verify IPC protocol compliance
   scaffold_worker(
       target_dir="models/custom_worker",
       tensor_format="numpy",
       verify=True,
   )

To inspect or verify an existing worker from the command line:

.. code-block:: bash

   python -m snoozeml.worker --check models/custom_worker
