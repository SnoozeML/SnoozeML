Event Streaming & Brokers
=========================

SnoozeML connects real-time model inference directly to distributed event streaming brokers (Apache Kafka, RabbitMQ, Redis Streams, and Mock streams).

The :mod:`snoozeml.streams` module provides serialization envelopes and broker clients for streaming pipelines.

Streaming Request & Response Envelopes
--------------------------------------

:class:`~snoozeml.streams.StreamInferRequest` and :class:`~snoozeml.streams.StreamInferResponse` format payloads for event streaming ingestion:

.. code-block:: python

   from snoozeml.streams import StreamInferRequest, StreamWireFormat

   # Construct streaming inference request
   req = StreamInferRequest(
       model_id="iris_classifier",
       inputs={"input": [[5.1, 3.5, 1.4, 0.2]]},
       request_id="event-1234",
       client_timeout_ms=1000,
   )

   # Serialize to JSON wire format
   json_payload = req.to_json()

   # Or serialize to binary Protobuf bytes
   proto_payload = req.to_protobuf()

In-Memory Testing with ``MockStreamClient``
-------------------------------------------

Test stream processing logic in unit tests or local pipelines without running broker clusters:

.. code-block:: python

   from snoozeml import SnoozeClient
   from snoozeml.streams import MockStreamClient

   client = SnoozeClient("http://localhost:8080")
   stream_client = MockStreamClient(client=client, default_model_id="iris_classifier")

   # Simulates streaming ingestion through the standard SnoozeClient
   response = stream_client.infer(
       inputs={"input": [[5.1, 3.5, 1.4, 0.2]]},
       request_id="tx-99",
   )
   print(f"Output: {response.outputs['output']}")

Redis Streams Ingestion (``RedisStreamClient``)
-----------------------------------------------

Use :class:`~snoozeml.streams.RedisStreamClient` to produce to and consume from live Redis Streams (using ``XADD`` and ``XREAD``):

.. code-block:: python

   from snoozeml.streams import RedisStreamClient, StreamWireFormat

   redis_client = RedisStreamClient(
       endpoint="redis://127.0.0.1:6379",
       input_topic="snooze.requests",
       output_topic="snooze.responses",
       default_model_id="fraud_detector",
       format=StreamWireFormat.JSON,
   )

   # Publishes to snooze.requests and awaits response on snooze.responses
   result = redis_client.infer(
       inputs={"features": [[0.5, 1.2, -0.4, 3.1]]},
       timeout=5.0,
   )
   print(f"Prediction: {result.outputs}")
