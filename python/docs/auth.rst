Authentication, RBAC & JWT Minting
===================================

SnoozeML uses a 3-tier Role-Based Access Control (RBAC) hierarchy to secure inference and model management endpoints. The Python SDK includes built-in RFC 7519 HMAC JSON Web Token minting and decoding functions in :mod:`snoozeml.auth`.

Role Hierarchy
--------------

.. list-table:: 3-Tier RBAC Roles
   :header-rows: 1
   :widths: 20 15 65

   * - Role (:class:`~snoozeml.auth.Role`)
     - Level
     - Allowed Operations
   * - ``Role.READ_ONLY``
     - 1
     - Query registered models (``GET /v1/models``), inspect metadata.
   * - ``Role.INFERENCE``
     - 2
     - All ReadOnly actions + execute forward inference passes (``POST /v1/models/:id/infer`` and gRPC ``Infer``).
   * - ``Role.ADMIN``
     - 3
     - All Inference actions + load/unload/pin/unpin models, upload models, scrape protected Prometheus metrics.

Minting HMAC JSON Web Tokens
----------------------------

Use :func:`~snoozeml.auth.create_jwt_token` to generate HMAC-signed JWTs matching SnoozeML's claim format:

.. code-block:: python

   from snoozeml import Role, create_jwt_token

   # Generate a token valid for 1 hour
   token = create_jwt_token(
       secret="my-cluster-secret",
       subject="worker-service-01",
       role=Role.INFERENCE,
       expires_in_secs=3600,
       issuer="snooze-auth-server",
       algorithm="HS256",  # Supported: "HS256", "HS384", "HS512"
   )
   print(f"Minted JWT: {token}")

Decoding and Verifying Claims Locally
-------------------------------------

Inspect and validate claims without contacting the server:

.. code-block:: python

   from snoozeml import decode_jwt_token

   claims = decode_jwt_token(
       token=token,
       secret="my-cluster-secret",
       issuer="snooze-auth-server",
   )
   print(f"Subject: {claims['sub']}")
   print(f"Role: {claims['role']}")
   print(f"Expires at: {claims['exp']}")

Authenticating Client Connections
---------------------------------

Pass either an API key or a minted JWT to :class:`~snoozeml.SnoozeClient`:

.. code-block:: python

   from snoozeml import SnoozeClient

   # Option A: Static API key (sent via x-api-key or Bearer)
   client = SnoozeClient("http://localhost:8080", api_key="sk-admin-key")

   # Option B: Dynamic JWT Bearer token
   client = SnoozeClient("http://localhost:8080", token=token)
