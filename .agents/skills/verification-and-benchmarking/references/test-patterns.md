# SnoozeML Integration Test Patterns (`tests/`)

All test files in `tests/` begin with:
```rust
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
```
*(Remember: `.unwrap()`, `.expect()`, and `panic!()` are strictly forbidden in `src/`, but allowed in `tests/`.)*

---

## 1. In-Process HTTP Router Test (`tower::ServiceExt::oneshot`)

Use `oneshot` for fast, port-free HTTP REST and RBAC testing:

```rust
use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use snooze_ml::model::{ModelConfig, ModelController};
use snooze_ml::registry::ModelRegistry;
use snooze_ml::server::http::{create_router, AppState};
use snooze_ml::server::metrics::ServerMetrics;
use std::time::Duration;
use tower::ServiceExt;

#[tokio::test]
async fn test_http_endpoint_example() {
    let registry = ModelRegistry::new();
    let metrics = ServerMetrics::new_shared();

    let config = ModelConfig {
        id: "test-model".to_string(),
        onnx_path: "mock:test".to_string(),
        idle_timeout: Duration::from_secs(60),
        max_batch_size: 4,
        max_batch_delay: Duration::from_millis(5),
        ..Default::default()
    };
    registry.register(ModelController::new(config)).await;

    let app = create_router(AppState::new(registry, metrics));

    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/models/test-model/pin")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["pinned"], true);
}
```

---

## 2. Ephemeral Port Server Test (gRPC / CLI / TLS)

When testing `run_server`, Tonic gRPC clients, or `ureq` CLI helpers, bind to port `0` first to obtain a free OS port:

```rust
async fn get_free_port() -> u16 {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    listener.local_addr().unwrap().port()
}
```

> **Note on Environment Variable Tests**: Tests that mutate process environment variables (`SNOOZE_MODELS_DIR`, `SNOOZE_GRPC`, etc.) must acquire a test-local `static ENV_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());` so parallel test threads do not race.

---

## 3. Reactive OOM & LRU Eviction Test Pattern

When testing `mock:oom_configurable` using `set_mock_oom_failures`, serialize access to the global atomic counter (`MOCK_OOM_COUNT`) if multiple OOM tests run in the same binary, and set `min_residency: Duration::from_millis(0)` on candidate models so they are immediately eligible for eviction.
