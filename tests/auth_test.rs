//! Integration and behavioral test suite for authentication and RBAC middleware.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use http_body_util::BodyExt;
use jsonwebtoken::{encode, EncodingKey, Header};
use snooze_ml::auth::{
    ApiKeyConfig, AuthConfig, AuthService, JwtAlgorithm, JwtClaims, JwtConfig, Role,
};
use snooze_ml::model::{ModelConfig, ModelController};
use snooze_ml::registry::ModelRegistry;
use snooze_ml::server::http::{create_router, AppState};
use snooze_ml::server::metrics::ServerMetrics;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tower::ServiceExt;

const TEST_JWT_SECRET: &str = "test-secret-key-that-is-at-least-32-characters-long!";

fn create_test_model(id: &str) -> Arc<ModelController> {
    let config = ModelConfig {
        id: id.to_string(),
        onnx_path: "mock:test".to_string(),
        idle_timeout: Duration::from_secs(60),
        max_batch_size: 4,
        max_batch_delay: Duration::from_millis(10),
        provider: None,
        description: Some("Test model for auth validation".to_string()),
        min_residency: Duration::from_millis(0),
        ..Default::default()
    };
    ModelController::new(config)
}

fn create_jwt_token(
    sub: &str,
    role: Option<Role>,
    roles: Option<Vec<Role>>,
    exp_offset_secs: i64,
) -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;
    let exp = (now + exp_offset_secs) as usize;

    let claims = JwtClaims {
        sub: sub.to_string(),
        exp,
        iat: Some(now as usize),
        nbf: None,
        iss: Some("snoozeml-test".to_string()),
        aud: Some("snoozeml-clients".to_string()),
        role,
        roles,
    };

    encode(
        &Header::default(),
        &claims,
        &EncodingKey::from_secret(TEST_JWT_SECRET.as_bytes()),
    )
    .unwrap()
}

fn sample_infer_request(model_id: &str) -> Request<Body> {
    let payload = serde_json::json!({
        "inputs": [
            {
                "name": "input_ids",
                "shape": [1, 2],
                "data": [10.0, 20.0]
            }
        ]
    });

    Request::builder()
        .method("POST")
        .uri(format!("/v1/models/{model_id}/infer"))
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap()
}

#[tokio::test]
async fn test_auth_disabled_allows_unauthenticated_access() {
    let registry = ModelRegistry::new();
    let metrics = ServerMetrics::new_shared();
    let model = create_test_model("model-open");
    registry.register(model).await;

    let app = create_router(AppState::new(registry, metrics));

    // 1. Health / Home
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/health/live")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // 2. Models list
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // 3. Infer
    let resp = app
        .clone()
        .oneshot(sample_infer_request("model-open"))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // 4. Admin Load & Unload
    let load_req = Request::builder()
        .method("POST")
        .uri("/v1/models/model-open/load")
        .body(Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(load_req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let unload_req = Request::builder()
        .method("POST")
        .uri("/v1/models/model-open/unload")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(unload_req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_auth_enabled_public_routes_always_accessible() {
    let registry = ModelRegistry::new();
    let metrics = ServerMetrics::new_shared();

    let auth_config = AuthConfig {
        enabled: true,
        protect_metrics: false,
        api_keys: vec![ApiKeyConfig::new("admin-key-123", Role::Admin).with_name("test-admin")],
        jwt: None,
    };
    let auth_svc = Arc::new(AuthService::new(auth_config).unwrap());
    let app = create_router(AppState::new(registry, metrics).with_auth(auth_svc));

    // GET /
    let resp = app
        .clone()
        .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // GET /health/live
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/health/live")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // GET /health/ready
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/health/ready")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // GET /metrics (default public)
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_auth_enabled_missing_credentials_returns_401() {
    let registry = ModelRegistry::new();
    let metrics = ServerMetrics::new_shared();

    let auth_config = AuthConfig {
        enabled: true,
        protect_metrics: false,
        api_keys: vec![ApiKeyConfig::new("secret-key", Role::Inference).with_name("client")],
        jwt: None,
    };
    let auth_svc = Arc::new(AuthService::new(auth_config).unwrap());
    let app = create_router(AppState::new(registry, metrics).with_auth(auth_svc));

    let resp = app
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(
        resp.headers().get(header::WWW_AUTHENTICATE).unwrap(),
        "Bearer realm=\"snoozeml\""
    );

    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert!(json["error"]
        .as_str()
        .unwrap()
        .contains("credentials are missing"));
}

#[tokio::test]
async fn test_api_key_role_matrix() {
    let registry = ModelRegistry::new();
    let metrics = ServerMetrics::new_shared();
    let model = create_test_model("model-matrix");
    registry.register(model).await;

    let auth_config = AuthConfig {
        enabled: true,
        protect_metrics: false,
        api_keys: vec![
            ApiKeyConfig::new("admin-key", Role::Admin).with_name("admin-user"),
            ApiKeyConfig::new("infer-key", Role::Inference).with_name("infer-user"),
            ApiKeyConfig::new("readonly-key", Role::ReadOnly).with_name("ro-user"),
        ],
        jwt: None,
    };
    let auth_svc = Arc::new(AuthService::new(auth_config).unwrap());
    let app = create_router(AppState::new(registry, metrics).with_auth(auth_svc));

    // 1. ReadOnly key:
    // Can read /v1/models
    let req = Request::builder()
        .uri("/v1/models")
        .header("x-api-key", "readonly-key")
        .body(Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // Cannot infer -> 403 Forbidden
    let mut infer_req = sample_infer_request("model-matrix");
    infer_req
        .headers_mut()
        .insert("x-api-key", "readonly-key".parse().unwrap());
    let resp = app.clone().oneshot(infer_req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);

    // Cannot load -> 403 Forbidden
    let load_req = Request::builder()
        .method("POST")
        .uri("/v1/models/model-matrix/load")
        .header("x-api-key", "readonly-key")
        .body(Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(load_req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);

    // Cannot refresh -> 403 Forbidden
    let refresh_req = Request::builder()
        .method("POST")
        .uri("/v1/models/refresh")
        .header("x-api-key", "readonly-key")
        .body(Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(refresh_req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);

    // 2. Inference key:
    // Can read /v1/models
    let req = Request::builder()
        .uri("/v1/models")
        .header("x-api-key", "infer-key")
        .body(Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // Can infer -> 200 OK
    let mut infer_req = sample_infer_request("model-matrix");
    infer_req
        .headers_mut()
        .insert("x-api-key", "infer-key".parse().unwrap());
    let resp = app.clone().oneshot(infer_req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // Cannot load -> 403 Forbidden
    let load_req = Request::builder()
        .method("POST")
        .uri("/v1/models/model-matrix/load")
        .header("x-api-key", "infer-key")
        .body(Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(load_req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);

    // Cannot refresh -> 403 Forbidden
    let refresh_req = Request::builder()
        .method("POST")
        .uri("/v1/models/refresh")
        .header("x-api-key", "infer-key")
        .body(Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(refresh_req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);

    // 3. Admin key:
    // Can infer -> 200 OK
    let mut infer_req = sample_infer_request("model-matrix");
    infer_req
        .headers_mut()
        .insert("x-api-key", "admin-key".parse().unwrap());
    let resp = app.clone().oneshot(infer_req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // Can load -> 200 OK
    let load_req = Request::builder()
        .method("POST")
        .uri("/v1/models/model-matrix/load")
        .header("x-api-key", "admin-key")
        .body(Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(load_req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // Can refresh -> 200 OK
    let refresh_req = Request::builder()
        .method("POST")
        .uri("/v1/models/refresh")
        .header("x-api-key", "admin-key")
        .body(Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(refresh_req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // Can unload -> 200 OK
    let unload_req = Request::builder()
        .method("POST")
        .uri("/v1/models/model-matrix/unload")
        .header("x-api-key", "admin-key")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(unload_req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_api_key_in_authorization_bearer_header() {
    let registry = ModelRegistry::new();
    let metrics = ServerMetrics::new_shared();
    let model = create_test_model("model-bearer-key");
    registry.register(model).await;

    let auth_config = AuthConfig {
        enabled: true,
        protect_metrics: false,
        api_keys: vec![
            ApiKeyConfig::new("sk-my-client-key-456", Role::Inference).with_name("sdk-client")
        ],
        jwt: None,
    };
    let auth_svc = Arc::new(AuthService::new(auth_config).unwrap());
    let app = create_router(AppState::new(registry, metrics).with_auth(auth_svc));

    // Presenting API key in standard "Authorization: Bearer <key>"
    let mut req = sample_infer_request("model-bearer-key");
    req.headers_mut().insert(
        header::AUTHORIZATION,
        "Bearer sk-my-client-key-456".parse().unwrap(),
    );

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_api_key_invalid_returns_401() {
    let registry = ModelRegistry::new();
    let metrics = ServerMetrics::new_shared();

    let auth_config = AuthConfig {
        enabled: true,
        protect_metrics: false,
        api_keys: vec![ApiKeyConfig::new("good-key", Role::Admin)],
        jwt: None,
    };
    let auth_svc = Arc::new(AuthService::new(auth_config).unwrap());
    let app = create_router(AppState::new(registry, metrics).with_auth(auth_svc));

    let req = Request::builder()
        .uri("/v1/models")
        .header("x-api-key", "bad-key-xyz")
        .body(Body::empty())
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn test_jwt_roles_and_lifecycle() {
    let registry = ModelRegistry::new();
    let metrics = ServerMetrics::new_shared();
    let model = create_test_model("model-jwt");
    registry.register(model).await;

    let auth_config = AuthConfig {
        enabled: true,
        protect_metrics: false,
        api_keys: Vec::new(),
        jwt: Some(JwtConfig {
            secret: Some(TEST_JWT_SECRET.to_string()),
            secret_env: None,
            algorithm: JwtAlgorithm::HS256,
            issuer: Some("snoozeml-test".to_string()),
            audience: Some("snoozeml-clients".to_string()),
            leeway_secs: 10,
        }),
    };
    let auth_svc = Arc::new(AuthService::new(auth_config).unwrap());
    let app = create_router(AppState::new(registry, metrics).with_auth(auth_svc));

    // 1. Valid Admin JWT
    let admin_token = create_jwt_token("admin-user", Some(Role::Admin), None, 3600);
    let mut req = sample_infer_request("model-jwt");
    req.headers_mut().insert(
        header::AUTHORIZATION,
        format!("Bearer {admin_token}").parse().unwrap(),
    );
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let load_req = Request::builder()
        .method("POST")
        .uri("/v1/models/model-jwt/load")
        .header(header::AUTHORIZATION, format!("Bearer {admin_token}"))
        .body(Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(load_req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // 2. Valid Inference JWT with multiple roles array
    let infer_token = create_jwt_token(
        "infer-user",
        None,
        Some(vec![Role::ReadOnly, Role::Inference]),
        3600,
    );
    let mut req = sample_infer_request("model-jwt");
    req.headers_mut().insert(
        header::AUTHORIZATION,
        format!("Bearer {infer_token}").parse().unwrap(),
    );
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // Inference user rejected from admin load endpoint -> 403
    let load_req = Request::builder()
        .method("POST")
        .uri("/v1/models/model-jwt/load")
        .header(header::AUTHORIZATION, format!("Bearer {infer_token}"))
        .body(Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(load_req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);

    // 3. Expired JWT -> 401
    let expired_token = create_jwt_token("expired-user", Some(Role::Admin), None, -100);
    let mut req = sample_infer_request("model-jwt");
    req.headers_mut().insert(
        header::AUTHORIZATION,
        format!("Bearer {expired_token}").parse().unwrap(),
    );
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert!(json["error"].as_str().unwrap().contains("expired"));

    // 4. JWT signed with wrong secret -> 401
    let wrong_secret_token = encode(
        &Header::default(),
        &JwtClaims {
            sub: "hacker".to_string(),
            exp: (SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs()
                + 3600) as usize,
            iat: None,
            nbf: None,
            iss: Some("snoozeml-test".to_string()),
            aud: Some("snoozeml-clients".to_string()),
            role: Some(Role::Admin),
            roles: None,
        },
        &EncodingKey::from_secret(b"completely-wrong-secret-signature-key-1234"),
    )
    .unwrap();

    let mut req = sample_infer_request("model-jwt");
    req.headers_mut().insert(
        header::AUTHORIZATION,
        format!("Bearer {wrong_secret_token}").parse().unwrap(),
    );
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn test_protect_metrics_flag() {
    let registry = ModelRegistry::new();
    let metrics = ServerMetrics::new_shared();

    // Configure with protect_metrics = true
    let auth_config = AuthConfig {
        enabled: true,
        protect_metrics: true,
        api_keys: vec![
            ApiKeyConfig::new("admin-key", Role::Admin).with_name("admin"),
            ApiKeyConfig::new("infer-key", Role::Inference).with_name("infer"),
        ],
        jwt: None,
    };
    let auth_svc = Arc::new(AuthService::new(auth_config).unwrap());
    let app = create_router(AppState::new(registry, metrics).with_auth(auth_svc));

    // 1. Unauthenticated /metrics -> 401
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

    // 2. Inference role on /metrics -> 403 Forbidden
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .header("x-api-key", "infer-key")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);

    // 3. Admin role on /metrics -> 200 OK
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .header("x-api-key", "admin-key")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_dual_mode_concurrent_auth() {
    let registry = ModelRegistry::new();
    let metrics = ServerMetrics::new_shared();
    let model = create_test_model("model-dual");
    registry.register(model).await;

    let auth_config = AuthConfig {
        enabled: true,
        protect_metrics: false,
        api_keys: vec![
            ApiKeyConfig::new("static-key-admin", Role::Admin).with_name("api-key-caller")
        ],
        jwt: Some(JwtConfig::new(TEST_JWT_SECRET)),
    };
    let auth_svc = Arc::new(AuthService::new(auth_config).unwrap());
    let app = create_router(AppState::new(registry, metrics).with_auth(auth_svc));

    // 1. Request via API key
    let mut req1 = sample_infer_request("model-dual");
    req1.headers_mut()
        .insert("x-api-key", "static-key-admin".parse().unwrap());
    let resp1 = app.clone().oneshot(req1).await.unwrap();
    assert_eq!(resp1.status(), StatusCode::OK);

    // 2. Request via JWT
    let token = create_jwt_token("jwt-caller", Some(Role::Inference), None, 3600);
    let mut req2 = sample_infer_request("model-dual");
    req2.headers_mut().insert(
        header::AUTHORIZATION,
        format!("Bearer {token}").parse().unwrap(),
    );
    let resp2 = app.oneshot(req2).await.unwrap();
    assert_eq!(resp2.status(), StatusCode::OK);
}

#[test]
fn test_auth_config_rejects_empty_api_key() {
    // Empty key
    let config_empty = AuthConfig {
        enabled: true,
        protect_metrics: false,
        api_keys: vec![ApiKeyConfig::new("", Role::Inference).with_name("empty")],
        jwt: None,
    };
    let err_empty = AuthService::new(config_empty).unwrap_err();
    assert!(matches!(err_empty, snooze_ml::auth::AuthError::Internal(_)));

    // Whitespace key
    let config_whitespace = AuthConfig {
        enabled: true,
        protect_metrics: false,
        api_keys: vec![ApiKeyConfig::new("   ", Role::Inference).with_name("whitespace")],
        jwt: None,
    };
    let err_ws = AuthService::new(config_whitespace).unwrap_err();
    assert!(matches!(err_ws, snooze_ml::auth::AuthError::Internal(_)));
}

#[tokio::test]
async fn test_jwt_rejects_alg_none_and_mismatched_algorithms() {
    let registry = ModelRegistry::new();
    let metrics = ServerMetrics::new_shared();
    let model = create_test_model("model-jwt-alg");
    registry.register(model).await;

    let auth_config = AuthConfig {
        enabled: true,
        protect_metrics: false,
        api_keys: vec![],
        jwt: Some(JwtConfig::new(TEST_JWT_SECRET)),
    };
    let auth_svc = Arc::new(AuthService::new(auth_config).unwrap());
    let app = create_router(AppState::new(registry, metrics).with_auth(auth_svc));

    // 1. Token forged with "alg": "none" header
    // Header: {"typ":"JWT","alg":"none"} -> eyJ0eXAiOiJKV1QiLCJhbGciOiJub25lIn0
    // Payload: {"sub":"attacker","role":"admin","exp":2500000000} -> eyJzdWIiOiJhdHRhY2tlciIsInJvbGUiOiJhZG1pbiIsImV4cCI6MjUwMDAwMDAwMH0
    let forged_none_token =
        "eyJ0eXAiOiJKV1QiLCJhbGciOiJub25lIn0.eyJzdWIiOiJhdHRhY2tlciIsInJvbGUiOiJhZG1pbiIsImV4cCI6MjUwMDAwMDAwMH0.";

    let mut req1 = sample_infer_request("model-jwt-alg");
    req1.headers_mut().insert(
        header::AUTHORIZATION,
        format!("Bearer {forged_none_token}").parse().unwrap(),
    );
    let resp1 = app.clone().oneshot(req1).await.unwrap();
    assert_eq!(
        resp1.status(),
        StatusCode::UNAUTHORIZED,
        "Tokens with alg: none must be rejected"
    );

    // 2. Token signed with mismatched algorithm (e.g. HS384 when server expects HS256)
    let header_hs384 = Header::new(jsonwebtoken::Algorithm::HS384);
    let claims_struct = JwtClaims {
        sub: "user-hs384".to_string(),
        exp: 2500000000,
        iat: None,
        nbf: None,
        iss: None,
        aud: None,
        role: Some(Role::Admin),
        roles: None,
    };
    let token_hs384 = encode(
        &header_hs384,
        &claims_struct,
        &EncodingKey::from_secret(TEST_JWT_SECRET.as_bytes()),
    )
    .unwrap();

    let mut req2 = sample_infer_request("model-jwt-alg");
    req2.headers_mut().insert(
        header::AUTHORIZATION,
        format!("Bearer {token_hs384}").parse().unwrap(),
    );
    let resp2 = app.oneshot(req2).await.unwrap();
    assert_eq!(
        resp2.status(),
        StatusCode::UNAUTHORIZED,
        "Tokens with mismatched algorithm must be rejected"
    );
}

static AUTH_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

struct EnvGuard {
    vars: Vec<&'static str>,
}

impl EnvGuard {
    fn set(vars: &[(&'static str, &str)]) -> Self {
        let mut names = Vec::new();
        for (k, v) in vars {
            std::env::set_var(k, v);
            names.push(*k);
        }
        Self { vars: names }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        for v in &self.vars {
            std::env::remove_var(v);
        }
    }
}

#[test]
fn test_auth_config_from_env_single_key_defaults_to_admin() {
    let _lock = AUTH_ENV_LOCK.lock().unwrap();
    let _guard = EnvGuard::set(&[("SNOOZE_API_KEY", "sk-env-admin-999")]);

    let config = AuthConfig {
        enabled: true,
        ..Default::default()
    };
    let auth = AuthService::new(config).unwrap();
    let identity = auth.authenticate_api_key("sk-env-admin-999").unwrap();
    assert_eq!(identity.role, Role::Admin);
    assert_eq!(identity.subject, "env-admin");
}

#[test]
fn test_auth_config_from_env_role_prefixed() {
    let _lock = AUTH_ENV_LOCK.lock().unwrap();
    let _guard = EnvGuard::set(&[("SNOOZE_API_KEY", "infer:sk-env-infer-888")]);

    let config = AuthConfig {
        enabled: true,
        ..Default::default()
    };
    let auth = AuthService::new(config).unwrap();
    let identity = auth.authenticate_api_key("sk-env-infer-888").unwrap();
    assert_eq!(identity.role, Role::Inference);
}

#[test]
fn test_auth_config_from_env_multiple_keys() {
    let _lock = AUTH_ENV_LOCK.lock().unwrap();
    let _guard = EnvGuard::set(&[(
        "SNOOZE_API_KEYS",
        "admin:sk-multi-adm,infer:sk-multi-inf,read_only:sk-multi-ro",
    )]);

    let config = AuthConfig {
        enabled: true,
        ..Default::default()
    };
    let auth = AuthService::new(config).unwrap();
    assert_eq!(
        auth.authenticate_api_key("sk-multi-adm").unwrap().role,
        Role::Admin
    );
    assert_eq!(
        auth.authenticate_api_key("sk-multi-inf").unwrap().role,
        Role::Inference
    );
    assert_eq!(
        auth.authenticate_api_key("sk-multi-ro").unwrap().role,
        Role::ReadOnly
    );
}

#[test]
fn test_auth_config_from_env_role_specific() {
    let _lock = AUTH_ENV_LOCK.lock().unwrap();
    let _guard = EnvGuard::set(&[
        ("SNOOZE_API_KEY_ADMIN", "sk-admin-direct"),
        ("SNOOZE_API_KEY_INFER", "sk-infer-direct"),
        ("SNOOZE_API_KEY_READONLY", "sk-ro-direct"),
    ]);

    let config = AuthConfig {
        enabled: true,
        ..Default::default()
    };
    let auth = AuthService::new(config).unwrap();
    assert_eq!(
        auth.authenticate_api_key("sk-admin-direct").unwrap().role,
        Role::Admin
    );
    assert_eq!(
        auth.authenticate_api_key("sk-infer-direct").unwrap().role,
        Role::Inference
    );
    assert_eq!(
        auth.authenticate_api_key("sk-ro-direct").unwrap().role,
        Role::ReadOnly
    );
}

#[test]
fn test_api_key_config_key_env() {
    let _lock = AUTH_ENV_LOCK.lock().unwrap();
    let _guard = EnvGuard::set(&[("MY_CUSTOM_SECRET_KEY", "sk-custom-secret-key-456")]);

    let key_cfg =
        ApiKeyConfig::from_env("MY_CUSTOM_SECRET_KEY", Role::Inference).with_name("custom-app");
    assert_eq!(
        key_cfg.resolve_key().as_deref(),
        Some("sk-custom-secret-key-456")
    );

    let config = AuthConfig {
        enabled: true,
        protect_metrics: false,
        api_keys: vec![key_cfg],
        jwt: None,
    };
    let auth = AuthService::new(config).unwrap();
    let identity = auth
        .authenticate_api_key("sk-custom-secret-key-456")
        .unwrap();
    assert_eq!(identity.role, Role::Inference);
    assert_eq!(identity.subject, "custom-app");
}

#[test]
fn test_api_key_config_dollar_expansion() {
    let _lock = AUTH_ENV_LOCK.lock().unwrap();
    let _guard = EnvGuard::set(&[("EXPANDED_SECRET_VAR", "sk-expanded-val-789")]);

    let key_cfg =
        ApiKeyConfig::new("$EXPANDED_SECRET_VAR", Role::Admin).with_name("expanded-admin");
    assert_eq!(
        key_cfg.resolve_key().as_deref(),
        Some("sk-expanded-val-789")
    );

    let config = AuthConfig {
        enabled: true,
        protect_metrics: false,
        api_keys: vec![key_cfg],
        jwt: None,
    };
    let auth = AuthService::new(config).unwrap();
    assert_eq!(
        auth.authenticate_api_key("sk-expanded-val-789")
            .unwrap()
            .role,
        Role::Admin
    );
}

#[test]
fn test_jwt_config_from_env() {
    let _lock = AUTH_ENV_LOCK.lock().unwrap();
    let _guard = EnvGuard::set(&[("SNOOZE_JWT_SECRET", TEST_JWT_SECRET)]);

    let config = AuthConfig {
        enabled: true,
        ..Default::default()
    };
    let auth = AuthService::new(config).unwrap();
    let token = create_jwt_token("env-jwt-user", Some(Role::Inference), None, 3600);
    let identity = auth.authenticate_jwt(&token).unwrap();
    assert_eq!(identity.subject, "env-jwt-user");
    assert_eq!(identity.role, Role::Inference);
}

#[test]
fn test_jwt_config_secret_env() {
    let _lock = AUTH_ENV_LOCK.lock().unwrap();
    let _guard = EnvGuard::set(&[("CUSTOM_JWT_SECRET_VAR", TEST_JWT_SECRET)]);

    let jwt_cfg = JwtConfig::from_env("CUSTOM_JWT_SECRET_VAR");
    assert_eq!(jwt_cfg.resolve_secret().as_deref(), Some(TEST_JWT_SECRET));

    let config = AuthConfig {
        enabled: true,
        protect_metrics: false,
        api_keys: vec![],
        jwt: Some(jwt_cfg),
    };
    let auth = AuthService::new(config).unwrap();
    let token = create_jwt_token("custom-jwt-user", Some(Role::Admin), None, 3600);
    let identity = auth.authenticate_jwt(&token).unwrap();
    assert_eq!(identity.subject, "custom-jwt-user");
    assert_eq!(identity.role, Role::Admin);
}

#[tokio::test]
async fn test_models_toml_only_enabled_true_with_env_key() {
    let _lock = AUTH_ENV_LOCK.lock().unwrap();
    let _guard = EnvGuard::set(&[("SNOOZE_API_KEY", "sk-pure-env-key-123")]);

    let toml_content = r#"
[auth]
enabled = true
protect_metrics = true
"#;
    let mut config = snooze_ml::config::ModelDirectoryConfig::parse_content(
        toml_content,
        std::path::Path::new("models.toml"),
    )
    .unwrap();

    let mut auth_cfg = config.auth.take().unwrap();
    auth_cfg.load_from_env();
    assert_eq!(auth_cfg.api_keys.len(), 1);
    assert_eq!(
        auth_cfg.api_keys[0].resolve_key().as_deref(),
        Some("sk-pure-env-key-123")
    );

    let auth_svc = AuthService::new(auth_cfg).unwrap();
    let identity = auth_svc
        .authenticate_api_key("sk-pure-env-key-123")
        .unwrap();
    assert_eq!(identity.role, Role::Admin);
}
