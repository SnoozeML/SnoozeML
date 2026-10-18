//! Authentication and Role-Based Access Control (RBAC) middleware for SnoozeML.
//!
//! Provides dual-mode authentication via static API keys and cryptographically
//! verified JSON Web Tokens (JWT), along with role-based endpoint protection
//! (`Admin`, `Inference`, `ReadOnly`).

#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use axum::{
    extract::{Request, State},
    http::{header, HeaderValue, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use jsonwebtoken::{Algorithm, DecodingKey, Validation};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;
use thiserror::Error;

/// Role granted to an authenticated caller defining access boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// Read-only access to model metadata and listing endpoints.
    #[serde(alias = "read", alias = "readonly")]
    ReadOnly = 1,
    /// Permission to execute inference and read model metadata.
    #[serde(alias = "infer", alias = "inference", alias = "user")]
    Inference = 2,
    /// Unrestricted administrative access (load, unload, configure).
    #[serde(alias = "admin", alias = "administrator")]
    Admin = 3,
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ReadOnly => write!(f, "read_only"),
            Self::Inference => write!(f, "inference"),
            Self::Admin => write!(f, "admin"),
        }
    }
}

/// Authentication mechanism used by the client.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthMethod {
    /// Static API key passed via `X-API-Key` or `Authorization: Bearer <key>`.
    ApiKey,
    /// Cryptographically verified JSON Web Token (JWT).
    Jwt,
    /// Anonymous identity used when authentication is disabled.
    Anonymous,
}

/// Authenticated user identity attached to request extensions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthIdentity {
    /// Subject or caller identifier (e.g. username, service account, key name).
    pub subject: String,
    /// Granted role / permission level.
    pub role: Role,
    /// Method used to authenticate.
    pub method: AuthMethod,
}

impl AuthIdentity {
    /// Creates an anonymous administrator identity for when authentication is disabled.
    #[must_use]
    pub fn anonymous_admin() -> Self {
        Self {
            subject: "anonymous-admin".to_string(),
            role: Role::Admin,
            method: AuthMethod::Anonymous,
        }
    }
}

/// Static API key configuration entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApiKeyConfig {
    /// Secret API key string.
    pub key: String,
    /// Role assigned to callers presenting this key.
    pub role: Role,
    /// Descriptive name or identifier for this key (e.g. "ci-runner", "client-app").
    #[serde(default)]
    pub name: Option<String>,
}

/// Supported cryptographic signature algorithms for JWT verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "UPPERCASE")]
pub enum JwtAlgorithm {
    /// HMAC using SHA-256 (default).
    #[default]
    HS256,
    /// HMAC using SHA-384.
    HS384,
    /// HMAC using SHA-512.
    HS512,
}

impl From<JwtAlgorithm> for Algorithm {
    fn from(algo: JwtAlgorithm) -> Self {
        match algo {
            JwtAlgorithm::HS256 => Self::HS256,
            JwtAlgorithm::HS384 => Self::HS384,
            JwtAlgorithm::HS512 => Self::HS512,
        }
    }
}

/// Configuration settings for validating JSON Web Tokens (JWT).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JwtConfig {
    /// Shared secret key used to verify HMAC signatures.
    pub secret: String,
    /// Expected cryptographic algorithm (defaults to `HS256`).
    #[serde(default)]
    pub algorithm: JwtAlgorithm,
    /// Expected issuer claim (`iss`), if validation is required.
    #[serde(default)]
    pub issuer: Option<String>,
    /// Expected audience claim (`aud`), if validation is required.
    #[serde(default)]
    pub audience: Option<String>,
    /// Clock skew leeway in seconds (defaults to 60s).
    #[serde(default = "default_leeway_secs")]
    pub leeway_secs: u64,
}

fn default_leeway_secs() -> u64 {
    60
}

/// Master authentication configuration for SnoozeML.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct AuthConfig {
    /// Whether authentication enforcement is enabled.
    #[serde(default)]
    pub enabled: bool,
    /// Whether to require authentication on the `/metrics` endpoint (defaults to false).
    #[serde(default)]
    pub protect_metrics: bool,
    /// List of configured static API keys.
    #[serde(default)]
    pub api_keys: Vec<ApiKeyConfig>,
    /// Optional JWT verification configuration.
    #[serde(default)]
    pub jwt: Option<JwtConfig>,
}

/// Standard and custom claims extracted from a decoded JSON Web Token.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JwtClaims {
    /// Subject identifier.
    pub sub: String,
    /// Expiration timestamp (epoch seconds).
    pub exp: usize,
    /// Optional issued-at timestamp.
    #[serde(default)]
    pub iat: Option<usize>,
    /// Optional not-before timestamp.
    #[serde(default)]
    pub nbf: Option<usize>,
    /// Optional token issuer.
    #[serde(default)]
    pub iss: Option<String>,
    /// Optional token audience.
    #[serde(default)]
    pub aud: Option<String>,
    /// Optional single role claim.
    #[serde(default)]
    pub role: Option<Role>,
    /// Optional list of roles claim.
    #[serde(default)]
    pub roles: Option<Vec<Role>>,
}

impl JwtClaims {
    /// Resolves the highest role assigned to this token.
    #[must_use]
    pub fn effective_role(&self) -> Role {
        let mut highest = self.role.unwrap_or(Role::ReadOnly);
        if let Some(roles) = &self.roles {
            for &r in roles {
                if r > highest {
                    highest = r;
                }
            }
        }
        highest
    }
}

/// Errors occurring during authentication or token verification.
#[derive(Debug, Error)]
pub enum AuthError {
    /// Request did not contain any authentication credentials.
    #[error("Authentication credentials are missing")]
    MissingCredentials,

    /// Provided API key does not match any configured keys.
    #[error("Invalid API key")]
    InvalidApiKey,

    /// JWT verification failed due to signature mismatch or malformed token.
    #[error("Invalid JSON Web Token: {0}")]
    InvalidJwt(String),

    /// JWT has expired.
    #[error("JSON Web Token has expired")]
    ExpiredJwt,

    /// Caller possesses insufficient permissions for the requested route.
    #[error("Insufficient permissions: requires {required:?} role, caller has {actual:?}")]
    InsufficientRole {
        /// Required minimum role.
        required: Role,
        /// Actual caller role.
        actual: Role,
    },

    /// Internal error initializing or executing authentication.
    #[error("Authentication internal error: {0}")]
    Internal(String),
}

/// Internal pre-hashed API key entry for timing-safe matching.
struct HashedApiKey {
    hash: [u8; 32],
    role: Role,
    name: String,
}

/// Authentication engine verifying API keys and decoding JSON Web Tokens.
pub struct AuthService {
    enabled: bool,
    protect_metrics: bool,
    api_keys: Vec<HashedApiKey>,
    jwt_config: Option<(JwtConfig, DecodingKey, Validation)>,
}

impl std::fmt::Debug for AuthService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AuthService")
            .field("enabled", &self.enabled)
            .field("protect_metrics", &self.protect_metrics)
            .field("api_keys_count", &self.api_keys.len())
            .field("jwt_configured", &self.jwt_config.is_some())
            .finish()
    }
}

impl AuthService {
    /// Creates a new [`AuthService`] from the provided [`AuthConfig`].
    ///
    /// # Errors
    /// Returns [`AuthError::Internal`] if JWT keys or validation rules cannot be created.
    ///
    /// # Panics
    /// This function does not panic.
    pub fn new(config: AuthConfig) -> Result<Self, AuthError> {
        let mut hashed_keys = Vec::with_capacity(config.api_keys.len());
        for key_cfg in config.api_keys {
            if key_cfg.key.trim().is_empty() {
                return Err(AuthError::Internal(
                    "Configured API key cannot be empty or whitespace".to_string(),
                ));
            }
            let digest = Sha256::digest(key_cfg.key.as_bytes());
            let mut hash = [0u8; 32];
            hash.copy_from_slice(&digest);
            let name = key_cfg.name.unwrap_or_else(|| "api-key".to_string());
            hashed_keys.push(HashedApiKey {
                hash,
                role: key_cfg.role,
                name,
            });
        }

        let jwt_config = if let Some(jwt) = config.jwt {
            let key = DecodingKey::from_secret(jwt.secret.as_bytes());
            let mut validation = Validation::new(jwt.algorithm.into());
            validation.leeway = jwt.leeway_secs;

            if let Some(ref iss) = jwt.issuer {
                validation.set_issuer(&[iss]);
            }
            if let Some(ref aud) = jwt.audience {
                validation.set_audience(&[aud]);
            } else {
                validation.validate_aud = false;
            }

            Some((jwt, key, validation))
        } else {
            None
        };

        Ok(Self {
            enabled: config.enabled,
            protect_metrics: config.protect_metrics,
            api_keys: hashed_keys,
            jwt_config,
        })
    }

    /// Creates an authentication service with authentication disabled.
    #[must_use]
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            protect_metrics: false,
            api_keys: Vec::new(),
            jwt_config: None,
        }
    }

    /// Checks whether authentication enforcement is active.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Checks whether the `/metrics` endpoint is protected.
    #[must_use]
    pub fn protect_metrics(&self) -> bool {
        self.protect_metrics
    }

    /// Authenticates an incoming HTTP request using headers.
    ///
    /// Supports:
    /// - `X-API-Key: <key>`
    /// - `Authorization: Bearer <token>` (checked as API key first, then as JWT)
    ///
    /// # Errors
    /// Returns [`AuthError`] if credentials are missing, malformed, or invalid.
    ///
    /// # Panics
    /// This function does not panic.
    pub fn authenticate_request<B>(&self, req: &Request<B>) -> Result<AuthIdentity, AuthError> {
        if !self.enabled {
            return Ok(AuthIdentity::anonymous_admin());
        }

        // 1. Check X-API-Key header
        if let Some(api_key_header) = req.headers().get("x-api-key") {
            let key_str = api_key_header
                .to_str()
                .map_err(|_| AuthError::InvalidApiKey)?;
            return self.authenticate_api_key(key_str.trim());
        }

        // 2. Check Authorization header
        if let Some(auth_header) = req.headers().get(header::AUTHORIZATION) {
            let auth_str = auth_header
                .to_str()
                .map_err(|_| AuthError::MissingCredentials)?;

            let token = if let Some(stripped) = auth_str.strip_prefix("Bearer ") {
                stripped.trim()
            } else if let Some(stripped) = auth_str.strip_prefix("bearer ") {
                stripped.trim()
            } else {
                return Err(AuthError::MissingCredentials);
            };

            // Check if token matches a configured API key
            if let Ok(identity) = self.authenticate_api_key(token) {
                return Ok(identity);
            }

            // Fall back to JWT validation
            if self.jwt_config.is_some() {
                return self.authenticate_jwt(token);
            }

            return Err(AuthError::InvalidApiKey);
        }

        Err(AuthError::MissingCredentials)
    }

    /// Verifies an API key against configured keys using constant-time comparison.
    ///
    /// # Errors
    /// Returns [`AuthError::InvalidApiKey`] if no configured key matches.
    ///
    /// # Panics
    /// This function does not panic.
    pub fn authenticate_api_key(&self, key: &str) -> Result<AuthIdentity, AuthError> {
        let digest = Sha256::digest(key.as_bytes());
        let mut matched: Option<AuthIdentity> = None;

        for entry in &self.api_keys {
            let matches: bool = entry.hash.ct_eq(&digest[..]).into();
            if matches && matched.is_none() {
                matched = Some(AuthIdentity {
                    subject: entry.name.clone(),
                    role: entry.role,
                    method: AuthMethod::ApiKey,
                });
            }
        }

        matched.ok_or(AuthError::InvalidApiKey)
    }

    /// Validates and decodes a JSON Web Token.
    ///
    /// # Errors
    /// Returns [`AuthError::InvalidJwt`] or [`AuthError::ExpiredJwt`] if token validation fails.
    ///
    /// # Panics
    /// This function does not panic.
    pub fn authenticate_jwt(&self, token: &str) -> Result<AuthIdentity, AuthError> {
        let (_cfg, key, validation) = self
            .jwt_config
            .as_ref()
            .ok_or_else(|| AuthError::Internal("JWT verification is not configured".to_string()))?;

        match jsonwebtoken::decode::<JwtClaims>(token, key, validation) {
            Ok(token_data) => {
                let role = token_data.claims.effective_role();
                Ok(AuthIdentity {
                    subject: token_data.claims.sub,
                    role,
                    method: AuthMethod::Jwt,
                })
            }
            Err(err) => match err.kind() {
                jsonwebtoken::errors::ErrorKind::ExpiredSignature => Err(AuthError::ExpiredJwt),
                _ => Err(AuthError::InvalidJwt(err.to_string())),
            },
        }
    }
}

/// Axum middleware function that authenticates requests and attaches [`AuthIdentity`].
///
/// # Panics
/// This function does not panic.
pub async fn auth_middleware(
    State(state): State<crate::server::http::AppState>,
    mut req: Request,
    next: Next,
) -> Response {
    let auth_service = match &state.auth {
        Some(svc) if svc.is_enabled() => svc,
        _ => {
            req.extensions_mut().insert(AuthIdentity::anonymous_admin());
            return next.run(req).await;
        }
    };

    match auth_service.authenticate_request(&req) {
        Ok(identity) => {
            tracing::debug!(
                subject = %identity.subject,
                role = %identity.role,
                method = ?identity.method,
                "Request authenticated successfully"
            );
            req.extensions_mut().insert(identity);
            next.run(req).await
        }
        Err(err) => {
            tracing::warn!(error = %err, "Authentication failed");
            let (status, msg) = match err {
                AuthError::MissingCredentials => (
                    StatusCode::UNAUTHORIZED,
                    "Authentication credentials are missing. Provide 'X-API-Key' or 'Authorization: Bearer <token>' header.".to_string(),
                ),
                AuthError::ExpiredJwt => (
                    StatusCode::UNAUTHORIZED,
                    "JSON Web Token has expired".to_string(),
                ),
                AuthError::InvalidJwt(ref details) => (
                    StatusCode::UNAUTHORIZED,
                    format!("Invalid JSON Web Token: {details}"),
                ),
                AuthError::InvalidApiKey => (
                    StatusCode::UNAUTHORIZED,
                    "Invalid API key provided".to_string(),
                ),
                AuthError::InsufficientRole { required, actual } => (
                    StatusCode::FORBIDDEN,
                    format!("Forbidden: endpoint requires {required:?} role, but caller has {actual:?}"),
                ),
                AuthError::Internal(ref details) => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Authentication internal error: {details}"),
                ),
            };

            let mut resp = (
                status,
                Json(crate::server::http::ErrorResponse { error: msg }),
            )
                .into_response();
            if status == StatusCode::UNAUTHORIZED {
                resp.headers_mut().insert(
                    header::WWW_AUTHENTICATE,
                    HeaderValue::from_static("Bearer realm=\"snoozeml\""),
                );
            }
            resp
        }
    }
}

/// Creates a middleware function enforcing a minimum required [`Role`].
///
/// # Panics
/// The returned middleware does not panic.
pub fn require_role(
    min_role: Role,
) -> impl Fn(Request, Next) -> std::pin::Pin<Box<dyn std::future::Future<Output = Response> + Send>>
       + Clone {
    move |req: Request, next: Next| {
        Box::pin(async move {
            let identity = req.extensions().get::<AuthIdentity>();
            match identity {
                Some(id) if id.role >= min_role => next.run(req).await,
                Some(id) => {
                    tracing::warn!(
                        caller = %id.subject,
                        caller_role = %id.role,
                        required_role = %min_role,
                        "Caller forbidden from accessing route"
                    );
                    (
                        StatusCode::FORBIDDEN,
                        Json(crate::server::http::ErrorResponse {
                            error: format!(
                                "Forbidden: requires {min_role:?} role, caller has {actual:?}",
                                actual = id.role
                            ),
                        }),
                    )
                        .into_response()
                }
                None => (
                    StatusCode::UNAUTHORIZED,
                    Json(crate::server::http::ErrorResponse {
                        error: "Unauthorized: authentication required".to_string(),
                    }),
                )
                    .into_response(),
            }
        })
    }
}
