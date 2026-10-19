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

impl Role {
    /// Parses a role from a string slice (case-insensitive: "admin", "infer", "read_only", etc.).
    #[must_use]
    pub fn parse_str(s: &str) -> Self {
        match s.trim().to_lowercase().as_str() {
            "admin" | "administrator" => Self::Admin,
            "read" | "readonly" | "read_only" | "ro" => Self::ReadOnly,
            _ => Self::Inference,
        }
    }
}

impl std::str::FromStr for Role {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self::parse_str(s))
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key: Option<String>,
    /// Environment variable name from which to load the API key.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key_env: Option<String>,
    /// Role assigned to callers presenting this key.
    pub role: Role,
    /// Descriptive name or identifier for this key (e.g. "ci-runner", "client-app").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl ApiKeyConfig {
    /// Creates a new API key configuration from a raw key string.
    #[must_use]
    pub fn new(key: impl Into<String>, role: Role) -> Self {
        Self {
            key: Some(key.into()),
            key_env: None,
            role,
            name: None,
        }
    }

    /// Creates a new API key configuration resolved from an environment variable name.
    #[must_use]
    pub fn from_env(key_env: impl Into<String>, role: Role) -> Self {
        Self {
            key: None,
            key_env: Some(key_env.into()),
            role,
            name: None,
        }
    }

    /// Sets the descriptive name for this API key.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Resolves the actual secret key, checking `key_env` first, then `$VAR` expansion in `key`, then raw `key`.
    #[must_use]
    pub fn resolve_key(&self) -> Option<String> {
        if let Some(ref env_var) = self.key_env {
            if let Ok(val) = std::env::var(env_var) {
                let trimmed = val.trim();
                if !trimmed.is_empty() {
                    return Some(trimmed.to_string());
                }
            }
        }
        if let Some(ref k) = self.key {
            let trimmed = k.trim();
            if let Some(var_name) = trimmed.strip_prefix('$') {
                if let Ok(val) = std::env::var(var_name) {
                    let env_trimmed = val.trim();
                    if !env_trimmed.is_empty() {
                        return Some(env_trimmed.to_string());
                    }
                }
            }
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
        None
    }
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub secret: Option<String>,
    /// Environment variable name from which to load the HMAC secret (defaults to `SNOOZE_JWT_SECRET`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub secret_env: Option<String>,
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

impl Default for JwtConfig {
    fn default() -> Self {
        Self {
            secret: None,
            secret_env: None,
            algorithm: JwtAlgorithm::HS256,
            issuer: None,
            audience: None,
            leeway_secs: 60,
        }
    }
}

impl JwtConfig {
    /// Creates a new JWT configuration with a raw shared secret.
    #[must_use]
    pub fn new(secret: impl Into<String>) -> Self {
        Self {
            secret: Some(secret.into()),
            secret_env: None,
            algorithm: JwtAlgorithm::HS256,
            issuer: None,
            audience: None,
            leeway_secs: 60,
        }
    }

    /// Creates a new JWT configuration resolved from an environment variable name.
    #[must_use]
    pub fn from_env(secret_env: impl Into<String>) -> Self {
        Self {
            secret: None,
            secret_env: Some(secret_env.into()),
            algorithm: JwtAlgorithm::HS256,
            issuer: None,
            audience: None,
            leeway_secs: 60,
        }
    }

    /// Resolves the HMAC secret string from `secret_env`, `$VAR` in `secret`, raw `secret`, or `SNOOZE_JWT_SECRET`.
    #[must_use]
    pub fn resolve_secret(&self) -> Option<String> {
        if let Some(ref env_var) = self.secret_env {
            if let Ok(val) = std::env::var(env_var) {
                let trimmed = val.trim();
                if !trimmed.is_empty() {
                    return Some(trimmed.to_string());
                }
            }
        }
        if let Some(ref s) = self.secret {
            let trimmed = s.trim();
            if let Some(var_name) = trimmed.strip_prefix('$') {
                if let Ok(val) = std::env::var(var_name) {
                    let env_trimmed = val.trim();
                    if !env_trimmed.is_empty() {
                        return Some(env_trimmed.to_string());
                    }
                }
            }
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
        if let Ok(val) = std::env::var("SNOOZE_JWT_SECRET") {
            let trimmed = val.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
        None
    }
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

impl AuthConfig {
    /// Populates API keys and JWT configuration from environment variables if not already specified.
    ///
    /// Inspects the following environment variables:
    /// - `SNOOZE_API_KEY`: Single API key string (e.g. `sk-admin123`) or `<role>:<key>`. If no role is specified, defaults to `Admin`.
    /// - `SNOOZE_API_KEYS`: Comma-separated list of `<role>:<key>` or keys.
    /// - `SNOOZE_API_KEY_ADMIN` / `SNOOZE_ADMIN_API_KEY`: API key with `Admin` role.
    /// - `SNOOZE_API_KEY_INFER` / `SNOOZE_API_KEY_INFERENCE`: API key with `Inference` role.
    /// - `SNOOZE_API_KEY_READONLY` / `SNOOZE_API_KEY_RO`: API key with `ReadOnly` role.
    /// - `SNOOZE_JWT_SECRET`: Shared HMAC secret for verifying JSON Web Tokens.
    pub fn load_from_env(&mut self) {
        // 1. SNOOZE_API_KEY
        if let Ok(val) = std::env::var("SNOOZE_API_KEY") {
            let trimmed = val.trim();
            if !trimmed.is_empty() {
                let (role, key, name) = if let Some((r, k)) = trimmed.split_once(':') {
                    let parsed_role = Role::parse_str(r);
                    (
                        parsed_role,
                        k.trim(),
                        format!("env-{}", r.trim().to_lowercase()),
                    )
                } else {
                    (Role::Admin, trimmed, "env-admin".to_string())
                };
                if !self
                    .api_keys
                    .iter()
                    .any(|k| k.resolve_key().as_deref() == Some(key))
                {
                    self.api_keys.push(ApiKeyConfig {
                        key: Some(key.to_string()),
                        key_env: None,
                        role,
                        name: Some(name),
                    });
                }
            }
        }

        // 2. SNOOZE_API_KEYS (comma-separated list of keys, optionally role-prefixed)
        if let Ok(val) = std::env::var("SNOOZE_API_KEYS") {
            for entry in val.split(',') {
                let trimmed = entry.trim();
                if !trimmed.is_empty() {
                    let (role, key, name) = if let Some((r, k)) = trimmed.split_once(':') {
                        let parsed_role = Role::parse_str(r);
                        (
                            parsed_role,
                            k.trim(),
                            format!("env-{}", r.trim().to_lowercase()),
                        )
                    } else {
                        (Role::Admin, trimmed, "env-admin".to_string())
                    };
                    if !self
                        .api_keys
                        .iter()
                        .any(|k| k.resolve_key().as_deref() == Some(key))
                    {
                        self.api_keys.push(ApiKeyConfig {
                            key: Some(key.to_string()),
                            key_env: None,
                            role,
                            name: Some(name),
                        });
                    }
                }
            }
        }

        // 3. SNOOZE_API_KEY_ADMIN / SNOOZE_ADMIN_API_KEY
        if let Ok(val) =
            std::env::var("SNOOZE_API_KEY_ADMIN").or_else(|_| std::env::var("SNOOZE_ADMIN_API_KEY"))
        {
            let trimmed = val.trim();
            if !trimmed.is_empty()
                && !self
                    .api_keys
                    .iter()
                    .any(|k| k.resolve_key().as_deref() == Some(trimmed))
            {
                self.api_keys.push(ApiKeyConfig {
                    key: Some(trimmed.to_string()),
                    key_env: None,
                    role: Role::Admin,
                    name: Some("env-admin-role".to_string()),
                });
            }
        }

        // 4. SNOOZE_API_KEY_INFER / SNOOZE_API_KEY_INFERENCE
        if let Ok(val) = std::env::var("SNOOZE_API_KEY_INFER")
            .or_else(|_| std::env::var("SNOOZE_API_KEY_INFERENCE"))
        {
            let trimmed = val.trim();
            if !trimmed.is_empty()
                && !self
                    .api_keys
                    .iter()
                    .any(|k| k.resolve_key().as_deref() == Some(trimmed))
            {
                self.api_keys.push(ApiKeyConfig {
                    key: Some(trimmed.to_string()),
                    key_env: None,
                    role: Role::Inference,
                    name: Some("env-infer-role".to_string()),
                });
            }
        }

        // 5. SNOOZE_API_KEY_READONLY / SNOOZE_API_KEY_RO
        if let Ok(val) =
            std::env::var("SNOOZE_API_KEY_READONLY").or_else(|_| std::env::var("SNOOZE_API_KEY_RO"))
        {
            let trimmed = val.trim();
            if !trimmed.is_empty()
                && !self
                    .api_keys
                    .iter()
                    .any(|k| k.resolve_key().as_deref() == Some(trimmed))
            {
                self.api_keys.push(ApiKeyConfig {
                    key: Some(trimmed.to_string()),
                    key_env: None,
                    role: Role::ReadOnly,
                    name: Some("env-readonly-role".to_string()),
                });
            }
        }

        // 6. SNOOZE_JWT_SECRET
        if self.jwt.is_none() {
            if let Ok(val) = std::env::var("SNOOZE_JWT_SECRET") {
                let trimmed = val.trim();
                if !trimmed.is_empty() {
                    self.jwt = Some(JwtConfig {
                        secret: Some(trimmed.to_string()),
                        secret_env: Some("SNOOZE_JWT_SECRET".to_string()),
                        algorithm: JwtAlgorithm::HS256,
                        issuer: None,
                        audience: None,
                        leeway_secs: 60,
                    });
                }
            }
        }
    }
}

/// Standard and custom claims extracted from a decoded JSON Web Token.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JwtClaims {
    /// Subject identifier.
    pub sub: String,
    /// Expiration timestamp (epoch seconds).
    pub exp: usize,
    /// Optional issued-at timestamp.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub iat: Option<usize>,
    /// Optional not-before timestamp.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nbf: Option<usize>,
    /// Optional token issuer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub iss: Option<String>,
    /// Optional token audience.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aud: Option<String>,
    /// Optional single role claim.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<Role>,
    /// Optional list of roles claim.
    #[serde(default, skip_serializing_if = "Option::is_none")]
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
    pub fn new(mut config: AuthConfig) -> Result<Self, AuthError> {
        if config.enabled {
            config.load_from_env();
        }

        let mut hashed_keys = Vec::with_capacity(config.api_keys.len());
        for key_cfg in config.api_keys {
            let name = key_cfg.name.as_deref().unwrap_or("api-key").to_string();
            let resolved_key = key_cfg.resolve_key().ok_or_else(|| {
                AuthError::Internal(format!(
                    "API key '{name}' cannot be resolved: key is empty or environment variable is not set"
                ))
            })?;
            if resolved_key.trim().is_empty() {
                return Err(AuthError::Internal(
                    "Configured API key cannot be empty or whitespace".to_string(),
                ));
            }
            if key_cfg.key.is_some()
                && key_cfg.key_env.is_none()
                && !key_cfg.key.as_deref().unwrap_or("").starts_with('$')
            {
                tracing::warn!(
                    key_name = %name,
                    "API key is stored in plaintext in the configuration file; consider using environment variables (e.g. SNOOZE_API_KEY or key_env) instead."
                );
            }
            let digest = Sha256::digest(resolved_key.as_bytes());
            let mut hash = [0u8; 32];
            hash.copy_from_slice(&digest);
            hashed_keys.push(HashedApiKey {
                hash,
                role: key_cfg.role,
                name,
            });
        }

        let jwt_config = if let Some(jwt) = config.jwt {
            let secret = jwt.resolve_secret().ok_or_else(|| {
                AuthError::Internal(
                    "JWT secret not provided; configure 'secret_env' or set environment variable SNOOZE_JWT_SECRET".to_string(),
                )
            })?;
            if jwt.secret.is_some()
                && jwt.secret_env.is_none()
                && !jwt.secret.as_deref().unwrap_or("").starts_with('$')
            {
                tracing::warn!(
                    "JWT secret is stored in plaintext in the configuration file; consider using environment variable SNOOZE_JWT_SECRET or secret_env instead."
                );
            }
            let key = DecodingKey::from_secret(secret.as_bytes());
            let mut validation = Validation::new(jwt.algorithm.into());
            validation.leeway = jwt.leeway_secs;
            validation.validate_nbf = true;

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

        if config.enabled && hashed_keys.is_empty() && jwt_config.is_none() {
            tracing::warn!(
                "Authentication is enabled ([auth] enabled = true), but no API keys or JWT secrets were configured. Set SNOOZE_API_KEY or SNOOZE_JWT_SECRET in the environment, or pass --api-key on the CLI."
            );
        }

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
