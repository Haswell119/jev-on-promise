use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Structured API error. Mirrors the HTTP error body returned by the server.
#[derive(Debug, Clone, Error, Serialize, Deserialize, PartialEq, Eq)]
#[error("{code}: {message}")]
pub struct ApiError {
    /// HTTP status this error maps to.
    pub status: u16,
    /// Machine readable error code (e.g. `invalid_request`, `unknown_model`).
    pub code: String,
    /// Human readable message.
    pub message: String,
    /// JSON-pointer-ish path to the offending field, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub field: Option<String>,
}

impl ApiError {
    pub fn invalid(field: impl Into<String>, message: impl Into<String>) -> Self {
        ApiError { status: 422, code: "invalid_request".into(), message: message.into(), field: Some(field.into()) }
    }

    pub fn unknown_model(model: &str) -> Self {
        ApiError {
            status: 404,
            code: "unknown_model".into(),
            message: format!("unknown model `{model}`; call GET /v1/models for the list"),
            field: Some("model".into()),
        }
    }

    pub fn too_large(message: impl Into<String>) -> Self {
        ApiError { status: 413, code: "payload_too_large".into(), message: message.into(), field: None }
    }

    pub fn internal(message: impl Into<String>) -> Self {
        ApiError { status: 500, code: "internal_error".into(), message: message.into(), field: None }
    }
}
