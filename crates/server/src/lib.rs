//! HTTP server for the Sextant engine (axum). Routes:
//! `GET /health`, `GET /v1/models`, `POST /v1/systemone`, `GET /openapi.yaml`.

use axum::{
    body::Bytes,
    extract::{Query, State},
    http::{header, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use sextant_core::api::{validate::MODEL_ID, ApiError, SystemOneRequest};
use sextant_core::Engine;
use std::sync::Arc;
use std::time::Instant;
use tower_http::limit::RequestBodyLimitLayer;

pub const OPENAPI_YAML: &str = include_str!("../../../docs/openapi.yaml");

#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<Engine>,
    pub started: Instant,
    pub max_body_bytes: usize,
}

#[derive(Debug, Serialize)]
pub struct Health {
    pub status: &'static str,
    pub version: &'static str,
    pub model: &'static str,
    pub uptime_seconds: u64,
    pub neural: bool,
    pub network_required: bool,
}

#[derive(Debug, Deserialize, Default)]
pub struct EvalQuery {
    #[serde(default)]
    pub explain: Option<bool>,
}

#[derive(Debug, Serialize)]
struct ErrorBody {
    error: ApiError,
}

struct AppError(ApiError);

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let status = StatusCode::from_u16(self.0.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        (status, Json(ErrorBody { error: self.0 })).into_response()
    }
}

async fn health(State(st): State<AppState>) -> Json<Health> {
    Json(Health {
        status: "ok",
        version: sextant_core::VERSION,
        model: MODEL_ID,
        uptime_seconds: st.started.elapsed().as_secs(),
        neural: false,
        network_required: false,
    })
}

async fn models(State(st): State<AppState>) -> Json<serde_json::Value> {
    let cards = st.engine.models();
    // Mirrors the public list shape (`models: [{name, description, release_date}]`)
    // while also exposing the richer card fields.
    let models: Vec<serde_json::Value> = cards
        .iter()
        .map(|c| {
            serde_json::json!({
                "name": c.id,
                "id": c.id,
                "object": c.object,
                "description": c.description,
                "release_date": "2026-09-22",
                "aliases": c.aliases,
                "max_choice_options": c.max_choice_options,
                "max_score_levels": c.max_score_levels,
                "neural": c.neural,
            })
        })
        .collect();
    Json(serde_json::json!({ "object": "list", "models": models, "data": models }))
}

async fn openapi() -> Response {
    let mut r = OPENAPI_YAML.into_response();
    r.headers_mut().insert(header::CONTENT_TYPE, HeaderValue::from_static("application/yaml; charset=utf-8"));
    r
}

async fn systemone(
    State(st): State<AppState>,
    Query(q): Query<EvalQuery>,
    body: Bytes,
) -> Result<Json<sextant_core::SystemOneResponse>, AppError> {
    if body.is_empty() {
        return Err(AppError(ApiError::invalid("body", "empty request body")));
    }
    let mut req: SystemOneRequest = serde_json::from_slice(&body)
        .map_err(|e| AppError(ApiError::invalid("body", format!("malformed JSON body: {e}"))))?;
    if q.explain == Some(true) {
        req.explain = Some(true);
    }
    let engine = st.engine.clone();
    // CPU-bound work runs on the blocking pool so the async runtime keeps serving.
    let resp = tokio::task::spawn_blocking(move || engine.evaluate(&req))
        .await
        .map_err(|e| AppError(ApiError::internal(format!("worker panicked: {e}"))))?;
    match resp {
        Ok(r) => Ok(Json(r)),
        Err(e) => Err(AppError(e)),
    }
}

async fn not_found() -> AppError {
    AppError(ApiError {
        status: 404,
        code: "not_found".into(),
        message: "unknown route; see GET /openapi.yaml".into(),
        field: None,
    })
}

/// Build the router.
pub fn app(engine: Arc<Engine>, max_body_bytes: usize) -> Router {
    let state = AppState { engine, started: Instant::now(), max_body_bytes };
    Router::new()
        .route("/health", get(health))
        .route("/healthz", get(health))
        .route("/v1/models", get(models))
        .route("/v1/systemone", post(systemone))
        .route("/openapi.yaml", get(openapi))
        .fallback(not_found)
        .layer(RequestBodyLimitLayer::new(max_body_bytes))
        .layer(tower_http::timeout::TimeoutLayer::with_status_code(
            StatusCode::REQUEST_TIMEOUT,
            std::time::Duration::from_secs(60),
        ))
        .with_state(state)
}

/// Serve until Ctrl-C.
pub async fn serve(engine: Arc<Engine>, addr: std::net::SocketAddr, max_body_bytes: usize) -> std::io::Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(%addr, "sextant listening");
    axum::serve(listener, app(engine, max_body_bytes)).with_graceful_shutdown(shutdown_signal()).await
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    tracing::info!("shutdown signal received");
}
