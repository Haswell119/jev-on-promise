//! HTTP integration tests for the axum server. Requests go straight into the
//! router through `tower::ServiceExt::oneshot`: no sockets, no network.

use axum::body::Body;
use axum::http::{header, HeaderMap, Request, StatusCode};
use axum::Router;
use http_body_util::BodyExt;
use serde_json::{json, Value};
use sextant_core::Engine;
use std::sync::{Arc, OnceLock};
use tower::ServiceExt;

const EXAMPLE: &str = include_str!("../../../examples/request_basic.json");
const DEFAULT_MAX_BODY: usize = 4 * 1024 * 1024;

/// One engine per test binary: building one parses the embedded lexicon.
fn engine() -> Arc<Engine> {
    static ENGINE: OnceLock<Arc<Engine>> = OnceLock::new();
    ENGINE.get_or_init(|| Arc::new(Engine::default_embedded())).clone()
}

fn app() -> Router {
    sextant_server::app(engine(), DEFAULT_MAX_BODY)
}

async fn send(app: Router, req: Request<Body>) -> (StatusCode, HeaderMap, String) {
    let resp = app.oneshot(req).await.expect("router is infallible");
    let status = resp.status();
    let headers = resp.headers().clone();
    let bytes = resp.into_body().collect().await.expect("body").to_bytes();
    (status, headers, String::from_utf8_lossy(&bytes).into_owned())
}

fn get(uri: &str) -> Request<Body> {
    Request::builder().method("GET").uri(uri).body(Body::empty()).unwrap()
}

fn post_json(uri: &str, body: impl Into<Body>) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(uri)
        .header(header::CONTENT_TYPE, "application/json")
        .body(body.into())
        .unwrap()
}

fn parse(body: &str) -> Value {
    serde_json::from_str(body).unwrap_or_else(|e| panic!("response is not JSON ({e}): {body}"))
}

fn content_type(headers: &HeaderMap) -> String {
    headers.get(header::CONTENT_TYPE).map(|v| v.to_str().unwrap_or("").to_string()).unwrap_or_default()
}

#[tokio::test]
async fn health_reports_ok_and_non_neural() {
    let (status, headers, body) = send(app(), get("/health")).await;
    assert_eq!(status, StatusCode::OK);
    assert!(content_type(&headers).starts_with("application/json"), "{headers:?}");
    let v = parse(&body);
    assert_eq!(v["status"], "ok");
    assert_eq!(v["neural"], false);
    assert_eq!(v["network_required"], false);
    assert_eq!(v["model"], "sextant-1");
    assert!(v["version"].is_string());
    assert!(v["uptime_seconds"].is_u64());

    let (status, _, body) = send(app(), get("/healthz")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(parse(&body)["status"], "ok");
}

#[tokio::test]
async fn models_lists_sextant_1() {
    let (status, _, body) = send(app(), get("/v1/models")).await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.contains("sextant-1"));
    let v = parse(&body);
    assert_eq!(v["object"], "list");
    let data = v["data"].as_array().expect("data array");
    assert!(data.iter().any(|m| m["id"] == "sextant-1" && m["name"] == "sextant-1"));
    let card = &data[0];
    assert_eq!(card["neural"], false);
    assert_eq!(card["max_choice_options"], 255);
    assert_eq!(card["max_score_levels"], 10);
    assert!(card["aliases"].as_array().unwrap().iter().any(|a| a == "sextant-latest"));
    assert_eq!(v["models"], v["data"]);
}

#[tokio::test]
async fn systemone_answers_the_basic_example() {
    let (status, headers, body) = send(app(), post_json("/v1/systemone", EXAMPLE)).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(content_type(&headers).starts_with("application/json"));
    let v = parse(&body);
    assert_eq!(v["model"], "sextant-1");
    let answers = v["answers"].as_object().expect("answers object");
    let expected: Vec<&str> = vec!["department", "is_urgent", "frustration"];
    assert_eq!(answers.keys().map(String::as_str).collect::<Vec<_>>(), expected);

    let department = &answers["department"];
    assert_eq!(department["type"], "choice");
    let probs = department["probabilities"].as_object().unwrap();
    assert_eq!(probs.keys().map(String::as_str).collect::<Vec<_>>(), vec!["billing", "technical", "sales"]);
    assert!(probs.contains_key(department["choice"].as_str().unwrap()));
    let sum: f64 = probs.values().map(|p| p.as_f64().unwrap()).sum();
    assert!((sum - 1.0).abs() < 1e-6);
    assert!(department["confidence"].as_f64().unwrap() <= 1.0);
    assert!(department.get("explain").is_none(), "no explain block unless requested");

    let urgent = &answers["is_urgent"];
    assert_eq!(urgent["type"], "noul");
    let p = urgent["noul"].as_f64().unwrap();
    assert!((0.0..=1.0).contains(&p));

    let frustration = &answers["frustration"];
    assert_eq!(frustration["type"], "score");
    assert_eq!(frustration["legend"], json!({ "0": "Calm", "1": "Frustrated", "2": "Very angry" }));
    let score = frustration["score"].as_f64().unwrap();
    assert!((0.0..=2.0).contains(&score));
    assert_eq!(frustration["probabilities"].as_object().unwrap().len(), 3);

    assert!(v["usage"]["input_tokens"].as_u64().unwrap() > 0);
    assert!(v["usage"]["output_tokens"].as_u64().unwrap() > 0);
}

#[tokio::test]
async fn malformed_json_is_422_invalid_request() {
    let (status, headers, body) = send(app(), post_json("/v1/systemone", "{not json")).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
    assert!(content_type(&headers).starts_with("application/json"));
    let v = parse(&body);
    assert_eq!(v["error"]["code"], "invalid_request");
    assert_eq!(v["error"]["status"], 422);
    assert!(v["error"]["message"].as_str().unwrap().contains("malformed"));

    // Valid JSON, wrong shape.
    let (status, _, body) = send(app(), post_json("/v1/systemone", r#"{"model": "sextant-1"}"#)).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(parse(&body)["error"]["code"], "invalid_request");

    // Empty body.
    let (status, _, body) = send(app(), post_json("/v1/systemone", "")).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(parse(&body)["error"]["code"], "invalid_request");
}

#[tokio::test]
async fn unknown_model_is_404() {
    let mut req: Value = serde_json::from_str(EXAMPLE).unwrap();
    req["model"] = json!("gpt-4");
    let (status, _, body) = send(app(), post_json("/v1/systemone", req.to_string())).await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
    let v = parse(&body);
    assert_eq!(v["error"]["code"], "unknown_model");
    assert_eq!(v["error"]["field"], "model");
}

#[tokio::test]
async fn validation_errors_are_422_with_a_field_path() {
    let req = json!({
        "model": "sextant-1",
        "state": "x",
        "questions": { "only": { "type": "choice", "instructions": "Pick", "criteria": { "one": null } } }
    });
    let (status, _, body) = send(app(), post_json("/v1/systemone", req.to_string())).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
    let v = parse(&body);
    assert_eq!(v["error"]["code"], "invalid_request");
    assert_eq!(v["error"]["field"], "questions.only.criteria");
}

#[tokio::test]
async fn oversized_body_is_413() {
    let small_limit = sextant_server::app(engine(), 1024);
    let mut req: Value = serde_json::from_str(EXAMPLE).unwrap();
    req["state"] = json!("x".repeat(5000));
    let body = req.to_string();
    assert!(body.len() > 1024);
    let (status, _, _) = send(small_limit, post_json("/v1/systemone", body)).await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);

    // The same request is fine under the default limit.
    let (status, _, _) = send(app(), post_json("/v1/systemone", req.to_string())).await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn unknown_route_is_404_json() {
    let (status, headers, body) = send(app(), get("/nope")).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(content_type(&headers).starts_with("application/json"));
    let v = parse(&body);
    assert_eq!(v["error"]["code"], "not_found");
    assert!(v["error"]["message"].as_str().unwrap().contains("openapi"));

    let (status, _, body) = send(app(), post_json("/v2/systemone", EXAMPLE)).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(parse(&body)["error"]["code"], "not_found");
}

#[tokio::test]
async fn wrong_method_is_405() {
    let (status, _, _) = send(app(), get("/v1/systemone")).await;
    assert_eq!(status, StatusCode::METHOD_NOT_ALLOWED);
}

#[tokio::test]
async fn explain_query_adds_explain_blocks() {
    let (status, _, body) = send(app(), post_json("/v1/systemone?explain=true", EXAMPLE)).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let v = parse(&body);
    for (id, a) in v["answers"].as_object().unwrap() {
        let ex = a.get("explain").unwrap_or_else(|| panic!("answer {id} has no explain block"));
        assert!(ex["family"].is_string());
        assert!(ex["path"].is_string());
        assert!(ex["top_evidence"].is_array());
        assert!(ex["features"].is_object());
        assert!(ex["calibration"].is_object());
    }

    // The answers themselves are identical with and without explain.
    let (_, _, plain) = send(app(), post_json("/v1/systemone", EXAMPLE)).await;
    let mut explained = v.clone();
    for a in explained["answers"].as_object_mut().unwrap().values_mut() {
        a.as_object_mut().unwrap().remove("explain");
    }
    assert_eq!(explained, parse(&plain));

    // `explain=false` and a missing parameter both mean "no explain".
    let (status, _, body) = send(app(), post_json("/v1/systemone?explain=false", EXAMPLE)).await;
    assert_eq!(status, StatusCode::OK);
    assert!(parse(&body)["answers"].as_object().unwrap().values().all(|a| a.get("explain").is_none()));
}

#[tokio::test]
async fn openapi_yaml_is_served() {
    let (status, headers, body) = send(app(), get("/openapi.yaml")).await;
    assert_eq!(status, StatusCode::OK);
    assert!(content_type(&headers).contains("yaml"), "{headers:?}");
    assert!(body.contains("openapi:"));
    assert!(body.contains("/v1/systemone"));
}
