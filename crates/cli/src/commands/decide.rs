use super::common::{build_engine, read_input};
use sextant_core::api::validate::{validate_request, Limits};
use sextant_core::SystemOneRequest;
use std::path::PathBuf;

pub fn run(model_dir: Option<PathBuf>, threads: usize, file: Option<PathBuf>, explain: bool, pretty: bool) -> i32 {
    let text = match read_input(file) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error: {e}");
            return 2;
        }
    };
    let mut req: SystemOneRequest = match serde_json::from_str(&text) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: malformed request JSON: {e}");
            return 2;
        }
    };
    if explain {
        req.explain = Some(true);
    }
    let engine = match build_engine(model_dir, threads) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("error: {e}");
            return 2;
        }
    };
    match engine.evaluate(&req) {
        Ok(resp) => {
            let s = if pretty { serde_json::to_string_pretty(&resp) } else { serde_json::to_string(&resp) }.expect("serialize");
            println!("{s}");
            0
        }
        Err(e) => {
            eprintln!("{}", serde_json::to_string(&serde_json::json!({ "error": e })).unwrap());
            1
        }
    }
}

pub fn validate(file: Option<PathBuf>) -> i32 {
    let text = match read_input(file) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error: {e}");
            return 2;
        }
    };
    let req: SystemOneRequest = match serde_json::from_str(&text) {
        Ok(r) => r,
        Err(e) => {
            println!("{}", serde_json::json!({"valid": false, "error": {"status": 422, "code": "invalid_request", "message": format!("malformed request JSON: {e}")}}));
            return 1;
        }
    };
    match validate_request(&req, &Limits::default()) {
        Ok(()) => {
            println!("{}", serde_json::json!({"valid": true, "questions": req.questions.len()}));
            0
        }
        Err(e) => {
            println!("{}", serde_json::json!({"valid": false, "error": e}));
            1
        }
    }
}
