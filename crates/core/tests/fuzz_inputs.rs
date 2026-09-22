//! Robustness: whatever the input, the engine returns `Ok` or a structured
//! `ApiError`. It never panics, never emits NaN, and `validate_request`
//! agrees with `Engine::evaluate` on what is acceptable.

mod common;

use common::*;
use proptest::prelude::*;
use serde_json::{json, Map, Value};
use sextant_core::api::validate::{validate_request, Limits};
use sextant_core::model::{EMBEDDED_CALIBRATION, EMBEDDED_WEIGHTS};
use sextant_core::{Engine, EngineConfig, Model, Resources, SystemOneRequest, SystemOneResponse};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Strategies for hostile inputs
// ---------------------------------------------------------------------------

const ODD_KEYS: &[&str] =
    &["", " ", "a", "a ", "A", " a", "a\u{200b}", "true", "false", "null", "0", "-1", "__proto__", "..", "a.b[0]"];

const KINDS: &[&str] = &["choice", "score", "noul", "regress", "", "CHOICE"];

fn arb_string() -> impl Strategy<Value = String> {
    prop_oneof![
        3 => text(),
        1 => "[\\x00-\\x1f]{0,6}",
        1 => "\\PC{0,64}",
        1 => Just("a".repeat(600)),
        1 => "[a ]{0,30}",
    ]
}

fn arb_key() -> impl Strategy<Value = String> {
    prop_oneof![
        3 => key(),
        2 => prop::sample::select(ODD_KEYS).prop_map(String::from),
        1 => "[\\x00-\\x1f]{1,4}",
        1 => Just("k".repeat(10_000)),
        1 => Just("k".repeat(513)),
    ]
}

/// Arbitrary JSON, nested up to 6 levels.
fn arb_value() -> impl Strategy<Value = Value> {
    let leaf = prop_oneof![
        4 => arb_string().prop_map(Value::String),
        1 => any::<i64>().prop_map(|n| json!(n)),
        1 => any::<f64>().prop_filter("finite", |f| f.is_finite()).prop_map(|f| json!(f)),
        1 => any::<bool>().prop_map(Value::Bool),
        1 => Just(Value::Null),
    ];
    leaf.prop_recursive(6, 64, 8, |inner| {
        prop_oneof![
            prop::collection::vec(inner.clone(), 0..8).prop_map(Value::Array),
            prop::collection::vec((arb_key(), inner), 0..8).prop_map(|kv| {
                let mut m = Map::new();
                for (k, v) in kv {
                    m.insert(k, v);
                }
                Value::Object(m)
            }),
        ]
    })
}

/// Questions with arbitrary instructions and criteria, including malformed shapes.
fn arb_question() -> impl Strategy<Value = Value> {
    prop_oneof![
        4 => (arb_value(), prop::collection::vec((arb_key(), arb_value()), 0..8)).prop_map(|(ins, crit)| {
            let mut m = Map::new();
            for (k, v) in crit {
                m.insert(k, v);
            }
            json!({ "type": "choice", "instructions": ins, "criteria": Value::Object(m) })
        }),
        3 => (arb_value(), prop::collection::vec(arb_value(), 0..13))
            .prop_map(|(ins, levels)| json!({ "type": "score", "instructions": ins, "criteria": levels })),
        3 => (arb_value(), prop::option::of(arb_value()), prop::option::of(arb_value())).prop_map(|(ins, yes, no)| {
            let mut q = json!({ "type": "noul", "instructions": ins });
            let mut c = Map::new();
            if let Some(y) = yes {
                c.insert("true".into(), y);
            }
            if let Some(n) = no {
                c.insert("false".into(), n);
            }
            if !c.is_empty() {
                q["criteria"] = Value::Object(c);
            }
            q
        }),
        1 => arb_value(),
        1 => (prop::sample::select(KINDS), arb_value())
            .prop_map(|(t, v)| json!({ "type": t, "instructions": v, "criteria": v })),
    ]
}

fn arb_model() -> impl Strategy<Value = String> {
    prop_oneof![
        4 => Just("sextant-1".to_string()),
        1 => Just("sextant".to_string()),
        1 => Just("sextant-latest".to_string()),
        1 => Just(String::new()),
        1 => Just("gpt-4".to_string()),
        1 => "\\PC{0,12}",
    ]
}

fn check_ok(resp: &SystemOneResponse, req: &SystemOneRequest) -> Result<(), TestCaseError> {
    prop_assert!(all_finite(resp), "non-finite number in response");
    prop_assert_eq!(resp.answers.len(), req.questions.len());
    prop_assert!(resp.answers.keys().eq(req.questions.keys()), "answers keyed and ordered like the questions");
    if let Some(diff) = roundtrip_error(resp) {
        return Err(TestCaseError::fail(format!("JSON round trip: {diff}")));
    }
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 128, max_shrink_iters: 200, ..ProptestConfig::default() })]

    /// Arbitrary JSON states and questions (deep nesting, huge keys, control
    /// characters, empty strings, null descriptions, look-alike options,
    /// invalid model names): validate + evaluate never panic and agree.
    #[test]
    fn arbitrary_requests_never_panic(
        state in arb_value(),
        qs in prop::collection::vec((arb_key(), arb_question()), 0..4),
        model in arb_model(),
        explain in prop::option::of(any::<bool>()),
    ) {
        let mut questions = Map::new();
        for (id, q) in qs {
            questions.insert(id, q);
        }
        let mut raw = json!({ "model": model, "state": state, "questions": questions });
        if let Some(e) = explain {
            raw["explain"] = json!(e);
        }
        let req: SystemOneRequest = match serde_json::from_value(raw) {
            Ok(r) => r,
            Err(_) => return Ok(()), // not even the schema: nothing to evaluate
        };
        let validation = validate_request(&req, &Limits::default());
        let result = engine().evaluate(&req);
        prop_assert_eq!(
            validation.is_ok(),
            result.is_ok(),
            "validate ({:?}) and evaluate ({:?}) disagree",
            validation.as_ref().err(),
            result.as_ref().err()
        );
        match result {
            Ok(resp) => check_ok(&resp, &req)?,
            Err(e) => {
                prop_assert!(matches!(e.status, 404 | 413 | 422), "unexpected status: {:?}", e);
                prop_assert!(!e.code.is_empty() && !e.message.is_empty());
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Deterministic edge cases
// ---------------------------------------------------------------------------

fn noul(instructions: &str) -> Value {
    json!({ "type": "noul", "instructions": instructions })
}

fn nested(depth: usize) -> Value {
    let mut v = json!("leaf");
    for i in 0..depth {
        v = if i % 2 == 0 { json!([v]) } else { json!({ "k": v }) };
    }
    v
}

fn err_of(v: Value) -> sextant_core::ApiError {
    engine().evaluate(&request(v)).expect_err("request must be rejected")
}

#[test]
fn deeply_nested_values_are_rejected_not_panicking() {
    let e = err_of(json!({ "model": "sextant-1", "state": nested(80), "questions": { "q": noul("Is it deep?") } }));
    assert_eq!((e.status, e.code.as_str(), e.field.as_deref()), (422, "invalid_request", Some("state")));

    let e = err_of(json!({
        "model": "sextant-1",
        "state": "shallow",
        "questions": { "q": { "type": "choice", "instructions": "Pick", "criteria": { "a": nested(80), "b": null } } }
    }));
    assert_eq!(e.status, 422);
    assert!(e.field.as_deref().unwrap_or("").contains("criteria"), "{e:?}");

    let e = err_of(json!({ "model": "sextant-1", "state": "shallow", "questions": { "q": noul_with(nested(80)) } }));
    assert_eq!(e.status, 422);

    // 60 levels is within the limit and must evaluate normally.
    let ok = evaluate(json!({ "model": "sextant-1", "state": nested(60), "questions": { "q": noul("Is it deep?") } }));
    assert!(all_finite(&ok));
}

fn noul_with(instructions: Value) -> Value {
    json!({ "type": "noul", "instructions": instructions })
}

#[test]
fn oversized_option_keys_are_rejected() {
    let choice = |key: String| {
        json!({
            "model": "sextant-1",
            "state": "x",
            "questions": { "q": { "type": "choice", "instructions": "Pick", "criteria": { key: null, "other": null } } }
        })
    };
    let e = err_of(choice("k".repeat(10_000)));
    assert_eq!((e.status, e.code.as_str()), (422, "invalid_request"));
    assert!(e.message.contains("option key"), "{e:?}");
    let e = err_of(choice("k".repeat(513)));
    assert_eq!(e.status, 422);
    let ok = evaluate(choice("k".repeat(512)));
    assert!(all_finite(&ok));

    let e = err_of(json!({
        "model": "sextant-1",
        "state": "x",
        "questions": { "q": { "type": "choice", "instructions": "Pick", "criteria": { "": null, "b": null } } }
    }));
    assert_eq!(e.status, 422);
    let e = err_of(json!({ "model": "sextant-1", "state": "x", "questions": { "": noul("Empty id?") } }));
    assert_eq!(e.status, 422);
    let e = err_of(json!({ "model": "sextant-1", "state": "x", "questions": { "i".repeat(600): noul("Long id?") } }));
    assert_eq!(e.status, 422);
}

#[test]
fn too_many_options_or_levels_are_rejected_with_422() {
    let choice = |n: usize| {
        let mut c = Map::new();
        for i in 0..n {
            c.insert(format!("o{i}"), Value::Null);
        }
        json!({ "model": "sextant-1", "state": "x", "questions": { "q": { "type": "choice", "instructions": "Pick", "criteria": c } } })
    };
    let e = err_of(choice(1000));
    assert_eq!((e.status, e.code.as_str(), e.field.as_deref()), (422, "invalid_request", Some("questions.q.criteria")));
    assert!(e.message.contains("255"), "{e:?}");
    assert_eq!(err_of(choice(256)).status, 422);
    assert_eq!(err_of(choice(1)).status, 422);
    assert_eq!(err_of(choice(0)).status, 422);
    assert!(all_finite(&evaluate(choice(255))));
    assert!(all_finite(&evaluate(choice(2))));

    let score = |k: usize| {
        let levels: Vec<Value> = (0..k).map(|i| json!(format!("level {i}"))).collect();
        json!({ "model": "sextant-1", "state": "x", "questions": { "q": { "type": "score", "instructions": "Rate", "criteria": levels } } })
    };
    let e = err_of(score(11));
    assert_eq!((e.status, e.code.as_str(), e.field.as_deref()), (422, "invalid_request", Some("questions.q.criteria")));
    assert_eq!(err_of(score(1)).status, 422);
    assert_eq!(err_of(score(0)).status, 422);
    assert_eq!(err_of(score(100)).status, 422);
    assert!(all_finite(&evaluate(score(10))));
    assert!(all_finite(&evaluate(score(2))));

    // Too many questions.
    let mut qs = Map::new();
    for i in 0..1025 {
        qs.insert(format!("q{i}"), noul("Too many?"));
    }
    assert_eq!(err_of(json!({ "model": "sextant-1", "state": "x", "questions": qs })).status, 422);
    assert_eq!(err_of(json!({ "model": "sextant-1", "state": "x", "questions": {} })).status, 422);
}

#[test]
fn invalid_model_names_are_rejected() {
    for m in ["gpt-4", "sextant-2", "SEXTANT-1", "sextant-1 ", " sextant-1", "sextant_1", "\u{1f680}", "sextant-1\u{0}"]
    {
        let e = err_of(json!({ "model": m, "state": "x", "questions": { "q": noul("?") } }));
        assert_eq!((e.status, e.code.as_str(), e.field.as_deref()), (404, "unknown_model", Some("model")), "{m:?}");
    }
    let e = err_of(json!({ "model": "", "state": "x", "questions": { "q": noul("?") } }));
    assert_eq!((e.status, e.code.as_str()), (422, "invalid_request"));
    for alias in ["sextant-1", "sextant", "sextant-latest"] {
        let resp = evaluate(json!({ "model": alias, "state": "x", "questions": { "q": noul("?") } }));
        assert_eq!(resp.model, "sextant-1");
    }
}

#[test]
fn invalid_state_shapes_are_rejected() {
    for state in [json!(null), json!(12), json!(1.5), json!(true), json!(false)] {
        let e = err_of(json!({ "model": "sextant-1", "state": state, "questions": { "q": noul("?") } }));
        assert_eq!((e.status, e.field.as_deref()), (422, Some("state")), "{state}");
    }
    // Oversized state: rejected with 413 rather than evaluated.
    let e = err_of(
        json!({ "model": "sextant-1", "state": "x".repeat(2 * 1024 * 1024 + 1), "questions": { "q": noul("?") } }),
    );
    assert_eq!((e.status, e.code.as_str()), (413, "payload_too_large"));
    let many: Vec<Value> = vec![json!("leaf"); 50_001];
    let e = err_of(json!({ "model": "sextant-1", "state": many, "questions": { "q": noul("?") } }));
    assert_eq!(e.status, 413);
}

#[test]
fn noul_without_instructions_or_criteria_is_rejected() {
    let e = err_of(json!({ "model": "sextant-1", "state": "x", "questions": { "q": { "type": "noul" } } }));
    assert_eq!(e.status, 422);
    let e = err_of(
        json!({ "model": "sextant-1", "state": "x", "questions": { "q": { "type": "noul", "instructions": null } } }),
    );
    assert_eq!(e.status, 422);
    let e =
        err_of(json!({ "model": "sextant-1", "state": "x", "questions": { "q": { "type": "noul", "criteria": {} } } }));
    assert_eq!(e.status, 422);
    let ok = evaluate(json!({
        "model": "sextant-1",
        "state": "x",
        "questions": { "q": { "type": "noul", "criteria": { "true": "it is x", "false": "it is not x" } } }
    }));
    assert!(all_finite(&ok));
}

#[test]
fn unknown_question_types_do_not_parse() {
    for t in ["regress", "", "CHOICE", "Noul", "score "] {
        let raw =
            json!({ "model": "sextant-1", "state": "x", "questions": { "q": { "type": t, "instructions": "?" } } });
        assert!(serde_json::from_value::<SystemOneRequest>(raw).is_err(), "type {t:?} must not parse");
    }
    let raw = json!({ "model": "sextant-1", "state": "x", "questions": { "q": { "instructions": "?" } } });
    assert!(serde_json::from_value::<SystemOneRequest>(raw).is_err(), "missing type must not parse");
}

/// Strange but valid requests: empty and whitespace states, empty containers,
/// control characters, look-alike option keys, null / empty / numeric
/// descriptions, structured or empty instructions, unicode ids.
#[test]
fn unusual_but_valid_requests_evaluate_without_panicking() {
    let states = vec![
        json!(""),
        json!(" "),
        json!("\n\t"),
        json!([]),
        json!({}),
        json!([null, null]),
        json!({ "": "" }),
        json!({ "a": { "b": { "c": [] } } }),
        json!([[], {}, "", 0, false, null]),
        json!("\u{0}\u{1}\u{2}\u{7f}"),
        json!("!!!???..."),
        json!("a".repeat(5000)),
        Value::Array(vec![json!("x"); 300]),
        json!({ "refund_requested": false, "paid": true, "amount": 0, "note": null, "n/a": "N/A" }),
        json!("Order 12/13/2026 at 10:30am for $12,840.00 net 30, 12% off, mail bob@x.io see https://x.io/a?b=1"),
        json!("\u{645}\u{631}\u{62d}\u{628}\u{627} \u{5e9}\u{5dc}\u{5d5}\u{5dd} \u{65e5}\u{672c}\u{8a9e} \u{1f680}"),
    ];
    let questions = json!({
        "dupes": {
            "type": "choice",
            "instructions": "Pick one",
            "criteria": { "a": null, "a ": null, "A": null, " a": "", "a\u{200b}": [] }
        },
        "nulls": { "type": "choice", "instructions": "", "criteria": { "x": null, "y": null } },
        "no_instructions": { "type": "choice", "criteria": { "x": "first", "y": "second" } },
        "obj_instructions": { "type": "noul", "instructions": {} },
        "arr_instructions": { "type": "noul", "instructions": [] },
        "empty_instructions": { "type": "noul", "instructions": "" },
        "criteria_only": { "type": "noul", "criteria": { "true": "yes", "false": "no" } },
        "levels": { "type": "score", "instructions": "How?", "criteria": [null, "", {}, [], 0, false, "level"] },
        "numbers": {
            "type": "score",
            "instructions": "How many items?",
            "criteria": ["0-10 items", "11-20 items", "more than 20 items"]
        },
        "comparisons": { "type": "noul", "instructions": "Is the total greater than $500?" },
        "dates": { "type": "noul", "instructions": "Was it before 2026-04-01?" },
        "unicode id \u{1f680}": { "type": "noul", "instructions": "\u{645}\u{631}\u{62d}\u{628}\u{627}?" },
        "control": { "type": "noul", "instructions": "\u{0}\u{1}?" }
    });
    let n = questions.as_object().unwrap().len();
    for state in states {
        for explain in [false, true] {
            let resp =
                evaluate(json!({ "model": "sextant-1", "state": state, "questions": questions, "explain": explain }));
            assert_eq!(resp.answers.len(), n, "state {state}");
            assert!(all_finite(&resp), "state {state}");
            let dupes = as_choice(&resp.answers["dupes"]);
            assert_eq!(dupes.probabilities.len(), 5);
            let sum: f64 = dupes.probabilities.values().sum();
            assert!((sum - 1.0).abs() < 1e-9);
            let levels = as_score(&resp.answers["levels"]);
            assert_eq!(levels.legend.len(), 7);
            assert_eq!(levels.legend["0"], "");
            assert_eq!(levels.legend["4"], "0");
            assert_eq!(levels.legend["5"], "false");
            assert_eq!(roundtrip_error(&resp), None, "state {state}");
        }
    }
}

// ---------------------------------------------------------------------------
// Model artifacts
// ---------------------------------------------------------------------------

struct TempDir(std::path::PathBuf);

impl TempDir {
    fn new(name: &str) -> TempDir {
        let dir = std::env::temp_dir().join(format!("sextant-fuzz-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create temp dir");
        TempDir(dir)
    }
    fn write(&self, name: &str, contents: &str) {
        std::fs::write(self.0.join(name), contents).expect("write temp file");
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn calibration_with(edit: impl FnOnce(&mut Value)) -> String {
    let mut v: Value = serde_json::from_str(EMBEDDED_CALIBRATION).expect("embedded calibration parses");
    edit(&mut v);
    serde_json::to_string_pretty(&v).unwrap()
}

#[test]
fn model_load_dir_rejects_invalid_calibration() {
    let dir = TempDir::new("calibration");
    dir.write("weights.json", EMBEDDED_WEIGHTS);

    dir.write("calibration.json", &calibration_with(|c| c["temperature"]["choice"] = json!(-1.0)));
    let e = Model::load_dir(&dir.0).expect_err("negative temperature must be rejected");
    assert!(e.contains("temperature"), "{e}");

    dir.write("calibration.json", &calibration_with(|c| c["symbolic_temperature"]["score"] = json!(0.0)));
    assert!(Model::load_dir(&dir.0).is_err(), "zero temperature must be rejected");

    dir.write("calibration.json", &calibration_with(|c| c["ordinal_lambda"] = json!(1.5)));
    let e = Model::load_dir(&dir.0).expect_err("ordinal_lambda >= 1 must be rejected");
    assert!(e.contains("ordinal_lambda"), "{e}");

    // NaN is not valid JSON: the artifact must fail to load rather than poison inference.
    let nan = calibration_with(|c| c["temperature"]["choice"] = json!(12345.5)).replace("12345.5", "NaN");
    assert!(nan.contains("NaN"));
    dir.write("calibration.json", &nan);
    let e = Model::load_dir(&dir.0).expect_err("NaN temperature must be rejected");
    assert!(e.contains("calibration.json"), "{e}");

    let inf = calibration_with(|c| c["noul_platt"]["a"] = json!(12345.5)).replace("12345.5", "1e999");
    dir.write("calibration.json", &inf);
    assert!(Model::load_dir(&dir.0).is_err(), "out-of-range platt parameter must be rejected");

    dir.write("calibration.json", "{ not json");
    assert!(Model::load_dir(&dir.0).is_err());

    // A pristine copy of the embedded artifact loads and behaves identically.
    dir.write("calibration.json", EMBEDDED_CALIBRATION);
    let model = Model::load_dir(&dir.0).expect("valid artifact loads");
    assert!(model.source.contains("sextant-fuzz"), "source = {}", model.source);
    let loaded = Engine::new(Resources::embedded(), Arc::new(model), EngineConfig::default());
    let req = request(json!({
        "model": "sextant-1",
        "state": "Help! My payouts have been failing for 3 days. I need this fixed today or I will cancel.",
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": { "billing": "Payments, invoicing, refunds, payouts", "technical": "Bugs, outages", "sales": "Pricing" }
            },
            "is_urgent": noul("Does this convey urgency?"),
            "frustration": { "type": "score", "instructions": "How frustrated is the customer?", "criteria": ["Calm", "Frustrated", "Very angry"] }
        }
    }));
    assert_eq!(to_json(&loaded.evaluate(&req).unwrap()), to_json(&engine().evaluate(&req).unwrap()));

    // Missing files are reported by name.
    std::fs::remove_file(dir.0.join("weights.json")).unwrap();
    let e = Model::load_dir(&dir.0).expect_err("missing weights must be rejected");
    assert!(e.contains("weights.json"), "{e}");
    assert!(Model::load_dir(&dir.0.join("does-not-exist")).is_err());
}

#[test]
fn model_load_dir_rejects_invalid_weights() {
    let dir = TempDir::new("weights");
    dir.write("calibration.json", EMBEDDED_CALIBRATION);
    dir.write("weights.json", "{ \"version\": \"x\" }");
    let e = Model::load_dir(&dir.0).expect_err("incomplete weights must be rejected");
    assert!(e.contains("weights.json"), "{e}");
    dir.write("weights.json", "[]");
    assert!(Model::load_dir(&dir.0).is_err());
}
