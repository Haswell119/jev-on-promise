//! The engine: validate → index state once → evaluate questions in parallel
//! on a bounded worker pool → typed answers. Deterministic given the same
//! binary, artifact and request.

use crate::api::validate::{validate_request, Limits, MODEL_ALIASES, MODEL_ID};
use crate::api::{
    Answer, ApiError, ChoiceAnswer, Explanation, ModelCard, NoulAnswer, Question, QuestionKind, ScoreAnswer,
    SystemOneRequest, SystemOneResponse, Usage,
};
use crate::calibration::{cardinality_bucket, platt};
use crate::features::{extract_features, F};
use crate::lexicon::Resources;
use crate::model::Model;
use crate::question::QuestionView;
use crate::resolvers;
use crate::scoring::{confidence, ordinal_smooth, softmax_temp};
use crate::state::index::StateIndex;
use indexmap::IndexMap;
use serde_json::Value;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub limits: Limits,
    /// Worker threads for question evaluation (0 = number of CPUs).
    pub threads: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        EngineConfig { limits: Limits::default(), threads: 0 }
    }
}

pub struct Engine {
    pub res: Arc<Resources>,
    pub model: Arc<Model>,
    pub config: EngineConfig,
    pool: rayon::ThreadPool,
}

impl std::fmt::Debug for Engine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Engine(model={}, threads={})", self.model.source, self.pool.current_num_threads())
    }
}

/// Everything the per-question worker needs.
struct Ctx<'a> {
    res: &'a Resources,
    model: &'a Model,
    state: &'a StateIndex,
    explain: bool,
}

impl Engine {
    pub fn new(res: Arc<Resources>, model: Arc<Model>, config: EngineConfig) -> Engine {
        let threads = if config.threads == 0 { std::thread::available_parallelism().map(|n| n.get()).unwrap_or(2) } else { config.threads };
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).thread_name(|i| format!("sextant-worker-{i}")).build().expect("thread pool");
        Engine { res, model, config, pool }
    }

    /// Engine with embedded resources and the embedded model artifact.
    pub fn default_embedded() -> Engine {
        Engine::new(Resources::embedded(), Arc::new(Model::embedded()), EngineConfig::default())
    }

    pub fn models(&self) -> Vec<ModelCard> {
        vec![ModelCard {
            id: MODEL_ID.into(),
            object: "model".into(),
            description: "Sextant non-neural probabilistic decision engine (Choice / Score / Noul); deterministic, calibrated, no external AI services.".into(),
            aliases: MODEL_ALIASES.iter().map(|s| s.to_string()).collect(),
            max_choice_options: self.config.limits.max_choice_options,
            max_score_levels: self.config.limits.max_score_levels,
            neural: false,
        }]
    }

    /// Evaluate a request. Validation errors are returned as `ApiError`.
    pub fn evaluate(&self, req: &SystemOneRequest) -> Result<SystemOneResponse, ApiError> {
        validate_request(req, &self.config.limits)?;
        let explain = req.explain.unwrap_or(false);
        let state = StateIndex::build(&req.state, &self.res);
        let ctx = Ctx { res: &self.res, model: &self.model, state: &state, explain };
        let questions: Vec<(&String, &Question)> = req.questions.iter().collect();
        let answers: Vec<Answer> = if questions.len() <= 1 {
            questions.iter().map(|(_, q)| answer_question(q, &ctx)).collect()
        } else {
            self.pool.install(|| {
                use rayon::prelude::*;
                questions.par_iter().map(|(_, q)| answer_question(q, &ctx)).collect()
            })
        };
        let mut out = IndexMap::with_capacity(answers.len());
        let mut output_tokens = 0u64;
        for ((id, _), a) in questions.iter().zip(answers) {
            output_tokens += match &a {
                Answer::Noul(_) => 2,
                Answer::Choice(c) => 2 + c.probabilities.len() as u64,
                Answer::Score(s) => 2 + s.probabilities.len() as u64,
            };
            out.insert((*id).clone(), a);
        }
        let question_tokens: u64 = req.questions.values().map(|q| approx_tokens(q)).sum();
        Ok(SystemOneResponse {
            model: MODEL_ID.into(),
            answers: out,
            usage: Usage { input_tokens: state.token_count + question_tokens, output_tokens },
        })
    }

    /// Evaluate a single question against a pre-built index (used by tests, benches and training).
    pub fn answer_one(&self, q: &Question, state: &StateIndex, explain: bool) -> Answer {
        let ctx = Ctx { res: &self.res, model: &self.model, state, explain };
        answer_question(q, &ctx)
    }
}

fn approx_tokens(q: &Question) -> u64 {
    fn count(v: &Value) -> u64 {
        match v {
            Value::String(s) => s.split_whitespace().count() as u64,
            Value::Array(a) => a.iter().map(count).sum(),
            Value::Object(o) => o.iter().map(|(k, v)| 1 + count(v) + k.split('_').count() as u64).sum(),
            Value::Null => 0,
            _ => 1,
        }
    }
    let base = count(q.instructions());
    match q {
        Question::Noul(n) => base + n.criteria.as_ref().map(|c| c.yes.as_ref().map(count).unwrap_or(0) + c.no.as_ref().map(count).unwrap_or(0)).unwrap_or(0),
        Question::Choice(c) => base + c.criteria.iter().map(|(k, v)| 1 + count(v) + k.split('_').count() as u64).sum::<u64>(),
        Question::Score(s) => base + s.criteria.iter().map(count).sum::<u64>(),
    }
}

/// Argmax with lexicographic key tie-breaking (order-invariant).
fn argmax_key(probs: &IndexMap<String, f64>) -> String {
    let mut best: Option<(&String, f64)> = None;
    for (k, &p) in probs {
        match best {
            None => best = Some((k, p)),
            Some((bk, bp)) => {
                if p > bp || (p == bp && k < bk) {
                    best = Some((k, p));
                }
            }
        }
    }
    best.map(|(k, _)| k.clone()).unwrap_or_default()
}

fn answer_question(q: &Question, ctx: &Ctx<'_>) -> Answer {
    let view = QuestionView::build(q, ctx.state, ctx.res);
    let feats = extract_features(&view, ctx.state, ctx.res);
    let resolved = resolvers::resolve(&view, ctx.state, &feats);
    let cal = &ctx.model.calibration;
    let family = view.family;
    let k = view.criteria.len();

    match view.kind {
        QuestionKind::Noul => {
            let (logit_raw, path) = match &resolved {
                Some(r) => (r.logits[0], format!("symbolic:{}", r.name)),
                None => (ctx.model.dense.noul.logit(&feats.rows[0], &feats.rows[1], family), "semantic".to_string()),
            };
            let p = match &resolved {
                Some(_) => {
                    let eps = cal.symbolic_epsilon_for(QuestionKind::Noul);
                    (1.0 - eps) * platt(logit_raw as f64, cal.noul_symbolic_platt.as_ref().unwrap_or(&cal.noul_platt)) + eps * 0.5
                }
                None => platt(logit_raw as f64, cal.platt_for(family)),
            };
            let explain = ctx.explain.then(|| {
                let mut raw: IndexMap<String, f64> = IndexMap::new();
                raw.insert("logit".to_string(), logit_raw as f64);
                let mut ex = build_explanation(&view, &feats, ctx, family.as_str(), path.clone(), raw);
                if let Some(r) = &resolved {
                    ex.notes = r.notes.clone();
                }
                let pl = if resolved.is_some() { cal.noul_symbolic_platt.clone().unwrap_or_else(|| cal.noul_platt.clone()) } else { cal.platt_for(family).clone() };
                ex.calibration.insert("method".into(), Value::String("platt".into()));
                ex.calibration.insert("platt_a".into(), serde_json::json!(pl.a));
                ex.calibration.insert("platt_b".into(), serde_json::json!(pl.b));
                ex
            });
            Answer::Noul(NoulAnswer { noul: p, explain })
        }
        QuestionKind::Choice | QuestionKind::Score => {
            let head = if view.kind == QuestionKind::Choice { &ctx.model.dense.choice } else { &ctx.model.dense.score };
            let (z, temperature, path): (Vec<f32>, f64, String) = match &resolved {
                Some(r) => (r.logits.clone(), cal.symbolic_temperature_for(view.kind), format!("symbolic:{}", r.name)),
                None => (feats.rows.iter().map(|f| head.score(f, family)).collect(), cal.temperature_for(view.kind, family, k), "semantic".to_string()),
            };
            let mut probs = softmax_temp(&z, temperature as f32);
            if resolved.is_some() {
                let eps = cal.symbolic_epsilon_for(view.kind);
                let u = 1.0 / probs.len() as f64;
                for p in probs.iter_mut() {
                    *p = (1.0 - eps) * *p + eps * u;
                }
                crate::scoring::softmax::fix_sum(&mut probs);
            }
            if view.kind == QuestionKind::Score {
                probs = ordinal_smooth(&probs, cal.ordinal_lambda);
            }
            let evidence = feats.rows.iter().map(|f| f.get(F::cov_w)).fold(0.0f32, f32::max) as f64;
            let ood = feats.rows.iter().map(|f| f.get(F::ood)).fold(1.0f32, f32::min) as f64;
            let conf = confidence(&probs, evidence, ood, &cal.confidence_for(view.kind));
            let explain = ctx.explain.then(|| {
                let raw: IndexMap<String, f64> = view.criteria.iter().zip(z.iter()).map(|(c, v)| (c.key.clone(), *v as f64)).collect();
                let mut ex = build_explanation(&view, &feats, ctx, family.as_str(), path.clone(), raw);
                if let Some(r) = &resolved {
                    ex.notes = r.notes.clone();
                }
                ex.calibration.insert("method".into(), Value::String(cal.method.clone()));
                ex.calibration.insert("temperature".into(), serde_json::json!(temperature));
                ex.calibration.insert("bucket".into(), Value::String(cardinality_bucket(k).into()));
                if view.kind == QuestionKind::Score {
                    ex.calibration.insert("ordinal_lambda".into(), serde_json::json!(cal.ordinal_lambda));
                }
                ex
            });
            if view.kind == QuestionKind::Choice {
                let probabilities: IndexMap<String, f64> = view.criteria.iter().zip(probs.iter()).map(|(c, p)| (c.key.clone(), *p)).collect();
                let choice = argmax_key(&probabilities);
                Answer::Choice(ChoiceAnswer { choice, probabilities, confidence: conf, explain })
            } else {
                let legend: IndexMap<String, String> = match q {
                    Question::Score(s) => s.criteria.iter().enumerate().map(|(i, v)| (i.to_string(), legend_text(v))).collect(),
                    _ => IndexMap::new(),
                };
                let probabilities: IndexMap<String, f64> = probs.iter().enumerate().map(|(i, p)| (i.to_string(), *p)).collect();
                let score: f64 = probs.iter().enumerate().map(|(i, p)| i as f64 * p).sum();
                Answer::Score(ScoreAnswer { score, legend, probabilities, confidence: conf, explain })
            }
        }
    }
}

fn legend_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

fn build_explanation(view: &QuestionView<'_>, feats: &crate::features::FeatureMatrix, ctx: &Ctx<'_>, family: &str, path: String, raw_scores: IndexMap<String, f64>) -> Explanation {
    // Evidence of the best-scoring criterion (or the yes hypothesis for Noul).
    let best = raw_scores.iter().enumerate().fold((0usize, f64::NEG_INFINITY), |acc, (i, (_, v))| if *v > acc.1 { (i, *v) } else { acc }).0;
    let top_evidence = feats.evidence.get(best).map(|e| crate::features::extract::evidence_items(ctx.state, e, 5)).unwrap_or_default();
    let features: IndexMap<String, IndexMap<String, f64>> = view.criteria.iter().zip(feats.rows.iter()).map(|(c, f)| (c.key.clone(), f.named())).collect();
    let mut calibration = IndexMap::new();
    calibration.insert("family".into(), Value::String(family.into()));
    Explanation { family: family.to_string(), path, top_evidence, raw_scores, features, calibration, notes: Vec::new() }
}
