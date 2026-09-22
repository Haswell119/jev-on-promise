//! Accuracy, NLL, Brier, ECE and schema validity over evaluated answers.

use serde::Serialize;
use serde_json::Value;
use sextant_core::api::Answer;

/// One evaluated question.
#[derive(Debug, Clone, Serialize)]
pub struct Outcome {
    pub record_id: String,
    pub question_id: String,
    pub kind: String,
    pub correct: bool,
    /// Probability assigned to the gold label.
    pub p_gold: f64,
    /// Probability of the predicted label (top-label confidence for ECE).
    pub p_max: f64,
    pub brier: f64,
    pub nll: f64,
    pub schema_valid: bool,
    pub schema_valid_strict: bool,
    pub predicted: String,
    pub gold: String,
    pub confidence: Option<f64>,
    pub n_labels: usize,
    /// Total variation distance to gold soft labels when available.
    pub tvd: Option<f64>,
    /// Expected-value error for score questions.
    pub abs_err: Option<f64>,
    pub path: Option<String>,
    pub family: Option<String>,
}

fn gold_label_string(gold: &Value) -> String {
    match gold {
        Value::String(s) => s.clone(),
        Value::Bool(b) => if *b { "yes".into() } else { "no".into() },
        Value::Number(n) => n.to_string(),
        other => other.to_string(),
    }
}

pub fn score_answer(record_id: &str, qid: &str, answer: &Answer, gold: &Value, gold_probs: Option<&Value>) -> Outcome {
    let eps = 1e-12f64;
    match answer {
        Answer::Noul(a) => {
            let p = a.noul;
            let valid = (0.0..=1.0).contains(&p) && p.is_finite();
            let gold_yes = match gold {
                Value::Bool(b) => *b,
                Value::String(s) => matches!(s.to_ascii_lowercase().as_str(), "yes" | "true" | "1"),
                Value::Number(n) => n.as_f64().unwrap_or(0.0) >= 0.5,
                _ => false,
            };
            let p_gold = if gold_yes { p } else { 1.0 - p };
            let pred_yes = p >= 0.5;
            let tvd = gold_probs.and_then(|g| match g {
                Value::Number(n) => n.as_f64().map(|gp| (gp - p).abs()),
                Value::Object(o) => o.get("yes").and_then(|v| v.as_f64()).map(|gp| (gp - p).abs()),
                _ => None,
            });
            Outcome {
                record_id: record_id.into(),
                question_id: qid.into(),
                kind: "noul".into(),
                correct: pred_yes == gold_yes,
                p_gold,
                p_max: p.max(1.0 - p),
                brier: 2.0 * (p - if gold_yes { 1.0 } else { 0.0 }).powi(2),
                nll: -(p_gold.max(eps)).ln(),
                schema_valid: valid,
                schema_valid_strict: valid,
                predicted: if pred_yes { "yes".into() } else { "no".into() },
                gold: if gold_yes { "yes".into() } else { "no".into() },
                confidence: None,
                n_labels: 2,
                tvd,
                abs_err: None,
                path: a.explain.as_ref().map(|e| e.path.clone()),
                family: a.explain.as_ref().map(|e| e.family.clone()),
            }
        }
        Answer::Choice(a) => {
            let gold_s = gold_label_string(gold);
            let sum: f64 = a.probabilities.values().sum();
            let strict = (sum - 1.0).abs() <= 1e-3 && a.probabilities.values().all(|p| (0.0..=1.0).contains(p) && p.is_finite()) && a.probabilities.contains_key(&a.choice);
            let valid = (sum - 1.0).abs() <= 2e-2 && a.probabilities.values().all(|p| (0.0..=1.0).contains(p) && p.is_finite()) && a.probabilities.contains_key(&a.choice);
            let p_gold = a.probabilities.get(&gold_s).copied().unwrap_or(0.0);
            let p_max = a.probabilities.values().cloned().fold(0.0, f64::max);
            let brier: f64 = a.probabilities.iter().map(|(k, p)| (p - if *k == gold_s { 1.0 } else { 0.0 }).powi(2)).sum();
            let tvd = gold_probs.and_then(|g| g.as_object()).map(|g| 0.5 * a.probabilities.iter().map(|(k, p)| (p - g.get(k).and_then(|v| v.as_f64()).unwrap_or(0.0)).abs()).sum::<f64>());
            Outcome {
                record_id: record_id.into(),
                question_id: qid.into(),
                kind: "choice".into(),
                correct: a.choice == gold_s,
                p_gold,
                p_max,
                brier,
                nll: -(p_gold.max(eps)).ln(),
                schema_valid: valid,
                schema_valid_strict: strict,
                predicted: a.choice.clone(),
                gold: gold_s,
                confidence: Some(a.confidence),
                n_labels: a.probabilities.len(),
                tvd,
                abs_err: None,
                path: a.explain.as_ref().map(|e| e.path.clone()),
                family: a.explain.as_ref().map(|e| e.family.clone()),
            }
        }
        Answer::Score(a) => {
            let gold_s = gold_label_string(gold);
            let gold_i: f64 = gold_s.parse().unwrap_or(-1.0);
            let sum: f64 = a.probabilities.values().sum();
            let strict = (sum - 1.0).abs() <= 1e-3 && a.probabilities.values().all(|p| (0.0..=1.0).contains(p) && p.is_finite());
            let valid = (sum - 1.0).abs() <= 2e-2 && a.probabilities.values().all(|p| (0.0..=1.0).contains(p) && p.is_finite());
            // argmax with lexicographic tie-break, as JevBench does
            let mut pred = String::new();
            let mut best = f64::NEG_INFINITY;
            for (k, p) in &a.probabilities {
                if *p > best || (*p == best && k < &pred) {
                    best = *p;
                    pred = k.clone();
                }
            }
            let p_gold = a.probabilities.get(&gold_s).copied().unwrap_or(0.0);
            let brier: f64 = a.probabilities.iter().map(|(k, p)| (p - if *k == gold_s { 1.0 } else { 0.0 }).powi(2)).sum();
            let tvd = gold_probs.and_then(|g| match g {
                Value::Array(arr) => Some(0.5 * a.probabilities.iter().enumerate().map(|(i, (_, p))| (p - arr.get(i).and_then(|v| v.as_f64()).unwrap_or(0.0)).abs()).sum::<f64>()),
                Value::Object(o) => Some(0.5 * a.probabilities.iter().map(|(k, p)| (p - o.get(k).and_then(|v| v.as_f64()).unwrap_or(0.0)).abs()).sum::<f64>()),
                _ => None,
            });
            Outcome {
                record_id: record_id.into(),
                question_id: qid.into(),
                kind: "score".into(),
                correct: pred == gold_s,
                p_gold,
                p_max: best,
                brier,
                nll: -(p_gold.max(eps)).ln(),
                schema_valid: valid,
                schema_valid_strict: strict,
                predicted: pred,
                gold: gold_s,
                confidence: Some(a.confidence),
                n_labels: a.probabilities.len(),
                tvd,
                abs_err: if gold_i >= 0.0 { Some((a.score - gold_i).abs()) } else { None },
                path: a.explain.as_ref().map(|e| e.path.clone()),
                family: a.explain.as_ref().map(|e| e.family.clone()),
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct Summary {
    pub n: usize,
    pub accuracy: f64,
    pub nll: f64,
    pub brier: f64,
    pub ece: f64,
    pub ece_15: f64,
    pub schema_validity: f64,
    pub schema_validity_strict: f64,
    pub mean_tvd: Option<f64>,
    pub mae: Option<f64>,
    /// Accuracy of the top 50% by confidence (choice/score only).
    pub selective_acc_50: Option<f64>,
    pub confidence_auroc: Option<f64>,
}

/// Top-label expected calibration error with `bins` equal-width bins.
pub fn ece(outcomes: &[&Outcome], bins: usize) -> f64 {
    if outcomes.is_empty() {
        return 0.0;
    }
    let mut sum_conf = vec![0.0f64; bins];
    let mut sum_acc = vec![0.0f64; bins];
    let mut count = vec![0usize; bins];
    for o in outcomes {
        let b = ((o.p_max * bins as f64).floor() as usize).min(bins - 1);
        sum_conf[b] += o.p_max;
        sum_acc[b] += if o.correct { 1.0 } else { 0.0 };
        count[b] += 1;
    }
    let n = outcomes.len() as f64;
    (0..bins).filter(|&b| count[b] > 0).map(|b| (count[b] as f64 / n) * ((sum_acc[b] / count[b] as f64) - (sum_conf[b] / count[b] as f64)).abs()).sum()
}

/// AUROC of `confidence` for predicting correctness (Mann–Whitney).
fn auroc(pairs: &[(f64, bool)]) -> Option<f64> {
    let pos: Vec<f64> = pairs.iter().filter(|(_, c)| *c).map(|(s, _)| *s).collect();
    let neg: Vec<f64> = pairs.iter().filter(|(_, c)| !*c).map(|(s, _)| *s).collect();
    if pos.is_empty() || neg.is_empty() {
        return None;
    }
    let mut wins = 0.0f64;
    for p in &pos {
        for q in &neg {
            wins += if p > q { 1.0 } else if p == q { 0.5 } else { 0.0 };
        }
    }
    Some(wins / (pos.len() as f64 * neg.len() as f64))
}

pub fn summarize(outcomes: &[&Outcome]) -> Summary {
    let n = outcomes.len();
    if n == 0 {
        return Summary::default();
    }
    let nf = n as f64;
    let accuracy = outcomes.iter().filter(|o| o.correct && o.schema_valid).count() as f64 / nf;
    let nll = outcomes.iter().map(|o| o.nll).sum::<f64>() / nf;
    let brier = outcomes.iter().map(|o| o.brier).sum::<f64>() / nf;
    let tvds: Vec<f64> = outcomes.iter().filter_map(|o| o.tvd).collect();
    let maes: Vec<f64> = outcomes.iter().filter_map(|o| o.abs_err).collect();
    let conf_pairs: Vec<(f64, bool)> = outcomes.iter().filter_map(|o| o.confidence.map(|c| (c, o.correct))).collect();
    let selective = if conf_pairs.is_empty() {
        None
    } else {
        let mut sorted = conf_pairs.clone();
        sorted.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let half = (sorted.len() / 2).max(1);
        Some(sorted.iter().take(half).filter(|(_, c)| *c).count() as f64 / half as f64)
    };
    Summary {
        n,
        accuracy,
        nll,
        brier,
        ece: ece(outcomes, 10),
        ece_15: ece(outcomes, 15),
        schema_validity: outcomes.iter().filter(|o| o.schema_valid).count() as f64 / nf,
        schema_validity_strict: outcomes.iter().filter(|o| o.schema_valid_strict).count() as f64 / nf,
        mean_tvd: if tvds.is_empty() { None } else { Some(tvds.iter().sum::<f64>() / tvds.len() as f64) },
        mae: if maes.is_empty() { None } else { Some(maes.iter().sum::<f64>() / maes.len() as f64) },
        selective_acc_50: selective,
        confidence_auroc: auroc(&conf_pairs),
    }
}
