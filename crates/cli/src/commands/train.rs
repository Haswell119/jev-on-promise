#![allow(clippy::too_many_arguments)]
//! `sextant train`: fit the fusion weights.
//!
//! * Choice / Score: conditional logit (shared weight vector over per-option
//!   features, softmax over options) with optional per-family deltas
//!   (mixture of linear experts) shrunk toward the base expert.
//! * Noul: logistic regression over (f_yes − f_no) ++ f_yes.
//!
//! Full-batch Adam with a fixed seed; small, inspectable parameter count.
//! Examples answered by a symbolic resolver are excluded (the resolver
//! path is calibrated separately).

use super::dataset::load_records;
use super::learn::{extract_examples, parse_drop, Example};
use indexmap::IndexMap;
use sextant_core::api::QuestionKind;
use sextant_core::features::{FEATURE_NAMES, N_FEATURES};
use sextant_core::question::Family;
use sextant_core::scoring::{Head, NoulFamilyDelta, NoulHead, Weights};
use sextant_core::Resources;
use std::path::PathBuf;

const MIN_FAMILY_EXAMPLES: usize = 40;
/// Ordinal label smoothing for Score: mass given to each adjacent level.
const ORDINAL_SMOOTH: f32 = 0.08;

struct Adam {
    m: Vec<f32>,
    v: Vec<f32>,
    t: usize,
    lr: f32,
}

impl Adam {
    fn new(n: usize, lr: f32) -> Self {
        Adam { m: vec![0.0; n], v: vec![0.0; n], t: 0, lr }
    }
    fn step(&mut self, w: &mut [f32], g: &[f32]) {
        self.t += 1;
        let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
        let lr_t = self.lr * (1.0 - b2.powi(self.t as i32)).sqrt() / (1.0 - b1.powi(self.t as i32));
        for i in 0..w.len() {
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * g[i];
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * g[i] * g[i];
            w[i] -= lr_t * self.m[i] / (self.v[i].sqrt() + eps);
        }
    }
}

fn family_slot(f: Family) -> usize {
    Family::all().iter().position(|x| *x == f).unwrap_or(0)
}

/// Parameters for one multinomial head: base + per-family deltas (flattened).
struct HeadParams {
    base: Vec<f32>,
    deltas: Vec<Vec<f32>>, // per family slot
    active: Vec<bool>,
}

impl HeadParams {
    fn new(n_fam: usize) -> Self {
        HeadParams { base: vec![0.0; N_FEATURES], deltas: vec![vec![0.0; N_FEATURES]; n_fam], active: vec![false; n_fam] }
    }
    fn weights_for(&self, slot: usize, out: &mut [f32]) {
        for j in 0..N_FEATURES {
            out[j] = self.base[j] + if self.active[slot] { self.deltas[slot][j] } else { 0.0 };
        }
    }
}

fn targets(ex: &Example) -> Vec<f32> {
    let k = ex.rows.len();
    if let Some(soft) = &ex.soft {
        if soft.len() == k && soft.iter().sum::<f64>() > 0.99 {
            return soft.iter().map(|v| *v as f32).collect();
        }
    }
    let mut t = vec![0.0f32; k];
    t[ex.gold] = 1.0;
    if ex.kind == QuestionKind::Score && k >= 3 {
        // ordinal smoothing toward neighbours
        let mut s = vec![0.0f32; k];
        for i in 0..k {
            if i == ex.gold {
                s[i] += 1.0 - ORDINAL_SMOOTH * (((i > 0) as u8) + ((i + 1 < k) as u8)) as f32;
            } else if i + 1 == ex.gold || i == ex.gold + 1 {
                s[i] += ORDINAL_SMOOTH;
            }
        }
        return s;
    }
    t
}

/// Train a multinomial head. Returns (params, final loss).
fn train_multinomial(examples: &[&Example], epochs: usize, l2: f32, family_l2: f32, use_families: bool, lr: f32) -> (HeadParams, f32) {
    let n_fam = Family::all().len();
    let mut p = HeadParams::new(n_fam);
    let mut counts = vec![0usize; n_fam];
    for e in examples {
        counts[family_slot(e.family)] += 1;
    }
    if use_families {
        for s in 0..n_fam {
            p.active[s] = counts[s] >= MIN_FAMILY_EXAMPLES;
        }
    }
    let n_params = N_FEATURES * (1 + n_fam);
    let mut adam = Adam::new(n_params, lr);
    let mut grad = vec![0.0f32; n_params];
    let mut w = vec![0.0f32; N_FEATURES];
    let mut flat = vec![0.0f32; n_params];
    let mut loss = 0.0f32;
    let n = examples.len().max(1) as f32;
    for _epoch in 0..epochs {
        for g in grad.iter_mut() {
            *g = 0.0;
        }
        loss = 0.0;
        for e in examples {
            let slot = family_slot(e.family);
            p.weights_for(slot, &mut w);
            let z: Vec<f32> = e.rows.iter().map(|r| r.dot(&w)).collect();
            let m = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let ex: Vec<f32> = z.iter().map(|v| (v - m).exp()).collect();
            let s: f32 = ex.iter().sum();
            let probs: Vec<f32> = ex.iter().map(|v| v / s).collect();
            let t = targets(e);
            for k in 0..probs.len() {
                if t[k] > 0.0 {
                    loss -= t[k] * probs[k].max(1e-9).ln();
                }
                let coef = (probs[k] - t[k]) / n;
                for j in 0..N_FEATURES {
                    let g = coef * e.rows[k].0[j];
                    grad[j] += g;
                    if p.active[slot] {
                        grad[N_FEATURES * (1 + slot) + j] += g;
                    }
                }
            }
        }
        loss /= n;
        // L2
        for j in 0..N_FEATURES {
            grad[j] += 2.0 * l2 * p.base[j];
            loss += l2 * p.base[j] * p.base[j];
            for s in 0..n_fam {
                if p.active[s] {
                    let d = p.deltas[s][j];
                    grad[N_FEATURES * (1 + s) + j] += 2.0 * family_l2 * d;
                    loss += family_l2 * d * d;
                }
            }
        }
        flat[..N_FEATURES].copy_from_slice(&p.base);
        for s in 0..n_fam {
            flat[N_FEATURES * (1 + s)..N_FEATURES * (2 + s)].copy_from_slice(&p.deltas[s]);
        }
        adam.step(&mut flat, &grad);
        p.base.copy_from_slice(&flat[..N_FEATURES]);
        for s in 0..n_fam {
            p.deltas[s].copy_from_slice(&flat[N_FEATURES * (1 + s)..N_FEATURES * (2 + s)]);
        }
    }
    (p, loss)
}

struct NoulParams {
    diff: Vec<f32>,
    yes: Vec<f32>,
    bias: f32,
    fam_diff: Vec<Vec<f32>>,
    fam_yes: Vec<Vec<f32>>,
    fam_bias: Vec<f32>,
    active: Vec<bool>,
}

fn train_noul(examples: &[&Example], epochs: usize, l2: f32, family_l2: f32, use_families: bool, lr: f32) -> (NoulParams, f32) {
    let n_fam = Family::all().len();
    let mut p = NoulParams { diff: vec![0.0; N_FEATURES], yes: vec![0.0; N_FEATURES], bias: 0.0, fam_diff: vec![vec![0.0; N_FEATURES]; n_fam], fam_yes: vec![vec![0.0; N_FEATURES]; n_fam], fam_bias: vec![0.0; n_fam], active: vec![false; n_fam] };
    let mut counts = vec![0usize; n_fam];
    for e in examples {
        counts[family_slot(e.family)] += 1;
    }
    if use_families {
        for s in 0..n_fam {
            p.active[s] = counts[s] >= MIN_FAMILY_EXAMPLES;
        }
    }
    // layout: diff[N], yes[N], bias, then per family: diff[N], yes[N], bias
    let stride = 2 * N_FEATURES + 1;
    let n_params = stride * (1 + n_fam);
    let mut adam = Adam::new(n_params, lr);
    let mut flat = vec![0.0f32; n_params];
    let mut grad = vec![0.0f32; n_params];
    let n = examples.len().max(1) as f32;
    let mut loss = 0.0f32;
    for _ in 0..epochs {
        for g in grad.iter_mut() {
            *g = 0.0;
        }
        loss = 0.0;
        for e in examples {
            let slot = family_slot(e.family);
            let fy = &e.rows[0].0;
            let fn_ = &e.rows[1].0;
            let mut z = p.bias + if p.active[slot] { p.fam_bias[slot] } else { 0.0 };
            for j in 0..N_FEATURES {
                let d = fy[j] - fn_[j];
                z += p.diff[j] * d + p.yes[j] * fy[j];
                if p.active[slot] {
                    z += p.fam_diff[slot][j] * d + p.fam_yes[slot][j] * fy[j];
                }
            }
            let prob = 1.0 / (1.0 + (-z).exp());
            let target = match &e.soft {
                Some(s) if s.len() == 2 => s[1] as f32,
                _ => e.gold as f32,
            };
            loss -= target * prob.max(1e-9).ln() + (1.0 - target) * (1.0 - prob).max(1e-9).ln();
            let coef = (prob - target) / n;
            grad[2 * N_FEATURES] += coef;
            if p.active[slot] {
                grad[stride * (1 + slot) + 2 * N_FEATURES] += coef;
            }
            for j in 0..N_FEATURES {
                let d = fy[j] - fn_[j];
                grad[j] += coef * d;
                grad[N_FEATURES + j] += coef * fy[j];
                if p.active[slot] {
                    grad[stride * (1 + slot) + j] += coef * d;
                    grad[stride * (1 + slot) + N_FEATURES + j] += coef * fy[j];
                }
            }
        }
        loss /= n;
        for j in 0..N_FEATURES {
            grad[j] += 2.0 * l2 * p.diff[j];
            grad[N_FEATURES + j] += 2.0 * l2 * p.yes[j];
            loss += l2 * (p.diff[j] * p.diff[j] + p.yes[j] * p.yes[j]);
            for s in 0..n_fam {
                if p.active[s] {
                    grad[stride * (1 + s) + j] += 2.0 * family_l2 * p.fam_diff[s][j];
                    grad[stride * (1 + s) + N_FEATURES + j] += 2.0 * family_l2 * p.fam_yes[s][j];
                    loss += family_l2 * (p.fam_diff[s][j].powi(2) + p.fam_yes[s][j].powi(2));
                }
            }
        }
        flat[..N_FEATURES].copy_from_slice(&p.diff);
        flat[N_FEATURES..2 * N_FEATURES].copy_from_slice(&p.yes);
        flat[2 * N_FEATURES] = p.bias;
        for s in 0..n_fam {
            let o = stride * (1 + s);
            flat[o..o + N_FEATURES].copy_from_slice(&p.fam_diff[s]);
            flat[o + N_FEATURES..o + 2 * N_FEATURES].copy_from_slice(&p.fam_yes[s]);
            flat[o + 2 * N_FEATURES] = p.fam_bias[s];
        }
        adam.step(&mut flat, &grad);
        p.diff.copy_from_slice(&flat[..N_FEATURES]);
        p.yes.copy_from_slice(&flat[N_FEATURES..2 * N_FEATURES]);
        p.bias = flat[2 * N_FEATURES];
        for s in 0..n_fam {
            let o = stride * (1 + s);
            p.fam_diff[s].copy_from_slice(&flat[o..o + N_FEATURES]);
            p.fam_yes[s].copy_from_slice(&flat[o + N_FEATURES..o + 2 * N_FEATURES]);
            p.fam_bias[s] = flat[o + 2 * N_FEATURES];
        }
    }
    (p, loss)
}

fn to_map(v: &[f32]) -> IndexMap<String, f32> {
    FEATURE_NAMES.iter().zip(v.iter()).filter(|(_, w)| **w != 0.0).map(|(n, w)| (n.to_string(), (*w * 10000.0).round() / 10000.0)).collect()
}

fn head_to_artifact(p: &HeadParams) -> Head {
    let mut families = IndexMap::new();
    for (s, fam) in Family::all().iter().enumerate() {
        if p.active[s] {
            families.insert(fam.as_str().to_string(), to_map(&p.deltas[s]));
        }
    }
    Head { base: to_map(&p.base), families }
}

pub fn run(inputs: Vec<PathBuf>, out_dir: PathBuf, seed: u64, l2: f32, family_l2: f32, epochs: usize, no_families: bool, drop_features: Option<String>) -> i32 {
    let _ = seed; // full-batch training is deterministic; the seed is recorded for provenance only
    let records = match load_records(&inputs) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {e}");
            return 2;
        }
    };
    let res = Resources::embedded();
    let drop = parse_drop(&drop_features);
    let t0 = std::time::Instant::now();
    let (examples, skipped) = extract_examples(&records, &res, &drop);
    eprintln!("extracted {} examples from {} records ({} skipped) in {:.1}s", examples.len(), records.len(), skipped, t0.elapsed().as_secs_f64());
    let semantic: Vec<&Example> = examples.iter().filter(|e| e.symbolic.is_none()).collect();
    let choice: Vec<&Example> = semantic.iter().copied().filter(|e| e.kind == QuestionKind::Choice).collect();
    let score: Vec<&Example> = semantic.iter().copied().filter(|e| e.kind == QuestionKind::Score).collect();
    let noul: Vec<&Example> = semantic.iter().copied().filter(|e| e.kind == QuestionKind::Noul).collect();
    eprintln!("semantic examples: choice={} score={} noul={} (symbolic={})", choice.len(), score.len(), noul.len(), examples.len() - semantic.len());
    let lr = 0.05;
    let (pc, lc) = train_multinomial(&choice, epochs, l2, family_l2, !no_families, lr);
    let (ps, ls) = train_multinomial(&score, epochs, l2, family_l2, !no_families, lr);
    let (pn, ln_) = train_noul(&noul, epochs, l2, family_l2, !no_families, lr);
    eprintln!("final losses: choice={lc:.4} score={ls:.4} noul={ln_:.4}");
    let mut noul_fams = IndexMap::new();
    for (s, fam) in Family::all().iter().enumerate() {
        if pn.active[s] {
            noul_fams.insert(fam.as_str().to_string(), NoulFamilyDelta { diff: to_map(&pn.fam_diff[s]), yes: to_map(&pn.fam_yes[s]), bias: pn.fam_bias[s] });
        }
    }
    let weights = Weights {
        version: format!("trained-{}", chrono_like_stamp()),
        description: format!(
            "Fitted by `sextant train` on {} records / {} semantic examples (choice={}, score={}, noul={}); epochs={epochs} l2={l2} family_l2={family_l2} seed={seed} dropped_features={:?}. Choice/Score: conditional logit; Noul: logistic over (f_yes-f_no, f_yes).",
            records.len(),
            semantic.len(),
            choice.len(),
            score.len(),
            noul.len(),
            drop_features.clone().unwrap_or_default()
        ),
        features: FEATURE_NAMES.iter().map(|s| s.to_string()).collect(),
        choice: head_to_artifact(&pc),
        score: head_to_artifact(&ps),
        noul: NoulHead { diff: to_map(&pn.diff), yes: to_map(&pn.yes), bias: (pn.bias * 10000.0).round() / 10000.0, families: noul_fams },
    };
    if let Err(e) = std::fs::create_dir_all(&out_dir) {
        eprintln!("error: {e}");
        return 1;
    }
    let path = out_dir.join("weights.json");
    std::fs::write(&path, serde_json::to_string_pretty(&weights).unwrap() + "\n").expect("write weights");
    println!("wrote {}", path.display());
    0
}

/// Date stamp without pulling in a date crate: seconds since epoch.
fn chrono_like_stamp() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    format!("{secs}")
}
