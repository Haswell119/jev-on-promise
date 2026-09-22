//! `sextant calibrate`: fit calibration artifacts on a SEPARATE split using
//! the already-trained weights.
//!
//! * temperature scaling per primitive, per family (≥ MIN examples) and per
//!   cardinality bucket (multiplier);
//! * temperature for symbolic (resolver) logits;
//! * Platt scaling for Noul (global, per family, symbolic);
//! * ordinal smoothing λ for Score (NLL grid search);
//! * confidence map coefficients (logistic regression of correctness on
//!   distribution-shape + evidence features);
//! * optional comparison of Platt vs isotonic regression on a held-out half.

use super::dataset::load_records;
use super::learn::{extract_examples, Example};
use indexmap::IndexMap;
use sextant_core::api::QuestionKind;
use sextant_core::calibration::{cardinality_bucket, Calibration, PlattParams};
use sextant_core::features::F;
use sextant_core::model::Model;
use sextant_core::question::Family;
use sextant_core::scoring::{confidence::shape, ordinal_smooth, softmax_temp, ConfidenceParams};
use sextant_core::Resources;
use std::path::PathBuf;

const MIN_GROUP: usize = 40;

struct Multi {
    z: Vec<f32>,
    target: Vec<f64>,
    gold: usize,
    family: Family,
    k: usize,
    evidence: f64,
    ood: f64,
}

struct Binary {
    logit: f64,
    y: f64,
    family: Family,
}

fn nll_multi(items: &[&Multi], t: f64, lambda: f64, ordinal: bool) -> f64 {
    let mut s = 0.0;
    for it in items {
        let mut p = softmax_temp(&it.z, t as f32);
        if ordinal {
            p = ordinal_smooth(&p, lambda);
        }
        for k in 0..p.len() {
            if it.target[k] > 0.0 {
                s -= it.target[k] * p[k].max(1e-12).ln();
            }
        }
    }
    s / items.len().max(1) as f64
}

/// Golden-section search over log-temperature.
fn fit_temperature(items: &[&Multi], lambda: f64, ordinal: bool) -> f64 {
    if items.is_empty() {
        return 1.0;
    }
    let f = |lt: f64| nll_multi(items, lt.exp(), lambda, ordinal);
    let (mut a, mut b) = ((0.03f64).ln(), (8.0f64).ln());
    // coarse grid first
    let mut best = (f(0.0), 0.0);
    for i in 0..=40 {
        let lt = a + (b - a) * i as f64 / 40.0;
        let v = f(lt);
        if v < best.0 {
            best = (v, lt);
        }
    }
    a = best.1 - 0.4;
    b = best.1 + 0.4;
    let gr = (5f64.sqrt() - 1.0) / 2.0;
    let (mut c, mut d) = (b - gr * (b - a), a + gr * (b - a));
    for _ in 0..40 {
        if f(c) < f(d) {
            b = d;
        } else {
            a = c;
        }
        c = b - gr * (b - a);
        d = a + gr * (b - a);
    }
    ((a + b) / 2.0).exp()
}

fn fit_platt(items: &[&Binary]) -> PlattParams {
    if items.len() < 5 {
        return PlattParams::default();
    }
    // Gradient descent on BCE with light L2 on `a` toward 1; deterministic.
    let (mut a, mut b) = (1.0f64, 0.0f64);
    let n = items.len() as f64;
    let lr = 0.05;
    for _ in 0..4000 {
        let (mut ga, mut gb) = (0.0f64, 0.0f64);
        for it in items {
            let p = 1.0 / (1.0 + (-(a * it.logit + b)).exp());
            let d = p - it.y;
            ga += d * it.logit;
            gb += d;
        }
        ga = ga / n + 1e-3 * (a - 1.0);
        gb /= n;
        a -= lr * ga;
        b -= lr * gb;
    }
    PlattParams { a, b }
}

fn nll_binary(items: &[&Binary], p: &PlattParams) -> f64 {
    let mut s = 0.0;
    for it in items {
        let pr = (1.0 / (1.0 + (-(p.a * it.logit + p.b)).exp())).clamp(1e-12, 1.0 - 1e-12);
        s -= it.y * pr.ln() + (1.0 - it.y) * (1.0 - pr).ln();
    }
    s / items.len().max(1) as f64
}

fn ece_binary(pairs: &[(f64, f64)]) -> f64 {
    let bins = 10usize;
    let mut sum_c = vec![0.0; bins];
    let mut sum_y = vec![0.0; bins];
    let mut cnt = vec![0usize; bins];
    for (p, y) in pairs {
        let conf = p.max(1.0 - p);
        let correct = if (*p >= 0.5) == (*y >= 0.5) { 1.0 } else { 0.0 };
        let b = ((conf * bins as f64).floor() as usize).min(bins - 1);
        sum_c[b] += conf;
        sum_y[b] += correct;
        cnt[b] += 1;
    }
    let n = pairs.len().max(1) as f64;
    (0..bins).filter(|&b| cnt[b] > 0).map(|b| cnt[b] as f64 / n * (sum_y[b] / cnt[b] as f64 - sum_c[b] / cnt[b] as f64).abs()).sum()
}

/// Pool-adjacent-violators isotonic regression: returns (thresholds, values).
fn isotonic(mut pairs: Vec<(f64, f64)>) -> Vec<(f64, f64)> {
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let mut blocks: Vec<(f64, f64, f64)> = Vec::new(); // (x_max, sum_y, count)
    for (x, y) in pairs {
        blocks.push((x, y, 1.0));
        while blocks.len() >= 2 {
            let n = blocks.len();
            let (a, b) = (blocks[n - 2], blocks[n - 1]);
            if a.1 / a.2 > b.1 / b.2 {
                blocks.pop();
                blocks.pop();
                blocks.push((b.0, a.1 + b.1, a.2 + b.2));
            } else {
                break;
            }
        }
    }
    blocks.iter().map(|(x, sy, c)| (*x, sy / c)).collect()
}

fn isotonic_predict(model: &[(f64, f64)], x: f64) -> f64 {
    for (xm, v) in model {
        if x <= *xm {
            return *v;
        }
    }
    model.last().map(|(_, v)| *v).unwrap_or(0.5)
}

fn fit_confidence(rows: &[(Vec<f64>, bool)]) -> ConfidenceParams {
    // logistic regression: x = [concentration, margin, evidence, ood, 1]
    let mut w = [0.0f64; 5];
    if rows.len() < 20 {
        return ConfidenceParams::default();
    }
    let n = rows.len() as f64;
    let lr = 0.1;
    for _ in 0..5000 {
        let mut g = [0.0f64; 5];
        for (x, y) in rows {
            let z: f64 = (0..5).map(|i| w[i] * x[i]).sum();
            let p = 1.0 / (1.0 + (-z).exp());
            let d = p - if *y { 1.0 } else { 0.0 };
            for i in 0..5 {
                g[i] += d * x[i];
            }
        }
        for i in 0..5 {
            w[i] -= lr * (g[i] / n + 1e-4 * w[i]);
        }
    }
    ConfidenceParams { a_concentration: w[0], b_margin: w[1], c_evidence: w[2], d_ood: w[3], bias: w[4] }
}


fn to_multi(e: &Example, model: &Model) -> Multi {
    let head = if e.kind == QuestionKind::Choice { &model.dense.choice } else { &model.dense.score };
    let z: Vec<f32> = match &e.symbolic {
        Some(l) => l.clone(),
        None => e.rows.iter().map(|r| head.score(r, e.family)).collect(),
    };
    let k = e.rows.len();
    let target = match &e.soft {
        Some(s) if s.len() == k => s.clone(),
        _ => {
            let mut t = vec![0.0; k];
            t[e.gold] = 1.0;
            t
        }
    };
    let evidence = e.rows.iter().map(|r| r.get(F::cov_w)).fold(0.0f32, f32::max) as f64;
    let ood = e.rows.iter().map(|r| r.get(F::ood)).fold(1.0f32, f32::min) as f64;
    Multi { z, target, gold: e.gold, family: e.family, k, evidence, ood }
}

pub fn run(inputs: Vec<PathBuf>, out_dir: PathBuf, compare_isotonic: bool) -> i32 {
    let model = match std::fs::read_to_string(out_dir.join("weights.json")).map_err(|e| e.to_string()).and_then(|w| serde_json::from_str::<sextant_core::scoring::Weights>(&w).map_err(|e| e.to_string())) {
        Ok(w) => Model::from_parts(w, Calibration::default(), &out_dir.display().to_string()),
        Err(e) => {
            eprintln!("error: weights.json: {e} (run `sextant train` first)");
            return 2;
        }
    };
    let records = match load_records(&inputs) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {e}");
            return 2;
        }
    };
    let res = Resources::embedded();
    let (examples, skipped) = extract_examples(&records, &res, &[]);
    eprintln!("calibration examples: {} ({} skipped)", examples.len(), skipped);
    let mut cal = Calibration::default();
    cal.version = format!("calibrated-{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0));
    let mut report: Vec<String> = Vec::new();

    for kind in [QuestionKind::Choice, QuestionKind::Score] {
        let ordinal = kind == QuestionKind::Score;
        let sem: Vec<Multi> = examples.iter().filter(|e| e.kind == kind && e.symbolic.is_none()).map(|e| to_multi(e, &model)).collect();
        let sym: Vec<Multi> = examples.iter().filter(|e| e.kind == kind && e.symbolic.is_some()).map(|e| to_multi(e, &model)).collect();
        let refs: Vec<&Multi> = sem.iter().collect();
        let lambda0 = if ordinal { cal.ordinal_lambda } else { 0.0 };
        let t_global = fit_temperature(&refs, lambda0, ordinal);
        cal.temperature.insert(kind.as_str().into(), t_global);
        report.push(format!("{}: n_semantic={} n_symbolic={} T_global={:.3} NLL(before)={:.4} NLL(after)={:.4}", kind.as_str(), sem.len(), sym.len(), t_global, nll_multi(&refs, 1.0, lambda0, ordinal), nll_multi(&refs, t_global, lambda0, ordinal)));
        // ordinal lambda
        let mut lambda = lambda0;
        if ordinal && !refs.is_empty() {
            let mut best = (f64::INFINITY, 0.0);
            for i in 0..=12 {
                let l = i as f64 * 0.05;
                let v = nll_multi(&refs, t_global, l, true);
                if v < best.0 {
                    best = (v, l);
                }
            }
            lambda = best.1;
            cal.ordinal_lambda = lambda;
            report.push(format!("score: ordinal_lambda={lambda:.2} NLL={:.4}", best.0));
        }
        // per family
        let mut fam_map = IndexMap::new();
        for fam in Family::all() {
            let sub: Vec<&Multi> = sem.iter().filter(|m| m.family == *fam).collect();
            if sub.len() >= MIN_GROUP {
                let t = fit_temperature(&sub, lambda, ordinal);
                fam_map.insert(fam.as_str().to_string(), t);
                report.push(format!("  family {}: n={} T={:.3}", fam.as_str(), sub.len(), t));
            }
        }
        cal.family_temperature.insert(kind.as_str().into(), fam_map);
        // per bucket multiplier
        let mut bucket_map = IndexMap::new();
        for bucket in ["2", "3-4", "5-8", "9-16", "17-64", "65+"] {
            let sub: Vec<&Multi> = sem.iter().filter(|m| cardinality_bucket(m.k) == bucket).collect();
            if sub.len() >= MIN_GROUP {
                // temperature relative to the family/global base of each item
                let base: Vec<f64> = sub.iter().map(|m| cal.temperature_for(kind, m.family, 1)).collect();
                let mean_base = base.iter().sum::<f64>() / base.len() as f64;
                let t = fit_temperature(&sub, lambda, ordinal);
                bucket_map.insert(bucket.to_string(), (t / mean_base).clamp(0.25, 4.0));
                report.push(format!("  bucket {bucket}: n={} T={:.3} mult={:.3}", sub.len(), t, t / mean_base));
            }
        }
        cal.bucket_temperature.insert(kind.as_str().into(), bucket_map);
        // symbolic: keep T = 1 (resolver logits are already on a fixed scale) and
        // estimate the Laplace-smoothed error rate for uniform mixing.
        let sym_refs: Vec<&Multi> = sym.iter().collect();
        if !sym_refs.is_empty() {
            let errors = sym_refs.iter().filter(|m| {
                let pred = m.z.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |acc, (i, v)| if *v > acc.1 { (i, *v) } else { acc }).0;
                pred != m.gold
            }).count();
            let eps = (errors as f64 + 1.0) / (sym_refs.len() as f64 + 2.0);
            cal.symbolic_epsilon.insert(kind.as_str().into(), eps);
            report.push(format!("  symbolic: n={} errors={} epsilon={:.4}", sym.len(), errors, eps));
        }
        // confidence map on calibrated probabilities
        let mut rows: Vec<(Vec<f64>, bool)> = Vec::new();
        for m in sem.iter().chain(sym.iter()) {
            let t = if m.z.len() == m.k && sem.iter().any(|x| std::ptr::eq(x, m)) { cal.temperature_for(kind, m.family, m.k) } else { cal.symbolic_temperature_for(kind) };
            let mut p = softmax_temp(&m.z, t as f32);
            if ordinal {
                p = ordinal_smooth(&p, lambda);
            }
            let (conc, margin) = shape(&p);
            let pred = p.iter().enumerate().fold((0usize, f64::NEG_INFINITY), |acc, (i, v)| if *v > acc.1 { (i, *v) } else { acc }).0;
            rows.push((vec![conc, margin, m.evidence, m.ood, 1.0], pred == m.gold));
        }
        let cp = fit_confidence(&rows);
        // report confidence calibration quality
        let pairs: Vec<(f64, f64)> = rows
            .iter()
            .map(|(x, y)| {
                let z = cp.a_concentration * x[0] + cp.b_margin * x[1] + cp.c_evidence * x[2] + cp.d_ood * x[3] + cp.bias;
                (1.0 / (1.0 + (-z).exp()), if *y { 1.0 } else { 0.0 })
            })
            .collect();
        let mean_conf = pairs.iter().map(|p| p.0).sum::<f64>() / pairs.len().max(1) as f64;
        let acc = pairs.iter().map(|p| p.1).sum::<f64>() / pairs.len().max(1) as f64;
        report.push(format!("  confidence: a_conc={:.3} b_margin={:.3} c_evid={:.3} d_ood={:.3} bias={:.3} | mean_conf={:.3} acc={:.3} n={}", cp.a_concentration, cp.b_margin, cp.c_evidence, cp.d_ood, cp.bias, mean_conf, acc, rows.len()));
        cal.confidence.insert(kind.as_str().into(), cp);
    }

    // Noul
    let sem: Vec<Binary> = examples
        .iter()
        .filter(|e| e.kind == QuestionKind::Noul && e.symbolic.is_none())
        .map(|e| Binary { logit: model.dense.noul.logit(&e.rows[0], &e.rows[1], e.family) as f64, y: e.soft.as_ref().map(|s| s[1]).unwrap_or(e.gold as f64), family: e.family })
        .collect();
    let sym: Vec<Binary> = examples
        .iter()
        .filter(|e| e.kind == QuestionKind::Noul && e.symbolic.is_some())
        .map(|e| Binary { logit: e.symbolic.as_ref().unwrap()[0] as f64, y: e.gold as f64, family: e.family })
        .collect();
    let refs: Vec<&Binary> = sem.iter().collect();
    let platt = fit_platt(&refs);
    report.push(format!("noul: n_semantic={} n_symbolic={} platt a={:.3} b={:.3} NLL(before)={:.4} NLL(after)={:.4}", sem.len(), sym.len(), platt.a, platt.b, nll_binary(&refs, &PlattParams::default()), nll_binary(&refs, &platt)));
    cal.noul_platt = platt.clone();
    for fam in Family::all() {
        let sub: Vec<&Binary> = sem.iter().filter(|b| b.family == *fam).collect();
        if sub.len() >= MIN_GROUP {
            let p = fit_platt(&sub);
            report.push(format!("  family {}: n={} a={:.3} b={:.3}", fam.as_str(), sub.len(), p.a, p.b));
            cal.noul_family_platt.insert(fam.as_str().to_string(), p);
        }
    }
    let sym_refs: Vec<&Binary> = sym.iter().collect();
    if !sym_refs.is_empty() {
        let errors = sym_refs.iter().filter(|b| (b.logit >= 0.0) != (b.y >= 0.5)).count();
        let eps = (errors as f64 + 1.0) / (sym_refs.len() as f64 + 2.0);
        cal.symbolic_epsilon.insert("noul".into(), eps);
        // Platt on symbolic logits with `a` fixed near 1 keeps the fixed resolver scale.
        cal.noul_symbolic_platt = Some(PlattParams { a: 1.0, b: 0.0 });
        report.push(format!("  symbolic: n={} errors={} epsilon={:.4}", sym.len(), errors, eps));
    }
    if compare_isotonic && sem.len() >= 40 {
        // 2-fold comparison: fit on even, evaluate on odd (and vice versa)
        let mut tot = [0.0f64; 4]; // platt nll, iso nll, platt ece, iso ece
        for fold in 0..2 {
            let train: Vec<&Binary> = sem.iter().enumerate().filter(|(i, _)| i % 2 == fold).map(|(_, b)| b).collect();
            let test: Vec<&Binary> = sem.iter().enumerate().filter(|(i, _)| i % 2 != fold).map(|(_, b)| b).collect();
            let p = fit_platt(&train);
            let iso = isotonic(train.iter().map(|b| (b.logit, b.y)).collect());
            let platt_pairs: Vec<(f64, f64)> = test.iter().map(|b| (1.0 / (1.0 + (-(p.a * b.logit + p.b)).exp()), b.y)).collect();
            let iso_pairs: Vec<(f64, f64)> = test.iter().map(|b| (isotonic_predict(&iso, b.logit).clamp(0.01, 0.99), b.y)).collect();
            let nll = |pairs: &[(f64, f64)]| pairs.iter().map(|(p, y)| -(y * p.max(1e-12).ln() + (1.0 - y) * (1.0 - p).max(1e-12).ln())).sum::<f64>() / pairs.len() as f64;
            tot[0] += nll(&platt_pairs) / 2.0;
            tot[1] += nll(&iso_pairs) / 2.0;
            tot[2] += ece_binary(&platt_pairs) / 2.0;
            tot[3] += ece_binary(&iso_pairs) / 2.0;
        }
        report.push(format!("noul 2-fold comparison: platt NLL={:.4} ECE={:.4} | isotonic NLL={:.4} ECE={:.4} (artifact keeps Platt: monotone, 2 parameters, no binning artefacts)", tot[0], tot[2], tot[1], tot[3]));
    }
    for line in &report {
        println!("{line}");
    }
    cal.method = "temperature+platt".into();
    let path = out_dir.join("calibration.json");
    std::fs::write(&path, serde_json::to_string_pretty(&cal).unwrap() + "\n").expect("write calibration");
    println!("wrote {}", path.display());
    0
}
