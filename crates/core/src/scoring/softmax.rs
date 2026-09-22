//! Numerically stable softmax, normalized entropy and ordinal smoothing.

/// Softmax of `z / t`. Returns probabilities that sum to 1 (within f64 eps).
pub fn softmax_temp(z: &[f32], t: f32) -> Vec<f64> {
    if z.is_empty() {
        return Vec::new();
    }
    let t = if t > 1e-3 { t as f64 } else { 1e-3 };
    let m = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let mut out: Vec<f64> = z.iter().map(|&v| ((v as f64 - m) / t).exp()).collect();
    let s: f64 = out.iter().sum();
    for v in out.iter_mut() {
        *v /= s;
    }
    fix_sum(&mut out);
    out
}

/// Make the vector sum to exactly 1.0 by pushing the residual onto the largest entry.
pub fn fix_sum(p: &mut [f64]) {
    if p.is_empty() {
        return;
    }
    let s: f64 = p.iter().sum();
    if s > 0.0 && (s - 1.0).abs() > 0.0 {
        for v in p.iter_mut() {
            *v /= s;
        }
    }
    let s: f64 = p.iter().sum();
    let residual = 1.0 - s;
    if residual != 0.0 {
        let (i, _) =
            p.iter().enumerate().fold((0usize, f64::NEG_INFINITY), |acc, (i, &v)| if v > acc.1 { (i, v) } else { acc });
        p[i] = (p[i] + residual).clamp(0.0, 1.0);
    }
}

/// H(p) / log(K) in [0, 1]; 0 for K == 1.
pub fn entropy_normalized(p: &[f64]) -> f64 {
    let k = p.len();
    if k <= 1 {
        return 0.0;
    }
    let h: f64 = p.iter().filter(|&&v| v > 0.0).map(|&v| -v * v.ln()).sum();
    (h / (k as f64).ln()).clamp(0.0, 1.0)
}

/// Ordinal adjacency smoothing: p'_k ∝ Σ_j p_j λ^{|k-j|} restricted to |k-j| ≤ 2.
/// Keeps probability mass on plausible, contiguous levels and penalizes
/// discontinuous distributions. λ = 0 is the identity.
pub fn ordinal_smooth(p: &[f64], lambda: f64) -> Vec<f64> {
    let k = p.len();
    if k <= 1 || lambda <= 0.0 {
        return p.to_vec();
    }
    let mut out = vec![0.0f64; k];
    for (j, pj) in p.iter().enumerate() {
        for (i, o) in out.iter_mut().enumerate() {
            let d = (i as i64 - j as i64).unsigned_abs() as u32;
            if d <= 2 {
                *o += pj * lambda.powi(d as i32);
            }
        }
    }
    let s: f64 = out.iter().sum();
    for v in out.iter_mut() {
        *v /= s;
    }
    fix_sum(&mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_properties() {
        let p = softmax_temp(&[1.0, 2.0, 3.0], 1.0);
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(p[2] > p[1] && p[1] > p[0]);
        let p = softmax_temp(&[1000.0, -1000.0], 1.0);
        assert!(p[0] > 0.999);
        let p = softmax_temp(&[0.0, 0.0], 0.5);
        assert_eq!(p[0], 0.5);
    }

    #[test]
    fn entropy_and_smoothing() {
        assert!((entropy_normalized(&[0.5, 0.5]) - 1.0).abs() < 1e-12);
        assert_eq!(entropy_normalized(&[1.0, 0.0]), 0.0);
        let s = ordinal_smooth(&[1.0, 0.0, 0.0], 0.3);
        assert!(s[0] > s[1] && s[1] > s[2]);
        assert!((s.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }
}
