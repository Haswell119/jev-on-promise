//! Valence lexicon (VADER-style `token<TAB>mean_valence[<TAB>…]`).
//! Valence is normalized to [-1, 1].

use crate::text::stem::stem;
use rustc_hash::FxHashMap;

#[derive(Debug, Default)]
pub struct SentimentLexicon {
    by_surface: FxHashMap<Box<str>, f32>,
    by_stem: FxHashMap<Box<str>, f32>,
    pub entries: usize,
}

impl SentimentLexicon {
    pub fn parse(tsv: &str) -> SentimentLexicon {
        let mut by_surface: FxHashMap<Box<str>, f32> = FxHashMap::default();
        let mut stem_acc: FxHashMap<String, (f32, u32)> = FxHashMap::default();
        let mut max_abs: f32 = 0.0;
        let mut rows = Vec::new();
        for line in tsv.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut it = line.split('\t');
            let (Some(w), Some(v)) = (it.next(), it.next()) else { continue };
            let Ok(v) = v.trim().parse::<f32>() else { continue };
            let w = w.trim().to_lowercase();
            if w.is_empty() || w.contains(' ') {
                continue;
            }
            max_abs = max_abs.max(v.abs());
            rows.push((w, v));
        }
        if max_abs <= 0.0 {
            return SentimentLexicon::default();
        }
        for (w, v) in rows {
            let nv = v / max_abs;
            let e = stem_acc.entry(stem(&w)).or_insert((0.0, 0));
            e.0 += nv;
            e.1 += 1;
            by_surface.insert(w.into_boxed_str(), nv);
        }
        let by_stem = stem_acc.into_iter().map(|(s, (sum, n))| (s.into_boxed_str(), sum / n as f32)).collect();
        SentimentLexicon { entries: by_surface.len(), by_surface, by_stem }
    }

    /// Valence in [-1, 1] for a lowercase surface token (stem fallback); 0 if unknown.
    #[inline]
    pub fn valence(&self, surface: &str, stem: &str) -> f32 {
        if let Some(&v) = self.by_surface.get(surface) {
            return v;
        }
        self.by_stem.get(stem).copied().unwrap_or(0.0)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valence_lookup() {
        let l = SentimentLexicon::parse("good\t1.9\nbad\t-2.5\nangry\t-2.3\t0.5\t[..]\n");
        assert!(l.valence("good", "good") > 0.5);
        assert!(l.valence("bad", "bad") < -0.9);
        assert!(l.valence("angrily", "angri") < 0.0 || l.valence("angry", "angri") < 0.0);
        assert_eq!(l.valence("table", "tabl"), 0.0);
    }
}
