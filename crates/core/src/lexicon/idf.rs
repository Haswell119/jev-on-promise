//! Background inverse-document-frequency prior derived from a unigram
//! frequency list. Values are normalized to [0, 1]: 0 for the most common
//! word, 1 for words rarer than the table's floor (or unknown).

use crate::text::stem::stem;
use rustc_hash::FxHashMap;

#[derive(Debug, Default)]
pub struct IdfTable {
    by_stem: FxHashMap<Box<str>, f32>,
    by_surface: FxHashMap<Box<str>, f32>,
    pub entries: usize,
}

impl IdfTable {
    /// Parse `word<TAB>count` lines. Lines starting with `#` are ignored.
    pub fn parse(tsv: &str) -> IdfTable {
        let mut rows: Vec<(String, f64)> = Vec::new();
        for line in tsv.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut it = line.split('\t');
            let (Some(w), Some(c)) = (it.next(), it.next()) else { continue };
            let Ok(c) = c.trim().parse::<f64>() else { continue };
            if c <= 0.0 {
                continue;
            }
            rows.push((w.trim().to_lowercase(), c));
        }
        if rows.is_empty() {
            return IdfTable::default();
        }
        let total: f64 = rows.iter().map(|(_, c)| c).sum();
        let min_count = rows.iter().map(|(_, c)| *c).fold(f64::INFINITY, f64::min);
        let max_count = rows.iter().map(|(_, c)| *c).fold(0.0, f64::max);
        let denom = (max_count / min_count).ln().max(1e-9);
        let mut by_surface: FxHashMap<Box<str>, f32> = FxHashMap::default();
        let mut stem_counts: FxHashMap<String, f64> = FxHashMap::default();
        for (w, c) in &rows {
            let v = ((max_count / c).ln() / denom).clamp(0.0, 1.0) as f32;
            by_surface.insert(w.clone().into_boxed_str(), v);
            *stem_counts.entry(stem(w)).or_insert(0.0) += c;
        }
        let mut by_stem: FxHashMap<Box<str>, f32> = FxHashMap::default();
        for (s, c) in stem_counts {
            let v = ((max_count / c).ln() / denom).clamp(0.0, 1.0) as f32;
            by_stem.insert(s.into_boxed_str(), v);
        }
        let _ = total;
        IdfTable { entries: rows.len(), by_stem, by_surface }
    }

    /// IDF for a stem, falling back to the surface form, then to a default
    /// that depends on the token shape (numbers are specific but not topical).
    pub fn idf(&self, stem: &str, surface: &str) -> f32 {
        if let Some(&v) = self.by_stem.get(stem) {
            return v;
        }
        if let Some(&v) = self.by_surface.get(surface) {
            return v;
        }
        if self.entries == 0 {
            // No table: crude length-based prior so the engine still works.
            return match surface.len() {
                0..=2 => 0.2,
                3 => 0.45,
                4..=5 => 0.6,
                _ => 0.75,
            };
        }
        if surface.chars().all(|c| c.is_ascii_digit() || c == '.' || c == ',') {
            0.7
        } else if surface.len() <= 2 {
            0.5
        } else {
            1.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_normalizes() {
        let t = IdfTable::parse("the\t1000000\nrefund\t1000\nrefunds\t500\nzyx\t1\n");
        assert!(t.idf("the", "the") < 0.01);
        assert!(t.idf("refund", "refund") > 0.3 && t.idf("refund", "refund") < 0.7);
        assert_eq!(t.idf("unknownword", "unknownword"), 1.0);
        assert!(t.idf("42", "42") < 1.0);
    }
}
