//! Parse numeric ranges out of level / option descriptions such as
//! "Under $1,000", "$1,000 to $10,000", "Over 1,000,000", "3-5 days",
//! "at least 30", "Net 30", "more than 10%".

use crate::text::numbers::{parse_number, NumKind};
use crate::text::tokenize::{tokenize, TokenKind};

#[derive(Debug, Clone, PartialEq)]
pub struct NumericRange {
    pub lo: Option<f64>,
    pub hi: Option<f64>,
    pub lo_inclusive: bool,
    pub hi_inclusive: bool,
    pub kind: NumKind,
    /// Unit word following the number ("days", "units", "%") when present.
    pub unit: Option<String>,
    /// True when the description is a single exact value ("Net 30", "3").
    pub exact: bool,
}

impl NumericRange {
    pub fn contains(&self, v: f64) -> bool {
        if let Some(lo) = self.lo {
            if self.lo_inclusive {
                if v < lo {
                    return false;
                }
            } else if v <= lo {
                return false;
            }
        }
        if let Some(hi) = self.hi {
            if self.hi_inclusive {
                if v > hi {
                    return false;
                }
            } else if v >= hi {
                return false;
            }
        }
        true
    }

    /// Distance (0 when inside) used for soft scoring; relative to magnitude.
    pub fn distance(&self, v: f64) -> f64 {
        if self.contains(v) {
            return 0.0;
        }
        let d = match (self.lo, self.hi) {
            (Some(lo), _) if v < lo => lo - v,
            (_, Some(hi)) if v > hi => v - hi,
            (Some(lo), Some(hi)) => (v - lo).abs().min((v - hi).abs()),
            _ => 0.0,
        };
        let scale = self.lo.or(self.hi).map(|x| x.abs()).unwrap_or(1.0).max(1.0);
        d / scale
    }
}

fn is_num_kind(k: TokenKind) -> bool {
    matches!(k, TokenKind::Number | TokenKind::Currency | TokenKind::Percent)
}

/// Extract a numeric range from a short description. Returns None when the
/// text carries no number.
pub fn parse_range(text: &str) -> Option<NumericRange> {
    let toks = tokenize(&text.to_lowercase());
    let nums: Vec<(usize, f64, NumKind)> = toks
        .iter()
        .enumerate()
        .filter_map(|(i, t)| {
            if is_num_kind(t.kind) {
                parse_number(&t.text, t.kind).map(|p| (i, p.value, p.kind))
            } else if t.kind == TokenKind::Word {
                parse_number(&t.text, t.kind).map(|p| (i, p.value, p.kind))
            } else {
                None
            }
        })
        .collect();
    if nums.is_empty() {
        return None;
    }
    // Reject descriptions that are mostly prose with an incidental number.
    let words: Vec<&str> = toks.iter().filter(|t| t.kind == TokenKind::Word).map(|t| t.text.as_str()).collect();
    if words.len() > 12 {
        return None;
    }
    let kind = nums.iter().map(|n| n.2).find(|k| *k != NumKind::Plain).unwrap_or(NumKind::Plain);
    let unit = nums
        .last()
        .and_then(|(i, _, _)| toks.get(i + 1))
        .filter(|t| t.kind == TokenKind::Word && !matches!(t.text.as_str(), "or" | "and" | "to" | "of" | "in" | "per"))
        .map(|t| t.text.clone());
    let joined = words.join(" ");
    let before_first: Vec<&str> = words.iter().copied().take_while(|_| true).collect::<Vec<_>>();
    let _ = before_first;
    let has = |p: &str| joined.contains(p);
    let (a, b) = (nums[0].1, nums.get(1).map(|n| n.1));
    if let Some(b) = b {
        let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
        return Some(NumericRange {
            lo: Some(lo),
            hi: Some(hi),
            lo_inclusive: true,
            hi_inclusive: true,
            kind,
            unit,
            exact: false,
        });
    }
    let lower_words = [
        "under",
        "below",
        "less",
        "fewer",
        "up to",
        "at most",
        "maximum",
        "max",
        "no more than",
        "within",
        "or less",
        "or fewer",
        "or under",
        "shorter",
        "cheaper",
        "smaller",
        "lower",
    ];
    let upper_words = [
        "over", "above", "more", "greater", "at least", "minimum", "min", "exceed", "exceeds", "beyond", "or more",
        "or above", "or over", "longer", "larger", "higher", "+",
    ];
    let is_lower = lower_words.iter().any(|w| has(w)) || text.trim_end().ends_with("or less");
    let is_upper = upper_words.iter().any(|w| has(w)) || text.contains('+');
    if is_lower && !is_upper {
        let inclusive = has("up to")
            || has("at most")
            || has("or less")
            || has("or fewer")
            || has("or under")
            || has("max")
            || has("within")
            || has("no more");
        return Some(NumericRange {
            lo: None,
            hi: Some(a),
            lo_inclusive: false,
            hi_inclusive: inclusive,
            kind,
            unit,
            exact: false,
        });
    }
    if is_upper && !is_lower {
        let inclusive =
            has("at least") || has("or more") || has("or above") || has("or over") || has("min") || text.contains('+');
        return Some(NumericRange {
            lo: Some(a),
            hi: None,
            lo_inclusive: inclusive,
            hi_inclusive: false,
            kind,
            unit,
            exact: false,
        });
    }
    Some(NumericRange { lo: Some(a), hi: Some(a), lo_inclusive: true, hi_inclusive: true, kind, unit, exact: true })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges() {
        let r = parse_range("Under $1,000").unwrap();
        assert!(r.contains(500.0) && !r.contains(1000.0));
        let r = parse_range("$1,000 to $10,000").unwrap();
        assert!(r.contains(1000.0) && r.contains(10_000.0) && !r.contains(10_001.0));
        let r = parse_range("Over $1,000,000").unwrap();
        assert!(r.contains(2e6) && !r.contains(1e6));
        let r = parse_range("Net 30").unwrap();
        assert!(r.exact && r.contains(30.0));
        let r = parse_range("3-5 business days").unwrap();
        assert_eq!(r.unit.as_deref(), Some("business"));
        assert!(r.contains(4.0));
        assert!(parse_range("Calm and collected").is_none());
        let r = parse_range("at least 10%").unwrap();
        assert!(r.contains(10.0) && !r.contains(9.9));
    }
}
