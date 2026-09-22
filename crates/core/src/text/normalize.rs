use unicode_normalization::UnicodeNormalization;

/// NFKC-normalize a string, unify quotes/dashes/whitespace, and strip
/// zero-width / control characters. Case is preserved (lowercasing happens
/// per token so the original casing can be inspected).
pub fn normalize_nfkc(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for ch in input.nfkc() {
        let mapped = match ch {
            '\u{2018}' | '\u{2019}' | '\u{201A}' | '\u{2032}' | '`' | '\u{00B4}' => '\'',
            '\u{201C}' | '\u{201D}' | '\u{201E}' | '\u{2033}' => '"',
            '\u{2010}' | '\u{2011}' | '\u{2012}' | '\u{2013}' | '\u{2014}' | '\u{2015}' | '\u{2212}' => '-',
            '\u{00A0}' | '\u{2007}' | '\u{202F}' | '\u{2009}' | '\u{200A}' | '\u{2002}' | '\u{2003}' | '\u{3000}' => {
                ' '
            }
            '\u{200B}' | '\u{200C}' | '\u{200D}' | '\u{FEFF}' | '\u{2060}' | '\u{00AD}' => continue,
            '\u{2026}' => {
                out.push_str("...");
                continue;
            }
            c if c.is_control() && c != '\n' && c != '\t' && c != '\r' => continue,
            c => c,
        };
        out.push(mapped);
    }
    out
}

/// Lowercase for matching. Uses Unicode lowercase; ASCII fast path.
#[inline]
pub fn lowercase(s: &str) -> String {
    if s.is_ascii() {
        s.to_ascii_lowercase()
    } else {
        s.to_lowercase()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nfkc_and_punct_unification() {
        assert_eq!(normalize_nfkc("ﬁle “quoted” – dash…"), "file \"quoted\" - dash...");
        assert_eq!(normalize_nfkc("zero\u{200B}width"), "zerowidth");
        assert_eq!(normalize_nfkc("don’t"), "don't");
    }
}
