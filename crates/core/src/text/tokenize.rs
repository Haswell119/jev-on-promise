//! Unicode-aware tokenizer with typed spans (words, numbers, currency,
//! percentages, dates, times, emails, URLs, identifiers, punctuation).
//!
//! Regexes are compiled exactly once. Output tokens carry byte offsets into
//! the (NFKC-normalized) input so evidence can be displayed verbatim.

use regex::Regex;
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TokenKind {
    Word,
    Number,
    Currency,
    Percent,
    Date,
    Time,
    Email,
    Url,
    Identifier,
    Punct,
    Other,
}

impl TokenKind {
    #[inline]
    pub fn is_content(self) -> bool {
        !matches!(self, TokenKind::Punct | TokenKind::Other)
    }
}

/// A token span in normalized text. `text` is the lowercase matching form.
#[derive(Debug, Clone, PartialEq)]
pub struct RawToken {
    pub start: usize,
    pub end: usize,
    pub text: String,
    pub kind: TokenKind,
    /// True when the original token started with an uppercase letter.
    pub capitalized: bool,
}

fn special_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        // Order matters: earlier alternatives win at the same position.
        Regex::new(concat!(
            r"(?i)",
            r#"(?P<url>(?:https?://|www\.)[^\s<>"']+)"#,
            r"|(?P<email>[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,})",
            r"|(?P<isodate>\b\d{4}-\d{2}-\d{2}(?:t\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?z?)?\b)",
            r"|(?P<slashdate>\b\d{1,2}[/.]\d{1,2}[/.]\d{2,4}\b)",
            r"|(?P<time>\b\d{1,2}:\d{2}(?::\d{2})?\s?(?:am|pm)?\b)",
            r"|(?P<currency>(?:[$€£¥₹]|usd|eur|gbp|cad|aud|chf|jpy)\s?-?\d[\d,]*(?:\.\d+)?(?:\s?[kmb])?\b|\b-?\d[\d,]*(?:\.\d+)?\s?(?:usd|eur|gbp|cad|aud|chf|jpy|dollars?|euros?|pounds?|cents?|bucks)\b)",
            r"|(?P<percent>-?\d[\d,]*(?:\.\d+)?\s?(?:%|percent\b|pct\b))",
            r"|(?P<ordinal>\b\d+(?:st|nd|rd|th)\b)",
            r"|(?P<ident>\b[a-z]{1,6}[-_]?\d{2,}[a-z0-9\-_]*\b|\b\d+[-_][a-z0-9\-_]*[a-z][a-z0-9\-_]*\b|#\s?\d+\b|\b[a-z]{2,}\d+[a-z0-9]*\b|\b\d+[a-z]{2,}[a-z0-9]*\b)",
            r"|(?P<number>-?\d[\d,]*(?:\.\d+)?\b)",
        ))
        .expect("tokenizer regex compiles")
    })
}

fn kind_of(caps: &regex::Captures<'_>) -> TokenKind {
    if caps.name("url").is_some() {
        TokenKind::Url
    } else if caps.name("email").is_some() {
        TokenKind::Email
    } else if caps.name("isodate").is_some() || caps.name("slashdate").is_some() {
        TokenKind::Date
    } else if caps.name("time").is_some() {
        TokenKind::Time
    } else if caps.name("currency").is_some() {
        TokenKind::Currency
    } else if caps.name("percent").is_some() {
        TokenKind::Percent
    } else if caps.name("ident").is_some() {
        TokenKind::Identifier
    } else {
        TokenKind::Number
    }
}

#[inline]
fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '\'' || c == '_'
}

fn push_word(out: &mut Vec<RawToken>, text: &str, start: usize, end: usize) {
    if text.is_empty() {
        return;
    }
    let capitalized = text.chars().next().map(|c| c.is_uppercase()).unwrap_or(false);
    let lower = super::normalize::lowercase(text);
    // Expand common English contractions into separate tokens.
    let (base, suffix): (&str, Option<&str>) = if let Some(b) = lower.strip_suffix("n't") {
        // can't -> can not, won't -> will not, shan't -> shall not
        let b = match b {
            "ca" => "can",
            "wo" => "will",
            "sha" => "shall",
            other => other,
        };
        (b, Some("not"))
    } else if let Some(b) = lower.strip_suffix("'re") {
        (b, Some("are"))
    } else if let Some(b) = lower.strip_suffix("'ve") {
        (b, Some("have"))
    } else if let Some(b) = lower.strip_suffix("'ll") {
        (b, Some("will"))
    } else if let Some(b) = lower.strip_suffix("'d") {
        (b, Some("would"))
    } else if let Some(b) = lower.strip_suffix("'m") {
        (b, Some("am"))
    } else if let Some(b) = lower.strip_suffix("'s") {
        (b, None)
    } else {
        (lower.as_str(), None)
    };
    let base = base.trim_matches('\'');
    if base.is_empty() && suffix.is_none() {
        return;
    }
    if !base.is_empty() {
        let kind = if base.chars().all(|c| c.is_ascii_digit()) {
            TokenKind::Number
        } else {
            TokenKind::Word
        };
        out.push(RawToken { start, end, text: base.to_string(), kind, capitalized });
    }
    if let Some(s) = suffix {
        out.push(RawToken { start, end, text: s.to_string(), kind: TokenKind::Word, capitalized: false });
    }
}

fn tokenize_plain(out: &mut Vec<RawToken>, text: &str, offset: usize) {
    let mut word_start: Option<usize> = None;
    for (i, c) in text.char_indices() {
        if is_word_char(c) {
            if word_start.is_none() {
                word_start = Some(i);
            }
        } else {
            if let Some(ws) = word_start.take() {
                push_word(out, &text[ws..i], offset + ws, offset + i);
            }
            if c.is_whitespace() {
                continue;
            }
            let kind = if c.is_ascii_punctuation() || c.is_alphanumeric() { TokenKind::Punct } else { TokenKind::Other };
            let kind = if c.is_ascii_punctuation() { TokenKind::Punct } else { kind };
            out.push(RawToken {
                start: offset + i,
                end: offset + i + c.len_utf8(),
                text: c.to_string(),
                kind,
                capitalized: false,
            });
        }
    }
    if let Some(ws) = word_start {
        push_word(out, &text[ws..], offset + ws, offset + text.len());
    }
}

/// Tokenize normalized text into typed tokens.
pub fn tokenize(text: &str) -> Vec<RawToken> {
    let mut out = Vec::with_capacity(text.len() / 5 + 4);
    let mut last = 0usize;
    for caps in special_re().captures_iter(text) {
        let m = caps.get(0).unwrap();
        // Skip matches glued to a preceding word char (e.g. "abc123" inside "xabc123y" handled by \b already).
        if m.start() > last {
            tokenize_plain(&mut out, &text[last..m.start()], last);
        } else if m.start() < last {
            continue;
        }
        let kind = kind_of(&caps);
        let mut start_pos = m.start();
        let raw_full = &text[m.start()..m.end()];
        // "3-5": a leading minus glued to a preceding word char is a dash, not a sign.
        if raw_full.starts_with('-') && text[..m.start()].chars().next_back().map(|c| c.is_alphanumeric()).unwrap_or(false) {
            out.push(RawToken { start: m.start(), end: m.start() + 1, text: "-".into(), kind: TokenKind::Punct, capitalized: false });
            start_pos += 1;
        }
        let raw = &text[start_pos..m.end()];
        let lower = super::normalize::lowercase(raw);
        out.push(RawToken { start: start_pos, end: m.end(), text: lower, kind, capitalized: false });
        last = m.end();
    }
    if last < text.len() {
        tokenize_plain(&mut out, &text[last..], last);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(s: &str) -> Vec<(String, TokenKind)> {
        tokenize(s).into_iter().map(|t| (t.text, t.kind)).collect()
    }

    #[test]
    fn words_and_contractions() {
        let t = kinds("I can't believe it's Don's, we're here!");
        let words: Vec<&str> = t.iter().map(|(s, _)| s.as_str()).collect();
        assert_eq!(words, vec!["i", "can", "not", "believe", "it", "don", ",", "we", "are", "here", "!"]);
    }

    #[test]
    fn typed_spans() {
        let t = kinds("Invoice #4471 issued 2026-03-03 to ACME for $12,840.00, net 30, 12% off, mail bob@x.io see https://x.io/a?b=1 at 10:30am");
        let find = |k: TokenKind| t.iter().filter(|(_, kk)| *kk == k).map(|(s, _)| s.clone()).collect::<Vec<_>>();
        assert_eq!(find(TokenKind::Identifier), vec!["#4471"]);
        assert_eq!(find(TokenKind::Date), vec!["2026-03-03"]);
        assert_eq!(find(TokenKind::Currency), vec!["$12,840.00"]);
        assert_eq!(find(TokenKind::Percent), vec!["12%"]);
        assert_eq!(find(TokenKind::Email), vec!["bob@x.io"]);
        assert_eq!(find(TokenKind::Url), vec!["https://x.io/a?b=1"]);
        assert_eq!(find(TokenKind::Time), vec!["10:30am"]);
        assert!(find(TokenKind::Number).contains(&"30".to_string()));
    }

    #[test]
    fn unicode_words() {
        let t = kinds("Café naïve 日本語 مرحبا");
        assert_eq!(t.len(), 4);
        assert_eq!(t[0].0, "café");
    }

    #[test]
    fn offsets_are_consistent() {
        let s = "Hello, world 42!";
        for t in tokenize(s) {
            let slice = &s[t.start..t.end];
            assert!(!slice.is_empty());
        }
    }
}
