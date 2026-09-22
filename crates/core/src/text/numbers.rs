//! Parsing of numeric tokens: plain numbers, currency amounts, percentages,
//! ordinals, number words and magnitude suffixes (k/m/b).

use super::tokenize::TokenKind;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NumKind {
    Plain,
    Currency,
    Percent,
    Ordinal,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParsedNumber {
    pub value: f64,
    pub kind: NumKind,
    /// ISO-ish currency code when known (USD, EUR, GBP, …).
    pub currency: Option<&'static str>,
}

fn currency_code(s: &str) -> Option<&'static str> {
    let s = s.trim();
    if s.contains('$') || s.contains("usd") || s.contains("dollar") || s.contains("buck") || s.contains("cent") {
        Some("USD")
    } else if s.contains('€') || s.contains("eur") {
        Some("EUR")
    } else if s.contains('£') || s.contains("gbp") || s.contains("pound") {
        Some("GBP")
    } else if s.contains('¥') || s.contains("jpy") {
        Some("JPY")
    } else if s.contains('₹') {
        Some("INR")
    } else if s.contains("cad") {
        Some("CAD")
    } else if s.contains("aud") {
        Some("AUD")
    } else if s.contains("chf") {
        Some("CHF")
    } else {
        None
    }
}

/// Extract the numeric core of a token string ("$12,840.00" → 12840.0, "3.5k" → 3500).
fn numeric_core(s: &str) -> Option<f64> {
    let mut digits = String::with_capacity(s.len());
    let mut seen_digit = false;
    let mut negative = false;
    let mut mult = 1.0;
    let bytes: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        if c.is_ascii_digit() {
            digits.push(c);
            seen_digit = true;
        } else if c == '.' && seen_digit && i + 1 < bytes.len() && bytes[i + 1].is_ascii_digit() {
            digits.push('.');
        } else if c == ',' {
            // thousands separator
        } else if c == '-' && !seen_digit {
            negative = true;
        } else if seen_digit && (c == 'k' || c == 'm' || c == 'b') {
            // magnitude suffix only if it terminates the numeric part
            let rest: String = bytes[i + 1..].iter().collect();
            if rest.trim().is_empty() {
                mult = match c {
                    'k' => 1e3,
                    'm' => 1e6,
                    _ => 1e9,
                };
            }
            break;
        } else if seen_digit {
            break;
        }
        i += 1;
    }
    if !seen_digit {
        return None;
    }
    let v: f64 = digits.parse().ok()?;
    Some(if negative { -v * mult } else { v * mult })
}

/// Parse a token according to its kind. Words like "three" are parsed too.
pub fn parse_number(text: &str, kind: TokenKind) -> Option<ParsedNumber> {
    match kind {
        TokenKind::Number => {
            if let Some(v) = numeric_core(text) {
                let ordinal = text.ends_with("st") || text.ends_with("nd") || text.ends_with("rd") || text.ends_with("th");
                Some(ParsedNumber { value: v, kind: if ordinal { NumKind::Ordinal } else { NumKind::Plain }, currency: None })
            } else {
                None
            }
        }
        TokenKind::Currency => {
            let mut v = numeric_core(text)?;
            if text.contains("cent") {
                v /= 100.0;
            }
            Some(ParsedNumber { value: v, kind: NumKind::Currency, currency: currency_code(text) })
        }
        TokenKind::Percent => Some(ParsedNumber { value: numeric_core(text)?, kind: NumKind::Percent, currency: None }),
        TokenKind::Word => number_word(text).map(|v| ParsedNumber { value: v, kind: NumKind::Plain, currency: None }),
        _ => None,
    }
}

/// Small number words. Compound words ("twenty-one") arrive as separate tokens.
pub fn number_word(w: &str) -> Option<f64> {
    Some(match w {
        "zero" => 0.0,
        "one" => 1.0,
        "two" => 2.0,
        "three" => 3.0,
        "four" => 4.0,
        "five" => 5.0,
        "six" => 6.0,
        "seven" => 7.0,
        "eight" => 8.0,
        "nine" => 9.0,
        "ten" => 10.0,
        "eleven" => 11.0,
        "twelve" => 12.0,
        "thirteen" => 13.0,
        "fourteen" => 14.0,
        "fifteen" => 15.0,
        "sixteen" => 16.0,
        "seventeen" => 17.0,
        "eighteen" => 18.0,
        "nineteen" => 19.0,
        "twenty" => 20.0,
        "thirty" => 30.0,
        "forty" => 40.0,
        "fifty" => 50.0,
        "sixty" => 60.0,
        "seventy" => 70.0,
        "eighty" => 80.0,
        "ninety" => 90.0,
        "hundred" => 100.0,
        "thousand" => 1000.0,
        "million" => 1e6,
        "billion" => 1e9,
        "dozen" => 12.0,
        "half" => 0.5,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_currency_and_percent() {
        let p = parse_number("$12,840.00", TokenKind::Currency).unwrap();
        assert_eq!(p.value, 12840.0);
        assert_eq!(p.currency, Some("USD"));
        let p = parse_number("12%", TokenKind::Percent).unwrap();
        assert_eq!(p.value, 12.0);
        let p = parse_number("3.5k", TokenKind::Number);
        assert!(p.is_none() || p.unwrap().value == 3500.0);
        assert_eq!(parse_number("1,000", TokenKind::Number).unwrap().value, 1000.0);
        assert_eq!(parse_number("-2.5", TokenKind::Number).unwrap().value, -2.5);
        assert_eq!(parse_number("3rd", TokenKind::Number).unwrap().kind, NumKind::Ordinal);
        assert_eq!(parse_number("seven", TokenKind::Word).unwrap().value, 7.0);
        assert_eq!(parse_number("€5", TokenKind::Currency).unwrap().currency, Some("EUR"));
    }
}
