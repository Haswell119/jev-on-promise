//! Sentence / fragment segmentation over normalized text. Deterministic and
//! abbreviation-aware; long fragments are chunked at clause punctuation so
//! retrieval works at a useful granularity.

const ABBREVIATIONS: &[&str] = &[
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e", "eg", "ie", "inc", "ltd", "co",
    "corp", "no", "fig", "approx", "dept", "est", "u.s", "u.k", "a.m", "p.m", "jan", "feb", "mar", "apr", "jun",
    "jul", "aug", "sep", "sept", "oct", "nov", "dec", "mt", "ave", "blvd", "rd",
];

/// Maximum characters per segment before chunking at clause boundaries.
pub const MAX_SEGMENT_CHARS: usize = 400;

fn is_abbreviation(word: &str) -> bool {
    let w = word.to_ascii_lowercase();
    ABBREVIATIONS.contains(&w.as_str()) || (w.len() == 1 && w.chars().all(|c| c.is_alphabetic()))
}

/// Return byte ranges of segments in `text`.
pub fn segment_ranges(text: &str) -> Vec<(usize, usize)> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut start = 0usize;
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let n = chars.len();
    let mut i = 0usize;
    while i < n {
        let (pos, c) = chars[i];
        let mut boundary = false;
        if c == '\n' {
            // newline is a hard boundary (bullets, chat turns, records)
            boundary = true;
        } else if c == '.' || c == '!' || c == '?' {
            // consume runs of terminal punctuation / closing quotes
            let mut j = i + 1;
            while j < n && matches!(chars[j].1, '.' | '!' | '?' | '"' | '\'' | ')' | ']') {
                j += 1;
            }
            let next_is_space = j < n && chars[j].1.is_whitespace();
            let at_end = j >= n;
            if c == '.' {
                // decimal number: 3.5
                let prev_digit = i > 0 && chars[i - 1].1.is_ascii_digit();
                let next_digit = i + 1 < n && chars[i + 1].1.is_ascii_digit();
                if prev_digit && next_digit {
                    i += 1;
                    continue;
                }
                // abbreviation check: word before the period
                let mut k = i;
                while k > 0 && (chars[k - 1].1.is_alphanumeric() || chars[k - 1].1 == '.') {
                    k -= 1;
                }
                let word = &text[chars[k].0..pos];
                if is_abbreviation(word) && !at_end {
                    i += 1;
                    continue;
                }
            }
            if next_is_space || at_end {
                // require the next non-space char to look like a sentence start
                let mut k = j;
                while k < n && chars[k].1.is_whitespace() {
                    k += 1;
                }
                if k >= n || chars[k].1.is_uppercase() || chars[k].1.is_ascii_digit() || matches!(chars[k].1, '"' | '\'' | '(' | '[' | '-' | '*' | '#') || c != '.' || chars[k].1.is_alphabetic() {
                    boundary = true;
                }
                if boundary {
                    // include trailing punctuation in the segment
                    i = j;
                    let end = if i < n { chars[i].0 } else { bytes.len() };
                    push_range(&mut out, text, start, end);
                    start = end;
                    continue;
                }
            }
        }
        if boundary {
            let end = pos;
            push_range(&mut out, text, start, end);
            start = pos + c.len_utf8();
        }
        i += 1;
    }
    push_range(&mut out, text, start, bytes.len());
    out
}

fn push_range(out: &mut Vec<(usize, usize)>, text: &str, start: usize, end: usize) {
    if end <= start {
        return;
    }
    let slice = &text[start..end];
    let trimmed_start = start + (slice.len() - slice.trim_start().len());
    let trimmed_end = end - (slice.len() - slice.trim_end().len());
    if trimmed_end <= trimmed_start {
        return;
    }
    if trimmed_end - trimmed_start > MAX_SEGMENT_CHARS {
        chunk_long(out, text, trimmed_start, trimmed_end);
    } else {
        out.push((trimmed_start, trimmed_end));
    }
}

/// Split an over-long fragment at clause punctuation (`,;:`) or whitespace.
fn chunk_long(out: &mut Vec<(usize, usize)>, text: &str, start: usize, end: usize) {
    let mut s = start;
    while end - s > MAX_SEGMENT_CHARS {
        let window = &text[s..end];
        let limit = MAX_SEGMENT_CHARS.min(window.len());
        // find a safe char boundary at or below limit
        let mut cut = limit;
        while cut > 0 && !window.is_char_boundary(cut) {
            cut -= 1;
        }
        let head = &window[..cut];
        let split_at = head
            .rfind([';', ':', ','])
            .map(|p| p + 1)
            .or_else(|| head.rfind(char::is_whitespace))
            .unwrap_or(cut);
        let split_at = if split_at == 0 { cut } else { split_at };
        let seg_end = s + split_at;
        let piece = &text[s..seg_end];
        let te = seg_end - (piece.len() - piece.trim_end().len());
        if te > s {
            out.push((s, te));
        }
        s = seg_end;
        while s < end && text[s..].starts_with(char::is_whitespace) {
            s += text[s..].chars().next().unwrap().len_utf8();
        }
    }
    if end > s {
        out.push((s, end));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segs(s: &str) -> Vec<&str> {
        segment_ranges(s).into_iter().map(|(a, b)| &s[a..b]).collect()
    }

    #[test]
    fn splits_sentences() {
        assert_eq!(
            segs("Hello there. My order is late! Can you help? Thanks."),
            vec!["Hello there.", "My order is late!", "Can you help?", "Thanks."]
        );
    }

    #[test]
    fn keeps_abbreviations_and_decimals() {
        assert_eq!(segs("Dr. Smith paid 3.5 dollars. Then left."), vec!["Dr. Smith paid 3.5 dollars.", "Then left."]);
        assert_eq!(segs("Ship to the U.S. office. Done."), vec!["Ship to the U.S. office.", "Done."]);
    }

    #[test]
    fn newlines_and_long_chunks() {
        assert_eq!(segs("line one\nline two\n\nline three"), vec!["line one", "line two", "line three"]);
        let long = "word ".repeat(200);
        let s = segs(&long);
        assert!(s.len() >= 2);
        assert!(s.iter().all(|x| x.len() <= MAX_SEGMENT_CHARS));
    }
}
