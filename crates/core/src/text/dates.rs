//! Lightweight date parsing (ISO, slash, and textual month forms) without
//! external dependencies. Dates are reduced to days-since-1970 for
//! comparisons.

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Date {
    pub year: i32,
    pub month: u32,
    pub day: u32,
}

impl Date {
    pub fn new(year: i32, month: u32, day: u32) -> Option<Date> {
        if !(1..=12).contains(&month) || day == 0 || day > days_in_month(year, month) || !(1..=9999).contains(&year) {
            return None;
        }
        Some(Date { year, month, day })
    }

    /// Days since 1970-01-01 (proleptic Gregorian). Howard Hinnant's algorithm.
    pub fn days_from_epoch(&self) -> i64 {
        let y = if self.month <= 2 { self.year as i64 - 1 } else { self.year as i64 };
        let era = if y >= 0 { y } else { y - 399 } / 400;
        let yoe = y - era * 400;
        let m = self.month as i64;
        let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + self.day as i64 - 1;
        let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
        era * 146097 + doe - 719468
    }
}

/// Inverse of `days_from_epoch` (Howard Hinnant's civil_from_days).
pub fn from_days(z: i64) -> Option<Date> {
    let z = z + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = if m <= 2 { y + 1 } else { y };
    Date::new(y as i32, m, d)
}

fn is_leap(y: i32) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

fn days_in_month(y: i32, m: u32) -> u32 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if is_leap(y) {
                29
            } else {
                28
            }
        }
        _ => 0,
    }
}

pub fn month_from_name(s: &str) -> Option<u32> {
    let s = s.trim_end_matches('.');
    Some(match s {
        "jan" | "january" => 1,
        "feb" | "february" => 2,
        "mar" | "march" => 3,
        "apr" | "april" => 4,
        "may" => 5,
        "jun" | "june" => 6,
        "jul" | "july" => 7,
        "aug" | "august" => 8,
        "sep" | "sept" | "september" => 9,
        "oct" | "october" => 10,
        "nov" | "november" => 11,
        "dec" | "december" => 12,
        _ => return None,
    })
}

fn expand_year(y: i64) -> i32 {
    if y < 100 {
        (if y < 70 { 2000 + y } else { 1900 + y }) as i32
    } else {
        y as i32
    }
}

/// Parse ISO `2026-03-03` (optionally with a time suffix) or slash/dot form
/// `03/03/2026`, `3.3.26`. Slash dates default to month-first when the first
/// component can be a month, otherwise day-first.
pub fn parse_date_token(text: &str) -> Option<Date> {
    let t = text.trim();
    if t.len() >= 10 && t.as_bytes()[4] == b'-' && t.as_bytes()[7] == b'-' {
        let y: i32 = t[0..4].parse().ok()?;
        let m: u32 = t[5..7].parse().ok()?;
        let d: u32 = t[8..10].parse().ok()?;
        return Date::new(y, m, d);
    }
    let parts: Vec<&str> = t.split(['/', '.']).collect();
    if parts.len() == 3 {
        let a: i64 = parts[0].parse().ok()?;
        let b: i64 = parts[1].parse().ok()?;
        let c: i64 = parts[2].parse().ok()?;
        let year = expand_year(c);
        if (1..=12).contains(&a) && (1..=31).contains(&b) {
            if let Some(d) = Date::new(year, a as u32, b as u32) {
                return Some(d);
            }
        }
        if (1..=12).contains(&b) && (1..=31).contains(&a) {
            return Date::new(year, b as u32, a as u32);
        }
    }
    None
}

/// Try to assemble a textual date from a window of lowercase word/number
/// tokens starting at `i`: `march 3, 2026`, `3 march 2026`, `mar 3`,
/// `march 2026`. Returns the date and the number of tokens consumed.
pub fn parse_textual_date(tokens: &[&str], i: usize, default_year: i32) -> Option<(Date, usize)> {
    let num = |s: &str| -> Option<i64> {
        let s = s.trim_end_matches("st").trim_end_matches("nd").trim_end_matches("rd").trim_end_matches("th");
        s.parse::<i64>().ok()
    };
    let t0 = tokens.get(i)?;
    if let Some(m) = month_from_name(t0) {
        // month day(, year) | month year
        if let Some(t1) = tokens.get(i + 1) {
            if let Some(n1) = num(t1) {
                let mut consumed = 2;
                let mut idx = i + 2;
                if tokens.get(idx) == Some(&",") {
                    idx += 1;
                    consumed += 1;
                }
                if (1..=31).contains(&n1) {
                    if let Some(t2) = tokens.get(idx) {
                        if let Some(y) = num(t2) {
                            if y >= 1000 {
                                return Date::new(y as i32, m, n1 as u32).map(|d| (d, consumed + 1));
                            }
                        }
                    }
                    return Date::new(default_year, m, n1 as u32).map(|d| (d, 2));
                }
                if n1 >= 1000 {
                    return Date::new(n1 as i32, m, 1).map(|d| (d, 2));
                }
            }
        }
        return None;
    }
    // day month (year)
    let d = num(t0)?;
    if !(1..=31).contains(&d) {
        return None;
    }
    let t1 = tokens.get(i + 1)?;
    let (t1, skip) = if *t1 == "of" { (tokens.get(i + 2)?, 1) } else { (t1, 0) };
    let m = month_from_name(t1)?;
    if let Some(t2) = tokens.get(i + 2 + skip) {
        if let Some(y) = num(t2) {
            if y >= 1000 {
                return Date::new(y as i32, m, d as u32).map(|dd| (dd, 3 + skip));
            }
        }
    }
    Date::new(default_year, m, d as u32).map(|dd| (dd, 2 + skip))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iso_and_slash() {
        assert_eq!(parse_date_token("2026-03-03"), Date::new(2026, 3, 3));
        assert_eq!(parse_date_token("2026-03-03T10:00:00Z"), Date::new(2026, 3, 3));
        assert_eq!(parse_date_token("03/04/2026"), Date::new(2026, 3, 4));
        assert_eq!(parse_date_token("25/04/2026"), Date::new(2026, 4, 25));
        assert_eq!(parse_date_token("3.3.26"), Date::new(2026, 3, 3));
        assert_eq!(parse_date_token("2026-13-03"), None);
    }

    #[test]
    fn textual() {
        let toks = ["issued", "march", "3", ",", "2026", "to"];
        assert_eq!(parse_textual_date(&toks, 1, 2000), Some((Date::new(2026, 3, 3).unwrap(), 4)));
        let toks = ["on", "3", "march", "2026"];
        assert_eq!(parse_textual_date(&toks, 1, 2000), Some((Date::new(2026, 3, 3).unwrap(), 3)));
        let toks = ["mar", "3"];
        assert_eq!(parse_textual_date(&toks, 0, 2024), Some((Date::new(2024, 3, 3).unwrap(), 2)));
    }

    #[test]
    fn from_days_roundtrip() {
        for (y, m, d) in [(1970, 1, 1), (2000, 2, 29), (2026, 3, 3), (2024, 12, 31)] {
            let date = Date::new(y, m, d).unwrap();
            assert_eq!(from_days(date.days_from_epoch()), Some(date));
        }
        assert_eq!(from_days(Date::new(2026, 3, 10).unwrap().days_from_epoch() - 3), Date::new(2026, 3, 7));
    }

    #[test]
    fn epoch_days() {
        assert_eq!(Date::new(1970, 1, 1).unwrap().days_from_epoch(), 0);
        assert_eq!(Date::new(2000, 3, 1).unwrap().days_from_epoch(), 11017);
        assert!(Date::new(2026, 3, 3).unwrap() > Date::new(2026, 2, 28).unwrap());
    }
}
