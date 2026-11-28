//! Port of `shamwari/gateway/src/lang.ts`: a crude language tag.
//!
//! For training-set stratification and Ground filtering. Deliberately not a
//! model call: this runs on every request.
//!
//! Replace once groundMisses and the corpus show where it misfires.

/// The TypeScript patterns are `\b(word|...)\b` with the `i` flag and no `u`
/// flag: ASCII word boundaries and ASCII-only case folding. So a word matches
/// exactly when some maximal run of `[A-Za-z0-9_]` equals it, ignoring ASCII
/// case, which is what [`has_word`] checks.
const SN: &[&str] = &[
    "ndeipi",
    "mutero",
    "ndinoda",
    "zvakanaka",
    "chii",
    "sei",
    "vanhu",
    "mari",
    "ndiri",
    "tinoda",
    "handisi",
    "munhu",
    "basa",
    "mhoro",
];
const ND: &[&str] = &[
    "yini",
    "ngingathanda",
    "kuhle",
    "ngicela",
    "umuntu",
    "imali",
    "kanjani",
    "ngiyabonga",
    "sikhona",
    "lokhu",
    "umsebenzi",
    "sawubona",
];
const EN: &[&str] = &[
    "the", "and", "what", "how", "is", "are", "please", "tax", "law", "price",
];

fn has_word(text: &str, words: &[&str]) -> bool {
    text.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .any(|run| words.iter().any(|w| run.eq_ignore_ascii_case(w)))
}

pub fn detect_language(text: &str) -> Option<&'static str> {
    let sn = has_word(text, SN);
    let nd = has_word(text, ND);
    let en = has_word(text, EN);
    if sn && en {
        Some("sn-en")
    } else if nd && en {
        Some("nd-en")
    } else if sn {
        Some("sn")
    } else if nd {
        Some("nd")
    } else if en {
        Some("en")
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::detect_language;

    #[test]
    fn tags_single_languages() {
        assert_eq!(detect_language("Mhoro, ndinoda rubatsiro"), Some("sn"));
        assert_eq!(detect_language("Sawubona, ngicela usizo"), Some("nd"));
        assert_eq!(detect_language("What is the tax rate"), Some("en"));
    }

    #[test]
    fn tags_code_switched_input_which_is_the_common_case() {
        assert_eq!(detect_language("What is the mutero rate?"), Some("sn-en"));
        assert_eq!(detect_language("How much imali do I need?"), Some("nd-en"));
    }

    #[test]
    fn returns_none_when_nothing_matches_rather_than_guessing() {
        assert_eq!(detect_language("zzz 12345"), None);
        assert_eq!(detect_language(""), None);
    }

    #[test]
    fn matches_whole_ascii_words_only_as_the_typescript_boundaries_do() {
        // `\b` without the `u` flag: a non-ASCII letter is a boundary, an underscore is not.
        assert_eq!(detect_language("éthe"), Some("en"));
        assert_eq!(detect_language("the_x"), None);
        assert_eq!(detect_language("THE"), Some("en"));
        // No Unicode case folding: the Kelvin sign is not a "k".
        assert_eq!(detect_language("\u{212A}uhle"), None);
    }
}

/// JavaScript string semantics the ports in this crate keep, so a length, a
/// cut or a trim gives what the TypeScript Worker gives: lengths and cuts in
/// UTF-16 code units, and JavaScript's own whitespace set.
pub(crate) mod js {
    /// `\s` in a JavaScript regular expression, and what `trim()` removes.
    /// Not Rust's `char::is_whitespace`, which also takes U+0085 and leaves
    /// U+FEFF.
    pub fn is_space(c: char) -> bool {
        matches!(
            c,
            '\t' | '\n' | '\u{0B}' | '\u{0C}' | '\r' | ' ' | '\u{A0}' | '\u{1680}' | '\u{2000}'
                ..='\u{200A}'
                    | '\u{2028}'
                    | '\u{2029}'
                    | '\u{202F}'
                    | '\u{205F}'
                    | '\u{3000}'
                    | '\u{FEFF}'
        )
    }

    /// The same set as a regular expression class, for `\s` and `\S`.
    pub const SPACE: &str = r"\t\n\x0B\x0C\r \x{A0}\x{1680}\x{2000}-\x{200A}\x{2028}\x{2029}\x{202F}\x{205F}\x{3000}\x{FEFF}";

    pub fn trim(s: &str) -> &str {
        s.trim_matches(is_space)
    }

    pub fn trim_start(s: &str) -> &str {
        s.trim_start_matches(is_space)
    }

    pub fn trim_end(s: &str) -> &str {
        s.trim_end_matches(is_space)
    }

    /// `s.length`.
    pub fn len(s: &str) -> usize {
        s.encode_utf16().count()
    }

    /// `s.slice(0, n)`. A cut through a surrogate pair leaves U+FFFD where
    /// JavaScript leaves a lone surrogate: the same length either way.
    pub fn slice(s: &str, n: usize) -> String {
        let units: Vec<u16> = s.encode_utf16().collect();
        if n >= units.len() {
            return s.to_owned();
        }
        String::from_utf16_lossy(&units[..n])
    }

    /// `a < b` on strings, as JavaScript's default sort compares them (by
    /// UTF-16 code units, not by UTF-8 bytes).
    pub fn cmp(a: &str, b: &str) -> std::cmp::Ordering {
        a.encode_utf16().cmp(b.encode_utf16())
    }

    /// `encodeURIComponent`.
    pub fn encode_uri_component(s: &str) -> String {
        let mut out = String::with_capacity(s.len());
        for b in s.bytes() {
            if b.is_ascii_alphanumeric() || b"-_.!~*'()".contains(&b) {
                out.push(b as char);
            } else {
                out.push_str(&format!("%{b:02X}"));
            }
        }
        out
    }

    /// `Number(s)` for a `[vars]` value: `undefined` is NaN, an empty or
    /// blank string is 0, anything that is not a number is NaN.
    pub fn number(s: Option<&str>) -> f64 {
        let Some(s) = s else { return f64::NAN };
        let t = trim(s);
        if t.is_empty() {
            return 0.0;
        }
        let (sign, body) = match t.as_bytes()[0] {
            b'+' => (1.0, &t[1..]),
            b'-' => (-1.0, &t[1..]),
            _ => (1.0, t),
        };
        if body == "Infinity" {
            return sign * f64::INFINITY;
        }
        for (prefix, radix) in [
            ("0x", 16),
            ("0X", 16),
            ("0o", 8),
            ("0O", 8),
            ("0b", 2),
            ("0B", 2),
        ] {
            if let Some(digits) = t.strip_prefix(prefix) {
                return u64::from_str_radix(digits, radix)
                    .map(|v| v as f64)
                    .unwrap_or(f64::NAN);
            }
        }
        let numeric = body
            .bytes()
            .all(|b| b.is_ascii_digit() || matches!(b, b'.' | b'e' | b'E' | b'+' | b'-'));
        if !numeric || !body.bytes().any(|b| b.is_ascii_digit()) {
            return f64::NAN;
        }
        body.parse::<f64>().map(|v| sign * v).unwrap_or(f64::NAN)
    }

    /// A number as `JSON.stringify` writes it: an integer without a fraction,
    /// and NaN or an infinity as `null`.
    pub fn number_json(v: f64) -> serde_json::Value {
        if !v.is_finite() {
            serde_json::Value::Null
        } else if v.fract() == 0.0 && v.abs() < 9_007_199_254_740_992.0 {
            serde_json::Value::from(v as i64)
        } else {
            serde_json::Value::from(v)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use serde_json::json;

        #[test]
        fn number_reads_a_var_as_javascript_does() {
            assert_eq!(number(Some("6")), 6.0);
            assert_eq!(number(Some(" 6 ")), 6.0);
            assert_eq!(number(Some("")), 0.0);
            assert_eq!(number(Some("0x10")), 16.0);
            assert_eq!(number(Some("-Infinity")), f64::NEG_INFINITY);
            assert!(number(None).is_nan());
            assert!(number(Some("six")).is_nan());
            assert!(number(Some("inf")).is_nan());
            assert!(number(Some("1_0")).is_nan());
            assert_eq!(number_json(6.0), json!(6));
            assert_eq!(number_json(f64::NAN), json!(null));
            assert_eq!(number_json(0.5), json!(0.5));
        }

        #[test]
        fn lengths_cuts_and_trims_are_javascripts() {
            assert_eq!(len("a\u{1F600}"), 3);
            assert_eq!(slice("abc", 2), "ab");
            assert_eq!(slice("ab", 9), "ab");
            assert_eq!(trim("\u{FEFF} x \u{3000}"), "x");
            assert_eq!(trim("\u{85}x"), "\u{85}x");
            assert_eq!(encode_uri_component("a b/é"), "a%20b%2F%C3%A9");
        }
    }
}
