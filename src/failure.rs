//! Port of `shamwari/gateway/src/failure.ts`: why an answer could not be
//! given, in a form that can travel to the app.
//!
//! Before this, each inference step logged a bare status ("gateway 424") and
//! the caller answered `inference_unavailable`, so a person, and the owner, saw
//! "having trouble" while AI Gateway was saying exactly what was wrong ("Prompt
//! blocked due to security configurations", 2026-11-23). Each step now records
//! a reason, and the 503 carries them with the request id that finds the logs.
//!
//! What leaves the Worker is cleaned first: no URLs (they carry the account
//! id), no long hex or token-like strings, at most [`MESSAGE_MAX`] characters,
//! at most [`REASONS_MAX`] entries. The repository and the API are public; a
//! reason says what failed, never a credential or an internal address.
//!
//! Also here, because every other port of agent C logs and measures text the
//! way JavaScript does: [`console`] (the Worker log) and [`js_len`] /
//! [`js_prefix`] (string lengths in UTF-16 code units, as `.length` counts).

use regex_lite::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::fmt;
use std::sync::OnceLock;

/// Which step failed, in the order they were tried.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FailureStep {
    AiGateway,
    WorkersAi,
    Core,
    Worker,
}

impl FailureStep {
    pub fn as_str(self) -> &'static str {
        match self {
            FailureStep::AiGateway => "ai-gateway",
            FailureStep::WorkersAi => "workers-ai",
            FailureStep::Core => "core",
            FailureStep::Worker => "worker",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StepFailure {
    pub step: FailureStep,
    /// The upstream HTTP status, when there was one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<u16>,
    /// The upstream's own error code (AI Gateway's `internalCode`, e.g. 2016),
    /// when it gave one: a JSON number or string.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<Value>,
    /// What the upstream said, cleaned (see [`clean`]).
    pub message: String,
}

pub const MESSAGE_MAX: usize = 200;
pub const REASONS_MAX: usize = 5;

/// JavaScript's `\s`, spelled out: regex-lite's `\s` is ASCII only.
const JS_SPACE: &str = r"\t\n\x0B\x0C\r \x{A0}\x{1680}\x{2000}-\x{200A}\x{2028}\x{2029}\x{202F}\x{205F}\x{3000}\x{FEFF}";

struct Cleaners {
    url: Regex,
    credential: Regex,
    id: Regex,
    token: Regex,
    space: Regex,
}

fn cleaners() -> &'static Cleaners {
    static C: OnceLock<Cleaners> = OnceLock::new();
    C.get_or_init(|| Cleaners {
        url: Regex::new(&format!(r"(?i)\bhttps?://[^{JS_SPACE}]+")).expect("url pattern"),
        credential: Regex::new(&format!(
            r"(?i)\b(?:Bearer|Basic)[{JS_SPACE}]+[^{JS_SPACE}]+"
        ))
        .expect("credential pattern"),
        id: Regex::new(r"(?i)\b[0-9a-f]{24,}\b").expect("id pattern"),
        token: Regex::new(r"\b[A-Za-z0-9_-]{40,}\b").expect("token pattern"),
        space: Regex::new(&format!(r"[{JS_SPACE}]+")).expect("space pattern"),
    })
}

/// A message safe to show: no URLs, no long hex ids or tokens, one line, bounded.
pub fn clean(raw: &str) -> String {
    let c = cleaners();
    let s = c.url.replace_all(raw, "[url]");
    let s = c.credential.replace_all(&s, "[credential]");
    let s = c.id.replace_all(&s, "[id]");
    let s = c.token.replace_all(&s, "[token]");
    let s = c.space.replace_all(&s, " ");
    let mut s = s.trim().to_owned();
    if js_len(&s) > MESSAGE_MAX {
        s = format!("{}…", js_prefix(&s, MESSAGE_MAX - 1));
    }
    if s.is_empty() {
        "no detail given".to_owned()
    } else {
        s
    }
}

/// Read an upstream error body for its code and message. AI Gateway answers
/// `{ error: [{ code, message }], internalCode, message }`; OpenAI-style bodies
/// answer `{ error: { code, message } }`. Anything else is kept as text.
pub fn from_body(step: FailureStep, status: u16, body: &str) -> StepFailure {
    let mut code: Option<Value> = None;
    let mut message: Option<String> = None;
    match serde_json::from_str::<Value>(body) {
        // `JSON.parse` succeeded but reading `.error` of null threw: the catch keeps the text.
        Err(_) | Ok(Value::Null) => message = Some(body.to_owned()),
        Ok(Value::Object(j)) => {
            let e = present(j.get("error"));
            let first: Option<&Map<String, Value>> = match e {
                Some(Value::Array(a)) => a.first().and_then(Value::as_object),
                Some(Value::Object(o)) => Some(o),
                _ => None,
            };
            let c = present(j.get("internalCode"))
                .or_else(|| first.and_then(|f| present(f.get("code"))));
            if let Some(c @ (Value::Number(_) | Value::String(_))) = c {
                code = Some(c.clone());
            }
            let m = present(j.get("message"))
                .or_else(|| first.and_then(|f| present(f.get("message"))))
                .or_else(|| e.filter(|e| e.is_string()));
            if let Some(Value::String(m)) = m {
                message = Some(m.clone());
            }
        }
        // A number, string, boolean or array has no `error` to read.
        Ok(_) => {}
    }
    let message = match message {
        Some(m) if !m.is_empty() => m,
        _ => format!("HTTP {status}"),
    };
    StepFailure {
        step,
        status: Some(status),
        code,
        message: clean(&message),
    }
}

/// `??`: absent and null are both missing.
fn present(v: Option<&Value>) -> Option<&Value> {
    v.filter(|v| !v.is_null())
}

/// A thrown error as a failure of `step`.
pub fn from_error(step: FailureStep, e: &str) -> StepFailure {
    StepFailure {
        step,
        status: None,
        code: None,
        message: clean(e),
    }
}

/// Every inference step failed. Carries what each one said.
#[derive(Debug, Clone, PartialEq)]
pub struct InferenceFailed {
    pub failures: Vec<StepFailure>,
}

impl fmt::Display for InferenceFailed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let each: Vec<String> = self
            .failures
            .iter()
            .map(|x| {
                let status = match x.status {
                    Some(s) if s != 0 => format!(" {s}"),
                    _ => String::new(),
                };
                format!("{}{status}: {}", x.step.as_str(), x.message)
            })
            .collect();
        write!(f, "every inference step failed: {}", each.join("; "))
    }
}

impl std::error::Error for InferenceFailed {}

/// The reasons as they go out: bounded in number, each already cleaned.
pub fn public_reasons(failures: &[StepFailure]) -> Vec<StepFailure> {
    failures
        .iter()
        .take(REASONS_MAX)
        .map(|f| StepFailure {
            message: clean(&f.message),
            ..f.clone()
        })
        .collect()
}

/// A string's length as JavaScript's `.length` counts it (UTF-16 code units).
pub fn js_len(s: &str) -> usize {
    s.encode_utf16().count()
}

/// `s.slice(0, n)` in UTF-16 code units. A cut that would split a surrogate
/// pair stops before it (JavaScript would keep half of the pair).
pub fn js_prefix(s: &str, n: usize) -> &str {
    let mut units = 0;
    for (i, ch) in s.char_indices() {
        let w = ch.len_utf16();
        if units + w > n {
            return &s[..i];
        }
        units += w;
    }
    s
}

/// The Worker log (`console.warn` / `console.error`). Under `cargo test` each
/// line is kept on the test's own thread so a test can assert what was logged.
pub mod console {
    #[cfg(test)]
    thread_local! {
        static LINES: std::cell::RefCell<Vec<String>> = const { std::cell::RefCell::new(Vec::new()) };
    }

    #[allow(unused_variables)]
    fn emit(level: &str, line: &str) {
        #[cfg(test)]
        LINES.with(|l| l.borrow_mut().push(format!("{level}: {line}")));
        #[cfg(all(target_arch = "wasm32", not(test)))]
        match level {
            "warn" => worker::console_warn!("{}", line),
            _ => worker::console_error!("{}", line),
        }
        #[cfg(all(not(target_arch = "wasm32"), not(test)))]
        eprintln!("{level}: {line}");
    }

    pub fn warn(line: &str) {
        emit("warn", line);
    }

    pub fn error(line: &str) {
        emit("error", line);
    }

    /// Everything logged on this thread since the last call.
    #[cfg(test)]
    pub fn take() -> Vec<String> {
        LINES.with(|l| std::mem::take(&mut *l.borrow_mut()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn v(f: &StepFailure) -> Value {
        serde_json::to_value(f).unwrap()
    }

    #[test]
    fn reads_ai_gateways_own_error_body_the_guardrail_block_that_hid_behind_a_bare_424() {
        let body = json!({
            "success": false, "result": [], "messages": [],
            "error": [{ "code": 2016, "message": "Prompt blocked due to security configurations" }],
            "name": "AiGatewayError", "httpCode": 424, "internalCode": 2016,
            "message": "Prompt blocked due to security configurations",
        })
        .to_string();
        assert_eq!(
            v(&from_body(FailureStep::AiGateway, 424, &body)),
            json!({ "step": "ai-gateway", "status": 424, "code": 2016,
                    "message": "Prompt blocked due to security configurations" })
        );
    }

    #[test]
    fn reads_an_openai_style_error_and_a_body_that_is_not_json_at_all() {
        let body = json!({ "error": { "code": "rate_limited", "message": "Too many requests" } });
        assert_eq!(
            v(&from_body(FailureStep::AiGateway, 429, &body.to_string())),
            json!({ "step": "ai-gateway", "status": 429, "code": "rate_limited", "message": "Too many requests" })
        );
        assert_eq!(
            v(&from_body(
                FailureStep::AiGateway,
                502,
                "<html>Bad gateway</html>"
            )),
            json!({ "step": "ai-gateway", "status": 502, "message": "<html>Bad gateway</html>" })
        );
        assert_eq!(
            v(&from_body(FailureStep::AiGateway, 500, "")),
            json!({ "step": "ai-gateway", "status": 500, "message": "HTTP 500" })
        );
    }

    #[test]
    fn never_lets_a_url_a_credential_a_long_id_or_a_token_out() {
        let leaky = "POST https://gateway.ai.cloudflare.com/v1/125a2dfbc21f76a25c980609609e8218/shamwari/compat failed: Bearer sk_live_abc123 \
            id 125a2dfbc21f76a25c980609609e8218 token eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9eyJzdWIiOiIxMjM0NTY3ODkwIn0";
        let out = clean(leaky);
        assert!(!out.contains("http:") && !out.contains("https:"), "{out}");
        assert!(!out.contains("125a2dfbc21f76a25c980609609e8218"));
        assert!(!out.contains("sk_live_abc123"));
        assert!(!out.contains("eyJhbGci"));
        assert!(out.contains("[url]"));
        assert!(out.contains("[credential]"));
    }

    #[test]
    fn keeps_a_message_to_one_bounded_line_and_never_empty() {
        assert_eq!(clean("a\n\n  b"), "a b");
        assert_eq!(js_len(&clean(&"word ".repeat(300))), MESSAGE_MAX);
        assert_eq!(clean(""), "no detail given");
        assert_eq!(clean("kv down"), "kv down");
        assert_eq!(
            v(&from_error(
                FailureStep::WorkersAi,
                "3040: capacity exceeded"
            )),
            json!({ "step": "workers-ai", "message": "3040: capacity exceeded" })
        );
    }

    #[test]
    fn sends_at_most_a_few_reasons_each_cleaned_again_on_the_way_out() {
        let many: Vec<StepFailure> = (0..9)
            .map(|i| StepFailure {
                step: FailureStep::AiGateway,
                status: None,
                code: None,
                message: format!("try {i} https://x.example/secret"),
            })
            .collect();
        let out = public_reasons(&many);
        assert_eq!(out.len(), REASONS_MAX);
        assert_eq!(out[0].message, "try 0 [url]");
    }

    #[test]
    fn inference_failed_carries_every_step_and_says_them_in_its_own_message_for_the_logs() {
        let e = InferenceFailed {
            failures: vec![
                StepFailure {
                    step: FailureStep::AiGateway,
                    status: Some(424),
                    code: Some(json!(2016)),
                    message: "Prompt blocked due to security configurations".into(),
                },
                StepFailure {
                    step: FailureStep::WorkersAi,
                    status: None,
                    code: None,
                    message: "workers ai returned no text".into(),
                },
            ],
        };
        assert_eq!(e.failures.len(), 2);
        assert_eq!(
            e.to_string(),
            "every inference step failed: ai-gateway 424: Prompt blocked due to security configurations; workers-ai: workers ai returned no text"
        );
    }

    #[test]
    fn a_null_body_is_kept_as_text_and_a_null_internal_code_falls_back_to_the_first_error() {
        assert_eq!(from_body(FailureStep::Core, 500, "null").message, "null");
        let body = json!({ "internalCode": null, "error": [{ "code": 7, "message": "m" }] });
        let f = from_body(FailureStep::AiGateway, 400, &body.to_string());
        assert_eq!(f.code, Some(json!(7)));
        assert_eq!(f.message, "m");
    }

    #[test]
    fn utf16_lengths_count_as_javascript_does() {
        assert_eq!(js_len("a\u{1F600}"), 3);
        assert_eq!(js_prefix("ab\u{1F600}c", 3), "ab");
        assert_eq!(js_prefix("ab\u{1F600}c", 4), "ab\u{1F600}");
        assert_eq!(js_prefix("abc", 10), "abc");
    }
}
