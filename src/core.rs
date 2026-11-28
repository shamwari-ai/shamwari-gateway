//! Port of `shamwari/gateway/src/core.ts`: the client for Shamwari Core, the
//! service that owns MongoDB.
//!
//! The Worker holds no database credentials. Core runs on Nyuchi
//! infrastructure, which is also why scope enforcement lives there: it needs
//! to be somewhere Cloudflare cannot see.

use crate::config::Config;
use crate::platform::{HttpRequest, Platform};
use serde_json::Value;
use std::cell::Cell;
use std::fmt;

/// How long a Core call may take before it counts as down. Without a limit a
/// Core that accepts the connection and then hangs would hold every chat
/// request until the page's own limit and show "offline" instead of
/// answering ungrounded.
pub const CORE_TIMEOUT_MS: u32 = 4000;

/// THE BREAKER. Once Core has failed to answer (no connection, the timeout,
/// or a 5xx), the calls after it in this isolate skip Core for this long and
/// fail at once, instead of each waiting out the timeout.
///
/// Skipping is NOT "Core said nothing": a skipped call fails with status 503,
/// exactly what a down Core gives, so every caller keeps its own meaning for
/// it. A 4xx is Core answering (a scope refusal, a bad key), so it never
/// opens the breaker; any answer closes it.
pub const CORE_COOLDOWN_MS: u64 = 30_000;

thread_local! {
    // Per isolate, as the TypeScript module-level `let downUntil`. A Worker
    // isolate is one thread; under `cargo test` each test has its own.
    static DOWN_UNTIL: Cell<u64> = const { Cell::new(0) };
}

/// True while the breaker is open.
pub fn core_down(now_ms: u64) -> bool {
    now_ms < DOWN_UNTIL.with(Cell::get)
}

/// For tests: close the breaker.
pub fn reset_core_breaker() {
    DOWN_UNTIL.with(|d| d.set(0));
}

fn set_down_until(v: u64) {
    DOWN_UNTIL.with(|d| d.set(v));
}

/// `CoreError`: Core answered with a status that is not 2xx, or was skipped
/// while the breaker is open (503).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoreError {
    pub status: u16,
    pub detail: String,
}

impl fmt::Display for CoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "core {}: {}", self.status, self.detail)
    }
}

/// Why a Core call gave no value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoreFailure {
    /// Core answered, or was skipped, with a status (the TypeScript `CoreError`).
    Status(CoreError),
    /// No answer: no connection or the timeout (the fetch threw), or a 2xx
    /// whose body was not JSON. The TypeScript rethrows these as they are.
    Threw(String),
}

impl CoreFailure {
    /// The status, when there was one.
    pub fn status(&self) -> Option<u16> {
        match self {
            CoreFailure::Status(e) => Some(e.status),
            CoreFailure::Threw(_) => None,
        }
    }
}

impl fmt::Display for CoreFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CoreFailure::Status(e) => e.fmt(f),
            CoreFailure::Threw(e) => f.write_str(e),
        }
    }
}

/// JavaScript truthiness, for `init.body ? ... : ...`.
fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0 && !f.is_nan()),
        Value::String(s) => !s.is_empty(),
        Value::Array(_) | Value::Object(_) => true,
    }
}

/// `core(env, path, { method, body })`: a call to Core with the Worker's Core
/// token. `method` defaults to GET.
pub async fn core<P: Platform>(
    p: &P,
    cfg: &Config,
    path: &str,
    method: Option<&str>,
    body: Option<&Value>,
) -> Result<Value, CoreFailure> {
    if core_down(p.now_ms()) {
        return Err(CoreFailure::Status(CoreError {
            status: 503,
            detail: "core_unavailable: skipped while the breaker is open".into(),
        }));
    }
    let body = body.filter(|b| truthy(b));
    let mut headers = vec![(
        "Authorization".to_owned(),
        format!(
            "Bearer {}",
            cfg.shamwari_core_token.as_deref().unwrap_or_default()
        ),
    )];
    if body.is_some() {
        headers.push(("Content-Type".to_owned(), "application/json".to_owned()));
    }
    let req = HttpRequest {
        method: method.unwrap_or("GET").to_owned(),
        url: format!("{}{path}", cfg.core_url.as_deref().unwrap_or_default()),
        headers,
        body: body.map(Value::to_string),
        timeout_ms: Some(CORE_TIMEOUT_MS),
    };
    let res = match p.fetch(req).await {
        Ok(r) => r,
        Err(e) => {
            set_down_until(p.now_ms() + CORE_COOLDOWN_MS);
            return Err(CoreFailure::Threw(e));
        }
    };
    if !res.ok() {
        if res.status >= 500 {
            set_down_until(p.now_ms() + CORE_COOLDOWN_MS);
        } else {
            set_down_until(0);
        }
        return Err(CoreFailure::Status(CoreError {
            status: res.status,
            detail: res.body,
        }));
    }
    set_down_until(0);
    serde_json::from_str(&res.body).map_err(|e| CoreFailure::Threw(e.to_string()))
}

/// `POST path` with a JSON body, returning Core's parsed JSON, and any failure
/// as its message (`core 503: ...`, or what the fetch said). The shape
/// `ground::ground` takes for Core's search:
/// `|body| core::core_json(p, cfg, "/ground/search", body)`. Same breaker.
pub async fn core_json<P: Platform>(
    p: &P,
    cfg: &Config,
    path: &str,
    body: Value,
) -> Result<Value, String> {
    core(p, cfg, path, Some("POST"), Some(&body))
        .await
        .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{block_on, FakePlatform};
    use serde_json::json;
    use std::cell::RefCell;
    use std::rc::Rc;

    fn cfg() -> Config {
        Config {
            core_url: Some("https://core.invalid".into()),
            shamwari_core_token: Some("t".into()),
            ..Default::default()
        }
    }

    fn call(p: &FakePlatform, path: &str, body: Option<&Value>) -> Result<Value, CoreFailure> {
        let method = body.map(|_| "POST");
        block_on(core(p, &cfg(), path, method, body))
    }

    // core-breaker.test.ts

    #[test]
    fn after_no_connection_the_next_call_fails_at_once_as_a_503_without_calling_core() {
        reset_core_breaker();
        let p = FakePlatform::new();
        p.on_fn(|_| Some(Err("connection refused".into())));
        let e = call(&p, "/ground/search", Some(&json!({}))).unwrap_err();
        assert!(e.to_string().contains("connection refused"));
        assert!(core_down(p.now_ms()));
        let skipped = call(&p, "/rollup", None).unwrap_err();
        assert_eq!(skipped.status(), Some(503));
        assert!(matches!(skipped, CoreFailure::Status(_)));
        assert_eq!(p.requests.borrow().len(), 1);
    }

    #[test]
    fn a_5xx_opens_it_it_closes_by_itself_after_the_cool_down_and_core_is_asked_again() {
        reset_core_breaker();
        let p = FakePlatform::new();
        let n = Rc::new(RefCell::new(0));
        let k = n.clone();
        p.on_fn(move |_| {
            *k.borrow_mut() += 1;
            Some(Ok(if *k.borrow() == 1 {
                crate::platform::HttpResponse {
                    status: 502,
                    headers: vec![],
                    body: "down".into(),
                }
            } else {
                crate::platform::HttpResponse {
                    status: 200,
                    headers: vec![],
                    body: r#"{"ok":true}"#.into(),
                }
            }))
        });
        assert_eq!(call(&p, "/rollup", None).unwrap_err().status(), Some(502));
        assert_eq!(call(&p, "/rollup", None).unwrap_err().status(), Some(503));
        assert_eq!(*n.borrow(), 1);
        *p.now.borrow_mut() += CORE_COOLDOWN_MS + 1;
        assert_eq!(call(&p, "/rollup", None).unwrap(), json!({ "ok": true }));
        assert_eq!(*n.borrow(), 2);
        assert!(!core_down(p.now_ms()));
    }

    #[test]
    fn a_4xx_core_refusing_a_scope_or_a_bad_key_never_opens_it() {
        reset_core_breaker();
        let p = FakePlatform::new();
        p.on(
            "https://core.invalid",
            403,
            r#"{"error":"forbidden_scope"}"#,
        );
        let body = json!({});
        assert_eq!(
            call(&p, "/ground/search", Some(&body))
                .unwrap_err()
                .status(),
            Some(403)
        );
        assert!(!core_down(p.now_ms()));
        assert_eq!(
            call(&p, "/ground/search", Some(&body))
                .unwrap_err()
                .status(),
            Some(403)
        );
        assert_eq!(p.requests.borrow().len(), 2);
    }

    // core-timeout.test.ts: core() never hangs.

    #[test]
    fn passes_a_timeout_so_a_silent_core_is_cut_off() {
        reset_core_breaker();
        let p = FakePlatform::new();
        p.on("https://core.invalid", 200, r#"{"ok":true}"#);
        call(&p, "/health", None).unwrap();
        assert_eq!(p.requests.borrow()[0].timeout_ms, Some(CORE_TIMEOUT_MS));
        const { assert!(CORE_TIMEOUT_MS <= 5000) };
    }

    #[test]
    fn fails_when_core_accepts_the_connection_and_never_answers() {
        reset_core_breaker();
        let p = FakePlatform::new();
        // The runtime aborts the fetch at the timeout and it fails.
        p.on_fn(|r| r.timeout_ms.map(|_| Err("timeout".into())));
        let e = call(&p, "/ground/search", Some(&json!({ "q": "x" }))).unwrap_err();
        assert_eq!(e, CoreFailure::Threw("timeout".into()));
    }

    #[test]
    fn core_json_posts_the_body_and_shares_the_breaker() {
        reset_core_breaker();
        let p = FakePlatform::new();
        p.on("https://core.invalid", 502, "down");
        let e = block_on(core_json(&p, &cfg(), "/ground/search", json!({ "q": "x" }))).unwrap_err();
        assert_eq!(e, "core 502: down");
        assert_eq!(p.requests.borrow()[0].method, "POST");
        // The breaker is open for every caller now, `core` included.
        assert_eq!(call(&p, "/rollup", None).unwrap_err().status(), Some(503));
        assert_eq!(p.requests.borrow().len(), 1);
    }

    #[test]
    fn sends_the_core_token_and_a_content_type_only_with_a_body() {
        reset_core_breaker();
        let p = FakePlatform::new();
        p.on("https://core.invalid", 200, "{}");
        call(&p, "/rollup", None).unwrap();
        call(&p, "/ground/search", Some(&json!({ "q": "x" }))).unwrap();
        let reqs = p.requests.borrow();
        assert_eq!(reqs[0].url, "https://core.invalid/rollup");
        assert_eq!(reqs[0].method, "GET");
        assert_eq!(
            reqs[0].headers,
            vec![("Authorization".to_owned(), "Bearer t".to_owned())]
        );
        assert_eq!(reqs[0].body, None);
        assert_eq!(reqs[1].method, "POST");
        assert_eq!(reqs[1].headers.len(), 2);
        assert_eq!(reqs[1].body.as_deref(), Some(r#"{"q":"x"}"#));
    }
}
