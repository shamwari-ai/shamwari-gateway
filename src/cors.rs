//! Port of `shamwari/gateway/src/cors.ts`: which browser origins may call the
//! API, so the Shamwari Chat web app on shamwari.ai can call api.shamwari.ai.
//!
//! Deliberately narrow:
//!
//! - An exact-match allowlist of origins (`CORS_ORIGINS`, comma separated).
//!   Never `*`, never `null`, and an origin is never reflected unless it is on
//!   the list. Unset means the Shamwari sites; set but empty means NO origins,
//!   so CORS can be switched off by configuration.
//! - Only the chat endpoints (`/v1/chat/completions` and `/v1/responses`)
//!   answer a preflight and carry CORS headers. The ground, SMS and USSD
//!   endpoints and `/health` are untouched, so a page cannot read them.
//! - No credentials: callers send a Bearer key in a header, not a cookie, so
//!   `Access-Control-Allow-Credentials` is never set.
//! - This widens who may READ a response in a browser. It does not
//!   authenticate anyone and does not touch the scope gate.

use crate::auth::js_trim;
use crate::config::Config;
use crate::http::{Req, Resp};

/// The app origin only. The marketing site (www) makes no API calls.
pub const DEFAULT_CORS_ORIGINS: [&str; 1] = ["https://shamwari.ai"];

/// Request headers a browser may send. `x-shamwari-channel` can only withhold
/// a page's own traffic from training, never add to it.
const ALLOW_HEADERS: &str = "authorization, content-type, idempotency-key, x-shamwari-channel";

/// Response headers the page may read: `Retry-After` (503 and the in-progress
/// 409) and `Idempotent-Replayed`.
const EXPOSE_HEADERS: &str = "Retry-After, Idempotent-Replayed";

/// The chat endpoint, and Open Responses, which is the same chat path in another wire shape.
pub const CORS_PATHS: [&str; 2] = ["/v1/chat/completions", "/v1/responses"];

/// Trimmed, with every trailing slash removed (a linear scan).
fn normalise(o: &str) -> &str {
    js_trim(o).trim_end_matches('/')
}

pub fn allowed_origins(cfg: &Config) -> Vec<String> {
    match cfg.cors_origins.as_deref() {
        None => DEFAULT_CORS_ORIGINS
            .iter()
            .map(|o| (*o).to_owned())
            .collect(),
        Some(list) => list
            .split(',')
            .map(normalise)
            .filter(|o| !o.is_empty() && *o != "*" && *o != "null")
            .map(str::to_owned)
            .collect(),
    }
}

/// The origin to echo, or `None`. Exact match only.
pub fn cors_origin(cfg: &Config, req: &Req) -> Option<String> {
    let origin = req.header("origin")?;
    if origin.is_empty() || origin == "*" || origin == "null" {
        return None;
    }
    allowed_origins(cfg)
        .iter()
        .any(|o| o == origin)
        .then(|| origin.to_owned())
}

fn on_chat_path(req: &Req) -> bool {
    CORS_PATHS.contains(&req.path().as_str())
}

/// A CORS preflight answer for the chat endpoint, or `None` to carry on to the routes.
pub fn preflight(cfg: &Config, req: &Req) -> Option<Resp> {
    if req.method != "OPTIONS" || !on_chat_path(req) {
        return None;
    }
    // A real preflight names its origin and the method it wants. Anything else
    // is not CORS traffic and is left to the router, as before.
    if req.header("origin").is_none() || req.header("access-control-request-method").is_none() {
        return None;
    }
    let mut res = Resp {
        status: 204,
        headers: vec![],
        body: String::new(),
    };
    // A disallowed origin gets a bare 204 with no allow headers: the browser
    // then blocks the real request, which is the intended outcome.
    if let Some(origin) = cors_origin(cfg, req) {
        res.set_header("access-control-allow-origin", &origin);
        res.set_header("access-control-allow-methods", "POST, OPTIONS");
        res.set_header("access-control-allow-headers", ALLOW_HEADERS);
        res.set_header("access-control-max-age", "600");
    }
    res.set_header("vary", "Origin");
    Some(res)
}

/// `res` with the headers a browser needs, for the chat endpoint and an
/// allowed origin only. `Vary: Origin` goes on every chat response so a shared
/// cache can never hand one origin's variant to another.
pub fn with_cors(cfg: &Config, req: &Req, mut res: Resp) -> Resp {
    if !on_chat_path(req) {
        return res;
    }
    // `headers.append('Vary', 'Origin')`: joined to any Vary already there.
    let vary = match res.header("vary") {
        Some(v) => format!("{v}, Origin"),
        None => "Origin".to_owned(),
    };
    res.set_header("vary", &vary);
    if let Some(origin) = cors_origin(cfg, req) {
        res.set_header("access-control-allow-origin", &origin);
        res.set_header("access-control-expose-headers", EXPOSE_HEADERS);
    }
    res
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::{auth_failure, AuthResult};
    use crate::router::handle;
    use crate::testing::{block_on, FakePlatform};
    use serde_json::json;

    const SITE: &str = "https://shamwari.ai";

    fn env() -> Config {
        Config {
            auth_cache_ttl: Some("60".into()),
            ..Default::default()
        }
    }

    fn request(method: &str, path: &str, headers: &[(&str, &str)]) -> Req {
        Req {
            method: method.into(),
            url: format!("https://gw.example{path}"),
            headers: headers
                .iter()
                .map(|(k, v)| (k.to_ascii_lowercase(), (*v).to_owned()))
                .collect(),
            body: String::new(),
        }
    }

    fn preflight_via_router(origin: Option<&str>, path: &str, cfg: &Config) -> Resp {
        let mut h = vec![
            ("Access-Control-Request-Method", "POST"),
            (
                "Access-Control-Request-Headers",
                "authorization, content-type, idempotency-key, x-shamwari-channel",
            ),
        ];
        if let Some(o) = origin {
            h.push(("Origin", o));
        }
        block_on(handle(
            &FakePlatform::new(),
            cfg,
            request("OPTIONS", path, &h),
        ))
    }

    // ---- preflight

    #[test]
    fn lets_the_app_origin_send_the_headers_the_chat_app_sends() {
        for path in CORS_PATHS {
            let res = preflight_via_router(Some(SITE), path, &env());
            assert_eq!(res.status, 204);
            assert_eq!(res.header("Access-Control-Allow-Origin"), Some(SITE));
            assert_eq!(
                res.header("Access-Control-Allow-Methods"),
                Some("POST, OPTIONS")
            );
            let allowed: Vec<&str> = res
                .header("Access-Control-Allow-Headers")
                .unwrap()
                .split(", ")
                .collect();
            for h in [
                "authorization",
                "content-type",
                "idempotency-key",
                "x-shamwari-channel",
            ] {
                assert!(allowed.contains(&h), "{h}");
            }
            assert_eq!(res.header("Access-Control-Max-Age"), Some("600"));
            assert!(res.header("Vary").unwrap().contains("Origin"));
            assert!(res.body.is_empty());
        }
    }

    #[test]
    fn gives_an_unlisted_origin_nothing_so_the_browser_blocks_the_real_request() {
        for origin in [
            "https://evil.example",
            "https://shamwari.ai.evil.example",
            "http://shamwari.ai",
            "null",
        ] {
            let res = preflight_via_router(Some(origin), "/v1/chat/completions", &env());
            assert_eq!(res.status, 204, "{origin}");
            assert_eq!(res.header("Access-Control-Allow-Origin"), None);
            assert_eq!(res.header("Access-Control-Allow-Headers"), None);
            assert_eq!(res.header("Vary"), Some("Origin"));
        }
    }

    #[test]
    fn is_never_a_wildcard_and_never_allows_credentials() {
        let res = preflight_via_router(Some(SITE), "/v1/chat/completions", &env());
        assert_ne!(res.header("Access-Control-Allow-Origin"), Some("*"));
        assert_eq!(res.header("Access-Control-Allow-Credentials"), None);
    }

    #[test]
    fn only_answers_for_the_chat_endpoint() {
        for path in [
            "/v1/ground/context",
            "/v1/channels/sms",
            "/v1/channels/ussd",
        ] {
            let res = preflight_via_router(Some(SITE), path, &env());
            assert!(!(200..300).contains(&res.status), "{path}");
            assert_eq!(res.header("Access-Control-Allow-Methods"), None);
            assert_eq!(res.header("Access-Control-Allow-Headers"), None);
        }
    }

    // ---- responses

    fn post(origin: Option<&str>) -> Req {
        let h: Vec<(&str, &str)> = origin.map(|o| vec![("Origin", o)]).unwrap_or_default();
        request("POST", "/v1/chat/completions", &h)
    }

    #[test]
    fn carry_the_allowed_origin_and_expose_retry_after_on_errors_too() {
        // No key: a 401, which a browser must still be able to read.
        let res = with_cors(
            &env(),
            &post(Some(SITE)),
            auth_failure(&AuthResult::Invalid),
        );
        assert_eq!(res.status, 401);
        assert_eq!(res.header("Access-Control-Allow-Origin"), Some(SITE));
        assert_eq!(
            res.header("Access-Control-Expose-Headers"),
            Some("Retry-After, Idempotent-Replayed")
        );
        assert!(res.header("Vary").unwrap().contains("Origin"));
    }

    #[test]
    fn carry_nothing_for_an_unlisted_origin_or_for_no_origin_at_all() {
        for origin in [Some("https://evil.example"), None] {
            let res = with_cors(&env(), &post(origin), auth_failure(&AuthResult::Invalid));
            assert_eq!(res.status, 401);
            assert_eq!(res.header("Access-Control-Allow-Origin"), None);
        }
    }

    #[test]
    fn do_not_change_auth_cors_is_not_a_credential() {
        let res = with_cors(
            &env(),
            &post(Some(SITE)),
            auth_failure(&AuthResult::Invalid),
        );
        assert!(res.body_json()["error"].is_string());
        assert_eq!(res.status, 401);
    }

    // ---- scope of the change

    #[test]
    fn puts_no_cors_header_on_any_other_endpoint() {
        for (path, method) in [
            ("/health", "GET"),
            ("/v1/ground/context", "POST"),
            ("/v1/channels/sms", "POST"),
            ("/v1/channels/ussd", "POST"),
        ] {
            let res = block_on(handle(
                &FakePlatform::new(),
                &env(),
                request(method, path, &[("Origin", SITE)]),
            ));
            assert_eq!(
                res.header("Access-Control-Allow-Origin"),
                None,
                "{method} {path}"
            );
            assert_eq!(res.header("Vary"), None, "{method} {path}");
        }
    }

    #[test]
    fn answers_an_options_that_is_not_a_cors_preflight_exactly_as_before() {
        let res = block_on(handle(
            &FakePlatform::new(),
            &env(),
            request("OPTIONS", "/v1/chat/completions", &[]),
        ));
        assert_eq!(res.status, 404);
        // An origin without a requested method is not a preflight either.
        let res = block_on(handle(
            &FakePlatform::new(),
            &env(),
            request("OPTIONS", "/v1/chat/completions", &[("Origin", SITE)]),
        ));
        assert_eq!(res.status, 404);
    }

    #[test]
    fn keeps_status_body_and_headers_when_it_adds_cors_and_always_varies_on_origin() {
        let mut inner = Resp {
            status: 429,
            headers: vec![("content-type".into(), "application/json".into())],
            body: r#"{"error":"quota_exceeded"}"#.into(),
        };
        inner.set_header("Retry-After", "30");
        let out = with_cors(&env(), &post(Some(SITE)), inner);
        assert_eq!(out.status, 429);
        assert_eq!(out.header("Retry-After"), Some("30"));
        assert_eq!(out.header("content-type"), Some("application/json"));
        assert_eq!(out.body, r#"{"error":"quota_exceeded"}"#);

        let other = with_cors(&env(), &post(None), Resp::json(&json!("x"), 200));
        assert_eq!(other.header("Access-Control-Allow-Origin"), None);
        assert!(other.header("Vary").unwrap().contains("Origin"));

        let mut varied = Resp::json(&json!({}), 200);
        varied.set_header("Vary", "Accept-Encoding");
        let out = with_cors(&env(), &post(None), varied);
        assert_eq!(out.header("Vary"), Some("Accept-Encoding, Origin"));
    }

    // ---- configuration

    #[test]
    fn an_origin_set_in_the_environment_is_honoured() {
        let e = Config {
            cors_origins: Some("https://preview.example".into()),
            ..env()
        };
        let res = with_cors(
            &e,
            &post(Some("https://preview.example")),
            Resp::json(&json!({}), 401),
        );
        assert_eq!(
            res.header("Access-Control-Allow-Origin"),
            Some("https://preview.example")
        );
        let blocked = with_cors(&e, &post(Some(SITE)), Resp::json(&json!({}), 401));
        assert_eq!(blocked.header("Access-Control-Allow-Origin"), None);
        let pre = preflight_via_router(Some("https://preview.example"), "/v1/chat/completions", &e);
        assert_eq!(
            pre.header("Access-Control-Allow-Origin"),
            Some("https://preview.example")
        );
    }

    fn origins(v: Option<&str>) -> Vec<String> {
        allowed_origins(&Config {
            cors_origins: v.map(Into::into),
            ..Default::default()
        })
    }

    #[test]
    fn set_but_empty_switches_cors_off_only_unset_means_the_defaults() {
        assert!(origins(Some("")).is_empty());
        assert!(origins(Some("  ,  ")).is_empty());
        assert_eq!(origins(None), ["https://shamwari.ai"]);
        assert!(!origins(None).contains(&"https://www.shamwari.ai".to_owned()));
    }

    #[test]
    fn never_allows_the_null_origin_even_if_configured() {
        assert_eq!(
            origins(Some("null,https://a.example")),
            ["https://a.example"]
        );
        let cfg = Config {
            cors_origins: Some("null".into()),
            ..Default::default()
        };
        assert_eq!(
            cors_origin(&cfg, &request("GET", "/", &[("Origin", "null")])),
            None
        );
    }

    #[test]
    fn forgives_a_trailing_slash_in_the_configured_origin() {
        assert_eq!(
            origins(Some("https://shamwari.ai/")),
            ["https://shamwari.ai"]
        );
    }

    #[test]
    fn strips_every_trailing_slash_quickly_however_many_there_are() {
        let many = format!("https://shamwari.ai{}", "/".repeat(100_000));
        let start = std::time::Instant::now();
        assert_eq!(origins(Some(&many)), ["https://shamwari.ai"]);
        assert!(start.elapsed().as_millis() < 500);
        assert_eq!(
            origins(Some("///, https://a.example/")),
            ["https://a.example"]
        );
    }

    #[test]
    fn can_be_set_and_trims_and_ignores_blanks() {
        assert_eq!(
            origins(Some(" https://a.example , ,https://b.example ")),
            ["https://a.example", "https://b.example"]
        );
    }

    #[test]
    fn refuses_a_wildcard_even_if_one_is_configured() {
        let cfg = Config {
            cors_origins: Some("*".into()),
            ..Default::default()
        };
        assert_eq!(
            cors_origin(
                &cfg,
                &request("GET", "/", &[("Origin", "https://anything.example")])
            ),
            None
        );
        assert_eq!(
            cors_origin(&cfg, &request("GET", "/", &[("Origin", "*")])),
            None
        );
    }
}
