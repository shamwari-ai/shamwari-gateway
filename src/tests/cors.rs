//! CORS through the real fetch handler, errors included: `test/cors.test.ts`
//! and the CORS cases of `test/responses.test.ts`.
//!
//! What it must NOT do matters as much: no wildcard, no reflecting an
//! unlisted origin, no preflight for other endpoints, no credentials, and no
//! change to the scope gate or to auth.

use super::world::*;
use crate::http::Req;
use serde_json::json;

fn req(method: &str, path: &str, headers: &[(&str, &str)], body: &str) -> Req {
    Req {
        method: method.into(),
        url: format!("https://gw.example{path}"),
        headers: headers
            .iter()
            .map(|(k, v)| (k.to_ascii_lowercase(), (*v).to_owned()))
            .collect(),
        body: body.into(),
    }
}

fn preflight(w: &World, origin: Option<&str>, path: &str) -> crate::http::Resp {
    let mut h = vec![
        ("access-control-request-method", "POST"),
        (
            "access-control-request-headers",
            "authorization, content-type, idempotency-key, x-shamwari-channel",
        ),
    ];
    if let Some(o) = origin {
        h.push(("origin", o));
    }
    w.send(req("OPTIONS", path, &h, ""))
}

#[test]
fn a_preflight_from_the_app_origin_is_answered_with_no_body_on_both_chat_paths() {
    let w = World::new();
    for path in ["/v1/chat/completions", "/v1/responses"] {
        let r = preflight(&w, Some(SITE), path);
        assert_eq!(r.status, 204, "{path}");
        assert!(r.body.is_empty());
        assert!(crate::http::Resp::is_null_body_status(r.status));
        assert_eq!(r.header("access-control-allow-origin"), Some(SITE));
        assert_eq!(
            r.header("access-control-allow-methods"),
            Some("POST, OPTIONS")
        );
        assert_eq!(r.header("access-control-allow-credentials"), None);
        assert_eq!(r.header("vary"), Some("Origin"));
    }
    let r = preflight(&w, Some("https://evil.example"), "/v1/chat/completions");
    assert_eq!(r.status, 204);
    assert_eq!(r.header("access-control-allow-origin"), None);
    for path in ["/v1/ground/context", "/v1/models", "/v1/channels/sms"] {
        assert_eq!(preflight(&w, Some(SITE), path).status, 404, "{path}");
    }
    assert!(w.requests().is_empty());
}

#[test]
fn an_options_that_is_not_a_preflight_is_answered_as_before() {
    let w = World::new();
    let r = w.send(req("OPTIONS", "/v1/chat/completions", &[], ""));
    assert_eq!(r.status, 404);
}

#[test]
fn errors_carry_the_allowed_origin_and_expose_retry_after() {
    let w = World::new();
    // No key: a 401, which a browser must still be able to read.
    let r = w.send(req(
        "POST",
        "/v1/chat/completions",
        &[("origin", SITE), ("content-type", "application/json")],
        "{}",
    ));
    assert_eq!(r.status, 401);
    assert_eq!(r.header("access-control-allow-origin"), Some(SITE));
    assert_eq!(
        r.header("access-control-expose-headers"),
        Some("Retry-After, Idempotent-Replayed")
    );
    assert!(r.header("vary").unwrap().contains("Origin"));
    for origin in [Some("https://evil.example"), None] {
        let mut h = vec![("content-type", "application/json")];
        if let Some(o) = origin {
            h.push(("origin", o));
        }
        let r = w.send(req("POST", "/v1/chat/completions", &h, "{}"));
        assert_eq!(r.status, 401);
        assert_eq!(r.header("access-control-allow-origin"), None);
        assert!(r.header("vary").unwrap().contains("Origin"));
    }
}

#[test]
fn a_409_a_429_and_a_503_carry_cors_with_their_own_headers_kept() {
    let w = World::new();
    let o = [("origin", SITE)];
    let r = w.chat_with(LIFE_Q, json!({ "scope": "personal" }), &o);
    assert_eq!(r.status, 409);
    assert_eq!(r.header("access-control-allow-origin"), Some(SITE));

    let r = w.chat_with(PERSONAL_Q, json!({}), &[("origin", SITE)]);
    assert_eq!(r.status, 200);
    assert_eq!(r.header("access-control-allow-origin"), Some(SITE));

    w.settings().core_status = Some(503);
    let r = w.chat_with(PERSONAL_Q, json!({}), &o);
    assert_eq!(r.status, 503);
    assert_eq!(r.header("retry-after"), Some("30"));
    assert_eq!(r.header("access-control-allow-origin"), Some(SITE));

    // A fresh world: the 503 above opened Core's breaker, and a rollup that
    // cannot be asked fails open (no 429), as it does in TypeScript.
    let w = World::new();
    w.settings().rollup_total = 1e9;
    let r = w.chat_with(LIFE_Q, json!({}), &o);
    assert_eq!(r.status, 429);
    assert_eq!(r.header("access-control-allow-origin"), Some(SITE));
}

#[test]
fn an_unexpected_error_is_a_readable_500_with_cors_not_an_opaque_failure() {
    let w = World::new();
    let r = w.send(req(
        "POST",
        "/v1/chat/completions",
        &[
            ("origin", SITE),
            ("authorization", &format!("Bearer {}", api_key())),
            ("content-type", "application/json"),
        ],
        "null",
    ));
    assert_eq!(r.status, 500);
    assert_eq!(r.header("access-control-allow-origin"), Some(SITE));
    assert_eq!(
        r.body_json(),
        json!({ "error": "internal_error", "reasons": [{ "step": "worker", "message": "Cannot read properties of null (reading 'messages')" }] })
    );
    assert!(crate::failure::console::take()
        .iter()
        .any(|l| l.contains("unhandled error")));
}

#[test]
fn a_replay_keeps_idempotent_replayed_and_exposes_it() {
    let w = World::new();
    let h = [("origin", SITE), ("idempotency-key", "cors-1")];
    w.chat_with(LIFE_Q, json!({}), &h);
    let r = w.chat_with(LIFE_Q, json!({}), &h);
    assert_eq!(r.header("idempotent-replayed"), Some("true"));
    assert_eq!(r.header("access-control-allow-origin"), Some(SITE));
}

#[test]
fn no_other_endpoint_carries_cors_so_a_page_cannot_read_it() {
    let w = World::new();
    for (path, method) in [
        ("/health", "GET"),
        ("/v1/ground/context", "POST"),
        ("/v1/channels/sms", "POST"),
        ("/v1/channels/ussd", "POST"),
        ("/v1/models", "GET"),
        ("/nope", "GET"),
    ] {
        let r = w.send(req(method, path, &[("origin", SITE)], "{}"));
        assert_eq!(
            r.header("access-control-allow-origin"),
            None,
            "{method} {path}"
        );
        assert_eq!(r.header("vary"), None, "{method} {path}");
    }
}

#[test]
fn the_responses_endpoint_carries_cors_on_errors_too() {
    let w = World::new();
    let r = w.send(req(
        "POST",
        "/v1/responses",
        &[("origin", SITE)],
        r#"{"input":"x"}"#,
    ));
    assert_eq!(r.status, 401);
    assert_eq!(r.header("access-control-allow-origin"), Some(SITE));
    let r = w.send(req(
        "POST",
        "/v1/responses",
        &[("origin", "https://evil.example")],
        r#"{"input":"x"}"#,
    ));
    assert_eq!(r.header("access-control-allow-origin"), None);
}
