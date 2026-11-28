//! `GET /v1/models` through the router: `test/models.test.ts`.

use super::world::*;
use crate::http::Req;
use crate::platform::HttpResponse;
use serde_json::json;

fn get(w: &World, headers: &[(&str, &str)], method: &str) -> crate::http::Resp {
    w.send(Req {
        method: method.into(),
        url: "https://gw.example/v1/models".into(),
        headers: headers
            .iter()
            .map(|(k, v)| (k.to_ascii_lowercase(), (*v).to_owned()))
            .collect(),
        body: String::new(),
    })
}

fn bearer() -> String {
    format!("Bearer {}", api_key())
}

#[test]
fn answers_an_authenticated_caller_with_the_list() {
    let w = World::new();
    let r = get(&w, &[("authorization", &bearer())], "GET");
    assert_eq!(r.status, 200);
    assert_eq!(r.header("content-type"), Some("application/json"));
    assert_eq!(r.body_json(), crate::models::model_list(&w.cfg));
    assert!(
        r.body.contains("\n  \"object\": \"list\""),
        "pretty, as json()"
    );
}

#[test]
fn is_not_metered_no_rollup_and_no_key_service_call_when_the_key_is_cached() {
    let w = World::new();
    let r = get(&w, &[("authorization", &bearer())], "GET");
    assert_eq!(r.status, 200);
    assert!(w.requests().is_empty());
    assert!(w.sink().is_empty());
}

#[test]
fn refuses_a_missing_or_invalid_key_like_the_chat_endpoint() {
    let w = World::new();
    assert_eq!(get(&w, &[], "GET").status, 401);
    w.p.fake.kv.borrow_mut().insert(
        format!("auth:{}", crate::auth::sha256_hex(api_key())),
        "\"invalid\"".into(),
    );
    let r = get(&w, &[("authorization", &bearer())], "GET");
    assert_eq!(r.status, 401);
    assert_eq!(r.body_json(), json!({ "error": "invalid_api_key" }));
}

#[test]
fn is_a_503_with_retry_after_when_keys_cannot_be_checked() {
    let w = World::new();
    w.p.fake.kv.borrow_mut().clear();
    w.p.fake.responders.borrow_mut().insert(
        0,
        Box::new(|_| Some(Err::<HttpResponse, _>("down".to_owned()))),
    );
    let r = get(&w, &[("authorization", &bearer())], "GET");
    assert_eq!(r.status, 503);
    assert_eq!(r.header("retry-after"), Some("5"));
    assert_eq!(r.body_json(), json!({ "error": "auth_unavailable" }));
}

#[test]
fn only_answers_get() {
    let w = World::new();
    for m in ["POST", "PUT", "DELETE"] {
        assert_eq!(
            get(&w, &[("authorization", &bearer())], m).status,
            404,
            "{m}"
        );
    }
}

#[test]
fn carries_no_cors_headers() {
    let w = World::new();
    let r = get(&w, &[("authorization", &bearer()), ("origin", SITE)], "GET");
    assert_eq!(r.status, 200);
    assert_eq!(r.header("access-control-allow-origin"), None);
}
