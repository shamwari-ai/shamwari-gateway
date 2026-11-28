//! `POST /v1/ground/context`, the retrieval-only endpoint Shamwari Mind
//! calls: the worker-level cases of `test/scope-decision.test.ts`,
//! `test/extractive-handler.test.ts` and `test/e2e.test.ts`.
//!
//! It is the only endpoint that may see personal scope, and only because
//! nothing on it reaches a provider: no edge embedding for a `mind`
//! destination, and no inference at all.

use super::world::*;
use serde_json::{json, Value};

#[test]
fn rule1_with_mind_not_available_explicit_personal_is_a_409_and_nothing_is_touched() {
    let w = World::new();
    w.scopes(&["personal", "community", "platform"]);
    let r = w.context("what is on my payslip?", json!({ "scope": "personal" }));
    assert_eq!(r.status, 409);
    let b = r.body_json();
    assert_eq!(b["error"], "scope_requires_local_inference");
    assert_eq!(b["capability"], "retrieval_only");
    assert_eq!(w.touched(), 0);
}

#[test]
fn rule1_with_mind_available_retrieval_only_no_embedding_no_provider() {
    let w = World::mind();
    w.scopes(&["personal", "community", "platform"]);
    let r = w.context(
        "All persons are equal before the law",
        json!({ "scope": "personal" }),
    );
    assert_eq!(r.status, 200);
    let j = r.body_json();
    assert_eq!(j["shamwari"]["destination"], "mind");
    assert_eq!(j["shamwari"]["scope"], "personal");
    let core = w.core();
    assert_eq!(core.len(), 1);
    assert_eq!(core[0]["scope"], "personal");
    assert_eq!(core[0]["destination"], "mind");
    assert!(core[0].get("embedding").is_none());
    assert_eq!(w.ai(), 0, "no edge embedding of the user's words");
    assert_eq!(w.completions().len(), 0, "no provider");
}

#[test]
fn rule1_a_key_without_personal_cannot_make_a_personal_request() {
    let w = World::mind(); // default scopes: community and platform
    let r = w.context("my payslip", json!({ "scope": "personal" }));
    assert_eq!(r.status, 403);
    assert_eq!(r.body_json()["error"], "scope_not_permitted_for_key");
    assert_eq!(w.touched(), 0);
}

#[test]
fn a_key_that_holds_personal_may_use_it_here() {
    let w = World::mind();
    w.scopes(&["personal", "platform"]);
    let r = w.context("my payslip", json!({ "scope": "personal" }));
    assert_eq!(r.status, 200);
    assert_eq!(w.completions().len(), 0);
}

#[test]
fn a_client_sent_scope_is_not_read_unless_it_is_an_explicit_personal() {
    let w = World::mind();
    let r = w.context(PLAIN_Q, json!({ "scope": "community" }));
    assert_eq!(r.status, 200);
    assert_eq!(
        w.core()
            .iter()
            .map(|b| b["scope"].clone())
            .collect::<Vec<_>>(),
        [json!("platform")]
    );
}

#[test]
fn the_mind_endpoint_never_says_answered_without_own_information_whatever_core_says() {
    let w = World::mind();
    w.scopes(&["personal", "platform"]);
    w.settings().core = Some(std::rc::Rc::new(
        |b: &Value| json!({ "chunks": [chunk(b["scope"].as_str().unwrap())], "hit": true, "scope": b["scope"], "needs_own_information": true }),
    ));
    let j = w
        .context(PERSONAL_Q, json!({ "scope": "personal" }))
        .body_json();
    assert_eq!(j["shamwari"]["destination"], "mind");
    assert_eq!(j["shamwari"]["answered_without_own_information"], false);
    assert_eq!(j["shamwari"]["notice"], Value::Null);
}

#[test]
fn rule1_a_personal_looking_question_is_not_embedded_at_the_edge_either() {
    let w = World::new();
    let r = w.context(PERSONAL_Q, json!({}));
    assert_eq!(r.status, 200);
    assert_eq!(w.provider_calls(), 0);
    assert!(w.core()[0].get("embedding").is_none());
    assert_eq!(
        r.body_json()["shamwari"]["answered_without_own_information"],
        true
    );
}

#[test]
fn a_plain_question_on_cloud_is_embedded_once_and_never_generated() {
    let w = World::new();
    let r = w.context(PLAIN_Q, json!({}));
    assert_eq!(r.status, 200);
    assert_eq!(w.embeds(), 1);
    assert_eq!(w.completions().len(), 0);
    let j = r.body_json();
    assert!(j["system_prompt"]
        .as_str()
        .unwrap()
        .starts_with("You are Shamwari"));
    assert_eq!(j["shamwari"]["inference_path"], "mind");
    assert_eq!(j["shamwari"]["grounded"], true);
    assert_eq!(
        j["shamwari"]["citations"][0]["heading"],
        json!("Section 48")
    );
}

#[test]
fn it_ignores_fields_it_has_never_read_and_still_honours_the_flag() {
    let a = World::new();
    assert_eq!(
        a.context(PLAIN_Q, json!({ "skills": "nonsense" })).status,
        200
    );
    let b = World::new();
    b.context(PLAIN_Q, json!({ "own_information": true }));
    assert_eq!(b.ai(), 0);
    assert!(b.core()[0].get("embedding").is_none());
    let c = World::mind();
    c.scopes(&["personal", "community", "platform"]);
    let r = c.context(
        "ndiri kuda alert",
        json!({ "scope": "personal", "mode": "chat", "reach": "r9", "format": "x" }),
    );
    assert_eq!(r.status, 200);
    let usage = c.doc("usageEvents").unwrap();
    assert_eq!(usage["scope"], "personal");
    assert_eq!(usage["inferencePath"], "mind");
}

#[test]
fn rule2_is_metered_but_writes_no_conversation() {
    let w = World::new();
    w.context(PLAIN_Q, json!({}));
    assert_eq!(w.collections(), ["usageEvents"]);
    let u = w.doc("usageEvents").unwrap();
    assert_eq!(u["tier"], "retrieval");
    assert_eq!(u["inferencePath"], "mind");
    assert_eq!(u["edgeEmbedding"], true);
    assert_eq!(u["provider"], Value::Null);
    assert!(!u.to_string().contains("section 48"));
}

#[test]
fn the_answer_is_byte_for_byte_what_typescript_writes() {
    let w = World::mind();
    w.scopes(&["personal", "platform"]);
    let r = w.context(PLAIN_Q, json!({ "scope": "personal" }));
    let j = r.body_json();
    let want = format!(
        r#"{{
  "id": "{id}",
  "system_prompt": {prompt},
  "shamwari": {{
    "scope": "personal",
    "destination": "mind",
    "language": "en",
    "grounded": true,
    "answered_without_own_information": false,
    "notice": null,
    "inference_path": "mind",
    "latency_ms": 0,
    "citations": [
      {{
        "title": "Constitution of Zimbabwe",
        "heading": "Section 48",
        "authority": "Parliament of Zimbabwe",
        "effective_from": "2013-05-22",
        "url": "https://example.invalid/constitution"
      }}
    ]
  }}
}}"#,
        id = j["id"].as_str().unwrap(),
        prompt = j["system_prompt"],
    );
    assert_eq!(r.body, want);
}

#[test]
fn it_is_authenticated_and_quota_checked_like_the_chat_endpoint() {
    let w = World::new();
    let r = w.post_with(
        "/v1/ground/context",
        &ask(PLAIN_Q, json!({})),
        &[("authorization", "")],
    );
    assert_eq!(r.status, 401);
    w.settings().rollup_total = 1e9;
    let r = w.context(PLAIN_Q, json!({}));
    assert_eq!(r.status, 429);
    assert_eq!(r.body_json()["error"], "quota_exceeded");
    assert_eq!(w.core().len(), 0);
}
