//! `Idempotency-Key` through the chat handler: `test/idempotency.test.ts`.
//!
//! The counts are the tests: a replay must make no Ground, inference or sink
//! call, so a retry over a flaky link is neither slow nor billed twice; and
//! personal scope (Rule 1) is refused before anything is stored, so a stored
//! answer is always a platform or community one.

use super::world::*;
use crate::idempotency::{DONE_TTL_SECONDS, PENDING_TTL_SECONDS};
use serde_json::{json, Value};

fn with_key(w: &World, key: &str, extra: Value) -> crate::http::Resp {
    w.chat_with(LIFE_Q, extra, &[("Idempotency-Key", key)])
}

fn counts(w: &World) -> (usize, usize, usize) {
    (w.core().len(), w.completions().len(), w.sink().len())
}

#[test]
fn a_retry_with_the_same_key_is_the_same_answer_with_nothing_run_again() {
    let w = World::new();
    let first = with_key(&w, "retry-1", json!({}));
    assert_eq!(first.status, 200);
    assert_eq!(first.header("idempotent-replayed"), None);
    let after_first = counts(&w);

    let second = with_key(&w, "retry-1", json!({}));
    assert_eq!(second.status, 200);
    assert_eq!(second.header("idempotent-replayed"), Some("true"));
    assert_eq!(second.header("content-type"), Some("application/json"));
    assert_eq!(second.body, first.body);
    assert_eq!(counts(&w), after_first);
}

#[test]
fn without_a_key_every_request_runs_and_nothing_is_stored() {
    let w = World::new();
    w.chat(LIFE_Q, json!({}));
    w.chat(LIFE_Q, json!({}));
    assert_eq!(w.completions().len(), 2);
    assert!(w.idem_keys().is_empty());
}

#[test]
fn the_same_key_with_a_different_request_is_refused_not_answered() {
    let w = World::new();
    with_key(&w, "k", json!({}));
    let before = w.completions().len();
    let r = w.post_with(
        "/v1/chat/completions",
        &ask("Something else entirely", json!({})),
        &[("Idempotency-Key", "k")],
    );
    assert_eq!(r.status, 422);
    assert_eq!(
        r.body_json(),
        json!({ "error": "idempotency_key_reuse", "message": "This key was used for a different request." })
    );
    assert_eq!(w.completions().len(), before);
}

#[test]
fn the_format_is_part_of_the_request_so_the_same_key_cannot_change_shape() {
    let w = World::new();
    with_key(&w, "k", json!({}));
    assert_eq!(
        with_key(&w, "k", json!({ "format": "compact" })).status,
        422
    );
}

#[test]
fn a_field_sent_as_null_is_a_different_request_from_one_not_sent() {
    let w = World::new();
    with_key(&w, "k", json!({}));
    assert_eq!(with_key(&w, "k", json!({ "model": null })).status, 422);
}

#[test]
fn is_scoped_to_the_calling_entity() {
    let w = World::new();
    w.auth_with(|a| a.owner_entity_id = "entity-a".into());
    with_key(&w, "shared", json!({}));
    w.auth_with(|a| a.owner_entity_id = "entity-b".into());
    let r = with_key(&w, "shared", json!({}));
    assert_eq!(r.header("idempotent-replayed"), None);
    assert_eq!(w.completions().len(), 2);
}

#[test]
fn a_request_still_in_flight_is_a_409_with_retry_after_not_a_second_run() {
    let w = World::new();
    with_key(&w, "slow", json!({}));
    // Turn the stored answer back into the in-flight marker a retry would see.
    let done = w
        .idem_keys()
        .into_iter()
        .find(|k| k.ends_with(":d"))
        .unwrap();
    let mut kv = w.p.fake.kv.borrow_mut();
    let fp = serde_json::from_str::<Value>(&kv[&done]).unwrap()["fp"].clone();
    kv.remove(&done);
    kv.insert(
        format!("{}:p", done.trim_end_matches(":d")),
        json!({ "fp": fp }).to_string(),
    );
    drop(kv);
    let before = w.completions().len();
    let r = with_key(&w, "slow", json!({}));
    assert_eq!(r.status, 409);
    assert_eq!(r.header("retry-after"), Some("2"));
    assert_eq!(
        r.body_json(),
        json!({ "error": "request_in_progress", "retry": true })
    );
    assert_eq!(w.completions().len(), before);
}

#[test]
fn a_failed_attempt_is_not_an_answer_the_retry_runs() {
    let w = World::new();
    w.settings().gateway_fails = true;
    w.settings().fallback_fails = true;
    w.settings().core = Some(std::rc::Rc::new(
        |b: &Value| json!({ "chunks": [], "hit": false, "scope": b["scope"] }),
    ));
    assert_eq!(with_key(&w, "boom", json!({})).status, 503);
    assert_eq!(with_key(&w, "boom", json!({})).status, 503);
    assert!(w.completions().len() >= 2);
    assert!(w.idem_keys().is_empty(), "the marker is released");
}

#[test]
fn a_non_200_answer_is_not_stored_either() {
    let w = World::new();
    w.settings().core_status = Some(503);
    let r = with_key(&w, "g", json!({ "mode": "extractive" }));
    assert_eq!(r.status, 503);
    assert!(w.idem_keys().is_empty());
}

#[test]
fn a_malformed_key_is_rejected_before_anything_is_called() {
    let w = World::new();
    for bad in ["has space", &"x".repeat(65), "café", ""] {
        let r = with_key(&w, bad, json!({}));
        assert_eq!(r.status, 400, "{bad:?}");
        assert_eq!(
            r.body_json(),
            json!({ "error": "invalid_idempotency_key", "pattern": "^[A-Za-z0-9_.:-]{1,64}$" })
        );
    }
    assert_eq!(w.core().len() + w.completions().len(), 0);
}

#[test]
fn rule1_stores_nothing_for_a_personal_scope_request_which_is_refused_first() {
    let w = World::new();
    let r = with_key(&w, "p1", json!({ "scope": "personal" }));
    assert_eq!(r.status, 409);
    assert!(w.idem_keys().is_empty());
    assert_eq!(w.touched(), 0);
}

#[test]
fn stores_only_the_response_for_ten_minutes_under_a_hashed_key() {
    let w = World::new();
    with_key(&w, "my-secret-looking-key", json!({}));
    let stored: Vec<(String, Option<u64>)> =
        w.p.kv_puts
            .borrow()
            .iter()
            .filter(|(k, _)| k.starts_with("idem:"))
            .cloned()
            .collect();
    assert_eq!(
        stored.iter().map(|(_, t)| *t).collect::<Vec<_>>(),
        [Some(PENDING_TTL_SECONDS), Some(DONE_TTL_SECONDS)]
    );
    assert!(stored
        .iter()
        .all(|(k, _)| !k.contains("my-secret-looking-key")));
    assert!(stored.iter().all(|(k, _)| k.starts_with("idem:e1:")));
    // The answer goes to its own key, so a fast answer never rewrites the marker key.
    assert_ne!(stored[0].0, stored[1].0);
}

#[test]
fn a_retry_of_an_answered_request_is_replayed_even_when_the_quota_is_now_spent() {
    let w = World::new();
    assert_eq!(with_key(&w, "paid", json!({})).status, 200);
    w.settings().rollup_total = 1e9;
    let retry = with_key(&w, "paid", json!({}));
    assert_eq!(retry.status, 200);
    assert_eq!(retry.header("idempotent-replayed"), Some("true"));
    // A genuinely new request is still refused, and its marker is released.
    assert_eq!(with_key(&w, "new-one", json!({})).status, 429);
    w.settings().rollup_total = 0.0;
    assert_eq!(with_key(&w, "new-one", json!({})).status, 200);
    assert_eq!(w.completions().len(), 2);
}

#[test]
fn a_degraded_answer_is_not_stored_so_the_retry_can_get_a_proper_one() {
    let w = World::new();
    w.settings().gateway_fails = true;
    w.settings().fallback_fails = true;
    let first = with_key(&w, "degrade", json!({}));
    assert_eq!(first.status, 200);
    assert_eq!(first.body_json()["shamwari"]["degraded"], true);
    assert!(!w.idem_keys().iter().any(|k| k.ends_with(":d")));
    let before = w.core().len();
    with_key(&w, "degrade", json!({}));
    assert!(w.core().len() > before);
}

#[test]
fn a_degraded_compact_answer_is_not_stored_either() {
    let w = World::new();
    w.settings().gateway_fails = true;
    w.settings().fallback_fails = true;
    let first = with_key(&w, "dc", json!({ "format": "compact" }));
    assert_eq!(first.body_json()["degraded"], true);
    assert!(!w.idem_keys().iter().any(|k| k.ends_with(":d")));
}

#[test]
fn a_kv_outage_runs_the_request_unprotected_rather_than_failing_it() {
    let w = World::new();
    w.p.idem_kv_down.set(true);
    let r = with_key(&w, "outage", json!({}));
    assert_eq!(r.status, 200);
    assert_eq!(w.completions().len(), 1);
}

#[test]
fn a_replay_of_a_compact_answer_is_compact() {
    let w = World::new();
    let first = with_key(&w, "c1", json!({ "format": "compact" }));
    assert!(!first.body.contains('\n'));
    let again = with_key(&w, "c1", json!({ "format": "compact" }));
    assert_eq!(again.header("idempotent-replayed"), Some("true"));
    assert_eq!(again.body, first.body);
}
