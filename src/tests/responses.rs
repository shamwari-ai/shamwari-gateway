//! `POST /v1/responses` through the real chat path, re-dispatched in-process:
//! `test/responses.test.ts`. Every rule of the chat path applies to it
//! unchanged, so most cases send the same question both ways and compare
//! what each world saw.

use super::world::*;
use crate::personal::OWN_INFORMATION_CODE;
use serde_json::{json, Value};

const ANSWER: &str = "Section 48 protects the right to life.";

fn world() -> World {
    let w = World::new();
    w.settings().answer = ANSWER.into();
    w.settings().usage = json!({ "prompt_tokens": 120, "completion_tokens": 9 });
    w
}

fn responses(w: &World, body: Value, headers: &[(&str, &str)]) -> crate::http::Resp {
    w.post_with("/v1/responses", &body, headers)
}

/// What a world saw: embeddings, Core bodies, completions, sink.
fn seen(w: &World) -> Value {
    json!({
        "embed": w.ai(),
        "core": w.core(),
        "completions": w.completions().into_iter().map(|(_, b)| b["messages"].clone()).collect::<Vec<_>>(),
        "sink": w.collections(),
    })
}

struct Both {
    chat: crate::http::Resp,
    resp: crate::http::Resp,
    chat_w: World,
    resp_w: World,
}

/// The same question on both endpoints, each in its own world.
fn both(q: &str, setup: impl Fn(&World), extra: Value, headers: &[(&str, &str)]) -> Both {
    let chat_w = world();
    setup(&chat_w);
    let chat = chat_w.chat_with(q, extra.clone(), headers);
    let resp_w = world();
    setup(&resp_w);
    let mut body = json!({ "input": q });
    if let Value::Object(m) = extra {
        for (k, v) in m {
            body[k] = v;
        }
    }
    let resp = responses(&resp_w, body, headers);
    Both {
        chat,
        resp,
        chat_w,
        resp_w,
    }
}

#[test]
fn a_string_input_comes_back_as_a_completed_response() {
    let w = world();
    let r = responses(
        &w,
        json!({ "model": "shamwari", "input": PLAIN_Q, "metadata": { "ticket": "A-1" } }),
        &[],
    );
    assert_eq!(r.status, 200);
    assert_eq!(r.header("content-type"), Some("application/json"));
    assert!(!r.body.contains('\n'), "JSON.stringify(out), unindented");
    let j = r.body_json();
    assert_eq!(j["object"], "response");
    assert!(j["id"].as_str().unwrap().starts_with("resp_"));
    assert_eq!(j["status"], "completed");
    assert_eq!(j["output"][0]["content"][0]["text"], ANSWER);
    assert_eq!(
        j["usage"],
        json!({
            "input_tokens": 120, "output_tokens": 9, "total_tokens": 129,
            "input_tokens_details": { "cached_tokens": 0 },
            "output_tokens_details": { "reasoning_tokens": 0 },
        })
    );
    assert_eq!(j["metadata"], json!({ "ticket": "A-1" }));
    // The question reached the model once, as the user's turn, after Ground's system prompt.
    let c = w.completions();
    assert_eq!(c.len(), 1);
    let msgs = c[0].1["messages"].as_array().unwrap();
    assert_eq!(msgs[0]["role"], "system");
    assert_eq!(
        msgs.last().unwrap(),
        &json!({ "role": "user", "content": PLAIN_Q })
    );
    assert_eq!(j["shamwari"]["citations"][0]["heading"], "Section 48");
}

#[test]
fn is_metered_and_recorded_exactly_as_the_same_request_on_chat() {
    let b = both(PLAIN_Q, |_| {}, json!({}), &[]);
    assert_eq!(b.chat.status, 200);
    assert_eq!(b.resp.status, 200);
    assert_eq!(seen(&b.resp_w), seen(&b.chat_w));
    let usage = b.resp_w.doc("usageEvents").unwrap();
    assert_eq!(usage["ownerEntityId"], "e1");
    assert_eq!(usage["apiKeyId"], "k1");
    assert_eq!(usage["inputTokens"], 120);
    assert_eq!(usage["outputTokens"], 9);
    assert!(usage.get("costUsdEstimate").is_some());
}

#[test]
fn an_idempotent_retry_is_the_same_response_run_once() {
    let w = world();
    let h = [("Idempotency-Key", "retry-1")];
    let first = responses(&w, json!({ "input": PLAIN_Q }), &h);
    let again = responses(&w, json!({ "input": PLAIN_Q }), &h);
    assert_eq!(again.header("idempotent-replayed"), Some("true"));
    assert_eq!(first.header("idempotent-replayed"), None);
    assert_eq!(again.body_json()["output"], first.body_json()["output"]);
    assert_eq!(w.completions().len(), 1);
    assert_eq!(w.core().len(), 1);
}

#[test]
fn rule1_an_explicit_personal_scope_is_the_same_409_and_nothing_is_called() {
    for q in [PLAIN_Q, PERSONAL_Q] {
        for mode in [None, Some("extractive")] {
            let mut extra = json!({ "scope": "personal" });
            if let Some(m) = mode {
                extra["mode"] = json!(m);
            }
            let b = both(
                q,
                |w| w.scopes(&["personal", "community", "platform"]),
                extra,
                &[],
            );
            assert_eq!(b.chat.status, 409);
            assert_eq!(b.resp.status, 409);
            let err = &b.resp.body_json()["error"];
            assert_eq!(err["type"], "invalid_request");
            assert_eq!(err["code"], "scope_requires_local_inference");
            assert_eq!(err["param"], "scope");
            assert_eq!(err["message"], b.chat.body_json()["message"]);
            assert_eq!(b.resp_w.touched(), 0);
            assert_eq!(seen(&b.resp_w), seen(&b.chat_w));
        }
    }
}

#[test]
fn rule1_a_personal_looking_question_reaches_no_model_exactly_as_on_chat() {
    let b = both(PERSONAL_Q, |_| {}, json!({}), &[]);
    assert_eq!(b.resp.status, 200);
    assert_eq!(b.resp_w.ai(), 0);
    assert_eq!(b.resp_w.completions().len(), 0);
    assert_eq!(
        b.resp_w
            .core()
            .iter()
            .map(|c| c["scope"].clone())
            .collect::<Vec<_>>(),
        [json!("platform")]
    );
    assert!(b.resp_w.doc("conversations").is_none());
    assert_eq!(seen(&b.resp_w), seen(&b.chat_w));
    let r = b.resp.body_json();
    let c = b.chat.body_json();
    assert_eq!(r["model"], "shamwari-extractive");
    assert_eq!(
        r["output"][0]["content"][0]["text"],
        c["choices"][0]["message"]["content"]
    );
    assert_eq!(r["shamwari"]["inference_path"], "extractive");
    assert_eq!(r["shamwari"]["answered_without_own_information"], true);
    assert_eq!(r["shamwari"]["notice"]["code"], OWN_INFORMATION_CODE);
}

#[test]
fn rule1_personal_text_in_an_input_text_part_is_caught() {
    let w = world();
    let r = responses(
        &w,
        json!({ "input": [{ "type": "message", "role": "user", "content": [{ "type": "input_text", "text": PERSONAL_Q }] }] }),
        &[],
    );
    assert_eq!(r.status, 200);
    assert_eq!(w.provider_calls(), 0);
}

#[test]
fn a_key_with_no_cloud_safe_scope_is_refused_as_on_chat() {
    let b = both(PLAIN_Q, |w| w.scopes(&["personal"]), json!({}), &[]);
    assert_eq!(b.chat.status, 403);
    assert_eq!(b.resp.status, 403);
    assert_eq!(
        b.resp.body_json()["error"]["code"],
        "scope_not_permitted_for_key"
    );
    assert_eq!(b.resp_w.completions().len(), 0);
}

#[test]
fn rule2_provenance_and_consent_are_stamped_as_on_chat() {
    let b = both(PLAIN_Q, |_| {}, json!({}), &[]);
    let (r, c) = (
        b.resp_w.doc("conversations").unwrap(),
        b.chat_w.doc("conversations").unwrap(),
    );
    assert_eq!(r["licenseClass"], c["licenseClass"]);
    assert_eq!(r["teacherModel"], c["teacherModel"]);
    assert_eq!(r["trainingConsent"], true);
    assert_eq!(r["trainingEligible"], c["trainingEligible"]);

    for consent in [None, Some(false)] {
        let b = both(
            PLAIN_Q,
            |w| w.auth_with(|a| a.training_consent = consent),
            json!({}),
            &[],
        );
        let r = b.resp_w.doc("conversations").unwrap();
        assert_eq!(r["trainingConsent"], false);
        assert_eq!(r["trainingEligible"], false);
        assert_eq!(
            r["trainingEligible"],
            b.chat_w.doc("conversations").unwrap()["trainingEligible"]
        );
    }
}

#[test]
fn rule2_the_channel_marker_is_forwarded() {
    let sms = both(PLAIN_Q, |_| {}, json!({}), &[("X-Shamwari-Channel", "sms")]);
    assert_eq!(
        sms.resp_w.doc("conversations").unwrap()["trainingEligible"],
        false
    );
    assert_eq!(seen(&sms.resp_w), seen(&sms.chat_w));
    let pwa = both(PLAIN_Q, |_| {}, json!({}), &[("X-Shamwari-Channel", "pwa")]);
    assert!(pwa.resp_w.doc("conversations").is_none());
    assert_eq!(seen(&pwa.resp_w), seen(&pwa.chat_w));
}

#[test]
fn no_key_is_a_401_in_the_spec_error_shape_and_nothing_is_called() {
    let w = world();
    let r = responses(
        &w,
        json!({ "input": PLAIN_Q, "stream": true }),
        &[("authorization", "")],
    );
    assert_eq!(r.status, 401);
    let e = &r.body_json()["error"];
    assert_eq!(e["type"], "invalid_request");
    assert_eq!(e["code"], "invalid_api_key");
    assert_eq!(e["param"], Value::Null);
    assert_eq!(w.touched(), 0);
}

#[test]
fn a_spent_quota_is_a_429_too_many_requests_and_nothing_is_generated() {
    let w = world();
    w.settings().rollup_total = 1e9;
    let r = responses(&w, json!({ "input": PLAIN_Q }), &[]);
    assert_eq!(r.status, 429);
    let e = &r.body_json()["error"];
    assert_eq!(e["type"], "too_many_requests");
    assert_eq!(e["code"], "quota_exceeded");
    assert_eq!(w.completions().len(), 0);
}

#[test]
fn only_post_is_served() {
    let w = world();
    let r = w.send(crate::http::Req {
        method: "GET".into(),
        url: "https://gw.example/v1/responses".into(),
        ..Default::default()
    });
    assert_eq!(r.status, 404);
}

#[test]
fn a_chat_path_validation_error_keeps_its_status_in_the_spec_shape() {
    let w = world();
    let r = responses(&w, json!({ "input": PLAIN_Q, "reach": "r9" }), &[]);
    assert_eq!(r.status, 400);
    assert_eq!(r.body_json()["error"]["type"], "invalid_request");
}

#[test]
fn every_inference_path_failing_with_nothing_from_ground_is_a_retryable_model_error() {
    let w = world();
    w.settings().gateway_fails = true;
    w.settings().fallback_fails = true;
    w.settings().core = Some(std::rc::Rc::new(
        |b: &Value| json!({ "chunks": [], "hit": false, "scope": b["scope"] }),
    ));
    let r = responses(&w, json!({ "input": PLAIN_Q }), &[]);
    assert_eq!(r.status, 503);
    assert_eq!(r.body_json()["error"]["type"], "model_error");
    assert_eq!(r.header("retry-after"), Some("15"));
}
