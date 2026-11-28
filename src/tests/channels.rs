//! The SMS and USSD webhooks through the real chat path, re-dispatched
//! in-process: the worker-level cases of `test/channels.test.ts`.
//!
//! The sender (a phone number) is personal data: it is in no outbound call,
//! no stored document and no log line. Channel traffic is never training
//! material (Rule 2).

use super::world::*;
use crate::channels::messages;
use crate::channels::sms::SMS_MAX_CHARS;
use crate::channels::ussd::USSD_MAX_CHARS;
use crate::http::Req;
use hmac::{Hmac, KeyInit, Mac};
use serde_json::{json, Value};
use sha2::Sha256;

const PHONE: &str = "+263771234567";

fn sign(secret: &str, raw: &str) -> String {
    let mut mac = <Hmac<Sha256> as KeyInit>::new_from_slice(secret.as_bytes()).unwrap();
    mac.update(raw.as_bytes());
    mac.finalize()
        .into_bytes()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// The fake operator: it signs what it sends, as the real one would.
fn operator(w: &World, path: &str, body: &Value, signature: Option<&str>) -> crate::http::Resp {
    let raw = body.to_string();
    let sig = signature.map_or_else(|| sign(operator_secret(), &raw), str::to_owned);
    w.send(Req {
        method: "POST".into(),
        url: format!("https://gw.example{path}"),
        headers: vec![
            ("content-type".into(), "application/json".into()),
            ("x-signature".into(), sig),
        ],
        body: raw,
    })
}

fn text(id: &str) -> Value {
    json!({ "id": id, "from": PHONE, "text": LIFE_Q })
}

fn world(answer: &str) -> World {
    let w = World::new();
    w.settings().answer = answer.into();
    w
}

fn reply(r: &crate::http::Resp) -> String {
    r.body_json()["reply"]
        .as_str()
        .unwrap_or_default()
        .to_owned()
}

#[test]
fn sms_answers_a_signed_message_with_the_answer_and_its_source_inside_three_segments() {
    let w = world("Section 48 protects the right to life.");
    let r = operator(&w, "/v1/channels/sms", &text("m1"), None);
    assert_eq!(r.status, 200);
    assert!(!r.body.contains('\n') || r.body.starts_with("{\"reply\""));
    let reply = reply(&r);
    assert!(reply.contains("Section 48 protects the right to life."));
    assert!(reply.contains("Source: Constitution of Zimbabwe - Section 48"));
    assert!(crate::lang::js::len(&reply) <= SMS_MAX_CHARS);
    assert_eq!(w.completions().len(), 1);
    // The chat request was the channel's: r0, compact, platform, with the marker.
    let c = &w.core()[0];
    assert_eq!(c["scope"], "platform");
}

#[test]
fn the_reply_is_plain_ascii_so_one_curly_quote_cannot_force_ucs2() {
    let w = world("It\u{2019}s \u{201C}protected\u{201D} \u{2013} always\u{2026}");
    w.settings().core = Some(std::rc::Rc::new(|b: &Value| {
        let mut c = chunk(b["scope"].as_str().unwrap());
        c["title"] = json!("Constitution \u{2013} Zimbabwe");
        json!({ "chunks": [c], "hit": true, "scope": b["scope"] })
    }));
    let r = operator(&w, "/v1/channels/sms", &text("m2"), None);
    assert!(reply(&r).is_ascii(), "{}", reply(&r));
}

#[test]
fn a_long_answer_is_cut_inside_the_limit_and_keeps_its_source_line() {
    let w = world(&"word ".repeat(400));
    let r = operator(&w, "/v1/channels/sms", &text("m3"), None);
    let reply = reply(&r);
    assert!(crate::lang::js::len(&reply) <= SMS_MAX_CHARS);
    assert!(reply.contains("Source: "));
}

#[test]
fn an_operator_retry_of_the_same_message_is_the_same_answer_run_once() {
    let w = world("Section 48 protects the right to life.");
    let a = operator(&w, "/v1/channels/sms", &text("dup"), None);
    let b = operator(&w, "/v1/channels/sms", &text("dup"), None);
    assert_eq!(a.body_json(), b.body_json());
    assert_eq!(w.completions().len(), 1);
    assert_eq!(w.core().len(), 1);
}

#[test]
fn the_sender_is_in_no_outbound_call_stored_document_or_log_line() {
    let w = world("Section 48 protects the right to life.");
    operator(&w, "/v1/channels/sms", &text("m4"), None);
    assert_eq!(w.completions().len(), 1);
    let everything = format!(
        "{:?} {} {:?}",
        w.requests(),
        serde_json::to_string(&w.sink()).unwrap(),
        crate::failure::console::take()
    );
    assert!(!everything.contains("771234567"));
}

#[test]
fn is_metered_like_any_caller_a_spent_quota_is_a_short_message() {
    let w = world("x");
    w.settings().rollup_total = 1e9;
    let r = operator(&w, "/v1/channels/sms", &text("m5"), None);
    assert_eq!(r.status, 200);
    assert_eq!(reply(&r), messages::QUOTA);
    assert_eq!(w.completions().len(), 0);
}

#[test]
fn a_bad_or_missing_signature_is_a_401_and_nothing_is_called() {
    let w = world("x");
    let wrong = sign(
        &format!("{}-not", operator_secret()),
        &text("s1").to_string(),
    );
    for sig in [wrong.as_str(), "", "abc"] {
        let r = operator(&w, "/v1/channels/sms", &text("s1"), Some(sig));
        assert_eq!(r.status, 401);
        assert_eq!(r.body, r#"{"error":"invalid_signature"}"#);
    }
    assert_eq!(w.requests().len(), 0);
}

#[test]
fn an_unconfigured_channel_answers_503_and_nothing_else() {
    let mut cfg = env();
    cfg.channel_webhook_secret = None;
    let w = World::with(cfg);
    let r = operator(&w, "/v1/channels/ussd", &text("u"), None);
    assert_eq!(r.status, 503);
    assert_eq!(r.body_json(), json!({ "error": "channel_not_configured" }));
    assert_eq!(w.requests().len(), 0);
}

#[test]
fn ussd_an_empty_first_request_opens_the_session_and_calls_nothing() {
    let w = world("x");
    let r = operator(
        &w,
        "/v1/channels/ussd",
        &json!({ "sessionId": "s1", "msisdn": PHONE, "input": "" }),
        None,
    );
    assert_eq!(
        r.body_json(),
        json!({ "message": "Shamwari. Type your question:", "end": false })
    );
    assert_eq!(w.touched(), 0);
}

#[test]
fn ussd_answers_inside_one_screen_and_closes_the_session() {
    let w = world(&"x ".repeat(300));
    let r = operator(
        &w,
        "/v1/channels/ussd",
        &json!({ "sessionId": "s2", "msisdn": PHONE, "input": LIFE_Q }),
        None,
    );
    let j = r.body_json();
    assert_eq!(j["end"], true);
    let m = j["message"].as_str().unwrap();
    assert!(crate::lang::js::len(m) <= USSD_MAX_CHARS);
    assert!(m.contains("Source: "));
}

#[test]
fn rule1_a_personal_looking_question_on_sms_reaches_no_model() {
    let w = world("x");
    let r = operator(
        &w,
        "/v1/channels/sms",
        &json!({ "id": "p1", "from": PHONE, "text": PERSONAL_Q }),
        None,
    );
    assert_eq!(r.status, 200);
    assert_eq!(w.provider_calls(), 0);
    assert!(reply(&r).contains("Answered without your own info"));
}

#[test]
fn rule2_channel_traffic_is_never_stored_as_training_material() {
    // The control: the same question through the API is trainable, so the
    // channel result is the header's doing, not an artefact of the harness.
    let direct = world("Section 48 protects the right to life.");
    direct.chat(LIFE_Q, json!({}));
    assert_eq!(
        direct.doc("conversations").unwrap()["trainingEligible"],
        true
    );
    let via = world("Section 48 protects the right to life.");
    operator(&via, "/v1/channels/sms", &text("tr1"), None);
    assert_eq!(via.doc("conversations").unwrap()["trainingEligible"], false);
}
