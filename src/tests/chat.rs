//! `POST /v1/chat/completions` through the real handler: the worker-level
//! cases of `test/scope-decision.test.ts`, `test/extractive-handler.test.ts`
//! and `test/reach.test.ts`, and the bytes of the response.
//!
//! Rule 1 is stated as counts: every provider-bound call (an embedding, a
//! completion, the Workers AI fallback) is counted with the scope it was made
//! under. A personal path to a provider is a leak even if the response is an
//! error, so each case says how many calls it made.

use super::world::*;
use crate::model_router::{policy, Reach, DEFAULT_MAX_OUTPUT_TOKENS};
use crate::personal::{OWN_INFORMATION_CODE, OWN_INFORMATION_NOTICE};
use serde_json::{json, Value};

const ESCALATING: &str = "Please calculate the total rainfall for three fields step by step.";

/// `{...extra}` with only the fields that are set: `undefined` is not sent.
fn fields(pairs: &[(&str, Option<Value>)]) -> Value {
    let mut m = serde_json::Map::new();
    for (k, v) in pairs {
        if let Some(v) = v {
            m.insert((*k).to_owned(), v.clone());
        }
    }
    Value::Object(m)
}

// ---------------------------------------------------------------------------
// (c) the client cannot force a scope

#[test]
fn a_client_sent_community_reaches_core_and_the_provider_as_platform() {
    let w = World::new();
    let r = w.chat(PLAIN_Q, json!({ "scope": "community" }));
    assert_eq!(r.status, 200);
    assert_eq!(w.scopes_seen(), ["platform", "platform"]);
    assert_eq!(r.body_json()["shamwari"]["scope"], "platform");
}

#[test]
fn the_scope_follows_the_keys_grants_not_the_request() {
    let w = World::new();
    w.scopes(&["community"]);
    let r = w.chat(PLAIN_Q, json!({ "scope": "platform" }));
    assert_eq!(r.status, 200);
    assert_eq!(r.body_json()["shamwari"]["scope"], "community");
    assert_eq!(w.scopes_seen(), ["community", "community"]);
}

// ---------------------------------------------------------------------------
// (a) no personal path to a provider

#[test]
fn rule1_an_explicit_personal_request_is_a_409_and_nothing_is_called() {
    for mind in [false, true] {
        for q in [PLAIN_Q, PERSONAL_Q] {
            for mode in [None, Some(json!("extractive"))] {
                let w = if mind { World::mind() } else { World::new() };
                w.scopes(&["personal", "community", "platform"]);
                let r = w.chat(
                    q,
                    fields(&[("scope", Some(json!("personal"))), ("mode", mode.clone())]),
                );
                assert_eq!(r.status, 409, "{mind} {q} {mode:?}");
                let b = r.body_json();
                assert_eq!(b["error"], "scope_requires_local_inference");
                assert_eq!(b["scope"], "personal");
                assert_eq!(b["capability"], "cloud_inference");
                assert_eq!(w.touched(), 0, "{mind} {q} {mode:?}");
            }
        }
    }
}

#[test]
fn rule1_a_key_that_does_not_hold_personal_gets_the_same_409_not_a_403() {
    let w = World::new(); // default scopes: community, platform
    let r = w.chat(PERSONAL_Q, json!({ "scope": "personal" }));
    assert_eq!(r.status, 409);
    assert_eq!(r.body_json()["error"], "scope_requires_local_inference");
    assert_eq!(w.touched(), 0);
}

#[test]
fn rule1_counts_a_personal_looking_question_makes_no_embedding_and_no_completion() {
    let w = World::new(); // Cloud-only mode: MIND_AVAILABLE=false
    let r = w.chat(PERSONAL_Q, json!({}));
    assert_eq!(r.status, 200);
    assert_eq!(w.ai(), 0);
    assert_eq!(w.completions().len(), 0);
    assert_eq!(w.core().len(), 1);
    assert_eq!(r.body_json()["shamwari"]["inference_path"], "extractive");
    assert!(w.unsafe_scopes().is_empty());
    for m in w.sink() {
        assert_eq!(m["doc"]["scope"], "platform");
    }
}

#[test]
fn rule1_holds_for_every_combination_of_grants_mind_mode_and_client_scope() {
    let grants: [Option<&[&str]>; 4] = [
        None,
        Some(&["personal", "community", "platform"]),
        Some(&["community"]),
        Some(&["platform"]),
    ];
    for g in grants {
        for mind in [false, true] {
            for mode in [None, Some(json!("extractive"))] {
                for scope in [None, Some("community"), Some("platform"), Some("bogus")] {
                    for q in [PLAIN_Q, PERSONAL_Q] {
                        let w = if mind { World::mind() } else { World::new() };
                        if let Some(g) = g {
                            w.scopes(g);
                        }
                        w.chat(
                            q,
                            fields(&[("scope", scope.map(|s| json!(s))), ("mode", mode.clone())]),
                        );
                        assert!(
                            w.unsafe_scopes().is_empty(),
                            "{g:?} {mind} {mode:?} {scope:?}"
                        );
                        if mode.is_some() {
                            assert_eq!(w.completions().len(), 0);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn rule1_still_refuses_personal_scope_on_every_reach_with_nothing_called() {
    for reach in ["r0", "r1", "r2"] {
        let w = World::new();
        let r = w.chat(LIFE_Q, json!({ "reach": reach, "scope": "personal" }));
        assert_eq!(r.status, 409);
        assert_eq!(r.body_json()["error"], "scope_requires_local_inference");
        assert_eq!(w.touched(), 0);
    }
}

#[test]
fn rule1_personal_scope_is_refused_on_the_degraded_path_as_well() {
    let w = World::new();
    w.settings().gateway_fails = true;
    w.settings().fallback_fails = true;
    let r = w.chat(LIFE_Q, json!({ "scope": "personal" }));
    assert_eq!(r.status, 409);
    assert_eq!(w.touched(), 0);
}

#[test]
fn rule1_holding_personal_never_lets_personal_reach_cloud() {
    let w = World::new();
    w.scopes(&["personal", "community", "platform"]);
    let r = w.chat(LIFE_Q, json!({ "scope": "personal" }));
    assert_eq!(r.status, 409);
    assert_eq!(w.touched(), 0);
}

#[test]
fn a_key_that_holds_neither_community_nor_platform_is_refused_whatever_is_asked() {
    let w = World::new();
    w.scopes(&["personal"]);
    let r = w.chat(LIFE_Q, json!({ "scope": "community" }));
    assert_eq!(r.status, 403);
    let b = r.body_json();
    assert_eq!(b["error"], "scope_not_permitted_for_key");
    assert_eq!(b["scope"], "platform");
    assert_eq!(b["permitted_scopes"], json!(["personal"]));
    assert_eq!(b["hint"], "this key holds no community or platform scope");
    assert_eq!(w.touched(), 0);
}

#[test]
fn the_default_allows_community_and_platform() {
    let w = World::new();
    assert_eq!(w.chat(LIFE_Q, json!({ "scope": "platform" })).status, 200);
}

// ---------------------------------------------------------------------------
// (b) a personal-looking question is never answered without the marker

#[test]
fn cloud_only_mode_says_it_answered_without_the_persons_own_information() {
    let w = World::new();
    let j = w.chat(PERSONAL_Q, json!({})).body_json();
    assert_eq!(j["shamwari"]["answered_without_own_information"], true);
    assert_eq!(
        j["shamwari"]["notice"],
        json!({ "code": OWN_INFORMATION_CODE, "message": OWN_INFORMATION_NOTICE })
    );
    let msg = j["shamwari"]["notice"]["message"].as_str().unwrap();
    assert!(msg.contains("not sent to an AI model"));
    assert!(!msg.contains("Mind"));
    assert_eq!(j["shamwari"]["inference_path"], "extractive");
    assert_eq!(j["shamwari"]["scope"], "platform");
}

#[test]
fn every_shared_personal_looking_case_carries_the_marker_through_the_worker() {
    for (text, needs) in crate::personal::tests::scope_cases() {
        if !needs {
            continue;
        }
        let w = World::new();
        let r = w.chat(&text, json!({}));
        assert_eq!(r.status, 200, "{text}");
        assert_eq!(
            r.body_json()["shamwari"]["answered_without_own_information"],
            true,
            "{text}"
        );
        assert_eq!(w.provider_calls(), 0, "{text}");
    }
}

#[test]
fn a_plain_question_carries_no_notice_but_the_field_is_present() {
    let w = World::new();
    let j = w.chat(PLAIN_Q, json!({})).body_json();
    assert_eq!(j["shamwari"]["answered_without_own_information"], false);
    assert_eq!(j["shamwari"]["notice"], Value::Null);
}

#[test]
fn the_extractive_rung_carries_the_marker_too_and_calls_no_model() {
    let w = World::new();
    let j = w
        .chat(PERSONAL_Q, json!({ "mode": "extractive" }))
        .body_json();
    assert_eq!(j["shamwari"]["answered_without_own_information"], true);
    assert_eq!(w.completions().len(), 0);
}

#[test]
fn a_personal_looking_question_never_reaches_the_degraded_path() {
    let w = World::new();
    w.settings().gateway_fails = true;
    let j = w.chat(PERSONAL_Q, json!({})).body_json();
    assert_eq!(j["shamwari"]["degraded"], false);
    assert_eq!(j["shamwari"]["answered_without_own_information"], true);
    assert_eq!(w.provider_calls(), 0);
}

#[test]
fn a_follow_up_straight_after_a_personal_turn_is_still_flagged_and_kept_from_the_model() {
    let w = World::new();
    let r = w.post(
        "/v1/chat/completions",
        &json!({ "messages": [
            { "role": "user", "content": "I earn $400 a month" },
            { "role": "assistant", "content": "Ok." },
            { "role": "user", "content": "and how much tax is that?" },
        ] }),
    );
    assert_eq!(
        r.body_json()["shamwari"]["answered_without_own_information"],
        true
    );
    assert_eq!(w.provider_calls(), 0);
}

#[test]
fn the_compact_envelope_carries_the_notice_as_one_string_only_when_true() {
    let w = World::new();
    let yes = w
        .chat(PERSONAL_Q, json!({ "format": "compact" }))
        .body_json();
    assert_eq!(yes["notice"], OWN_INFORMATION_NOTICE);
    let w = World::new();
    let no = w.chat(PLAIN_Q, json!({ "format": "compact" })).body_json();
    assert!(no.get("notice").is_none());
}

// ---------------------------------------------------------------------------
// (d) Core and the gateway cannot disagree about a scope decision

#[test]
fn core_is_sent_the_grants_and_the_gateways_own_view() {
    let w = World::new();
    w.scopes(&["community", "platform"]);
    w.chat(PERSONAL_Q, json!({}));
    let c = &w.core()[0];
    assert_eq!(c["scope"], "platform");
    assert_eq!(c["destination"], "cloud");
    assert_eq!(c["granted_scopes"], json!(["community", "platform"]));
    assert_eq!(c["needs_own_information"], true);
    assert!(c.get("embedding").is_none());
}

#[test]
fn core_answering_for_another_scope_is_not_used_and_nothing_is_cited() {
    let answers: Vec<Value> = vec![
        json!({ "chunks": [{ "scope": "personal", "content": "PAYSLIP" }], "hit": true, "scope": "personal" }),
        json!({ "chunks": [chunk("community")], "hit": true, "scope": "community" }),
        json!({ "chunks": [{ "scope": "personal", "content": "PAYSLIP" }], "hit": true, "scope": "platform" }),
        json!({ "chunks": [chunk("platform")], "hit": true }),
    ];
    for a in answers {
        let w = World::new();
        let fixed = a.clone();
        w.settings().core = Some(std::rc::Rc::new(move |_| fixed.clone()));
        let r = w.chat(PLAIN_Q, json!({}));
        assert_eq!(r.status, 200);
        let j = r.body_json();
        assert_eq!(j["shamwari"]["grounded"], false, "{a}");
        assert_eq!(j["shamwari"]["citations"], json!([]));
        assert!(!r.body.contains("PAYSLIP"));
        assert!(w.unsafe_scopes().is_empty());
    }
}

fn core_says_needs_own_information(w: &World, needs: bool) {
    w.settings().core = Some(std::rc::Rc::new(
        move |b: &Value| json!({ "chunks": [chunk(b["scope"].as_str().unwrap())], "hit": true, "scope": b["scope"], "needs_own_information": needs }),
    ));
}

#[test]
fn core_saying_yes_on_its_own_adds_the_notice_and_no_completion_is_made() {
    let w = World::new();
    core_says_needs_own_information(&w, true);
    let j = w.chat(PLAIN_Q, json!({})).body_json();
    assert_eq!(j["shamwari"]["answered_without_own_information"], true);
    assert_eq!(j["shamwari"]["inference_path"], "extractive");
    assert_eq!(w.completions().len(), 0);
}

#[test]
fn core_saying_false_cannot_remove_a_notice_the_gateway_raised() {
    let w = World::new();
    core_says_needs_own_information(&w, false);
    let j = w.chat(PERSONAL_Q, json!({})).body_json();
    assert_eq!(j["shamwari"]["answered_without_own_information"], true);
}

#[test]
fn core_refusing_the_scope_leaves_the_answer_ungrounded_never_retried() {
    for status in [403, 409] {
        let w = World::new();
        w.settings().core_status = Some(status);
        let r = w.chat(PLAIN_Q, json!({}));
        assert_eq!(r.status, 200);
        assert_eq!(w.core().len(), 1);
        assert_eq!(r.body_json()["shamwari"]["grounded"], false);
        assert!(w.unsafe_scopes().is_empty());
    }
}

#[test]
fn a_personal_looking_question_with_ground_unreachable_is_a_retryable_503() {
    for status in [403, 409] {
        let w = World::new();
        w.settings().core_status = Some(status);
        let r = w.chat(PERSONAL_Q, json!({}));
        assert_eq!(r.status, 503);
        assert_eq!(r.header("retry-after"), Some("30"));
        assert_eq!(
            r.body_json(),
            json!({ "error": "ground_unavailable", "retry": true })
        );
        assert_eq!(w.core().len(), 1);
        assert_eq!(w.provider_calls(), 0);
    }
}

// ---------------------------------------------------------------------------
// Skills and the own-information flag

fn skill() -> Value {
    json!({ "name": "short-answers", "description": "Keep answers short for a small screen.", "instructions": "Answer in three sentences or fewer." })
}

#[test]
fn own_information_true_takes_the_personal_path_with_no_provider_call() {
    let w = World::new();
    let r = w.chat(PLAIN_Q, json!({ "own_information": true }));
    assert_eq!(r.status, 200);
    let j = r.body_json();
    assert_eq!(w.provider_calls(), 0);
    assert_eq!(w.core().len(), 1);
    assert!(w.core()[0].get("embedding").is_none());
    assert_eq!(w.core()[0]["needs_own_information"], true);
    assert_eq!(j["shamwari"]["inference_path"], "extractive");
    assert_eq!(j["shamwari"]["answered_without_own_information"], true);
    assert!(!j["shamwari"]["citations"].as_array().unwrap().is_empty());
    assert!(w.unsafe_scopes().is_empty());
}

#[test]
fn own_information_false_does_not_clear_a_question_the_rules_flag() {
    let w = World::new();
    let j = w
        .chat(PERSONAL_Q, json!({ "own_information": false }))
        .body_json();
    assert_eq!(w.provider_calls(), 0);
    assert_eq!(j["shamwari"]["inference_path"], "extractive");
}

#[test]
fn an_explicit_personal_scope_is_still_the_409_with_the_flag_and_skills() {
    let w = World::new();
    w.scopes(&["personal", "community", "platform"]);
    let r = w.chat(
        PLAIN_Q,
        json!({ "scope": "personal", "own_information": true, "skills": [skill()] }),
    );
    assert_eq!(r.status, 409);
    assert_eq!(w.touched(), 0);
}

#[test]
fn the_flag_holds_for_every_mode_and_key_grant() {
    let grants: [Option<&[&str]>; 4] = [
        None,
        Some(&["platform"]),
        Some(&["community"]),
        Some(&["personal", "community", "platform"]),
    ];
    for g in grants {
        for mode in [None, Some(json!("extractive"))] {
            let w = World::new();
            if let Some(g) = g {
                w.scopes(g);
            }
            w.chat(
                PLAIN_Q,
                fields(&[
                    ("own_information", Some(json!(true))),
                    ("mode", mode.clone()),
                ]),
            );
            assert_eq!(w.provider_calls(), 0, "{g:?} {mode:?}");
        }
    }
}

#[test]
fn the_flag_with_ground_unreachable_is_the_retryable_503() {
    let w = World::new();
    w.settings().core_status = Some(503);
    let r = w.chat(PLAIN_Q, json!({ "own_information": true }));
    assert_eq!(r.status, 503);
    assert_eq!(w.provider_calls(), 0);
}

#[test]
fn a_platform_skill_goes_after_the_constant_prompt() {
    let w = World::new();
    let r = w.chat(PLAIN_Q, json!({ "skills": [skill()] }));
    assert_eq!(r.status, 200);
    assert_eq!(w.completions().len(), 1);
    let system = w.system_prompt(0);
    assert!(system.starts_with("You are Shamwari"));
    assert!(system.contains("--- SOURCES ---"));
    assert!(system.contains("--- SKILLS TURNED ON FOR THIS CONVERSATION ---"));
    assert!(system.contains(
        "[short-answers] Keep answers short for a small screen.\nAnswer in three sentences or fewer."
    ));
    assert!(system.find("SKILLS TURNED ON").unwrap() > system.find("END SOURCES").unwrap());
}

#[test]
fn without_skills_the_prompt_is_exactly_what_it_was() {
    let a = World::new();
    a.chat(PLAIN_Q, json!({}));
    let b = World::new();
    b.chat(PLAIN_Q, json!({ "skills": [] }));
    assert_eq!(a.system_prompt(0), b.system_prompt(0));
    assert!(!a.system_prompt(0).contains("SKILLS"));
}

#[test]
fn a_skill_whose_own_words_look_personal_makes_the_request_personal_scope() {
    let mut s1 = skill();
    s1["instructions"] = json!("Use my farm size of 4 hectares.");
    let mut s2 = skill();
    s2["description"] = json!("Answers about our co-operative.");
    let mut s3 = skill();
    s3["name"] = json!("wangu");
    for s in [s1, s2, s3] {
        let w = World::new();
        let j = w
            .chat(PLAIN_Q, json!({ "skills": [s.clone()] }))
            .body_json();
        assert_eq!(w.provider_calls(), 0, "{s}");
        assert_eq!(j["shamwari"]["inference_path"], "extractive");
        assert_eq!(w.core()[0]["needs_own_information"], true);
    }
}

#[test]
fn malformed_skills_or_a_non_boolean_flag_are_a_400_before_anything_is_called() {
    let s = skill();
    let with = |k: &str, v: Value| {
        let mut x = s.clone();
        x[k] = v;
        x
    };
    let bad = [
        json!({ "skills": "short-answers" }),
        json!({ "skills": [{ "name": "a" }] }),
        json!({ "skills": [with("name", json!(""))] }),
        json!({ "skills": [with("name", json!("a".repeat(65)))] }),
        json!({ "skills": [with("description", json!("d".repeat(1025)))] }),
        json!({ "skills": [with("instructions", json!("i".repeat(12_001)))] }),
        json!({ "skills": [s.clone(), s.clone(), s.clone(), s.clone()] }),
        json!({ "own_information": "yes" }),
        json!({ "own_information": null }),
    ];
    for extra in bad {
        let w = World::new();
        let r = w.chat(PLAIN_Q, extra.clone());
        assert_eq!(r.status, 400, "{extra}");
        assert_eq!(w.provider_calls() + w.core().len(), 0);
    }
    let w = World::new();
    let b = w.chat(PLAIN_Q, json!({ "skills": "x" })).body_json();
    assert_eq!(
        b,
        json!({ "error": "invalid_skills", "max": 3, "hint": "skills is a list of {name, description, instructions}" })
    );
    let b = w.chat(PLAIN_Q, json!({ "own_information": 1 })).body_json();
    assert_eq!(
        b,
        json!({ "error": "invalid_own_information", "allowed": [true, false] })
    );
}

// ---------------------------------------------------------------------------
// The extractive rung (extractive-handler.test.ts)

#[test]
fn mode_extractive_answers_with_the_cited_passage_and_makes_no_inference_call() {
    let w = World::new();
    let r = w.chat(LIFE_Q, json!({ "mode": "extractive" }));
    assert_eq!(r.status, 200);
    let j = r.body_json();
    let text = j["choices"][0]["message"]["content"].as_str().unwrap();
    assert!(text.contains("Every person has the right to life."));
    assert!(text.contains("Closest source: Constitution of Zimbabwe"));
    assert_eq!(j["model"], "shamwari-extractive");
    assert_eq!(j["shamwari"]["inference_path"], "extractive");
    assert_eq!(j["shamwari"]["degraded"], false);
    assert_eq!(j["shamwari"]["citations"].as_array().unwrap().len(), 1);
    assert_eq!(j["usage"]["total_tokens"], 0);
    assert_eq!(w.completions().len(), 0);
}

#[test]
fn rule2_extractive_is_metered_and_never_stored_as_a_conversation() {
    let w = World::new();
    w.chat(LIFE_Q, json!({ "mode": "extractive" }));
    assert_eq!(w.collections(), ["usageEvents"]);
    let doc = w.doc("usageEvents").unwrap();
    assert_eq!(doc["scope"], "platform");
    assert_eq!(doc["inferencePath"], "extractive");
    assert_eq!(doc["inputTokens"], 0);
    assert_eq!(doc["outputTokens"], 0);
    assert_eq!(doc["degraded"], false);
}

#[test]
fn extractive_says_plainly_it_found_nothing_when_ground_is_empty() {
    let w = World::new();
    w.settings().core = Some(std::rc::Rc::new(
        |b: &Value| json!({ "chunks": [], "hit": false, "scope": b["scope"] }),
    ));
    let j = w.chat(LIFE_Q, json!({ "mode": "extractive" })).body_json();
    assert_eq!(j["shamwari"]["grounded"], false);
    assert_eq!(j["shamwari"]["citations"], json!([]));
    assert!(j["choices"][0]["message"]["content"]
        .as_str()
        .unwrap()
        .contains("will not guess"));
}

#[test]
fn an_unknown_mode_is_rejected_before_anything_is_called() {
    for bad in [json!("turbo"), json!(null), json!(1)] {
        let w = World::new();
        let r = w.chat(LIFE_Q, json!({ "mode": bad }));
        assert_eq!(r.status, 400);
        assert_eq!(
            r.body_json(),
            json!({ "error": "invalid_mode", "allowed": ["extractive"] })
        );
        assert_eq!(w.provider_calls() + w.core().len(), 0);
    }
}

#[test]
fn every_inference_path_failing_answers_from_ground_and_says_it_degraded() {
    let w = World::new();
    w.settings().gateway_fails = true;
    w.settings().fallback_fails = true;
    let r = w.chat(LIFE_Q, json!({}));
    assert_eq!(r.status, 200);
    let j = r.body_json();
    assert_eq!(j["shamwari"]["inference_path"], "extractive");
    assert_eq!(j["shamwari"]["degraded"], true);
    assert!(j["choices"][0]["message"]["content"]
        .as_str()
        .unwrap()
        .contains("Closest source: Constitution of Zimbabwe"));
    // Metered as degraded, never stored as a conversation.
    assert_eq!(w.collections(), ["usageEvents"]);
    let doc = w.doc("usageEvents").unwrap();
    assert_eq!(doc["inferencePath"], "extractive");
    assert_eq!(doc["degraded"], true);
}

#[test]
fn without_a_ground_hit_the_failure_stands_as_a_503_that_says_why() {
    let w = World::new();
    w.settings().gateway_fails = true;
    w.settings().fallback_fails = true;
    w.settings().core = Some(std::rc::Rc::new(
        |b: &Value| json!({ "chunks": [], "hit": false, "scope": b["scope"] }),
    ));
    let r = w.chat(LIFE_Q, json!({}));
    assert_eq!(r.status, 503);
    assert_eq!(r.header("retry-after"), Some("15"));
    let b = r.body_json();
    assert_eq!(b["error"], "inference_unavailable");
    assert_eq!(b["retry"], true);
    let id = b["request_id"].as_str().unwrap();
    assert_eq!(id.len(), 36);
    assert!(id.chars().all(|c| c.is_ascii_hexdigit() || c == '-'));
    assert_eq!(
        b["reasons"],
        json!([
            { "step": "ai-gateway", "status": 503, "message": "down" },
            { "step": "workers-ai", "message": "workers ai down" },
        ])
    );
    // Logged with the request id; the answer text and the question are not in it.
    let logs = crate::failure::console::take().join("\n");
    assert!(logs.contains("inference unavailable and no Ground passage to give"));
    assert!(logs.contains(id));
    assert!(!logs.contains(LIFE_Q));
}

#[test]
fn a_normal_answer_still_goes_to_inference_and_stores_the_conversation() {
    let w = World::new();
    let j = w.chat(LIFE_Q, json!({})).body_json();
    assert_eq!(j["shamwari"]["inference_path"], "gateway");
    assert_eq!(w.completions().len(), 1);
    let mut c = w.collections();
    c.sort();
    assert_eq!(c, ["conversations", "usageEvents"]);
    let usage = w.doc("usageEvents").unwrap();
    assert_eq!(usage["scope"], "platform");
    assert!(usage["priceTableVersion"]
        .as_str()
        .is_some_and(|s| !s.is_empty()));
    assert!(usage["costUsdEstimate"].is_null() || usage["costUsdEstimate"].is_number());
}

#[test]
fn the_web_chat_writes_no_conversation_but_still_writes_the_usage_event() {
    let w = World::new();
    let r = w.chat_with(LIFE_Q, json!({}), &[("X-Shamwari-Channel", "PWA")]);
    assert_eq!(r.status, 200);
    assert_eq!(w.collections(), ["usageEvents"]);
    let usage = w.doc("usageEvents").unwrap();
    assert_eq!(usage["scope"], "platform");
    assert_eq!(usage["inferencePath"], "gateway");
    assert!(usage["requestId"].is_string());
    let stored = serde_json::to_string(&w.sink()).unwrap();
    assert!(!stored.contains(LIFE_Q));
    assert!(!stored.contains("generated"));
    assert!(!crate::failure::console::take()
        .iter()
        .any(|l| l.contains("sink enqueue failed")));
}

#[test]
fn rule2_sms_traffic_is_stored_but_never_trainable() {
    let w = World::new();
    w.chat_with(LIFE_Q, json!({}), &[("X-Shamwari-Channel", "sms")]);
    let conv = w.doc("conversations").expect("stored");
    assert_eq!(conv["trainingEligible"], false);
}

#[test]
fn rule1_the_web_chat_is_still_refused_personal_scope() {
    let w = World::new();
    w.scopes(&["personal", "community", "platform"]);
    let r = w.chat_with(
        LIFE_Q,
        json!({ "scope": "personal" }),
        &[("X-Shamwari-Channel", "pwa")],
    );
    assert_eq!(r.status, 409);
    assert_eq!(w.touched(), 0);
}

#[test]
fn still_answers_when_the_embedding_model_is_down_by_keyword_search() {
    let w = World::new();
    w.settings().embed_fails = true;
    let r = w.chat(LIFE_Q, json!({ "mode": "extractive" }));
    assert_eq!(r.status, 200);
    assert_eq!(r.body_json()["shamwari"]["grounded"], true);
    assert_eq!(w.core().len(), 1);
}

#[test]
fn does_not_say_the_corpus_lacks_something_when_core_could_not_be_asked() {
    let w = World::new();
    w.settings().core_status = Some(503);
    let r = w.chat(LIFE_Q, json!({ "mode": "extractive" }));
    assert_eq!(r.status, 503);
    assert_eq!(r.header("retry-after"), Some("30"));
    let b = r.body_json();
    assert_eq!(b["error"], "ground_unavailable");
    assert_eq!(b["retry"], true);
    assert_eq!(
        b["reasons"],
        json!([{ "step": "core", "message": "Core could not be reached, so the cited passage could not be fetched" }])
    );
}

#[test]
fn core_down_the_model_still_answers_ungrounded_and_core_is_tried_once() {
    let w = World::new();
    w.settings().core_status = Some(503);
    for _ in 0..3 {
        let r = w.chat(LIFE_Q, json!({}));
        assert_eq!(r.status, 200);
        let j = r.body_json();
        assert_eq!(j["choices"][0]["message"]["content"], "generated");
        assert_eq!(j["shamwari"]["grounded"], false);
        assert_eq!(j["shamwari"]["citations"], json!([]));
    }
    // The first question found Core down; the breaker skipped it for the next two.
    assert_eq!(w.core().len(), 1);
    assert_eq!(w.completions().len(), 3);
}

#[test]
fn rule1_holds_with_the_breaker_open_the_flag_gets_no_model_only_a_503() {
    let w = World::new();
    w.settings().core_status = Some(503);
    w.chat(LIFE_Q, json!({})); // opens the breaker
    let before = w.completions().len();
    let r = w.chat(LIFE_Q, json!({ "own_information": true }));
    assert_eq!(r.status, 503);
    assert_eq!(r.body_json()["error"], "ground_unavailable");
    assert_eq!(w.completions().len(), before);
}

#[test]
fn the_key_path_shares_cores_breaker_a_key_check_fails_at_once_while_core_is_down() {
    let w = World::new();
    w.settings().core_status = Some(503);
    w.chat(LIFE_Q, json!({})); // opens the breaker
    let before = w.requests().len();
    // An uncached key: verified through Core, which is skipped while the breaker is open.
    let other = format!("{}x", api_key());
    let r = w.post_with(
        "/v1/chat/completions",
        &ask(LIFE_Q, json!({})),
        &[("authorization", &format!("Bearer {other}"))],
    );
    assert_eq!(r.status, 503);
    assert_eq!(r.body_json(), json!({ "error": "auth_unavailable" }));
    assert_eq!(r.header("retry-after"), Some("5"));
    assert_eq!(w.requests().len(), before, "Core was not called");
}

// ---------------------------------------------------------------------------
// Rule 2 on the stored conversation

#[test]
fn rule2_a_conversation_the_system_judged_personal_is_never_stored() {
    let w = World::new();
    w.chat(PERSONAL_Q, json!({}));
    assert!(w.doc("conversations").is_none());

    let plain = World::new();
    plain.chat(PLAIN_Q, json!({}));
    let conv = plain.doc("conversations").unwrap();
    assert_eq!(conv["trainingEligible"], true);
    assert_eq!(conv["trainingConsent"], true);
    assert_eq!(conv["licenseClass"], "open_weight");
}

#[test]
fn rule2_a_caller_without_recorded_consent_is_stored_but_not_trainable() {
    for consent in [None, Some(false)] {
        let w = World::new();
        w.auth_with(|a| a.training_consent = consent);
        w.chat(PLAIN_Q, json!({}));
        let conv = w.doc("conversations").unwrap();
        assert_eq!(conv["trainingEligible"], false, "{consent:?}");
        assert_eq!(conv["trainingConsent"], false);
    }
}

#[test]
fn rule2_a_workos_sign_in_is_never_trainable_even_with_consent() {
    let w = World::new();
    w.auth_with(|a| a.key_type = "workos".into());
    w.chat(PLAIN_Q, json!({}));
    assert_eq!(w.doc("conversations").unwrap()["trainingEligible"], false);
}

#[test]
fn rule2_provenance_is_the_served_model_restricted_when_substituted() {
    let w = World::new();
    w.settings().served = Some(("someone".into(), "unlisted-model-v1".into()));
    let j = w.chat(PLAIN_Q, json!({})).body_json();
    assert_eq!(j["shamwari"]["model_substituted"], true);
    assert_eq!(j["shamwari"]["model_served"], "someone/unlisted-model-v1");
    let conv = w.doc("conversations").unwrap();
    assert_eq!(conv["licenseClass"], "restricted");
    assert_eq!(conv["trainingEligible"], false);
    assert_eq!(conv["teacherModel"], "someone/unlisted-model-v1");
    assert_eq!(conv["modelSubstituted"], true);

    let w = World::new();
    w.settings().served = Some(("workers-ai".into(), "@cf/zai-org/glm-5.3".into()));
    let j = w.chat(PLAIN_Q, json!({})).body_json();
    assert_eq!(
        j["shamwari"]["model_served"],
        "workers-ai/@cf/zai-org/glm-5.3"
    );
    assert_eq!(
        w.doc("conversations").unwrap()["licenseClass"],
        "open_weight"
    );
}

#[test]
fn metering_carries_no_content() {
    let w = World::new();
    w.chat(PLAIN_Q, json!({}));
    let usage = w.doc("usageEvents").unwrap().to_string();
    assert!(!usage.contains("section 48"));
    assert!(!usage.contains("generated"));
}

// ---------------------------------------------------------------------------
// Reach (reach.test.ts, through the handler)

fn max_tokens_sent(w: &World) -> Value {
    w.completions()[0].1["max_tokens"].clone()
}

#[test]
fn a_request_with_no_reach_behaves_exactly_as_before() {
    let w = World::new();
    let j = w.chat(LIFE_Q, json!({})).body_json();
    assert_eq!(max_tokens_sent(&w), json!(DEFAULT_MAX_OUTPUT_TOKENS));
    assert_eq!(j["shamwari"]["reach"], Value::Null);
    assert_eq!(j["shamwari"]["reach_clamped"], false);
}

#[test]
fn r0_clamps_an_escalating_question_to_economy_and_caps_the_output() {
    let w = World::new();
    let j = w.chat(ESCALATING, json!({ "reach": "r0" })).body_json();
    assert_eq!(w.completions()[0].0["tier"], "economy");
    assert_eq!(
        max_tokens_sent(&w),
        json!(policy().reach(Reach::R0).max_output_tokens)
    );
    assert_eq!(j["shamwari"]["tier"], "economy");
    assert_eq!(j["shamwari"]["reach"], "r0");
    assert_eq!(j["shamwari"]["reach_clamped"], true);
    let usage = w.doc("usageEvents").unwrap();
    assert_eq!(usage["tier"], "economy");
    assert_eq!(usage["routedTier"], "standard");
    assert_eq!(usage["reachClamped"], true);
}

#[test]
fn r1_lets_the_same_question_reach_standard_with_its_own_cap() {
    let w = World::new();
    let j = w.chat(ESCALATING, json!({ "reach": "r1" })).body_json();
    assert_eq!(w.completions()[0].0["tier"], "standard");
    assert_eq!(
        max_tokens_sent(&w),
        json!(policy().reach(Reach::R1).max_output_tokens)
    );
    assert_eq!(j["shamwari"]["reach_clamped"], false);
    assert_eq!(j["model"], "shamwari-standard");
}

#[test]
fn r2_has_the_largest_cap_and_the_reach_is_recorded() {
    let w = World::new();
    w.chat(LIFE_Q, json!({ "reach": "r2" }));
    assert_eq!(
        max_tokens_sent(&w),
        json!(policy().reach(Reach::R2).max_output_tokens)
    );
    let w = World::new();
    w.chat(LIFE_Q, json!({ "reach": "r1" }));
    let usage = w.doc("usageEvents").unwrap();
    assert_eq!(usage["reach"], "r1");
    assert_eq!(usage["reachClamped"], false);
    assert_eq!(w.doc("conversations").unwrap()["reach"], "r1");
}

#[test]
fn an_unknown_reach_is_rejected_before_anything_is_called() {
    let w = World::new();
    for bad in [
        json!("r3"),
        json!("R0"),
        json!("sms"),
        json!(5),
        json!(null),
    ] {
        let r = w.chat(LIFE_Q, json!({ "reach": bad }));
        assert_eq!(r.status, 400);
        assert_eq!(
            r.body_json(),
            json!({ "error": "invalid_reach", "allowed": ["r0", "r1", "r2"] })
        );
    }
    assert_eq!(w.provider_calls() + w.core().len(), 0);
}

#[test]
fn an_unknown_format_is_rejected_before_anything_is_called() {
    let w = World::new();
    let r = w.chat(LIFE_Q, json!({ "format": "tiny" }));
    assert_eq!(r.status, 400);
    assert_eq!(
        r.body_json(),
        json!({ "error": "invalid_format", "allowed": ["compact"] })
    );
    assert_eq!(w.provider_calls() + w.core().len(), 0);
}

#[test]
fn an_extractive_answer_is_capped_by_reach() {
    let small = World::new();
    let a = small
        .chat(LIFE_Q, json!({ "reach": "r0", "mode": "extractive" }))
        .body_json();
    let big = World::new();
    big.settings().core = Some(std::rc::Rc::new(|b: &Value| {
        let mut c = chunk(b["scope"].as_str().unwrap());
        c["content"] = json!("Every person has the right to life. ".repeat(40));
        json!({ "chunks": [c], "hit": true, "scope": b["scope"] })
    }));
    small.settings().core = big.settings().core.clone();
    let a2 = small
        .chat(LIFE_Q, json!({ "reach": "r0", "mode": "extractive" }))
        .body_json();
    let b = big
        .chat(LIFE_Q, json!({ "reach": "r2", "mode": "extractive" }))
        .body_json();
    let len =
        |v: &Value| crate::lang::js::len(v["choices"][0]["message"]["content"].as_str().unwrap());
    assert!(len(&a2) <= policy().reach(Reach::R0).extractive_max_chars);
    assert!(len(&b) > len(&a2));
    assert!(len(&b) <= policy().reach(Reach::R2).extractive_max_chars);
    assert_eq!(a["shamwari"]["reach"], "r0");
    assert_eq!(small.completions().len(), 0);
}

#[test]
fn the_caller_is_told_when_the_output_cap_cut_the_answer_off() {
    let w = World::new();
    w.settings().finish_reason = Some("length".into());
    w.settings().answer = "cut off mid".into();
    let j = w.chat(LIFE_Q, json!({ "reach": "r0" })).body_json();
    assert_eq!(j["choices"][0]["finish_reason"], "length");
    assert_eq!(j["shamwari"]["truncated"], true);
}

#[test]
fn r0_extractive_text_is_plain_so_an_sms_stays_in_one_encoding() {
    let w = World::new();
    w.settings().core = Some(std::rc::Rc::new(|b: &Value| {
        let mut c = chunk(b["scope"].as_str().unwrap());
        c["title"] = json!("Constitution \u{2014} Zimbabwe");
        c["content"] = json!("It\u{2019}s \u{201C}the right\u{201D} \u{2013} and more\u{2026}");
        json!({ "chunks": [c], "hit": true, "scope": b["scope"] })
    }));
    let r0 = w
        .chat(LIFE_Q, json!({ "reach": "r0", "mode": "extractive" }))
        .body_json();
    let text = r0["choices"][0]["message"]["content"].as_str().unwrap();
    assert!(!text.contains([
        '\u{2013}', '\u{2014}', '\u{2018}', '\u{2019}', '\u{201C}', '\u{201D}', '\u{2026}'
    ]));
    assert!(text.contains("It's \"the right\" - and more..."));
    let r1 = w
        .chat(LIFE_Q, json!({ "reach": "r1", "mode": "extractive" }))
        .body_json();
    assert!(r1["choices"][0]["message"]["content"]
        .as_str()
        .unwrap()
        .contains('\u{2014}'));
}

// ---------------------------------------------------------------------------
// Admission: what is refused before anything runs

#[test]
fn a_missing_or_invalid_key_is_a_401_and_nothing_is_called() {
    let w = World::new();
    for auth in ["", "Bearer ", "Bearer nope", "Basic x"] {
        let r = w.post_with(
            "/v1/chat/completions",
            &ask(LIFE_Q, json!({})),
            &[("authorization", auth)],
        );
        assert_eq!(r.status, 401, "{auth}");
        assert_eq!(r.body_json(), json!({ "error": "invalid_api_key" }));
    }
    assert_eq!(w.touched(), 0);
}

#[test]
fn a_body_that_is_not_json_or_has_no_messages_is_a_400() {
    let w = World::new();
    let send = |body: &str| {
        w.send(crate::http::Req {
            method: "POST".into(),
            url: "https://gw.example/v1/chat/completions".into(),
            headers: vec![("authorization".into(), format!("Bearer {}", api_key()))],
            body: body.into(),
        })
    };
    assert_eq!(send("{").body_json(), json!({ "error": "invalid_json" }));
    for b in [
        r#"{}"#,
        r#"{"messages":[]}"#,
        r#"{"messages":null}"#,
        r#"[]"#,
        r#""x""#,
    ] {
        let r = send(b);
        assert_eq!(r.status, 400, "{b}");
        assert_eq!(
            r.body_json(),
            json!({ "error": "messages_required" }),
            "{b}"
        );
    }
    for b in [
        r#"{"messages":"hello"}"#,
        r#"{"messages":[{"role":"tool","content":"x"}]}"#,
        r#"{"messages":[{"role":"user"}]}"#,
    ] {
        let r = send(b);
        assert_eq!(r.status, 400, "{b}");
        assert_eq!(r.body_json()["error"], "invalid_messages", "{b}");
    }
    // `null` has no `.messages`: in TypeScript the exception reaches the 500.
    let r = send("null");
    assert_eq!(r.status, 500);
    assert_eq!(r.body_json()["error"], "internal_error");
    assert_eq!(w.touched(), 0);
}

#[test]
fn a_spent_quota_is_a_429_and_nothing_is_generated() {
    let w = World::new();
    w.settings().rollup_total = 1e9;
    let r = w.chat(LIFE_Q, json!({}));
    assert_eq!(r.status, 429);
    assert_eq!(
        r.body_json(),
        json!({ "error": "quota_exceeded", "tier": "community", "upgrade": "https://platform.shamwari.ai/billing" })
    );
    assert_eq!(w.touched(), 0);
    // The quota is checked after the body is read on the chat endpoint, so
    // a malformed body is still its own 400.
    let r = w.chat(LIFE_Q, json!({ "mode": "turbo" }));
    assert_eq!(r.status, 400);
}

#[test]
fn a_rollup_outage_fails_open_and_says_so() {
    let w = World::new();
    w.p.fake.responders.borrow_mut().insert(
        0,
        Box::new(|r| r.url.contains("/rollup").then(|| Err("down".to_owned()))),
    );
    let r = w.chat(LIFE_Q, json!({}));
    assert_eq!(r.status, 200);
}

// ---------------------------------------------------------------------------
// The bytes: TypeScript's json() is JSON.stringify(body, null, 2)

#[test]
fn the_extractive_answer_is_byte_for_byte_what_typescript_writes() {
    let w = World::new();
    let r = w.chat(LIFE_Q, json!({ "mode": "extractive" }));
    let id = r.body_json()["id"].as_str().unwrap().to_owned();
    let text = r.body_json()["choices"][0]["message"]["content"]
        .as_str()
        .unwrap()
        .to_owned();
    let want = format!(
        r#"{{
  "id": "{id}",
  "model": "shamwari-extractive",
  "choices": [
    {{
      "index": 0,
      "message": {{
        "role": "assistant",
        "content": {text}
      }},
      "finish_reason": "stop"
    }}
  ],
  "usage": {{
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
  }},
  "shamwari": {{
    "tier": "extractive",
    "scope": "platform",
    "language": null,
    "grounded": true,
    "answered_without_own_information": false,
    "notice": null,
    "cache_hit": false,
    "inference_path": "extractive",
    "reach": null,
    "degraded": false,
    "truncated": false,
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
        text = Value::String(text.clone())
    );
    assert_eq!(r.body, want);
    assert_eq!(r.header("content-type"), Some("application/json"));
}

#[test]
fn the_generated_answer_is_byte_for_byte_what_typescript_writes() {
    let w = World::new();
    w.settings().usage = json!({ "prompt_tokens": 120, "completion_tokens": 9 });
    w.settings().finish_reason = Some("stop".into());
    let r = w.chat(PLAIN_Q, json!({}));
    let j = r.body_json();
    let id = j["id"].as_str().unwrap();
    let served = j["shamwari"]["model_served"].as_str().unwrap();
    let want = format!(
        r#"{{
  "id": "{id}",
  "model": "shamwari-economy",
  "choices": [
    {{
      "index": 0,
      "message": {{
        "role": "assistant",
        "content": "generated"
      }},
      "finish_reason": "stop"
    }}
  ],
  "usage": {{
    "prompt_tokens": 120,
    "completion_tokens": 9,
    "total_tokens": 129
  }},
  "shamwari": {{
    "tier": "economy",
    "reach": null,
    "reach_clamped": false,
    "truncated": false,
    "scope": "platform",
    "language": "en",
    "grounded": true,
    "answered_without_own_information": false,
    "notice": null,
    "cache_hit": false,
    "inference_path": "gateway",
    "model_served": "{served}",
    "model_substituted": false,
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
}}"#
    );
    assert_eq!(r.body, want);
    assert_eq!(served, "workers-ai/@cf/zai-org/glm-5.3-flash");
}

#[test]
fn the_compact_envelope_is_unindented_and_r0_labels_are_plain_ascii() {
    let w = World::new();
    w.settings().core = Some(std::rc::Rc::new(|b: &Value| {
        let mut c = chunk(b["scope"].as_str().unwrap());
        c["title"] = json!("Constitution \u{2013} \u{201C}Zimbabwe\u{201D}");
        c["heading"] = json!("S\u{00A0}48");
        json!({ "chunks": [c], "hit": true, "scope": b["scope"] })
    }));
    let r = w.chat(LIFE_Q, json!({ "format": "compact", "reach": "r0" }));
    assert!(!r.body.contains('\n'));
    assert_eq!(
        r.body_json()["cites"][0]["label"],
        "Constitution - \"Zimbabwe\" - S 48"
    );
}

#[test]
fn the_compact_envelope_has_only_what_a_narrow_channel_needs() {
    let w = World::new();
    w.settings().answer = "ok".into();
    let j = w.chat(LIFE_Q, json!({ "format": "compact" })).body_json();
    let mut keys: Vec<&String> = j.as_object().unwrap().keys().collect();
    keys.sort();
    assert_eq!(
        keys,
        [
            "answer",
            "cites",
            "degraded",
            "id",
            "scope",
            "truncated",
            "via"
        ]
    );
    assert_eq!(j["answer"], "ok");
    assert_eq!(j["via"], "gateway");
    assert_eq!(
        j["cites"],
        json!([{ "n": 1, "label": "Constitution of Zimbabwe - Section 48", "url": "https://example.invalid/constitution" }])
    );

    let a = World::new();
    let verbose = a.chat(LIFE_Q, json!({})).body;
    let b = World::new();
    let compact = b.chat(LIFE_Q, json!({ "format": "compact" })).body;
    assert!(compact.len() < verbose.len());

    let c = World::new();
    let j = c
        .chat(LIFE_Q, json!({ "format": "compact", "mode": "extractive" }))
        .body_json();
    assert_eq!(j["via"], "extractive");
    assert_eq!(c.completions().len(), 0);

    let d = World::new();
    let r = d.chat(LIFE_Q, json!({ "format": "compact", "scope": "personal" }));
    assert_eq!(r.status, 409);
    assert_eq!(r.body_json()["error"], "scope_requires_local_inference");
}
