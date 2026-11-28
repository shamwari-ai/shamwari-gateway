//! Port of `shamwari/gateway/src/envelope.ts`: the compact response envelope
//! (docs/tiered-access-design.md), what a narrow channel needs and nothing
//! else. The default response stays OpenAI-shaped; a request opts in with
//! `"format": "compact"`.
//!
//! ```text
//! { id, answer, cites: [{ n, label, url? }], scope, via, truncated, degraded, notice? }
//! ```
//!
//! `notice` is present only when the answer was given without the person's
//! own information (shamwari-ai/shamwari#81); a client renders it as one line.
//! `via` is the rung that served it (gateway, workers-ai, extractive). Usage,
//! model names, latency and language are left out: they cost bytes a 2G link
//! pays for. On r0 the labels are made plain, so an SMS adapter does not meet
//! a dash that forces UCS-2.

use serde_json::{json, Map, Value};

pub const FORMATS: [&str; 1] = ["compact"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Compact,
}

/// `isFormat`: a string naming a known format.
pub fn is_format(v: &Value) -> Option<Format> {
    match v.as_str() {
        Some("compact") => Some(Format::Compact),
        _ => None,
    }
}

fn truthy_str(v: Option<&Value>) -> Option<&str> {
    v.and_then(Value::as_str).filter(|s| !s.is_empty())
}

/// The verbose chat body as the compact envelope. `plain` is the extractive
/// rung's typographic fold (`extractive::plain`, agent B's), given on r0 so
/// the labels stay in one SMS encoding; `None` keeps them as they are.
pub fn to_compact(v: &Value, plain: Option<&dyn Fn(&str) -> String>) -> Value {
    let label = |t: String| match plain {
        Some(f) => f(&t),
        None => t,
    };
    let s = v.get("shamwari").cloned().unwrap_or(Value::Null);
    let cites: Vec<Value> = s
        .get("citations")
        .and_then(Value::as_array)
        .map(|cs| {
            cs.iter()
                .enumerate()
                .map(|(i, c)| {
                    let parts: Vec<&str> = [c.get("title"), c.get("heading")]
                        .into_iter()
                        .filter_map(truthy_str)
                        .collect();
                    let mut o = Map::new();
                    o.insert("n".into(), json!(i + 1));
                    o.insert("label".into(), label(parts.join(" - ")).into());
                    if truthy_str(c.get("url")).is_some() {
                        o.insert("url".into(), c["url"].clone());
                    }
                    Value::Object(o)
                })
                .collect()
        })
        .unwrap_or_default();

    let mut out = Map::new();
    if let Some(id) = v.get("id") {
        out.insert("id".into(), id.clone());
    }
    let answer = v
        .pointer("/choices/0/message/content")
        .filter(|a| !a.is_null())
        .cloned()
        .unwrap_or_else(|| json!(""));
    out.insert("answer".into(), answer);
    out.insert("cites".into(), Value::Array(cites));
    if let Some(scope) = s.get("scope") {
        out.insert("scope".into(), scope.clone());
    }
    if let Some(via) = s.get("inference_path") {
        out.insert("via".into(), via.clone());
    }
    out.insert(
        "truncated".into(),
        json!(s.get("truncated") == Some(&json!(true))),
    );
    out.insert(
        "degraded".into(),
        json!(s.get("degraded") == Some(&json!(true))),
    );
    // Only when there is something to say: bytes cost on a 2G link.
    if let Some(notice) = s.get("notice").filter(|n| n.is_object()) {
        if let Some(m) = notice.get("message") {
            out.insert("notice".into(), m.clone());
        }
    }
    Value::Object(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn verbose() -> Value {
        json!({
            "id": "req-1",
            "model": "shamwari-economy",
            "choices": [{ "index": 0, "message": { "role": "assistant", "content": "The right to life." } }],
            "usage": { "prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3 },
            "shamwari": {
                "tier": "economy", "scope": "platform", "inference_path": "gateway",
                "truncated": false, "notice": null, "latency_ms": 12,
                "citations": [
                    { "title": "Constitution \u{2014} Zimbabwe", "heading": "Section 48", "authority": "Parliament", "effective_from": null, "url": "https://example.invalid/c" },
                    { "title": "Finance Act", "heading": null, "authority": "Parliament", "effective_from": null, "url": null },
                ],
            },
        })
    }

    #[test]
    fn keeps_only_what_a_narrow_channel_needs() {
        assert_eq!(
            to_compact(&verbose(), None),
            json!({
                "id": "req-1",
                "answer": "The right to life.",
                "cites": [
                    { "n": 1, "label": "Constitution \u{2014} Zimbabwe - Section 48", "url": "https://example.invalid/c" },
                    { "n": 2, "label": "Finance Act" },
                ],
                "scope": "platform",
                "via": "gateway",
                "truncated": false,
                "degraded": false,
            })
        );
    }

    #[test]
    fn labels_are_folded_when_asked_and_the_notice_rides_only_when_there_is_one() {
        let mut v = verbose();
        v["shamwari"]["notice"] =
            json!({ "code": "c", "message": "Answered without your own information." });
        v["shamwari"]["degraded"] = json!(true);
        let fold = |s: &str| s.replace('\u{2014}', "-");
        let c = to_compact(&v, Some(&fold));
        assert_eq!(
            c["cites"][0]["label"],
            "Constitution - Zimbabwe - Section 48"
        );
        assert_eq!(c["notice"], "Answered without your own information.");
        assert_eq!(c["degraded"], true);
        assert!(to_compact(&verbose(), None).get("notice").is_none());
    }

    #[test]
    fn only_compact_is_a_format() {
        assert_eq!(is_format(&json!("compact")), Some(Format::Compact));
        for bad in [json!("verbose"), json!("Compact"), json!(1), json!(null)] {
            assert_eq!(is_format(&bad), None);
        }
    }
}
