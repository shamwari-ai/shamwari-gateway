//! Port of `shamwari/gateway/src/channels/channel.ts`: the channel-adapter
//! interface (docs/tiered-access-design.md, "Channel adapters"). One adapter
//! per channel ([`sms`], [`ussd`]), one pipeline for all of them:
//!
//! ```text
//! verify the operator -> parse -> RULE 1 GATE -> the gateway's own chat path
//! ```
//!
//! An adapter is a client of the gateway, like Mind: it has no model access
//! and no Core access. It reaches Ground and inference only by calling
//! `/v1/chat/completions` in-process with the channel's own key, so every rule
//! that holds for an API caller (quota, scope, provenance, idempotency) holds
//! here without being re-implemented.
//!
//! RULE 1. The scope is a property of the adapter, fixed in code, never read
//! from the message: an SMS or USSD user cannot declare `personal`, and a
//! misconfigured adapter that does is refused here, before the chat path and
//! therefore before a byte reaches a provider. The gate runs again inside the
//! chat path; two checks because one is a single bug away from a leak.
//!
//! The sender (a phone number) is personal data and is never forwarded: it is
//! not in the chat request, the idempotency key, the stored conversation or a
//! log line.
//!
//! The routes are wired in [`crate::gateway::handle_sms`] and
//! [`crate::gateway::handle_ussd`], which pass the gateway's own chat path to
//! [`handle_channel`].

use crate::config::Config;
use crate::failure::console;
use crate::http::{Req, Resp};
use crate::lang::js;
use crate::model_router::Reach;
use crate::types::Scope;
use hmac::{Hmac, KeyInit, Mac};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::future::Future;
use unicode_normalization::UnicodeNormalization;

pub mod sms;
pub mod ussd;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelName {
    Sms,
    Ussd,
}

impl ChannelName {
    pub fn as_str(self) -> &'static str {
        match self {
            ChannelName::Sms => "sms",
            ChannelName::Ussd => "ussd",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Inbound {
    /// The operator's message or session id. Used only to derive an idempotency key.
    pub id: String,
    /// What the person typed.
    pub text: String,
    /// The sender. Held here for the operator's reply, never forwarded.
    pub sender: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reply {
    pub text: String,
    /// USSD: close the session.
    pub end: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct ChannelAdapter {
    pub channel: ChannelName,
    pub reach: Reach,
    /// What this channel can carry. Cloud-bound channels are platform.
    pub scope: Scope,
    /// The length the operator allows for one reply.
    pub max_chars: usize,
    pub parse: fn(&Value) -> Option<Inbound>,
    /// The first message of a session, before a question has been asked.
    pub greeting: Option<fn(&Inbound) -> Option<Reply>>,
    /// The operator's wire format for a reply.
    pub render: fn(&Reply) -> Value,
}

/// What [`compose`] needs from the extractive rung and the personal rules
/// (agent B's): `extractive::plain`, `extractive::clip(..).text`, and
/// `personal::OWN_INFORMATION_NOTICE_SHORT` / `_MIN`.
#[derive(Debug, Clone, Copy)]
pub struct ChannelText {
    pub plain: fn(&str) -> String,
    /// `clip(text, max, ellipsis).text`.
    pub clip: fn(&str, usize, &str) -> String,
    pub notice_short: &'static str,
    pub notice_min: &'static str,
}

pub const MAX_INPUT_CHARS: usize = 500;
const MAX_BODY_BYTES: usize = 8192;

/// What the person is told, in every case the chat path did not answer.
pub mod messages {
    pub const PERSONAL: &str =
        "Personal questions cannot be answered over this channel. Use the Shamwari app on your phone.";
    pub const RETRY: &str =
        "Shamwari is still working on your last message. Please wait a moment and send it again.";
    pub const BUSY: &str = "Shamwari cannot answer right now. Please try again later.";
    pub const QUOTA: &str = "This service has reached its limit for now. Please try again later.";
    pub const EMPTY: &str = "Send your question as text and Shamwari will answer with its source.";
}

/// The operator's signature: HMAC-SHA256 of the raw body, as 64 hex digits.
pub fn hmac_ok(secret: &str, body: &str, signature: Option<&str>) -> bool {
    let Some(sig) = signature else {
        return false;
    };
    if sig.len() != 64 || !sig.bytes().all(|b| b.is_ascii_hexdigit()) {
        return false;
    }
    let bytes: Vec<u8> = (0..32)
        .map(|i| u8::from_str_radix(&sig[i * 2..i * 2 + 2], 16).unwrap_or(0))
        .collect();
    let Ok(mut mac) = <Hmac<Sha256> as KeyInit>::new_from_slice(secret.as_bytes()) else {
        return false;
    };
    mac.update(body.as_bytes());
    // Constant time, as crypto.subtle.verify.
    mac.verify_slice(&bytes).is_ok()
}

/// `sha256Hex`: lowercase hex of SHA-256 of the UTF-8 text.
pub fn sha256_hex(text: &str) -> String {
    Sha256::digest(text.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Reduce text to what one GSM-7 septet can carry. `plain` folds the
/// typographic characters; this also strips accents, replaces the extension
/// characters that cost two septets each (`[ ] { } ^ ~ | \`), and drops what
/// has no plain form (emoji). One character outside the set would put the
/// whole SMS in UCS-2 at 70 characters a segment, so the 459 limit counts the
/// right thing.
pub fn gsm7(text: &str, plain: fn(&str) -> String) -> String {
    let s = plain(text).replace('\u{20AC}', "EUR");
    let s: String = s
        .nfd()
        .filter(|c| !('\u{0300}'..='\u{036f}').contains(c))
        .filter_map(|c| match c {
            '[' | '{' => Some('('),
            ']' | '}' => Some(')'),
            '~' | '^' => None,
            '|' | '\\' => Some('/'),
            '\n' | '\x20'..='\x7E' => Some(c),
            _ => None,
        })
        .collect();
    // / {2,}/g -> ' ', then /[ \t]+\n/g -> '\n' (no tab survives the filter above).
    let mut collapsed = String::with_capacity(s.len());
    for c in s.chars() {
        if c == ' ' && collapsed.ends_with(' ') {
            continue;
        }
        if c == '\n' {
            while collapsed.ends_with(' ') {
                collapsed.pop();
            }
        }
        collapsed.push(c);
    }
    collapsed
}

pub const NO_SOURCE_LINE: &str = "(No source found. Please check it.)";

/// The answer plus one source line, inside the channel's limit. An extractive
/// answer already carries its own hedged "Closest source" line, which stays
/// hedged: it is moved to the end, never rewritten as "Source". `notice` is
/// present when the answer was given without the person's own information.
pub fn compose(
    answer: &str,
    cite_labels: &[&str],
    max_chars: usize,
    via: Option<&str>,
    notice: Option<&str>,
    t: &ChannelText,
) -> String {
    let (body, source) = if via == Some("extractive") {
        let first: Vec<&str> = answer
            .split("\n\n")
            .next()
            .unwrap_or_default()
            .split('\n')
            .collect();
        let last = first.last().copied().unwrap_or_default();
        if last.starts_with("Closest source") {
            (first[..first.len() - 1].join("\n"), last.to_owned())
        } else {
            (first.join("\n"), String::new())
        }
    } else {
        let source = match cite_labels.iter().find(|l| !l.is_empty()) {
            Some(l) => format!("Source: {l}"),
            None => NO_SOURCE_LINE.to_owned(),
        };
        (answer.to_owned(), source)
    };
    let source = gsm7(&source, t.plain);
    let src = if source.is_empty() {
        String::new()
    } else {
        (t.clip)(&source, max_chars * 2 / 5, "...")
    };
    // The notice is never the part that gets clipped away: the body gives up
    // room for it. The full short form on SMS; on a screen of about 180
    // characters the marker alone, or the answer would be squeezed to a few words.
    let note = match notice.filter(|n| !n.is_empty()) {
        Some(_) => gsm7(
            if max_chars >= 300 {
                t.notice_short
            } else {
                t.notice_min
            },
            t.plain,
        ),
        None => String::new(),
    };
    let used = |s: &str| {
        if s.is_empty() {
            0
        } else {
            js::len(s) as i64 + 1
        }
    };
    let room = (max_chars as i64 - used(&src) - used(&note)).max(20) as usize;
    let text = (t.clip)(&gsm7(&body, t.plain), room, "...");
    [text, src, note]
        .into_iter()
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

/// `json()` in `channels/channel.ts`: `JSON.stringify(body)`, unindented.
fn json_resp(body: &Value, status: u16) -> Resp {
    Resp::json_compact(body, status)
}

/// `Number(header ?? 0)`: blank is 0, anything unparseable is NaN (never too large).
fn js_number(h: Option<&str>) -> f64 {
    match h.map(str::trim) {
        None | Some("") => 0.0,
        Some(s) => s.parse::<f64>().unwrap_or(f64::NAN),
    }
}

/// The internal chat request a channel makes: the channel's own key, the
/// adapter's scope and reach, the compact envelope, an idempotency key from
/// the operator's id and the text, and the channel marker. Nothing about the
/// sender.
pub fn chat_request(adapter: &ChannelAdapter, api_key: &str, inbound_id: &str, text: &str) -> Req {
    // The operator's id and the text, hashed: a retry of the same message is
    // the same key, and nothing about the sender is in it.
    let key = sha256_hex(&format!(
        "{}\n{inbound_id}\n{text}",
        adapter.channel.as_str()
    ));
    Req {
        method: "POST".into(),
        url: "https://gateway.internal/v1/chat/completions".into(),
        headers: vec![
            ("authorization".into(), format!("Bearer {api_key}")),
            ("content-type".into(), "application/json".into()),
            ("idempotency-key".into(), key),
            // Marks channel traffic: it is never stored as training material.
            ("x-shamwari-channel".into(), adapter.channel.as_str().into()),
        ],
        body: json!({
            "messages": [{ "role": "user", "content": text }],
            "scope": adapter.scope.as_str(),
            "reach": adapter.reach.as_str(),
            "format": "compact",
        })
        .to_string(),
    }
}

/// The whole pipeline for one inbound webhook. `chat` is the gateway's own
/// chat handler, called in-process with an ordinary API request; `Err` from
/// it is an exception that escaped the chat path.
pub async fn handle_channel<F, Fut>(
    adapter: &ChannelAdapter,
    req: &Req,
    cfg: &Config,
    text_deps: &ChannelText,
    chat: F,
) -> Resp
where
    F: FnOnce(Req) -> Fut,
    Fut: Future<Output = Result<Resp, String>>,
{
    // Fail closed: an unconfigured channel answers nothing.
    let (Some(secret), Some(api_key)) = (
        cfg.get(&cfg.channel_webhook_secret),
        cfg.get(&cfg.channel_api_key),
    ) else {
        return json_resp(&json!({ "error": "channel_not_configured" }), 503);
    };

    // Bounded before it is read: this route is public until the signature passes.
    if js_number(req.header("content-length")) > MAX_BODY_BYTES as f64 {
        return json_resp(&json!({ "error": "too_large" }), 413);
    }
    let raw = &req.body;
    if js::len(raw) > MAX_BODY_BYTES {
        return json_resp(&json!({ "error": "too_large" }), 413);
    }
    if !hmac_ok(secret, raw, req.header("x-signature")) {
        return json_resp(&json!({ "error": "invalid_signature" }), 401);
    }

    let Ok(parsed) = serde_json::from_str::<Value>(raw) else {
        return json_resp(&json!({ "error": "invalid_json" }), 400);
    };
    let Some(inbound) = (adapter.parse)(&parsed) else {
        return json_resp(&json!({ "error": "invalid_message" }), 400);
    };

    let reply = |text: &str| {
        json_resp(
            &(adapter.render)(&Reply {
                text: text.to_owned(),
                end: true,
            }),
            200,
        )
    };

    if let Some(greeting) = adapter.greeting.and_then(|g| g(&inbound)) {
        return json_resp(&(adapter.render)(&greeting), 200);
    }

    let text = js::prefix(js::trim(&inbound.text), MAX_INPUT_CHARS);
    if text.is_empty() {
        return reply(messages::EMPTY);
    }

    // RULE 1 GATE. Before the chat path, so before Ground, embedding or
    // inference. A cloud-bound endpoint refuses personal scope whatever
    // MIND_AVAILABLE says (`decideDestination(.., 'cloud_inference')`).
    if adapter.scope == Scope::Personal {
        return reply(messages::PERSONAL);
    }

    let channel = adapter.channel.as_str();
    let res = match chat(chat_request(adapter, api_key, &inbound.id, text)).await {
        Ok(r) => r,
        Err(e) => {
            // An outage on every inference path. The operator still gets an
            // answer in its own format, and the sender is not in the log line.
            console::error(&format!("channel chat path threw {channel} {e}"));
            return reply(messages::BUSY);
        }
    };

    if res.status != 200 {
        // Never silent: a rotated key or a dead path would otherwise look like
        // a polite "busy" to every user and a healthy webhook to the operator.
        let code = serde_json::from_str::<Value>(&res.body)
            .ok()
            .and_then(|b| b.get("error").filter(|e| !e.is_null()).cloned())
            .map(|e| match e {
                Value::String(s) => s,
                other => other.to_string(),
            })
            .unwrap_or_default();
        console::error(&format!(
            "channel chat path refused {channel} {} {code}",
            res.status
        ));
        if code == "scope_requires_local_inference" {
            return reply(messages::PERSONAL);
        }
        if res.status == 429 {
            return reply(messages::QUOTA);
        }
        if code == "request_in_progress" {
            return reply(messages::RETRY);
        }
        return reply(messages::BUSY);
    }

    let out: Value = serde_json::from_str(&res.body).unwrap_or(Value::Null);
    let answer = out
        .get("answer")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if answer.is_empty() {
        return reply(messages::BUSY);
    }
    let labels: Vec<&str> = out
        .get("cites")
        .and_then(Value::as_array)
        .map(|cs| {
            cs.iter()
                .map(|c| c.get("label").and_then(Value::as_str).unwrap_or_default())
                .collect()
        })
        .unwrap_or_default();
    reply(&compose(
        answer,
        &labels,
        adapter.max_chars,
        out.get("via").and_then(Value::as_str),
        out.get("notice").and_then(Value::as_str),
        text_deps,
    ))
}

#[cfg(test)]
mod tests {
    use super::sms::{SMS, SMS_MAX_CHARS};
    use super::ussd::{USSD, USSD_MAX_CHARS};
    use super::*;
    use crate::testing::block_on;
    use std::cell::RefCell;

    // Test copies of agent B's `extractive::plain` and `extractive::clip`
    // and the personal notices, ported verbatim from the TypeScript, so these
    // tests do not depend on another port. The wiring passes the real ones.

    fn plain(text: &str) -> String {
        text.replace(['\u{2018}', '\u{2019}', '\u{201B}'], "'")
            .replace(['\u{201C}', '\u{201D}'], "\"")
            .replace(['\u{2013}', '\u{2014}'], "-")
            .replace('\u{2026}', "...")
            .replace('\u{00A0}', " ")
    }

    fn clip(text: &str, max: usize, ellipsis: &str) -> String {
        let mut t = String::new();
        for c in text.trim().chars() {
            if (c == ' ' || c == '\t') && (t.ends_with(' ')) {
                continue;
            }
            t.push(if c == '\t' { ' ' } else { c });
        }
        while t.contains("\n\n\n") {
            t = t.replace("\n\n\n", "\n\n");
        }
        if js::len(&t) <= max {
            return t;
        }
        let cut = js::prefix(&t, max.saturating_sub(js::len(ellipsis)));
        let at = cut.rfind(' ').max(cut.rfind('\n'));
        let kept = match at {
            Some(i) if (js::len(&cut[..i]) as f64) > max as f64 * 0.5 => &cut[..i],
            _ => cut,
        };
        format!("{}{ellipsis}", kept.trim_end())
    }

    const TEXT: ChannelText = ChannelText {
        plain,
        clip,
        notice_short: "Answered without your own info. Not sent to an AI: public sources only.",
        notice_min: "Answered without your own info.",
    };
    const NOTICE: &str = "Answered without your own information, from public sources only. It looks personal, so it was not sent to an AI model.";

    /// 16 random bytes, hex: test secrets are drawn each run, so the repository
    /// holds no secret value (CodeQL rightly flags a literal one).
    fn random_secret() -> String {
        let mut b = [0u8; 16];
        getrandom::fill(&mut b).expect("random bytes");
        b.iter().map(|x| format!("{x:02x}")).collect()
    }

    /// The operator's shared secret for these tests, the same for the whole run.
    fn secret() -> &'static str {
        use std::sync::OnceLock;
        static S: OnceLock<String> = OnceLock::new();
        S.get_or_init(random_secret)
    }

    /// A secret that is never the operator's: a second, independent draw.
    fn wrong_secret() -> String {
        loop {
            let s = random_secret();
            if s != secret() {
                return s;
            }
        }
    }
    const PHONE: &str = "+263771234567";

    fn cfg() -> Config {
        Config {
            channel_webhook_secret: Some(secret().into()),
            channel_api_key: Some("sk_shamwari_channel".into()),
            ..Default::default()
        }
    }

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
    fn operator(body: &Value, secret: &str, signature: Option<&str>) -> Req {
        let raw = body.to_string();
        let sig = signature.map_or_else(|| sign(secret, &raw), str::to_owned);
        Req {
            method: "POST".into(),
            url: "https://gw.example/v1/channels/sms".into(),
            headers: vec![
                ("content-type".into(), "application/json".into()),
                ("x-signature".into(), sig),
            ],
            body: raw,
        }
    }

    fn text_msg(id: &str) -> Value {
        json!({ "id": id, "from": PHONE, "text": "Do I have a right to life?" })
    }

    fn compact(answer: &str) -> Value {
        json!({
            "id": "r1", "answer": answer,
            "cites": [{ "n": 1, "label": "Constitution of Zimbabwe - Section 48", "url": "https://example.invalid/constitution" }],
            "scope": "platform", "via": "gateway", "truncated": false, "degraded": false,
        })
    }

    /// Run the channel against a chat handler; returns the response and
    /// every request the chat handler was sent.
    fn run(
        adapter: &ChannelAdapter,
        req: Req,
        cfg: &Config,
        answer: impl Fn() -> Result<Resp, String>,
    ) -> (Resp, Vec<Req>) {
        let seen = RefCell::new(Vec::new());
        let r = block_on(handle_channel(adapter, &req, cfg, &TEXT, |chat| {
            seen.borrow_mut().push(chat);
            let a = answer();
            async move { a }
        }));
        (r, seen.into_inner())
    }

    fn ok(v: Value) -> Result<Resp, String> {
        Ok(Resp::json(&v, 200))
    }

    fn reply_of(r: &Resp) -> String {
        r.body_json()["reply"].as_str().unwrap().to_owned()
    }

    // SMS.

    #[test]
    fn answers_a_signed_message_with_the_answer_and_its_source_inside_three_segments() {
        let (res, sent) = run(
            &SMS,
            operator(&text_msg("m1"), secret(), None),
            &cfg(),
            || ok(compact("Section 48 protects the right to life.")),
        );
        assert_eq!(res.status, 200);
        let reply = reply_of(&res);
        assert!(reply.contains("Section 48 protects the right to life."));
        assert!(reply.contains("Source: Constitution of Zimbabwe - Section 48"));
        assert!(reply.len() <= SMS_MAX_CHARS);
        assert_eq!(sent.len(), 1);
    }

    #[test]
    fn the_chat_request_carries_the_adapters_scope_and_reach_a_hashed_key_and_never_the_sender() {
        let (_, sent) = run(
            &SMS,
            operator(
                &json!({ "id": "m4", "from": PHONE, "text": "  Do I have a right to life?  ", "scope": "personal", "reach": "r2" }),
                secret(),
                None,
            ),
            &cfg(),
            || ok(compact("ok")),
        );
        let r = &sent[0];
        let body: Value = serde_json::from_str(&r.body).unwrap();
        // The scope sent to the gateway is the adapter's, never the message's.
        assert_eq!(
            body,
            json!({
                "messages": [{ "role": "user", "content": "Do I have a right to life?" }],
                "scope": "platform", "reach": "r0", "format": "compact",
            })
        );
        assert_eq!(
            r.header("authorization"),
            Some("Bearer sk_shamwari_channel")
        );
        assert_eq!(r.header("x-shamwari-channel"), Some("sms"));
        assert_eq!(
            r.header("idempotency-key"),
            Some(sha256_hex("sms\nm4\nDo I have a right to life?").as_str())
        );
        let everything = format!("{r:?}");
        assert!(!everything.contains("263771234567"));
        assert!(!everything.contains("771234567"));
    }

    #[test]
    fn an_operator_retry_of_the_same_message_is_the_same_idempotency_key() {
        let key = |id: &str| {
            let (_, sent) = run(
                &SMS,
                operator(&text_msg(id), secret(), None),
                &cfg(),
                || ok(compact("ok")),
            );
            sent[0].header("idempotency-key").unwrap().to_owned()
        };
        assert_eq!(key("dup"), key("dup"));
        assert_ne!(key("dup"), key("other"));
    }

    #[test]
    fn the_reply_is_plain_ascii_so_one_curly_quote_cannot_force_ucs_2() {
        let (res, _) = run(
            &SMS,
            operator(&text_msg("m2"), secret(), None),
            &cfg(),
            || {
                let mut c =
                    compact("It\u{2019}s \u{201C}protected\u{201D} \u{2013} always\u{2026}");
                c["cites"][0]["label"] = json!("Constitution \u{2013} Zimbabwe");
                ok(c)
            },
        );
        assert!(reply_of(&res).is_ascii());
    }

    #[test]
    fn a_long_answer_is_cut_inside_the_limit_and_keeps_its_source_line() {
        let long = "word ".repeat(400);
        let (res, _) = run(
            &SMS,
            operator(&text_msg("m3"), secret(), None),
            &cfg(),
            || ok(compact(&long)),
        );
        let reply = reply_of(&res);
        assert!(reply.len() <= SMS_MAX_CHARS);
        assert!(reply.contains("Source: "));
    }

    #[test]
    fn a_spent_quota_is_a_short_message_not_a_500() {
        let (res, _) = run(
            &SMS,
            operator(&text_msg("m5"), secret(), None),
            &cfg(),
            || Ok(Resp::json(&json!({ "error": "quota_exceeded" }), 429)),
        );
        assert_eq!(res.status, 200);
        assert_eq!(reply_of(&res), messages::QUOTA);
    }

    // The operator is authenticated first.

    #[test]
    fn a_bad_signature_is_a_401_and_nothing_is_called() {
        let (res, sent) = run(
            &SMS,
            operator(&text_msg("x1"), &wrong_secret(), None),
            &cfg(),
            || ok(json!({})),
        );
        assert_eq!(res.status, 401);
        assert!(sent.is_empty());
    }

    #[test]
    fn a_missing_or_malformed_signature_is_a_401_and_nothing_is_called() {
        for sig in ["", "zz", &"a".repeat(64)] {
            let (res, sent) = run(
                &SMS,
                operator(&text_msg("x2"), secret(), Some(sig)),
                &cfg(),
                || ok(json!({})),
            );
            assert_eq!(res.status, 401, "{sig}");
            assert!(sent.is_empty());
        }
        let mut req = operator(&text_msg("x2"), secret(), None);
        req.headers.retain(|(k, _)| k != "x-signature");
        let (res, _) = run(&SMS, req, &cfg(), || ok(json!({})));
        assert_eq!(res.status, 401);
    }

    #[test]
    fn a_signature_in_upper_case_hex_is_accepted() {
        let raw = text_msg("x9").to_string();
        let sig = sign(secret(), &raw).to_uppercase();
        assert!(hmac_ok(secret(), &raw, Some(&sig)));
    }

    #[test]
    fn an_unconfigured_channel_answers_503_and_nothing_else() {
        for c in [
            Config {
                channel_api_key: None,
                ..cfg()
            },
            Config {
                channel_webhook_secret: None,
                ..cfg()
            },
            Config {
                channel_api_key: Some(String::new()),
                ..cfg()
            },
        ] {
            let (res, sent) = run(&SMS, operator(&text_msg("x3"), secret(), None), &c, || {
                ok(json!({}))
            });
            assert_eq!(res.status, 503);
            assert_eq!(res.body_json()["error"], "channel_not_configured");
            assert!(sent.is_empty());
        }
    }

    #[test]
    fn a_malformed_message_is_a_400_and_nothing_is_called() {
        let (res, sent) = run(
            &SMS,
            operator(&json!({ "id": 1 }), secret(), None),
            &cfg(),
            || ok(json!({})),
        );
        assert_eq!(res.status, 400);
        assert_eq!(res.body_json()["error"], "invalid_message");
        assert!(sent.is_empty());
        let mut req = operator(&json!({}), secret(), None);
        req.body = "{not json".into();
        let sig = sign(secret(), &req.body);
        req.headers.retain(|(k, _)| k != "x-signature");
        req.headers.push(("x-signature".into(), sig));
        let (res, _) = run(&SMS, req, &cfg(), || ok(json!({})));
        assert_eq!(res.body_json()["error"], "invalid_json");
    }

    #[test]
    fn an_oversized_body_is_refused_before_it_is_read_with_nothing_called() {
        let req = Req {
            method: "POST".into(),
            url: "https://gw.example/v1/channels/sms".into(),
            headers: vec![
                ("content-length".into(), "1000000".into()),
                ("x-signature".into(), "a".repeat(64)),
            ],
            body: "x".into(),
        };
        let (res, sent) = run(&SMS, req, &cfg(), || ok(json!({})));
        assert_eq!(res.status, 413);
        assert!(sent.is_empty());
        let mut big = operator(
            &json!({ "id": "b", "from": PHONE, "text": "x".repeat(9000) }),
            secret(),
            None,
        );
        big.headers.retain(|(k, _)| k != "content-length");
        let (res, _) = run(&SMS, big, &cfg(), || ok(json!({})));
        assert_eq!(res.status, 413);
    }

    // Rule 1 on a channel.

    #[test]
    fn rule_1_an_adapter_that_carries_personal_scope_is_refused_at_the_adapter_before_the_chat_path(
    ) {
        let personal = ChannelAdapter {
            scope: Scope::Personal,
            ..SMS
        };
        let (res, sent) = run(
            &personal,
            operator(&text_msg("p1"), secret(), None),
            &cfg(),
            || Ok(Resp::json(&json!("should not be reached"), 200)),
        );
        assert_eq!(res.status, 200);
        assert_eq!(reply_of(&res), messages::PERSONAL);
        assert!(sent.is_empty());
    }

    #[test]
    fn rule_1_personal_scope_is_a_message_the_person_can_act_on_not_a_silent_downgrade() {
        assert!(messages::PERSONAL.to_lowercase().contains("app"));
        assert!(messages::PERSONAL.len() <= 160);
    }

    #[test]
    fn rule_1_the_chat_paths_scope_refusal_is_the_personal_message() {
        let (res, _) = run(
            &SMS,
            operator(&text_msg("p3"), secret(), None),
            &cfg(),
            || {
                Ok(Resp::json(
                    &json!({ "error": "scope_requires_local_inference" }),
                    409,
                ))
            },
        );
        assert_eq!(reply_of(&res), messages::PERSONAL);
    }

    // USSD.

    #[test]
    fn an_empty_first_ussd_request_opens_the_session_with_a_prompt_and_calls_nothing() {
        let (res, sent) = run(
            &USSD,
            operator(
                &json!({ "sessionId": "s1", "msisdn": PHONE, "input": "" }),
                secret(),
                None,
            ),
            &cfg(),
            || ok(json!({})),
        );
        assert_eq!(
            res.body_json(),
            json!({ "message": "Shamwari. Type your question:", "end": false })
        );
        assert!(sent.is_empty());
    }

    #[test]
    fn answers_a_ussd_question_inside_one_screen_and_closes_the_session() {
        let long = "x ".repeat(300);
        let (res, sent) = run(
            &USSD,
            operator(
                &json!({ "sessionId": "s2", "msisdn": PHONE, "input": "Do I have a right to life?" }),
                secret(),
                None,
            ),
            &cfg(),
            || ok(compact(&long)),
        );
        let j = res.body_json();
        assert_eq!(j["end"], true);
        assert!(j["message"].as_str().unwrap().len() <= USSD_MAX_CHARS);
        assert!(j["message"].as_str().unwrap().contains("Source: "));
        assert_eq!(sent[0].header("x-shamwari-channel"), Some("ussd"));
    }

    // compose.

    #[test]
    fn compose_never_returns_more_than_the_limit_with_or_without_a_source() {
        let long = "a".repeat(2000);
        let label = "L".repeat(500);
        for max in [60, 182, 459] {
            assert!(compose(&long, &[&label], max, None, None, &TEXT).len() <= max);
            assert!(compose(&long, &[], max, None, None, &TEXT).len() <= max);
        }
    }

    #[test]
    fn the_notice_does_not_squeeze_a_ussd_answer_to_a_few_words() {
        let out = compose(
            &"x".repeat(2000),
            &[&"y".repeat(200)],
            182,
            Some("gateway"),
            Some(NOTICE),
            &TEXT,
        );
        let body = out.split('\n').next().unwrap();
        assert!(out.len() <= 182);
        assert!(body.len() >= 70, "{}", body.len());
        assert!(out.ends_with("Answered without your own info."));
        let sms = compose("ok", &["Act"], 459, Some("gateway"), Some(NOTICE), &TEXT);
        assert!(sms.ends_with(TEXT.notice_short));
    }

    // Review findings.

    #[test]
    fn a_message_still_in_flight_is_told_to_wait_not_that_its_question_was_personal() {
        let (res, _) = run(
            &SMS,
            operator(&text_msg("inflight"), secret(), None),
            &cfg(),
            || {
                Ok(Resp::json(
                    &json!({ "error": "request_in_progress", "retry": true }),
                    409,
                ))
            },
        );
        assert_eq!(reply_of(&res), messages::RETRY);
    }

    #[test]
    fn a_chat_path_that_errs_still_gets_the_operator_an_answer_in_its_own_format() {
        console::take();
        let (res, _) = run(
            &SMS,
            operator(&text_msg("t1"), secret(), None),
            &cfg(),
            || Err("all inference paths failed".into()),
        );
        assert_eq!(res.status, 200);
        assert_eq!(reply_of(&res), messages::BUSY);
        assert!(!console::take().join("\n").contains("771234567"));
    }

    #[test]
    fn a_refusal_from_the_chat_path_is_logged_with_its_status_and_code_never_the_sender() {
        console::take();
        let (res, _) = run(
            &SMS,
            operator(&text_msg("t2"), secret(), None),
            &cfg(),
            || Ok(Resp::json(&json!({ "error": "invalid_api_key" }), 401)),
        );
        assert_eq!(reply_of(&res), messages::BUSY);
        let logged = console::take().join("\n");
        assert!(logged.contains("401"));
        assert!(logged.contains("invalid_api_key"));
        assert!(!logged.contains("771234567"));
    }

    #[test]
    fn a_reply_is_gsm7_clean_no_emoji_accents_euro_sign_or_extension_characters() {
        assert_eq!(
            gsm7(
                "caf\u{00E9} \u{20AC}5 [Section 48] {x} ^~|\\ \u{1F600} ok",
                plain
            ),
            "cafe EUR5 (Section 48) (x) // ok"
        );
        let (res, _) = run(
            &SMS,
            operator(&text_msg("g1"), secret(), None),
            &cfg(),
            || ok(compact("Right to life \u{1F600} see [s.48] \u{20AC}")),
        );
        let reply = reply_of(&res);
        assert!(reply
            .chars()
            .all(|c| c == '\n' || ('\x20'..='\x7E').contains(&c)));
        assert!(!reply.contains(['[', ']', '{', '}', '^', '~', '|', '\\']));
    }

    #[test]
    fn an_extractive_answer_keeps_its_hedged_closest_source_line_and_is_never_relabelled() {
        let extractive = "Every person has the right to life.\nClosest source: Constitution - Section 48 (Parliament)\n\nOther passage.\nClosest source: Other";
        for max in [182, 459] {
            let out = compose(
                extractive,
                &["Constitution - Section 48"],
                max,
                Some("extractive"),
                None,
                &TEXT,
            );
            assert!(out.contains("Closest source: Constitution"));
            assert!(!out.starts_with("Source:") && !out.contains("\nSource:"));
            assert!(out.len() <= max);
        }
    }

    #[test]
    fn an_answer_with_no_source_says_so_instead_of_going_out_unsourced() {
        assert!(compose(
            "From general knowledge.",
            &[],
            459,
            Some("gateway"),
            None,
            &TEXT
        )
        .contains(NO_SOURCE_LINE));
        assert!(
            compose("x", &["", "Later"], 459, Some("gateway"), None, &TEXT)
                .contains("Source: Later")
        );
    }

    #[test]
    fn an_empty_answer_or_an_empty_question_is_a_short_message() {
        let (res, _) = run(
            &SMS,
            operator(&text_msg("e1"), secret(), None),
            &cfg(),
            || ok(json!({ "answer": "" })),
        );
        assert_eq!(reply_of(&res), messages::BUSY);
        let (res, sent) = run(
            &SMS,
            operator(
                &json!({ "id": "e2", "from": PHONE, "text": "   " }),
                secret(),
                None,
            ),
            &cfg(),
            || ok(json!({})),
        );
        assert_eq!(reply_of(&res), messages::EMPTY);
        assert!(sent.is_empty());
    }
}
