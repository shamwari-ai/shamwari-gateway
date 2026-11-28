//! Port of `shamwari/gateway/src/channels/sms.ts`: the SMS adapter.
//!
//! The wire format is a placeholder until an operator or aggregator is chosen
//! (docs/tiered-access-design.md, open decisions): a signed JSON POST
//! `{id, from, text}` answered with `{reply}`. Another operator's shape changes
//! `parse` and `render` only; the gate and the pipeline stay in `channels`.
//!
//! 459 characters is three concatenated GSM-7 segments (3 x 153). Replies are
//! plain ASCII so one curly quote does not push the message into UCS-2.

use super::{ChannelAdapter, ChannelName, Inbound, Reply};
use crate::failure::js_len;
use crate::model_router::Reach;
use crate::types::Scope;
use serde_json::{json, Value};

pub const SMS_MAX_CHARS: usize = 459;

/// A non-empty string of at most `max` characters.
fn str_field(v: Option<&Value>, max: usize) -> Option<String> {
    v.and_then(Value::as_str)
        .filter(|s| !s.is_empty() && js_len(s) <= max)
        .map(str::to_owned)
}

fn parse(body: &Value) -> Option<Inbound> {
    let id = str_field(body.get("id"), 128)?;
    let sender = str_field(body.get("from"), 32)?;
    let text = body.get("text")?.as_str()?.to_owned();
    Some(Inbound { id, text, sender })
}

fn render(r: &Reply) -> Value {
    json!({ "reply": r.text })
}

pub const SMS: ChannelAdapter = ChannelAdapter {
    channel: ChannelName::Sms,
    reach: Reach::R0,
    scope: Scope::Platform,
    max_chars: SMS_MAX_CHARS,
    parse,
    greeting: None,
    render,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_operators_message_and_refuses_anything_malformed() {
        let m = (SMS.parse)(&json!({ "id": "m1", "from": "+263771234567", "text": "hi" })).unwrap();
        assert_eq!(
            (m.id.as_str(), m.sender.as_str(), m.text.as_str()),
            ("m1", "+263771234567", "hi")
        );
        for bad in [
            json!({ "id": 1 }),
            json!({ "id": "", "from": "x", "text": "t" }),
            json!({ "id": "m", "from": "x".repeat(33), "text": "t" }),
            json!({ "id": "m".repeat(129), "from": "x", "text": "t" }),
            json!({ "id": "m", "from": "x", "text": 5 }),
            json!(null),
            json!([]),
        ] {
            assert!((SMS.parse)(&bad).is_none(), "{bad}");
        }
    }

    #[test]
    fn renders_a_reply_and_is_a_platform_r0_channel() {
        assert_eq!(
            (SMS.render)(&Reply {
                text: "ok".into(),
                end: true
            }),
            json!({ "reply": "ok" })
        );
        assert_eq!(SMS.scope, Scope::Platform);
        assert_eq!(SMS.reach, Reach::R0);
        assert!(SMS.greeting.is_none());
    }
}
