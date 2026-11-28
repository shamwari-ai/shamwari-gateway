//! Port of `shamwari/gateway/src/channels/ussd.ts`: the USSD adapter.
//!
//! Placeholder wire format, as for SMS: a signed JSON POST
//! `{sessionId, msisdn, input}` answered with `{message, end}`. One question
//! per session: an empty input opens the session with a prompt, and the
//! question's answer closes it. A screen holds about 182 characters.

use super::{ChannelAdapter, ChannelName, Inbound, Reply};
use crate::lang::js;
use crate::model_router::Reach;
use crate::types::Scope;
use serde_json::{json, Value};

pub const USSD_MAX_CHARS: usize = 182;

fn parse(body: &Value) -> Option<Inbound> {
    let field = |k: &str, max: usize| {
        body.get(k)
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty() && js::len(s) <= max)
            .map(str::to_owned)
    };
    let id = field("sessionId", 128)?;
    let sender = field("msisdn", 32)?;
    let text = body
        .get("input")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    Some(Inbound { id, text, sender })
}

fn greeting(i: &Inbound) -> Option<Reply> {
    i.text.trim().is_empty().then(|| Reply {
        text: "Shamwari. Type your question:".into(),
        end: false,
    })
}

fn render(r: &Reply) -> Value {
    json!({ "message": r.text, "end": r.end })
}

pub const USSD: ChannelAdapter = ChannelAdapter {
    channel: ChannelName::Ussd,
    reach: Reach::R0,
    scope: Scope::Platform,
    max_chars: USSD_MAX_CHARS,
    parse,
    greeting: Some(greeting),
    render,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_session_and_reads_a_missing_input_as_empty() {
        let m = (USSD.parse)(&json!({ "sessionId": "s1", "msisdn": "+263771234567" })).unwrap();
        assert_eq!(m.text, "");
        assert!((USSD.parse)(&json!({ "sessionId": "", "msisdn": "x" })).is_none());
        assert!((USSD.parse)(&json!({ "sessionId": "s", "msisdn": "x".repeat(33) })).is_none());
        assert!((USSD.parse)(&json!({ "sessionId": "s", "input": "q" })).is_none());
    }

    #[test]
    fn an_empty_input_opens_the_session_and_a_question_does_not() {
        let open = Inbound {
            id: "s".into(),
            text: "  ".into(),
            sender: "x".into(),
        };
        let g = (USSD.greeting.unwrap())(&open).unwrap();
        assert_eq!(
            (USSD.render)(&g),
            json!({ "message": "Shamwari. Type your question:", "end": false })
        );
        let asked = Inbound {
            text: "q".into(),
            ..open
        };
        assert!((USSD.greeting.unwrap())(&asked).is_none());
    }
}
