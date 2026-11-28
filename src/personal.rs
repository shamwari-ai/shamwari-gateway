//! Port of `shamwari/gateway/src/personal.ts`: does a question need the
//! person's OWN information? (shamwari-ai/shamwari#81)
//!
//! The server decides this, never the user: there is no scope dropdown and a
//! client-sent scope is not read ([`crate::scope::resolve_scope`]). This is
//! the first version of that decision. It is a short list of deterministic
//! rules, no model, and it is deliberately conservative: when in doubt it
//! says yes.
//!
//! # What this is and is not
//!
//! It is an honesty control. When it says yes and the answer can only be
//! given from community and platform knowledge, the response carries
//! "answered without your own information" so a person is never left thinking
//! the answer used their own data.
//!
//! It is NOT the privacy control. A wrong answer here, in either direction,
//! cannot send personal-layer content to a provider: Cloud never holds any,
//! and the scope sent to a provider is community or platform whatever this
//! says ([`crate::scope::provider_scope`]). A false negative means the notice
//! is missing on one answer, so the rules lean towards a false positive.
//!
//! The same rules, word for word, live in `core/personal.py` and
//! `gateway/src/personal.ts`. Core's copy is authoritative for what it returns
//! and the gateway's is the fast path. `core/scope_cases.json` holds the cases
//! every suite runs (the copy here is in the tests below), so the copies
//! cannot drift apart without a test failing.
//!
//! # Languages
//!
//! English, Shona and Ndebele. The Shona and Ndebele rules are possessive
//! suffixes and a few first-person verb forms. No native speaker has reviewed
//! them yet; treat them as a starting point that errs towards flagging.

use crate::lang::js;
use crate::types::{Message, Role};
use regex::Regex;
use std::sync::OnceLock;
use unicode_normalization::UnicodeNormalization;

/// What a client may see as one line. Plain ASCII, so it survives every
/// channel.
///
/// It says what actually happened (shamwari-ai/shamwari#291). A question this
/// module flags is treated as personal scope: it is not sent to any hosted
/// model, including the Cloudflare-hosted embedding, and the answer is the
/// cited Ground passage with nothing generated. The earlier wording promised
/// on-device Mind, which has not shipped.
pub const OWN_INFORMATION_NOTICE: &str =
    "Answered without your own information, from public sources only. It looks personal, so it was not sent to an AI model.";

/// For SMS and USSD, where every character is billed and a segment is 160.
pub const OWN_INFORMATION_NOTICE_SHORT: &str =
    "Answered without your own info. Not sent to an AI: public sources only.";

/// For a screen of about 180 characters (USSD): the marker alone, so the
/// answer keeps its room.
pub const OWN_INFORMATION_NOTICE_MIN: &str = "Answered without your own info.";

/// Stable machine-readable name for the notice. Clients key their rendering
/// on this.
pub const OWN_INFORMATION_CODE: &str = "answered_without_your_own_information";

/// The bodies of the TypeScript rules, in order. Each is wrapped by [`word`].
const RULE_BODIES: &[&str] = &[
    // English: possessives and first-person contractions.
    "my|mine|myself|our|ours|ourselves|i'm|i've|i'll|i'd|im|ive",
    // English: "I" followed by something that is about the speaker's own situation.
    // A bare "I" is not enough ("Can I vote?" is a general question).
    "i (?:am|was|own|owe|earn|paid|pay|receive|received|lost|bought|sold|saved|spent|uploaded)",
    // "I have 3 rental properties", "I make $900": a first-person verb near a number or currency.
    // (`\d` in a JavaScript `u` regular expression is ASCII.)
    r"i (?:have|had|get|got|make|made|need) [^.?!]{0,20}?(?:\$|[0-9])",
    "i get paid|i work (?:at|for|as)",
    "(?:about|for|to|on|with|from) me",
    // English: things only the person holds.
    "payslips?|bank statements?|medical records?",
    // Shona: possessive -ngu (wangu, yangu, dzangu, changu, zvangu, rangu, vangu, kwangu, pangu, ...) and "ini".
    "(?:w|y|dz|ch|zv|r|v|kw|p|rw|tw|bw|sw)angu|ini",
    // Shona: first-person verbs (ndiri, ndine, ndakabhadhara, ndichada, ndinoda, ndiku...).
    r"ndiri|ndine|nda[ck]a[\p{L}\p{N}_]*|ndichi[\p{L}\p{N}_]*|ndino[\p{L}\p{N}_]*|ndiku[\p{L}\p{N}_]*",
    // Ndebele: possessive -mi (wami, yami, lami, zami, sami, bami, kwami) and "mina".
    "(?:w|y|l|z|s|b|kw)ami|mina",
    // Ndebele: first-person verbs.
    "ngine|ngikhokhile|ngisebenza|ngihlala|ngiyakhokha",
];

/// Whole-word match that also works next to apostrophes and non-ASCII
/// letters. TypeScript writes it with a lookbehind and a lookahead, which the
/// `regex` crate has not got; consuming the neighbouring character instead
/// answers the same question ("is there such a match anywhere?"), and that is
/// the only question asked here.
fn word(body: &str) -> String {
    format!(r"(?:^|[^\p{{L}}\p{{N}}_])(?:{body})(?:[^\p{{L}}\p{{N}}_]|$)")
}

fn rules() -> &'static [Regex] {
    static RULES: OnceLock<Vec<Regex>> = OnceLock::new();
    RULES.get_or_init(|| {
        RULE_BODIES
            .iter()
            .map(|b| Regex::new(&word(b)).expect("a personal rule compiles"))
            .collect()
    })
}

/// Fold the cheap evasions: width and case, curly apostrophes, zero-width
/// characters.
pub fn normalise(text: &str) -> String {
    let folded: String = text
        .nfkc()
        .filter(|c| !matches!(c, '\u{200B}'..='\u{200D}' | '\u{2060}' | '\u{FEFF}'))
        .map(|c| match c {
            '\u{2018}' | '\u{2019}' | '\u{02BC}' | '`' => '\'',
            c => c,
        })
        .collect();
    let lower = folded.to_lowercase();
    // `.replace(/\s+/g, ' ')`, with JavaScript's whitespace set.
    let mut out = String::with_capacity(lower.len());
    let mut in_space = false;
    for c in lower.chars() {
        if js::is_space(c) {
            if !in_space {
                out.push(' ');
            }
            in_space = true;
        } else {
            out.push(c);
            in_space = false;
        }
    }
    out
}

pub fn needs_own_information(text: &str) -> bool {
    let t = normalise(text);
    rules().iter().any(|r| r.is_match(&t))
}

/// Looks at the latest two user turns. The latest alone would miss "and how
/// much tax is that?" straight after "my salary is...".
pub fn conversation_needs_own_information(messages: &[Message]) -> bool {
    let users: Vec<&Message> = messages.iter().filter(|m| m.role == Role::User).collect();
    let from = users.len().saturating_sub(2);
    users[from..]
        .iter()
        .any(|m| needs_own_information(&m.content))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use serde_json::Value;

    /// Every case of `core/scope_cases.json` in shamwari-ai/shamwari (main,
    /// 2026-11-28), in its order and with its escapes. Core and the
    /// TypeScript gateway run the same file; if it changes there, copy it here.
    const SCOPE_CASES: &str = r#"{
  "cases": [
    {"text": "What is my tax for this month?", "needs_own_information": true},
    {"text": "MY SALARY after PAYE", "needs_own_information": true},
    {"text": "How do I read this payslip?", "needs_own_information": true},
    {"text": "Explain this bank statement", "needs_own_information": true},
    {"text": "I earn $400 a month, how much PAYE do I pay?", "needs_own_information": true},
    {"text": "I owe ZIMRA, what are my options?", "needs_own_information": true},
    {"text": "I'm a landlord, what must I register?", "needs_own_information": true},
    {"text": "I\u2019m a landlord, what must I register?", "needs_own_information": true},
    {"text": "Tell me about me", "needs_own_information": true},
    {"text": "m\u200by salary", "needs_own_information": true},
    {"text": "\uff4dy salary", "needs_own_information": true},
    {"text": "Mari yangu iri kupi?", "needs_own_information": true},
    {"text": "Ndiri kuda kuziva mutero wangu", "needs_own_information": true},
    {"text": "Ndakabhadhara mutero here?", "needs_own_information": true},
    {"text": "ndakabhadhara\u00e9", "needs_own_information": true},
    {"text": "I have 3 rental properties, how much tax do I owe?", "needs_own_information": true},
    {"text": "I make $900 a month, what is my PAYE band?", "needs_own_information": true},
    {"text": "I get paid weekly, is that taxable?", "needs_own_information": true},
    {"text": "I work at a bank, what is the minimum wage?", "needs_own_information": true},
    {"text": "Imali yami ikuphi?", "needs_own_information": true},
    {"text": "Ngine imali encane", "needs_own_information": true},
    {"text": "What does section 56 of the Constitution say?", "needs_own_information": false},
    {"text": "Can I vote?", "needs_own_information": false},
    {"text": "Do I have a right to life?", "needs_own_information": false},
    {"text": "Do I need a lawyer to register a company?", "needs_own_information": false},
    {"text": "Can I work in another country?", "needs_own_information": false},
    {"text": "What is the VAT rate in Zimbabwe?", "needs_own_information": false},
    {"text": "Tell me the VAT rate", "needs_own_information": false},
    {"text": "Who may stand for president?", "needs_own_information": false},
    {"text": "Ndeipi mitemo yekubhadhara mutero?", "needs_own_information": false},
    {"text": "Imithetho yelizwe isithini?", "needs_own_information": false},
    {"text": "mystery novel", "needs_own_information": false},
    {"text": "immigration policy", "needs_own_information": false},
    {"text": "common law marriage", "needs_own_information": false},
    {"text": "Chinese investment law", "needs_own_information": false},
    {"text": "", "needs_own_information": false}
  ]
}"#;

    pub(crate) fn scope_cases() -> Vec<(String, bool)> {
        let v: Value = serde_json::from_str(SCOPE_CASES).unwrap();
        v["cases"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| {
                (
                    c["text"].as_str().unwrap().to_owned(),
                    c["needs_own_information"].as_bool().unwrap(),
                )
            })
            .collect()
    }

    fn user(content: &str) -> Message {
        Message {
            role: Role::User,
            content: content.into(),
        }
    }

    // scope-decision.test.ts: "the rules agree with Core (shared cases)".
    #[test]
    fn gives_the_answer_core_scope_cases_gives_for_every_case() {
        let cases = scope_cases();
        assert_eq!(cases.len(), 36);
        for (text, want) in cases {
            assert_eq!(needs_own_information(&text), want, "{text:?}");
        }
    }

    #[test]
    fn folds_width_case_curly_apostrophes_and_zero_width_characters() {
        assert_eq!(
            normalise("\u{FF2D}Y\u{200B} \u{2019}x\u{2060}  \t y"),
            "my 'x y"
        );
        assert_eq!(normalise("a\u{FEFF}b`c\u{02BC}"), "ab'c'");
    }

    #[test]
    fn a_bare_i_is_not_enough_and_whole_words_only() {
        assert!(!needs_own_information("Can I vote?"));
        assert!(!needs_own_information("mystery"));
        assert!(needs_own_information("(my) salary"));
        assert!(needs_own_information("my"));
        assert!(!needs_own_information("my_salary"));
        // A first-person verb near a number, within 20 characters.
        assert!(needs_own_information("I had about 20 goats"));
        assert!(!needs_own_information(
            "I had a very long list of things before 20"
        ));
        assert!(!needs_own_information("I had cows. 20 of them"));
    }

    #[test]
    fn looks_at_the_latest_two_user_turns() {
        let assistant = Message {
            role: Role::Assistant,
            content: "my my my".into(),
        };
        let follow_up = [
            user("I earn $400 a month"),
            assistant.clone(),
            user("and how much tax is that?"),
        ];
        assert!(conversation_needs_own_information(&follow_up));
        let older = [
            user("I earn $400 a month"),
            user("What is VAT?"),
            user("and how much tax is that?"),
        ];
        assert!(!conversation_needs_own_information(&older));
        // An assistant turn never flags a conversation.
        assert!(!conversation_needs_own_information(&[assistant]));
        assert!(!conversation_needs_own_information(&[]));
    }

    #[test]
    fn the_notice_says_what_happened_and_promises_nothing_unshipped() {
        assert!(OWN_INFORMATION_NOTICE.contains("not sent to an AI model"));
        assert!(!OWN_INFORMATION_NOTICE.contains("Mind"));
        for n in [
            OWN_INFORMATION_NOTICE,
            OWN_INFORMATION_NOTICE_SHORT,
            OWN_INFORMATION_NOTICE_MIN,
        ] {
            assert!(n.is_ascii());
            assert!(n.contains("Answered without your own info"));
        }
        assert!(OWN_INFORMATION_NOTICE_SHORT.len() < 160);
        assert_eq!(
            OWN_INFORMATION_CODE,
            "answered_without_your_own_information"
        );
    }
}
