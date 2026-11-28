//! Port of `shamwari/gateway/src/responses.ts`: Open Responses,
//! `POST /v1/responses` (shamwari#291), pinned to the dated specification
//! 2026-04-24 (`src/specifications/2026-04-24.mdx` and
//! `public/openapi/2026-04-24/openapi.json` in openresponses/openresponses).
//!
//! A TRANSLATOR, NOT A SECOND CHAT PATH. This module has no model access, no
//! Ground access and no Core access, the same as a channel adapter. It turns
//! an Open Responses request into an ordinary chat request, calls
//! `/v1/chat/completions` IN-PROCESS with the caller's own `Authorization`,
//! and turns the answer back. So every rule that holds for a chat caller holds
//! here without being re-implemented, and none can be skipped:
//!
//! - Rule 1: the scope gate, the own-information rule, and the provider scope
//!   check at every provider call. An explicit `"scope": "personal"` is
//!   forwarded untouched, so it is the same 409, and the user's text is
//!   forwarded verbatim.
//! - Rule 2: provenance is resolved from the model that served, and the
//!   conversation row's `trainingEligible` comes from `conversation_trainable`,
//!   both in the chat path. `X-Shamwari-Channel` is forwarded as sent, never
//!   added or removed.
//!
//! WHAT IS REFUSED, AND WHY (never silently dropped). The chat path has no
//! tools, no stored responses, no streaming, no structured output, no caller
//! system prompt (Ground sets it) and no sampling controls. A field asking for
//! one of those is a 400 `invalid_request` naming the `param`, as "Errors"
//! requires, rather than an answer that quietly ignored it.
//!
//! [`handle`] answers 501 until the wiring lands: [`handle_responses`] takes
//! the caller's authentication (agent A's) as [`AuthGate`] and the chat
//! handler as a function.

use crate::config::Config;
use crate::failure::{console, from_error, js_len, public_reasons, FailureStep};
use crate::http::{Req, Resp};
use crate::platform::Platform;
use serde_json::{json, Map, Value};
use std::future::Future;

/// The spec's error types ("Errors" > "Error Types"), by HTTP status.
fn known_error(status: u16) -> &'static str {
    match status {
        400 | 401 | 403 | 409 | 422 => "invalid_request",
        404 => "not_found",
        429 => "too_many_requests",
        _ => "server_error",
    }
}

/// Headers a client needs on a failure: how long to wait, and whether it was a replay.
const PASS_HEADERS: [&str; 2] = ["Retry-After", "Idempotent-Replayed"];

/// `{ error: { type, code, message, param } }` with the status, as `errorResponse`.
/// `kind` defaults to the spec's type for the status.
pub fn error_response(
    status: u16,
    code: &str,
    message: &str,
    param: Option<&str>,
    headers: &[(String, String)],
    kind: Option<&str>,
) -> Resp {
    let kind = kind.unwrap_or_else(|| known_error(status));
    let mut r = Resp::json(
        &json!({ "error": { "type": kind, "code": code, "message": message, "param": param } }),
        status,
    );
    for (k, v) in headers {
        r.set_header(k, v);
    }
    r
}

fn unsupported(param: &str, message: &str) -> Resp {
    error_response(
        400,
        "unsupported_parameter",
        message,
        Some(param),
        &[],
        None,
    )
}

fn invalid(param: &str, message: &str) -> Resp {
    error_response(400, "invalid_value", message, Some(param), &[], None)
}

/// Present with a meaningful value: not undefined, not null.
fn given(v: Option<&Value>) -> bool {
    v.is_some_and(|v| !v.is_null())
}

/// `typeof v === 'object' && v !== null` (arrays included).
fn is_object_like(v: &Value) -> bool {
    v.is_object() || v.is_array()
}

/// A property of anything; only an object has one.
fn prop<'a>(v: &'a Value, k: &str) -> Option<&'a Value> {
    v.as_object().and_then(|o| o.get(k))
}

/// `JSON.stringify(v)`, with `undefined` for a missing value.
fn shown(v: Option<&Value>) -> String {
    v.map_or_else(|| "undefined".to_owned(), Value::to_string)
}

/// What the caller asked for that the response echoes ("ResponseResource").
#[derive(Debug, Clone, PartialEq)]
pub struct Echo {
    pub model: Option<String>,
    pub metadata: Map<String, Value>,
    /// `auto` or `disabled`.
    pub truncation: &'static str,
    /// `none` or `auto`.
    pub tool_choice: &'static str,
    pub parallel_tool_calls: bool,
    pub max_tool_calls: Option<Value>,
    pub safety_identifier: Option<String>,
    pub prompt_cache_key: Option<String>,
}

/// An Open Responses request as a chat request: the chat body, and what the
/// response echoes.
#[derive(Debug, Clone, PartialEq)]
pub struct Translated {
    pub chat: Value,
    pub echo: Echo,
}

/// One input item's text. User content is `input_text`, assistant content is
/// `output_text`; anything else (images, files, video, refusals) is a modality
/// the chat path cannot carry, so it is refused, not dropped.
fn text_of(content: Option<&Value>, role: &str, at: &str) -> Result<String, Resp> {
    if let Some(Value::String(s)) = content {
        return Ok(s.clone());
    }
    let Some(Value::Array(parts_in)) = content else {
        return Err(invalid(
            &format!("{at}.content"),
            "content must be a string or an array of content parts.",
        ));
    };
    let want = if role == "user" {
        "input_text"
    } else {
        "output_text"
    };
    let mut parts = Vec::new();
    for (j, p) in parts_in.iter().enumerate() {
        let here = format!("{at}.content[{j}]");
        if !is_object_like(p) {
            return Err(invalid(&here, "a content part must be an object."));
        }
        let t = prop(p, "type");
        if t.and_then(Value::as_str) != Some(want) {
            return Err(error_response(
                400,
                "unsupported_content_type",
                &format!(
                    "Content of type {} is not supported on a {role} message here: this endpoint accepts {want} only.",
                    shown(t)
                ),
                Some(&format!("{here}.type")),
                &[],
                None,
            ));
        }
        match prop(p, "text") {
            Some(Value::String(s)) => parts.push(s.clone()),
            _ => return Err(invalid(&format!("{here}.text"), "text must be a string.")),
        }
    }
    Ok(parts.join("\n"))
}

/// `input` as chat messages: a string is one user turn; items must be messages.
fn messages_of(input: Option<&Value>) -> Result<Vec<Value>, Resp> {
    if let Some(Value::String(s)) = input {
        if s.trim().is_empty() {
            return Err(invalid("input", "input must not be empty."));
        }
        return Ok(vec![json!({ "role": "user", "content": s })]);
    }
    let items = match input {
        Some(Value::Array(a)) if !a.is_empty() => a,
        _ => {
            return Err(error_response(
                400,
                "missing_required_parameter",
                "input is required: a string or a non-empty array of message items.",
                Some("input"),
                &[],
                None,
            ))
        }
    };
    let mut out = Vec::new();
    for (i, item) in items.iter().enumerate() {
        let at = format!("input[{i}]");
        if !is_object_like(item) {
            return Err(invalid(&at, "an input item must be an object."));
        }
        // `type` defaults to "message" in the schema.
        let kind = prop(item, "type").filter(|t| !t.is_null());
        if kind.is_some_and(|t| t.as_str() != Some("message")) {
            // function_call, function_call_output, reasoning, item_reference,
            // compaction, and any implementor-prefixed type.
            return Err(error_response(
                400,
                "unsupported_input_type",
                &format!(
                    "Input items of type {} are not supported here: this endpoint accepts message items only.",
                    shown(kind)
                ),
                Some(&format!("{at}.type")),
                &[],
                None,
            ));
        }
        let role = prop(item, "role").and_then(Value::as_str);
        if matches!(role, Some("system" | "developer")) {
            return Err(error_response(
                400,
                "unsupported_input_type",
                "System and developer messages are not supported: Shamwari sets the system prompt from its Ground \
                 context, and /v1/chat/completions does not use a caller system prompt either.",
                Some(&format!("{at}.role")),
                &[],
                None,
            ));
        }
        let role = match role {
            Some(r @ ("user" | "assistant")) => r,
            _ => {
                return Err(invalid(
                    &format!("{at}.role"),
                    "role must be \"user\" or \"assistant\".",
                ))
            }
        };
        let text = text_of(prop(item, "content"), role, &at)?;
        out.push(json!({ "role": role, "content": text }));
    }
    if !out.iter().any(|m| m["role"] == "user") {
        return Err(error_response(
            400,
            "missing_required_parameter",
            "input must contain at least one user message.",
            Some("input"),
            &[],
            None,
        ));
    }
    Ok(out)
}

/// MetadataParam: at most 16 pairs, keys up to 64 characters, string values up to 512.
fn metadata_of(v: Option<&Value>) -> Result<Map<String, Value>, Resp> {
    if !given(v) {
        return Ok(Map::new());
    }
    let Some(Value::Object(m)) = v else {
        return Err(invalid("metadata", "metadata must be an object."));
    };
    if m.len() > 16 {
        return Err(invalid(
            "metadata",
            "metadata holds at most 16 key-value pairs.",
        ));
    }
    for (k, val) in m {
        if js_len(k) > 64 {
            return Err(invalid(
                "metadata",
                "metadata keys are at most 64 characters.",
            ));
        }
        match val {
            Value::String(s) if js_len(s) <= 512 => {}
            _ => {
                return Err(invalid(
                    "metadata",
                    "metadata values are strings of at most 512 characters.",
                ))
            }
        }
    }
    Ok(m.clone())
}

/// The Open Responses request as a chat request, or the error that says why
/// it cannot be one. Pure: no I/O, so every refusal is testable without a world.
pub fn translate_request(body: &Value) -> Result<Translated, Resp> {
    let Some(b) = body.as_object() else {
        return Err(invalid("body", "The request body must be a JSON object."));
    };
    let g = |k: &str| b.get(k);
    let is_true = |k: &str| g(k) == Some(&Value::Bool(true));

    // --- Fields the chat path cannot honour: refused, never ignored. -------
    // Streaming: the chat path does not stream, so there is no honest event
    // sequence to send.
    if is_true("stream") {
        return Err(unsupported(
            "stream",
            "Streaming is not supported on this endpoint yet. Send \"stream\": false or omit it.",
        ));
    }
    if is_true("background") {
        return Err(unsupported(
            "background",
            "Background responses are not supported.",
        ));
    }
    // A server given previous_response_id MUST load that response. Nothing is stored here.
    if given(g("previous_response_id")) {
        return Err(unsupported(
            "previous_response_id",
            "Responses are not stored on this server, so previous_response_id cannot be loaded. Send the earlier turns in input.",
        ));
    }
    if is_true("store") {
        return Err(unsupported(
            "store",
            "Responses are not stored for retrieval on this server. Send \"store\": false or omit it.",
        ));
    }
    if given(g("conversation")) {
        return Err(unsupported(
            "conversation",
            "Server-side conversations are not supported.",
        ));
    }
    // Tools: the chat path has no tools. "required" or a named function MUST
    // call a tool, and "allowed_tools" MUST be enforced, so neither can be accepted.
    if given(g("tools"))
        && !g("tools")
            .and_then(Value::as_array)
            .is_some_and(Vec::is_empty)
    {
        return Err(unsupported(
            "tools",
            "Tools and function calling are not supported on this endpoint.",
        ));
    }
    if given(g("tool_choice"))
        && !matches!(
            g("tool_choice").and_then(Value::as_str),
            Some("auto" | "none")
        )
    {
        return Err(unsupported(
            "tool_choice",
            "No tools are available here, so tool_choice can only be \"auto\" or \"none\".",
        ));
    }
    if given(g("instructions")) && g("instructions") != Some(&json!("")) {
        return Err(unsupported(
            "instructions",
            "Caller instructions are not supported: Shamwari sets the system prompt from its Ground context.",
        ));
    }
    let include_given = match g("include") {
        Some(Value::Array(a)) => !a.is_empty(),
        other => given(other),
    };
    if include_given {
        return Err(unsupported(
            "include",
            "No optional output (logprobs, encrypted reasoning) is available here.",
        ));
    }
    if let Some(text) = g("text").filter(|t| !t.is_null()) {
        if !is_object_like(text) {
            return Err(invalid("text", "text must be an object."));
        }
        let format = prop(text, "format");
        if given(format) && format.and_then(|f| prop(f, "type")) != Some(&json!("text")) {
            return Err(unsupported(
                "text.format",
                "Only plain text output is supported; structured output is not.",
            ));
        }
        if given(prop(text, "verbosity")) {
            return Err(unsupported(
                "text.verbosity",
                "Output verbosity cannot be set here.",
            ));
        }
    }
    if let Some(r) = g("reasoning").filter(|r| !r.is_null()) {
        if !is_object_like(r) || given(prop(r, "effort")) || given(prop(r, "summary")) {
            return Err(unsupported(
                "reasoning",
                "Reasoning effort and summaries cannot be set here.",
            ));
        }
    }
    // Sampling controls: the routed model's settings are fixed by the gateway.
    for p in [
        "temperature",
        "top_p",
        "presence_penalty",
        "frequency_penalty",
        "max_output_tokens",
    ] {
        if given(g(p)) {
            return Err(unsupported(
                p,
                &format!("{p} cannot be set here: the gateway chooses the model and its settings."),
            ));
        }
    }
    if given(g("top_logprobs")) && g("top_logprobs").and_then(Value::as_f64) != Some(0.0) {
        return Err(unsupported(
            "top_logprobs",
            "Log probabilities are not available here.",
        ));
    }
    if given(g("service_tier"))
        && !matches!(
            g("service_tier").and_then(Value::as_str),
            Some("auto" | "default")
        )
    {
        return Err(unsupported(
            "service_tier",
            "Only the \"default\" service tier is available.",
        ));
    }
    // The chat endpoint's compact envelope is a different wire shape.
    if given(g("format")) {
        return Err(unsupported(
            "format",
            "format is a /v1/chat/completions option; this endpoint answers in the Open Responses shape.",
        ));
    }

    // --- Fields that are checked and echoed. -------------------------------
    if given(g("truncation"))
        && !matches!(
            g("truncation").and_then(Value::as_str),
            Some("auto" | "disabled")
        )
    {
        return Err(invalid(
            "truncation",
            "truncation must be \"auto\" or \"disabled\".",
        ));
    }
    if given(g("model")) && !g("model").is_some_and(Value::is_string) {
        return Err(invalid("model", "model must be a string."));
    }
    for p in ["safety_identifier", "prompt_cache_key"] {
        if given(g(p))
            && !g(p)
                .and_then(Value::as_str)
                .is_some_and(|s| js_len(s) <= 64)
        {
            return Err(invalid(
                p,
                &format!("{p} must be a string of at most 64 characters."),
            ));
        }
    }
    let metadata = metadata_of(g("metadata"))?;
    let messages = messages_of(g("input"))?;

    // The chat request. `scope`, `reach` and `mode` are Shamwari's own request
    // fields, forwarded exactly as sent: above all an explicit "personal"
    // scope, which the chat path's gate refuses with 409. Dropping it here
    // would turn the caller's "this is personal" into an ordinary Cloud request.
    let mut chat = Map::new();
    chat.insert("messages".into(), Value::Array(messages));
    if given(g("model")) {
        chat.insert("model".into(), g("model").cloned().unwrap_or_default());
    }
    for k in ["scope", "reach", "mode"] {
        if let Some(v) = g(k) {
            chat.insert(k.into(), v.clone());
        }
    }

    let string = |k: &str| g(k).and_then(Value::as_str).map(str::to_owned);
    Ok(Translated {
        chat: Value::Object(chat),
        echo: Echo {
            model: string("model"),
            metadata,
            truncation: if g("truncation") == Some(&json!("auto")) {
                "auto"
            } else {
                "disabled"
            },
            tool_choice: if g("tool_choice") == Some(&json!("none")) {
                "none"
            } else {
                "auto"
            },
            parallel_tool_calls: is_true("parallel_tool_calls"),
            max_tool_calls: g("max_tool_calls").filter(|v| v.is_number()).cloned(),
            safety_identifier: string("safety_identifier"),
            prompt_cache_key: string("prompt_cache_key"),
        },
    })
}

/// Deterministic from the chat id, so an idempotent replay returns the same
/// ids. `fallback` (32 hex digits, a fresh UUID's) is used only when the id
/// has no hex digit at all.
fn hex(id: &str, fallback: impl FnOnce() -> String) -> String {
    let h: String = id
        .chars()
        .filter(char::is_ascii_hexdigit)
        .collect::<String>()
        .to_lowercase();
    if h.is_empty() {
        fallback()
    } else {
        h
    }
}

/// The chat answer as a ResponseResource. Every field the 2026-04-24 schema
/// requires is present; Shamwari's citations, scope decision, provenance and
/// the own-information notice ride in one optional `shamwari` object. No core
/// field is invented or repurposed: `annotations` stays empty because a
/// `url_citation` needs a character span and a URL, and Ground's citations
/// have neither span nor, always, a URL.
pub fn to_response(
    chat: &Value,
    echo: &Echo,
    created_at: u64,
    completed_at: u64,
    fallback_id: impl FnOnce() -> String,
) -> Value {
    let choice = chat.pointer("/choices/0");
    let text = choice
        .and_then(|c| c.pointer("/message/content"))
        .filter(|t| !t.is_null())
        .cloned()
        .unwrap_or_else(|| json!(""));
    // "Items are state machines": an item cut off by the token budget is
    // `incomplete`, and then the response MUST be `incomplete` too.
    let cut = choice.and_then(|c| c.get("finish_reason")) == Some(&json!("length"));
    let id = hex(
        chat.get("id").and_then(Value::as_str).unwrap_or_default(),
        fallback_id,
    );
    let usage = chat.get("usage");
    let num = |k: &str| {
        usage
            .and_then(|u| u.get(k))
            .filter(|v| !v.is_null())
            .cloned()
    };
    let input = num("prompt_tokens").unwrap_or_else(|| json!(0));
    let output = num("completion_tokens").unwrap_or_else(|| json!(0));
    let total = num("total_tokens").unwrap_or_else(|| {
        json!(input.as_f64().unwrap_or(0.0) as u64 + output.as_f64().unwrap_or(0.0) as u64)
    });
    let status = if cut { "incomplete" } else { "completed" };
    let mut out = json!({
        "id": format!("resp_{id}"),
        "object": "response",
        "created_at": created_at,
        "completed_at": if cut { Value::Null } else { json!(completed_at) },
        "status": status,
        "incomplete_details": if cut { json!({ "reason": "max_output_tokens" }) } else { Value::Null },
        "model": chat.get("model").cloned().unwrap_or(Value::Null),
        "previous_response_id": null,
        "instructions": null,
        "output": [{
            "type": "message",
            "id": format!("msg_{id}"),
            "status": status,
            "role": "assistant",
            "content": [{ "type": "output_text", "text": text, "annotations": [], "logprobs": [] }],
        }],
        "error": null,
        "tools": [],
        "tool_choice": echo.tool_choice,
        "truncation": echo.truncation,
        "parallel_tool_calls": echo.parallel_tool_calls,
        "text": { "format": { "type": "text" } },
        "top_p": 1,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "top_logprobs": 0,
        "temperature": 1,
        "reasoning": null,
        "usage": {
            "input_tokens": input,
            "output_tokens": output,
            "total_tokens": total,
            "input_tokens_details": { "cached_tokens": 0 },
            "output_tokens_details": { "reasoning_tokens": 0 },
        },
        "max_output_tokens": null,
        "max_tool_calls": echo.max_tool_calls.clone().unwrap_or(Value::Null),
        "store": false,
        "background": false,
        "service_tier": "default",
        "metadata": echo.metadata,
        "safety_identifier": echo.safety_identifier,
        "prompt_cache_key": echo.prompt_cache_key,
        "shamwari": chat.get("shamwari").cloned().unwrap_or(Value::Null),
    });
    if chat.get("model").is_none() {
        // `model: undefined` is left out by JSON.stringify.
        if let Some(o) = out.as_object_mut() {
            o.remove("model");
        }
    }
    out
}

/// `String(v)` for a JSON value.
fn js_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Number(n) => match n.as_f64() {
            Some(f) if f.fract() == 0.0 && f.abs() < 1e21 => format!("{}", f as i64),
            _ => n.to_string(),
        },
        Value::Bool(b) => b.to_string(),
        Value::Null => "null".into(),
        Value::Array(a) => a
            .iter()
            .map(|x| {
                if x.is_null() {
                    String::new()
                } else {
                    js_string(x)
                }
            })
            .collect::<Vec<_>>()
            .join(","),
        Value::Object(_) => "[object Object]".into(),
    }
}

/// A chat-path refusal (`status` other than 200, its headers and body) in the
/// spec's error shape, keeping its status, code and wait.
pub fn translate_error(res: &Resp) -> Resp {
    let payload: Map<String, Value> = match serde_json::from_str::<Value>(&res.body) {
        Ok(Value::Object(m)) => m,
        _ => Map::new(),
    };
    let s = |k: &str| payload.get(k).and_then(Value::as_str);
    let code = s("error").unwrap_or("error").to_owned();
    let message = s("message")
        .or_else(|| s("hint"))
        .map(str::to_owned)
        .unwrap_or_else(|| code.replace('_', " "));
    let param = match code.as_str() {
        "invalid_mode" => Some("mode"),
        "invalid_reach" => Some("reach"),
        "invalid_idempotency_key" | "idempotency_key_reuse" => Some("Idempotency-Key"),
        "scope_requires_local_inference" => Some("scope"),
        _ => None,
    };
    let headers: Vec<(String, String)> = PASS_HEADERS
        .iter()
        .filter_map(|h| res.header(h).map(|v| (h.to_string(), v.to_owned())))
        .collect();
    // Every inference path failed with nothing from Ground: the chat path says
    // so as a retryable 503. In the spec's shape that is a model_error.
    if code == "inference_unavailable" {
        // The spec's error has no field for a list, so the reasons go in the
        // message, already cleaned by the chat path.
        let why: Vec<String> = payload
            .get("reasons")
            .and_then(Value::as_array)
            .map(|rs| {
                rs.iter()
                    .map(|r| {
                        let f = |k: &str| r.get(k).filter(|v| !v.is_null());
                        let step = f("step").map_or_else(|| "step".to_owned(), js_string);
                        let status = match r.get("status") {
                            Some(n @ Value::Number(_)) => format!(" {}", js_string(n)),
                            _ => String::new(),
                        };
                        let msg = f("message").map(js_string).unwrap_or_default();
                        format!("{step}{status}: {msg}")
                    })
                    .collect()
            })
            .unwrap_or_default();
        let why = why.join("; ");
        let id = s("request_id")
            .map(|i| format!(" (request {i})"))
            .unwrap_or_default();
        let colon = if why.is_empty() {
            String::new()
        } else {
            format!(": {why}")
        };
        let message = format!("No model could answer this request{colon}{id}. Retry later.");
        return error_response(
            res.status,
            "model_unavailable",
            &message,
            None,
            &headers,
            Some("model_error"),
        );
    }
    error_response(res.status, &code, &message, param, &headers, None)
}

/// The caller's authentication, as the wiring found it (agent A's
/// `authenticate`), checked before anything about the request is looked at.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthGate {
    Ok,
    /// Keys cannot be checked: a 503, never a 401.
    Unavailable,
    /// A real sign-in that is not provisioned for Shamwari; the detail says why.
    Forbidden(String),
    /// Missing or invalid.
    Invalid,
}

/// The internal chat request `/v1/responses` makes for the caller.
pub fn chat_request(req: &Req, chat_body: &Value) -> Req {
    let mut headers = vec![
        (
            "authorization".to_owned(),
            req.header("authorization").unwrap_or_default().to_owned(),
        ),
        ("content-type".to_owned(), "application/json".to_owned()),
    ];
    // Forwarded as sent and never invented: the channel marker decides what is
    // stored and whether it can ever be training material (Rule 2), and the
    // key makes a retry a replay.
    for h in ["idempotency-key", "x-shamwari-channel"] {
        if let Some(v) = req.header(h) {
            headers.push((h.to_owned(), v.to_owned()));
        }
    }
    Req {
        method: "POST".into(),
        url: "https://gateway.internal/v1/chat/completions".into(),
        headers,
        body: chat_body.to_string(),
    }
}

/// `POST /v1/responses`. `auth` is the caller's authentication; `chat` is the
/// gateway's own chat handler, called in-process with an ordinary chat
/// request carrying the caller's own credentials. `Err` from `chat` is an
/// exception that escaped the chat path.
pub async fn handle_responses<P, F, Fut>(p: &P, req: &Req, auth: AuthGate, chat: F) -> Resp
where
    P: Platform,
    F: FnOnce(Req) -> Fut,
    Fut: Future<Output = Result<Resp, String>>,
{
    let created_at = p.now_ms() / 1000;

    // Who is asking, before anything about the request is looked at. The chat
    // path authenticates again; this only keeps an unauthenticated caller from
    // learning how the body would have been judged.
    match auth {
        AuthGate::Ok => {}
        AuthGate::Unavailable => {
            return error_response(
                503,
                "auth_unavailable",
                "Authentication is unavailable. Retry shortly.",
                None,
                &[("Retry-After".into(), "5".into())],
                None,
            )
        }
        AuthGate::Forbidden(detail) => {
            return error_response(403, "not_provisioned", &detail, None, &[], None)
        }
        AuthGate::Invalid => {
            return error_response(
                401,
                "invalid_api_key",
                "The API key or token is missing or invalid.",
                None,
                &[],
                None,
            )
        }
    }

    let Ok(body) = serde_json::from_str::<Value>(&req.body) else {
        return error_response(
            400,
            "invalid_json",
            "The request body must be JSON.",
            Some("body"),
            &[],
            None,
        );
    };
    let t = match translate_request(&body) {
        Ok(t) => t,
        Err(r) => return r,
    };

    let res = match chat(chat_request(req, &t.chat)).await {
        Ok(r) => r,
        Err(e) => {
            console::error(&format!("responses: chat path threw {e}"));
            return error_response(
                500,
                "model_unavailable",
                "No model could answer this request. Retry later.",
                None,
                &[],
                Some("model_error"),
            );
        }
    };
    if res.status != 200 {
        return translate_error(&res);
    }

    let chat_body: Value = match serde_json::from_str(&res.body) {
        Ok(v) => v,
        // An exception here reaches the Worker's own catch: a 500 internal_error.
        Err(e) => {
            let reasons = public_reasons(&[from_error(FailureStep::Worker, &e.to_string())]);
            return Resp::json(
                &json!({ "error": "internal_error", "reasons": reasons }),
                500,
            );
        }
    };
    let out = to_response(&chat_body, &t.echo, created_at, p.now_ms() / 1000, || {
        crate::sink::random_uuid(p).replace('-', "")
    });
    let mut r = Resp::json(&out, 200);
    if let Some(v) = res.header("idempotent-replayed").filter(|v| !v.is_empty()) {
        r.set_header("Idempotent-Replayed", v);
    }
    r
}

pub async fn handle<P: Platform>(_p: &P, _cfg: &Config, _req: &Req) -> Resp {
    crate::router::not_ported("POST /v1/responses")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{block_on, FakePlatform};
    use std::cell::RefCell;

    const PLAIN_Q: &str = "What does section 48 of the Constitution say?";

    fn body(extra: Value) -> Value {
        let mut b = json!({ "input": PLAIN_Q });
        if let Value::Object(e) = extra {
            for (k, v) in e {
                b[k] = v;
            }
        }
        b
    }

    fn refused(r: Result<Translated, Resp>) -> (u16, Value) {
        let r = r.expect_err("expected a refusal");
        (r.status, r.body_json()["error"].clone())
    }

    /// The chat path's 200 answer, as the chat handler writes it.
    fn chat_answer(finish: &str) -> Value {
        json!({
            "id": "0b8f2c1e-6a4d-4f7e-9b1a-2c3d4e5f6a7b",
            "model": "shamwari-economy",
            "choices": [{ "index": 0, "message": { "role": "assistant", "content": "Section 48 protects the right to life." }, "finish_reason": finish }],
            "usage": { "prompt_tokens": 120, "completion_tokens": 9, "total_tokens": 129 },
            "shamwari": {
                "scope": "platform", "inference_path": "gateway", "model_served": "workers-ai/@cf/zai-org/glm-5.3",
                "model_substituted": false, "answered_without_own_information": false, "notice": null,
                "citations": [{ "title": "Constitution of Zimbabwe", "heading": "Section 48", "authority": "Parliament of Zimbabwe", "effective_from": "2013-05-22", "url": "https://example.invalid/constitution" }],
            },
        })
    }

    fn echo() -> Echo {
        translate_request(&body(json!({ "metadata": { "ticket": "A-1" } })))
            .unwrap()
            .echo
    }

    fn post(body: &str, headers: &[(&str, &str)]) -> Req {
        let mut h = vec![
            (
                "authorization".to_owned(),
                "Bearer sk_shamwari_test".to_owned(),
            ),
            ("content-type".to_owned(), "application/json".to_owned()),
        ];
        h.extend(headers.iter().map(|(k, v)| (k.to_string(), v.to_string())));
        Req {
            method: "POST".into(),
            url: "https://gw.example/v1/responses".into(),
            headers: h,
            body: body.into(),
        }
    }

    /// Runs `/v1/responses` against a chat handler that answers `answer`, and
    /// returns what the chat handler was sent.
    fn run(req: Req, auth: AuthGate, answer: Result<Resp, String>) -> (Resp, Option<Req>) {
        let p = FakePlatform::new();
        let seen = RefCell::new(None);
        let r = block_on(handle_responses(&p, &req, auth, |chat| {
            *seen.borrow_mut() = Some(chat);
            async { answer }
        }));
        (r, seen.into_inner())
    }

    fn ok(v: Value) -> Result<Resp, String> {
        Ok(Resp::json(&v, 200))
    }

    // Round trip.

    #[test]
    fn a_string_input_comes_back_as_a_completed_response_with_one_assistant_message() {
        let (res, sent) = run(
            post(
                &json!({ "model": "shamwari", "input": PLAIN_Q, "metadata": { "ticket": "A-1" } })
                    .to_string(),
                &[],
            ),
            AuthGate::Ok,
            ok(chat_answer("stop")),
        );
        assert_eq!(res.status, 200);
        assert_eq!(res.header("content-type"), Some("application/json"));
        let r = res.body_json();
        assert_eq!(r["object"], "response");
        let id = r["id"].as_str().unwrap();
        assert!(
            id.starts_with("resp_")
                && id.len() == 37
                && id[5..].chars().all(|c| c.is_ascii_hexdigit())
        );
        assert_eq!(r["status"], "completed");
        assert_eq!(r["error"], Value::Null);
        assert_eq!(r["incomplete_details"], Value::Null);
        assert!(r["created_at"].is_number());
        assert!(r["completed_at"].as_u64() >= r["created_at"].as_u64());
        assert_eq!(
            r["output"],
            json!([{
                "type": "message", "id": format!("msg_{}", &id[5..]), "status": "completed", "role": "assistant",
                "content": [{ "type": "output_text", "text": "Section 48 protects the right to life.", "annotations": [], "logprobs": [] }],
            }])
        );
        assert_eq!(
            r["usage"],
            json!({
                "input_tokens": 120, "output_tokens": 9, "total_tokens": 129,
                "input_tokens_details": { "cached_tokens": 0 }, "output_tokens_details": { "reasoning_tokens": 0 },
            })
        );
        assert_eq!(r["metadata"], json!({ "ticket": "A-1" }));
        assert_eq!(r["store"], false);
        assert_eq!(r["previous_response_id"], Value::Null);
        assert_eq!(r["tools"], json!([]));
        // The question reached the chat path once, as the user's turn.
        let sent: Value = serde_json::from_str(&sent.unwrap().body).unwrap();
        assert_eq!(
            sent,
            json!({ "messages": [{ "role": "user", "content": PLAIN_Q }], "model": "shamwari" })
        );
    }

    #[test]
    fn every_field_the_2026_04_24_response_resource_requires_is_present() {
        let r = to_response(&chat_answer("stop"), &echo(), 1, 2, String::new);
        for k in [
            "id",
            "object",
            "created_at",
            "completed_at",
            "status",
            "incomplete_details",
            "model",
            "previous_response_id",
            "instructions",
            "output",
            "error",
            "tools",
            "tool_choice",
            "truncation",
            "parallel_tool_calls",
            "text",
            "top_p",
            "presence_penalty",
            "frequency_penalty",
            "top_logprobs",
            "temperature",
            "reasoning",
            "usage",
            "max_output_tokens",
            "max_tool_calls",
            "store",
            "background",
            "service_tier",
            "metadata",
            "safety_identifier",
            "prompt_cache_key",
        ] {
            assert!(r.get(k).is_some(), "{k}");
        }
    }

    #[test]
    fn message_items_with_input_text_and_earlier_assistant_turns_become_the_chat_messages() {
        let t = translate_request(&json!({
            "input": [
                { "type": "message", "role": "user", "content": [{ "type": "input_text", "text": "What is section 48?" }] },
                { "type": "message", "role": "assistant", "content": [{ "type": "output_text", "text": "The right to life." }] },
                { "role": "user", "content": [{ "type": "input_text", "text": "Who" }, { "type": "input_text", "text": "does it cover?" }] },
            ],
        }))
        .unwrap();
        assert_eq!(
            t.chat["messages"],
            json!([
                { "role": "user", "content": "What is section 48?" },
                { "role": "assistant", "content": "The right to life." },
                { "role": "user", "content": "Who\ndoes it cover?" },
            ])
        );
    }

    #[test]
    fn carries_the_ground_citations_and_provenance_in_the_optional_shamwari_extension() {
        let r = to_response(&chat_answer("stop"), &echo(), 1, 2, String::new);
        assert_eq!(r["shamwari"], chat_answer("stop")["shamwari"]);
        assert_eq!(r["output"][0]["content"][0]["annotations"], json!([]));
    }

    #[test]
    fn an_answer_cut_off_by_the_token_budget_is_an_incomplete_item_in_an_incomplete_response() {
        let r = to_response(&chat_answer("length"), &echo(), 1, 2, String::new);
        assert_eq!(r["status"], "incomplete");
        assert_eq!(
            r["incomplete_details"],
            json!({ "reason": "max_output_tokens" })
        );
        assert_eq!(r["output"][0]["status"], "incomplete");
        assert_eq!(r["completed_at"], Value::Null);
    }

    #[test]
    fn ids_are_deterministic_from_the_chat_id_so_a_replay_returns_the_same_ids() {
        let a = to_response(&chat_answer("stop"), &echo(), 1, 2, || "x".into());
        let b = to_response(&chat_answer("stop"), &echo(), 3, 4, || "y".into());
        assert_eq!(a["id"], b["id"]);
        assert_eq!(a["id"], "resp_0b8f2c1e6a4d4f7e9b1a2c3d4e5f6a7b");
        let mut no_hex = chat_answer("stop");
        no_hex["id"] = json!("zzz");
        assert_eq!(
            to_response(&no_hex, &echo(), 1, 2, || "f00d".into())["id"],
            "resp_f00d"
        );
    }

    #[test]
    fn an_idempotent_replay_says_so() {
        let mut replay = Resp::json(&chat_answer("stop"), 200);
        replay.set_header("Idempotent-Replayed", "true");
        let (res, sent) = run(
            post(
                &json!({ "input": PLAIN_Q }).to_string(),
                &[("idempotency-key", "retry-1")],
            ),
            AuthGate::Ok,
            Ok(replay),
        );
        assert_eq!(res.header("idempotent-replayed"), Some("true"));
        assert_eq!(sent.unwrap().header("idempotency-key"), Some("retry-1"));
    }

    // Rule 1 holds on the new path: an explicit personal scope is forwarded
    // untouched, so the chat path's gate gives the same 409.

    #[test]
    fn rule_1_an_explicit_personal_scope_is_forwarded_and_its_409_keeps_its_message() {
        let refusal = json!({
            "error": "scope_requires_local_inference", "scope": "personal",
            "message": "Personal-layer data cannot be sent to an external inference provider.",
            "capability": "cloud_inference", "docs": "https://docs.shamwari.ai/scopes",
        });
        let (res, sent) = run(
            post(
                &json!({ "input": PLAIN_Q, "scope": "personal", "mode": "extractive" }).to_string(),
                &[],
            ),
            AuthGate::Ok,
            Ok(Resp::json(&refusal, 409)),
        );
        let sent: Value = serde_json::from_str(&sent.unwrap().body).unwrap();
        assert_eq!(sent["scope"], "personal");
        assert_eq!(sent["mode"], "extractive");
        assert_eq!(res.status, 409);
        let err = &res.body_json()["error"];
        assert_eq!(err["type"], "invalid_request");
        assert_eq!(err["code"], "scope_requires_local_inference");
        assert_eq!(err["param"], "scope");
        assert_eq!(err["message"], refusal["message"]);
    }

    #[test]
    fn rule_1_the_own_information_rule_sees_the_same_turns_text_is_forwarded_verbatim() {
        let q = "How much PAYE do I owe on my salary?";
        let t = translate_request(&json!({
            "input": [{ "type": "message", "role": "user", "content": [{ "type": "input_text", "text": q }] }],
        }))
        .unwrap();
        assert_eq!(
            t.chat["messages"],
            json!([{ "role": "user", "content": q }])
        );
    }

    #[test]
    fn a_key_that_holds_no_cloud_safe_scope_is_refused_as_on_the_chat_endpoint() {
        let e = Resp::json(
            &json!({ "error": "scope_not_permitted_for_key", "scope": "platform", "hint": "this key holds no community or platform scope" }),
            403,
        );
        let r = translate_error(&e);
        assert_eq!(r.status, 403);
        assert_eq!(
            r.body_json()["error"]["code"],
            "scope_not_permitted_for_key"
        );
        assert_eq!(
            r.body_json()["error"]["message"],
            "this key holds no community or platform scope"
        );
    }

    // Rule 2 holds on the new path: the channel marker is forwarded as sent,
    // never added or removed, so channel traffic is never trainable and the
    // web chat stores no conversation.

    #[test]
    fn rule_2_the_channel_marker_is_forwarded_as_sent_and_never_invented() {
        for ch in ["sms", "pwa"] {
            let (_, sent) = run(
                post(
                    &json!({ "input": PLAIN_Q }).to_string(),
                    &[("x-shamwari-channel", ch)],
                ),
                AuthGate::Ok,
                ok(chat_answer("stop")),
            );
            assert_eq!(sent.unwrap().header("x-shamwari-channel"), Some(ch));
        }
        let (_, sent) = run(
            post(&json!({ "input": PLAIN_Q }).to_string(), &[]),
            AuthGate::Ok,
            ok(chat_answer("stop")),
        );
        let sent = sent.unwrap();
        assert_eq!(sent.header("x-shamwari-channel"), None);
        assert_eq!(
            sent.header("authorization"),
            Some("Bearer sk_shamwari_test")
        );
        assert_eq!(sent.url, "https://gateway.internal/v1/chat/completions");
    }

    // Auth.

    #[test]
    fn no_key_is_a_401_in_the_spec_error_shape_and_nothing_is_called() {
        let (res, sent) = run(
            post(&json!({ "input": PLAIN_Q }).to_string(), &[]),
            AuthGate::Invalid,
            ok(json!({})),
        );
        assert_eq!(res.status, 401);
        let e = res.body_json();
        assert_eq!(e["error"]["type"], "invalid_request");
        assert_eq!(e["error"]["code"], "invalid_api_key");
        assert!(e["error"]["message"].is_string());
        assert_eq!(e["error"]["param"], Value::Null);
        assert!(sent.is_none());
    }

    #[test]
    fn an_unauthenticated_caller_gets_a_401_even_for_a_body_that_would_be_refused() {
        let (res, _) = run(
            post(&json!({ "input": "x", "stream": true }).to_string(), &[]),
            AuthGate::Invalid,
            ok(json!({})),
        );
        assert_eq!(res.status, 401);
    }

    #[test]
    fn auth_unavailable_is_a_503_with_retry_after_and_forbidden_says_why() {
        let (res, _) = run(post("{}", &[]), AuthGate::Unavailable, ok(json!({})));
        assert_eq!(res.status, 503);
        assert_eq!(res.header("retry-after"), Some("5"));
        assert_eq!(res.body_json()["error"]["code"], "auth_unavailable");
        let (res, _) = run(
            post("{}", &[]),
            AuthGate::Forbidden("no seat".into()),
            ok(json!({})),
        );
        assert_eq!(res.status, 403);
        assert_eq!(res.body_json()["error"]["message"], "no seat");
    }

    #[test]
    fn a_spent_quota_is_a_429_too_many_requests() {
        let r = translate_error(&Resp::json(
            &json!({ "error": "quota_exceeded", "tier": "community" }),
            429,
        ));
        assert_eq!(r.status, 429);
        assert_eq!(r.body_json()["error"]["type"], "too_many_requests");
        assert_eq!(r.body_json()["error"]["code"], "quota_exceeded");
        assert_eq!(r.body_json()["error"]["message"], "quota exceeded");
    }

    // Unsupported features are a 400 naming the param, never silently dropped.

    #[test]
    fn unsupported_features_are_a_400_naming_the_param() {
        let cases: Vec<(Value, &str)> = vec![
            (json!({ "stream": true }), "stream"),
            (json!({ "background": true }), "background"),
            (
                json!({ "previous_response_id": "resp_123" }),
                "previous_response_id",
            ),
            (json!({ "store": true }), "store"),
            (json!({ "conversation": "conv_1" }), "conversation"),
            (
                json!({ "tools": [{ "type": "function", "name": "f", "parameters": {} }] }),
                "tools",
            ),
            (json!({ "tool_choice": "required" }), "tool_choice"),
            (
                json!({ "tool_choice": { "type": "function", "name": "f" } }),
                "tool_choice",
            ),
            (
                json!({ "tool_choice": { "type": "allowed_tools", "tools": [] } }),
                "tool_choice",
            ),
            (
                json!({ "instructions": "Answer in French." }),
                "instructions",
            ),
            (
                json!({ "include": ["message.output_text.logprobs"] }),
                "include",
            ),
            (
                json!({ "text": { "format": { "type": "json_schema", "name": "x", "schema": {} } } }),
                "text.format",
            ),
            (json!({ "text": { "verbosity": "low" } }), "text.verbosity"),
            (json!({ "reasoning": { "effort": "high" } }), "reasoning"),
            (json!({ "temperature": 0.2 }), "temperature"),
            (json!({ "top_p": 0.5 }), "top_p"),
            (json!({ "max_output_tokens": 64 }), "max_output_tokens"),
            (json!({ "top_logprobs": 3 }), "top_logprobs"),
            (json!({ "service_tier": "priority" }), "service_tier"),
            (json!({ "format": "compact" }), "format"),
        ];
        for (extra, param) in cases {
            let (status, err) = refused(translate_request(&body(extra.clone())));
            assert_eq!(status, 400, "{extra}");
            assert_eq!(err["type"], "invalid_request", "{extra}");
            assert_eq!(err["code"], "unsupported_parameter", "{extra}");
            assert!(err["message"].is_string());
            assert_eq!(err["param"], param, "{extra}");
        }
    }

    #[test]
    fn unsupported_or_malformed_input_is_a_400_naming_where() {
        let cases: Vec<(Value, &str, &str)> = vec![
            (
                json!([{ "type": "message", "role": "user", "content": [{ "type": "input_image", "image_url": "https://x.example/a.png" }] }]),
                "unsupported_content_type",
                "input[0].content[0].type",
            ),
            (
                json!([{ "role": "user", "content": [{ "type": "input_text", "text": "see" }, { "type": "input_file", "file_url": "https://x.example/a.pdf" }] }]),
                "unsupported_content_type",
                "input[0].content[1].type",
            ),
            (
                json!([{ "type": "function_call_output", "call_id": "c1", "output": "{}" }]),
                "unsupported_input_type",
                "input[0].type",
            ),
            (
                json!([{ "type": "reasoning", "summary": [] }]),
                "unsupported_input_type",
                "input[0].type",
            ),
            (
                json!([{ "type": "acme:search_result", "id": "x", "status": "completed" }]),
                "unsupported_input_type",
                "input[0].type",
            ),
            (
                json!([{ "role": "system", "content": "Be terse." }, { "role": "user", "content": PLAIN_Q }]),
                "unsupported_input_type",
                "input[0].role",
            ),
            (
                json!([{ "role": "developer", "content": "Be terse." }, { "role": "user", "content": PLAIN_Q }]),
                "unsupported_input_type",
                "input[0].role",
            ),
            (
                json!([{ "role": "assistant", "content": "Hello." }]),
                "missing_required_parameter",
                "input",
            ),
            (Value::Null, "missing_required_parameter", "input"),
            (json!("   "), "invalid_value", "input"),
            (
                json!([{ "role": "tool", "content": "x" }]),
                "invalid_value",
                "input[0].role",
            ),
        ];
        for (input, code, param) in cases {
            let b = if input.is_null() {
                json!({})
            } else {
                json!({ "input": input })
            };
            let (status, err) = refused(translate_request(&b));
            assert_eq!(status, 400, "{b}");
            assert_eq!(err["type"], "invalid_request", "{b}");
            assert_eq!(err["code"], code, "{b}");
            assert_eq!(err["param"], param, "{b}");
        }
    }

    #[test]
    fn malformed_json_and_bad_metadata_are_400_invalid_request() {
        let (res, sent) = run(post("{not json", &[]), AuthGate::Ok, ok(json!({})));
        assert_eq!(res.status, 400);
        assert_eq!(res.body_json()["error"]["code"], "invalid_json");
        assert!(sent.is_none());
        let many: Map<String, Value> = (0..17).map(|i| (format!("k{i}"), json!("v"))).collect();
        for metadata in [
            Value::Object(many),
            json!({ "k": 1 }),
            json!({ "k": "x".repeat(513) }),
            json!(["a"]),
        ] {
            let (_, err) = refused(translate_request(&body(json!({ "metadata": metadata }))));
            assert_eq!(err["type"], "invalid_request");
            assert_eq!(err["param"], "metadata");
        }
        let (_, err) = refused(translate_request(&json!([1])));
        assert_eq!(err["param"], "body");
    }

    #[test]
    fn the_harmless_defaults_are_accepted() {
        let t = translate_request(&json!({
            "input": PLAIN_Q, "tools": [], "tool_choice": "none", "store": false, "stream": false,
            "instructions": null, "include": [], "text": { "format": { "type": "text" } },
            "truncation": "auto", "service_tier": "auto", "top_logprobs": 0, "temperature": null,
        }))
        .unwrap();
        assert_eq!(
            t.chat,
            json!({ "messages": [{ "role": "user", "content": PLAIN_Q }] })
        );
        assert_eq!(t.echo.tool_choice, "none");
        assert_eq!(t.echo.truncation, "auto");
    }

    #[test]
    fn forwards_the_shamwari_request_fields_scope_reach_and_mode_untouched_and_nothing_else() {
        let t = translate_request(&json!({
            "input": PLAIN_Q, "model": "shamwari", "scope": "personal", "reach": "r1", "mode": "extractive", "user": "u",
        }))
        .unwrap();
        assert_eq!(
            t.chat,
            json!({
                "messages": [{ "role": "user", "content": PLAIN_Q }],
                "model": "shamwari", "scope": "personal", "reach": "r1", "mode": "extractive",
            })
        );
    }

    #[test]
    fn a_chat_path_validation_error_keeps_its_status_in_the_spec_shape() {
        let r = translate_error(&Resp::json(
            &json!({ "error": "invalid_reach", "allowed": ["r0", "r1", "r2"] }),
            400,
        ));
        assert_eq!(r.status, 400);
        let e = &r.body_json()["error"];
        assert_eq!(e["type"], "invalid_request");
        assert_eq!(e["code"], "invalid_reach");
        assert_eq!(e["param"], "reach");
    }

    #[test]
    fn a_failure_on_every_inference_path_is_a_retryable_model_error_not_an_opaque_500() {
        let mut down = Resp::json(
            &json!({
                "error": "inference_unavailable", "retry": true, "request_id": "req-9",
                "reasons": [
                    { "step": "ai-gateway", "status": 503, "message": "HTTP 503" },
                    { "step": "workers-ai", "message": "workers ai down" },
                ],
            }),
            503,
        );
        down.set_header("Retry-After", "15");
        let r = translate_error(&down);
        assert_eq!(r.status, 503);
        assert_eq!(r.header("retry-after"), Some("15"));
        let e = &r.body_json()["error"];
        assert_eq!(e["type"], "model_error");
        assert_eq!(e["code"], "model_unavailable");
        assert_eq!(
            e["message"],
            "No model could answer this request: ai-gateway 503: HTTP 503; workers-ai: workers ai down (request req-9). Retry later."
        );
    }

    #[test]
    fn a_chat_path_that_errs_is_a_500_model_error() {
        let (res, _) = run(
            post(&json!({ "input": PLAIN_Q }).to_string(), &[]),
            AuthGate::Ok,
            Err("boom".into()),
        );
        assert_eq!(res.status, 500);
        assert_eq!(res.body_json()["error"]["type"], "model_error");
        assert_eq!(res.body_json()["error"]["code"], "model_unavailable");
    }

    #[test]
    fn a_refused_body_never_reaches_the_chat_path() {
        let (res, sent) = run(
            post(
                &json!({ "input": PLAIN_Q, "stream": true }).to_string(),
                &[],
            ),
            AuthGate::Ok,
            ok(json!({})),
        );
        assert_eq!(res.status, 400);
        assert!(sent.is_none());
    }
}
