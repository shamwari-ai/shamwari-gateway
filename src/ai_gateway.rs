//! Port of `shamwari/gateway/src/gateway.ts`: calls to inference through the
//! Cloudflare AI Gateway, and the Workers AI binding behind it.
//! (`gateway.rs` is the chat route from `index.ts`; this is the client it calls.)
//!
//! Two-step degradation:
//!
//! 1. AI Gateway: the dynamic route `dynamic/<CF_ROUTE_NAME>`, with caching,
//!    spend limits, observability, the tier branch and its own fallbacks.
//! 2. Workers AI: the binding directly, no AI Gateway in the path.
//!
//! Everything runs on Cloudflare (no third-party provider, no provider key),
//! so Cloudflare is the dependency; the second step only removes the AI
//! Gateway from the path. The steps are separate functions so that
//! [`crate::probe`] can exercise each one **on its own**: in one composed
//! function the second step is only reachable when the first fails, which is
//! exactly why fallbacks rot.
//!
//! Guardrails and DLP are AI Gateway settings, applied by Cloudflare on the
//! route; the Worker sends no guardrail options. What it does is keep what
//! the gateway says when it blocks a prompt (a 424 with `internalCode` 2016),
//! through [`crate::failure::from_body`], so the reason reaches the app.
//! The Workers AI call passes no gateway options (`AI.run`'s third argument),
//! exactly as the TypeScript does.

use crate::config::Config;
use crate::failure::{console, from_body, from_error, FailureStep, InferenceFailed, StepFailure};
use crate::model_router::{tier_name, Target, DEFAULT_MAX_OUTPUT_TOKENS};
use crate::platform::{HttpRequest, Platform};
use crate::provenance::{resolve_license_class, WORKERS_AI_FALLBACK_MODEL};
use crate::types::{LicenseClass, Message, Scope, Tier};
use serde_json::{json, Map, Value};
use std::fmt;

/// The Secrets Store binding that holds the AI Gateway token.
pub const CF_AIG_TOKEN: &str = "CF_AIG_TOKEN";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferencePath {
    Gateway,
    WorkersAi,
}

impl InferencePath {
    pub fn as_str(self) -> &'static str {
        match self {
            InferencePath::Gateway => "gateway",
            InferencePath::WorkersAi => "workers-ai",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct InferenceResult {
    pub text: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_hit: bool,
    /// The provider that actually served, per `cf-aig-provider` where present.
    pub provider: String,
    /// The model that actually served, per `cf-aig-model` where present.
    pub model: String,
    pub requested_provider: String,
    pub requested_model: String,
    /// Resolved from the served model, never from the requested tier.
    pub license_class: LicenseClass,
    /// True when a dynamic route or a fallback answered with something else.
    pub substituted: bool,
    pub path: InferencePath,
    /// The model's own stop reason: `length` means the output cap cut it off.
    pub finish_reason: String,
}

/// Context attached to the AI Gateway log entry, and available to Conditional
/// nodes inside a dynamic route.
///
/// Nothing derived from message content belongs here. These values are stored
/// in Cloudflare's logs, so the rule is the same shape as Rule 1's:
/// identifiers and routing dimensions only, never the user's words.
///
/// `scope` can only ever be community or platform on this path (personal is
/// refused before inference is reachable), but it is sent so a route and its
/// log can show that rather than assume it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestMetadata {
    pub tier: Tier,
    pub scope: Scope,
    pub owner_entity_id: String,
    pub surface: String,
    pub request_id: String,
}

/// AI Gateway accepts at most FIVE custom metadata entries per request and
/// silently ignores the rest, so a sixth would not error, it would quietly
/// stop reaching the dynamic route that branches on it. Values must be
/// string, number or boolean. Keys beginning `cf.` are reserved.
pub const AIG_METADATA_MAX_ENTRIES: usize = 5;

pub fn build_metadata(meta: &RequestMetadata) -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("tier".into(), tier_name(meta.tier).into());
    m.insert("scope".into(), meta.scope.as_str().into());
    m.insert(
        "owner_entity_id".into(),
        meta.owner_entity_id.clone().into(),
    );
    m.insert("surface".into(), meta.surface.clone().into());
    m.insert("request_id".into(), meta.request_id.clone().into());
    m
}

/// Thinking budget for the reasoning models: low, so the token limit goes to
/// the answer, not the reasoning.
pub const REASONING_EFFORT: &str = "low";

/// Rule 1 refused a provider call: `ScopeRefusal` in `scope.ts`. A refusal is
/// a bug in a caller and surfaces as itself, never as "unavailable"; the
/// chat handler answers it with the 409 `scope.rs` builds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScopeRefused(pub Scope);

impl fmt::Display for ScopeRefused {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("scope_requires_local_inference")
    }
}

impl std::error::Error for ScopeRefused {}

/// `providerScope({ scope, destination: "cloud" })`, called immediately
/// before every provider call. Only community and platform may reach one.
pub fn provider_scope(scope: Scope) -> Result<Scope, ScopeRefused> {
    match scope {
        Scope::Community | Scope::Platform => Ok(scope),
        Scope::Personal => Err(ScopeRefused(scope)),
    }
}

/// Why [`infer`] gave no answer.
#[derive(Debug, Clone, PartialEq)]
pub enum InferError {
    /// Rule 1. Never degrade on this.
    ScopeRefusal(ScopeRefused),
    /// Every step failed; each says why.
    Failed(InferenceFailed),
}

impl fmt::Display for InferError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InferError::ScopeRefusal(s) => s.fmt(f),
            InferError::Failed(e) => e.fmt(f),
        }
    }
}

/// Why [`via_workers_ai`] gave no answer.
#[derive(Debug, Clone, PartialEq)]
pub enum StepError {
    ScopeRefusal(ScopeRefused),
    /// The binding threw, or answered with no text.
    Failed(String),
}

impl fmt::Display for StepError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StepError::ScopeRefusal(s) => s.fmt(f),
            StepError::Failed(e) => f.write_str(e),
        }
    }
}

/// The AI Gateway URL for the account and gateway.
pub fn gateway_url(cfg: &Config) -> String {
    format!(
        "https://gateway.ai.cloudflare.com/v1/{}/{}/compat/chat/completions",
        cfg.cf_account_id.as_deref().unwrap_or_default(),
        cfg.cf_gateway_id.as_deref().unwrap_or_default()
    )
}

/// The request [`via_gateway`] sends. No provider key: `cf-aig-authorization`
/// is the only credential.
pub fn gateway_request(
    cfg: &Config,
    aig_token: &str,
    messages: &[Message],
    meta: &RequestMetadata,
    skip_cache: bool,
    max_tokens: u32,
) -> HttpRequest {
    let mut headers = vec![
        ("Content-Type".to_owned(), "application/json".to_owned()),
        (
            "cf-aig-authorization".to_owned(),
            format!("Bearer {aig_token}"),
        ),
        (
            "cf-aig-metadata".to_owned(),
            Value::Object(build_metadata(meta)).to_string(),
        ),
    ];
    if skip_cache {
        headers.push(("cf-aig-skip-cache".to_owned(), "true".to_owned()));
    }
    let body = json!({
        "model": format!("dynamic/{}", cfg.cf_route_name.as_deref().unwrap_or_default()),
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": max_tokens.min(DEFAULT_MAX_OUTPUT_TOKENS),
        // The route's models (GLM-5.3) are reasoning models that cannot switch
        // thinking off. At the default effort a real question spent all 2048
        // tokens thinking and came back with no text (checked live,
        // 2026-11-12); at "low" it answers in about 300 tokens.
        "reasoning_effort": REASONING_EFFORT,
    });
    HttpRequest {
        method: "POST".into(),
        url: gateway_url(cfg),
        headers,
        body: Some(body.to_string()),
        timeout_ms: None,
    }
}

fn tokens(usage: Option<&Value>, k: &str) -> u64 {
    usage
        .and_then(|u| u.get(k))
        .and_then(Value::as_u64)
        .unwrap_or(0)
}

fn parse(
    j: &Value,
    provider_header: Option<&str>,
    model_header: Option<&str>,
    target: &Target,
    cache_hit: bool,
) -> InferenceResult {
    let choice = j.get("choices").and_then(|c| c.get(0));
    // A dynamic route reports what it chose in these headers. When they are
    // absent, what was asked for is taken to be what served.
    let pick = |h: Option<&str>, asked: &str| {
        h.map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or(asked)
            .to_owned()
    };
    let provider = pick(provider_header, &target.provider);
    let model = pick(model_header, &target.model);
    let usage = j.get("usage");
    InferenceResult {
        text: choice
            .and_then(|c| c.pointer("/message/content"))
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        finish_reason: choice
            .and_then(|c| c.get("finish_reason"))
            .and_then(Value::as_str)
            .unwrap_or("stop")
            .to_owned(),
        input_tokens: tokens(usage, "prompt_tokens"),
        output_tokens: tokens(usage, "completion_tokens"),
        cache_hit,
        license_class: resolve_license_class(&provider, &model),
        substituted: provider != target.provider || model != target.model,
        provider,
        model,
        requested_provider: target.provider.clone(),
        requested_model: target.model.clone(),
        path: InferencePath::Gateway,
    }
}

/// Step 1: the AI Gateway's dynamic route.
///
/// `Ok(None)` rather than an error when the step is unavailable, so the
/// caller decides whether to degrade or to report; the reason goes to
/// `failures` when given. `skip_cache` exists for the probe: its prompt is
/// constant, so a cached HIT would report a healthy gateway while the route
/// behind it was failing. `Err` is only ever Rule 1.
#[allow(clippy::too_many_arguments)]
pub async fn via_gateway<P: Platform>(
    p: &P,
    cfg: &Config,
    target: &Target,
    messages: &[Message],
    meta: &RequestMetadata,
    skip_cache: bool,
    max_tokens: u32,
    failures: Option<&mut Vec<StepFailure>>,
) -> Result<Option<InferenceResult>, ScopeRefused> {
    // RULE 1, where the call is made. Outside the fallible part: a refusal is
    // a bug in a caller, and must not be swallowed as "the gateway is unavailable".
    provider_scope(meta.scope)?;
    let failure = match attempt(p, cfg, target, messages, meta, skip_cache, max_tokens).await {
        Ok(out) => return Ok(Some(out)),
        Err(f) => f,
    };
    if let Some(fs) = failures {
        fs.push(failure);
    }
    Ok(None)
}

async fn attempt<P: Platform>(
    p: &P,
    cfg: &Config,
    target: &Target,
    messages: &[Message],
    meta: &RequestMetadata,
    skip_cache: bool,
    max_tokens: u32,
) -> Result<InferenceResult, StepFailure> {
    let threw = |e: String| {
        console::warn(&format!("gateway threw {e}"));
        from_error(FailureStep::AiGateway, &e)
    };
    // A Secrets Store binding, so the token is read at request time and never
    // sits in the Worker's own secrets. If the read fails the step is
    // unavailable and the caller degrades, like any other gateway failure.
    let token = match p.secret_store_get(CF_AIG_TOKEN).await {
        Ok(Some(t)) => t,
        Ok(None) => return Err(threw(format!("{CF_AIG_TOKEN} has no value"))),
        Err(e) => return Err(threw(e)),
    };
    let req = gateway_request(cfg, &token, messages, meta, skip_cache, max_tokens);
    let res = p.fetch(req).await.map_err(threw)?;
    if res.ok() {
        let j: Value = serde_json::from_str(&res.body).map_err(|e| threw(e.to_string()))?;
        let out = parse(
            &j,
            res.header("cf-aig-provider"),
            res.header("cf-aig-model"),
            target,
            res.header("cf-aig-cache-status") == Some("HIT"),
        );
        // A 200 with no text is not an answer. Treated as a failure so the
        // next rung engages, and so a blank is never stored as a teacher output.
        if !out.text.trim().is_empty() {
            return Ok(out);
        }
        console::warn("gateway returned no text");
        return Err(StepFailure {
            step: FailureStep::AiGateway,
            status: Some(res.status),
            code: None,
            message: "the model returned no text".into(),
        });
    }
    // Keep what AI Gateway said, not just its status: "424" alone hid a
    // guardrail block for days.
    let why = from_body(FailureStep::AiGateway, res.status, &res.body);
    let code = match &why.code {
        Some(Value::String(s)) => s.clone(),
        Some(c) => c.to_string(),
        None => String::new(),
    };
    console::warn(&format!("gateway {} {code} {}", res.status, why.message));
    Err(why)
}

/// The model's text from either shape Workers AI answers with: the older
/// `{ response }` or the OpenAI-style `{ choices: [{ message: { content } }] }`.
/// Empty when there is none (a reasoning-only reply has no content).
pub fn workers_ai_text(out: &Value) -> String {
    if let Some(s) = out.get("response").and_then(Value::as_str) {
        return s.to_owned();
    }
    out.pointer("/choices/0/message/content")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned()
}

/// The input [`via_workers_ai`] gives the binding. `max_tokens` only when a
/// reach asks for less: a request with no reach is sent exactly as before.
pub fn workers_ai_input(messages: &[Message], max_tokens: u32) -> Value {
    let mut input = json!({ "messages": messages, "reasoning_effort": REASONING_EFFORT });
    if max_tokens < DEFAULT_MAX_OUTPUT_TOKENS {
        input["max_tokens"] = json!(max_tokens);
    }
    input
}

/// Step 2. Last resort: the Workers AI binding with no AI Gateway in the path.
/// Lower quality, still answers. A general model, not a coder variant: this
/// answers Shona and Ndebele questions about law and tax.
pub async fn via_workers_ai<P: Platform>(
    p: &P,
    target: &Target,
    messages: &[Message],
    scope: Scope,
    max_tokens: u32,
) -> Result<InferenceResult, StepError> {
    provider_scope(scope).map_err(StepError::ScopeRefusal)?;
    let fb = p
        .ai_run(
            WORKERS_AI_FALLBACK_MODEL,
            &workers_ai_input(messages, max_tokens),
            None,
        )
        .await
        .map_err(StepError::Failed)?;

    let text = workers_ai_text(&fb);
    // No text is a failure, not an answer. This used to substitute a canned
    // apology, which then travelled as the model's output: stored as a
    // conversation under an open-weight teacher label and flagged trainable.
    if text.trim().is_empty() {
        return Err(StepError::Failed("workers ai returned no text".into()));
    }
    let usage = fb.get("usage");
    Ok(InferenceResult {
        text,
        input_tokens: tokens(usage, "prompt_tokens"),
        output_tokens: tokens(usage, "completion_tokens"),
        cache_hit: false,
        provider: "workers-ai".into(),
        model: WORKERS_AI_FALLBACK_MODEL.into(),
        requested_provider: target.provider.clone(),
        requested_model: target.model.clone(),
        license_class: resolve_license_class("workers-ai", WORKERS_AI_FALLBACK_MODEL),
        substituted: true,
        path: InferencePath::WorkersAi,
        finish_reason: "stop".into(),
    })
}

/// Both steps, in order: the gateway, then Workers AI.
pub async fn infer<P: Platform>(
    p: &P,
    cfg: &Config,
    target: &Target,
    messages: &[Message],
    meta: &RequestMetadata,
    max_tokens: u32,
) -> Result<InferenceResult, InferError> {
    let mut failures = Vec::new();
    let first = via_gateway(
        p,
        cfg,
        target,
        messages,
        meta,
        false,
        max_tokens,
        Some(&mut failures),
    )
    .await
    .map_err(InferError::ScopeRefusal)?;
    if let Some(out) = first {
        return Ok(out);
    }
    match via_workers_ai(p, target, messages, meta.scope, max_tokens).await {
        Ok(out) => Ok(out),
        // A Rule 1 refusal is a bug in a caller and must surface as itself.
        Err(StepError::ScopeRefusal(s)) => Err(InferError::ScopeRefusal(s)),
        Err(StepError::Failed(e)) => {
            failures.push(from_error(FailureStep::WorkersAi, &e));
            Err(InferError::Failed(InferenceFailed { failures }))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_router::targets;
    use crate::platform::HttpResponse;
    use crate::testing::{block_on, FakePlatform};
    use crate::types::Role;

    fn messages() -> Vec<Message> {
        vec![Message {
            role: Role::User,
            content: "What is the VAT rate?".into(),
        }]
    }

    fn meta(tier: Tier) -> RequestMetadata {
        RequestMetadata {
            tier,
            scope: Scope::Platform,
            owner_entity_id: "e-1".into(),
            surface: "shamwari.ai".into(),
            request_id: "r-1".into(),
        }
    }

    fn cfg() -> Config {
        Config {
            cf_account_id: Some("acct".into()),
            cf_gateway_id: Some("gw".into()),
            cf_route_name: Some("shamwari".into()),
            economy_model: Some("@cf/zai-org/glm-5.3-flash".into()),
            standard_model: Some("@cf/zai-org/glm-5.3".into()),
            ..Default::default()
        }
    }

    /// A world whose Workers AI binding answers `fallback answer`.
    fn world() -> FakePlatform {
        let p = FakePlatform::new();
        p.secrets
            .borrow_mut()
            .insert(CF_AIG_TOKEN.into(), "aig".into());
        p.on_ai(|_, _| Some(Ok(json!({ "response": "fallback answer" }))));
        p
    }

    const BODY: &str = r#"{"choices":[{"message":{"content":"hello"}}],"usage":{"prompt_tokens":10,"completion_tokens":5}}"#;

    /// The gateway answers 200 with the given extra headers.
    fn stub_gateway(p: &FakePlatform, headers: &[(&str, &str)]) {
        let headers: Vec<(String, String)> = headers
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        p.on_fn(move |_| {
            Some(Ok(HttpResponse {
                status: 200,
                headers: headers.clone(),
                body: BODY.into(),
            }))
        });
    }

    fn header<'a>(r: &'a HttpRequest, name: &str) -> Option<&'a str> {
        r.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }

    fn run_infer(p: &FakePlatform, tier: Tier) -> Result<InferenceResult, InferError> {
        let c = cfg();
        block_on(infer(
            p,
            &c,
            targets(&c).get(tier),
            &messages(),
            &meta(Tier::Standard),
            DEFAULT_MAX_OUTPUT_TOKENS,
        ))
    }

    // gateway.test.ts: AI Gateway custom metadata.

    #[test]
    fn sends_at_most_the_five_entries_ai_gateway_will_actually_keep() {
        assert!(build_metadata(&meta(Tier::Standard)).len() <= AIG_METADATA_MAX_ENTRIES);
    }

    #[test]
    fn uses_only_scalar_values_objects_are_not_supported_as_metadata() {
        for v in build_metadata(&meta(Tier::Standard)).values() {
            assert!(v.is_string() || v.is_number() || v.is_boolean());
        }
    }

    #[test]
    fn sends_no_cf_keys_which_cloudflare_reserves_and_strips() {
        for k in build_metadata(&meta(Tier::Standard)).keys() {
            assert!(!k.starts_with("cf."));
        }
    }

    #[test]
    fn carries_the_tier_and_owner_so_a_route_can_branch_and_meter_per_tenant() {
        let m = build_metadata(&meta(Tier::Standard));
        assert_eq!(m["tier"], "standard");
        assert_eq!(m["owner_entity_id"], "e-1");
        assert_eq!(m["scope"], "platform");
    }

    #[test]
    fn never_puts_message_content_in_metadata_it_lands_in_cloudflare_logs() {
        for v in build_metadata(&meta(Tier::Standard)).values() {
            assert!(!v.as_str().unwrap().contains("VAT"));
        }
    }

    #[test]
    fn attaches_the_metadata_header_on_the_gateway_request() {
        let p = world();
        stub_gateway(&p, &[]);
        run_infer(&p, Tier::Standard).unwrap();
        let reqs = p.requests.borrow();
        let sent: Value =
            serde_json::from_str(header(&reqs[0], "cf-aig-metadata").unwrap()).unwrap();
        assert_eq!(sent["request_id"], "r-1");
    }

    // Provenance follows the model that actually served.

    #[test]
    fn uses_the_requested_model_when_the_gateway_names_no_substitute() {
        let p = world();
        stub_gateway(&p, &[]);
        let r = run_infer(&p, Tier::Standard).unwrap();
        assert_eq!(r.provider, "workers-ai");
        assert_eq!(r.model, "@cf/zai-org/glm-5.3");
        assert!(!r.substituted);
        assert_eq!(r.license_class, LicenseClass::OpenWeight);
        assert_eq!((r.input_tokens, r.output_tokens), (10, 5));
        assert_eq!(r.finish_reason, "stop");
        assert_eq!(r.path, InferencePath::Gateway);
    }

    #[test]
    fn reads_cf_aig_model_and_cf_aig_provider_when_a_dynamic_route_substitutes() {
        let p = world();
        stub_gateway(
            &p,
            &[
                ("cf-aig-provider", "workers-ai"),
                ("cf-aig-model", "@cf/zai-org/glm-5.3-flash"),
            ],
        );
        let r = run_infer(&p, Tier::Standard).unwrap();
        assert_eq!(r.provider, "workers-ai");
        assert_eq!(r.model, "@cf/zai-org/glm-5.3-flash");
        assert_eq!(r.requested_model, "@cf/zai-org/glm-5.3");
        assert!(r.substituted);
        assert_eq!(r.license_class, LicenseClass::OpenWeight);
    }

    #[test]
    fn stamps_open_weight_when_the_route_serves_glm_5_3_from_workers_ai() {
        let p = world();
        stub_gateway(
            &p,
            &[
                ("cf-aig-provider", "workers-ai"),
                ("cf-aig-model", "@cf/zai-org/glm-5.3"),
            ],
        );
        let r = run_infer(&p, Tier::Standard).unwrap();
        // The standard tier asks for exactly this pair, so nothing was swapped.
        assert!(!r.substituted);
        assert_eq!(r.license_class, LicenseClass::OpenWeight);
    }

    #[test]
    fn stamps_restricted_when_a_route_substitutes_a_restricted_model() {
        // THE ASSERTION THAT WOULD HAVE CAUGHT THE DEFECT. The Worker asked
        // for an open-weight tier; the route answered with Claude. Nothing in
        // the Worker changed, and nothing was deployed.
        let p = world();
        stub_gateway(
            &p,
            &[
                ("cf-aig-provider", "anthropic"),
                ("cf-aig-model", "claude-opus-5"),
            ],
        );
        let r = run_infer(&p, Tier::Standard).unwrap();
        assert_eq!(r.license_class, LicenseClass::Restricted);
        assert!(r.substituted);
    }

    #[test]
    fn reports_the_real_fallback_model_id_not_the_string_fallback() {
        let p = world();
        p.on("https://", 503, "nope");
        let r = run_infer(&p, Tier::Standard).unwrap();
        assert_eq!(r.path, InferencePath::WorkersAi);
        assert_eq!(r.provider, "workers-ai");
        assert_eq!(r.model, "@cf/zai-org/glm-5.3");
        assert_ne!(r.model, "fallback");
        assert_eq!(r.license_class, LicenseClass::OpenWeight);
        assert!(r.substituted);
    }

    #[test]
    fn degrades_gateway_to_workers_ai_when_the_route_fails() {
        let p = world();
        p.on("https://", 500, "nope");
        let r = run_infer(&p, Tier::Standard).unwrap();
        assert_eq!(r.path, InferencePath::WorkersAi);
        assert!(r.substituted);
        assert_eq!(r.license_class, LicenseClass::OpenWeight);
    }

    #[test]
    fn degrades_to_workers_ai_when_the_gateway_token_cannot_be_read() {
        let p = world();
        p.secrets.borrow_mut().clear();
        stub_gateway(&p, &[]);
        let r = run_infer(&p, Tier::Standard).unwrap();
        assert_eq!(r.path, InferencePath::WorkersAi);
        assert!(p.requests.borrow().is_empty());
    }

    #[test]
    fn calls_the_dynamic_route_and_sends_no_provider_key() {
        let p = world();
        stub_gateway(&p, &[]);
        run_infer(&p, Tier::Standard).unwrap();
        let reqs = p.requests.borrow();
        assert_eq!(reqs.len(), 1);
        assert_eq!(
            reqs[0].url,
            "https://gateway.ai.cloudflare.com/v1/acct/gw/compat/chat/completions"
        );
        assert_eq!(header(&reqs[0], "cf-aig-authorization"), Some("Bearer aig"));
        // No third-party provider: Authorization would carry a provider key.
        assert_eq!(header(&reqs[0], "Authorization"), None);
    }

    #[test]
    fn names_the_route_as_dynamic_cf_route_name_in_the_request_body() {
        let p = world();
        stub_gateway(&p, &[]);
        run_infer(&p, Tier::Economy).unwrap();
        let reqs = p.requests.borrow();
        let body: Value = serde_json::from_str(reqs[0].body.as_deref().unwrap()).unwrap();
        assert_eq!(body["model"], "dynamic/shamwari");
        assert_eq!(body["temperature"], 0.3);
        assert_eq!(body["reasoning_effort"], "low");
        assert_eq!(
            body["messages"][0],
            json!({ "role": "user", "content": "What is the VAT rate?" })
        );
    }

    #[test]
    fn every_step_failing_carries_each_reason_the_guardrail_block_included() {
        let p = FakePlatform::new();
        p.secrets
            .borrow_mut()
            .insert(CF_AIG_TOKEN.into(), "aig".into());
        let blocked = json!({
            "error": [{ "code": 2016, "message": "Prompt blocked due to security configurations" }],
            "internalCode": 2016, "message": "Prompt blocked due to security configurations",
        });
        p.on("https://", 424, &blocked.to_string());
        p.on_ai(|_, _| Some(Err("3040: capacity exceeded".into())));
        let Err(InferError::Failed(f)) = run_infer(&p, Tier::Standard) else {
            panic!("expected every step to fail");
        };
        assert_eq!(f.failures.len(), 2);
        assert_eq!(f.failures[0].status, Some(424));
        assert_eq!(f.failures[0].code, Some(json!(2016)));
        assert_eq!(
            f.failures[0].message,
            "Prompt blocked due to security configurations"
        );
        assert_eq!(f.failures[1].step, FailureStep::WorkersAi);
        assert_eq!(f.failures[1].message, "3040: capacity exceeded");
    }

    #[test]
    fn rule_1_personal_scope_reaches_neither_step() {
        let p = world();
        stub_gateway(&p, &[]);
        let c = cfg();
        let m = RequestMetadata {
            scope: Scope::Personal,
            ..meta(Tier::Economy)
        };
        let r = block_on(infer(&p, &c, &targets(&c).economy, &messages(), &m, 2048));
        assert_eq!(
            r,
            Err(InferError::ScopeRefusal(ScopeRefused(Scope::Personal)))
        );
        let w = block_on(via_workers_ai(
            &p,
            &targets(&c).economy,
            &messages(),
            Scope::Personal,
            2048,
        ));
        assert_eq!(
            w,
            Err(StepError::ScopeRefusal(ScopeRefused(Scope::Personal)))
        );
        assert!(p.requests.borrow().is_empty());
        assert!(p.ai_calls.borrow().is_empty());
    }

    // reach.test.ts: the token cap at the inference steps.

    #[test]
    fn the_gateway_step_sends_the_lower_cap_and_never_a_higher_one() {
        let p = world();
        stub_gateway(&p, &[]);
        let c = cfg();
        let m = meta(Tier::Economy);
        block_on(via_gateway(
            &p,
            &c,
            &targets(&c).economy,
            &messages(),
            &m,
            false,
            200,
            None,
        ))
        .unwrap();
        block_on(via_gateway(
            &p,
            &c,
            &targets(&c).economy,
            &messages(),
            &m,
            false,
            999_999,
            None,
        ))
        .unwrap();
        let sent: Vec<Value> = p
            .requests
            .borrow()
            .iter()
            .map(|r| {
                serde_json::from_str::<Value>(r.body.as_deref().unwrap()).unwrap()["max_tokens"]
                    .clone()
            })
            .collect();
        assert_eq!(sent, [json!(200), json!(DEFAULT_MAX_OUTPUT_TOKENS)]);
    }

    #[test]
    fn the_workers_ai_step_sends_max_tokens_only_when_a_reach_asks_for_less() {
        let p = world();
        let c = cfg();
        block_on(via_workers_ai(
            &p,
            &targets(&c).economy,
            &messages(),
            Scope::Platform,
            2048,
        ))
        .unwrap();
        block_on(via_workers_ai(
            &p,
            &targets(&c).economy,
            &messages(),
            Scope::Platform,
            200,
        ))
        .unwrap();
        let calls = p.ai_calls.borrow();
        assert!(calls[0].1.get("max_tokens").is_none());
        assert_eq!(calls[1].1["max_tokens"], 200);
        // No gateway options as AI.run's third argument, as in TypeScript.
        assert!(calls.iter().all(|(_, _, o)| o.is_none()));
        assert_eq!(calls[0].0, WORKERS_AI_FALLBACK_MODEL);
    }

    #[test]
    fn tells_the_caller_when_the_output_cap_cut_the_answer_off() {
        let p = world();
        p.on(
            "https://",
            200,
            r#"{"choices":[{"message":{"content":"cut off mid"},"finish_reason":"length"}]}"#,
        );
        let r = run_infer(&p, Tier::Economy).unwrap();
        assert_eq!(r.finish_reason, "length");
    }

    // empty-answer.test.ts: no text is a failure, not an answer.

    fn empty_world(response: Value) -> FakePlatform {
        let p = FakePlatform::new();
        p.secrets
            .borrow_mut()
            .insert(CF_AIG_TOKEN.into(), "t".into());
        p.on_ai(move |_, _| Some(Ok(response.clone())));
        p
    }

    fn gateway_once(p: &FakePlatform) -> Option<InferenceResult> {
        let c = cfg();
        block_on(via_gateway(
            p,
            &c,
            &targets(&c).economy,
            &messages(),
            &meta(Tier::Economy),
            false,
            2048,
            None,
        ))
        .unwrap()
    }

    fn workers_once(p: &FakePlatform) -> Result<InferenceResult, StepError> {
        let c = cfg();
        block_on(via_workers_ai(
            p,
            &targets(&c).economy,
            &messages(),
            Scope::Platform,
            2048,
        ))
    }

    #[test]
    fn the_gateway_step_returns_none_for_a_200_with_no_content() {
        let p = empty_world(json!({}));
        p.on(
            "https://",
            200,
            r#"{"choices":[{"message":{"content":"  "}}]}"#,
        );
        assert_eq!(gateway_once(&p), None);
    }

    #[test]
    fn the_gateway_step_still_returns_a_real_answer() {
        let p = empty_world(json!({}));
        p.on(
            "https://",
            200,
            r#"{"choices":[{"message":{"content":"ok"}}]}"#,
        );
        assert_eq!(gateway_once(&p).unwrap().text, "ok");
    }

    #[test]
    fn the_workers_ai_step_errs_rather_than_substituting_an_apology() {
        let p = empty_world(json!({}));
        let e = workers_once(&p).unwrap_err();
        assert!(e.to_string().contains("no text"), "{e}");
    }

    #[test]
    fn the_workers_ai_step_still_returns_a_real_answer() {
        let p = empty_world(json!({ "response": "ok" }));
        assert_eq!(workers_once(&p).unwrap().text, "ok");
    }

    #[test]
    fn the_workers_ai_step_reads_the_openai_style_answer_the_glm_models_give() {
        let p = empty_world(json!({
            "choices": [{ "message": { "role": "assistant", "content": "Hello there!" }, "finish_reason": "stop" }],
            "usage": { "prompt_tokens": 19, "completion_tokens": 5 },
        }));
        let out = workers_once(&p).unwrap();
        assert_eq!(out.text, "Hello there!");
        assert_eq!(out.input_tokens, 19);
        assert_eq!(out.output_tokens, 5);
    }

    #[test]
    fn an_openai_style_reply_with_no_content_is_still_a_failure_not_an_answer() {
        let p = empty_world(
            json!({ "choices": [{ "message": { "role": "assistant", "content": null } }] }),
        );
        assert!(workers_once(&p)
            .unwrap_err()
            .to_string()
            .contains("no text"));
    }
}
