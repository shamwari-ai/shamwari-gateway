//! Port of what is left of `shamwari/gateway/src/index.ts` once the modules
//! it calls are ported: `admit`, the chat completions handler,
//! `/v1/ground/context`, `/v1/models`, and the in-process re-dispatch that
//! `/v1/responses` and the SMS and USSD channels make into the chat path.
//!
//! The dispatch table itself is [`crate::router`]; every handler here answers
//! `Result<Resp, String>`, where `Err` is what an exception escaping the
//! TypeScript `worker.fetch` was: the router answers it `500 internal_error`
//! with [`public_reasons`], as the TypeScript top-level `catch` does.
//!
//! Where TypeScript hands a promise to `ctx.waitUntil` (the usage meter, the
//! sink queue, the idempotency store), this port awaits it before answering.
//! What is written, and when it can fail, is unchanged; only the response
//! waits for it.
//!
//! The two rules hold here exactly as in `index.ts`:
//!
//! 1. Personal content never reaches third-party inference: [`admit`] answers
//!    an explicit personal scope 409 before Ground, an embedding or a model
//!    is reached, and a question the gateway's own rules (or Core) say needs
//!    the person's own information is answered from the cited Ground passage
//!    with nothing generated ([`handle_chat`]).
//! 2. Only open-weight output with explicit consent trains Mind: the
//!    conversation row's `trainingEligible` comes from
//!    [`crate::provenance::conversation_trainable`] (inside
//!    [`crate::sink::conversation_doc`]), and an extractive answer writes no
//!    conversation row at all.
//!
//! The person's WorkOS token ([`person_session_token`]) is read for the one
//! request, handed to [`ground`] for the Nyuchi API, and never logged, cached
//! or stored.

use crate::ai_gateway::{infer, InferError, RequestMetadata};
use crate::auth::{auth_failure, authenticate, js_truthy, person_session_token, AuthResult};
use crate::channels::{handle_channel, sms::SMS, ussd::USSD, ChannelAdapter, ChannelText};
use crate::config::Config;
use crate::envelope::{is_format, to_compact, Format, FORMATS};
use crate::extractive::{
    clip, extractive_answer, plain, ExtractiveOptions, EXTRACTIVE_MAX_CHARS, NO_SOURCE_MESSAGE,
};
use crate::failure::{console, from_error, public_reasons, FailureStep};
use crate::ground::{ground, GroundResult};
use crate::http::{Req, Resp};
use crate::idempotency::{self as idem, Begin, StoredResponse};
use crate::keys::{permitted_scopes, usage_meter};
use crate::lang::detect_language;
use crate::model_router::{
    apply_reach, is_reach, policy, route, Reach, DEFAULT_MAX_OUTPUT_TOKENS, REACHES,
};
use crate::personal::{
    needs_own_information, OWN_INFORMATION_CODE, OWN_INFORMATION_NOTICE,
    OWN_INFORMATION_NOTICE_MIN, OWN_INFORMATION_NOTICE_SHORT,
};
use crate::platform::Platform;
use crate::pricing::{estimate_cost_usd, over_quota, PRICE_TABLE_VERSION};
use crate::responses::{handle_responses, AuthGate};
use crate::scope::{resolve_scope, Capability, Destination, ScopeInput, ScopeResolution};
use crate::sink::{
    conversation_doc, enqueue, extractive_usage_doc, is_pwa, iso_timestamp, random_uuid,
    retrieval_usage_doc, split_for_delivery, usage_event_doc, ConversationInput, UsageInput,
};
use crate::skills::{parse_skills, skills_need_own_information, with_skills, AppliedSkill, SKILLS_MAX};
use crate::types::{AuthContext, GroundChunk, Message, Role};
use serde_json::{json, Map, Value};

/// What a handler answers. `Err` is an exception that escaped the TypeScript
/// handler: the router turns it into `500 internal_error`.
pub type Handled = Result<Resp, String>;

/// Where the 429 sends a caller who has spent the month's tokens.
pub const UPGRADE_URL: &str = "https://platform.shamwari.ai/billing";

/// `X-Shamwari-Channel`: marks traffic that is never kept as training material.
const CHANNEL_HEADER: &str = "x-shamwari-channel";

/// What `compose` in `channels/` needs from the extractive rung and the
/// personal rules.
pub const CHANNEL_TEXT: ChannelText = ChannelText {
    plain,
    clip: clip_text,
    notice_short: OWN_INFORMATION_NOTICE_SHORT,
    notice_min: OWN_INFORMATION_NOTICE_MIN,
};

/// `clip(text, max, ellipsis).text`.
fn clip_text(text: &str, max: usize, ellipsis: &str) -> String {
    clip(text, max, ellipsis).0
}

fn with_header(mut res: Resp, name: &str, value: &str) -> Resp {
    res.set_header(name, value);
    res
}

/// The citations a caller sees, in the shape every response uses (`citationsOf`).
pub fn citations_of(chunks: &[GroundChunk]) -> Value {
    Value::Array(
        chunks
            .iter()
            .map(|c| {
                json!({
                    "title": c.title,
                    "heading": c.heading,
                    "authority": c.authority,
                    "effective_from": c.effective_from,
                    "url": c.source_url,
                })
            })
            .collect(),
    )
}

/// Whether this answer lacks the person's own information (`ownInformation`).
///
/// Either side saying yes is enough, and neither can unsay the other: Core is
/// authoritative for what it returns, the gateway for what it saw first. Only
/// a Cloud answer can lack the person's information; on the Mind endpoint the
/// device holds it and uses it, so the notice there would be wrong.
pub fn answered_without_own_information(decision: &ScopeResolution, g: &GroundResult) -> bool {
    decision.destination == Destination::Cloud
        && (decision.without_own_information || g.core_needs_own_information)
}

/// The one line a client renders when an answer was given without the
/// person's own information. Present on every chat and retrieval response,
/// true or false, so a client never has to guess from an absent field.
/// `notice` is null when there is nothing to say. Inserted into `out` in the
/// order the TypeScript spread puts them.
fn put_own_information(out: &mut Map<String, Value>, decision: &ScopeResolution, g: &GroundResult) {
    let without = answered_without_own_information(decision, g);
    out.insert("answered_without_own_information".into(), without.into());
    out.insert(
        "notice".into(),
        if without {
            json!({ "code": OWN_INFORMATION_CODE, "message": OWN_INFORMATION_NOTICE })
        } else {
            Value::Null
        },
    );
}

/// `{ ...head, ...ownInformation(decision, g), ...tail }`, in that order.
fn shamwari_block(
    head: Value,
    decision: &ScopeResolution,
    g: &GroundResult,
    tail: Value,
) -> Value {
    let mut out = match head {
        Value::Object(m) => m,
        _ => Map::new(),
    };
    put_own_information(&mut out, decision, g);
    if let Value::Object(t) = tail {
        out.extend(t);
    }
    Value::Object(out)
}

/// The language of the latest user turn, as `detectLanguage` reads it.
fn language_of(messages: &[Message]) -> Option<&'static str> {
    messages
        .iter()
        .rev()
        .find(|m| m.role == Role::User)
        .and_then(|m| detect_language(&m.content))
}

/// The 429 a spent caller gets, or `None` while there is quota left.
async fn quota<P: Platform>(p: &P, cfg: &Config, auth: &AuthContext) -> Option<Resp> {
    over_quota(p, cfg, auth).await.then(|| {
        Resp::json(
            &json!({ "error": "quota_exceeded", "tier": auth.tier, "upgrade": UPGRADE_URL }),
            429,
        )
    })
}

/// A request that passed [`admit`].
#[derive(Debug, Clone)]
pub struct Admitted {
    pub auth: AuthContext,
    /// The request body as sent.
    pub body: Map<String, Value>,
    /// `body.messages`, exactly as sent: stored and fingerprinted as it came.
    pub raw_messages: Value,
    pub messages: Vec<Message>,
    pub decision: ScopeResolution,
    /// Valid skills from the body; `None` when `skills` was sent but
    /// malformed (the chat endpoint answers 400).
    pub skills: Option<Vec<AppliedSkill>>,
}

/// Everything both endpoints do before they diverge: authenticate, meter,
/// parse, and gate on scope. The capability argument is what makes the gate
/// mean different things on the two endpoints (see `scope.rs`).
///
/// `defer_quota`: a retry of an answered request is replayed without a quota
/// check, so the chat endpoint checks the quota itself once it knows the
/// request is new.
///
/// `Ok(Err(resp))` is a refusal to answer with; `Err` an exception.
pub async fn admit<P: Platform>(
    p: &P,
    cfg: &Config,
    req: &Req,
    capability: Capability,
    defer_quota: bool,
) -> Result<Result<Admitted, Resp>, String> {
    let authed = authenticate(p, cfg, req).await;
    let auth = match authed {
        AuthResult::Ok(a) => a,
        other => return Ok(Err(auth_failure(&other))),
    };
    if !defer_quota {
        if let Some(over) = quota(p, cfg, &auth).await {
            return Ok(Err(over));
        }
    }

    let Ok(body) = serde_json::from_str::<Value>(&req.body) else {
        return Ok(Err(Resp::json(&json!({ "error": "invalid_json" }), 400)));
    };
    let body = match body {
        Value::Object(m) => m,
        // `body.messages` of null throws in TypeScript: the Worker's 500.
        Value::Null => return Err("Cannot read properties of null (reading 'messages')".into()),
        // Any other JSON value has no `messages`: `?? []`.
        _ => Map::new(),
    };

    let raw_messages = body.get("messages").cloned().unwrap_or(Value::Null);
    let empty = match &raw_messages {
        Value::Null => true,
        Value::Array(a) => a.is_empty(),
        Value::String(s) => s.is_empty(),
        _ => false,
    };
    if empty {
        return Ok(Err(Resp::json(
            &json!({ "error": "messages_required" }),
            400,
        )));
    }
    // TypeScript trusts the shape and fails somewhere later (a 500, or a role
    // it does not know passed on to a provider). Here a list that is not
    // `{role: system|user|assistant, content: string}` is refused up front.
    let Ok(messages) = serde_json::from_value::<Vec<Message>>(raw_messages.clone()) else {
        return Ok(Err(Resp::json(
            &json!({ "error": "invalid_messages", "hint": "messages is a list of {role, content} with role system, user or assistant and content a string" }),
            400,
        )));
    };

    // Scope gate, before anything expensive happens. The server decides the
    // scope: body.scope is read only to see whether the client explicitly
    // asked for personal.
    //
    // The sovereignty gate first, so an explicit personal request bound for
    // Cloud is still the 409 that says why. Then the key: a key may only hold
    // the scopes it holds. A personal-scope skill is never sent: the client
    // says `own_information: true` instead. A skill that was sent is checked
    // with the same personal rules as the question (skills.rs).
    let skills = parse_skills(body.get("skills"));
    let grants = permitted_scopes(&auth).to_vec();
    let own_information = body.get("own_information") == Some(&Value::Bool(true))
        || skills_need_own_information(skills.as_deref().unwrap_or(&[]), needs_own_information);
    let decision = match resolve_scope(ScopeInput {
        messages: &messages,
        requested: body.get("scope"),
        grants: &grants,
        cfg,
        capability,
        own_information,
    }) {
        Ok(d) => d,
        Err(refusal) => return Ok(Err(Resp::json(&refusal.to_payload(), 409))),
    };
    if !grants.contains(&decision.scope) {
        return Ok(Err(Resp::json(
            &json!({
                "error": "scope_not_permitted_for_key",
                "scope": decision.scope.as_str(),
                "permitted_scopes": grants,
                "hint": "this key holds no community or platform scope",
            }),
            403,
        )));
    }
    Ok(Ok(Admitted {
        auth,
        body,
        raw_messages,
        messages,
        decision,
        skills,
    }))
}

/// `admit`, with a refusal already answered.
macro_rules! admitted {
    ($e:expr) => {
        match $e? {
            Ok(a) => a,
            Err(resp) => return Ok(resp),
        }
    };
}

/// Ground the request: Core's search through [`crate::core`], so it shares
/// Core's breaker, and the person's own token for the Nyuchi API.
async fn ground_for<P: Platform>(
    p: &P,
    cfg: &Config,
    req: &Req,
    a: &Admitted,
    language: Option<&str>,
) -> Result<GroundResult, String> {
    let person_token = person_session_token(req, &a.auth);
    ground(
        p,
        cfg,
        |body| crate::core::core_json(p, cfg, "/ground/search", body),
        &a.messages,
        &a.auth.owner_entity_id,
        &a.decision,
        language,
        person_token.as_deref(),
    )
    .await
    .map_err(|e| e.to_string())
}

/// `GET /v1/models`: the model ids a caller may name. Authenticated like the
/// rest of /v1, but not metered and not quota-checked: listing costs nothing
/// to serve.
pub async fn handle_models<P: Platform>(p: &P, cfg: &Config, req: &Req) -> Handled {
    match authenticate(p, cfg, req).await {
        AuthResult::Ok(_) => Ok(crate::models::list_response(cfg)),
        other => Ok(auth_failure(&other)),
    }
}

/// `POST /v1/ground/context`: retrieval without generation. This is the
/// endpoint Shamwari Mind calls.
///
/// It is the only endpoint that may see personal scope, and it can only do so
/// because nothing here reaches a provider: `ground()` skips the edge
/// embedding for a `mind` destination, and inference is never called. The
/// device gets the system prompt and the citations, and answers for itself.
pub async fn handle_ground_context<P: Platform>(p: &P, cfg: &Config, req: &Req) -> Handled {
    let a = admitted!(admit(p, cfg, req, Capability::RetrievalOnly, false).await);

    let request_id = random_uuid(p);
    let started = p.now_ms();
    let language = language_of(&a.messages);
    let g = ground_for(p, cfg, req, &a, language).await?;
    let latency_ms = p.now_ms().saturating_sub(started);

    // Metered, but no conversation document. The gateway never sees the
    // answer on this path, and a personal-scope exchange is not Mind training
    // data.
    let doc = retrieval_usage_doc(
        &a.auth,
        &request_id,
        a.decision.scope,
        g.hit,
        a.decision.destination == Destination::Cloud,
        latency_ms,
        p.now_ms(),
    );
    usage_meter(cfg).record(p, cfg, &doc).await;

    let shamwari = shamwari_block(
        json!({
            "scope": a.decision.scope.as_str(),
            "destination": a.decision.destination.as_str(),
            "language": language,
            "grounded": g.hit,
        }),
        &a.decision,
        &g,
        json!({
            "inference_path": "mind",
            "latency_ms": latency_ms,
            "citations": citations_of(&g.chunks),
        }),
    );
    Ok(Resp::json(
        &json!({
            "id": request_id,
            "system_prompt": g.system_prompt,
            "shamwari": shamwari,
        }),
        200,
    ))
}

/// What both answering rungs of the chat path share.
struct Rung<'a> {
    auth: &'a AuthContext,
    decision: &'a ScopeResolution,
    g: &'a GroundResult,
    language: Option<&'static str>,
    request_id: &'a str,
    started: u64,
    reach: Option<Reach>,
}

/// The extractive rung: the cited passage from Ground, with no generation.
///
/// Reached two ways: the caller asks for it (`mode: "extractive"`, the cheap
/// path for narrow channels), or every inference path failed and Ground has
/// something to say (`degraded`). Either way the scope gate has already run
/// in [`admit`], and nothing here reaches a provider.
///
/// It writes a usage event and NO conversation row. An extractive answer has
/// no teacher model, so there is no honest `licenseClass` to stamp, and the
/// chunks it quotes carry their own `mind_eligible` terms that are not
/// visible here. Storing it as a conversation would put text of unknown
/// training status in the corpus that rule 2 protects, so it is metered and
/// nothing more.
async fn extractive_response<P: Platform>(
    p: &P,
    cfg: &Config,
    r: &Rung<'_>,
    degraded: bool,
) -> Resp {
    // r0 is SMS/USSD: typographic characters would force UCS-2 and turn three
    // segments into seven, so the text is plain there.
    let max_chars = r
        .reach
        .map_or(EXTRACTIVE_MAX_CHARS, |x| policy().reach(x).extractive_max_chars);
    let built = extractive_answer(
        &r.g.chunks,
        max_chars,
        ExtractiveOptions {
            plain_text: r.reach == Some(Reach::R0),
        },
    );
    let (text, used, truncated) = match &built {
        Some(b) => (b.text.as_str(), b.used.as_slice(), b.truncated),
        None => (NO_SOURCE_MESSAGE, &[][..], false),
    };
    let latency_ms = p.now_ms().saturating_sub(r.started);

    let doc = extractive_usage_doc(
        r.auth,
        r.request_id,
        r.decision.scope,
        r.g.hit,
        r.reach,
        degraded,
        latency_ms,
        p.now_ms(),
    );
    usage_meter(cfg).record(p, cfg, &doc).await;

    let reach = r.reach.map(Reach::as_str);
    let shamwari = shamwari_block(
        json!({
            "tier": "extractive",
            "scope": r.decision.scope.as_str(),
            "language": r.language,
            "grounded": r.g.hit,
        }),
        r.decision,
        r.g,
        json!({
            "cache_hit": false,
            "inference_path": "extractive",
            "reach": reach,
            "degraded": degraded,
            "truncated": truncated,
            "latency_ms": latency_ms,
            "citations": citations_of(used),
        }),
    );
    Resp::json(
        &json!({
            "id": r.request_id,
            "model": "shamwari-extractive",
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": text },
                "finish_reason": "stop",
            }],
            "usage": { "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0 },
            "shamwari": shamwari,
        }),
        200,
    )
}

/// `"Ground could not be asked"`, with the wait before a retry.
fn ground_unavailable(body: Value) -> Resp {
    with_header(Resp::json(&body, 503), "retry-after", "30")
}

/// The request fields that identify it for idempotency. A field the caller
/// did not send is left out (the TypeScript `canonical` drops `undefined`);
/// one sent as null is kept.
fn fingerprint_of(body: &Map<String, Value>) -> String {
    let mut o = Map::new();
    for k in ["messages", "model", "scope", "mode", "reach", "format"] {
        if let Some(v) = body.get(k) {
            o.insert(k.to_owned(), v.clone());
        }
    }
    idem::fingerprint(&Value::Object(o))
}

/// A degraded answer (every inference path failed, Ground answered) is real
/// but not final: the retry should get the chance of a proper one.
fn worth_storing(text: &str) -> bool {
    let Ok(o) = serde_json::from_str::<Value>(text) else {
        return false;
    };
    let present = |v: Option<&Value>| v.filter(|v| !v.is_null()).cloned();
    let degraded = present(o.get("degraded"))
        .or_else(|| present(o.get("shamwari").and_then(|s| s.get("degraded"))))
        .unwrap_or(Value::Null);
    !js_truthy(&degraded)
}

/// `POST /v1/chat/completions`.
pub async fn handle_chat<P: Platform>(p: &P, cfg: &Config, req: &Req) -> Handled {
    let a = admitted!(admit(p, cfg, req, Capability::CloudInference, true).await);

    // Skills and the own-information flag are read only here:
    // /v1/ground/context generates nothing and must not start refusing
    // fields it has always ignored.
    let Some(skills) = a.skills.as_deref() else {
        return Ok(Resp::json(
            &json!({
                "error": "invalid_skills",
                "max": SKILLS_MAX,
                "hint": "skills is a list of {name, description, instructions}",
            }),
            400,
        ));
    };
    if a.body.get("own_information").is_some_and(|v| !v.is_boolean()) {
        return Ok(Resp::json(
            &json!({ "error": "invalid_own_information", "allowed": [true, false] }),
            400,
        ));
    }

    // Only this endpoint has modes.
    let mode = a.body.get("mode");
    if mode.is_some_and(|m| m.as_str() != Some("extractive")) {
        return Ok(Resp::json(
            &json!({ "error": "invalid_mode", "allowed": ["extractive"] }),
            400,
        ));
    }
    let extractive_mode = mode.is_some();

    let reach = match a.body.get("reach") {
        None => None,
        Some(v) => match is_reach(v) {
            Some(r) => Some(r),
            None => {
                let allowed: Vec<&str> = REACHES.iter().map(|r| r.as_str()).collect();
                return Ok(Resp::json(
                    &json!({ "error": "invalid_reach", "allowed": allowed }),
                    400,
                ));
            }
        },
    };

    let format = match a.body.get("format") {
        None => None,
        Some(v) => match is_format(v) {
            Some(f) => Some(f),
            None => {
                return Ok(Resp::json(
                    &json!({ "error": "invalid_format", "allowed": FORMATS }),
                    400,
                ))
            }
        },
    };

    let raw_key = req.header("idempotency-key").map(str::to_owned);
    if let Some(k) = &raw_key {
        if !idem::is_valid_key(k) {
            return Ok(Resp::json(
                &json!({ "error": "invalid_idempotency_key", "pattern": idem::KEY_PATTERN }),
                400,
            ));
        }
    }

    let chat = Chat {
        p,
        cfg,
        req,
        a: &a,
        skills,
        extractive_mode,
        reach,
        format,
    };

    let Some(raw_key) = raw_key else {
        if let Some(over) = quota(p, cfg, &a.auth).await {
            return Ok(over);
        }
        return chat.shaped().await;
    };

    // The key is scoped to the entity and to the whole request: the same key
    // with a different request is a mistake, not a retry.
    let k = idem::storage_key(&a.auth.owner_entity_id, &raw_key);
    let fp = fingerprint_of(&a.body);
    match idem::begin(p, &k, &fp).await {
        Begin::Conflict => {
            return Ok(Resp::json(
                &json!({
                    "error": "idempotency_key_reuse",
                    "message": "This key was used for a different request.",
                }),
                422,
            ))
        }
        Begin::InProgress => {
            return Ok(with_header(
                Resp::json(&json!({ "error": "request_in_progress", "retry": true }), 409),
                "retry-after",
                "2",
            ))
        }
        // No quota check, no Ground, no inference, no usage event: a replay is
        // free, is not stored again, and is an answer the caller already earned.
        Begin::Replay { status, body } => {
            let mut res = Resp {
                status,
                headers: vec![],
                body,
            };
            res.set_header("content-type", "application/json");
            res.set_header("idempotent-replayed", "true");
            return Ok(res);
        }
        Begin::New => {}
    }

    if let Some(over) = quota(p, cfg, &a.auth).await {
        idem::release(p, &k).await;
        return Ok(over);
    }
    let res = match chat.shaped().await {
        Ok(r) => r,
        Err(e) => {
            idem::release(p, &k).await;
            return Err(e);
        }
    };
    if res.status == 200 && worth_storing(&res.body) {
        let stored = StoredResponse {
            status: res.status,
            body: res.body.clone(),
        };
        idem::complete(p, &k, &fp, &stored, a.decision.scope.as_str()).await;
    } else {
        idem::release(p, &k).await;
    }
    Ok(res)
}

/// One admitted chat request, between validation and the answer.
struct Chat<'a, P> {
    p: &'a P,
    cfg: &'a Config,
    req: &'a Req,
    a: &'a Admitted,
    skills: &'a [AppliedSkill],
    extractive_mode: bool,
    reach: Option<Reach>,
    format: Option<Format>,
}

impl<P: Platform> Chat<'_, P> {
    /// Compact is a shape, applied last, so both rungs and the replay agree.
    async fn shaped(&self) -> Handled {
        let res = self.produce().await?;
        if self.format != Some(Format::Compact) || res.status != 200 {
            return Ok(res);
        }
        let fold = |t: &str| plain(t);
        let plain_labels: Option<&dyn Fn(&str) -> String> =
            (self.reach == Some(Reach::R0)).then_some(&fold as &dyn Fn(&str) -> String);
        Ok(Resp::json_compact(
            &to_compact(&res.body_json(), plain_labels),
            200,
        ))
    }

    async fn produce(&self) -> Handled {
        let (p, cfg, a) = (self.p, self.cfg, self.a);
        let request_id = random_uuid(p);
        let started = p.now_ms();
        let language = language_of(&a.messages);

        let g = ground_for(p, cfg, self.req, a, language).await?;
        let rung = Rung {
            auth: &a.auth,
            decision: &a.decision,
            g: &g,
            language,
            request_id: &request_id,
            started,
            reach: self.reach,
        };

        if self.extractive_mode {
            // "Ground could not be asked" is not "Ground has nothing": only
            // the second may be said to a person as a fact about the corpus.
            if g.unavailable {
                return Ok(ground_unavailable(json!({
                    "error": "ground_unavailable",
                    "retry": true,
                    "request_id": request_id,
                    "reasons": [{
                        "step": "core",
                        "message": "Core could not be reached, so the cited passage could not be fetched",
                    }],
                })));
            }
            return Ok(extractive_response(p, cfg, &rung, false).await);
        }

        // PERSONAL SCOPE, BY THE GATEWAY'S OWN RULES (owner decision,
        // shamwari#291).
        //
        // A question that looks like it needs the person's own information is
        // treated as personal scope, not answered by a hosted model with a
        // notice attached. Every inference path is hosted by someone else
        // (the AI Gateway providers and the Workers AI fallback on
        // Cloudflare), so none is called: the answer is the cited Ground
        // passage, with nothing generated, and no conversation row is written
        // (extractive_response). ground() already skipped the edge embedding
        // for the same reason. Core saying yes counts too.
        if a.decision.without_own_information || g.core_needs_own_information {
            if g.unavailable {
                return Ok(ground_unavailable(
                    json!({ "error": "ground_unavailable", "retry": true }),
                ));
            }
            return Ok(extractive_response(p, cfg, &rung, false).await);
        }

        let requested = a.body.get("model").and_then(Value::as_str);
        let routed = apply_reach(route(&a.messages, requested, cfg), self.reach, cfg);
        let target = &routed.target;
        let mut to_model = vec![Message {
            role: Role::System,
            // Skills sit beside the constant prompt, never in place of it.
            // Only reached when the request is not personal scope (the branch
            // above returned the cited passage).
            content: with_skills(&g.system_prompt, self.skills),
        }];
        to_model.extend(a.messages.iter().filter(|m| m.role != Role::System).cloned());
        let max_tokens = self
            .reach
            .map_or(DEFAULT_MAX_OUTPUT_TOKENS, |r| policy().reach(r).max_output_tokens);
        let meta = RequestMetadata {
            tier: target.tier,
            scope: a.decision.scope,
            owner_entity_id: a.auth.owner_entity_id.clone(),
            surface: cfg.surface.clone().unwrap_or_default(),
            request_id: request_id.clone(),
        };

        let result = match infer(p, cfg, target, &to_model, &meta, max_tokens).await {
            Ok(r) => r,
            Err(e) => {
                // Every inference path failed. With a Ground hit there is
                // still a cited answer to give; without one there is nothing
                // honest to say, so the failure stands, as a 503 the caller
                // can retry (with when), not a 500: nothing is wrong with the
                // request, the models are just not answering.
                if !g.hit {
                    // Say WHY, all the way to the app: each step's reason
                    // (cleaned, failure.rs) and the request id that finds this
                    // in the logs.
                    let failures = match &e {
                        InferError::Failed(f) => f.failures.clone(),
                        InferError::ScopeRefusal(s) => {
                            vec![from_error(FailureStep::Worker, &s.to_string())]
                        }
                    };
                    let reasons = public_reasons(&failures);
                    console::error(&format!(
                        "inference unavailable and no Ground passage to give {request_id} {}",
                        serde_json::to_string(&reasons).unwrap_or_default()
                    ));
                    return Ok(with_header(
                        Resp::json(
                            &json!({
                                "error": "inference_unavailable",
                                "retry": true,
                                "request_id": request_id,
                                "reasons": reasons,
                            }),
                            503,
                        ),
                        "retry-after",
                        "15",
                    ));
                }
                console::error(&format!("inference unavailable, answering from Ground {e}"));
                return Ok(extractive_response(p, cfg, &rung, true).await);
            }
        };
        let latency_ms = p.now_ms().saturating_sub(started);

        let now_ms = p.now_ms();
        let now = iso_timestamp(now_ms);
        let reach = self.reach;
        let served = format!("{}/{}", result.provider, result.model);
        let without = answered_without_own_information(&a.decision, &g);
        let sink = vec![
            conversation_doc(&ConversationInput {
                auth: &a.auth,
                request_id: &request_id,
                surface: cfg.surface.as_deref(),
                scope: a.decision.scope,
                language,
                messages: &a.raw_messages,
                result: &result,
                chunks: &g.chunks,
                tier: target.tier,
                reach,
                grounded: g.hit,
                // Channel traffic (SMS, USSD) is never trainable: any value of
                // the marker counts.
                via_channel: self.req.header(CHANNEL_HEADER).is_some(),
                answered_without_own_information: without,
                now: &now,
            }),
            usage_event_doc(&UsageInput {
                auth: &a.auth,
                request_id: &request_id,
                scope: a.decision.scope,
                tier: target.tier,
                reach,
                reach_clamped: routed.clamped,
                routed_tier: routed.routed_tier,
                result: &result,
                // An estimate from a versioned table, never billing-grade;
                // null when the model that served is not priced.
                cost_usd_estimate: estimate_cost_usd(
                    Some(&served),
                    result.input_tokens,
                    result.output_tokens,
                ),
                price_table_version: PRICE_TABLE_VERSION,
                ground_hit: g.hit,
                latency_ms,
                now_ms,
            }),
        ];

        // The web chat (X-Shamwari-Channel: pwa) keeps no content on the
        // server (owner decision, 2026-10-31): no conversation document. What
        // remains is the usage event, which never carries any text. Usage
        // goes to the meter, which may be the Servers API; everything else
        // stays on the sink queue (never an empty batch).
        let (queued, metered) = split_for_delivery(sink, is_pwa(self.req.header(CHANNEL_HEADER)));
        enqueue(p, cfg, &queued, "sink").await;
        let meter = usage_meter(cfg);
        for m in &metered {
            meter.record(p, cfg, &m.doc).await;
        }

        let tier = crate::model_router::tier_name(target.tier);
        let shamwari = shamwari_block(
            json!({
                "tier": tier,
                "reach": reach.map(Reach::as_str),
                "reach_clamped": routed.clamped,
                "truncated": result.finish_reason == "length",
                "scope": a.decision.scope.as_str(),
                "language": language,
                "grounded": g.hit,
            }),
            &a.decision,
            &g,
            json!({
                "cache_hit": result.cache_hit,
                "inference_path": result.path.as_str(),
                "model_served": served,
                // Surfaced rather than buried in the usage event: a dynamic
                // route substituting the model is otherwise invisible to the
                // caller.
                "model_substituted": result.substituted,
                "latency_ms": latency_ms,
                "citations": citations_of(&g.chunks),
            }),
        );
        Ok(Resp::json(
            &json!({
                "id": request_id,
                "model": format!("shamwari-{tier}"),
                "choices": [{
                    "index": 0,
                    "message": { "role": "assistant", "content": result.text },
                    "finish_reason": result.finish_reason,
                }],
                "usage": {
                    "prompt_tokens": result.input_tokens,
                    "completion_tokens": result.output_tokens,
                    "total_tokens": result.input_tokens + result.output_tokens,
                },
                "shamwari": shamwari,
            }),
            200,
        ))
    }
}

/// `POST /v1/responses` (Open Responses): a translator in front of the chat
/// path, called in-process like the channels, so every rule of the chat path
/// applies to it unchanged.
pub async fn handle_open_responses<P: Platform>(p: &P, cfg: &Config, req: &Req) -> Handled {
    let gate = match authenticate(p, cfg, req).await {
        AuthResult::Ok(_) => AuthGate::Ok,
        AuthResult::Unavailable => AuthGate::Unavailable,
        AuthResult::Forbidden(d) => AuthGate::Forbidden(d),
        AuthResult::Invalid => AuthGate::Invalid,
    };
    Ok(handle_responses(p, req, gate, |r| crate::router::dispatch(p, cfg, r)).await)
}

/// `POST /v1/channels/sms` and `POST /v1/channels/ussd`: the operator's
/// webhook, re-dispatched in-process to the chat path with the channel's own
/// key.
pub async fn handle_channel_route<P: Platform>(
    p: &P,
    cfg: &Config,
    req: &Req,
    adapter: &ChannelAdapter,
) -> Handled {
    Ok(handle_channel(adapter, req, cfg, &CHANNEL_TEXT, |r| {
        crate::router::dispatch(p, cfg, r)
    })
    .await)
}

pub async fn handle_sms<P: Platform>(p: &P, cfg: &Config, req: &Req) -> Handled {
    handle_channel_route(p, cfg, req, &SMS).await
}

pub async fn handle_ussd<P: Platform>(p: &P, cfg: &Config, req: &Req) -> Handled {
    handle_channel_route(p, cfg, req, &USSD).await
}

/// The Worker's `queue` handler: drain a `shamwari-sink` batch into Core.
/// What it acknowledges is `acked`; an `Err` fails the batch, so the queue
/// retries what was not acknowledged (five times, then the DLQ).
pub async fn handle_queue<P: Platform>(
    p: &P,
    cfg: &Config,
    batch: &[crate::types::SinkMessage],
) -> crate::sink::Drained {
    crate::sink::drain(p, cfg, batch).await
}

/// The Worker's `scheduled` handler: the weekly degradation probe. It writes
/// through the same queue as everything else, so a Core outage delays the
/// reading rather than losing it.
pub async fn handle_scheduled<P: Platform>(p: &P, cfg: &Config) -> crate::probe::ProbeReport {
    crate::probe::scheduled(p, cfg).await
}
