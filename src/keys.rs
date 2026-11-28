//! Port of `shamwari/gateway/src/keys.ts`: the seam between the gateway and
//! whatever owns API keys and usage.
//!
//! Today that is Core (Mongo `platform.apiKeys` and `platform.usageEvents`).
//! It is expected to move to the Nyuchi Servers API, where keys live in the
//! ledger's `platform.api_keys`. Nothing outside this file knows which one is
//! answering; `SERVERS_API_ENABLED = "true"` switches.
//!
//! Two properties hold for every implementation, and the tests below fail if
//! either breaks:
//!
//! - Auth fails closed. `Unavailable` is never treated as `Ok`, and is never
//!   cached as a verdict either way.
//! - Nothing here carries prompt, answer or scope-gated content. Key checks
//!   send a hash; metering sends counts and identifiers (rule 1). The one
//!   exception is the scope *label*, which [`usage_event_payload`] enforces as
//!   exactly one of the three names and drops otherwise.

use crate::auth::{console, Level};
use crate::config::Config;
use crate::platform::{HttpRequest, HttpResponse, Platform};
use crate::types::{AuthContext, Scope};
use serde_json::{json, Map, Value};

#[derive(Debug, Clone, PartialEq)]
pub enum VerifyOutcome {
    Ok(AuthContext),
    Invalid,
    Unavailable,
}

pub const ALL_SCOPES: [Scope; 3] = [Scope::Personal, Scope::Community, Scope::Platform];
pub const DEFAULT_SCOPES: [Scope; 2] = [Scope::Community, Scope::Platform];

/// The scopes a key may ask for. Personal is never a default.
pub fn permitted_scopes(auth: &AuthContext) -> &[Scope] {
    auth.scopes.as_deref().unwrap_or(&DEFAULT_SCOPES)
}

/// `scopes` from a verify response. `Ok(None)`: absent or null, the default.
/// `Err`: present but not an array of strings, a malformed response. Names
/// outside the three are dropped, so a response can narrow but never invent.
fn parse_key_scopes(v: Option<&Value>) -> Result<Option<Vec<Scope>>, ()> {
    let Some(v) = v.filter(|v| !v.is_null()) else {
        return Ok(None);
    };
    let Some(list) = v.as_array() else {
        return Err(());
    };
    let names: Vec<&str> = list
        .iter()
        .map(Value::as_str)
        .collect::<Option<_>>()
        .ok_or(())?;
    Ok(Some(
        ALL_SCOPES
            .into_iter()
            .filter(|s| names.contains(&s.as_str()))
            .collect(),
    ))
}

/// What a key is issued for decides its type; Shamwari customer keys are this type.
pub const SHAMWARI_KEY_TYPE: &str = "shamwari";

const VERIFY_TIMEOUT_MS: u32 = 800;
const ROLLUP_TIMEOUT_MS: u32 = 800;
const METER_TIMEOUT_MS: u32 = 2000;

/// Fields a usage event may carry to the Servers API. An allowlist, so a field
/// added to the event later (or a content field added by mistake) does not
/// leave the gateway until someone decides it should. Identical to Core's
/// `core/usage_event_fields.json`, which Core replays spilled events with.
pub const USAGE_EVENT_FIELDS: [&str; 26] = [
    "ownerEntityId",
    "apiKeyId",
    "requestId",
    "billingPeriod",
    "scope",
    "tier",
    "routedTier",
    "reach",
    "reachClamped",
    "provider",
    "model",
    "requestedProvider",
    "requestedModel",
    "modelSubstituted",
    "licenseClass",
    "inputTokens",
    "outputTokens",
    "costUsdEstimate",
    "priceTableVersion",
    "cacheHit",
    "groundHit",
    "inferencePath",
    "edgeEmbedding",
    "degraded",
    "latencyMs",
    "createdAt",
];

/// The allowlisted fields of `doc`, in allowlist order.
pub fn usage_event_payload(doc: &Map<String, Value>) -> Map<String, Value> {
    let mut out = Map::new();
    for k in USAGE_EVENT_FIELDS {
        if let Some(v) = doc.get(k) {
            out.insert(k.to_owned(), v.clone());
        }
    }
    // A label, never a value: anything but the three scope names is dropped, so a
    // scope-gated identifier or payload put here by mistake cannot leave.
    let label_ok = out.get("scope").is_none_or(|s| {
        s.as_str()
            .is_some_and(|s| ALL_SCOPES.iter().any(|x| x.as_str() == s))
    });
    if !label_ok {
        out.remove("scope");
    }
    out
}

struct ServersApiConn {
    url: String,
    token: String,
    /// Second factor the internal routes require, when the owner has set one.
    internal_key: Option<String>,
}

fn servers_api(cfg: &Config) -> Option<ServersApiConn> {
    let url = cfg.get(&cfg.servers_api_url)?;
    let token = cfg.get(&cfg.servers_api_token)?;
    Some(ServersApiConn {
        url: url.strip_suffix('/').unwrap_or(url).to_owned(),
        token: token.to_owned(),
        internal_key: cfg.get(&cfg.servers_api_internal_key).map(str::to_owned),
    })
}

/// Bearer token plus, when configured, the internal key the Servers API requires.
fn auth_headers(api: &ServersApiConn) -> Vec<(String, String)> {
    let mut h = vec![("Authorization".to_owned(), format!("Bearer {}", api.token))];
    if let Some(k) = &api.internal_key {
        h.push(("x-nyuchi-internal-key".to_owned(), k.clone()));
    }
    h
}

fn non_empty(v: Option<&Value>) -> Option<&str> {
    v.and_then(Value::as_str).filter(|s| !s.is_empty())
}

/// `Number.isInteger(v) && v > 0`, as the cap a context may carry.
fn positive_integer(v: Option<&Value>) -> Option<u64> {
    let f = v?.as_f64()?;
    (f.is_finite() && f.fract() == 0.0 && f > 0.0).then_some(f as u64)
}

/// A 200 that does not satisfy this is `Unavailable`, never a context we cache.
fn is_auth_context(r: &Map<String, Value>) -> bool {
    non_empty(r.get("api_key_id")).is_some()
        && non_empty(r.get("owner_entity_id")).is_some()
        && non_empty(r.get("key_type")).is_some()
        && non_empty(r.get("tier")).is_some()
        && positive_integer(r.get("monthly_token_cap")).is_some()
}

/// Only a body that says so makes a 401/403/404 the key's fault and not ours.
fn says_invalid_key(res: &HttpResponse) -> bool {
    serde_json::from_str::<Value>(&res.body)
        .ok()
        .and_then(|v| v.get("error").cloned())
        == Some(json!("invalid_key"))
}

fn str_field(r: &Map<String, Value>, k: &str) -> String {
    r.get(k)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned()
}

enum CoreFailure {
    /// Core answered with this status and body (or was skipped while Core's
    /// breaker is open: a 503).
    Status(u16, String),
    /// No answer: a network error, a timeout, an unreadable body, or no `CORE_URL`.
    Other(String),
}

/// A call to Shamwari Core through [`crate::core`], so the key path shares
/// Core's breaker: while Core is down a key check fails at once as
/// `Unavailable` instead of waiting out the timeout on every request.
async fn core_call<P: Platform>(
    p: &P,
    cfg: &Config,
    method: &str,
    path: &str,
    body: Option<&Value>,
) -> Result<Value, CoreFailure> {
    // TypeScript would fetch `undefined/...`, which throws before any request.
    if cfg.core_url.is_none() {
        return Err(CoreFailure::Other("CORE_URL is not set".into()));
    }
    crate::core::core(p, cfg, path, Some(method), body)
        .await
        .map_err(|e| match e {
            crate::core::CoreFailure::Status(e) => CoreFailure::Status(e.status, e.detail),
            crate::core::CoreFailure::Threw(m) => CoreFailure::Other(m),
        })
}

/// Who verifies keys and answers quota rollups.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyService {
    /// Core's `/auth/verify` and `/rollup`. The default.
    Core,
    /// The Nyuchi Servers API. Only with `SERVERS_API_ENABLED = "true"`.
    ServersApi,
}

impl KeyService {
    pub async fn verify<P: Platform>(
        &self,
        p: &P,
        cfg: &Config,
        key_prefix: &str,
        key_hash: &str,
    ) -> VerifyOutcome {
        match self {
            KeyService::Core => core_verify(p, cfg, key_prefix, key_hash).await,
            KeyService::ServersApi => servers_verify(p, cfg, key_prefix, key_hash).await,
        }
    }

    /// Tokens used by an owner in a billing period (`YYYY-MM`). `Err` when it cannot say.
    pub async fn usage_total<P: Platform>(
        &self,
        p: &P,
        cfg: &Config,
        owner_entity_id: &str,
        billing_period: &str,
    ) -> Result<f64, String> {
        match self {
            KeyService::Core => {
                let path = format!(
                    "/rollup?owner_entity_id={owner_entity_id}&billing_period={billing_period}"
                );
                let r = core_call(p, cfg, "GET", &path, None)
                    .await
                    .map_err(|e| match e {
                        CoreFailure::Status(s, _) => format!("core {s}"),
                        CoreFailure::Other(m) => m,
                    })?;
                r.get("total_tokens")
                    .and_then(Value::as_f64)
                    .ok_or_else(|| "core rollup malformed".to_owned())
            }
            KeyService::ServersApi => {
                let api = servers_api(cfg).ok_or("servers api not configured")?;
                let q = url::form_urlencoded::Serializer::new(String::new())
                    .append_pair("owner_entity_id", owner_entity_id)
                    .append_pair("billing_period", billing_period)
                    .finish();
                let res = p
                    .fetch(HttpRequest {
                        method: "GET".into(),
                        url: format!("{}/v1/shamwari/usage/rollup?{q}", api.url),
                        headers: auth_headers(&api),
                        body: None,
                        timeout_ms: Some(ROLLUP_TIMEOUT_MS),
                    })
                    .await?;
                if !res.ok() {
                    return Err(format!("servers api rollup {}", res.status));
                }
                serde_json::from_str::<Value>(&res.body)
                    .ok()
                    .and_then(|r| r.get("total_tokens").and_then(Value::as_f64))
                    .ok_or_else(|| "servers api rollup malformed".to_owned())
            }
        }
    }
}

async fn core_verify<P: Platform>(
    p: &P,
    cfg: &Config,
    key_prefix: &str,
    key_hash: &str,
) -> VerifyOutcome {
    // keyPrefix is indexed in platform.apiKeys; Core finds by prefix then does
    // a constant-time compare on the hash.
    let body = json!({ "key_prefix": key_prefix, "key_hash": key_hash });
    match core_call(p, cfg, "POST", "/auth/verify", Some(&body)).await {
        Ok(r) => {
            // TypeScript trusts Core's shape; a context that cannot be built is
            // not one to cache, so it is unavailable here.
            let Some(o) = r.as_object() else {
                return VerifyOutcome::Unavailable;
            };
            let s = |k: &str| o.get(k).and_then(Value::as_str).map(str::to_owned);
            let cap = o
                .get("monthly_token_cap")
                .and_then(Value::as_f64)
                .filter(|f| f.is_finite() && f.fract() == 0.0 && *f >= 0.0);
            match (
                s("api_key_id"),
                s("owner_entity_id"),
                s("key_type"),
                s("tier"),
                cap,
            ) {
                (
                    Some(api_key_id),
                    Some(owner_entity_id),
                    Some(key_type),
                    Some(tier),
                    Some(cap),
                ) => VerifyOutcome::Ok(AuthContext {
                    api_key_id,
                    owner_entity_id,
                    owner_person_id: s("owner_person_id"),
                    key_type,
                    tier,
                    monthly_token_cap: cap as u64,
                    scopes: None,
                    training_consent: None,
                }),
                _ => VerifyOutcome::Unavailable,
            }
        }
        // Core answers a bad key with 401 {"detail":{"error":"invalid_key"}}. Only
        // that is a verdict on the key, and only that is cached. Anything else (Core
        // down, a timeout, a 5xx, Core refusing the Worker's own token) says nothing
        // about the key: it is unavailable, denies the request, and is not cached.
        Err(CoreFailure::Status(401, detail)) if detail.contains("\"invalid_key\"") => {
            VerifyOutcome::Invalid
        }
        Err(_) => VerifyOutcome::Unavailable,
    }
}

async fn servers_verify<P: Platform>(
    p: &P,
    cfg: &Config,
    key_prefix: &str,
    key_hash: &str,
) -> VerifyOutcome {
    // Enabled but unconfigured: closed, not open.
    let Some(api) = servers_api(cfg) else {
        return VerifyOutcome::Unavailable;
    };
    let mut headers = auth_headers(&api);
    headers.push(("Content-Type".to_owned(), "application/json".to_owned()));
    let Ok(res) = p
        .fetch(HttpRequest {
            method: "POST".into(),
            url: format!("{}/v1/shamwari/keys/verify", api.url),
            headers,
            body: Some(json!({ "key_prefix": key_prefix, "key_hash": key_hash }).to_string()),
            timeout_ms: Some(VERIFY_TIMEOUT_MS),
        })
        .await
    else {
        return VerifyOutcome::Unavailable;
    };
    // A definitive "no" needs the body to say so. A bare 401/403/404 is as likely
    // our own rotated SERVERS_API_TOKEN or a wrong route, and caching that as
    // "invalid" would reject every good key for the TTL.
    if [401, 403, 404].contains(&res.status) {
        return if says_invalid_key(&res) {
            VerifyOutcome::Invalid
        } else {
            VerifyOutcome::Unavailable
        };
    }
    if res.status != 200 {
        return VerifyOutcome::Unavailable;
    }
    let Ok(Value::Object(r)) = serde_json::from_str::<Value>(&res.body) else {
        return VerifyOutcome::Unavailable;
    };
    if !is_auth_context(&r) {
        return VerifyOutcome::Unavailable;
    }
    // The shared key table holds every Nyuchi product's keys. A key issued for
    // something else is a valid key that is not valid here.
    if r.get("key_type").and_then(Value::as_str) != Some(SHAMWARI_KEY_TYPE) {
        return VerifyOutcome::Invalid;
    }
    let Ok(scopes) = parse_key_scopes(r.get("scopes")) else {
        return VerifyOutcome::Unavailable;
    };
    VerifyOutcome::Ok(AuthContext {
        api_key_id: str_field(&r, "api_key_id"),
        owner_entity_id: str_field(&r, "owner_entity_id"),
        owner_person_id: r
            .get("owner_person_id")
            .and_then(Value::as_str)
            .map(str::to_owned),
        key_type: str_field(&r, "key_type"),
        tier: str_field(&r, "tier"),
        monthly_token_cap: positive_integer(r.get("monthly_token_cap")).unwrap_or_default(),
        scopes,
        training_consent: None,
    })
}

/// Where usage events go. `record` never fails a user request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UsageMeter {
    /// The `shamwari-sink` queue, drained into Core. `spilled` marks an event
    /// the Servers API did not take: Core's replay job (core/replay.py)
    /// delivers exactly the rows carrying it, and nothing else.
    Queue { spilled: bool },
    /// The Servers API, falling back to the queue (spilled) so an outage there
    /// does not lose the event. A timeout after the Servers API accepted can
    /// spill a duplicate; the replay's `requestId` dedup makes that harmless.
    ServersApi,
}

impl UsageMeter {
    pub async fn record<P: Platform>(&self, p: &P, cfg: &Config, doc: &Map<String, Value>) {
        match self {
            UsageMeter::Queue { spilled } => queue_record(p, cfg, doc, *spilled).await,
            UsageMeter::ServersApi => {
                if let Err(e) = servers_record(p, cfg, doc).await {
                    console(
                        Level::Warn,
                        &format!("servers api metering failed, queueing {e}"),
                    );
                    queue_record(p, cfg, doc, true).await;
                }
            }
        }
    }
}

async fn queue_record<P: Platform>(p: &P, cfg: &Config, doc: &Map<String, Value>, spilled: bool) {
    // Until the cutover this Worker must not write to shamwari-sink
    // (SINK_ENABLED = "false"), so parity traffic never double-writes Core.
    if !cfg.sink_enabled() {
        return;
    }
    let mut doc = doc.clone();
    if spilled {
        doc.insert("spilled".into(), Value::Bool(true));
    }
    let msg = json!({ "database": "platform", "collection": "usageEvents", "doc": doc });
    if let Err(e) = p.queue_send(&msg).await {
        console(Level::Error, &format!("sink enqueue failed {e}"));
    }
}

async fn servers_record<P: Platform>(
    p: &P,
    cfg: &Config,
    doc: &Map<String, Value>,
) -> Result<(), String> {
    let api = servers_api(cfg).ok_or("servers api not configured")?;
    let mut headers = auth_headers(&api);
    headers.push(("Content-Type".to_owned(), "application/json".to_owned()));
    let res = p
        .fetch(HttpRequest {
            method: "POST".into(),
            url: format!("{}/v1/shamwari/usage/events", api.url),
            headers,
            body: Some(json!({ "events": [usage_event_payload(doc)] }).to_string()),
            timeout_ms: Some(METER_TIMEOUT_MS),
        })
        .await?;
    if !res.ok() {
        return Err(format!("servers api usage {}", res.status));
    }
    Ok(())
}

pub fn servers_api_enabled(cfg: &Config) -> bool {
    cfg.servers_api_enabled.as_deref() == Some("true")
}

pub fn key_service(cfg: &Config) -> KeyService {
    if servers_api_enabled(cfg) {
        KeyService::ServersApi
    } else {
        KeyService::Core
    }
}

pub fn usage_meter(cfg: &Config) -> UsageMeter {
    if servers_api_enabled(cfg) {
        UsageMeter::ServersApi
    } else {
        UsageMeter::Queue { spilled: false }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::fakes::Rec;
    use crate::auth::{auth_failure, authenticate, sha256_hex, take_logs, AuthResult};
    use crate::http::Req;
    use crate::testing::block_on;
    use std::cell::Cell;
    use std::rc::Rc;

    const RAW: &str = "sk_shamwari_abcdefghijklmnop";
    const VERIFY: &str = "https://servers.example/v1/shamwari/keys/verify";
    const EVENTS: &str = "https://servers.example/v1/shamwari/usage/events";
    const ROLLUP: &str = "https://servers.example/v1/shamwari/usage/rollup";

    fn good() -> Value {
        json!({
            "api_key_id": "k1", "owner_entity_id": "e1", "owner_person_id": null,
            "key_type": "shamwari", "tier": "community", "monthly_token_cap": 100_000
        })
    }

    fn good_with(k: &str, v: Value) -> String {
        let mut g = good();
        if v.is_null() {
            g.as_object_mut().unwrap().remove(k);
        } else {
            g[k] = v;
        }
        g.to_string()
    }

    fn base() -> Config {
        Config {
            core_url: Some("https://core.example".into()),
            shamwari_core_token: Some("core-token".into()),
            auth_cache_ttl: Some("60".into()),
            ..Default::default()
        }
    }

    fn servers() -> Config {
        Config {
            servers_api_enabled: Some("true".into()),
            servers_api_url: Some("https://servers.example/".into()),
            servers_api_token: Some("servers-token".into()),
            ..base()
        }
    }

    fn req() -> Req {
        Req {
            method: "POST".into(),
            url: "https://x/v1/chat/completions".into(),
            headers: vec![("authorization".into(), format!("Bearer {RAW}"))],
            body: String::new(),
        }
    }

    fn header<'a>(r: &'a HttpRequest, name: &str) -> Option<&'a str> {
        r.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }

    fn auth(p: &Rec, cfg: &Config) -> AuthResult {
        block_on(authenticate(p, cfg, &req()))
    }

    fn cached_values(p: &Rec) -> Vec<String> {
        p.kv.borrow().values().cloned().collect()
    }

    fn doc(v: Value) -> Map<String, Value> {
        v.as_object().unwrap().clone()
    }

    // ---- default implementation

    #[test]
    fn is_core_and_the_queue_unless_the_flag_is_exactly_true() {
        for flag in [None, Some(""), Some("false"), Some("TRUE"), Some("1")] {
            let cfg = Config {
                servers_api_enabled: flag.map(Into::into),
                ..servers()
            };
            assert_eq!(key_service(&cfg), KeyService::Core);
            assert_eq!(usage_meter(&cfg), UsageMeter::Queue { spilled: false });
        }
        assert_eq!(key_service(&servers()), KeyService::ServersApi);
        assert_eq!(usage_meter(&servers()), UsageMeter::ServersApi);
    }

    #[test]
    fn verifies_against_core_and_never_touches_the_servers_api() {
        let p = Rec::new();
        p.on("https://", 200, &good().to_string());
        let r = auth(&p, &base());
        let AuthResult::Ok(ctx) = r else {
            panic!("expected ok, got {r:?}")
        };
        assert_eq!(
            (ctx.api_key_id.as_str(), ctx.owner_entity_id.as_str()),
            ("k1", "e1")
        );
        let reqs = p.requests.borrow();
        assert_eq!(
            reqs.iter().map(|r| r.url.as_str()).collect::<Vec<_>>(),
            ["https://core.example/auth/verify"]
        );
        assert_eq!(header(&reqs[0], "authorization"), Some("Bearer core-token"));
        assert_eq!(reqs[0].timeout_ms, Some(crate::core::CORE_TIMEOUT_MS));
        let body: Value = serde_json::from_str(reqs[0].body.as_deref().unwrap()).unwrap();
        assert_eq!(
            body,
            json!({ "key_prefix": "sk_shamwari_", "key_hash": sha256_hex(RAW) })
        );
    }

    #[test]
    fn a_core_outage_is_unavailable_and_is_not_cached_as_a_verdict_on_the_key() {
        let p = Rec::new();
        p.on("https://", 500, "boom");
        assert_eq!(auth(&p, &base()), AuthResult::Unavailable);
        assert!(cached_values(&p).is_empty());
        // And a network error, and no CORE_URL at all.
        let p = Rec::new();
        assert_eq!(auth(&p, &base()), AuthResult::Unavailable);
        let cfg = Config {
            core_url: None,
            ..base()
        };
        assert_eq!(auth(&p, &cfg), AuthResult::Unavailable);
        assert!(cached_values(&p).is_empty());
    }

    #[test]
    fn core_refusing_the_workers_own_token_is_unavailable_not_a_bad_key() {
        let p = Rec::new();
        p.on("https://", 401, r#"{"detail":"bad core token"}"#);
        assert_eq!(auth(&p, &base()), AuthResult::Unavailable);
        assert!(cached_values(&p).is_empty());
    }

    #[test]
    fn a_bad_key_is_invalid_and_the_negative_is_cached() {
        let p = Rec::new();
        p.on("https://", 401, r#"{"detail":{"error":"invalid_key"}}"#);
        assert_eq!(auth(&p, &base()), AuthResult::Invalid);
        assert_eq!(cached_values(&p), ["\"invalid\""]);
    }

    #[test]
    fn a_core_200_that_cannot_be_a_context_is_unavailable() {
        for body in ["<html>", "null", r#"{"api_key_id":"k1"}"#] {
            let p = Rec::new();
            p.on("https://", 200, body);
            assert_eq!(auth(&p, &base()), AuthResult::Unavailable, "{body}");
            assert!(cached_values(&p).is_empty());
        }
    }

    #[test]
    fn meters_through_the_queue() {
        let p = Rec::new();
        block_on(usage_meter(&base()).record(&p, &base(), &doc(json!({ "requestId": "r1" }))));
        assert_eq!(
            *p.queue.borrow(),
            [json!({
                "database": "platform", "collection": "usageEvents",
                "doc": { "requestId": "r1" }
            })]
        );
    }

    #[test]
    fn sink_enabled_false_keeps_this_worker_off_the_queue() {
        let p = Rec::new();
        let cfg = Config {
            sink_enabled: Some("false".into()),
            ..base()
        };
        block_on(usage_meter(&cfg).record(&p, &cfg, &doc(json!({ "requestId": "r1" }))));
        assert!(p.queue.borrow().is_empty());
    }

    #[test]
    fn a_queue_failure_is_logged_not_raised() {
        let p = Rec::new();
        p.faults.queue.set(true);
        block_on(usage_meter(&base()).record(&p, &base(), &doc(json!({ "requestId": "r1" }))));
        assert!(take_logs()
            .iter()
            .any(|l| l.contains("sink enqueue failed")));
    }

    // ---- Servers API key verification

    #[test]
    fn sends_only_the_prefix_and_the_hash_to_the_contract_path_with_the_service_token() {
        let p = Rec::new();
        p.on(VERIFY, 200, &good().to_string());
        let r = auth(&p, &servers());
        let AuthResult::Ok(ctx) = r else {
            panic!("expected ok, got {r:?}")
        };
        assert_eq!(
            (ctx.tier.as_str(), ctx.monthly_token_cap),
            ("community", 100_000)
        );
        let reqs = p.requests.borrow();
        assert_eq!(reqs.len(), 1);
        assert_eq!(reqs[0].url, VERIFY);
        assert_eq!(reqs[0].method, "POST");
        assert_eq!(
            header(&reqs[0], "authorization"),
            Some("Bearer servers-token")
        );
        assert_eq!(reqs[0].timeout_ms, Some(VERIFY_TIMEOUT_MS));
        let body: Value = serde_json::from_str(reqs[0].body.as_deref().unwrap()).unwrap();
        let mut keys: Vec<_> = body.as_object().unwrap().keys().cloned().collect();
        keys.sort();
        assert_eq!(keys, ["key_hash", "key_prefix"]);
        assert!(!body.to_string().contains(RAW));
    }

    #[test]
    fn a_401_403_404_saying_invalid_key_is_a_definitive_invalid_cached_as_one() {
        for status in [401, 403, 404] {
            let p = Rec::new();
            p.on(VERIFY, status, r#"{"error":"invalid_key"}"#);
            assert_eq!(auth(&p, &servers()), AuthResult::Invalid, "{status}");
            assert_eq!(cached_values(&p), ["\"invalid\""]);
        }
    }

    #[test]
    fn a_bare_401_403_404_is_unavailable_so_good_keys_are_not_poisoned() {
        for status in [401, 403, 404] {
            let p = Rec::new();
            p.on(VERIFY, status, "nope");
            assert_eq!(auth(&p, &servers()), AuthResult::Unavailable, "{status}");
            assert!(cached_values(&p).is_empty());
        }
    }

    #[test]
    fn fails_closed_on_every_outage_unavailable_not_ok_and_nothing_cached() {
        let outages: Vec<(&str, Option<(u16, String)>)> = vec![
            ("500", Some((500, "x".into()))),
            ("502", Some((502, "x".into()))),
            ("429", Some((429, "x".into()))),
            ("a thrown network error", None),
            ("a 200 that is not JSON", Some((200, "<html>".into()))),
            (
                "a 200 missing required fields",
                Some((200, r#"{"api_key_id":"k1"}"#.into())),
            ),
            (
                "a 200 with no key_type",
                Some((200, good_with("key_type", Value::Null))),
            ),
            (
                "a 200 with a zero cap",
                Some((200, good_with("monthly_token_cap", json!(0)))),
            ),
            (
                "a 200 with an empty owner",
                Some((200, good_with("owner_entity_id", json!("")))),
            ),
            (
                "a 200 with a string cap",
                Some((200, good_with("monthly_token_cap", json!("100000")))),
            ),
            (
                "a 200 with a fractional cap",
                Some((200, good_with("monthly_token_cap", json!(1.5)))),
            ),
            ("a 200 that is null", Some((200, "null".into()))),
        ];
        for (name, answer) in outages {
            let p = Rec::new();
            if let Some((status, body)) = answer {
                p.on(VERIFY, status, &body);
            }
            assert_eq!(auth(&p, &servers()), AuthResult::Unavailable, "{name}");
            assert!(cached_values(&p).is_empty(), "{name}");
        }
        // A timeout.
        let p = Rec::new();
        p.on_fn(|_| Some(Err("timeout".into())));
        assert_eq!(auth(&p, &servers()), AuthResult::Unavailable);
        assert!(cached_values(&p).is_empty());
    }

    #[test]
    fn fails_closed_when_enabled_but_unconfigured_and_does_not_fall_back_to_core() {
        let p = Rec::new();
        p.on("https://", 200, &good().to_string());
        let cfg = Config {
            servers_api_enabled: Some("true".into()),
            ..base()
        };
        assert_eq!(auth(&p, &cfg), AuthResult::Unavailable);
        assert!(p.requests.borrow().is_empty());
    }

    #[test]
    fn a_valid_key_of_another_type_is_invalid_here_and_is_cached_as_invalid() {
        let p = Rec::new();
        p.on(VERIFY, 200, &good_with("key_type", json!("nyuchi-api")));
        assert_eq!(auth(&p, &servers()), AuthResult::Invalid);
        assert_eq!(cached_values(&p), ["\"invalid\""]);
    }

    #[test]
    fn key_type_must_match_exactly() {
        for kt in [
            json!("Shamwari"),
            json!(" shamwari"),
            json!("shamwari "),
            json!(["shamwari"]),
            json!(123),
        ] {
            let p = Rec::new();
            p.on(VERIFY, 200, &good_with("key_type", kt.clone()));
            assert!(!matches!(auth(&p, &servers()), AuthResult::Ok(_)), "{kt}");
        }
    }

    #[test]
    fn ignores_a_returned_scope_label_key_data_never_decides_the_request_scope() {
        let p = Rec::new();
        p.on(VERIFY, 200, &good_with("scope_label", json!("personal")));
        let AuthResult::Ok(ctx) = auth(&p, &servers()) else {
            panic!("expected ok")
        };
        let v = serde_json::to_value(&ctx).unwrap();
        assert!(v.get("scope_label").is_none());
        assert_eq!(ctx.scopes, None);
    }

    #[test]
    fn sends_the_internal_key_second_factor_on_every_route_when_configured_and_not_otherwise() {
        let p = Rec::new();
        p.on(VERIFY, 200, &good().to_string())
            .on(ROLLUP, 200, r#"{"total_tokens":1}"#)
            .on(EVENTS, 200, "{}");
        let with_key = Config {
            servers_api_internal_key: Some("nyk_a.nys_b".into()),
            ..servers()
        };
        block_on(KeyService::ServersApi.verify(&p, &with_key, "sk_shamwari_", "h"));
        block_on(KeyService::ServersApi.usage_total(&p, &with_key, "e1", "2026-10")).unwrap();
        block_on(UsageMeter::ServersApi.record(&p, &with_key, &doc(json!({ "requestId": "r" }))));
        let keys: Vec<_> = p
            .requests
            .borrow()
            .iter()
            .map(|r| header(r, "x-nyuchi-internal-key").map(str::to_owned))
            .collect();
        assert_eq!(keys, vec![Some("nyk_a.nys_b".to_owned()); 3]);

        p.requests.borrow_mut().clear();
        block_on(KeyService::ServersApi.verify(&p, &servers(), "sk_shamwari_", "h"));
        assert_eq!(
            header(&p.requests.borrow()[0], "x-nyuchi-internal-key"),
            None
        );
    }

    #[test]
    fn a_recovered_service_is_believed_at_once() {
        let p = Rec::new();
        let up = Rc::new(Cell::new(false));
        let u = up.clone();
        p.on_fn(move |_| {
            Some(Ok(if u.get() {
                HttpResponse {
                    status: 200,
                    headers: vec![],
                    body: good().to_string(),
                }
            } else {
                HttpResponse {
                    status: 503,
                    headers: vec![],
                    body: "x".into(),
                }
            }))
        });
        assert!(!matches!(auth(&p, &servers()), AuthResult::Ok(_)));
        up.set(true);
        assert!(matches!(auth(&p, &servers()), AuthResult::Ok(_)));
    }

    #[test]
    fn a_failing_kv_write_does_not_deny_a_verified_key() {
        let p = Rec::new();
        p.on(VERIFY, 200, &good().to_string());
        p.faults.kv_put.set(true);
        assert!(matches!(auth(&p, &servers()), AuthResult::Ok(_)));
    }

    #[test]
    fn answers_from_the_kv_cache_without_calling_the_service() {
        let p = Rec::new();
        p.on(VERIFY, 200, &good().to_string());
        let first = auth(&p, &servers());
        let second = auth(&p, &servers());
        assert_eq!(p.requests.borrow().len(), 1);
        assert_eq!(first, second);
    }

    // ---- through the failure response

    #[test]
    fn a_servers_api_outage_is_503_auth_unavailable_with_retry_after() {
        let p = Rec::new();
        p.on(VERIFY, 500, "x");
        let res = auth_failure(&auth(&p, &servers()));
        assert_eq!(res.status, 503);
        assert_eq!(res.body_json(), json!({ "error": "auth_unavailable" }));
        assert_eq!(res.header("retry-after"), Some("5"));
    }

    #[test]
    fn an_invalid_key_is_401() {
        let p = Rec::new();
        p.on(VERIFY, 404, r#"{"error":"invalid_key"}"#);
        assert_eq!(auth_failure(&auth(&p, &servers())).status, 401);
    }

    // ---- Servers API quota rollup

    #[test]
    fn rollup_returns_the_total_and_errs_when_it_cannot_say() {
        let p = Rec::new();
        p.on(ROLLUP, 200, r#"{"total_tokens":42}"#);
        let svc = KeyService::ServersApi;
        assert_eq!(
            block_on(svc.usage_total(&p, &servers(), "e1", "2026-10")),
            Ok(42.0)
        );
        let r = &p.requests.borrow()[0];
        assert_eq!(r.method, "GET");
        assert_eq!(
            r.url,
            format!("{ROLLUP}?owner_entity_id=e1&billing_period=2026-10")
        );
        assert_eq!(r.timeout_ms, Some(ROLLUP_TIMEOUT_MS));

        for (status, body) in [(500, "x"), (200, "{}"), (200, "<html>")] {
            let p = Rec::new();
            p.on(ROLLUP, status, body);
            assert!(block_on(svc.usage_total(&p, &servers(), "e1", "2026-10")).is_err());
        }
        let unconfigured = Config {
            servers_api_url: None,
            ..servers()
        };
        assert!(block_on(svc.usage_total(&Rec::new(), &unconfigured, "e1", "2026-10")).is_err());
    }

    #[test]
    fn core_rollup_interpolates_the_owner_and_period_into_the_path() {
        let p = Rec::new();
        p.on("https://core.example/rollup", 200, r#"{"total_tokens":7}"#);
        let total = block_on(KeyService::Core.usage_total(&p, &base(), "person_abc123", "2026-01"));
        assert_eq!(total, Ok(7.0));
        assert_eq!(
            p.requests.borrow()[0].url,
            "https://core.example/rollup?owner_entity_id=person_abc123&billing_period=2026-01"
        );
    }

    // ---- spill replay contract with Core

    #[test]
    fn the_allowlist_is_identical_to_cores_usage_event_fields() {
        // core/usage_event_fields.json at the time of the port; Core replays with it.
        let mut core = vec![
            "ownerEntityId",
            "apiKeyId",
            "requestId",
            "billingPeriod",
            "scope",
            "tier",
            "routedTier",
            "reach",
            "reachClamped",
            "provider",
            "model",
            "requestedProvider",
            "requestedModel",
            "modelSubstituted",
            "licenseClass",
            "inputTokens",
            "outputTokens",
            "costUsdEstimate",
            "priceTableVersion",
            "cacheHit",
            "groundHit",
            "inferencePath",
            "edgeEmbedding",
            "degraded",
            "latencyMs",
            "createdAt",
        ];
        let mut ours = USAGE_EVENT_FIELDS.to_vec();
        core.sort_unstable();
        ours.sort_unstable();
        assert_eq!(ours, core);
    }

    #[test]
    fn the_default_meter_does_not_mark_events_spilled_only_the_fallback_does() {
        let p = Rec::new();
        block_on(UsageMeter::Queue { spilled: false }.record(
            &p,
            &base(),
            &doc(json!({ "requestId": "r" })),
        ));
        assert!(p.queue.borrow()[0]["doc"].get("spilled").is_none());
    }

    // ---- Servers API usage metering

    fn event() -> Map<String, Value> {
        doc(json!({
            "requestId": "r1", "ownerEntityId": "e1", "apiKeyId": "k1",
            "billingPeriod": "2026-10", "inputTokens": 10, "outputTokens": 5,
            "createdAt": "2026-10-29T00:00:00Z"
        }))
    }

    #[test]
    fn posts_an_idempotent_batch_and_does_not_queue_when_accepted() {
        let p = Rec::new();
        p.on(EVENTS, 200, r#"{"accepted":1}"#);
        block_on(UsageMeter::ServersApi.record(&p, &servers(), &event()));
        let reqs = p.requests.borrow();
        assert_eq!(reqs[0].url, EVENTS);
        assert_eq!(reqs[0].timeout_ms, Some(METER_TIMEOUT_MS));
        let body: Value = serde_json::from_str(reqs[0].body.as_deref().unwrap()).unwrap();
        assert_eq!(body, json!({ "events": [Value::Object(event())] }));
        assert!(p.queue.borrow().is_empty());
    }

    #[test]
    fn queues_the_event_spilled_instead_of_dropping_it_on_a_500_or_a_network_error() {
        for fails in [true, false] {
            let p = Rec::new();
            if fails {
                p.on(EVENTS, 500, "x");
            }
            block_on(UsageMeter::ServersApi.record(&p, &servers(), &event()));
            let mut spilled = event();
            spilled.insert("spilled".into(), json!(true));
            // `spilled` is what Core's replay job selects on.
            assert_eq!(
                *p.queue.borrow(),
                [json!({ "database": "platform", "collection": "usageEvents", "doc": spilled })]
            );
        }
    }

    #[test]
    fn rule_1_scope_is_a_label_only() {
        for scope in [
            json!({ "pod": "x", "data": "my clinic results" }),
            json!("pod:abc123"),
            json!("my clinic results"),
            json!(""),
        ] {
            let out = usage_event_payload(&doc(json!({ "requestId": "r1", "scope": scope })));
            assert_eq!(Value::Object(out.clone()), json!({ "requestId": "r1" }));
            assert!(!Value::Object(out).to_string().contains("clinic"));
        }
    }

    #[test]
    fn rule_1_every_key_on_the_wire_is_on_the_allowlist() {
        let p = Rec::new();
        p.on(EVENTS, 200, "{}");
        let junk = [
            "messages",
            "content",
            "prompt",
            "completion",
            "answer",
            "text",
            "citations",
            "systemPrompt",
            "embedding",
            "newField",
        ];
        let mut d = event();
        for k in junk {
            d.insert(k.into(), json!("x"));
        }
        block_on(UsageMeter::ServersApi.record(&p, &servers(), &d));
        let body: Value =
            serde_json::from_str(p.requests.borrow()[0].body.as_deref().unwrap()).unwrap();
        let sent: Vec<String> = body["events"][0]
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect();
        for k in &sent {
            assert!(USAGE_EVENT_FIELDS.contains(&k.as_str()), "{k}");
        }
        for k in junk {
            assert!(!sent.iter().any(|s| s == k), "{k}");
        }
    }

    #[test]
    fn rule_1_carries_no_prompt_answer_or_scope_gated_content() {
        let p = Rec::new();
        p.on(EVENTS, 200, "{}");
        let mut d = event();
        for (k, v) in [
            (
                "messages",
                json!([{ "role": "user", "content": "my clinic results" }]),
            ),
            ("content", json!("secret")),
            ("prompt", json!("secret")),
            ("answer", json!("secret")),
            ("scope", json!("personal")),
            ("systemPrompt", json!("secret")),
        ] {
            d.insert(k.into(), v);
        }
        block_on(UsageMeter::ServersApi.record(&p, &servers(), &d));
        let sent = p.requests.borrow()[0].body.clone().unwrap();
        assert!(!sent.contains("secret") && !sent.contains("clinic"));
        let v: Value = serde_json::from_str(&sent).unwrap();
        assert!(v["events"][0].get("messages").is_none());
        // The scope *label* is allowed; the scope word is not what must stay out.
        assert_eq!(v["events"][0]["scope"], "personal");
        let only = usage_event_payload(&doc(
            json!({ "messages": 1, "content": 1, "requestId": "r" }),
        ));
        assert_eq!(only.keys().collect::<Vec<_>>(), ["requestId"]);
    }

    // ---- key scopes

    #[test]
    fn absent_scopes_mean_community_and_platform_personal_is_never_a_default() {
        let ctx = AuthContext {
            api_key_id: "k".into(),
            owner_entity_id: "e".into(),
            owner_person_id: None,
            key_type: "shamwari".into(),
            tier: "community".into(),
            monthly_token_cap: 1,
            scopes: None,
            training_consent: None,
        };
        assert_eq!(permitted_scopes(&ctx), [Scope::Community, Scope::Platform]);
    }

    #[test]
    fn scopes_from_the_servers_api_are_kept_filtered_to_the_three_names_never_invented() {
        let p = Rec::new();
        p.on(
            VERIFY,
            200,
            &good_with("scopes", json!(["personal", "platform", "admin", "ALL"])),
        );
        let AuthResult::Ok(ctx) = auth(&p, &servers()) else {
            panic!("expected ok")
        };
        assert_eq!(ctx.scopes, Some(vec![Scope::Personal, Scope::Platform]));
    }

    #[test]
    fn malformed_scopes_are_unavailable_and_nothing_is_cached() {
        for scopes in [
            json!("personal"),
            json!({ "personal": true }),
            json!([1, 2]),
        ] {
            let p = Rec::new();
            p.on(VERIFY, 200, &good_with("scopes", scopes.clone()));
            assert_eq!(auth(&p, &servers()), AuthResult::Unavailable, "{scopes}");
            assert!(cached_values(&p).is_empty());
        }
    }

    #[test]
    fn a_null_scopes_means_the_default_not_an_outage() {
        let mut g = good();
        g["scopes"] = Value::Null;
        let p = Rec::new();
        p.on(VERIFY, 200, &g.to_string());
        let AuthResult::Ok(ctx) = auth(&p, &servers()) else {
            panic!("expected ok")
        };
        assert_eq!(ctx.scopes, None);
    }

    #[test]
    fn an_explicit_empty_list_stays_empty_that_key_may_ask_for_nothing() {
        let p = Rec::new();
        p.on(VERIFY, 200, &good_with("scopes", json!([])));
        let AuthResult::Ok(ctx) = auth(&p, &servers()) else {
            panic!("expected ok")
        };
        assert_eq!(ctx.scopes, Some(vec![]));
        assert!(permitted_scopes(&ctx).is_empty());
    }

    // ---- least privilege

    #[test]
    fn every_call_the_gateway_can_make_goes_to_one_of_three_routes_with_the_right_method() {
        let p = Rec::new();
        p.on(VERIFY, 200, &good().to_string())
            .on(ROLLUP, 200, r#"{"total_tokens":1}"#)
            .on(EVENTS, 200, "{}");
        let cfg = Config {
            servers_api_internal_key: Some("nyk_a.nys_b".into()),
            ..servers()
        };
        block_on(KeyService::ServersApi.verify(&p, &cfg, "sk_shamwari_", "h"));
        block_on(KeyService::ServersApi.usage_total(&p, &cfg, "e1", "2026-10")).unwrap();
        block_on(UsageMeter::ServersApi.record(&p, &cfg, &doc(json!({ "requestId": "r" }))));
        let mut calls: Vec<String> = p
            .requests
            .borrow()
            .iter()
            .map(|r| format!("{} {}", r.method, url::Url::parse(&r.url).unwrap().path()))
            .collect();
        calls.sort();
        assert_eq!(
            calls,
            [
                "GET /v1/shamwari/usage/rollup",
                "POST /v1/shamwari/keys/verify",
                "POST /v1/shamwari/usage/events",
            ]
        );
    }

    #[test]
    fn the_source_names_no_issuing_revoking_listing_or_deleting_route() {
        let src = include_str!("keys.rs");
        let code = src.split("#[cfg(test)]\nmod tests").next().unwrap();
        let mut paths: Vec<&str> = code
            .match_indices("/v1/shamwari/")
            .map(|(i, _)| {
                let rest = &code[i..];
                let end = rest
                    .find(|c: char| !(c.is_ascii_alphanumeric() || "_/-".contains(c)))
                    .unwrap_or(rest.len());
                &rest[..end]
            })
            .collect();
        paths.sort_unstable();
        paths.dedup();
        assert_eq!(
            paths,
            [
                "/v1/shamwari/keys/verify",
                "/v1/shamwari/usage/events",
                "/v1/shamwari/usage/rollup",
            ]
        );
    }

    // ---- no secret in logs, errors or usage events

    fn with_secrets() -> Config {
        Config {
            servers_api_token: Some("tok-SECRET-1".into()),
            servers_api_internal_key: Some("nyk_id.nys_SECRET-2".into()),
            shamwari_core_token: Some("core-SECRET-3".into()),
            ..servers()
        }
    }
    const SECRETS: [&str; 3] = ["tok-SECRET-1", "nyk_id.nys_SECRET-2", "core-SECRET-3"];

    #[test]
    fn failures_of_every_kind_never_print_a_credential_the_raw_key_or_its_hash() {
        take_logs();
        let cfg = with_secrets();
        for verify in [Some((500, "x")), None, Some((200, "<html>"))] {
            let p = Rec::new();
            if let Some((s, b)) = verify {
                p.on(VERIFY, s, b);
            }
            p.on(EVENTS, 500, "x");
            auth(&p, &cfg);
            block_on(UsageMeter::ServersApi.record(&p, &cfg, &doc(json!({ "requestId": "r1" }))));
        }
        let all = take_logs().join("\n");
        assert!(!all.is_empty());
        for v in SECRETS.iter().chain([&RAW]) {
            assert!(!all.contains(v), "{v}");
        }
        let hex_run = all
            .split(|c: char| !c.is_ascii_hexdigit())
            .any(|w| w.len() >= 64);
        assert!(!hex_run, "no full digest either");
    }

    #[test]
    fn a_usage_event_carries_none_of_the_credentials() {
        let p = Rec::new();
        p.on(EVENTS, 200, "{}");
        block_on(UsageMeter::ServersApi.record(
            &p,
            &with_secrets(),
            &doc(json!({ "requestId": "r", "apiKeyId": "k1" })),
        ));
        let body = p.requests.borrow()[0].body.clone().unwrap();
        for v in SECRETS {
            assert!(!body.contains(v));
        }
    }

    #[test]
    fn the_credentials_are_only_ever_sent_as_headers_never_in_a_url_or_a_body() {
        let p = Rec::new();
        p.on(VERIFY, 200, &good().to_string())
            .on(ROLLUP, 200, r#"{"total_tokens":1}"#)
            .on("https://core.example", 200, &good().to_string());
        let cfg = with_secrets();
        block_on(KeyService::ServersApi.verify(&p, &cfg, "sk_shamwari_", "h"));
        block_on(KeyService::ServersApi.usage_total(&p, &cfg, "e1", "2026-10")).unwrap();
        block_on(KeyService::Core.verify(&p, &cfg, "sk_shamwari_", "h"));
        for r in p.requests.borrow().iter() {
            for v in SECRETS {
                assert!(!r.url.contains(v));
                assert!(!r.body.clone().unwrap_or_default().contains(v));
            }
        }
    }
}
