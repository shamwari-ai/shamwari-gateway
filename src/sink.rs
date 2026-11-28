//! Port of `shamwari/gateway/src/sink.ts` (the queue consumer that drains into
//! Core), and of the sink documents `index.ts` builds: the conversation row,
//! the usage events, and the enqueue that honours `SINK_ENABLED`.
//!
//! The Worker never talks to Mongo. Messages go on the `shamwari-sink` queue
//! and the consumer hands them to Core (`/sink/bulk`), which owns the data
//! layer. A Core outage produces a queue backlog, not a failed user request.
//!
//! Field names are camelCase, as the TypeScript Worker writes them: both
//! Workers feed the same queue and Core reads one shape.

use crate::ai_gateway::InferenceResult;
use crate::config::Config;
use crate::failure::console;
use crate::model_router::{tier_name, Reach};
use crate::platform::{HttpRequest, Platform};
use crate::provenance::{conversation_trainable, license_class_name};
use crate::types::{AuthContext, GroundChunk, Scope, SinkDatabase, SinkMessage, Tier};
use serde_json::{json, Map, Value};

// ---------------------------------------------------------------------------
// Time and ids, as the TypeScript gets them from `Date` and `crypto`.

/// `new Date(ms).toISOString()`: `2026-01-01T00:00:00.000Z`.
pub fn iso_timestamp(ms: u64) -> String {
    let (y, m, d) = civil_from_days((ms / 86_400_000) as i64);
    let rem = ms % 86_400_000;
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}.{:03}Z",
        rem / 3_600_000,
        rem / 60_000 % 60,
        rem / 1000 % 60,
        rem % 1000
    )
}

/// The UTC month a request is billed in: `2026-01`.
pub fn billing_period(ms: u64) -> String {
    let (y, m, _) = civil_from_days((ms / 86_400_000) as i64);
    format!("{y:04}-{m:02}")
}

/// Days since 1970-01-01 to (year, month, day), proleptic Gregorian.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = yoe + era * 400 + i64::from(m <= 2);
    (y, m, d)
}

/// `crypto.randomUUID()`: a version 4 UUID from the platform's random bytes.
pub fn random_uuid<P: Platform>(p: &P) -> String {
    let mut b = p.random_bytes(16);
    b.resize(16, 0);
    b[6] = (b[6] & 0x0f) | 0x40;
    b[8] = (b[8] & 0x3f) | 0x80;
    let h: String = b.iter().map(|x| format!("{x:02x}")).collect();
    format!(
        "{}-{}-{}-{}-{}",
        &h[..8],
        &h[8..12],
        &h[12..16],
        &h[16..20],
        &h[20..]
    )
}

// ---------------------------------------------------------------------------
// The documents.

fn reach_value(reach: Option<Reach>) -> Value {
    reach.map_or(Value::Null, |r| r.as_str().into())
}

/// Everything the conversation row is built from. `messages` is the request's
/// `messages` exactly as the caller sent it.
pub struct ConversationInput<'a> {
    pub auth: &'a AuthContext,
    pub request_id: &'a str,
    /// `SURFACE`.
    pub surface: Option<&'a str>,
    pub scope: Scope,
    pub language: Option<&'a str>,
    pub messages: &'a Value,
    pub result: &'a InferenceResult,
    pub chunks: &'a [GroundChunk],
    /// The tier that was called, after the reach clamp.
    pub tier: Tier,
    pub reach: Option<Reach>,
    pub grounded: bool,
    /// The request carried `X-Shamwari-Channel` (any value).
    pub via_channel: bool,
    /// `ownInformation(decision, g).answered_without_own_information`.
    pub answered_without_own_information: bool,
    /// `new Date().toISOString()`, once for the row and the usage event.
    pub now: &'a str,
}

/// The `shamwari.conversations` row for an answered chat request.
///
/// Two independent gates decide `trainingEligible`, and a row needs both:
/// the licence class (the teacher model's terms, resolved from the model that
/// actually answered, see `provenance`) and the user's (never personal scope,
/// never channel traffic, never a WorkOS sign-in, never a question that looked
/// personal, and never without recorded consent: storing a conversation is
/// not consent to train on it). Core refuses a trainable row that does not
/// carry `trainingConsent: true`.
pub fn conversation_doc(c: &ConversationInput<'_>) -> SinkMessage {
    let r = c.result;
    let consented = c.auth.consented_to_training();
    let citations: Vec<Value> = c
        .chunks
        .iter()
        .map(|ch| {
            json!({
                "title": ch.title,
                "heading": ch.heading,
                "authority": ch.authority,
                "url": ch.source_url,
                "effectiveFrom": ch.effective_from,
            })
        })
        .collect();
    let mut doc = Map::new();
    let mut put = |k: &str, v: Value| {
        doc.insert(k.to_owned(), v);
    };
    put("ownerEntityId", c.auth.owner_entity_id.clone().into());
    put("ownerPersonId", json!(c.auth.owner_person_id));
    put("requestId", c.request_id.into());
    if let Some(s) = c.surface {
        put("surfaceContext", s.into());
    }
    put("scope", c.scope.as_str().into());
    put("language", json!(c.language));
    put("messages", c.messages.clone());
    put("response", r.text.clone().into());
    put("citations", Value::Array(citations));
    put("teacherModel", format!("{}/{}", r.provider, r.model).into());
    put("licenseClass", license_class_name(r.license_class).into());
    put(
        "requestedModel",
        format!("{}/{}", r.requested_provider, r.requested_model).into(),
    );
    put("modelSubstituted", r.substituted.into());
    put("tier", tier_name(c.tier).into());
    put("reach", reach_value(c.reach));
    put("grounded", c.grounded.into());
    put("trainingConsent", consented.into());
    put(
        "trainingEligible",
        conversation_trainable(
            c.scope,
            r.license_class,
            c.via_channel,
            &c.auth.key_type,
            c.answered_without_own_information,
            consented,
        )
        .into(),
    );
    put("promoted", false.into());
    put("lastMessageAt", c.now.into());
    put("createdAt", c.now.into());
    SinkMessage {
        database: SinkDatabase::Shamwari,
        collection: "conversations".into(),
        doc,
    }
}

/// Everything the chat path's usage event is built from.
pub struct UsageInput<'a> {
    pub auth: &'a AuthContext,
    pub request_id: &'a str,
    pub scope: Scope,
    pub tier: Tier,
    pub reach: Option<Reach>,
    pub reach_clamped: bool,
    pub routed_tier: Tier,
    pub result: &'a InferenceResult,
    /// `estimateCostUsd(...)` from `pricing` (agent A's): null when unpriced.
    pub cost_usd_estimate: Option<f64>,
    /// `PRICE_TABLE_VERSION` from `pricing`.
    pub price_table_version: &'a str,
    pub ground_hit: bool,
    pub latency_ms: u64,
    /// The same `now` as the conversation row; the billing period is read from it.
    pub now_ms: u64,
}

/// The `platform.usageEvents` document for an answered chat request. A label,
/// not content: it carries the request id, scope, model, token counts, path
/// and time, and never any text.
pub fn usage_event_doc(u: &UsageInput<'_>) -> SinkMessage {
    let r = u.result;
    let mut doc = Map::new();
    let mut put = |k: &str, v: Value| {
        doc.insert(k.to_owned(), v);
    };
    put("ownerEntityId", u.auth.owner_entity_id.clone().into());
    put("apiKeyId", u.auth.api_key_id.clone().into());
    put("requestId", u.request_id.into());
    put("billingPeriod", billing_period(u.now_ms).into());
    put("scope", u.scope.as_str().into());
    put("tier", tier_name(u.tier).into());
    put("reach", reach_value(u.reach));
    put("reachClamped", u.reach_clamped.into());
    put("routedTier", tier_name(u.routed_tier).into());
    put("provider", r.provider.clone().into());
    put("model", r.model.clone().into());
    put("requestedProvider", r.requested_provider.clone().into());
    put("requestedModel", r.requested_model.clone().into());
    put("modelSubstituted", r.substituted.into());
    put("licenseClass", license_class_name(r.license_class).into());
    put("inputTokens", r.input_tokens.into());
    put("outputTokens", r.output_tokens.into());
    put("costUsdEstimate", json!(u.cost_usd_estimate));
    put("priceTableVersion", u.price_table_version.into());
    put("cacheHit", r.cache_hit.into());
    put("groundHit", u.ground_hit.into());
    put("inferencePath", r.path.as_str().into());
    put("latencyMs", u.latency_ms.into());
    put("createdAt", iso_timestamp(u.now_ms).into());
    SinkMessage {
        database: SinkDatabase::Platform,
        collection: "usageEvents".into(),
        doc,
    }
}

/// The usage event of the extractive rung (`extractiveResponse`): metered,
/// and no conversation row, because an extractive answer has no teacher model
/// and so no honest licence class to stamp.
#[allow(clippy::too_many_arguments)]
pub fn extractive_usage_doc(
    auth: &AuthContext,
    request_id: &str,
    scope: Scope,
    ground_hit: bool,
    reach: Option<Reach>,
    degraded: bool,
    latency_ms: u64,
    now_ms: u64,
) -> Map<String, Value> {
    let v = json!({
        "ownerEntityId": auth.owner_entity_id,
        "apiKeyId": auth.api_key_id,
        "requestId": request_id,
        "billingPeriod": billing_period(now_ms),
        "scope": scope.as_str(),
        "tier": "extractive",
        "provider": null,
        "model": null,
        "licenseClass": null,
        "inputTokens": 0,
        "outputTokens": 0,
        "cacheHit": false,
        "groundHit": ground_hit,
        "inferencePath": "extractive",
        "reach": reach_value(reach),
        // No model tier is chosen on this rung, so there is nothing to clamp.
        "reachClamped": false,
        "degraded": degraded,
        "latencyMs": latency_ms,
        "createdAt": iso_timestamp(now_ms),
    });
    match v {
        Value::Object(m) => m,
        _ => unreachable!(),
    }
}

/// The usage event of `/v1/ground/context` (`groundContext`): metered, and no
/// conversation document, because the gateway never sees the answer there.
pub fn retrieval_usage_doc(
    auth: &AuthContext,
    request_id: &str,
    scope: Scope,
    ground_hit: bool,
    edge_embedding: bool,
    latency_ms: u64,
    now_ms: u64,
) -> Map<String, Value> {
    let v = json!({
        "ownerEntityId": auth.owner_entity_id,
        "apiKeyId": auth.api_key_id,
        "requestId": request_id,
        "billingPeriod": billing_period(now_ms),
        "scope": scope.as_str(),
        "tier": "retrieval",
        "provider": null,
        "model": null,
        "licenseClass": null,
        "inputTokens": 0,
        "outputTokens": 0,
        "cacheHit": false,
        "groundHit": ground_hit,
        "inferencePath": "mind",
        "edgeEmbedding": edge_embedding,
        "latencyMs": latency_ms,
        "createdAt": iso_timestamp(now_ms),
    });
    match v {
        Value::Object(m) => m,
        _ => unreachable!(),
    }
}

/// Web chat traffic (`X-Shamwari-Channel: pwa`, any case). The marker is the
/// channel name, so SMS and USSD are not caught by it.
pub fn is_pwa(channel_header: Option<&str>) -> bool {
    channel_header.is_some_and(|h| h.to_lowercase() == "pwa")
}

/// Where an answered request's documents go: `(queued, metered)`.
///
/// The web chat keeps no content on the server (owner decision, 2026-10-31):
/// no conversation document, so no message or answer text is stored and
/// nothing from it can become training material. Usage events go to the meter
/// (`keys::usage_meter`, which may be the Servers API); everything else stays
/// on the sink queue.
pub fn split_for_delivery(
    sink: Vec<SinkMessage>,
    pwa: bool,
) -> (Vec<SinkMessage>, Vec<SinkMessage>) {
    let sink: Vec<SinkMessage> = sink
        .into_iter()
        .filter(|m| !(pwa && m.collection == "conversations"))
        .collect();
    sink.into_iter()
        .partition(|m| m.collection != "usageEvents")
}

/// Put messages on the `shamwari-sink` queue, unless `SINK_ENABLED` is
/// `"false"`: then nothing is sent, so parity traffic never double-writes
/// Core. A queue refuses an empty batch, so nothing is sent for none. A
/// failure is logged and swallowed (`.catch(console.error)`): the answer has
/// already been given. `context` names the caller in the log line ("sink",
/// "probe sink"). Returns how many were sent.
pub async fn enqueue<P: Platform>(
    p: &P,
    cfg: &Config,
    messages: &[SinkMessage],
    context: &str,
) -> usize {
    if !cfg.sink_enabled() || messages.is_empty() {
        return 0;
    }
    let mut sent = 0;
    for m in messages {
        let v = match serde_json::to_value(m) {
            Ok(v) => v,
            Err(e) => {
                console::error(&format!("{context} enqueue failed {e}"));
                return sent;
            }
        };
        if let Err(e) = p.queue_send(&v).await {
            console::error(&format!("{context} enqueue failed {e}"));
            return sent;
        }
        sent += 1;
    }
    sent
}

// ---------------------------------------------------------------------------
// The consumer.

/// Statuses that mean "this document is wrong", not "the service is down".
const PERMANENT: [u16; 4] = [400, 409, 413, 422];

/// How long one `/sink/bulk` write may take before the batch is given back to the queue.
pub const SINK_TIMEOUT_MS: u32 = 15_000;

/// Only a plain identifier is ever logged; anything else (content, a hostile id) is not.
fn safe(v: Option<&Value>) -> String {
    match v.and_then(Value::as_str) {
        Some(s)
            if (1..=64).contains(&s.chars().count())
                && s.chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | ':' | '-')) =>
        {
            s.to_owned()
        }
        _ => "unknown".to_owned(),
    }
}

/// Core relays the Servers API's RFC 9457 `error` member as `detail.error` (or `error`).
fn error_code(text: &str) -> String {
    match serde_json::from_str::<Value>(text) {
        Ok(body) => {
            let detail = body
                .get("detail")
                .and_then(|d| d.get("error"))
                .filter(|v| !v.is_null());
            safe(detail.or_else(|| body.get("error")))
        }
        Err(_) => "unknown".to_owned(),
    }
}

/// What [`drain`] settled. The queue handler acknowledges every index in
/// `acked`; when `error` is set it then fails, so the queue retries whatever
/// is not acknowledged (five times, then the DLQ).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Drained {
    /// Indexes into the batch, in the order they were settled.
    pub acked: Vec<usize>,
    pub error: Option<String>,
}

struct Parked {
    index: usize,
    status: u16,
    error: String,
}

/// Hand a batch (up to 100) to Core, which owns the data-layer connection.
///
/// - Transient (a 5xx, 429, a refused credential, a missing route, or no
///   answer at all): an error. The queue retries what has not been
///   acknowledged. Nothing is lost, and a Core outage becomes a backlog.
/// - Permanent (400, 409, 413, 422: a poison or conflicting document): the
///   batch is split in half until the bad document is alone; the good ones are
///   stored and acknowledged, and the bad one is acknowledged too, after
///   logging its identifiers. It is dropped and the log line is the record.
/// - If EVERY document in a batch of more than one is refused, that is a fault
///   (a Core/API mismatch), not poison. Nothing is dropped: it is an error.
pub async fn drain<P: Platform>(p: &P, cfg: &Config, batch: &[SinkMessage]) -> Drained {
    let mut by_db: Vec<(SinkDatabase, Vec<usize>)> = Vec::new();
    for (i, m) in batch.iter().enumerate() {
        match by_db.iter_mut().find(|(d, _)| *d == m.database) {
            Some((_, v)) => v.push(i),
            None => by_db.push((m.database, vec![i])),
        }
    }

    let mut out = Drained::default();
    for (database, entries) in by_db {
        let mut parked = Vec::new();
        if let Err(e) = send(
            p,
            cfg,
            batch,
            database,
            &entries,
            &mut out.acked,
            &mut parked,
        )
        .await
        {
            out.error = Some(e);
            return out;
        }
        if parked.is_empty() {
            continue;
        }
        if parked.len() == entries.len() && entries.len() > 1 {
            out.error = Some(format!(
                "sink {}: every document in the batch was refused",
                parked[0].status
            ));
            return out;
        }
        for pk in parked {
            let m = &batch[pk.index];
            // Identifiers only: never the document, a message body or the response text.
            console::error(&format!(
                "sink parked a document: status={} error={} collection={} requestId={}",
                pk.status,
                pk.error,
                safe(Some(&Value::String(m.collection.clone()))),
                safe(m.doc.get("requestId"))
            ));
            out.acked.push(pk.index);
        }
    }
    out
}

/// Stores what it can, acknowledging as it goes; collects permanently refused
/// documents. Bisects depth first, first half before second, as the
/// TypeScript recursion does.
async fn send<P: Platform>(
    p: &P,
    cfg: &Config,
    batch: &[SinkMessage],
    database: SinkDatabase,
    entries: &[usize],
    acked: &mut Vec<usize>,
    parked: &mut Vec<Parked>,
) -> Result<(), String> {
    let mut stack: Vec<Vec<usize>> = vec![entries.to_vec()];
    while let Some(part) = stack.pop() {
        if part.is_empty() {
            continue;
        }
        let mut collections = Map::new();
        for &i in &part {
            let m = &batch[i];
            let list = collections
                .entry(m.collection.clone())
                .or_insert_with(|| Value::Array(vec![]));
            if let Value::Array(a) = list {
                a.push(Value::Object(m.doc.clone()));
            }
        }
        let body = json!({ "database": database, "collections": collections });
        let res = p
            .fetch(HttpRequest {
                method: "POST".into(),
                url: format!("{}/sink/bulk", cfg.core_url.as_deref().unwrap_or_default()),
                headers: vec![
                    ("Content-Type".into(), "application/json".into()),
                    (
                        "Authorization".into(),
                        format!(
                            "Bearer {}",
                            cfg.shamwari_core_token.as_deref().unwrap_or_default()
                        ),
                    ),
                ],
                body: Some(body.to_string()),
                // A Core that accepts the connection and never answers must not
                // hold the queue consumer: cut off, the batch is retried later.
                timeout_ms: Some(SINK_TIMEOUT_MS),
            })
            .await?;

        if res.ok() {
            acked.extend(part.iter().copied());
            continue;
        }
        if !PERMANENT.contains(&res.status) {
            return Err(format!("sink {}: {}", res.status, res.body));
        }
        if part.len() == 1 {
            parked.push(Parked {
                index: part[0],
                status: res.status,
                error: error_code(&res.body),
            });
            continue;
        }
        let mid = part.len().div_ceil(2);
        stack.push(part[mid..].to_vec());
        stack.push(part[..mid].to_vec());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai_gateway::InferencePath;
    use crate::platform::HttpResponse;
    use crate::testing::{block_on, FakePlatform};
    use crate::types::LicenseClass;
    use std::cell::RefCell;
    use std::rc::Rc;

    fn cfg() -> Config {
        Config {
            core_url: Some("https://core.example".into()),
            shamwari_core_token: Some("tok".into()),
            ..Default::default()
        }
    }

    fn doc(request_id: &str, extra: Value) -> SinkMessage {
        let mut d = json!({ "requestId": request_id, "secretPrompt": "PRIVATE TEXT" });
        if let Value::Object(e) = extra {
            for (k, v) in e {
                d[k] = v;
            }
        }
        SinkMessage {
            database: SinkDatabase::Shamwari,
            collection: "conversations".into(),
            doc: d.as_object().unwrap().clone(),
        }
    }

    fn docs(ids: &[&str]) -> Vec<SinkMessage> {
        ids.iter().map(|r| doc(r, json!({}))).collect()
    }

    fn ids_of(batch: &[SinkMessage], idx: &[usize]) -> Vec<String> {
        let mut v: Vec<String> = idx
            .iter()
            .map(|&i| batch[i].doc["requestId"].as_str().unwrap().to_owned())
            .collect();
        v.sort();
        v
    }

    #[derive(Default)]
    struct CoreSeen {
        stored: Vec<String>,
        calls: usize,
    }

    /// A Core that stores every document except those whose requestId is in `bad`.
    fn core(p: &FakePlatform, bad: &[&str], bad_status: u16) -> Rc<RefCell<CoreSeen>> {
        let seen = Rc::new(RefCell::new(CoreSeen::default()));
        let s = seen.clone();
        let bad: Vec<String> = bad.iter().map(|b| b.to_string()).collect();
        p.on_fn(move |r| {
            let body: Value = serde_json::from_str(r.body.as_deref()?).ok()?;
            let ids: Vec<String> = body["collections"]
                .as_object()?
                .values()
                .flat_map(|v| v.as_array().cloned().unwrap_or_default())
                .map(|d| d["requestId"].as_str().unwrap_or_default().to_owned())
                .collect();
            let mut s = s.borrow_mut();
            s.calls += 1;
            if ids.iter().any(|i| bad.contains(i)) {
                return Some(Ok(HttpResponse {
                    status: bad_status,
                    headers: vec![],
                    body: r#"{"detail":{"error":"invalid_field_name"}}"#.into(),
                }));
            }
            s.stored.extend(ids);
            Some(Ok(HttpResponse {
                status: 200,
                headers: vec![],
                body: r#"{"written":{}}"#.into(),
            }))
        });
        seen
    }

    fn errors_logged() -> String {
        console::take().join("\n")
    }

    // sink.test.ts: a batch with one poison document.

    #[test]
    fn stores_the_others_and_parks_only_the_bad_one() {
        for status in [400, 409, 413, 422] {
            let p = FakePlatform::new();
            let c = core(&p, &["r3"], status);
            let b = docs(&["r1", "r2", "r3", "r4", "r5"]);
            let out = block_on(drain(&p, &cfg(), &b));
            assert_eq!(out.error, None);
            let mut stored = c.borrow().stored.clone();
            stored.sort();
            assert_eq!(stored, ["r1", "r2", "r4", "r5"]);
            // Every message is settled once and none is left to retry or to reach the DLQ.
            assert_eq!(ids_of(&b, &out.acked), ["r1", "r2", "r3", "r4", "r5"]);
        }
    }

    #[test]
    fn isolates_it_by_bisecting_not_by_sending_documents_one_at_a_time() {
        let p = FakePlatform::new();
        let c = core(&p, &["r42"], 400);
        let ids: Vec<String> = (0..100).map(|i| format!("r{i}")).collect();
        let b: Vec<SinkMessage> = ids.iter().map(|i| doc(i, json!({}))).collect();
        block_on(drain(&p, &cfg(), &b));
        assert_eq!(c.borrow().stored.len(), 99);
        assert!(c.borrow().calls < 25, "{}", c.borrow().calls);
    }

    #[test]
    fn treats_a_batch_where_every_document_is_refused_as_a_fault_nothing_is_dropped() {
        let p = FakePlatform::new();
        let c = core(&p, &["r1", "r2", "r3", "r4"], 422);
        console::take();
        let out = block_on(drain(&p, &cfg(), &docs(&["r1", "r2", "r3", "r4"])));
        assert!(out.error.as_deref().unwrap().contains("sink 422"));
        assert!(out.acked.is_empty());
        assert!(c.borrow().stored.is_empty());
        assert!(!errors_logged().contains("parked"));
    }

    #[test]
    fn still_drops_a_lone_refused_document() {
        let p = FakePlatform::new();
        let c = core(&p, &["r1"], 400);
        let out = block_on(drain(&p, &cfg(), &docs(&["r1"])));
        assert_eq!(out.acked, [0]);
        assert_eq!(out.error, None);
        // Not retried: one call.
        assert_eq!(c.borrow().calls, 1);
    }

    #[test]
    fn logs_identifiers_only_never_content() {
        let p = FakePlatform::new();
        core(&p, &["r3"], 400);
        console::take();
        let b = vec![
            doc("r1", json!({})),
            doc("r3", json!({ "messages": [{ "content": "SECRET WORDS" }] })),
        ];
        block_on(drain(&p, &cfg(), &b));
        let text = errors_logged();
        for want in ["r3", "conversations", "400", "invalid_field_name"] {
            assert!(text.contains(want), "{want}: {text}");
        }
        for leaked in [
            "PRIVATE TEXT",
            "SECRET WORDS",
            "secretPrompt",
            "tok",
            "core.example",
        ] {
            assert!(!text.contains(leaked), "{leaked}");
        }
    }

    #[test]
    fn logs_nothing_that_is_not_a_plain_identifier_even_if_the_request_id_is_hostile() {
        let p = FakePlatform::new();
        core(&p, &["not an id: PRIVATE TEXT"], 400);
        console::take();
        block_on(drain(&p, &cfg(), &docs(&["not an id: PRIVATE TEXT"])));
        let text = errors_logged();
        assert!(text.contains("requestId=unknown"));
        assert!(!text.contains("PRIVATE TEXT"));
    }

    #[test]
    fn keeps_the_good_documents_of_other_databases_and_collections() {
        let p = FakePlatform::new();
        let c = core(&p, &["p2"], 400);
        let usage = |id: &str| SinkMessage {
            database: SinkDatabase::Platform,
            collection: "usageEvents".into(),
            doc: json!({ "requestId": id }).as_object().unwrap().clone(),
        };
        let b = vec![doc("r1", json!({})), usage("p1"), usage("p2")];
        let out = block_on(drain(&p, &cfg(), &b));
        let mut stored = c.borrow().stored.clone();
        stored.sort();
        assert_eq!(stored, ["p1", "r1"]);
        assert_eq!(out.acked.len(), 3);
    }

    // A transient failure retries as before.

    #[test]
    fn a_transient_status_is_an_error_so_the_queue_retries_and_nothing_is_acknowledged() {
        for status in [500, 502, 503, 504, 429, 401, 404] {
            let p = FakePlatform::new();
            p.on("https://core.example", status, "upstream broke");
            let out = block_on(drain(&p, &cfg(), &docs(&["r1", "r2"])));
            assert!(out
                .error
                .as_deref()
                .unwrap()
                .starts_with(&format!("sink {status}")));
            assert!(out.acked.is_empty());
        }
    }

    #[test]
    fn is_an_error_when_core_cannot_be_reached() {
        let p = FakePlatform::new();
        p.on_fn(|_| Some(Err("network down".into())));
        let out = block_on(drain(&p, &cfg(), &docs(&["r1"])));
        assert!(out.error.is_some());
        assert!(out.acked.is_empty());
    }

    #[test]
    fn after_isolating_a_poison_document_a_transient_failure_still_retries_only_what_is_unsettled()
    {
        let p = FakePlatform::new();
        let n = Rc::new(RefCell::new(0));
        let k = n.clone();
        p.on_fn(move |r| {
            *k.borrow_mut() += 1;
            let body: Value = serde_json::from_str(r.body.as_deref()?).ok()?;
            let ids: Vec<String> = body["collections"]
                .as_object()?
                .values()
                .flat_map(|v| v.as_array().cloned().unwrap_or_default())
                .map(|d| d["requestId"].as_str().unwrap_or_default().to_owned())
                .collect();
            let status = if ids == ["r2"] {
                503
            } else if ids.iter().any(|i| i == "r1" || i == "r2") {
                400
            } else {
                200
            };
            Some(Ok(HttpResponse {
                status,
                headers: vec![],
                body: "{}".into(),
            }))
        });
        let out = block_on(drain(&p, &cfg(), &docs(&["r1", "r2", "r3"])));
        assert!(out.error.as_deref().unwrap().starts_with("sink 503"));
        assert!(*n.borrow() > 1);
        // r1 was parked inside `send` but never acknowledged: it is retried with the rest.
        assert!(out.acked.is_empty());
    }

    // A good batch.

    #[test]
    fn a_good_batch_is_one_request_and_every_message_acknowledged() {
        let p = FakePlatform::new();
        let c = core(&p, &[], 400);
        let out = block_on(drain(&p, &cfg(), &docs(&["r1", "r2"])));
        assert_eq!(c.borrow().calls, 1);
        assert_eq!(out.acked.len(), 2);
    }

    #[test]
    fn sends_to_core_url_sink_bulk_with_the_core_token_and_nothing_for_the_servers_api() {
        let p = FakePlatform::new();
        core(&p, &[], 400);
        block_on(drain(&p, &cfg(), &docs(&["r1"])));
        let reqs = p.requests.borrow();
        assert_eq!(reqs[0].url, "https://core.example/sink/bulk");
        let mut names: Vec<&str> = reqs[0].headers.iter().map(|(k, _)| k.as_str()).collect();
        names.sort();
        assert_eq!(names, ["Authorization", "Content-Type"]);
        assert_eq!(
            reqs[0]
                .headers
                .iter()
                .find(|(k, _)| k == "Authorization")
                .unwrap()
                .1,
            "Bearer tok"
        );
        assert_eq!(reqs[0].timeout_ms, Some(SINK_TIMEOUT_MS));
        let body: Value = serde_json::from_str(reqs[0].body.as_deref().unwrap()).unwrap();
        assert_eq!(body["database"], "shamwari");
        assert_eq!(body["collections"]["conversations"][0]["requestId"], "r1");
    }

    // The documents, and Rule 2 on them.

    fn auth(key_type: &str, consent: Option<bool>) -> AuthContext {
        AuthContext {
            api_key_id: "k1".into(),
            owner_entity_id: "e1".into(),
            owner_person_id: None,
            key_type: key_type.into(),
            tier: "community".into(),
            monthly_token_cap: 100_000,
            scopes: None,
            training_consent: consent,
        }
    }

    fn result(license: LicenseClass) -> InferenceResult {
        InferenceResult {
            text: "Section 48 protects the right to life.".into(),
            input_tokens: 120,
            output_tokens: 9,
            cache_hit: false,
            provider: "workers-ai".into(),
            model: "@cf/zai-org/glm-5.3".into(),
            requested_provider: "workers-ai".into(),
            requested_model: "@cf/zai-org/glm-5.3".into(),
            license_class: license,
            substituted: false,
            path: InferencePath::Gateway,
            finish_reason: "stop".into(),
        }
    }

    fn chunk() -> GroundChunk {
        GroundChunk {
            content: "Every person has the right to life.".into(),
            heading: Some("Section 48".into()),
            title: "Constitution of Zimbabwe".into(),
            authority: "Parliament of Zimbabwe".into(),
            resource_type: "legislation".into(),
            source_url: Some("https://example.invalid/constitution".into()),
            effective_from: Some("2013-05-22".into()),
            scope: "platform".into(),
        }
    }

    fn conversation(
        a: &AuthContext,
        r: &InferenceResult,
        via_channel: bool,
        looks_personal: bool,
    ) -> SinkMessage {
        let messages = json!([{ "role": "user", "content": "What does section 48 say?" }]);
        conversation_doc(&ConversationInput {
            auth: a,
            request_id: "req-1",
            surface: Some("shamwari.ai"),
            scope: Scope::Platform,
            language: Some("en"),
            messages: &messages,
            result: r,
            chunks: &[chunk()],
            tier: Tier::Economy,
            reach: Some(Reach::R1),
            grounded: true,
            via_channel,
            answered_without_own_information: looks_personal,
            now: "2026-01-01T00:00:00.000Z",
        })
    }

    #[test]
    fn the_conversation_row_has_the_shape_core_reads() {
        let a = auth("standard", Some(true));
        let r = result(LicenseClass::OpenWeight);
        let m = conversation(&a, &r, false, false);
        assert_eq!(m.database, SinkDatabase::Shamwari);
        assert_eq!(m.collection, "conversations");
        let keys: Vec<&str> = m.doc.keys().map(String::as_str).collect();
        assert_eq!(
            keys,
            [
                "ownerEntityId",
                "ownerPersonId",
                "requestId",
                "surfaceContext",
                "scope",
                "language",
                "messages",
                "response",
                "citations",
                "teacherModel",
                "licenseClass",
                "requestedModel",
                "modelSubstituted",
                "tier",
                "reach",
                "grounded",
                "trainingConsent",
                "trainingEligible",
                "promoted",
                "lastMessageAt",
                "createdAt",
            ]
        );
        assert_eq!(
            m.doc["citations"],
            json!([{
                "title": "Constitution of Zimbabwe", "heading": "Section 48", "authority": "Parliament of Zimbabwe",
                "url": "https://example.invalid/constitution", "effectiveFrom": "2013-05-22",
            }])
        );
        assert_eq!(m.doc["teacherModel"], "workers-ai/@cf/zai-org/glm-5.3");
        assert_eq!(m.doc["licenseClass"], "open_weight");
        assert_eq!(m.doc["reach"], "r1");
        assert_eq!(m.doc["trainingConsent"], true);
        assert_eq!(m.doc["trainingEligible"], true);
        assert_eq!(m.doc["promoted"], false);
    }

    #[test]
    fn rule_2_no_consent_no_training_the_row_is_stored_but_never_trainable() {
        let r = result(LicenseClass::OpenWeight);
        for consent in [None, Some(false)] {
            let a = auth("standard", consent);
            let m = conversation(&a, &r, false, false);
            assert_eq!(m.doc["trainingConsent"], false);
            assert_eq!(m.doc["trainingEligible"], false);
        }
    }

    #[test]
    fn rule_2_a_restricted_teacher_a_channel_a_workos_caller_or_a_personal_looking_question_is_never_trainable(
    ) {
        let consented = auth("standard", Some(true));
        let open = result(LicenseClass::OpenWeight);
        let restricted = result(LicenseClass::Restricted);
        assert_eq!(
            conversation(&consented, &restricted, false, false).doc["trainingEligible"],
            false
        );
        assert_eq!(
            conversation(&consented, &restricted, false, false).doc["licenseClass"],
            "restricted"
        );
        assert_eq!(
            conversation(&consented, &open, true, false).doc["trainingEligible"],
            false
        );
        assert_eq!(
            conversation(&consented, &open, false, true).doc["trainingEligible"],
            false
        );
        let workos = auth("workos", Some(true));
        assert_eq!(
            conversation(&workos, &open, false, false).doc["trainingEligible"],
            false
        );
    }

    #[test]
    fn the_usage_event_carries_tokens_cost_and_the_reach_and_never_text() {
        let a = auth("standard", Some(true));
        let r = result(LicenseClass::OpenWeight);
        let m = usage_event_doc(&UsageInput {
            auth: &a,
            request_id: "req-1",
            scope: Scope::Platform,
            tier: Tier::Economy,
            reach: Some(Reach::R0),
            reach_clamped: true,
            routed_tier: Tier::Standard,
            result: &r,
            cost_usd_estimate: None,
            price_table_version: "2026-10-29",
            ground_hit: true,
            latency_ms: 42,
            now_ms: 1_767_225_600_000,
        });
        assert_eq!(m.database, SinkDatabase::Platform);
        assert_eq!(m.collection, "usageEvents");
        assert_eq!(
            Value::Object(m.doc.clone()),
            json!({
                "ownerEntityId": "e1", "apiKeyId": "k1", "requestId": "req-1", "billingPeriod": "2026-01",
                "scope": "platform", "tier": "economy", "reach": "r0", "reachClamped": true,
                "routedTier": "standard", "provider": "workers-ai", "model": "@cf/zai-org/glm-5.3",
                "requestedProvider": "workers-ai", "requestedModel": "@cf/zai-org/glm-5.3",
                "modelSubstituted": false, "licenseClass": "open_weight", "inputTokens": 120,
                "outputTokens": 9, "costUsdEstimate": null, "priceTableVersion": "2026-10-29",
                "cacheHit": false, "groundHit": true, "inferencePath": "gateway", "latencyMs": 42,
                "createdAt": "2026-01-01T00:00:00.000Z",
            })
        );
        assert!(!m
            .doc
            .values()
            .any(|v| v.as_str().is_some_and(|s| s.contains("right to life"))));
    }

    #[test]
    fn the_extractive_and_retrieval_usage_events_stamp_no_model_and_no_licence() {
        let a = auth("standard", None);
        let x = extractive_usage_doc(&a, "r", Scope::Platform, true, Some(Reach::R0), true, 5, 0);
        assert_eq!(x["tier"], "extractive");
        assert_eq!(x["licenseClass"], Value::Null);
        assert_eq!(x["reach"], "r0");
        assert_eq!(x["degraded"], true);
        assert_eq!(x["createdAt"], "1970-01-01T00:00:00.000Z");
        let g = retrieval_usage_doc(&a, "r", Scope::Personal, false, false, 5, 0);
        assert_eq!(g["tier"], "retrieval");
        assert_eq!(g["inferencePath"], "mind");
        assert_eq!(g["model"], Value::Null);
        assert_eq!(g["edgeEmbedding"], false);
    }

    #[test]
    fn the_web_chat_keeps_no_conversation_and_usage_goes_to_the_meter() {
        let a = auth("standard", Some(true));
        let r = result(LicenseClass::OpenWeight);
        let convo = conversation(&a, &r, true, false);
        let usage = SinkMessage {
            database: SinkDatabase::Platform,
            collection: "usageEvents".into(),
            doc: Map::new(),
        };
        let (q, m) = split_for_delivery(vec![convo.clone(), usage.clone()], false);
        assert_eq!(q, std::slice::from_ref(&convo));
        assert_eq!(m, std::slice::from_ref(&usage));
        let (q, m) = split_for_delivery(vec![convo, usage.clone()], true);
        assert!(q.is_empty());
        assert_eq!(m, [usage]);
        assert!(is_pwa(Some("PWA")));
        assert!(!is_pwa(Some("sms")));
        assert!(!is_pwa(None));
    }

    #[test]
    fn nothing_reaches_the_queue_while_sink_enabled_is_false() {
        let p = FakePlatform::new();
        let m = docs(&["r1"]);
        let off = Config {
            sink_enabled: Some("false".into()),
            ..cfg()
        };
        assert_eq!(block_on(enqueue(&p, &off, &m, "sink")), 0);
        assert!(p.queue.borrow().is_empty());
        assert_eq!(block_on(enqueue(&p, &cfg(), &[], "sink")), 0);
        assert_eq!(block_on(enqueue(&p, &cfg(), &m, "sink")), 1);
        let q = p.queue.borrow();
        assert_eq!(q[0]["database"], "shamwari");
        assert_eq!(q[0]["collection"], "conversations");
        assert_eq!(q[0]["doc"]["requestId"], "r1");
    }

    #[test]
    fn time_and_ids_read_as_javascript_writes_them() {
        assert_eq!(iso_timestamp(1_767_225_600_000), "2026-01-01T00:00:00.000Z");
        assert_eq!(iso_timestamp(951_827_696_789), "2000-02-29T12:34:56.789Z");
        assert_eq!(billing_period(1_767_225_599_999), "2025-12");
        let u = random_uuid(&FakePlatform::new());
        assert_eq!(u.len(), 36);
        assert_eq!(&u[14..15], "4");
        assert!(matches!(&u[19..20], "8" | "9" | "a" | "b"));
    }
}
