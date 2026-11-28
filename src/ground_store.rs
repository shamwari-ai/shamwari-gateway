//! Port of `shamwari/gateway/src/ground-store.ts`: Ground in MongoDB, through
//! the Nyuchi API (owner, 2026-11-23: "use mongodb for the store").
//!
//! The passages live in `shamwari.knowledgeBase` (Atlas vector and text
//! search indexes, bge-m3, 1024 numbers). The gateway holds no database login
//! and no service key: it calls `POST /v1/shamwari/ground/search` on the
//! Nyuchi API with the signed-in person's own WorkOS session token, so the API
//! acts with that person's base access and takes the owner from the token
//! (owner, 2026-11-23: no single key, no permission creep). A caller with an
//! API key has no person behind it and gets no Mongo search; our own sites
//! ([`crate::own_sites`]) still answer. The API makes the Rule 1 checks again
//! before it reads anything (personal scope to the cloud, an embedded mind
//! query), and answers in the shape Core did, so `ground()` treats both the
//! same.
//!
//! Getting to an answer must never wait on this: a short time limit, and a
//! breaker of its own, so a down API is skipped at once instead of being
//! waited on by every question.
//!
//! The person's token is sent only to the Nyuchi API URL, as a Bearer token.
//! It is never logged, cached or put in an error.

use crate::config::Config;
use crate::lang::js;
use crate::platform::{HttpRequest, Platform};
use serde_json::{json, Map, Value};
use std::cell::Cell;

/// The API answers a search in well under this from the edge; past it the
/// answer goes on without sources.
pub const GROUND_TIMEOUT_MS: u32 = 1500;
/// After a failure (no connection, the limit above, a 5xx), later searches
/// skip the API for this long.
pub const GROUND_COOLDOWN_MS: u64 = 30_000;
/// A refused token (401/403) here means the API does not accept our people's
/// sessions (not configured, or the first-party client not listed): the
/// gateway verified the token first, so it is not this person's. That will
/// not fix itself in seconds, so the API is skipped for longer and no question
/// pays for a call that cannot succeed.
pub const GROUND_REFUSED_KEY_COOLDOWN_MS: u64 = 5 * 60_000;

thread_local! {
    /// The TypeScript module's `downUntil`: one isolate, one thread.
    static DOWN_UNTIL: Cell<u64> = const { Cell::new(0) };
}

/// True while the breaker is open.
pub fn ground_store_down(now_ms: u64) -> bool {
    now_ms < DOWN_UNTIL.with(Cell::get)
}

/// For tests: close the breaker.
pub fn reset_ground_store_breaker() {
    DOWN_UNTIL.with(|d| d.set(0));
}

fn set_down_until(ms: u64) {
    DOWN_UNTIL.with(|d| d.set(ms));
}

/// Ground in Mongo is used when the API is configured and a signed-in person
/// is asking; otherwise Core.
pub fn ground_store_configured(cfg: &Config, person_token: Option<&str>) -> bool {
    cfg.get(&cfg.nyuchi_api_url).is_some() && person_token.is_some_and(|t| !t.is_empty())
}

/// What `ground()` asks for. Only these fields go to the API, which refuses
/// unknown ones (Core's `granted_scopes` and `needs_own_information` are not
/// among them).
#[derive(Debug, Clone, PartialEq)]
pub struct GroundQuery {
    pub query: String,
    pub embedding: Option<Vec<f64>>,
    pub owner_entity_id: String,
    pub scope: String,
    pub destination: String,
    pub language: Option<String>,
    /// `Number(GROUND_TOP_K)`, so NaN when the setting is absent or malformed.
    pub top_k: f64,
}

/// A refusal or failure from the store, with the status the API gave (503
/// for a skip or a timeout).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroundStoreError {
    pub status: u16,
    pub code: String,
}

impl GroundStoreError {
    fn new(status: u16, code: impl Into<String>) -> Self {
        GroundStoreError {
            status,
            code: code.into(),
        }
    }
}

impl std::fmt::Display for GroundStoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ground store {}: {}", self.status, self.code)
    }
}

impl std::error::Error for GroundStoreError {}

/// The request body: only the fields the API accepts, in the TypeScript order.
fn body(q: &GroundQuery) -> Value {
    let mut m = Map::new();
    m.insert("query".into(), json!(q.query));
    if let Some(e) = &q.embedding {
        m.insert("embedding".into(), json!(e));
    }
    m.insert("owner_entity_id".into(), json!(q.owner_entity_id));
    m.insert("scope".into(), json!(q.scope));
    m.insert("destination".into(), json!(q.destination));
    m.insert("language".into(), json!(q.language));
    // `Math.min(20, Math.max(1, top_k))`: NaN stays NaN, which JSON writes as null.
    let top_k = if q.top_k.is_nan() {
        f64::NAN
    } else {
        q.top_k.clamp(1.0, 20.0)
    };
    m.insert("top_k".into(), js::number_json(top_k));
    Value::Object(m)
}

/// `POST {NYUCHI_API_URL}/v1/shamwari/ground/search` with the person's own
/// session. The answer is the API's JSON as it came (`{chunks, hit, scope,
/// needs_own_information}`); `ground()` checks it.
pub async fn ground_search<P: Platform>(
    p: &P,
    cfg: &Config,
    q: &GroundQuery,
    person_token: &str,
) -> Result<Value, GroundStoreError> {
    if ground_store_down(p.now_ms()) {
        return Err(GroundStoreError::new(
            503,
            "ground_unavailable: skipped while the breaker is open",
        ));
    }
    let mut base = cfg.nyuchi_api_url.clone().unwrap_or_default();
    while base.ends_with('/') {
        base.pop();
    }
    let req = HttpRequest {
        method: "POST".into(),
        url: format!("{base}/v1/shamwari/ground/search"),
        headers: vec![
            ("Authorization".into(), format!("Bearer {person_token}")),
            ("Content-Type".into(), "application/json".into()),
        ],
        body: Some(body(q).to_string()),
        timeout_ms: Some(GROUND_TIMEOUT_MS),
    };
    let res = match p.fetch(req).await {
        Ok(res) => res,
        Err(e) => {
            set_down_until(p.now_ms() + GROUND_COOLDOWN_MS);
            // TypeScript reports the error's name only: `TimeoutError` for the
            // limit, `TypeError` for a fetch that could not connect.
            let name = if e == "timeout" {
                "TimeoutError"
            } else {
                "TypeError"
            };
            return Err(GroundStoreError::new(
                503,
                format!("ground_unavailable: {name}"),
            ));
        }
    };
    if !res.ok() {
        // A 5xx is an outage. A 401/403 is our people's sessions refused (see
        // above): every later search would fail the same way, so it is skipped
        // for longer. Any other 4xx is the API answering (a Rule 1 refusal, a
        // bad argument) and never opens the breaker.
        if res.status >= 500 {
            set_down_until(p.now_ms() + GROUND_COOLDOWN_MS);
        } else if res.status == 401 || res.status == 403 {
            set_down_until(p.now_ms() + GROUND_REFUSED_KEY_COOLDOWN_MS);
        } else {
            set_down_until(0);
        }
        let mut code = format!("http_{}", res.status);
        if let Ok(j) = serde_json::from_str::<Value>(&res.body) {
            match j.get("detail") {
                Some(Value::String(d)) => code = d.clone(),
                Some(d) => {
                    if let Some(Value::String(e)) = d.get("error") {
                        code = e.clone();
                    }
                }
                None => {}
            }
        }
        return Err(GroundStoreError::new(res.status, code));
    }
    set_down_until(0);
    serde_json::from_str(&res.body).map_err(|_| GroundStoreError::new(res.status, "invalid_json"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ground::ground;
    use crate::platform::HttpResponse;
    use crate::scope::{Destination, ScopeResolution};
    use crate::testing::{block_on, FakePlatform};
    use crate::types::{Message, Role, Scope};

    pub(crate) fn env() -> Config {
        Config {
            nyuchi_api_url: Some("https://api.example.com/".into()),
            core_url: Some("https://core.invalid".into()),
            shamwari_core_token: Some("t".into()),
            ground_top_k: Some("6".into()),
            embedding_model: Some("@cf/baai/bge-m3".into()),
            ..Default::default()
        }
    }

    const TOKEN: &str = "person.workos.token";

    fn q() -> GroundQuery {
        GroundQuery {
            query: "What does section 56 say?".into(),
            embedding: Some(vec![0.5, 0.25]),
            owner_entity_id: "e-1".into(),
            scope: "platform".into(),
            destination: "cloud".into(),
            language: Some("en".into()),
            top_k: 6.0,
        }
    }

    fn section_56() -> Value {
        json!({
            "title": "Constitution of Zimbabwe",
            "heading": "Section 56 — Equality and non-discrimination",
            "authority": "Parliament of Zimbabwe",
            "resource_type": "constitution",
            "source_url": null,
            "effective_from": "2013-05-22",
            "content": "All persons are equal before the law.",
            "scope": "platform",
        })
    }

    fn platform() -> FakePlatform {
        reset_ground_store_breaker();
        let p = FakePlatform::new();
        p.on_ai(|_, _| Some(Ok(json!({ "data": [[0.5, 0.25]] }))));
        p
    }

    fn cloud() -> ScopeResolution {
        ScopeResolution {
            scope: Scope::Platform,
            destination: Destination::Cloud,
            grants: vec![Scope::Platform],
            without_own_information: false,
        }
    }

    fn equal() -> Vec<Message> {
        vec![Message {
            role: Role::User,
            content: "Is everyone equal?".into(),
        }]
    }

    fn host(url: &str) -> String {
        url::Url::parse(url).unwrap().host_str().unwrap().to_owned()
    }

    /// Core's `/ground/search` through the fake's fetch, as the TypeScript
    /// tests reach Core through the stubbed global `fetch`.
    pub(crate) async fn core_via(
        p: &FakePlatform,
        cfg: &Config,
        body: Value,
    ) -> Result<Value, String> {
        let res = p
            .fetch(HttpRequest {
                method: "POST".into(),
                url: format!("{}/ground/search", cfg.core_url.clone().unwrap_or_default()),
                headers: vec![("Content-Type".into(), "application/json".into())],
                body: Some(body.to_string()),
                timeout_ms: Some(4000),
            })
            .await?;
        if !res.ok() {
            return Err(format!("core {}", res.status));
        }
        serde_json::from_str(&res.body).map_err(|e| e.to_string())
    }

    fn fetches(p: &FakePlatform) -> usize {
        p.requests.borrow().len()
    }

    // Ground in Mongo through the Nyuchi API: the call

    #[test]
    fn calls_the_apis_ground_search_with_the_persons_own_session_and_only_the_fields_it_accepts() {
        let p = platform();
        let answer = json!({ "scope": "platform", "hit": true, "chunks": [section_56()], "needs_own_information": false });
        p.on("https://", 200, &answer.to_string());
        let out = block_on(ground_search(&p, &env(), &q(), TOKEN)).unwrap();
        assert_eq!(out["hit"], true);
        let req = p.requests.borrow()[0].clone();
        assert_eq!(req.url, "https://api.example.com/v1/shamwari/ground/search");
        assert_eq!(req.method, "POST");
        assert_eq!(
            req.headers,
            vec![
                ("Authorization".to_owned(), format!("Bearer {TOKEN}")),
                ("Content-Type".to_owned(), "application/json".to_owned()),
            ]
        );
        assert!(!req.headers.iter().any(|(k, _)| k == "X-API-Key"));
        assert_eq!(req.timeout_ms, Some(GROUND_TIMEOUT_MS));
        let sent: Value = serde_json::from_str(req.body.as_deref().unwrap()).unwrap();
        assert_eq!(
            sent,
            json!({
                "query": "What does section 56 say?",
                "embedding": [0.5, 0.25],
                "owner_entity_id": "e-1",
                "scope": "platform",
                "destination": "cloud",
                "language": "en",
                "top_k": 6,
            })
        );
        // An integer, as JSON.stringify writes it, never 6.0.
        assert!(req.body.as_deref().unwrap().contains("\"top_k\":6}"));
    }

    #[test]
    fn sends_no_vector_at_all_when_there_is_none() {
        let p = platform();
        p.on(
            "https://",
            200,
            r#"{"scope":"platform","hit":false,"chunks":[]}"#,
        );
        let query = GroundQuery {
            embedding: None,
            ..q()
        };
        block_on(ground_search(&p, &env(), &query, TOKEN)).unwrap();
        let sent: Value =
            serde_json::from_str(p.requests.borrow()[0].body.as_deref().unwrap()).unwrap();
        assert!(sent.get("embedding").is_none());
    }

    #[test]
    fn never_sends_the_api_cores_extra_fields_and_clamps_top_k() {
        let p = platform();
        p.on(
            "https://",
            200,
            r#"{"scope":"platform","hit":false,"chunks":[]}"#,
        );
        for (top_k, want) in [
            (6.0, json!(6)),
            (0.0, json!(1)),
            (99.0, json!(20)),
            (f64::NAN, json!(null)),
        ] {
            block_on(ground_search(
                &p,
                &env(),
                &GroundQuery { top_k, ..q() },
                TOKEN,
            ))
            .unwrap();
            let last = p.requests.borrow().last().unwrap().clone();
            let sent: Value = serde_json::from_str(last.body.as_deref().unwrap()).unwrap();
            assert!(sent.get("granted_scopes").is_none());
            assert!(sent.get("needs_own_information").is_none());
            assert_eq!(sent["top_k"], want);
        }
    }

    #[test]
    fn is_used_only_for_a_signed_in_person_with_the_api_configured() {
        assert!(ground_store_configured(&env(), Some(TOKEN)));
        assert!(!ground_store_configured(&env(), None));
        assert!(!ground_store_configured(&env(), Some("")));
        let mut no_api = env();
        no_api.nyuchi_api_url = None;
        assert!(!ground_store_configured(&no_api, Some(TOKEN)));
        no_api.nyuchi_api_url = Some(String::new());
        assert!(!ground_store_configured(&no_api, Some(TOKEN)));
    }

    // A refusal is the API answering, an outage is skipped fast

    #[test]
    fn a_rule_1_refusal_409_comes_back_as_itself_and_does_not_open_the_breaker() {
        let p = platform();
        p.on(
            "https://",
            409,
            r#"{"detail":{"error":"edge_embedding_forbidden"}}"#,
        );
        let e = block_on(ground_search(&p, &env(), &q(), TOKEN)).unwrap_err();
        assert_eq!(e, GroundStoreError::new(409, "edge_embedding_forbidden"));
        assert!(!ground_store_down(p.now_ms()));
    }

    #[test]
    fn any_other_4xx_never_opens_the_breaker_and_closes_an_open_one() {
        for status in [400, 404, 409, 422, 429] {
            let p = platform();
            set_down_until(p.now_ms() - 1);
            p.on("https://", status, "not json");
            let e = block_on(ground_search(&p, &env(), &q(), TOKEN)).unwrap_err();
            assert_eq!(e, GroundStoreError::new(status, format!("http_{status}")));
            assert!(!ground_store_down(p.now_ms()), "{status}");
        }
    }

    #[test]
    fn a_refused_session_401_403_skips_the_api_for_minutes() {
        for status in [401, 403] {
            let p = platform();
            p.on(
                "https://",
                status,
                r#"{"detail":"Invalid or expired token"}"#,
            );
            let e = block_on(ground_search(&p, &env(), &q(), TOKEN)).unwrap_err();
            assert_eq!(e, GroundStoreError::new(status, "Invalid or expired token"));
            let e = block_on(ground_search(&p, &env(), &q(), TOKEN)).unwrap_err();
            assert_eq!(e.status, 503);
            assert_eq!(fetches(&p), 1);
            *p.now.borrow_mut() += GROUND_COOLDOWN_MS + 1;
            assert!(ground_store_down(p.now_ms()));
            *p.now.borrow_mut() += GROUND_REFUSED_KEY_COOLDOWN_MS;
            assert!(!ground_store_down(p.now_ms()));
        }
    }

    #[test]
    fn a_5xx_opens_the_breaker_the_next_search_fails_at_once_and_it_closes_after_the_cooldown() {
        let p = platform();
        p.on("https://", 502, "bad gateway");
        let e = block_on(ground_search(&p, &env(), &q(), TOKEN)).unwrap_err();
        assert_eq!(e.status, 502);
        assert!(ground_store_down(p.now_ms()));
        let e = block_on(ground_search(&p, &env(), &q(), TOKEN)).unwrap_err();
        assert_eq!(
            e,
            GroundStoreError::new(503, "ground_unavailable: skipped while the breaker is open")
        );
        assert_eq!(fetches(&p), 1);
        *p.now.borrow_mut() += GROUND_COOLDOWN_MS + 1;
        assert!(!ground_store_down(p.now_ms()));
    }

    #[test]
    fn a_network_failure_or_the_time_limit_counts_as_down_and_the_limit_is_short() {
        let p = platform();
        p.on_fn(|_| Some(Err("timeout".into())));
        let e = block_on(ground_search(&p, &env(), &q(), TOKEN)).unwrap_err();
        assert_eq!(
            e,
            GroundStoreError::new(503, "ground_unavailable: TimeoutError")
        );
        assert!(ground_store_down(p.now_ms()));
        const { assert!(GROUND_TIMEOUT_MS <= 1500) };

        let p = platform();
        let e = block_on(ground_search(&p, &env(), &q(), TOKEN)).unwrap_err();
        assert_eq!(
            e,
            GroundStoreError::new(503, "ground_unavailable: TypeError")
        );
        assert!(ground_store_down(p.now_ms()));
        *p.now.borrow_mut() += GROUND_COOLDOWN_MS;
        assert!(!ground_store_down(p.now_ms()));
    }

    #[test]
    fn the_persons_token_never_appears_in_an_error() {
        for status in [401, 500] {
            let p = platform();
            p.on(
                "https://",
                status,
                &format!(r#"{{"detail":{{"msg":"{TOKEN}"}}}}"#),
            );
            let e = block_on(ground_search(&p, &env(), &q(), TOKEN)).unwrap_err();
            assert!(!e.to_string().contains(TOKEN));
        }
    }

    // ground() with Ground in Mongo configured

    #[test]
    fn an_api_key_caller_never_reaches_the_nyuchi_api() {
        let p = platform();
        p.on_fn(|r| {
            assert_ne!(
                host(&r.url),
                "api.example.com",
                "the Nyuchi API must not be asked without a person"
            );
            Some(Ok(HttpResponse {
                status: 503,
                headers: vec![],
                body: "down".into(),
            }))
        });
        let cfg = env();
        block_on(ground(
            &p,
            &cfg,
            |b| core_via(&p, &cfg, b),
            &equal(),
            "e-1",
            &cloud(),
            Some("en"),
            None,
        ))
        .unwrap();
        assert!(p
            .requests
            .borrow()
            .iter()
            .all(|r| host(&r.url) != "api.example.com"));
        assert!(fetches(&p) >= 1);
    }

    #[test]
    fn cites_what_the_api_found_and_never_asks_core() {
        let p = platform();
        let answer = json!({ "scope": "platform", "hit": true, "chunks": [section_56()], "needs_own_information": false });
        p.on_fn(move |r| {
            assert_ne!(host(&r.url), "core.invalid", "Core must not be asked");
            Some(Ok(HttpResponse {
                status: 200,
                headers: vec![],
                body: answer.to_string(),
            }))
        });
        let cfg = env();
        let r = block_on(ground(
            &p,
            &cfg,
            |b| core_via(&p, &cfg, b),
            &equal(),
            "e-1",
            &cloud(),
            Some("en"),
            Some(TOKEN),
        ))
        .unwrap();
        assert!(r.hit);
        assert!(!r.unavailable);
        assert!(r.system_prompt.contains("--- SOURCES ---"));
        assert!(r.system_prompt.contains("Section 56"));
        assert!(p
            .requests
            .borrow()
            .iter()
            .all(|r| host(&r.url) == "api.example.com"));
        // The token went to the Nyuchi API URL and nowhere else.
        for r in p.requests.borrow().iter() {
            assert!(r.url.starts_with("https://api.example.com/"));
        }
    }

    #[test]
    fn an_api_outage_leaves_the_answer_to_go_on_without_sources_said_honestly() {
        let p = platform();
        p.on("https://", 503, "down");
        let cfg = env();
        let r = block_on(ground(
            &p,
            &cfg,
            |b| core_via(&p, &cfg, b),
            &equal(),
            "e-1",
            &cloud(),
            Some("en"),
            Some(TOKEN),
        ))
        .unwrap();
        assert!(!r.hit);
        assert!(r.unavailable);
    }
}
