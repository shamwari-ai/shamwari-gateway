//! The world the handler tests run in: the real router over a [`FakePlatform`]
//! with a scripted Core, AI Gateway and Workers AI, as the TypeScript tests
//! stub `fetch` and the bindings for `worker.fetch`.
//!
//! Rule 1 is stated as counts here, as it is in TypeScript: every
//! provider-bound call (an embedding, a completion, the Workers AI fallback)
//! is recorded with the scope it was made under, and a test asserts how many
//! there were. A personal path to a provider is a leak even if the response
//! is an error.
//!
//! No credential is a literal: the API keys and the operator's secret are
//! drawn at run time.

use crate::config::Config;
use crate::http::{Req, Resp};
use crate::platform::{HttpRequest, HttpResponse, Platform};
use crate::testing::{block_on, FakePlatform};
use crate::types::AuthContext;
use serde_json::{json, Value};
use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::OnceLock;

pub const PERSONAL_Q: &str = "How much PAYE do I owe on my salary?";
pub const PLAIN_Q: &str = "What does section 48 of the Constitution say?";
pub const LIFE_Q: &str = "Do I have a right to life?";
pub const SITE: &str = "https://shamwari.ai";
pub const GATEWAY: &str = "https://gateway.ai.cloudflare.com/";
pub const EMBEDDING_MODEL: &str = "@cf/baai/bge-m3";

fn random_hex(n: usize) -> String {
    let mut b = vec![0u8; n];
    getrandom::fill(&mut b).expect("random bytes");
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn once(cell: &'static OnceLock<String>, f: impl FnOnce() -> String) -> &'static str {
    cell.get_or_init(f)
}

/// The caller's API key, drawn once per run.
pub fn api_key() -> &'static str {
    static K: OnceLock<String> = OnceLock::new();
    once(&K, || format!("sk_shamwari_{}", random_hex(16)))
}

/// The channels' own API key, drawn once per run.
pub fn channel_key() -> &'static str {
    static K: OnceLock<String> = OnceLock::new();
    once(&K, || format!("sk_shamwari_{}", random_hex(16)))
}

/// The SMS/USSD operator's shared secret, drawn once per run.
pub fn operator_secret() -> &'static str {
    static K: OnceLock<String> = OnceLock::new();
    once(&K, || random_hex(16))
}

pub fn chunk(scope: &str) -> Value {
    json!({
        "content": "Every person has the right to life.",
        "heading": "Section 48",
        "title": "Constitution of Zimbabwe",
        "authority": "Parliament of Zimbabwe",
        "resource_type": "legislation",
        "source_url": "https://example.invalid/constitution",
        "effective_from": "2013-05-22",
        "scope": scope,
    })
}

/// The auth context the TypeScript tests cache: a standard key with consent.
pub fn auth() -> AuthContext {
    AuthContext {
        api_key_id: "k1".into(),
        owner_entity_id: "e1".into(),
        owner_person_id: None,
        key_type: "standard".into(),
        tier: "community".into(),
        monthly_token_cap: 100_000,
        scopes: None,
        training_consent: Some(true),
    }
}

type CoreAnswer = Rc<dyn Fn(&Value) -> Value>;

/// What the fakes answer. Changed at any time, read on every call.
pub struct Settings {
    /// Core's `/ground/search` answer, given the body it was sent. `None`:
    /// one chunk for the scope that was asked.
    pub core: Option<CoreAnswer>,
    /// Core answers this status (body `{}`) instead of an answer.
    pub core_status: Option<u16>,
    /// Core's search is unreachable (the fetch throws).
    pub core_unreachable: bool,
    /// The quota rollup's total.
    pub rollup_total: f64,
    /// The AI Gateway answers 503 `down`.
    pub gateway_fails: bool,
    /// Workers AI throws on anything that is not an embedding.
    pub fallback_fails: bool,
    /// Workers AI throws on an embedding.
    pub embed_fails: bool,
    /// The completion text.
    pub answer: String,
    /// `cf-aig-provider` / `cf-aig-model` on the gateway's answer.
    pub served: Option<(String, String)>,
    pub finish_reason: Option<String>,
    pub usage: Value,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            core: None,
            core_status: None,
            core_unreachable: false,
            rollup_total: 0.0,
            gateway_fails: false,
            fallback_fails: false,
            embed_fails: false,
            answer: "generated".into(),
            served: None,
            finish_reason: None,
            usage: json!({}),
        }
    }
}

/// [`FakePlatform`], with a KV that can fail for the idempotency keys alone
/// (as a KV outage would) and that records each write's TTL.
pub struct TestPlatform {
    pub fake: FakePlatform,
    pub idem_kv_down: Cell<bool>,
    pub kv_puts: RefCell<Vec<(String, Option<u64>)>>,
}

impl Platform for TestPlatform {
    async fn fetch(&self, req: HttpRequest) -> Result<HttpResponse, String> {
        self.fake.fetch(req).await
    }
    async fn ai_run(&self, m: &str, i: &Value, o: Option<&Value>) -> Result<Value, String> {
        self.fake.ai_run(m, i, o).await
    }
    async fn kv_get(&self, key: &str) -> Result<Option<String>, String> {
        if self.idem_kv_down.get() && key.starts_with("idem:") {
            return Err("KV down".into());
        }
        self.fake.kv_get(key).await
    }
    async fn kv_put(&self, key: &str, value: &str, ttl: Option<u64>) -> Result<(), String> {
        if self.idem_kv_down.get() && key.starts_with("idem:") {
            return Err("KV down".into());
        }
        self.kv_puts.borrow_mut().push((key.to_owned(), ttl));
        self.fake.kv_put(key, value, ttl).await
    }
    async fn kv_delete(&self, key: &str) -> Result<(), String> {
        self.fake.kv_delete(key).await
    }
    async fn queue_send(&self, message: &Value) -> Result<(), String> {
        self.fake.queue_send(message).await
    }
    async fn secret_store_get(&self, binding: &str) -> Result<Option<String>, String> {
        self.fake.secret_store_get(binding).await
    }
    fn now_ms(&self) -> u64 {
        self.fake.now_ms()
    }
    fn random_bytes(&self, n: usize) -> Vec<u8> {
        self.fake.random_bytes(n)
    }
}

pub struct World {
    pub p: TestPlatform,
    pub cfg: Config,
    pub s: Rc<RefCell<Settings>>,
}

/// The environment the TypeScript tests give the Worker.
pub fn env() -> Config {
    Config {
        core_url: Some("https://core.example".into()),
        shamwari_core_token: Some(random_hex(8)),
        embedding_model: Some(EMBEDDING_MODEL.into()),
        ground_top_k: Some("6".into()),
        auth_cache_ttl: Some("300".into()),
        economy_model: Some("@cf/zai-org/glm-5.3-flash".into()),
        standard_model: Some("@cf/zai-org/glm-5.3".into()),
        surface: Some("shamwari.ai".into()),
        mind_available: Some("false".into()),
        cf_account_id: Some("acct".into()),
        cf_gateway_id: Some("gw".into()),
        cf_route_name: Some("shamwari".into()),
        channel_webhook_secret: Some(operator_secret().into()),
        channel_api_key: Some(channel_key().into()),
        ..Default::default()
    }
}

impl World {
    pub fn new() -> World {
        World::with(env())
    }

    pub fn mind() -> World {
        let mut cfg = env();
        cfg.mind_available = Some("true".into());
        World::with(cfg)
    }

    pub fn with(cfg: Config) -> World {
        // Every breaker is per isolate (per test thread here): close them, so
        // one world's outage is not the next world's.
        crate::core::reset_core_breaker();
        crate::ground_store::reset_ground_store_breaker();
        crate::own_sites::reset_own_site_breakers();
        let _ = crate::failure::console::take();

        let s = Rc::new(RefCell::new(Settings::default()));
        let fake = FakePlatform::new();
        fake.secrets
            .borrow_mut()
            .insert(crate::ai_gateway::CF_AIG_TOKEN.into(), random_hex(8));

        let st = s.clone();
        fake.on_fn(move |r| {
            r.url.contains("/rollup").then(|| {
                Ok(HttpResponse {
                    status: 200,
                    headers: vec![],
                    body: json!({ "total_tokens": st.borrow().rollup_total }).to_string(),
                })
            })
        });
        let st = s.clone();
        fake.on_fn(move |r| {
            if !r.url.contains("/ground/search") {
                return None;
            }
            let st = st.borrow();
            if st.core_unreachable {
                return Some(Err("core unreachable".into()));
            }
            if let Some(status) = st.core_status {
                return Some(Ok(HttpResponse {
                    status,
                    headers: vec![],
                    body: "{}".into(),
                }));
            }
            let body: Value = serde_json::from_str(r.body.as_deref().unwrap_or("null")).ok()?;
            let answer = match &st.core {
                Some(f) => f(&body),
                None => {
                    json!({ "chunks": [chunk(body["scope"].as_str().unwrap_or(""))], "hit": true, "scope": body["scope"] })
                }
            };
            Some(Ok(HttpResponse {
                status: 200,
                headers: vec![],
                body: answer.to_string(),
            }))
        });
        let st = s.clone();
        fake.on_fn(move |r| {
            if !r.url.starts_with(GATEWAY) {
                return None;
            }
            let st = st.borrow();
            if st.gateway_fails {
                return Some(Ok(HttpResponse {
                    status: 503,
                    headers: vec![],
                    body: "down".into(),
                }));
            }
            let mut choice = json!({ "message": { "content": st.answer } });
            if let Some(f) = &st.finish_reason {
                choice["finish_reason"] = json!(f);
            }
            let headers = match &st.served {
                Some((provider, model)) => vec![
                    ("cf-aig-provider".to_owned(), provider.clone()),
                    ("cf-aig-model".to_owned(), model.clone()),
                ],
                None => vec![],
            };
            Some(Ok(HttpResponse {
                status: 200,
                headers,
                body: json!({ "choices": [choice], "usage": st.usage }).to_string(),
            }))
        });
        let st = s.clone();
        fake.on_ai(move |model, _input| {
            let st = st.borrow();
            if model == EMBEDDING_MODEL {
                return Some(if st.embed_fails {
                    Err("workers ai down".into())
                } else {
                    Ok(json!({ "data": [vec![0.1; 1024]] }))
                });
            }
            Some(if st.fallback_fails {
                Err("workers ai down".into())
            } else {
                Ok(json!({ "response": st.answer }))
            })
        });

        let w = World {
            p: TestPlatform {
                fake,
                idem_kv_down: Cell::new(false),
                kv_puts: RefCell::new(vec![]),
            },
            cfg,
            s,
        };
        w.set_auth(api_key(), &auth());
        w.set_auth(channel_key(), &auth());
        w
    }

    /// Cache `ctx` as the verified context of `key`, as the KV cache holds it.
    pub fn set_auth(&self, key: &str, ctx: &AuthContext) {
        self.p.fake.kv.borrow_mut().insert(
            format!("auth:{}", crate::auth::sha256_hex(key)),
            serde_json::to_string(ctx).unwrap(),
        );
    }

    /// The caller's context, changed by `f`.
    pub fn auth_with(&self, f: impl FnOnce(&mut AuthContext)) {
        let mut a = auth();
        f(&mut a);
        self.set_auth(api_key(), &a);
    }

    pub fn scopes(&self, scopes: &[&str]) {
        let v: Vec<crate::types::Scope> =
            serde_json::from_value(json!(scopes)).expect("scope names");
        self.auth_with(|a| a.scopes = Some(v));
    }

    pub fn settings(&self) -> std::cell::RefMut<'_, Settings> {
        self.s.borrow_mut()
    }

    /// Through the Worker's own fetch handler (CORS included).
    pub fn send(&self, req: Req) -> Resp {
        block_on(crate::router::handle(&self.p, &self.cfg, req))
    }

    pub fn post_with(&self, path: &str, body: &Value, headers: &[(&str, &str)]) -> Resp {
        let mut h = vec![
            ("authorization".to_owned(), format!("Bearer {}", api_key())),
            ("content-type".to_owned(), "application/json".to_owned()),
        ];
        for (k, v) in headers {
            h.retain(|(x, _)| !x.eq_ignore_ascii_case(k));
            h.push((k.to_ascii_lowercase(), (*v).to_owned()));
        }
        self.send(Req {
            method: "POST".into(),
            url: format!("https://gw.example{path}"),
            headers: h,
            body: body.to_string(),
        })
    }

    pub fn post(&self, path: &str, body: &Value) -> Resp {
        self.post_with(path, body, &[])
    }

    /// A chat completion with one user turn and `extra` fields.
    pub fn chat(&self, q: &str, extra: Value) -> Resp {
        self.post("/v1/chat/completions", &ask(q, extra))
    }

    pub fn chat_with(&self, q: &str, extra: Value, headers: &[(&str, &str)]) -> Resp {
        self.post_with("/v1/chat/completions", &ask(q, extra), headers)
    }

    pub fn context(&self, q: &str, extra: Value) -> Resp {
        self.post("/v1/ground/context", &ask(q, extra))
    }

    // What the world saw.

    pub fn requests(&self) -> Vec<HttpRequest> {
        self.p.fake.requests.borrow().clone()
    }

    /// Every body sent to Core's `/ground/search`.
    pub fn core(&self) -> Vec<Value> {
        self.requests()
            .iter()
            .filter(|r| r.url.contains("/ground/search"))
            .map(|r| serde_json::from_str(r.body.as_deref().unwrap_or("null")).unwrap())
            .collect()
    }

    /// Every completion sent to the AI Gateway: (its `cf-aig-metadata`, its body).
    pub fn completions(&self) -> Vec<(Value, Value)> {
        self.requests()
            .iter()
            .filter(|r| r.url.starts_with(GATEWAY))
            .map(|r| {
                let meta = r
                    .headers
                    .iter()
                    .find(|(k, _)| k.eq_ignore_ascii_case("cf-aig-metadata"))
                    .and_then(|(_, v)| serde_json::from_str::<Value>(v).ok())
                    .unwrap_or(Value::Null);
                let body = serde_json::from_str(r.body.as_deref().unwrap_or("null")).unwrap();
                (meta, body)
            })
            .collect()
    }

    /// The system prompt of the `i`th completion.
    pub fn system_prompt(&self, i: usize) -> String {
        self.completions()[i].1["messages"]
            .as_array()
            .and_then(|ms| ms.iter().find(|m| m["role"] == "system"))
            .and_then(|m| m["content"].as_str())
            .unwrap_or_default()
            .to_owned()
    }

    /// Every `AI.run` call: an embedding or the Workers AI fallback.
    pub fn ai(&self) -> usize {
        self.p.fake.ai_calls.borrow().len()
    }

    pub fn embeds(&self) -> usize {
        self.p
            .fake
            .ai_calls
            .borrow()
            .iter()
            .filter(|(m, _, _)| m == EMBEDDING_MODEL)
            .count()
    }

    /// Every provider-bound call: an embedding, the fallback, a completion.
    pub fn provider_calls(&self) -> usize {
        self.ai() + self.completions().len()
    }

    /// Every scope any request was made under, towards Core or a provider.
    pub fn scopes_seen(&self) -> Vec<String> {
        let mut out: Vec<String> = self
            .core()
            .iter()
            .map(|b| b["scope"].as_str().unwrap_or("?").to_owned())
            .collect();
        out.extend(
            self.completions()
                .into_iter()
                .map(|(m, _)| m["scope"].as_str().unwrap_or("?").to_owned()),
        );
        out
    }

    /// Scopes seen that are not community or platform.
    pub fn unsafe_scopes(&self) -> Vec<String> {
        self.scopes_seen()
            .into_iter()
            .filter(|s| s != "community" && s != "platform")
            .collect()
    }

    /// Everything put on the sink queue.
    pub fn sink(&self) -> Vec<Value> {
        self.p.fake.queue.borrow().clone()
    }

    pub fn collections(&self) -> Vec<String> {
        self.sink()
            .iter()
            .map(|m| m["collection"].as_str().unwrap_or_default().to_owned())
            .collect()
    }

    pub fn doc(&self, collection: &str) -> Option<Value> {
        self.sink()
            .into_iter()
            .find(|m| m["collection"] == collection)
            .map(|m| m["doc"].clone())
    }

    /// Everything touched by a request that must touch nothing.
    pub fn touched(&self) -> usize {
        self.provider_calls() + self.core().len() + self.sink().len()
    }

    pub fn idem_keys(&self) -> Vec<String> {
        self.p
            .fake
            .kv
            .borrow()
            .keys()
            .filter(|k| k.starts_with("idem:"))
            .cloned()
            .collect()
    }
}

/// A chat body with one user turn and `extra` fields (a JSON object).
pub fn ask(q: &str, extra: Value) -> Value {
    let mut b = json!({ "messages": [{ "role": "user", "content": q }] });
    if let Value::Object(m) = extra {
        for (k, v) in m {
            b[k] = v;
        }
    }
    b
}
