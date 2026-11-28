//! The Cloudflare Workers entry point: the only module that touches the
//! runtime. Everything it calls is plain Rust over [`Platform`].

use crate::config::Config;
use crate::failure::console;
use crate::http::{Req, Resp};
use crate::platform::{HttpRequest, HttpResponse, Platform};
use crate::router;
use crate::types::SinkMessage;
use js_sys::{Function, Reflect, JSON};
use serde_json::Value;
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;
use worker::{
    event, AbortController, Context, Env, Fetch, Headers, MessageBatch, MessageExt, Method,
    RequestInit, Response, Result, ScheduleContext, ScheduledEvent,
};

/// The Workers runtime behind [`Platform`].
struct WorkerPlatform {
    env: Env,
}

fn js_err(e: JsValue) -> String {
    e.as_string()
        .or_else(|| {
            Reflect::get(&e, &JsValue::from_str("message"))
                .ok()
                .and_then(|m| m.as_string())
        })
        .unwrap_or_else(|| "javascript exception".to_owned())
}

impl WorkerPlatform {
    fn binding(&self, name: &str) -> Option<JsValue> {
        Reflect::get(&self.env, &JsValue::from_str(name))
            .ok()
            .filter(|v| !v.is_undefined() && !v.is_null())
    }

    /// Call `binding.method(...args)` and await the promise it returns.
    async fn call(
        &self,
        binding: &str,
        method: &str,
        args: &[JsValue],
    ) -> std::result::Result<JsValue, String> {
        let target = self
            .binding(binding)
            .ok_or_else(|| format!("no {binding} binding"))?;
        let f: Function = Reflect::get(&target, &JsValue::from_str(method))
            .map_err(js_err)?
            .dyn_into()
            .map_err(|_| format!("{binding}.{method} is not a function"))?;
        let arr = js_sys::Array::new();
        for a in args {
            arr.push(a);
        }
        let out = f.apply(&target, &arr).map_err(js_err)?;
        JsFuture::from(js_sys::Promise::resolve(&out))
            .await
            .map_err(js_err)
    }
}

fn to_js(v: &Value) -> std::result::Result<JsValue, String> {
    JSON::parse(&v.to_string()).map_err(js_err)
}

impl Platform for WorkerPlatform {
    async fn fetch(&self, req: HttpRequest) -> std::result::Result<HttpResponse, String> {
        let headers = Headers::new();
        for (k, v) in &req.headers {
            headers.set(k, v).map_err(|e| e.to_string())?;
        }
        let mut init = RequestInit::new();
        init.with_method(Method::from(req.method.clone()))
            .with_headers(headers);
        if let Some(body) = &req.body {
            init.with_body(Some(JsValue::from_str(body)));
        }
        let request = worker::Request::new_with_init(&req.url, &init).map_err(|e| e.to_string())?;
        let mut response = match req.timeout_ms {
            Some(ms) => {
                let controller = AbortController::default();
                let signal = controller.signal();
                let timer = worker::Delay::from(std::time::Duration::from_millis(ms as u64));
                let fetcher = Fetch::Request(request);
                let fetch = fetcher.send_with_signal(&signal);
                futures_util_select(fetch, timer, controller).await?
            }
            None => Fetch::Request(request)
                .send()
                .await
                .map_err(|e| e.to_string())?,
        };
        let status = response.status_code();
        let headers = response.headers().entries().collect();
        let body = response.text().await.map_err(|e| e.to_string())?;
        Ok(HttpResponse {
            status,
            headers,
            body,
        })
    }

    async fn ai_run(
        &self,
        model: &str,
        input: &Value,
        options: Option<&Value>,
    ) -> std::result::Result<Value, String> {
        let mut args = vec![JsValue::from_str(model), to_js(input)?];
        if let Some(o) = options {
            args.push(to_js(o)?);
        }
        let out = self.call("AI", "run", &args).await?;
        let json = JSON::stringify(&out)
            .map_err(js_err)?
            .as_string()
            .ok_or("AI returned a non-serialisable value")?;
        serde_json::from_str(&json).map_err(|e| e.to_string())
    }

    async fn kv_get(&self, key: &str) -> std::result::Result<Option<String>, String> {
        let v = self
            .call("AUTH_CACHE", "get", &[JsValue::from_str(key)])
            .await?;
        Ok(v.as_string())
    }

    async fn kv_put(
        &self,
        key: &str,
        value: &str,
        ttl: Option<u64>,
    ) -> std::result::Result<(), String> {
        let mut args = vec![JsValue::from_str(key), JsValue::from_str(value)];
        if let Some(t) = ttl {
            args.push(to_js(&serde_json::json!({ "expirationTtl": t }))?);
        }
        self.call("AUTH_CACHE", "put", &args).await.map(|_| ())
    }

    async fn kv_delete(&self, key: &str) -> std::result::Result<(), String> {
        self.call("AUTH_CACHE", "delete", &[JsValue::from_str(key)])
            .await
            .map(|_| ())
    }

    async fn queue_send(&self, message: &Value) -> std::result::Result<(), String> {
        self.call("SINK", "send", &[to_js(message)?])
            .await
            .map(|_| ())
    }

    async fn secret_store_get(&self, binding: &str) -> std::result::Result<Option<String>, String> {
        let v = self.call(binding, "get", &[]).await?;
        Ok(v.as_string())
    }

    fn now_ms(&self) -> u64 {
        js_sys::Date::now() as u64
    }

    fn random_bytes(&self, n: usize) -> Vec<u8> {
        let mut buf = vec![0u8; n];
        getrandom::fill(&mut buf).expect("crypto.getRandomValues");
        buf
    }
}

/// The fetch, or a timeout that aborts it, whichever finishes first.
async fn futures_util_select(
    fetch: impl std::future::Future<Output = worker::Result<worker::Response>>,
    timer: worker::Delay,
    controller: AbortController,
) -> std::result::Result<worker::Response, String> {
    use futures_util::future::{select, Either};
    futures_util::pin_mut!(fetch);
    match select(fetch, timer).await {
        Either::Left((r, _)) => r.map_err(|e| e.to_string()),
        Either::Right(_) => {
            controller.abort();
            Err("timeout".to_owned())
        }
    }
}

/// A binding as a string, whether it is a `[vars]` entry or a Worker secret.
fn binding(env: &Env, name: &str) -> Option<String> {
    env.var(name)
        .map(|v| v.to_string())
        .or_else(|_| env.secret(name).map(|s| s.to_string()))
        .ok()
}

fn config_from(env: &Env) -> Config {
    let b = |n: &str| binding(env, n);
    Config {
        cf_account_id: b("CF_ACCOUNT_ID"),
        cf_gateway_id: b("CF_GATEWAY_ID"),
        cf_route_name: b("CF_ROUTE_NAME"),
        core_url: b("CORE_URL"),
        shamwari_core_token: b("SHAMWARI_CORE_TOKEN"),
        own_sites: b("OWN_SITES"),
        nyuchi_api_url: b("NYUCHI_API_URL"),
        servers_api_enabled: b("SERVERS_API_ENABLED"),
        servers_api_url: b("SERVERS_API_URL"),
        servers_api_token: b("SERVERS_API_TOKEN"),
        servers_api_internal_key: b("SERVERS_API_INTERNAL_KEY"),
        workos_client_id: b("WORKOS_CLIENT_ID"),
        workos_issuer: b("WORKOS_ISSUER"),
        workos_audience: b("WORKOS_AUDIENCE"),
        workos_jwks_url: b("WORKOS_JWKS_URL"),
        workos_monthly_token_cap: b("WORKOS_MONTHLY_TOKEN_CAP"),
        workos_max_token_age_seconds: b("WORKOS_MAX_TOKEN_AGE_SECONDS"),
        cors_origins: b("CORS_ORIGINS"),
        embedding_model: b("EMBEDDING_MODEL"),
        ground_top_k: b("GROUND_TOP_K"),
        auth_cache_ttl: b("AUTH_CACHE_TTL"),
        economy_model: b("ECONOMY_MODEL"),
        standard_model: b("STANDARD_MODEL"),
        surface: b("SURFACE"),
        mind_available: b("MIND_AVAILABLE"),
        channel_webhook_secret: b("CHANNEL_WEBHOOK_SECRET"),
        channel_api_key: b("CHANNEL_API_KEY"),
        sink_enabled: b("SINK_ENABLED"),
    }
}

#[event(fetch)]
async fn fetch(mut request: worker::Request, env: Env, _ctx: Context) -> Result<Response> {
    let method = request.method().to_string();
    let url = request.url()?.to_string();
    let headers: Vec<(String, String)> = request
        .headers()
        .entries()
        .map(|(k, v)| (k.to_ascii_lowercase(), v))
        .collect();
    let body = if matches!(method.as_str(), "GET" | "HEAD" | "OPTIONS") {
        String::new()
    } else {
        request.text().await.unwrap_or_default()
    };
    let platform = WorkerPlatform { env: env.clone() };
    let cfg = config_from(&env);
    let res = router::handle(
        &platform,
        &cfg,
        Req {
            method,
            url,
            headers,
            body,
        },
    )
    .await;
    to_response(res)
}

/// The runtime's `Response` for one of ours. A null-body status (204, 304,
/// ...) is built with no body at all, as `new Response(null, ...)`: the
/// runtime refuses one with a body, even an empty string. Headers are
/// appended, never set, so a header a route repeats stays repeated.
fn to_response(res: Resp) -> Result<Response> {
    let headers = Headers::new();
    for (k, v) in &res.headers {
        headers.append(k, v)?;
    }
    let builder = Response::builder()
        .with_status(res.status)
        .with_headers(headers);
    Ok(if Resp::is_null_body_status(res.status) {
        builder.empty()
    } else {
        builder.fixed(res.body.into_bytes())
    })
}

/// The `shamwari-sink` consumer (TypeScript `queue`): drain the batch into
/// Core, acknowledge what was settled, and fail the batch on a transient
/// error so the queue retries what is left (five times, then the DLQ).
///
/// Not bound in wrangler.toml until the cutover: a queue has one consumer,
/// and until then it is the TypeScript Worker.
#[event(queue)]
async fn queue(batch: MessageBatch<SinkMessage>, env: Env, _ctx: Context) -> Result<()> {
    let platform = WorkerPlatform { env: env.clone() };
    let cfg = config_from(&env);
    let raw: Vec<_> = batch.raw_iter().collect();
    let mut docs = Vec::with_capacity(raw.len());
    let mut index = Vec::with_capacity(raw.len());
    for (i, m) in raw.iter().enumerate() {
        let parsed = JSON::stringify(&m.body())
            .ok()
            .and_then(|s| s.as_string())
            .and_then(|s| serde_json::from_str::<SinkMessage>(&s).ok());
        match parsed {
            Some(doc) => {
                docs.push(doc);
                index.push(i);
            }
            // Core would refuse it as malformed (a permanent 400), and the
            // TypeScript consumer parks such a document: acknowledged, with
            // an identifier-free log line, never retried into the DLQ.
            None => {
                console::error("sink parked a document: status=400 error=unreadable_message");
                m.ack();
            }
        }
    }
    let drained = crate::gateway::handle_queue(&platform, &cfg, &docs).await;
    for i in &drained.acked {
        if let Some(m) = index.get(*i).and_then(|j| raw.get(*j)) {
            m.ack();
        }
    }
    match drained.error {
        Some(e) => Err(worker::Error::RustError(e)),
        None => Ok(()),
    }
}

/// The weekly degradation probe (TypeScript `scheduled`). No cron triggers it
/// until the cutover, so the probe never runs twice a week.
#[event(scheduled)]
async fn scheduled(_event: ScheduledEvent, env: Env, _ctx: ScheduleContext) {
    let platform = WorkerPlatform { env: env.clone() };
    let cfg = config_from(&env);
    crate::gateway::handle_scheduled(&platform, &cfg).await;
}
