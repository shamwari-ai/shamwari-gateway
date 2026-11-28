//! What the gateway needs from its runtime, as one trait.
//!
//! Production implements it with the Workers `fetch`, the `AI` binding, the
//! `AUTH_CACHE` KV namespace, the `SINK` queue, the Secrets Store and
//! `Date.now()` (see `worker_entry`). Tests implement it with scripted
//! answers. Nothing outside `worker_entry` touches the runtime, so every rule
//! the gateway enforces runs under `cargo test` on the host.

use serde_json::Value;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct HttpRequest {
    pub method: String,
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: Option<String>,
    /// Abort the request after this many milliseconds (the TypeScript
    /// gateway's `AbortSignal.timeout`). `None` means the runtime's own limit.
    pub timeout_ms: Option<u32>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct HttpResponse {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: String,
}

impl HttpResponse {
    pub fn ok(&self) -> bool {
        (200..300).contains(&self.status)
    }

    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
}

#[allow(async_fn_in_trait)]
pub trait Platform {
    /// An outbound HTTPS request. `Err` is a network-level failure or a timeout.
    async fn fetch(&self, req: HttpRequest) -> Result<HttpResponse, String>;

    /// Run a Workers AI model. `options` is the third argument of `AI.run`
    /// (the AI Gateway options object), never part of the input.
    async fn ai_run(
        &self,
        model: &str,
        input: &Value,
        options: Option<&Value>,
    ) -> Result<Value, String>;

    /// `AUTH_CACHE.get(key)`.
    async fn kv_get(&self, key: &str) -> Result<Option<String>, String>;

    /// `AUTH_CACHE.put(key, value, { expirationTtl })`.
    async fn kv_put(&self, key: &str, value: &str, ttl_seconds: Option<u64>) -> Result<(), String>;

    /// `AUTH_CACHE.delete(key)`.
    async fn kv_delete(&self, key: &str) -> Result<(), String>;

    /// `SINK.send(message)`: the queue that drains into Core.
    async fn queue_send(&self, message: &Value) -> Result<(), String>;

    /// A Secrets Store binding's value (`CF_AIG_TOKEN.get()`).
    async fn secret_store_get(&self, binding: &str) -> Result<Option<String>, String>;

    /// Milliseconds since the Unix epoch.
    fn now_ms(&self) -> u64;

    /// `n` cryptographically random bytes.
    fn random_bytes(&self, n: usize) -> Vec<u8>;
}
