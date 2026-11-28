//! A scripted [`Platform`] for tests: every port's tests use this one, so the
//! runtime is faked the same way everywhere.

use crate::platform::{HttpRequest, HttpResponse, Platform};
use serde_json::Value;
use std::cell::RefCell;
use std::collections::HashMap;

type Responder = Box<dyn Fn(&HttpRequest) -> Option<Result<HttpResponse, String>>>;
type AiResponder = Box<dyn Fn(&str, &Value) -> Option<Result<Value, String>>>;

#[derive(Default)]
pub struct FakePlatform {
    /// Tried in order; the first that answers wins. No answer is a network error.
    pub responders: RefCell<Vec<Responder>>,
    pub ai: RefCell<Vec<AiResponder>>,
    /// Every outbound request, in order.
    pub requests: RefCell<Vec<HttpRequest>>,
    /// Every `AI.run` call: (model, input, options).
    pub ai_calls: RefCell<Vec<(String, Value, Option<Value>)>>,
    pub kv: RefCell<HashMap<String, String>>,
    pub queue: RefCell<Vec<Value>>,
    pub secrets: RefCell<HashMap<String, String>>,
    pub now: RefCell<u64>,
}

impl FakePlatform {
    pub fn new() -> Self {
        let p = Self::default();
        *p.now.borrow_mut() = 1_767_225_600_000; // 2026-01-01T00:00:00Z
        p
    }

    /// Answer requests whose URL starts with `prefix`.
    pub fn on(&self, prefix: &str, status: u16, body: &str) -> &Self {
        let (prefix, body) = (prefix.to_owned(), body.to_owned());
        self.responders.borrow_mut().push(Box::new(move |r| {
            r.url.starts_with(&prefix).then(|| {
                Ok(HttpResponse {
                    status,
                    headers: vec![],
                    body: body.clone(),
                })
            })
        }));
        self
    }

    pub fn on_fn(
        &self,
        f: impl Fn(&HttpRequest) -> Option<Result<HttpResponse, String>> + 'static,
    ) -> &Self {
        self.responders.borrow_mut().push(Box::new(f));
        self
    }

    pub fn on_ai(
        &self,
        f: impl Fn(&str, &Value) -> Option<Result<Value, String>> + 'static,
    ) -> &Self {
        self.ai.borrow_mut().push(Box::new(f));
        self
    }
}

impl Platform for FakePlatform {
    async fn fetch(&self, req: HttpRequest) -> Result<HttpResponse, String> {
        self.requests.borrow_mut().push(req.clone());
        for r in self.responders.borrow().iter() {
            if let Some(out) = r(&req) {
                return out;
            }
        }
        Err(format!("no fake answer for {} {}", req.method, req.url))
    }

    async fn ai_run(
        &self,
        model: &str,
        input: &Value,
        options: Option<&Value>,
    ) -> Result<Value, String> {
        self.ai_calls
            .borrow_mut()
            .push((model.to_owned(), input.clone(), options.cloned()));
        for r in self.ai.borrow().iter() {
            if let Some(out) = r(model, input) {
                return out;
            }
        }
        Err(format!("no fake answer for AI model {model}"))
    }

    async fn kv_get(&self, key: &str) -> Result<Option<String>, String> {
        Ok(self.kv.borrow().get(key).cloned())
    }

    async fn kv_put(&self, key: &str, value: &str, _ttl: Option<u64>) -> Result<(), String> {
        self.kv
            .borrow_mut()
            .insert(key.to_owned(), value.to_owned());
        Ok(())
    }

    async fn kv_delete(&self, key: &str) -> Result<(), String> {
        self.kv.borrow_mut().remove(key);
        Ok(())
    }

    async fn queue_send(&self, message: &Value) -> Result<(), String> {
        self.queue.borrow_mut().push(message.clone());
        Ok(())
    }

    async fn secret_store_get(&self, binding: &str) -> Result<Option<String>, String> {
        Ok(self.secrets.borrow().get(binding).cloned())
    }

    fn now_ms(&self) -> u64 {
        *self.now.borrow()
    }

    fn random_bytes(&self, n: usize) -> Vec<u8> {
        (0..n)
            .map(|i| (i as u8).wrapping_mul(37).wrapping_add(11))
            .collect()
    }
}

/// Run a future to completion on the test thread.
pub fn block_on<F: std::future::Future>(f: F) -> F::Output {
    futures::executor::block_on(f)
}
