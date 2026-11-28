//! The gateway's routes, as `shamwari/gateway/src/index.ts` dispatches them.
//!
//! | Route                       | Handler                                  |
//! | --------------------------- | ---------------------------------------- |
//! | `GET /health` (any method)  | here                                     |
//! | `GET /v1/models`            | [`crate::gateway::handle_models`]        |
//! | `POST /v1/ground/context`   | [`crate::gateway::handle_ground_context`] |
//! | `POST /v1/channels/sms`     | [`crate::gateway::handle_sms`]           |
//! | `POST /v1/channels/ussd`    | [`crate::gateway::handle_ussd`]          |
//! | `POST /v1/responses`        | [`crate::gateway::handle_open_responses`] |
//! | `POST /v1/chat/completions` | [`crate::gateway::handle_chat`]          |
//!
//! Anything else is `404 {"error":"not_found"}`.
//!
//! Three layers, as in TypeScript:
//!
//! - [`dispatch`] is the TypeScript `worker.fetch`: the routes, no CORS, and
//!   an `Err` for an exception that escaped a handler. The channels and
//!   `/v1/responses` call it in-process for the chat path.
//! - [`route`] answers such an `Err` `500 internal_error` with the cleaned
//!   reason (the `try/catch` of the exported `fetch`).
//! - [`handle`] is the exported `fetch`: a CORS preflight answers early, and
//!   CORS wraps every other response from outside, so a browser sees it on
//!   errors, the 500 included.

use crate::config::Config;
use crate::failure::{console, from_error, public_reasons, FailureStep};
use crate::gateway::Handled;
use crate::http::{Req, Resp};
use crate::platform::Platform;
use serde_json::json;
use std::future::Future;
use std::pin::Pin;

/// The version `/health` reports. Keep in step with Cargo.toml.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// A boxed future, so the chat path can be re-entered from inside a handler
/// (`/v1/responses` and the channels call [`dispatch`] again).
pub type Dispatched<'a> = Pin<Box<dyn Future<Output = Handled> + 'a>>;

fn health(cfg: &Config) -> Resp {
    Resp::json(
        &json!({
            "ok": true,
            "service": "shamwari-gateway",
            "version": VERSION,
            "mind": cfg.mind_available(),
            "runtime": "rust",
        }),
        200,
    )
}

/// The routes without CORS (the TypeScript `worker.fetch`).
pub fn dispatch<'a, P: Platform>(p: &'a P, cfg: &'a Config, req: Req) -> Dispatched<'a> {
    Box::pin(async move {
        use crate::gateway as g;
        let path = req.path();
        match (req.method.as_str(), path.as_str()) {
            (_, "/health") => Ok(health(cfg)),
            ("GET", "/v1/models") => g::handle_models(p, cfg, &req).await,
            ("POST", "/v1/ground/context") => g::handle_ground_context(p, cfg, &req).await,
            ("POST", "/v1/channels/sms") => g::handle_sms(p, cfg, &req).await,
            ("POST", "/v1/channels/ussd") => g::handle_ussd(p, cfg, &req).await,
            ("POST", "/v1/responses") => g::handle_open_responses(p, cfg, &req).await,
            ("POST", "/v1/chat/completions") => g::handle_chat(p, cfg, &req).await,
            _ => Ok(Resp::json(&json!({ "error": "not_found" }), 404)),
        }
    })
}

/// The answer to an exception that escaped a handler: readable, with the
/// cleaned reason, never an opaque network failure.
pub fn internal_error(e: &str) -> Resp {
    console::error(&format!("unhandled error {e}"));
    Resp::json(
        &json!({
            "error": "internal_error",
            "reasons": public_reasons(&[from_error(FailureStep::Worker, e)]),
        }),
        500,
    )
}

/// [`dispatch`], with an escaped exception answered `500 internal_error`.
pub async fn route<P: Platform>(p: &P, cfg: &Config, req: Req) -> Resp {
    match dispatch(p, cfg, req).await {
        Ok(r) => r,
        Err(e) => internal_error(&e),
    }
}

/// The Worker's fetch handler: a CORS preflight answers early, and every other
/// response carries the CORS headers (the TypeScript `export default`).
pub async fn handle<P: Platform>(p: &P, cfg: &Config, req: Req) -> Resp {
    if let Some(early) = crate::cors::preflight(cfg, &req) {
        return early;
    }
    let res = route(p, cfg, req.clone()).await;
    crate::cors::with_cors(cfg, &req, res)
}
