//! The gateway's routes, as `shamwari/gateway/src/index.ts` dispatches them.
//!
//! | Route                       | Module          |
//! | --------------------------- | --------------- |
//! | `GET /health`               | here            |
//! | `GET /v1/models`            | [`crate::models`] |
//! | `POST /v1/ground/context`   | [`crate::ground`] |
//! | `POST /v1/channels/sms`     | [`crate::channels`] |
//! | `POST /v1/channels/ussd`    | [`crate::channels`] |
//! | `POST /v1/responses`        | [`crate::responses`] |
//! | `POST /v1/chat/completions` | [`crate::gateway`] |
//!
//! Anything else is `404 {"error":"not_found"}`. CORS wraps every response
//! from outside ([`handle`]), so a browser sees it on errors too.

use crate::config::Config;
use crate::http::{Req, Resp};
use crate::platform::Platform;
use serde_json::json;

/// The version `/health` reports. Keep in step with Cargo.toml.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// A route whose port from TypeScript has not landed yet. Never served in
/// production: `api.shamwari.ai` stays on the TypeScript Worker until every
/// route here passes parity.
pub fn not_ported(route: &str) -> Resp {
    Resp::json(&json!({ "error": "not_implemented", "route": route }), 501)
}

/// The routes without CORS (the TypeScript `worker.fetch`).
pub async fn route<P: Platform>(p: &P, cfg: &Config, req: Req) -> Resp {
    let path = req.path();
    let method = req.method.as_str();
    match (method, path.as_str()) {
        (_, "/health") => Resp::json(
            &json!({
                "ok": true,
                "service": "shamwari-gateway",
                "version": VERSION,
                "mind": cfg.mind_available(),
                "runtime": "rust",
            }),
            200,
        ),
        ("GET", "/v1/models") => crate::models::handle(p, cfg, &req).await,
        ("POST", "/v1/ground/context") => crate::ground::handle_context(p, cfg, &req).await,
        ("POST", "/v1/channels/sms") => crate::channels::handle_sms(p, cfg, &req).await,
        ("POST", "/v1/channels/ussd") => crate::channels::handle_ussd(p, cfg, &req).await,
        ("POST", "/v1/responses") => crate::responses::handle(p, cfg, &req).await,
        ("POST", "/v1/chat/completions") => crate::gateway::handle_chat(p, cfg, &req).await,
        _ => Resp::json(&json!({ "error": "not_found" }), 404),
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
