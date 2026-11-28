//! Port of `shamwari/gateway/src/ground.ts`: grounding, retrieval only
//!
//! Not ported yet: `POST /v1/ground/context` answers 501 until it is.

use crate::config::Config;
use crate::http::{Req, Resp};
use crate::platform::Platform;

pub async fn handle_context<P: Platform>(_p: &P, _cfg: &Config, _req: &Req) -> Resp {
    crate::router::not_ported("POST /v1/ground/context")
}
