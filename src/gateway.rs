//! Port of `shamwari/gateway/src/index.ts and gateway.ts`: routed inference
//!
//! Not ported yet: `POST /v1/chat/completions` answers 501 until it is.

use crate::config::Config;
use crate::http::{Req, Resp};
use crate::platform::Platform;

pub async fn handle_chat<P: Platform>(_p: &P, _cfg: &Config, _req: &Req) -> Resp {
    crate::router::not_ported("POST /v1/chat/completions")
}
