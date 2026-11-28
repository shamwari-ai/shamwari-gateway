//! Port of `shamwari/gateway/src/responses.ts`: Open Responses, a translator in front of chat completions
//!
//! Not ported yet: `POST /v1/responses` answers 501 until it is.

use crate::config::Config;
use crate::http::{Req, Resp};
use crate::platform::Platform;

pub async fn handle<P: Platform>(_p: &P, _cfg: &Config, _req: &Req) -> Resp {
    crate::router::not_ported("POST /v1/responses")
}
