//! Port of `shamwari/gateway/src/models.ts`: the model ids a caller may name
//!
//! Not ported yet: `GET /v1/models` answers 501 until it is.

use crate::config::Config;
use crate::http::{Req, Resp};
use crate::platform::Platform;

pub async fn handle<P: Platform>(_p: &P, _cfg: &Config, _req: &Req) -> Resp {
    crate::router::not_ported("GET /v1/models")
}
