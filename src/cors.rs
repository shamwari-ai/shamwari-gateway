//! Port of `shamwari/gateway/src/cors.ts`: which browser origins may call the API.
//!
//! Not ported yet: no preflight is answered and no CORS header is added, so a
//! browser cannot call this Worker. It serves no browser until the cutover.

use crate::config::Config;
use crate::http::{Req, Resp};

/// A CORS preflight answer, or `None` to carry on to the routes.
pub fn preflight(_cfg: &Config, _req: &Req) -> Option<Resp> {
    None
}

/// `res` with the CORS headers the request's origin is allowed.
pub fn with_cors(_cfg: &Config, _req: &Req, res: Resp) -> Resp {
    res
}
