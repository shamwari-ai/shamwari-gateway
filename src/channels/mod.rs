//! Port of `shamwari/gateway/src/channels/`: SMS and USSD.
//!
//! Not ported yet: both routes answer 501 until they are.

use crate::config::Config;
use crate::http::{Req, Resp};
use crate::platform::Platform;

pub mod sms;
pub mod ussd;

pub async fn handle_sms<P: Platform>(_p: &P, _cfg: &Config, _req: &Req) -> Resp {
    crate::router::not_ported("POST /v1/channels/sms")
}

pub async fn handle_ussd<P: Platform>(_p: &P, _cfg: &Config, _req: &Req) -> Resp {
    crate::router::not_ported("POST /v1/channels/ussd")
}
