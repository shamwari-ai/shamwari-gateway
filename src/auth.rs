//! Port of `shamwari/gateway/src/auth.ts`: who is calling. `authenticate`, the
//! signed-in person's own token (`person_session_token`), and the response for
//! a failed authentication (`auth_failure`, from `index.ts`).
//!
//! Two credential shapes, told apart by prefix and never tried against each
//! other: an API key (`sk_shamwari_…`), checked by the key service through the
//! `AUTH_CACHE` KV namespace ([`crate::keys`]), and a WorkOS access token,
//! verified offline ([`crate::workos`]).

use crate::config::Config;
use crate::http::{Req, Resp};
use crate::keys::{key_service, VerifyOutcome};
use crate::platform::Platform;
use crate::types::AuthContext;
use crate::workos::{looks_like_jwt, verify_workos_token, WorkosOutcome, WORKOS_KEY_TYPE};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

/// Every Shamwari API key starts with this. Anything else is never sent to the key service.
pub const KEY_PREFIX: &str = "sk_shamwari_";

#[derive(Debug, Clone, PartialEq)]
pub enum AuthResult {
    Ok(AuthContext),
    /// A verdict on the credential: 401.
    Invalid,
    /// No verdict could be reached (an outage): 503, never cached.
    Unavailable,
    /// A genuine sign-in that is not provisioned for Shamwari: 403 with this reason.
    Forbidden(String),
}

/// Lower-case hex SHA-256 of `raw`.
pub(crate) fn sha256_hex(raw: &str) -> String {
    Sha256::digest(raw.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// The bearer credential, as TypeScript reads it: `Authorization` starting with
/// `Bearer `, the rest trimmed; anything else is the empty string.
fn bearer(req: &Req) -> String {
    req.header("authorization")
        .and_then(|h| h.strip_prefix("Bearer "))
        .map(|t| js_trim(t).to_owned())
        .unwrap_or_default()
}

/// KV first, key service on miss. Only the hash is ever transmitted or cached,
/// so neither a KV leak nor a service log yields a usable key.
///
/// Short TTL is the revocation window: a revoked key stops working within
/// `AUTH_CACHE_TTL` seconds without any cache invalidation step.
///
/// Fails closed: `Unavailable` denies the request and is not cached, so a
/// recovered service is believed at once and an outage never becomes a
/// lasting verdict on a good key.
pub async fn authenticate<P: Platform>(p: &P, cfg: &Config, req: &Req) -> AuthResult {
    let raw = bearer(req);

    // A WorkOS access token is verified offline (workos.rs) and needs no cache: the
    // signature check is the whole cost, and a revoked session ends at the token's `exp`.
    if !raw.starts_with(KEY_PREFIX) {
        if !looks_like_jwt(&raw) {
            return AuthResult::Invalid;
        }
        return match verify_workos_token(p, cfg, &raw).await {
            WorkosOutcome::Ok(ctx) => AuthResult::Ok(ctx),
            WorkosOutcome::Forbidden(reason) => AuthResult::Forbidden(reason),
            WorkosOutcome::Invalid => AuthResult::Invalid,
            WorkosOutcome::Unavailable => AuthResult::Unavailable,
        };
    }

    let hash = sha256_hex(&raw);
    let cache_key = format!("auth:{hash}");
    // `Math.max(60, Number(env.AUTH_CACHE_TTL))`: NaN (unset or not a number)
    // stays NaN, KV refuses it, and the refused write is swallowed below.
    let ttl = cfg.auth_cache_ttl.as_deref().map_or(f64::NAN, js_number);
    let ttl = if ttl.is_nan() {
        f64::NAN
    } else {
        ttl.max(60.0)
    };

    // A KV read that fails, or a cached value that cannot be read, is not a
    // verdict on the key. TypeScript lets the throw reach the Worker's 500;
    // this signature has no error arm, so it denies as unavailable instead.
    let cached = match p.kv_get(&cache_key).await {
        Ok(v) => v,
        Err(_) => return AuthResult::Unavailable,
    };
    if let Some(text) = cached {
        let Ok(v) = serde_json::from_str::<Value>(&text) else {
            return AuthResult::Unavailable;
        };
        if v == json!("invalid") {
            return AuthResult::Invalid;
        }
        if js_truthy(&v) {
            return match serde_json::from_value::<AuthContext>(v) {
                Ok(ctx) => AuthResult::Ok(ctx),
                Err(_) => AuthResult::Unavailable,
            };
        }
    }

    // The prefix is exactly `sk_shamwari_` (12 characters), checked above.
    let outcome = key_service(cfg)
        .verify(p, cfg, &raw[..KEY_PREFIX.len()], &hash)
        .await;
    match outcome {
        VerifyOutcome::Ok(ctx) => {
            // Best-effort: a KV hiccup must not deny a verified key.
            if let Ok(v) = serde_json::to_string(&ctx) {
                remember(p, &cache_key, &v, ttl).await;
            }
            AuthResult::Ok(ctx)
        }
        VerifyOutcome::Invalid => {
            // Cache the negative so key-guessing cannot hammer the service.
            remember(p, &cache_key, "\"invalid\"", ttl).await;
            AuthResult::Invalid
        }
        VerifyOutcome::Unavailable => AuthResult::Unavailable,
    }
}

async fn remember<P: Platform>(p: &P, key: &str, value: &str, ttl: f64) {
    if !ttl.is_finite() {
        return; // KV refuses a non-finite expirationTtl; TypeScript swallows that refusal
    }
    let _ = p.kv_put(key, value, Some(ttl as u64)).await;
}

/// The signed-in person's own WorkOS access token, already verified by
/// [`authenticate`], for this request only. It is passed to the Nyuchi API so
/// the API acts with that person's own base access. `None` for an API key: no
/// person is behind it. Never cached, logged, stored or sent anywhere but the
/// Nyuchi API.
pub fn person_session_token(req: &Req, auth: &AuthContext) -> Option<String> {
    if auth.key_type != WORKOS_KEY_TYPE {
        return None;
    }
    Some(bearer(req)).filter(|t| !t.is_empty())
}

/// The response for a failed authentication, the same on every authenticated
/// endpoint (`authFailure` in `index.ts`). `AuthResult::Ok` is not a failure;
/// it is answered as an invalid credential so a misuse fails closed.
pub fn auth_failure(r: &AuthResult) -> Resp {
    match r {
        // A 503, not a 401: the caller's key may be fine, and a client that
        // drops a good key on a 401 would turn our outage into theirs.
        AuthResult::Unavailable => {
            let mut res = Resp::json(&json!({ "error": "auth_unavailable" }), 503);
            res.set_header("retry-after", "5");
            res
        }
        // A real sign-in that is not provisioned for Shamwari. Not a 401: the
        // client would send the person back to sign in and hand them a new
        // token with the same gap.
        AuthResult::Forbidden(detail) => Resp::json(
            &json!({ "error": "not_provisioned", "reason": detail }),
            403,
        ),
        AuthResult::Invalid | AuthResult::Ok(_) => {
            Resp::json(&json!({ "error": "invalid_api_key" }), 401)
        }
    }
}

// ---------------------------------------------------------------------------
// JavaScript semantics the ports need to match exactly.

/// `String.prototype.trim`: JavaScript's own whitespace set ([`crate::lang::js`]).
pub(crate) fn js_trim(s: &str) -> &str {
    crate::lang::js::trim(s)
}

/// JavaScript truthiness of a JSON value.
pub(crate) fn js_truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(_) | Value::Object(_) => true,
    }
}

/// `Number(s)` for a string: whitespace trimmed, empty is 0, decimal, `0x`,
/// `0o`, `0b` and `Infinity` literals, anything else NaN.
pub(crate) fn js_number(s: &str) -> f64 {
    let t = js_trim(s);
    if t.is_empty() {
        return 0.0;
    }
    match t {
        "Infinity" | "+Infinity" => return f64::INFINITY,
        "-Infinity" => return f64::NEG_INFINITY,
        _ => {}
    }
    for (prefix, radix) in [
        ("0x", 16),
        ("0X", 16),
        ("0o", 8),
        ("0O", 8),
        ("0b", 2),
        ("0B", 2),
    ] {
        if let Some(digits) = t.strip_prefix(prefix) {
            if digits.is_empty() || !digits.chars().all(|c| c.is_digit(radix)) {
                return f64::NAN;
            }
            return digits.chars().fold(0.0, |acc, c| {
                acc * f64::from(radix) + f64::from(c.to_digit(radix).unwrap_or(0))
            });
        }
    }
    let body = t.strip_prefix(['+', '-']).unwrap_or(t);
    let (mantissa, exponent) = match body.find(['e', 'E']) {
        Some(i) => (&body[..i], Some(&body[i + 1..])),
        None => (body, None),
    };
    let (int, frac) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let digits = |d: &str| d.chars().all(|c| c.is_ascii_digit());
    let mantissa_ok = digits(int)
        && digits(frac)
        && !(int.is_empty() && frac.is_empty())
        && (mantissa.contains('.') || !int.is_empty());
    let exponent_ok = exponent.is_none_or(|e| {
        let e = e.strip_prefix(['+', '-']).unwrap_or(e);
        !e.is_empty() && digits(e)
    });
    if !mantissa_ok || !exponent_ok {
        return f64::NAN;
    }
    t.parse::<f64>().unwrap_or(f64::NAN)
}

/// Where the gateway's log lines go: Workers Logs in production, stderr on
/// the host. Tests read them back with [`take_logs`].
#[derive(Debug, Clone, Copy)]
pub(crate) enum Level {
    Warn,
    Error,
}

#[cfg(test)]
thread_local! {
    static LOGS: std::cell::RefCell<Vec<String>> = const { std::cell::RefCell::new(Vec::new()) };
}

pub(crate) fn console(level: Level, line: &str) {
    #[cfg(test)]
    LOGS.with(|l| l.borrow_mut().push(line.to_owned()));
    #[cfg(target_arch = "wasm32")]
    match level {
        Level::Warn => worker::console_warn!("{}", line),
        Level::Error => worker::console_error!("{}", line),
    }
    #[cfg(not(any(test, target_arch = "wasm32")))]
    eprintln!("{level:?}: {line}");
    #[cfg(test)]
    let _ = level;
}

/// Every line logged on this thread since the last call.
#[cfg(test)]
pub(crate) fn take_logs() -> Vec<String> {
    LOGS.with(|l| std::mem::take(&mut *l.borrow_mut()))
}

/// Test doubles shared by the auth-area tests: a [`Platform`] that records
/// KV writes with their TTL and can be told to fail its KV or queue.
#[cfg(test)]
pub(crate) mod fakes {
    use crate::platform::{HttpRequest, HttpResponse, Platform};
    use crate::testing::FakePlatform;
    use serde_json::Value;
    use std::cell::{Cell, RefCell};

    #[derive(Default)]
    pub struct Faults {
        pub kv_get: Cell<bool>,
        pub kv_put: Cell<bool>,
        pub kv_delete: Cell<bool>,
        pub queue: Cell<bool>,
        /// Only keys starting with this fail (empty: every key).
        pub kv_prefix: RefCell<String>,
    }

    pub struct Rec {
        pub inner: FakePlatform,
        /// Every KV write: (key, value, ttl).
        pub puts: RefCell<Vec<(String, String, Option<u64>)>>,
        pub faults: Faults,
    }

    impl Rec {
        pub fn new() -> Self {
            Rec {
                inner: FakePlatform::new(),
                puts: RefCell::new(vec![]),
                faults: Faults::default(),
            }
        }

        fn kv_fails(&self, flag: &Cell<bool>, key: &str) -> bool {
            flag.get() && key.starts_with(self.faults.kv_prefix.borrow().as_str())
        }
    }

    impl std::ops::Deref for Rec {
        type Target = FakePlatform;
        fn deref(&self) -> &FakePlatform {
            &self.inner
        }
    }

    impl Platform for Rec {
        async fn fetch(&self, req: HttpRequest) -> Result<HttpResponse, String> {
            self.inner.fetch(req).await
        }
        async fn ai_run(&self, m: &str, i: &Value, o: Option<&Value>) -> Result<Value, String> {
            self.inner.ai_run(m, i, o).await
        }
        async fn kv_get(&self, key: &str) -> Result<Option<String>, String> {
            if self.kv_fails(&self.faults.kv_get, key) {
                return Err("KV down".into());
            }
            self.inner.kv_get(key).await
        }
        async fn kv_put(&self, key: &str, value: &str, ttl: Option<u64>) -> Result<(), String> {
            if self.kv_fails(&self.faults.kv_put, key) {
                return Err("kv 429".into());
            }
            self.puts
                .borrow_mut()
                .push((key.to_owned(), value.to_owned(), ttl));
            self.inner.kv_put(key, value, ttl).await
        }
        async fn kv_delete(&self, key: &str) -> Result<(), String> {
            if self.kv_fails(&self.faults.kv_delete, key) {
                return Err("KV down".into());
            }
            self.inner.kv_delete(key).await
        }
        async fn queue_send(&self, message: &Value) -> Result<(), String> {
            if self.faults.queue.get() {
                return Err("queue down".into());
            }
            self.inner.queue_send(message).await
        }
        async fn secret_store_get(&self, binding: &str) -> Result<Option<String>, String> {
            self.inner.secret_store_get(binding).await
        }
        fn now_ms(&self) -> u64 {
            self.inner.now_ms()
        }
        fn random_bytes(&self, n: usize) -> Vec<u8> {
            self.inner.random_bytes(n)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{block_on, FakePlatform};
    use crate::types::Scope;

    const RAW: &str = "sk_shamwari_abcdefghijklmnop";

    fn req_with(auth: Option<&str>) -> Req {
        Req {
            method: "POST".into(),
            url: "https://x/v1/chat/completions".into(),
            headers: auth
                .map(|a| vec![("authorization".to_owned(), a.to_owned())])
                .unwrap_or_default(),
            body: String::new(),
        }
    }

    fn core_cfg(ttl: &str) -> Config {
        Config {
            core_url: Some("https://core.example".into()),
            shamwari_core_token: Some("core-token".into()),
            auth_cache_ttl: Some(ttl.into()),
            ..Default::default()
        }
    }

    const GOOD: &str = r#"{"api_key_id":"k1","owner_entity_id":"e1","owner_person_id":null,"key_type":"shamwari","tier":"community","monthly_token_cap":100000}"#;

    fn ctx(key_type: &str) -> AuthContext {
        AuthContext {
            api_key_id: "k1".into(),
            owner_entity_id: "e1".into(),
            owner_person_id: None,
            key_type: key_type.into(),
            tier: "community".into(),
            monthly_token_cap: 100_000,
            scopes: None,
            training_consent: None,
        }
    }

    #[test]
    fn a_bearer_that_is_neither_shape_is_invalid_with_no_outbound_call() {
        let p = FakePlatform::new();
        for h in [
            None,
            Some("Bearer not-a-credential"),
            Some("Basic sk_shamwari_x"),
            Some("Bearer "),
        ] {
            assert_eq!(
                block_on(authenticate(&p, &core_cfg("60"), &req_with(h))),
                AuthResult::Invalid,
                "{h:?}"
            );
        }
        assert!(p.requests.borrow().is_empty());
    }

    #[test]
    fn only_the_hash_is_cached_under_auth_sha256_with_the_ttl_floor_of_60() {
        let p = fakes::Rec::new();
        p.on("https://core.example/auth/verify", 200, GOOD);
        let r = block_on(authenticate(
            &p,
            &core_cfg("5"),
            &req_with(Some(&format!("Bearer {RAW}"))),
        ));
        assert!(matches!(r, AuthResult::Ok(_)));
        let puts = p.puts.borrow();
        assert_eq!(puts.len(), 1);
        assert_eq!(puts[0].0, format!("auth:{}", sha256_hex(RAW)));
        assert_eq!(puts[0].2, Some(60));
        assert!(!puts[0].0.contains(RAW) && !puts[0].1.contains(RAW));
        // The key service saw the prefix and the hash, never the key.
        let sent = p.requests.borrow()[0].body.clone().unwrap();
        assert!(!sent.contains(RAW));
        assert!(sent.contains(&sha256_hex(RAW)));
    }

    #[test]
    fn a_longer_ttl_is_kept() {
        let p = fakes::Rec::new();
        p.on("https://core.example/auth/verify", 200, GOOD);
        block_on(authenticate(
            &p,
            &core_cfg(" 300 "),
            &req_with(Some(&format!("Bearer {RAW}"))),
        ));
        assert_eq!(p.puts.borrow()[0].2, Some(300));
    }

    #[test]
    fn an_unset_ttl_caches_nothing_and_still_answers() {
        // Math.max(60, NaN) is NaN, and KV refuses it: the verified key still works.
        let p = fakes::Rec::new();
        p.on("https://core.example/auth/verify", 200, GOOD);
        let cfg = Config {
            auth_cache_ttl: None,
            ..core_cfg("60")
        };
        let r = block_on(authenticate(
            &p,
            &cfg,
            &req_with(Some(&format!("Bearer {RAW}"))),
        ));
        assert!(matches!(r, AuthResult::Ok(_)));
        assert!(p.puts.borrow().is_empty());
    }

    #[test]
    fn a_cached_context_answers_without_the_service_and_a_cached_negative_is_invalid() {
        let p = FakePlatform::new();
        let key = format!("auth:{}", sha256_hex(RAW));
        p.kv.borrow_mut().insert(
            key.clone(),
            serde_json::to_string(&ctx("shamwari")).unwrap(),
        );
        let r = block_on(authenticate(
            &p,
            &core_cfg("60"),
            &req_with(Some(&format!("Bearer {RAW}"))),
        ));
        assert_eq!(r, AuthResult::Ok(ctx("shamwari")));
        p.kv.borrow_mut().insert(key, "\"invalid\"".into());
        let r = block_on(authenticate(
            &p,
            &core_cfg("60"),
            &req_with(Some(&format!("Bearer {RAW}"))),
        ));
        assert_eq!(r, AuthResult::Invalid);
        assert!(p.requests.borrow().is_empty());
    }

    #[test]
    fn a_failing_kv_read_denies_as_unavailable_and_calls_nothing() {
        let p = fakes::Rec::new();
        p.faults.kv_get.set(true);
        let r = block_on(authenticate(
            &p,
            &core_cfg("60"),
            &req_with(Some(&format!("Bearer {RAW}"))),
        ));
        assert_eq!(r, AuthResult::Unavailable);
        assert!(p.requests.borrow().is_empty());
    }

    #[test]
    fn the_person_session_token_is_only_for_a_workos_caller() {
        let req = req_with(Some("Bearer  eyJ.a.b  "));
        assert_eq!(
            person_session_token(&req, &ctx(WORKOS_KEY_TYPE)),
            Some("eyJ.a.b".into())
        );
        assert_eq!(person_session_token(&req, &ctx("shamwari")), None);
        assert_eq!(
            person_session_token(&req_with(Some("Bearer   ")), &ctx(WORKOS_KEY_TYPE)),
            None
        );
        assert_eq!(
            person_session_token(&req_with(None), &ctx(WORKOS_KEY_TYPE)),
            None
        );
    }

    #[test]
    fn auth_failure_answers_as_typescript_does() {
        let r = auth_failure(&AuthResult::Unavailable);
        assert_eq!(r.status, 503);
        assert_eq!(r.body_json(), json!({ "error": "auth_unavailable" }));
        assert_eq!(r.header("Retry-After"), Some("5"));

        let r = auth_failure(&AuthResult::Forbidden("no_person_id".into()));
        assert_eq!(r.status, 403);
        assert_eq!(
            r.body_json(),
            json!({ "error": "not_provisioned", "reason": "no_person_id" })
        );
        assert_eq!(r.header("Retry-After"), None);

        let r = auth_failure(&AuthResult::Invalid);
        assert_eq!(r.status, 401);
        assert_eq!(r.body_json(), json!({ "error": "invalid_api_key" }));
        assert_eq!(r.header("content-type"), Some("application/json"));
    }

    #[test]
    fn a_cached_context_without_scopes_gets_no_personal_scope() {
        let p = FakePlatform::new();
        let key = format!("auth:{}", sha256_hex(RAW));
        p.kv.borrow_mut().insert(
            key,
            r#"{"apiKeyId":"k","ownerEntityId":"e","ownerPersonId":null,"keyType":"shamwari","tier":"community","monthlyTokenCap":1}"#.into(),
        );
        let AuthResult::Ok(c) = block_on(authenticate(
            &p,
            &core_cfg("60"),
            &req_with(Some(&format!("Bearer {RAW}"))),
        )) else {
            panic!("expected ok");
        };
        assert!(!crate::keys::permitted_scopes(&c).contains(&Scope::Personal));
    }

    #[test]
    fn js_number_reads_as_javascript_does() {
        for (s, want) in [
            ("60", 60.0),
            (" 120 ", 120.0),
            ("", 0.0),
            ("  ", 0.0),
            ("1e3", 1000.0),
            ("0x10", 16.0),
            ("0b11", 3.0),
            ("1.", 1.0),
            (".5", 0.5),
            ("-7", -7.0),
            ("+7", 7.0),
            ("Infinity", f64::INFINITY),
        ] {
            assert_eq!(js_number(s), want, "{s:?}");
        }
        for s in [
            "lots", "1_000", "inf", "NaN", "0x", "1e", ".", "1 2", "0x1g", "infinity",
        ] {
            assert!(js_number(s).is_nan(), "{s:?}");
        }
    }
}
