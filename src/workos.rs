//! Port of `shamwari/gateway/src/workos.ts`: WorkOS AuthKit access tokens for
//! people, verified offline against the JWKS and turned into an
//! [`AuthContext`]. `docs/chat-signin.md` in `shamwari-ai/shamwari` has the
//! design.
//!
//! The properties that matter, each pinned by the tests below:
//!
//! - The algorithm is pinned to RS256 and the key comes from the JWKS, never
//!   from the token. `alg: none` and an HS256 token signed with the public key
//!   as a shared secret are both refused.
//! - `iss` and `client_id` must match configuration, `exp`, `iat` and `sub`
//!   must be present. `aud` is checked, and required, whenever
//!   `WORKOS_AUDIENCE` is set; when it is not set the check is off by owner
//!   decision and `client_id` is the isolation.
//! - Every valid token gets one fixed grant, decided here in code: community
//!   and platform, the free tier. No claim, permission or role is read to
//!   decide it, and `personal` is not a scope this file can grant (rule 1).
//! - Fails closed. A verdict on the token (`Invalid`) and a failure to reach a
//!   verdict (`Unavailable`) are different outcomes, so a JWKS outage is a 503
//!   and not a sign-out.
//!
//! TypeScript verifies with `jose` (`jwtVerify` over `createRemoteJWKSet`).
//! This file does the same checks, in the same order, with the same JWKS
//! cache, cooldown and timeout, so a token gets the same verdict from both
//! Workers. What `jose` counts as its own failure (the JWKS cannot be reached
//! or read, a key cannot be imported or is under 2048 bits) is `Unavailable`;
//! everything it raises about the token itself is `Invalid`.

use crate::auth::{console, js_number, js_trim, Level};
use crate::config::Config;
use crate::platform::{HttpRequest, Platform};
use crate::types::{AuthContext, Scope};
use base64::engine::{DecodePaddingMode, GeneralPurpose, GeneralPurposeConfig};
use base64::Engine;
use rsa::{BoxedUint, PublicKey};
use serde_json::{Map, Value};
use sha2::Sha256;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;

/// What a signed-in person may ask for, whoever they are. Not read from the
/// token. A paid tier later comes from the subscription / ledger entitlement,
/// not from a permission.
pub const FIXED_GRANT: [Scope; 2] = [Scope::Community, Scope::Platform];

pub const WORKOS_KEY_TYPE: &str = "workos";

/// The JWT-template claim carrying the platform's one canonical person id
/// (WorkOS `externalId`). Usage and quota key on this, never on WorkOS's own
/// `sub`, so a person is the same person on every Nyuchi surface.
pub const PERSON_ID_CLAIM: &str = "person_id";

/// The pattern [`is_person_id`] checks. The id is interpolated into a Core URL
/// on the default quota path, so it must be inert there, and Core validates
/// owner ids against exactly this pattern (core/main.py `_ID`). The session id
/// gets the same treatment because it is embedded in the api key id.
pub const PERSON_ID_PATTERN: &str = "^[A-Za-z0-9_.:-]{1,64}$";

/// `PERSON_ID_PATTERN.test(s)`.
pub fn is_person_id(s: &str) -> bool {
    (1..=64).contains(&s.len())
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"_.:-".contains(&b))
}

#[derive(Debug, Clone, PartialEq)]
pub enum WorkosOutcome {
    Ok(AuthContext),
    Invalid,
    /// A genuine token for an account with no usable canonical person id. That
    /// is provisioning, not a bad credential: a 403 that says so, not a 401
    /// that sends the person round the sign-in loop. The reason is `no_person_id`.
    Forbidden(String),
    Unavailable,
}

fn is_b64url_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_' || c == '-'
}

/// Three dot-separated base64url segments: shaped like a JWT, not yet believed.
pub fn looks_like_jwt(raw: &str) -> bool {
    let parts: Vec<&str> = raw.split('.').collect();
    parts.len() == 3
        && !parts[0].is_empty()
        && !parts[1].is_empty()
        && parts.iter().all(|p| p.chars().all(is_b64url_char))
}

/// WorkOS sign-in is off unless a client id is configured.
pub fn workos_enabled(cfg: &Config) -> bool {
    cfg.get(&cfg.workos_client_id).is_some()
}

struct WorkosConfig {
    /// Every client id whose tokens are accepted (one or more AuthKit applications).
    client_ids: Vec<String>,
    /// Each configured `iss`, and the same with a trailing slash added.
    issuers: Vec<String>,
    /// `None`: the `aud` claim is not checked.
    audience: Option<String>,
    jwks_url: String,
    monthly_token_cap: u64,
    max_token_age_seconds: f64,
}

fn js_integer(f: f64) -> bool {
    f.is_finite() && f.fract() == 0.0
}

/// `encodeURIComponent`.
fn encode_uri_component(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || b"-_.!~*'()".contains(&b) {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// `None` when enabled but incomplete: that is `Unavailable`, never "accept anyway".
fn config(cfg: &Config) -> Option<WorkosConfig> {
    // One or more client ids, comma separated. Blank entries are dropped; a list
    // with nothing left is a configuration error, never "accept any".
    let client_ids: Vec<String> = cfg
        .workos_client_id
        .as_deref()
        .unwrap_or_default()
        .split(',')
        .map(js_trim)
        .filter(|c| !c.is_empty())
        .map(str::to_owned)
        .collect();
    let client_id = client_ids.first()?.clone();
    let issuer = cfg.get(&cfg.workos_issuer)?;
    // Optional by owner decision, but when it IS configured it is required, and a
    // variable set to nothing is a mistake to refuse, not a request to stop checking.
    if cfg
        .workos_audience
        .as_deref()
        .is_some_and(|a| js_trim(a).is_empty())
    {
        return None;
    }
    let cap = cfg
        .workos_monthly_token_cap
        .as_deref()
        .map_or(f64::NAN, js_number);
    if !js_integer(cap) || cap <= 0.0 {
        return None;
    }
    // A ceiling on how long any token may have been valid, whatever `exp` it carries.
    let max_age = cfg
        .workos_max_token_age_seconds
        .as_deref()
        .map_or(3600.0, js_number);
    if !js_integer(max_age) || max_age <= 0.0 {
        return None;
    }
    let jwks = match cfg.get(&cfg.workos_jwks_url) {
        Some(u) => u.to_owned(),
        None => format!(
            "https://api.workos.com/sso/jwks/{}",
            encode_uri_component(&client_id)
        ),
    };
    let jwks_url = url::Url::parse(&jwks).ok()?;
    if jwks_url.scheme() != "https" {
        return None;
    }
    // WorkOS's own documentation shows the issuer both with and without a
    // trailing slash, so either form must work. Only that one character is
    // tolerated. WORKOS_ISSUER may name more than one issuer, comma separated.
    let mut issuers = vec![];
    for one in issuer.split(',') {
        let bare = js_trim(one).trim_end_matches('/');
        if bare.is_empty() {
            return None;
        }
        issuers.push(bare.to_owned());
        issuers.push(format!("{bare}/"));
    }
    Some(WorkosConfig {
        client_ids,
        issuers,
        audience: cfg.workos_audience.clone(),
        jwks_url: jwks_url.to_string(),
        monthly_token_cap: cap as u64,
        max_token_age_seconds: max_age,
    })
}

// One JWKS set per URL per isolate, as `jose`'s remote set keeps it: the keys
// are cached, re-fetched on an unknown `kid`, but not more than once per
// cooldown however many forged kids arrive, so a flood of bad tokens cannot
// become a flood at WorkOS.
const JWKS_CACHE_MAX_AGE_MS: u64 = 10 * 60 * 1000;
const JWKS_COOLDOWN_MS: u64 = 30 * 1000;
const JWKS_TIMEOUT_MS: u32 = 1500;

#[derive(Default)]
struct JwksSet {
    keys: Option<Vec<Map<String, Value>>>,
    fetched_at: Option<u64>,
}

thread_local! {
    static JWKS_SETS: RefCell<HashMap<String, JwksSet>> = RefCell::new(HashMap::new());
    /// (window start, lines logged in it).
    static LOG_WINDOW: Cell<(u64, u32)> = const { Cell::new((0, 0)) };
}

/// Forget cached keys and restart the log cap (tests).
pub fn reset_jwks_cache() {
    JWKS_SETS.with(|s| s.borrow_mut().clear());
    LOG_WINDOW.with(|w| w.set((0, 0)));
}

/// Why `jose` would have refused: a verdict on the token, with its error code
/// and the failing claim, or a failure that is ours.
#[derive(Debug, Clone, Copy, PartialEq)]
enum JoseErr {
    Verdict {
        code: &'static str,
        claim: Option<&'static str>,
    },
    Ours,
}

const ERR_JWS_INVALID: &str = "ERR_JWS_INVALID";
const ERR_JWT_INVALID: &str = "ERR_JWT_INVALID";
const ERR_JOSE_NOT_SUPPORTED: &str = "ERR_JOSE_NOT_SUPPORTED";
const ERR_JOSE_ALG_NOT_ALLOWED: &str = "ERR_JOSE_ALG_NOT_ALLOWED";
const ERR_JWKS_NO_MATCHING_KEY: &str = "ERR_JWKS_NO_MATCHING_KEY";
const ERR_JWKS_MULTIPLE_MATCHING_KEYS: &str = "ERR_JWKS_MULTIPLE_MATCHING_KEYS";
const ERR_JWS_SIGNATURE_VERIFICATION_FAILED: &str = "ERR_JWS_SIGNATURE_VERIFICATION_FAILED";
const ERR_JWT_CLAIM_VALIDATION_FAILED: &str = "ERR_JWT_CLAIM_VALIDATION_FAILED";
const ERR_JWT_EXPIRED: &str = "ERR_JWT_EXPIRED";

const fn verdict(code: &'static str) -> JoseErr {
    JoseErr::Verdict { code, claim: None }
}

const fn claim_failed(code: &'static str, claim: &'static str) -> JoseErr {
    JoseErr::Verdict {
        code,
        claim: Some(claim),
    }
}

/// base64url, padded or not, trailing bits tolerated (as `jose` decodes it).
const B64URL: GeneralPurpose = GeneralPurpose::new(
    &base64::alphabet::URL_SAFE,
    GeneralPurposeConfig::new()
        .with_decode_padding_mode(DecodePaddingMode::Indifferent)
        .with_decode_allow_trailing_bits(true),
);

fn b64url_decode(s: &str) -> Option<Vec<u8>> {
    B64URL.decode(s).ok()
}

/// base64url, then strict UTF-8, then JSON, then a plain object.
fn json_object(segment: &str) -> Option<Map<String, Value>> {
    let bytes = b64url_decode(segment)?;
    let text = String::from_utf8(bytes).ok()?;
    match serde_json::from_str(&text).ok()? {
        Value::Object(m) => Some(m),
        _ => None,
    }
}

fn is_fresh(at: Option<u64>, duration: u64, now: u64) -> bool {
    at.is_some_and(|t| now < t.saturating_add(duration))
}

/// Fetch the key set and make it the current one. Any failure is ours.
async fn reload<P: Platform>(p: &P, url: &str) -> Result<(), JoseErr> {
    let res = p
        .fetch(HttpRequest {
            method: "GET".into(),
            url: url.to_owned(),
            headers: vec![(
                "accept".into(),
                "application/json, application/jwk-set+json".into(),
            )],
            body: None,
            timeout_ms: Some(JWKS_TIMEOUT_MS),
        })
        .await
        .map_err(|_| JoseErr::Ours)?;
    if res.status != 200 {
        return Err(JoseErr::Ours);
    }
    let v: Value = serde_json::from_str(&res.body).map_err(|_| JoseErr::Ours)?;
    let keys = v
        .get("keys")
        .filter(|_| v.is_object())
        .and_then(Value::as_array)
        .and_then(|ks| {
            ks.iter()
                .map(|k| k.as_object().cloned())
                .collect::<Option<Vec<_>>>()
        })
        .ok_or(JoseErr::Ours)?;
    let now = p.now_ms();
    JWKS_SETS.with(|s| {
        s.borrow_mut().insert(
            url.to_owned(),
            JwksSet {
                keys: Some(keys),
                fetched_at: Some(now),
            },
        )
    });
    Ok(())
}

fn current(url: &str) -> (Option<Vec<Map<String, Value>>>, Option<u64>) {
    JWKS_SETS.with(|s| {
        s.borrow()
            .get(url)
            .map(|e| (e.keys.clone(), e.fetched_at))
            .unwrap_or_default()
    })
}

/// `isUsableJWK` for RS256.
fn usable(jwk: &Map<String, Value>, kid: Option<&Value>) -> bool {
    let ext_ok = jwk.get("ext").is_none_or(Value::is_boolean);
    let ops_ok = jwk.get("key_ops").is_none_or(|ops| {
        ops.as_array().is_some_and(|ops| {
            let names: Option<Vec<&str>> = ops.iter().map(Value::as_str).collect();
            names.is_some_and(|n| {
                n.iter().enumerate().all(|(i, o)| !n[..i].contains(o)) && n.contains(&"verify")
            })
        })
    });
    let kty_ok = jwk.get("kty").and_then(Value::as_str) == Some("RSA");
    let kid_ok = match kid {
        None => true,
        Some(Value::String(k)) => jwk.get("kid").and_then(Value::as_str) == Some(k),
        Some(_) => false,
    };
    let alg_ok = jwk.get("alg").is_none_or(|a| a.as_str() == Some("RS256"));
    let use_ok = jwk.get("use").is_none_or(|u| u.as_str() == Some("sig"));
    ext_ok && ops_ok && kty_ok && kid_ok && alg_ok && use_ok
}

fn biguint(v: Option<&Value>) -> Option<BoxedUint> {
    let mut bytes = b64url_decode(v?.as_str()?)?;
    while bytes.first() == Some(&0) {
        bytes.remove(0);
    }
    let bits = u32::try_from(bytes.len().max(1) * 8).ok()?;
    BoxedUint::from_be_slice(&bytes, bits).ok()
}

/// Import a JWK as an RSA public key. A failure is ours (WebCrypto's import
/// failing, or a private key in the set). The 2048-bit floor is checked later,
/// where `jose` checks it.
fn import(jwk: &Map<String, Value>) -> Result<PublicKey, JoseErr> {
    let private = ["d", "priv"]
        .iter()
        .any(|k| jwk.get(*k).is_some_and(crate::auth::js_truthy));
    if private {
        return Err(JoseErr::Ours);
    }
    let (Some(n), Some(e)) = (biguint(jwk.get("n")), biguint(jwk.get("e"))) else {
        return Err(JoseErr::Ours);
    };
    PublicKey::new_with_size_limits(n, e, 1..=16384).map_err(|_| JoseErr::Ours)
}

/// The one key in the set this token names, as `createLocalJWKSet` picks it.
fn select(keys: &[Map<String, Value>], header: &Map<String, Value>) -> Result<PublicKey, JoseErr> {
    let kid = header.get("kid");
    let candidates: Vec<&Map<String, Value>> = keys.iter().filter(|k| usable(k, kid)).collect();
    match candidates.as_slice() {
        [] => Err(verdict(ERR_JWKS_NO_MATCHING_KEY)),
        [one] => import(one),
        _ => Err(verdict(ERR_JWKS_MULTIPLE_MATCHING_KEYS)),
    }
}

/// `createRemoteJWKSet`'s key lookup: reload when there is no set or it is
/// older than the cache age; on an unknown key, reload once if the cooldown
/// has passed.
async fn jwks_key<P: Platform>(
    p: &P,
    url: &str,
    header: &Map<String, Value>,
) -> Result<PublicKey, JoseErr> {
    let (keys, at) = current(url);
    if keys.is_none() || !is_fresh(at, JWKS_CACHE_MAX_AGE_MS, p.now_ms()) {
        reload(p, url).await?;
    }
    let (keys, at) = current(url);
    match select(keys.as_deref().unwrap_or_default(), header) {
        Err(e)
            if e == verdict(ERR_JWKS_NO_MATCHING_KEY)
                && !is_fresh(at, JWKS_COOLDOWN_MS, p.now_ms()) =>
        {
            reload(p, url).await?;
            let (keys, _) = current(url);
            select(keys.as_deref().unwrap_or_default(), header)
        }
        r => r,
    }
}

/// `jwtVerify(token, jwks, { algorithms: ['RS256'], issuer, audience?,
/// requiredClaims: ['exp', 'iat', 'sub'], maxTokenAge })`, step for step.
async fn jwt_verify<P: Platform>(
    p: &P,
    wc: &WorkosConfig,
    token: &str,
) -> Result<Map<String, Value>, JoseErr> {
    let parts: Vec<&str> = token.split('.').collect();
    let [h, payload, signature] = parts.as_slice() else {
        return Err(verdict(ERR_JWS_INVALID));
    };
    let header = json_object(h).ok_or(verdict(ERR_JWS_INVALID))?;

    let mut b64 = true;
    if let Some(crit) = header.get("crit") {
        let names: Vec<&str> = crit
            .as_array()
            .filter(|l| !l.is_empty())
            .and_then(|l| l.iter().map(Value::as_str).collect::<Option<Vec<_>>>())
            .filter(|l| l.iter().all(|s| !s.is_empty()))
            .ok_or(verdict(ERR_JWS_INVALID))?;
        for name in &names {
            if *name != "b64" {
                return Err(verdict(ERR_JOSE_NOT_SUPPORTED));
            }
            if !header.contains_key(*name) {
                return Err(verdict(ERR_JWS_INVALID));
            }
        }
        if names.contains(&"b64") {
            b64 = header
                .get("b64")
                .and_then(Value::as_bool)
                .ok_or(verdict(ERR_JWS_INVALID))?;
        }
    }
    let alg = header
        .get("alg")
        .and_then(Value::as_str)
        .filter(|a| !a.is_empty())
        .ok_or(verdict(ERR_JWS_INVALID))?;
    if alg != "RS256" {
        return Err(verdict(ERR_JOSE_ALG_NOT_ALLOWED));
    }

    let key = jwks_key(p, &wc.jwks_url, &header).await?;
    let sig = b64url_decode(signature).ok_or(verdict(ERR_JWS_INVALID))?;
    {
        use rsa::traits::PublicKeyParts;
        if key.n().bit_width() < 2048 {
            return Err(JoseErr::Ours); // a TypeError in jose: the key set's fault, not the token's
        }
    }
    let signing_input = format!("{h}.{payload}");
    key.pkcs1v15_verify::<Sha256>(signing_input.as_bytes(), &sig)
        .map_err(|_| verdict(ERR_JWS_SIGNATURE_VERIFICATION_FAILED))?;
    if !b64 {
        return Err(verdict(ERR_JWT_INVALID)); // JWTs MUST NOT use unencoded payload
    }
    let claims_bytes = b64url_decode(payload).ok_or(verdict(ERR_JWS_INVALID))?;
    let claims = String::from_utf8(claims_bytes)
        .ok()
        .and_then(|t| serde_json::from_str::<Value>(&t).ok())
        .and_then(|v| match v {
            Value::Object(m) => Some(m),
            _ => None,
        })
        .ok_or(verdict(ERR_JWT_INVALID))?;

    // Presence, in jose's order: issuer, audience, maxTokenAge, then the required list reversed.
    let mut presence = vec!["iss"];
    if wc.audience.is_some() {
        presence.push("aud");
    }
    presence.extend(["iat", "sub", "exp"]);
    for c in presence {
        if !claims.contains_key(c) {
            return Err(claim_failed(ERR_JWT_CLAIM_VALIDATION_FAILED, c));
        }
    }
    let iss = claims.get("iss").and_then(Value::as_str);
    if !iss.is_some_and(|i| wc.issuers.iter().any(|x| x == i)) {
        return Err(claim_failed(ERR_JWT_CLAIM_VALIDATION_FAILED, "iss"));
    }
    if let Some(want) = &wc.audience {
        let present = match claims.get("aud") {
            Some(Value::String(a)) => a == want,
            Some(Value::Array(a)) => a.iter().any(|x| x.as_str() == Some(want)),
            _ => false,
        };
        if !present {
            return Err(claim_failed(ERR_JWT_CLAIM_VALIDATION_FAILED, "aud"));
        }
    }
    let now = (p.now_ms() / 1000) as f64;
    let number = |c: &'static str| -> Result<Option<f64>, JoseErr> {
        match claims.get(c) {
            None => Ok(None),
            Some(v) => v
                .as_f64()
                .map(Some)
                .ok_or(claim_failed(ERR_JWT_CLAIM_VALIDATION_FAILED, c)),
        }
    };
    let iat = number("iat")?.ok_or(claim_failed(ERR_JWT_CLAIM_VALIDATION_FAILED, "iat"))?;
    if let Some(nbf) = number("nbf")? {
        if nbf > now {
            return Err(claim_failed(ERR_JWT_CLAIM_VALIDATION_FAILED, "nbf"));
        }
    }
    if let Some(exp) = number("exp")? {
        if exp <= now {
            return Err(claim_failed(ERR_JWT_EXPIRED, "exp"));
        }
    }
    let age = now - iat;
    if age > wc.max_token_age_seconds {
        return Err(claim_failed(ERR_JWT_EXPIRED, "iat"));
    }
    if age < 0.0 {
        return Err(claim_failed(ERR_JWT_CLAIM_VALIDATION_FAILED, "iat"));
    }
    Ok(claims)
}

const LOG_PER_MINUTE: u32 = 20;

/// A value from a token nobody has verified, made safe for a log line:
/// control and non-ASCII characters out, length cut.
fn clean(v: Option<&Value>) -> String {
    match v {
        Some(Value::String(s)) => {
            let mut out = String::new();
            for c in s.chars() {
                if (' '..='~').contains(&c) {
                    out.push(c);
                } else {
                    // One '?' per UTF-16 code unit, as the JavaScript replace does.
                    out.extend(std::iter::repeat_n('?', c.len_utf16()));
                }
            }
            out.truncate(120);
            out
        }
        None => "undefined".into(),
        Some(Value::Bool(_)) => "boolean".into(),
        Some(Value::Number(_)) => "number".into(),
        Some(Value::Null | Value::Array(_) | Value::Object(_)) => "object".into(),
    }
}

/// `decodeJwt`: the claims of a token, unverified, or `None`.
fn decode_claims(token: &str) -> Option<Map<String, Value>> {
    let parts: Vec<&str> = token.split('.').collect();
    if parts.len() != 3 || parts[1].is_empty() {
        return None;
    }
    json_object(parts[1])
}

/// One log line for a token refused over a claim, so Workers Logs shows what a
/// real token carries. Only the NAME of the failing claim and the token's `iss`
/// and `client_id` values, cleaned: never the token, `sub`, email or any other
/// claim. Capped per isolate and minute, so a flood of forged tokens cannot
/// become a flood of logs.
fn log_refusal(now: u64, token: &str, claim: &str) {
    let allowed = LOG_WINDOW.with(|w| {
        let (mut start, mut logged) = w.get();
        if now.saturating_sub(start) > 60_000 {
            start = now;
            logged = 0;
        }
        logged += 1;
        w.set((start, logged));
        logged <= LOG_PER_MINUTE
    });
    if !allowed {
        return;
    }
    let c = decode_claims(token);
    let iss = c.as_ref().and_then(|c| c.get("iss"));
    let client_id = c.as_ref().and_then(|c| c.get("client_id"));
    console(
        Level::Error,
        &format!(
            "workos token refused: claim={} iss={} client_id={}",
            clean(Some(&Value::String(claim.to_owned()))),
            clean(iss),
            clean(client_id)
        ),
    );
}

fn non_empty_str<'a>(c: &'a Map<String, Value>, k: &str) -> Option<&'a str> {
    c.get(k).and_then(Value::as_str).filter(|s| !s.is_empty())
}

pub async fn verify_workos_token<P: Platform>(p: &P, cfg: &Config, token: &str) -> WorkosOutcome {
    if !workos_enabled(cfg) {
        return WorkosOutcome::Invalid;
    }
    let Some(wc) = config(cfg) else {
        console(
            Level::Error,
            "workos sign-in is enabled but its configuration is incomplete",
        );
        return WorkosOutcome::Unavailable;
    };

    let claims = match jwt_verify(p, &wc, token).await {
        Ok(c) => c,
        Err(JoseErr::Verdict { code, claim }) => {
            log_refusal(p.now_ms(), token, claim.unwrap_or(code));
            return WorkosOutcome::Invalid;
        }
        Err(JoseErr::Ours) => return WorkosOutcome::Unavailable,
    };

    if non_empty_str(&claims, "sub").is_none() {
        return WorkosOutcome::Invalid;
    }
    // A token minted for another WorkOS app that shares our issuer and audience.
    let client_ok = claims
        .get("client_id")
        .and_then(Value::as_str)
        .is_some_and(|c| wc.client_ids.iter().any(|x| x == c));
    if !client_ok {
        log_refusal(p.now_ms(), token, "client_id");
        return WorkosOutcome::Invalid;
    }
    // The session id is what revocation is keyed on; a token without one cannot be revoked.
    let Some(sid) = claims
        .get("sid")
        .and_then(Value::as_str)
        .filter(|s| is_person_id(s))
    else {
        return WorkosOutcome::Invalid;
    };
    // No canonical id means no metering identity. Falling back to the raw `sub`
    // would split one person across two quota identities, so it is refused.
    let Some(person_id) = claims
        .get(PERSON_ID_CLAIM)
        .and_then(Value::as_str)
        .filter(|s| is_person_id(s))
    else {
        return WorkosOutcome::Forbidden("no_person_id".into());
    };

    WorkosOutcome::Ok(AuthContext {
        // Metered and capped per person, on the canonical person id. The api key
        // id is the session, prefixed so it cannot be read as a real key id.
        api_key_id: format!("workos:sid:{sid}"),
        owner_entity_id: person_id.to_owned(),
        owner_person_id: Some(person_id.to_owned()),
        key_type: WORKOS_KEY_TYPE.to_owned(),
        tier: "community".to_owned(),
        monthly_token_cap: wc.monthly_token_cap,
        scopes: Some(FIXED_GRANT.to_vec()),
        training_consent: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::{auth_failure, authenticate, take_logs, AuthResult};
    use crate::http::Req;
    use crate::keys::permitted_scopes;
    use crate::platform::HttpResponse;
    use crate::testing::{block_on, FakePlatform};
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use rsa::traits::PublicKeyParts;
    use rsa::PrivateKey;
    use serde_json::json;
    use sha2::Digest;
    use std::rc::Rc;
    use std::sync::OnceLock;

    const CLIENT: &str = "client_test";
    const ISSUER: &str = "https://auth.example.test/user_management/client_test";
    const AUDIENCE: &str = "https://api.shamwari.example";
    const JWKS_URL: &str = "https://api.workos.com/sso/jwks/client_test";

    /// Throw-away keys generated in memory for the run. No key material is committed.
    fn signing() -> &'static PrivateKey {
        static K: OnceLock<PrivateKey> = OnceLock::new();
        K.get_or_init(|| PrivateKey::new(&mut getrandom::SysRng, 2048).expect("key"))
    }
    fn other_signing() -> &'static PrivateKey {
        static K: OnceLock<PrivateKey> = OnceLock::new();
        K.get_or_init(|| PrivateKey::new(&mut getrandom::SysRng, 2048).expect("key"))
    }

    fn b64u(bytes: &[u8]) -> String {
        URL_SAFE_NO_PAD.encode(bytes)
    }

    fn public_jwk() -> Value {
        let k = signing().to_public_key();
        json!({
            "kty": "RSA", "n": b64u(&k.n_bytes()), "e": b64u(&k.e_bytes()),
            "kid": "k1", "alg": "RS256", "use": "sig",
        })
    }

    fn sign_with(header: &Value, claims: &Value, key: &PrivateKey) -> String {
        let input = format!(
            "{}.{}",
            b64u(header.to_string().as_bytes()),
            b64u(claims.to_string().as_bytes())
        );
        let sig = key.pkcs1v15_sign::<Sha256>(input.as_bytes()).expect("sign");
        format!("{input}.{}", b64u(&sig))
    }

    fn hmac_sha256(key: &[u8], msg: &[u8]) -> Vec<u8> {
        let mut k = if key.len() > 64 {
            Sha256::digest(key).to_vec()
        } else {
            key.to_vec()
        };
        k.resize(64, 0);
        let pad = |x: u8| k.iter().map(|b| b ^ x).collect::<Vec<u8>>();
        let mut inner = Sha256::new();
        inner.update(pad(0x36));
        inner.update(msg);
        let mut outer = Sha256::new();
        outer.update(pad(0x5c));
        outer.update(inner.finalize());
        outer.finalize().to_vec()
    }

    struct World {
        p: FakePlatform,
        /// What the JWKS endpoint answers; `None` is a network error.
        jwks: Rc<RefCell<Option<(u16, String)>>>,
    }

    impl World {
        fn new() -> Self {
            reset_jwks_cache();
            take_logs();
            let p = FakePlatform::new();
            let jwks = Rc::new(RefCell::new(Some((
                200,
                json!({ "keys": [public_jwk()] }).to_string(),
            ))));
            let j = jwks.clone();
            p.on_fn(move |r| {
                (r.url == JWKS_URL).then(|| match j.borrow().clone() {
                    Some((status, body)) => Ok(HttpResponse {
                        status,
                        headers: vec![],
                        body,
                    }),
                    None => Err("network down".into()),
                })
            });
            p.on("https://core.example/rollup", 200, r#"{"total_tokens":0}"#);
            World { p, jwks }
        }

        fn now(&self) -> i64 {
            (self.p.now_ms() / 1000) as i64
        }

        fn claims(&self) -> Map<String, Value> {
            let now = self.now();
            json!({
                "person_id": "person_abc123", "sid": "session_1", "client_id": CLIENT,
                "iat": now, "exp": now + 300, "iss": ISSUER, "aud": AUDIENCE, "sub": "user_01",
            })
            .as_object()
            .unwrap()
            .clone()
        }

        /// A good token, after `f` has changed its claims.
        fn token(&self, f: impl FnOnce(&mut Map<String, Value>)) -> String {
            let mut c = self.claims();
            f(&mut c);
            sign_with(
                &json!({ "alg": "RS256", "kid": "k1" }),
                &Value::Object(c),
                signing(),
            )
        }

        fn good(&self) -> String {
            self.token(|_| {})
        }

        fn verify(&self, t: &str, cfg: &Config) -> WorkosOutcome {
            block_on(verify_workos_token(&self.p, cfg, t))
        }

        fn status(&self, t: &str, cfg: &Config) -> &'static str {
            match self.verify(t, cfg) {
                WorkosOutcome::Ok(_) => "ok",
                WorkosOutcome::Invalid => "invalid",
                WorkosOutcome::Forbidden(_) => "forbidden",
                WorkosOutcome::Unavailable => "unavailable",
            }
        }

        fn calls(&self) -> Vec<String> {
            self.p
                .requests
                .borrow()
                .iter()
                .map(|r| r.url.clone())
                .collect()
        }

        fn jwks_calls(&self) -> usize {
            self.calls().iter().filter(|u| *u == JWKS_URL).count()
        }
    }

    fn env() -> Config {
        Config {
            core_url: Some("https://core.example".into()),
            shamwari_core_token: Some("core-token".into()),
            auth_cache_ttl: Some("60".into()),
            mind_available: Some("false".into()),
            workos_client_id: Some(CLIENT.into()),
            workos_issuer: Some(ISSUER.into()),
            workos_audience: Some(AUDIENCE.into()),
            workos_monthly_token_cap: Some("50000".into()),
            ..Default::default()
        }
    }

    fn bearer(t: &str) -> Req {
        Req {
            method: "POST".into(),
            url: "https://x/v1/chat/completions".into(),
            headers: vec![("authorization".into(), format!("Bearer {t}"))],
            body: "{}".into(),
        }
    }

    fn header_of(t: &str) -> Value {
        let h = t.split('.').next().unwrap();
        serde_json::from_slice(&URL_SAFE_NO_PAD.decode(h).unwrap()).unwrap()
    }

    // ---- a good token

    #[test]
    fn a_good_token_becomes_a_per_person_context_with_the_fixed_grant() {
        let w = World::new();
        assert_eq!(
            w.verify(&w.good(), &env()),
            WorkosOutcome::Ok(AuthContext {
                api_key_id: "workos:sid:session_1".into(),
                owner_entity_id: "person_abc123".into(),
                owner_person_id: Some("person_abc123".into()),
                key_type: "workos".into(),
                tier: "community".into(),
                monthly_token_cap: 50000,
                scopes: Some(vec![Scope::Community, Scope::Platform]),
                training_consent: None,
            })
        );
        let r = &w.p.requests.borrow()[0];
        assert_eq!(
            (r.method.as_str(), r.timeout_ms),
            ("GET", Some(JWKS_TIMEOUT_MS))
        );
    }

    #[test]
    fn is_accepted_through_authenticate_and_the_key_service_is_never_asked() {
        let w = World::new();
        let r = block_on(authenticate(&w.p, &env(), &bearer(&w.good())));
        assert!(matches!(r, AuthResult::Ok(_)));
        assert_eq!(w.calls(), [JWKS_URL]);
        assert!(w.p.kv.borrow().is_empty(), "a WorkOS token is never cached");
    }

    #[test]
    fn fetches_the_jwks_once_for_many_tokens() {
        let w = World::new();
        for _ in 0..3 {
            assert_eq!(w.status(&w.good(), &env()), "ok");
        }
        assert_eq!(w.jwks_calls(), 1);
    }

    #[test]
    fn refetches_the_jwks_once_the_cache_age_has_passed() {
        let w = World::new();
        assert_eq!(w.status(&w.good(), &env()), "ok");
        *w.p.now.borrow_mut() += JWKS_CACHE_MAX_AGE_MS;
        assert_eq!(w.status(&w.good(), &env()), "ok");
        assert_eq!(w.jwks_calls(), 2);
        // A stale set whose refresh fails is unavailable, not believed.
        *w.p.now.borrow_mut() += JWKS_CACHE_MAX_AGE_MS;
        *w.jwks.borrow_mut() = Some((500, "boom".into()));
        assert_eq!(w.status(&w.good(), &env()), "unavailable");
    }

    // ---- a token that is not what it says

    #[test]
    fn refuses_alg_none() {
        let w = World::new();
        let c = Value::Object(w.claims());
        let t = format!(
            "{}.{}.",
            b64u(
                json!({ "alg": "none", "typ": "JWT" })
                    .to_string()
                    .as_bytes()
            ),
            b64u(c.to_string().as_bytes())
        );
        assert_eq!(w.status(&t, &env()), "invalid");
    }

    #[test]
    fn refuses_hs256_signed_with_the_public_key_as_the_shared_secret() {
        let w = World::new();
        let secret = public_jwk().to_string();
        let input = format!(
            "{}.{}",
            b64u(
                json!({ "alg": "HS256", "kid": "k1" })
                    .to_string()
                    .as_bytes()
            ),
            b64u(Value::Object(w.claims()).to_string().as_bytes())
        );
        let t = format!(
            "{input}.{}",
            b64u(&hmac_sha256(secret.as_bytes(), input.as_bytes()))
        );
        assert_eq!(w.status(&t, &env()), "invalid");
    }

    #[test]
    fn refuses_a_token_signed_by_a_key_that_is_not_in_the_jwks() {
        let w = World::new();
        let t = sign_with(
            &json!({ "alg": "RS256", "kid": "k1" }),
            &Value::Object(w.claims()),
            other_signing(),
        );
        assert_eq!(w.status(&t, &env()), "invalid");
    }

    #[test]
    fn refuses_an_unknown_kid_and_does_not_hammer_the_jwks_endpoint() {
        let w = World::new();
        for i in 0..4 {
            let t = sign_with(
                &json!({ "alg": "RS256", "kid": format!("forged-{i}") }),
                &Value::Object(w.claims()),
                signing(),
            );
            assert_eq!(w.status(&t, &env()), "invalid");
        }
        assert!(w.jwks_calls() <= 2);
        // After the cooldown one more fetch is allowed, still not one per kid.
        *w.p.now.borrow_mut() += JWKS_COOLDOWN_MS;
        let t = sign_with(
            &json!({ "alg": "RS256", "kid": "forged-x" }),
            &Value::Object(w.claims()),
            signing(),
        );
        assert_eq!(w.status(&t, &env()), "invalid");
        assert_eq!(w.jwks_calls(), 2);
    }

    #[test]
    fn refuses_each_claim_that_is_wrong_or_missing() {
        let w = World::new();
        let now = w.now();
        type Edit = Box<dyn Fn(&mut Map<String, Value>)>;
        let set = |k: &'static str, v: Value| -> Edit {
            Box::new(move |c| {
                c.insert(k.into(), v.clone());
            })
        };
        let drop = |k: &'static str| -> Edit {
            Box::new(move |c| {
                c.remove(k);
            })
        };
        let cases: Vec<(&str, Edit)> = vec![
            ("wrong audience", set("aud", json!("https://other.example"))),
            ("no audience at all", drop("aud")),
            ("wrong issuer", set("iss", json!("https://evil.example"))),
            ("no issuer at all", drop("iss")),
            ("no expiry (would be valid forever)", drop("exp")),
            ("no issued-at", drop("iat")),
            ("expired", set("exp", json!(now - 60))),
            (
                "another WorkOS app",
                set("client_id", json!("client_other")),
            ),
            ("no client id", drop("client_id")),
            ("no subject", drop("sub")),
            ("an empty subject", set("sub", json!(""))),
            ("no session id (cannot be revoked)", drop("sid")),
            (
                "a session id that is not inert",
                set("sid", json!("a b/../c")),
            ),
            (
                "a lifetime past the ceiling (iat 2h ago, exp still ahead)",
                Box::new(move |c| {
                    c.insert("iat".into(), json!(now - 7200));
                    c.insert("exp".into(), json!(now + 3600));
                }),
            ),
            ("an iat in the future", set("iat", json!(now + 60))),
            ("a not-before in the future", set("nbf", json!(now + 60))),
            ("an exp that is not a number", set("exp", json!("soon"))),
            ("an iss that is not a string", set("iss", json!([ISSUER]))),
        ];
        for (label, edit) in cases {
            let t = w.token(|c| edit(c));
            assert_eq!(w.status(&t, &env()), "invalid", "{label}");
        }
        // An aud array that contains ours is accepted, as jose accepts it.
        let t = w.token(|c| {
            c.insert("aud".into(), json!(["https://x.example", AUDIENCE]));
        });
        assert_eq!(w.status(&t, &env()), "ok");
    }

    #[test]
    fn refuses_garbage_that_only_looks_like_a_token() {
        let w = World::new();
        assert_eq!(w.status("aaa.bbb.ccc", &env()), "invalid");
        assert_eq!(w.status("a.b", &env()), "invalid");
        assert_eq!(w.jwks_calls(), 0, "refused before any key is fetched");
    }

    // ---- a real sign-in that is not provisioned is a 403, not a 401

    #[test]
    fn no_usable_person_id_is_forbidden_not_invalid() {
        let w = World::new();
        type Edit = Box<dyn Fn(&mut Map<String, Value>)>;
        let set = |v: Value| -> Edit {
            Box::new(move |c| {
                c.insert(PERSON_ID_CLAIM.into(), v.clone());
            })
        };
        let cases: Vec<(&str, Edit)> = vec![
            (
                "no canonical person id (never falls back to sub)",
                Box::new(|c| {
                    c.remove(PERSON_ID_CLAIM);
                }),
            ),
            ("an empty person id", set(json!(""))),
            ("a null person id", set(Value::Null)),
            ("a whitespace-only person id", set(json!("   "))),
            ("a non-string person id", set(json!(42))),
            (
                "a person id that is not inert in a URL",
                set(json!("p1&owner_entity_id=other")),
            ),
            ("a person id with a path in it", set(json!("../admin"))),
            (
                "a person id longer than Core accepts",
                set(json!("p".repeat(65))),
            ),
        ];
        for (label, edit) in cases {
            let t = w.token(|c| edit(c));
            assert_eq!(
                w.verify(&t, &env()),
                WorkosOutcome::Forbidden("no_person_id".into()),
                "{label}"
            );
        }
    }

    #[test]
    fn accepts_a_person_id_of_exactly_cores_maximum_length() {
        let w = World::new();
        let t = w.token(|c| {
            c.insert(PERSON_ID_CLAIM.into(), json!("p".repeat(64)));
        });
        assert_eq!(w.status(&t, &env()), "ok");
    }

    #[test]
    fn says_which_gap_it_is_over_http_and_does_not_look_like_a_bad_credential() {
        let w = World::new();
        let t = w.token(|c| {
            c.remove(PERSON_ID_CLAIM);
        });
        let res = auth_failure(&block_on(authenticate(&w.p, &env(), &bearer(&t))));
        assert_eq!(res.status, 403);
        assert_eq!(
            res.body_json(),
            json!({ "error": "not_provisioned", "reason": "no_person_id" })
        );
    }

    #[test]
    fn the_person_id_pattern_is_cores_owner_id_pattern() {
        // core/main.py: _ID = r"^[A-Za-z0-9_.:-]{1,64}$"
        assert_eq!(PERSON_ID_PATTERN, "^[A-Za-z0-9_.:-]{1,64}$");
        for (id, core) in [
            ("p".repeat(64), true),
            ("p".repeat(65), false),
            ("a b".into(), false),
            ("a&b".into(), false),
            (String::new(), false),
            ("ok_id.1:2-3".into(), true),
            ("é".into(), false),
        ] {
            assert_eq!(is_person_id(&id), core, "{id}");
        }
    }

    // ---- who is metered

    #[test]
    fn is_the_canonical_person_id_and_the_same_person_across_sessions() {
        let w = World::new();
        let a = w.verify(
            &w.token(|c| {
                c.insert("sub".into(), json!("user_A"));
                c.insert("sid".into(), json!("s1"));
            }),
            &env(),
        );
        let b = w.verify(
            &w.token(|c| {
                c.insert("sub".into(), json!("user_B"));
                c.insert("sid".into(), json!("s2"));
            }),
            &env(),
        );
        let (WorkosOutcome::Ok(a), WorkosOutcome::Ok(b)) = (a, b) else {
            panic!("expected ok")
        };
        assert_eq!(a.owner_entity_id, "person_abc123");
        assert_eq!(b.owner_entity_id, a.owner_entity_id);
        assert_ne!(a.api_key_id, b.api_key_id);
    }

    #[test]
    fn puts_only_the_validated_id_into_the_quota_request_url() {
        let w = World::new();
        let WorkosOutcome::Ok(ctx) = w.verify(&w.good(), &env()) else {
            panic!("expected ok")
        };
        assert!(!block_on(crate::pricing::over_quota(&w.p, &env(), &ctx)));
        let calls = w.calls();
        let rollup = calls
            .iter()
            .find(|c| c.starts_with("https://core.example/rollup"))
            .unwrap();
        assert!(rollup.contains("owner_entity_id=person_abc123&"));
    }

    // ---- one fixed grant, and nothing in the token can change it

    #[test]
    fn a_token_with_no_permissions_roles_or_org_still_gets_community_and_platform() {
        let w = World::new();
        let WorkosOutcome::Ok(ctx) = w.verify(&w.good(), &env()) else {
            panic!("expected ok")
        };
        assert_eq!(ctx.scopes, Some(vec![Scope::Community, Scope::Platform]));
        assert_eq!(permitted_scopes(&ctx), [Scope::Community, Scope::Platform]);
        assert_eq!(ctx.tier, "community");
    }

    #[test]
    fn no_claim_gains_anything_same_grant_same_tier_same_cap_never_personal() {
        let w = World::new();
        let plain = w.verify(&w.good(), &env());
        for extra in [
            json!({ "permissions": ["data:admin", "data:read", "console:admin", "mongodb:access", "github:access", "fundi:admin", "manage-production-app"] }),
            json!({ "role": "admin", "roles": ["admin", "owner"] }),
            json!({ "org_id": "org_01", "org": "nyuchi" }),
            json!({ "permissions": ["shamwari:personal", "personal", "scope:personal"], "scope": "personal", "scopes": ["personal"] }),
            json!({ "act": { "sub": "someone-else" } }),
            json!({ "permissions": ["shamwari:community"] }),
        ] {
            let t = w.token(|c| {
                for (k, v) in extra.as_object().unwrap() {
                    c.insert(k.clone(), v.clone());
                }
            });
            let loaded = w.verify(&t, &env());
            assert_eq!(loaded, plain, "{extra}");
            let WorkosOutcome::Ok(ctx) = loaded else {
                panic!()
            };
            assert!(!permitted_scopes(&ctx).contains(&Scope::Personal));
        }
    }

    #[test]
    fn fixed_grant_is_exactly_community_and_platform() {
        assert_eq!(FIXED_GRANT, [Scope::Community, Scope::Platform]);
    }

    // ---- audience is optional by config, required when configured

    fn no_aud() -> Config {
        Config {
            workos_audience: None,
            ..env()
        }
    }

    #[test]
    fn audience_unset_accepts_any_aud_or_none() {
        let w = World::new();
        for t in [
            w.token(|c| {
                c.remove("aud");
            }),
            w.token(|c| {
                c.insert("aud".into(), json!("https://anything.example"));
            }),
            w.good(),
        ] {
            assert_eq!(w.status(&t, &no_aud()), "ok");
        }
    }

    #[test]
    fn audience_unset_still_refuses_everything_else() {
        let w = World::new();
        let now = w.now();
        let tokens = [
            w.token(|c| {
                c.insert("client_id".into(), json!("client_other"));
            }),
            w.token(|c| {
                c.remove("client_id");
            }),
            w.token(|c| {
                c.insert("iss".into(), json!("https://evil.example"));
            }),
            w.token(|c| {
                c.remove("iss");
            }),
            w.token(|c| {
                c.insert("exp".into(), json!(now - 60));
            }),
            w.token(|c| {
                c.remove("exp");
            }),
            w.token(|c| {
                c.insert("iat".into(), json!(now - 7200));
                c.insert("exp".into(), json!(now + 3600));
            }),
            w.token(|c| {
                c.remove("sid");
            }),
            sign_with(
                &json!({ "alg": "RS256", "kid": "k1" }),
                &Value::Object(w.claims()),
                other_signing(),
            ),
        ];
        for (i, t) in tokens.iter().enumerate() {
            assert_eq!(w.status(t, &no_aud()), "invalid", "case {i}");
        }
    }

    #[test]
    fn audience_set_still_requires_it_exactly() {
        let w = World::new();
        assert_eq!(
            w.status(
                &w.token(|c| {
                    c.remove("aud");
                }),
                &env()
            ),
            "invalid"
        );
        assert_eq!(
            w.status(
                &w.token(|c| {
                    c.insert("aud".into(), json!("https://other.example"));
                }),
                &env()
            ),
            "invalid"
        );
        assert_eq!(w.status(&w.good(), &env()), "ok");
    }

    // ---- a comma-separated client id list

    fn two() -> Config {
        Config {
            workos_client_id: Some(format!("{CLIENT}, client_second ,")),
            ..env()
        }
    }

    fn with_client(w: &World, id: Option<&str>) -> String {
        w.token(|c| match id {
            Some(id) => {
                c.insert("client_id".into(), json!(id));
            }
            None => {
                c.remove("client_id");
            }
        })
    }

    #[test]
    fn accepts_a_token_from_either_listed_application() {
        let w = World::new();
        assert_eq!(w.status(&with_client(&w, Some(CLIENT)), &two()), "ok");
        assert_eq!(
            w.status(&with_client(&w, Some("client_second")), &two()),
            "ok"
        );
    }

    #[test]
    fn refuses_a_token_from_any_application_not_listed() {
        let w = World::new();
        assert_eq!(
            w.status(&with_client(&w, Some("client_third")), &two()),
            "invalid"
        );
        assert_eq!(
            w.status(&with_client(&w, Some("client_secon")), &two()),
            "invalid"
        );
        assert_eq!(w.status(&with_client(&w, None), &two()), "invalid");
    }

    #[test]
    fn a_list_with_no_id_in_it_is_unavailable_never_accept_any() {
        let w = World::new();
        for list in [",", " , ", ",,"] {
            let cfg = Config {
                workos_client_id: Some(list.into()),
                ..env()
            };
            assert_eq!(w.status(&w.good(), &cfg), "unavailable", "{list:?}");
            assert_eq!(
                w.status(&with_client(&w, Some("client_second")), &cfg),
                "unavailable"
            );
        }
    }

    #[test]
    fn the_keys_still_come_from_the_first_id() {
        let w = World::new();
        let mut c = w.claims();
        c.insert("client_id".into(), json!("client_second"));
        let t = sign_with(
            &json!({ "alg": "RS256", "kid": "k1" }),
            &Value::Object(c),
            other_signing(),
        );
        assert_eq!(w.status(&t, &two()), "invalid");
        assert!(w.calls().iter().all(|u| u == JWKS_URL));
    }

    // ---- every issuer the production config lists is accepted, and no other

    fn listed() -> Vec<String> {
        let toml = include_str!("../wrangler.toml");
        let line = toml
            .lines()
            .find(|l| l.starts_with("WORKOS_ISSUER"))
            .expect("WORKOS_ISSUER in wrangler.toml");
        let v = line.split_once('=').unwrap().1.trim().trim_matches('"');
        v.split(',').map(str::to_owned).collect()
    }

    #[test]
    fn lists_the_four_expected_issuers() {
        assert_eq!(
            listed(),
            [
                "https://auth.mukoko.com",
                "https://auth.mukoko.com/user_management/client_01KQBBSMQTSMTBN7HEC9KQBJC0",
                "https://accounts.mukoko.com",
                "https://accounts.mukoko.com/user_management/client_01KQBBSMQTSMTBN7HEC9KQBJC0",
            ]
        );
    }

    #[test]
    fn accepts_each_listed_issuer_with_or_without_a_trailing_slash_and_no_other() {
        let w = World::new();
        let cfg = Config {
            workos_issuer: Some(listed().join(",")),
            ..env()
        };
        for iss in listed() {
            for i in [iss.clone(), format!("{iss}/")] {
                let t = w.token(|c| {
                    c.insert("iss".into(), json!(i));
                });
                assert_eq!(w.status(&t, &cfg), "ok", "{i}");
            }
        }
        for iss in [
            "https://api.workos.com/user_management/client_01KQBBSMQTSMTBN7HEC9KQBJC0",
            "https://auth.mukoko.com/user_management/client_other",
            "https://auth.mukoko.com.evil.example",
            "http://auth.mukoko.com",
        ] {
            let t = w.token(|c| {
                c.insert("iss".into(), json!(iss));
            });
            assert_eq!(w.status(&t, &cfg), "invalid", "{iss}");
        }
    }

    // ---- a refused token is logged by claim name, iss and client_id only

    #[test]
    fn one_line_with_the_claim_name_and_the_two_values() {
        type Edit = Box<dyn Fn(&mut Map<String, Value>, i64)>;
        let cases: Vec<(&str, Edit, String)> = vec![
            (
                "iss",
                Box::new(|c, _| {
                    c.insert("iss".into(), json!("https://unknown.example"));
                }),
                "claim=iss iss=https://unknown.example client_id=client_test".into(),
            ),
            (
                "client_id",
                Box::new(|c, _| {
                    c.insert("client_id".into(), json!("client_zzz"));
                }),
                format!("claim=client_id iss={ISSUER} client_id=client_zzz"),
            ),
            (
                "exp",
                Box::new(|c, now| {
                    c.insert("exp".into(), json!(now - 60));
                }),
                "claim=exp".into(),
            ),
        ];
        for (name, edit, want) in cases {
            let w = World::new();
            let now = w.now();
            let t = w.token(|c| edit(c, now));
            assert_eq!(w.status(&t, &env()), "invalid");
            let lines = take_logs();
            assert_eq!(lines.len(), 1, "{name}: {lines:?}");
            assert!(lines[0].contains("workos token refused"), "{name}");
            assert!(lines[0].contains(&want), "{name}: {}", lines[0]);
        }
    }

    #[test]
    fn never_prints_the_token_or_the_person_and_cleans_control_characters_out_of_iss() {
        let w = World::new();
        let t = w.token(|c| {
            c.insert("iss".into(), json!("https://evil.example\nINJECTED line"));
        });
        w.status(&t, &env());
        let out = take_logs().join("\n");
        assert!(!out.contains(&t));
        assert_eq!(out.split('\n').count(), 1);
        assert!(out.contains("iss=https://evil.example?INJECTED"));
        for personal in ["person_abc123", "sub=", "sid=", "@"] {
            assert!(!out.contains(personal), "{personal}");
        }
    }

    #[test]
    fn a_bad_signature_on_a_good_token_is_not_logged_as_a_claim_failure() {
        let w = World::new();
        let t = sign_with(
            &json!({ "alg": "RS256", "kid": "k1" }),
            &Value::Object(w.claims()),
            other_signing(),
        );
        assert_eq!(w.status(&t, &env()), "invalid");
        let out = take_logs().join("\n");
        assert!(!out.contains("claim=iss"));
        assert!(out.contains("claim=ERR_JWS_SIGNATURE_VERIFICATION_FAILED"));
    }

    #[test]
    fn the_log_is_capped_so_a_flood_of_forged_tokens_is_not_a_flood_of_logs() {
        let w = World::new();
        for i in 0..60 {
            let t = w.token(|c| {
                c.insert("iss".into(), json!(format!("https://flood{i}.example")));
            });
            w.status(&t, &env());
        }
        assert!(take_logs().len() <= 20);
        // A new minute is a new allowance.
        *w.p.now.borrow_mut() += 60_001;
        let t = w.token(|c| {
            c.insert("iss".into(), json!("https://later.example"));
        });
        w.status(&t, &env());
        assert_eq!(take_logs().len(), 1);
    }

    #[test]
    fn clean_counts_utf16_units_and_cuts_at_120() {
        assert_eq!(clean(Some(&json!("a\u{1F600}b"))), "a??b");
        assert_eq!(clean(Some(&json!("x".repeat(200)))).len(), 120);
        assert_eq!(clean(None), "undefined");
        assert_eq!(clean(Some(&json!(5))), "number");
        assert_eq!(clean(Some(&Value::Null)), "object");
    }

    // ---- issuer lists and exact matching

    #[test]
    fn a_comma_separated_issuer_list_accepts_any_listed_and_refuses_one_from_none() {
        let w = World::new();
        let other = "https://other-issuer.example.test";
        let cfg = Config {
            workos_issuer: Some(format!("{other},{ISSUER}")),
            ..env()
        };
        assert_eq!(w.status(&w.good(), &cfg), "ok");
        let t = w.token(|c| {
            c.insert("iss".into(), json!(format!("{other}/")));
        });
        assert_eq!(w.status(&t, &cfg), "ok");
        let t = w.token(|c| {
            c.insert("iss".into(), json!("https://evil.example"));
        });
        assert_eq!(w.status(&t, &cfg), "invalid");
    }

    #[test]
    fn an_issuer_list_with_an_empty_entry_is_unavailable_not_permissive() {
        let w = World::new();
        let cfg = Config {
            workos_issuer: Some(format!("{ISSUER},")),
            ..env()
        };
        assert_eq!(w.status(&w.good(), &cfg), "unavailable");
    }

    #[test]
    fn the_issuer_is_matched_exactly_except_for_one_trailing_slash() {
        let w = World::new();
        let slash = format!("{ISSUER}/");
        for (configured, in_token) in [
            (ISSUER, slash.as_str()),
            (slash.as_str(), ISSUER),
            (ISSUER, ISSUER),
            (slash.as_str(), slash.as_str()),
        ] {
            let cfg = Config {
                workos_issuer: Some(configured.into()),
                ..env()
            };
            let t = w.token(|c| {
                c.insert("iss".into(), json!(in_token));
            });
            assert_eq!(w.status(&t, &cfg), "ok", "{configured} / {in_token}");
        }
        for in_token in [
            format!("{ISSUER}x"),
            ISSUER.replace("auth.example.test", "evil.example"),
            ISSUER[..ISSUER.len() - 4].to_owned(),
            format!("{ISSUER}//"),
            format!("{ISSUER}?a=b"),
            ISSUER.to_uppercase(),
        ] {
            let t = w.token(|c| {
                c.insert("iss".into(), json!(in_token));
            });
            assert_eq!(w.status(&t, &env()), "invalid", "{in_token}");
        }
    }

    #[test]
    fn an_issuer_of_only_slashes_is_unavailable() {
        let w = World::new();
        let cfg = Config {
            workos_issuer: Some("///".into()),
            ..env()
        };
        assert_eq!(w.status(&w.good(), &cfg), "unavailable");
    }

    // ---- failing closed

    #[test]
    fn is_unavailable_when_the_jwks_endpoint_is_down_or_throws() {
        let w = World::new();
        *w.jwks.borrow_mut() = Some((500, "boom".into()));
        assert_eq!(w.status(&w.good(), &env()), "unavailable");
        let w = World::new();
        *w.jwks.borrow_mut() = None;
        assert_eq!(w.status(&w.good(), &env()), "unavailable");
        // A failed fetch is not remembered: the next token fetches again.
        *w.jwks.borrow_mut() = Some((200, json!({ "keys": [public_jwk()] }).to_string()));
        assert_eq!(w.status(&w.good(), &env()), "ok");
    }

    #[test]
    fn is_invalid_not_unavailable_for_a_token_jose_cannot_make_sense_of() {
        let w = World::new();
        let good = w.good();
        let parts: Vec<&str> = good.split('.').collect();
        let mut hdr = header_of(&good);
        hdr["crit"] = json!(["exp"]);
        hdr["exp"] = json!(1);
        let crit = format!(
            "{}.{}.{}",
            b64u(hdr.to_string().as_bytes()),
            parts[1],
            parts[2]
        );
        assert_eq!(w.status(&crit, &env()), "invalid");
        let t = format!(
            "{}.{}.AAAA",
            b64u(
                json!({ "alg": "RS256", "kid": "k1" })
                    .to_string()
                    .as_bytes()
            ),
            b64u(b"{}")
        );
        assert_eq!(w.status(&t, &env()), "invalid");
    }

    #[test]
    fn is_unavailable_when_the_jwks_answer_is_malformed() {
        for body in [
            json!({ "not": "a key set" }).to_string(),
            "<html>".into(),
            json!({ "keys": ["x"] }).to_string(),
        ] {
            let w = World::new();
            *w.jwks.borrow_mut() = Some((200, body.clone()));
            assert_eq!(w.status(&w.good(), &env()), "unavailable", "{body}");
        }
    }

    #[test]
    fn a_key_under_2048_bits_or_a_private_key_in_the_set_is_ours_not_the_tokens() {
        let mut short = public_jwk();
        short["n"] = json!(b64u(&[0xff; 128]));
        let mut private = public_jwk();
        private["d"] = json!("AQAB");
        for jwk in [short, private] {
            let w = World::new();
            *w.jwks.borrow_mut() = Some((200, json!({ "keys": [jwk] }).to_string()));
            assert_eq!(w.status(&w.good(), &env()), "unavailable");
        }
    }

    #[test]
    fn two_keys_that_both_match_are_a_verdict() {
        let w = World::new();
        let mut second = public_jwk();
        second["n"] = json!(b64u(&other_signing().to_public_key().n_bytes()));
        *w.jwks.borrow_mut() = Some((200, json!({ "keys": [public_jwk(), second] }).to_string()));
        assert_eq!(w.status(&w.good(), &env()), "invalid");
    }

    #[test]
    fn is_a_503_through_auth_failure_when_the_jwks_endpoint_is_down() {
        let w = World::new();
        *w.jwks.borrow_mut() = Some((500, "boom".into()));
        let res = auth_failure(&block_on(authenticate(&w.p, &env(), &bearer(&w.good()))));
        assert_eq!(res.status, 503);
    }

    #[test]
    fn is_unavailable_never_permissive_with_a_broken_configuration() {
        let w = World::new();
        let t = w.good();
        let cases: Vec<(&str, Config)> = vec![
            (
                "no issuer",
                Config {
                    workos_issuer: None,
                    ..env()
                },
            ),
            (
                "a blank audience",
                Config {
                    workos_audience: Some(String::new()),
                    ..env()
                },
            ),
            (
                "a whitespace audience",
                Config {
                    workos_audience: Some("   ".into()),
                    ..env()
                },
            ),
            (
                "no cap",
                Config {
                    workos_monthly_token_cap: None,
                    ..env()
                },
            ),
            (
                "a non-numeric cap",
                Config {
                    workos_monthly_token_cap: Some("lots".into()),
                    ..env()
                },
            ),
            (
                "a zero cap",
                Config {
                    workos_monthly_token_cap: Some("0".into()),
                    ..env()
                },
            ),
            (
                "a fractional cap",
                Config {
                    workos_monthly_token_cap: Some("1.5".into()),
                    ..env()
                },
            ),
            (
                "a non-numeric max token age",
                Config {
                    workos_max_token_age_seconds: Some("forever".into()),
                    ..env()
                },
            ),
            (
                "a zero max token age",
                Config {
                    workos_max_token_age_seconds: Some("0".into()),
                    ..env()
                },
            ),
            (
                "a non-HTTPS JWKS URL",
                Config {
                    workos_jwks_url: Some("http://api.workos.com/sso/jwks/x".into()),
                    ..env()
                },
            ),
            (
                "an unparseable JWKS URL",
                Config {
                    workos_jwks_url: Some("not a url".into()),
                    ..env()
                },
            ),
        ];
        for (label, cfg) in cases {
            assert_eq!(w.status(&t, &cfg), "unavailable", "{label}");
        }
        assert!(w.calls().is_empty());
    }

    #[test]
    fn a_configured_jwks_url_and_max_age_are_used() {
        let w = World::new();
        w.p.on(
            "https://keys.example/jwks",
            200,
            &json!({ "keys": [public_jwk()] }).to_string(),
        );
        let cfg = Config {
            workos_jwks_url: Some("https://keys.example/jwks".into()),
            workos_max_token_age_seconds: Some("60".into()),
            ..env()
        };
        assert_eq!(w.status(&w.good(), &cfg), "ok");
        assert_eq!(w.calls(), ["https://keys.example/jwks"]);
        let now = w.now();
        let t = w.token(|c| {
            c.insert("iat".into(), json!(now - 61));
        });
        assert_eq!(w.status(&t, &cfg), "invalid");
    }

    #[test]
    fn is_off_and_says_invalid_when_workos_is_not_configured_at_all() {
        let w = World::new();
        let cfg = Config {
            workos_client_id: None,
            ..env()
        };
        assert_eq!(w.status(&w.good(), &cfg), "invalid");
        assert!(w.calls().is_empty());
    }

    // ---- the two credential shapes stay apart

    #[test]
    fn does_not_send_an_sk_key_to_the_workos_verifier() {
        let w = World::new();
        block_on(authenticate(
            &w.p,
            &env(),
            &bearer("sk_shamwari_abcdefghijklmnop"),
        ));
        assert!(!w.calls().iter().any(|u| u == JWKS_URL));
    }

    #[test]
    fn refuses_a_bearer_that_is_neither_shape_without_any_outbound_call() {
        let w = World::new();
        assert_eq!(
            block_on(authenticate(&w.p, &env(), &bearer("not-a-credential"))),
            AuthResult::Invalid
        );
        assert!(w.calls().is_empty());
    }

    #[test]
    fn looks_like_jwt_is_the_typescript_pattern() {
        for (s, want) in [
            ("a.b.c", true),
            ("a.b.", true),
            ("a..c", false),
            (".b.c", false),
            ("a.b", false),
            ("a.b.c.d", false),
            ("a+.b.c", false),
            ("a.b.c\n", false),
            ("", false),
        ] {
            assert_eq!(looks_like_jwt(s), want, "{s:?}");
        }
    }

    #[test]
    fn the_default_jwks_url_encodes_the_client_id() {
        assert_eq!(encode_uri_component("client_01/a b"), "client_01%2Fa%20b");
        let cfg = Config {
            workos_client_id: Some("client ü".into()),
            ..env()
        };
        assert_eq!(
            config(&cfg).unwrap().jwks_url,
            "https://api.workos.com/sso/jwks/client%20%C3%BC"
        );
    }
}
