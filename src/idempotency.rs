//! Port of `shamwari/gateway/src/idempotency.ts`: idempotent requests. The
//! same `Idempotency-Key` with the same request is the same answer, so a retry
//! on a flaky link costs nothing and is not billed or stored a second time.
//!
//! It lives in the `AUTH_CACHE` KV namespace under an `idem:` prefix. That is
//! the edge, and it is **best effort, not exactly-once**: KV is eventually
//! consistent and the read-then-write is not atomic. A first attempt that runs
//! past 60 s (KV's minimum lifetime for the in-flight marker) can be retried
//! into a second run.
//!
//! What is stored, and for how long: the response body of a request that
//! carried a key, for ten minutes, scoped to the calling entity. Personal
//! scope never reaches here (it is refused in `admit()` first), and
//! [`complete`] refuses it itself too.

use crate::auth::{console, Level};
use crate::platform::Platform;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// `KEY_PATTERN.source`, as the 400 for a malformed key reports it.
pub const KEY_PATTERN: &str = "^[A-Za-z0-9_.:-]{1,64}$";

/// `KEY_PATTERN.test(key)`.
pub fn is_valid_key(key: &str) -> bool {
    (1..=64).contains(&key.len())
        && key
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"_.:-".contains(&b))
}

/// Long enough for a client to retry over a bad link; short, because it holds an answer.
pub const DONE_TTL_SECONDS: u64 = 600;
/// A started request that never finished expires. 60 s is KV's minimum.
pub const PENDING_TTL_SECONDS: u64 = 60;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StoredResponse {
    pub status: u16,
    pub body: String,
}

#[derive(Serialize, Deserialize)]
struct Done {
    fp: String,
    response: StoredResponse,
}

#[derive(Serialize, Deserialize)]
struct Pending {
    fp: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Begin {
    New,
    Replay {
        status: u16,
        body: String,
    },
    /// The key was used before for a different request.
    Conflict,
    /// The first attempt has not finished.
    InProgress,
}

pub fn sha256_hex(text: &str) -> String {
    crate::auth::sha256_hex(text)
}

/// `Number.prototype.toString()` for a JSON number, so a fingerprint is the
/// one the TypeScript Worker computes for the same request.
fn js_number_string(n: &serde_json::Number) -> String {
    if n.is_i64() || n.is_u64() {
        return n.to_string();
    }
    let f = n.as_f64().unwrap_or(f64::NAN);
    if !f.is_finite() {
        return "null".into();
    }
    if f == 0.0 {
        return "0".into();
    }
    // Shortest round-trip digits and exponent, then JavaScript's layout.
    let sci = format!("{:e}", f.abs());
    let (mantissa, exp) = sci.split_once('e').unwrap_or((&sci, "0"));
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    let k = digits.len() as i32;
    let n_exp = exp.parse::<i32>().unwrap_or(0) + 1;
    let sign = if f < 0.0 { "-" } else { "" };
    let body = if k <= n_exp && n_exp <= 21 {
        format!("{digits}{}", "0".repeat((n_exp - k) as usize))
    } else if 0 < n_exp && n_exp <= 21 {
        format!(
            "{}.{}",
            &digits[..n_exp as usize],
            &digits[n_exp as usize..]
        )
    } else if -6 < n_exp && n_exp <= 0 {
        format!("0.{}{digits}", "0".repeat((-n_exp) as usize))
    } else {
        let e = n_exp - 1;
        let e = if e >= 0 {
            format!("+{e}")
        } else {
            e.to_string()
        };
        if k == 1 {
            format!("{digits}e{e}")
        } else {
            format!("{}.{}e{e}", &digits[..1], &digits[1..])
        }
    };
    format!("{sign}{body}")
}

/// JSON with object keys sorted (by UTF-16 code units, as JavaScript sorts),
/// so the same request always hashes the same.
fn canonical(v: &Value) -> String {
    match v {
        Value::Array(a) => format!(
            "[{}]",
            a.iter().map(canonical).collect::<Vec<_>>().join(",")
        ),
        Value::Object(o) => {
            let mut keys: Vec<&String> = o.keys().collect();
            keys.sort_by(|a, b| a.encode_utf16().cmp(b.encode_utf16()));
            let parts: Vec<String> = keys
                .into_iter()
                .map(|k| format!("{}:{}", Value::String(k.clone()), canonical(&o[k])))
                .collect();
            format!("{{{}}}", parts.join(","))
        }
        Value::Number(n) => js_number_string(n),
        other => other.to_string(),
    }
}

pub fn fingerprint(request: &Value) -> String {
    sha256_hex(&canonical(request))
}

/// Scoped to the entity, and the caller's key is hashed, never stored.
pub fn storage_key(owner_entity_id: &str, key: &str) -> String {
    format!("idem:{owner_entity_id}:{}", sha256_hex(key))
}

// Two keys per request, not one. Workers KV allows one write a second to a
// key, and a fast answer finishes inside that second: a `done` write on the
// key `begin` just wrote would be refused and the retry would find a stale
// `pending`. The answer goes to its own key, and `begin` reads it first.
fn pending_key(k: &str) -> String {
    format!("{k}:p")
}
fn done_key(k: &str) -> String {
    format!("{k}:d")
}

async fn read<P: Platform, T: for<'de> Deserialize<'de>>(
    p: &P,
    key: &str,
) -> Result<Option<T>, String> {
    match p.kv_get(key).await? {
        None => Ok(None),
        Some(text) => {
            let v: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
            if v.is_null() {
                return Ok(None);
            }
            serde_json::from_value(v)
                .map(Some)
                .map_err(|e| e.to_string())
        }
    }
}

/// KV is a convenience here, not a dependency: a KV error is logged and the
/// request runs unprotected, because a retry-safety feature must never make a
/// request less reliable than sending no key.
pub async fn begin<P: Platform>(p: &P, k: &str, fp: &str) -> Begin {
    let attempt = async {
        if let Some(done) = read::<P, Done>(p, &done_key(k)).await? {
            return Ok(if done.fp == fp {
                Begin::Replay {
                    status: done.response.status,
                    body: done.response.body,
                }
            } else {
                Begin::Conflict
            });
        }
        if let Some(pending) = read::<P, Pending>(p, &pending_key(k)).await? {
            return Ok(if pending.fp == fp {
                Begin::InProgress
            } else {
                Begin::Conflict
            });
        }
        let marker =
            serde_json::to_string(&Pending { fp: fp.to_owned() }).map_err(|e| e.to_string())?;
        p.kv_put(&pending_key(k), &marker, Some(PENDING_TTL_SECONDS))
            .await?;
        Ok::<Begin, String>(Begin::New)
    };
    match attempt.await {
        Ok(b) => b,
        Err(e) => {
            console(
                Level::Warn,
                &format!("idempotency store unavailable, running unprotected {e}"),
            );
            Begin::New
        }
    }
}

/// Stores the answer. It takes the scope and refuses personal: the store does
/// not rely on its caller having kept personal scope out, because a stored
/// answer would be a second copy of the user's own data outside their pod.
pub async fn complete<P: Platform>(
    p: &P,
    k: &str,
    fp: &str,
    response: &StoredResponse,
    scope: &str,
) {
    if scope == "personal" {
        return;
    }
    let done = Done {
        fp: fp.to_owned(),
        response: response.clone(),
    };
    let stored = match serde_json::to_string(&done) {
        Ok(s) => p.kv_put(&done_key(k), &s, Some(DONE_TTL_SECONDS)).await,
        Err(e) => Err(e.to_string()),
    };
    if let Err(e) = stored {
        console(Level::Warn, &format!("idempotency answer not stored {e}"));
    }
}

/// A failed attempt is not an answer: let the retry run.
pub async fn release<P: Platform>(p: &P, k: &str) {
    if let Err(e) = p.kv_delete(&pending_key(k)).await {
        console(
            Level::Warn,
            &format!("idempotency marker not released; it expires on its own {e}"),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::fakes::Rec;
    use crate::auth::take_logs;
    use crate::testing::block_on;
    use serde_json::json;

    fn ok(body: &str) -> StoredResponse {
        StoredResponse {
            status: 200,
            body: body.into(),
        }
    }

    #[test]
    fn a_retry_with_the_same_key_and_request_is_the_same_answer() {
        let p = Rec::new();
        let k = storage_key("e1", "retry-1");
        let fp = fingerprint(&json!({ "messages": [{ "role": "user", "content": "hi" }] }));
        assert_eq!(block_on(begin(&p, &k, &fp)), Begin::New);
        block_on(complete(&p, &k, &fp, &ok("{\"a\":1}"), "platform"));
        assert_eq!(
            block_on(begin(&p, &k, &fp)),
            Begin::Replay {
                status: 200,
                body: "{\"a\":1}".into()
            }
        );
    }

    #[test]
    fn the_same_key_with_a_different_request_is_refused_not_answered() {
        let p = Rec::new();
        let k = storage_key("e1", "k");
        let a = fingerprint(&json!({ "messages": [{ "role": "user", "content": "a" }] }));
        let b = fingerprint(&json!({ "messages": [{ "role": "user", "content": "b" }] }));
        block_on(begin(&p, &k, &a));
        // In flight, then done: a different request is a conflict either way.
        assert_eq!(block_on(begin(&p, &k, &b)), Begin::Conflict);
        block_on(complete(&p, &k, &a, &ok("{}"), "community"));
        assert_eq!(block_on(begin(&p, &k, &b)), Begin::Conflict);
    }

    #[test]
    fn the_format_is_part_of_the_request_so_the_same_key_cannot_change_shape() {
        let base = json!({ "messages": [{ "role": "user", "content": "a" }] });
        let mut compact = base.clone();
        compact["format"] = json!("compact");
        assert_ne!(fingerprint(&base), fingerprint(&compact));
    }

    #[test]
    fn is_scoped_to_the_calling_entity() {
        let p = Rec::new();
        let fp = fingerprint(&json!({}));
        let a = storage_key("entity-a", "shared");
        let b = storage_key("entity-b", "shared");
        assert_ne!(a, b);
        block_on(begin(&p, &a, &fp));
        block_on(complete(&p, &a, &fp, &ok("{}"), "platform"));
        assert_eq!(block_on(begin(&p, &b, &fp)), Begin::New);
    }

    #[test]
    fn a_request_still_in_flight_is_in_progress_not_a_second_run() {
        let p = Rec::new();
        let k = storage_key("e1", "slow");
        let fp = fingerprint(&json!({ "x": 1 }));
        assert_eq!(block_on(begin(&p, &k, &fp)), Begin::New);
        assert_eq!(block_on(begin(&p, &k, &fp)), Begin::InProgress);
    }

    #[test]
    fn a_failed_attempt_is_not_an_answer_the_retry_runs() {
        let p = Rec::new();
        let k = storage_key("e1", "boom");
        let fp = fingerprint(&json!({}));
        assert_eq!(block_on(begin(&p, &k, &fp)), Begin::New);
        block_on(release(&p, &k));
        assert_eq!(block_on(begin(&p, &k, &fp)), Begin::New);
    }

    #[test]
    fn rejects_a_malformed_key() {
        for bad in ["has space", &"x".repeat(65), "café", ""] {
            assert!(!is_valid_key(bad), "{bad:?}");
        }
        for good in ["retry-1", "a.b:c_d", &"x".repeat(64)] {
            assert!(is_valid_key(good), "{good:?}");
        }
        assert_eq!(KEY_PATTERN, "^[A-Za-z0-9_.:-]{1,64}$");
    }

    #[test]
    fn stores_only_what_it_needs_the_response_for_ten_minutes_under_a_hashed_key() {
        let p = Rec::new();
        let k = storage_key("e1", "my-secret-looking-key");
        let fp = fingerprint(&json!({}));
        block_on(begin(&p, &k, &fp));
        block_on(complete(&p, &k, &fp, &ok("{}"), "platform"));
        let puts = p.puts.borrow();
        let ttls: Vec<_> = puts.iter().map(|(_, _, t)| *t).collect();
        assert_eq!(ttls, [Some(PENDING_TTL_SECONDS), Some(DONE_TTL_SECONDS)]);
        assert!(puts
            .iter()
            .all(|(k, v, _)| !k.contains("my-secret-looking-key")
                && !v.contains("my-secret-looking-key")));
        assert!(puts.iter().all(|(k, _, _)| k.starts_with("idem:e1:")));
        let done: Value = serde_json::from_str(&puts[1].1).unwrap();
        assert_eq!(
            done,
            json!({ "fp": fp, "response": { "status": 200, "body": "{}" } })
        );
    }

    #[test]
    fn the_store_refuses_a_personal_scope_answer_itself_whatever_the_caller_did() {
        let p = Rec::new();
        block_on(complete(&p, "idem:e1:x", "fp", &ok("{}"), "personal"));
        assert!(p.puts.borrow().is_empty());
        block_on(complete(&p, "idem:e1:x", "fp", &ok("{}"), "platform"));
        assert_eq!(p.puts.borrow().len(), 1);
    }

    #[test]
    fn a_kv_outage_runs_the_request_unprotected_rather_than_failing_it() {
        for (get, put) in [(true, false), (false, true)] {
            let p = Rec::new();
            p.faults.kv_get.set(get);
            p.faults.kv_put.set(put);
            let k = storage_key("e1", "outage");
            assert_eq!(block_on(begin(&p, &k, "fp")), Begin::New);
            block_on(complete(&p, &k, "fp", &ok("{}"), "platform"));
            block_on(release(&p, &k));
        }
        let p = Rec::new();
        p.faults.kv_delete.set(true);
        block_on(release(&p, "idem:e1:x"));
        let logs = take_logs();
        assert!(logs.iter().any(|l| l.contains("running unprotected")));
        assert!(logs.iter().any(|l| l.contains("answer not stored")));
        assert!(logs.iter().any(|l| l.contains("expires on its own")));
    }

    #[test]
    fn an_unreadable_stored_entry_runs_unprotected() {
        let p = Rec::new();
        let k = storage_key("e1", "junk");
        p.kv.borrow_mut()
            .insert(format!("{k}:d"), "<not json>".into());
        assert_eq!(block_on(begin(&p, &k, "fp")), Begin::New);
    }

    #[test]
    fn the_answer_goes_to_its_own_key_so_a_fast_answer_never_rewrites_the_marker_key() {
        let p = Rec::new();
        let k = storage_key("e1", "fast");
        block_on(begin(&p, &k, "fp"));
        block_on(complete(&p, &k, "fp", &ok("{}"), "platform"));
        let keys: Vec<String> = p.puts.borrow().iter().map(|(k, _, _)| k.clone()).collect();
        let mut unique = keys.clone();
        unique.dedup();
        assert_eq!(unique, keys);
        assert_eq!(keys, [format!("{k}:p"), format!("{k}:d")]);
    }

    #[test]
    fn a_stored_answer_wins_over_a_lingering_marker() {
        let p = Rec::new();
        let k = storage_key("e1", "both");
        block_on(begin(&p, &k, "fp"));
        block_on(complete(&p, &k, "fp", &ok("{\"done\":true}"), "platform"));
        assert!(p.kv.borrow().contains_key(&format!("{k}:p")));
        assert!(matches!(
            block_on(begin(&p, &k, "fp")),
            Begin::Replay { .. }
        ));
    }

    #[test]
    fn the_fingerprint_ignores_key_order_and_matches_typescript() {
        let a = json!({ "b": 1, "a": [true, null, "x"], "c": { "z": 1.5, "y": -0.000001 } });
        let b = json!({ "c": { "y": -0.000001, "z": 1.5 }, "a": [true, null, "x"], "b": 1 });
        assert_eq!(fingerprint(&a), fingerprint(&b));
        assert_eq!(
            canonical(&a),
            r#"{"a":[true,null,"x"],"b":1,"c":{"y":-0.000001,"z":1.5}}"#
        );
        // sha256 of the canonical form, as sha256Hex(canonical(...)) in TypeScript.
        assert_eq!(
            fingerprint(&json!({})),
            "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
        );
    }

    #[test]
    fn numbers_print_as_javascript_prints_them() {
        for (n, js) in [
            (json!(1.0), "1"),
            (json!(1.5), "1.5"),
            (json!(-0.5), "-0.5"),
            (json!(1e21), "1e+21"),
            (json!(1.5e21), "1.5e+21"),
            (json!(1e20), "100000000000000000000"),
            (json!(0.000001), "0.000001"),
            (json!(1e-7), "1e-7"),
            (json!(1.25e-7), "1.25e-7"),
            (json!(123456.789), "123456.789"),
            (json!(42), "42"),
            (json!(-3), "-3"),
            (json!(0.0), "0"),
        ] {
            assert_eq!(canonical(&n), js, "{n}");
        }
        // Keys sort by UTF-16 code units: U+FF61 sorts after U+1F600 in UTF-8, before it in UTF-16.
        let v = json!({ "\u{1F600}": 1, "\u{FF61}": 2 });
        assert!(canonical(&v).starts_with("{\"\u{1F600}\""));
    }
}
