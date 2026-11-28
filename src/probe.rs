//! Port of `shamwari/gateway/src/probe.ts`: the weekly degradation probe.
//! MAKES THE CLAIM CHECKABLE.
//!
//! The fallback path is unreachable on a healthy day: step 2 runs only when
//! step 1 fails. So it rots silently, and the rot is invisible until the
//! outage that needed it. This runs on a cron instead and writes the result to
//! `platform.serviceHealth`, so a rotted path is a row you can query rather
//! than a discipline someone has to remember.
//!
//! What it does NOT do: assert the paths are healthy. It records what
//! happened. Alerting belongs where the data is, not in the Worker.

use crate::ai_gateway::{via_gateway, via_workers_ai, InferenceResult, RequestMetadata};
use crate::config::Config;
use crate::failure::console;
use crate::lang::js;
use crate::model_router::{targets, tier_name, DEFAULT_MAX_OUTPUT_TOKENS};
use crate::platform::Platform;
use crate::sink::{enqueue, iso_timestamp, random_uuid};
use crate::types::{Message, Role, Scope, SinkDatabase, SinkMessage, Tier};
use serde::Serialize;
use serde_json::{json, Map, Value};

/// Deliberately dull, deliberately constant, deliberately not a real user's
/// question. It reaches Workers AI on a schedule and lands in Cloudflare's
/// logs, so it must carry nothing personal, and being constant is what lets
/// a latency change mean something. Constant also means cacheable, which is
/// why the gateway step is probed with `cf-aig-skip-cache`.
pub const PROBE_PROMPT: &str = "Reply with the single word: ok";

fn probe_messages() -> Vec<Message> {
    vec![Message {
        role: Role::User,
        content: PROBE_PROMPT.into(),
    }]
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PathResult {
    /// `gateway` or `workers-ai`.
    pub path: String,
    pub ok: bool,
    pub latency_ms: u64,
    /// What answered, when something did. Null when the step was unavailable.
    pub served: Option<String>,
    /// Set when the step failed by erring rather than merely returning unavailable.
    pub error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProbeReport {
    pub probe_id: String,
    pub tier: String,
    pub started_at: String,
    pub results: Vec<PathResult>,
    /// True when every step answered. The claim holds only in this case.
    pub all_paths_healthy: bool,
    /// True when at least one step answered: a user request would succeed.
    pub serviceable: bool,
}

fn timed(
    path: &str,
    started: u64,
    now: u64,
    r: Result<Option<InferenceResult>, String>,
) -> PathResult {
    let latency_ms = now.saturating_sub(started);
    match r {
        Ok(r) => PathResult {
            path: path.into(),
            ok: r.is_some(),
            latency_ms,
            served: r.map(|r| format!("{}/{}", r.provider, r.model)),
            error: None,
        },
        // A step that fails is a failed step, not a failed probe. The whole
        // point is to get a reading on both. `String(e)` of an Error.
        Err(e) => PathResult {
            path: path.into(),
            ok: false,
            latency_ms,
            served: None,
            error: Some(js::prefix(&format!("Error: {e}"), 300).to_owned()),
        },
    }
}

/// Exercise both paths independently, in sequence. Sequential rather than
/// parallel: simultaneous calls to the same model can trip its own rate
/// limiting and report a fault that the probe itself caused. `tier` is
/// economy (the cron's) or standard.
pub async fn probe<P: Platform>(p: &P, cfg: &Config, tier: Tier) -> ProbeReport {
    let all = targets(cfg);
    let target = all.get(tier);
    let probe_id = random_uuid(p);
    let started_at = iso_timestamp(p.now_ms());
    let messages = probe_messages();

    let meta = RequestMetadata {
        tier: target.tier,
        // Platform scope, and it must stay that way: the probe carries no
        // user data, so it has no business claiming any other scope in the logs.
        scope: Scope::Platform,
        owner_entity_id: "__probe__".into(),
        surface: cfg.surface.clone().unwrap_or_default(),
        request_id: probe_id.clone(),
    };

    let t0 = p.now_ms();
    let g = via_gateway(
        p,
        cfg,
        target,
        &messages,
        &meta,
        true,
        DEFAULT_MAX_OUTPUT_TOKENS,
        None,
    )
    .await
    .map_err(|e| e.to_string());
    let gateway = timed("gateway", t0, p.now_ms(), g);

    let t1 = p.now_ms();
    let w = via_workers_ai(p, target, &messages, meta.scope, DEFAULT_MAX_OUTPUT_TOKENS)
        .await
        .map(Some)
        .map_err(|e| e.to_string());
    let workers = timed("workers-ai", t1, p.now_ms(), w);

    let results = vec![gateway, workers];
    ProbeReport {
        probe_id,
        tier: tier_name(target.tier).into(),
        started_at,
        all_paths_healthy: results.iter().all(|r| r.ok),
        serviceable: results.iter().any(|r| r.ok),
        results,
    }
}

/// One `platform.serviceHealth` row per probe. A single document rather than
/// one per path: the interesting query is "when did a path last work", and
/// that reads better off a flat set of per-path booleans.
pub fn health_doc(report: &ProbeReport, now_ms: u64) -> SinkMessage {
    let ok = |path: &str| report.results.iter().any(|r| r.path == path && r.ok);
    let v = json!({
        "service": "shamwari-gateway",
        "check": "inference-degradation",
        "probeId": report.probe_id,
        "tier": report.tier,
        "allPathsHealthy": report.all_paths_healthy,
        "serviceable": report.serviceable,
        "paths": report.results,
        // Hoisted so the common alert ("a fallback path has rotted") is an
        // equality match rather than an array traversal.
        "gatewayOk": ok("gateway"),
        "workersAiOk": ok("workers-ai"),
        "startedAt": report.started_at,
        "createdAt": iso_timestamp(now_ms),
    });
    let doc: Map<String, Value> = match v {
        Value::Object(m) => m,
        _ => unreachable!(),
    };
    SinkMessage {
        database: SinkDatabase::Platform,
        collection: "serviceHealth".into(),
        doc,
    }
}

/// The cron (`scheduled` in `index.ts`): probe the economy tier, say so in
/// the Worker log when a path is down, and write the reading through the
/// same queue as everything else (so a Core outage delays the reading rather
/// than losing it). Nothing is queued while `SINK_ENABLED` is "false".
pub async fn scheduled<P: Platform>(p: &P, cfg: &Config) -> ProbeReport {
    let report = probe(p, cfg, Tier::Economy).await;
    if !report.all_paths_healthy {
        // Loud in the Worker log as well as durable in Mongo.
        console::error(&format!(
            "degradation probe: not all inference paths healthy {}",
            serde_json::to_string(&report.results).unwrap_or_default()
        ));
    }
    enqueue(p, cfg, &[health_doc(&report, p.now_ms())], "probe sink").await;
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai_gateway::CF_AIG_TOKEN;
    use crate::platform::HttpRequest;
    use crate::testing::{block_on, FakePlatform};

    const OK_BODY: &str = r#"{"choices":[{"message":{"content":"ok"}}],"usage":{"prompt_tokens":5,"completion_tokens":1}}"#;

    fn cfg() -> Config {
        Config {
            cf_account_id: Some("acct".into()),
            cf_gateway_id: Some("gw".into()),
            cf_route_name: Some("shamwari".into()),
            economy_model: Some("@cf/zai-org/glm-5.3-flash".into()),
            standard_model: Some("@cf/zai-org/glm-5.3".into()),
            surface: Some("shamwari.ai".into()),
            ..Default::default()
        }
    }

    fn world(ai_ok: bool) -> FakePlatform {
        let p = FakePlatform::new();
        p.secrets
            .borrow_mut()
            .insert(CF_AIG_TOKEN.into(), "aig".into());
        p.on_ai(move |_, _| {
            Some(if ai_ok {
                Ok(json!({ "response": "ok" }))
            } else {
                Err("workers ai down".into())
            })
        });
        p
    }

    fn header<'a>(r: &'a HttpRequest, name: &str) -> Option<&'a str> {
        r.headers
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }

    fn find<'a>(r: &'a ProbeReport, path: &str) -> &'a PathResult {
        r.results.iter().find(|x| x.path == path).unwrap()
    }

    #[test]
    fn exercises_both_paths_even_when_the_first_one_succeeds() {
        // infer() stops at the first success, so a probe built on it would
        // only ever report on the gateway.
        let p = world(true);
        p.on("https://gateway.ai.cloudflare.com", 200, OK_BODY);
        let r = block_on(probe(&p, &cfg(), Tier::Economy));
        let paths: Vec<&str> = r.results.iter().map(|x| x.path.as_str()).collect();
        assert_eq!(paths, ["gateway", "workers-ai"]);
        assert!(r.all_paths_healthy);
        // The gateway step is HTTP; workers-ai goes through the AI binding.
        assert_eq!(p.requests.borrow().len(), 1);
        assert_eq!(p.ai_calls.borrow().len(), 1);
    }

    #[test]
    fn skips_the_ai_gateway_cache_or_a_hit_reports_a_dead_provider_as_healthy() {
        let p = world(true);
        p.on("https://", 200, OK_BODY);
        block_on(probe(&p, &cfg(), Tier::Economy));
        assert_eq!(
            header(&p.requests.borrow()[0], "cf-aig-skip-cache"),
            Some("true")
        );
    }

    #[test]
    fn reports_a_rotted_fallback_while_the_service_is_still_serviceable() {
        let p = world(false);
        p.on("https://", 200, OK_BODY);
        let r = block_on(probe(&p, &cfg(), Tier::Economy));
        assert!(find(&r, "gateway").ok);
        assert!(!find(&r, "workers-ai").ok);
        assert!(!r.all_paths_healthy);
        assert!(r.serviceable);
    }

    #[test]
    fn records_a_failed_step_as_a_failed_step_not_a_failed_probe() {
        let p = world(false);
        p.on_fn(|_| Some(Err("dns exploded".into())));
        let r = block_on(probe(&p, &cfg(), Tier::Economy));
        assert_eq!(r.results.len(), 2);
        assert!(!r.serviceable);
        assert!(r.results.iter().all(|x| !x.ok));
    }

    #[test]
    fn captures_an_error_string_for_the_step_that_failed() {
        let p = world(false);
        p.on("https://", 200, OK_BODY);
        let r = block_on(probe(&p, &cfg(), Tier::Economy));
        let wai = find(&r, "workers-ai");
        assert!(!wai.ok);
        assert!(wai.error.as_deref().unwrap().contains("workers ai down"));
    }

    #[test]
    fn names_what_served_each_healthy_path() {
        let p = world(true);
        p.on("https://", 200, OK_BODY);
        let r = block_on(probe(&p, &cfg(), Tier::Standard));
        assert_eq!(
            find(&r, "gateway").served.as_deref(),
            Some("workers-ai/@cf/zai-org/glm-5.3")
        );
        assert_eq!(
            find(&r, "workers-ai").served.as_deref(),
            Some("workers-ai/@cf/zai-org/glm-5.3")
        );
    }

    #[test]
    fn sends_nothing_personal_it_reaches_workers_ai_on_a_schedule() {
        let p = world(true);
        p.on("https://", 200, OK_BODY);
        block_on(probe(&p, &cfg(), Tier::Economy));
        let reqs = p.requests.borrow();
        let mut bodies: Vec<Value> = reqs
            .iter()
            .map(|r| serde_json::from_str(r.body.as_deref().unwrap()).unwrap())
            .collect();
        bodies.extend(
            p.ai_calls
                .borrow()
                .iter()
                .map(|(_, input, _)| input.clone()),
        );
        for body in bodies {
            assert_eq!(body["messages"].as_array().unwrap().len(), 1);
            assert_eq!(body["messages"][0]["content"], PROBE_PROMPT);
        }
        // And the metadata it tags the log with claims platform scope only.
        let meta: Value =
            serde_json::from_str(header(&reqs[0], "cf-aig-metadata").unwrap()).unwrap();
        assert_eq!(meta["scope"], "platform");
        assert_eq!(meta["owner_entity_id"], "__probe__");
    }

    #[test]
    fn health_doc_writes_to_platform_service_health_with_per_path_booleans_hoisted() {
        let p = world(false);
        p.on("https://", 200, OK_BODY);
        let doc = health_doc(&block_on(probe(&p, &cfg(), Tier::Economy)), p.now_ms());
        assert_eq!(doc.database, SinkDatabase::Platform);
        assert_eq!(doc.collection, "serviceHealth");
        assert_eq!(doc.doc["gatewayOk"], true);
        assert!(!doc.doc.contains_key("directOk"));
        assert_eq!(doc.doc["workersAiOk"], false);
        assert_eq!(doc.doc["allPathsHealthy"], false);
        assert_eq!(doc.doc["check"], "inference-degradation");
        assert_eq!(doc.doc["paths"][1]["latencyMs"], 0);
        assert_eq!(doc.doc["startedAt"], "2026-01-01T00:00:00.000Z");
    }

    #[test]
    fn health_doc_carries_no_message_content() {
        let p = world(true);
        p.on("https://", 200, OK_BODY);
        let doc = health_doc(&block_on(probe(&p, &cfg(), Tier::Economy)), p.now_ms());
        assert!(!Value::Object(doc.doc).to_string().contains("Reply with"));
    }

    #[test]
    fn the_cron_logs_a_rotted_path_and_queues_the_reading_unless_the_sink_is_off() {
        let p = world(false);
        p.on("https://", 200, OK_BODY);
        console::take();
        let r = block_on(scheduled(&p, &cfg()));
        assert!(!r.all_paths_healthy);
        assert!(console::take()
            .iter()
            .any(|l| l.contains("degradation probe")));
        assert_eq!(p.queue.borrow().len(), 1);
        assert_eq!(p.queue.borrow()[0]["collection"], "serviceHealth");

        let off = Config {
            sink_enabled: Some("false".into()),
            ..cfg()
        };
        let p = world(true);
        p.on("https://", 200, OK_BODY);
        block_on(scheduled(&p, &off));
        assert!(p.queue.borrow().is_empty());
    }
}
