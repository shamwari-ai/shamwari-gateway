//! The Worker's `queue` and `scheduled` handlers at the function level (the
//! TypeScript `worker.queue` and `worker.scheduled`), and what the entry point
//! relies on from [`Resp`]. `worker_entry` itself is wasm32 only; everything
//! it decides is here.

use super::world::*;
use crate::gateway::{handle_queue, handle_scheduled};
use crate::http::Resp;
use crate::platform::HttpResponse;
use crate::testing::block_on;
use crate::types::{SinkDatabase, SinkMessage};
use serde_json::{json, Map, Value};

fn msg(db: SinkDatabase, collection: &str, request_id: &str) -> SinkMessage {
    let mut doc = Map::new();
    doc.insert("requestId".into(), json!(request_id));
    SinkMessage {
        database: db,
        collection: collection.into(),
        doc,
    }
}

fn core_sink(w: &World, status: u16) {
    w.p.fake.responders.borrow_mut().insert(
        0,
        Box::new(move |r| {
            r.url.ends_with("/sink/bulk").then(|| {
                Ok(HttpResponse {
                    status,
                    headers: vec![],
                    body: "{}".into(),
                })
            })
        }),
    );
}

#[test]
fn the_queue_handler_acks_a_good_batch_in_one_request_per_database() {
    let w = World::new();
    core_sink(&w, 200);
    let batch = vec![
        msg(SinkDatabase::Shamwari, "conversations", "r1"),
        msg(SinkDatabase::Platform, "usageEvents", "r1"),
        msg(SinkDatabase::Shamwari, "conversations", "r2"),
    ];
    let d = block_on(handle_queue(&w.p, &w.cfg, &batch));
    assert_eq!(d.error, None);
    let mut acked = d.acked.clone();
    acked.sort();
    assert_eq!(acked, [0, 1, 2]);
    let sent: Vec<_> = w
        .requests()
        .into_iter()
        .filter(|r| r.url == "https://core.example/sink/bulk")
        .collect();
    assert_eq!(sent.len(), 2);
    let auth = sent[0]
        .headers
        .iter()
        .find(|(k, _)| k == "Authorization")
        .map(|(_, v)| v.clone());
    assert_eq!(
        auth,
        Some(format!(
            "Bearer {}",
            w.cfg.shamwari_core_token.as_deref().unwrap()
        ))
    );
}

#[test]
fn the_queue_handler_fails_the_batch_on_a_transient_error_and_acks_nothing() {
    for status in [500, 503, 429, 401] {
        let w = World::new();
        core_sink(&w, status);
        let batch = vec![msg(SinkDatabase::Platform, "usageEvents", "r1")];
        let d = block_on(handle_queue(&w.p, &w.cfg, &batch));
        assert!(d.error.is_some(), "{status}");
        assert!(d.acked.is_empty());
    }
}

#[test]
fn the_queue_handler_parks_a_poison_document_and_logs_identifiers_only() {
    let w = World::new();
    w.p.fake.responders.borrow_mut().insert(
        0,
        Box::new(|r| {
            if !r.url.ends_with("/sink/bulk") {
                return None;
            }
            let poisoned = r.body.as_deref().unwrap_or("").contains("\"bad\"");
            Some(Ok(HttpResponse {
                status: if poisoned { 422 } else { 200 },
                headers: vec![],
                body: r#"{"detail":{"error":"invalid_document"}}"#.into(),
            }))
        }),
    );
    let batch = vec![
        msg(SinkDatabase::Platform, "usageEvents", "good"),
        msg(SinkDatabase::Platform, "usageEvents", "bad"),
    ];
    let d = block_on(handle_queue(&w.p, &w.cfg, &batch));
    assert_eq!(d.error, None);
    let mut acked = d.acked.clone();
    acked.sort();
    assert_eq!(acked, [0, 1]);
    let logs = crate::failure::console::take().join("\n");
    assert!(logs.contains("sink parked a document"));
    assert!(logs.contains("requestId=bad"));
}

#[test]
fn the_cron_probes_and_queues_the_health_reading() {
    let w = World::new();
    let report = block_on(handle_scheduled(&w.p, &w.cfg));
    assert!(report.all_paths_healthy);
    assert_eq!(w.collections(), ["serviceHealth"]);
    let doc = w.doc("serviceHealth").unwrap();
    assert_eq!(doc["check"], "inference-degradation");
    assert_eq!(doc["gatewayOk"], true);
    // The probe carries no user data and says platform scope at the gateway.
    assert!(w
        .completions()
        .iter()
        .all(|(m, _)| m["scope"] == "platform"));
}

#[test]
fn the_cron_says_so_loudly_when_a_path_is_down() {
    let w = World::new();
    w.settings().gateway_fails = true;
    let report = block_on(handle_scheduled(&w.p, &w.cfg));
    assert!(!report.all_paths_healthy);
    assert!(report.serviceable);
    assert!(crate::failure::console::take()
        .iter()
        .any(|l| l.contains("degradation probe: not all inference paths healthy")));
    assert_eq!(w.doc("serviceHealth").unwrap()["gatewayOk"], false);
}

#[test]
fn with_sink_enabled_false_nothing_reaches_the_queue_from_any_route_or_the_cron() {
    let mut cfg = env();
    cfg.sink_enabled = Some("false".into());
    let w = World::with(cfg);
    assert_eq!(w.chat(PLAIN_Q, json!({})).status, 200);
    assert_eq!(w.chat(PERSONAL_Q, json!({})).status, 200);
    w.context(PLAIN_Q, json!({}));
    block_on(handle_scheduled(&w.p, &w.cfg));
    assert!(w.sink().is_empty());
}

#[test]
fn a_null_body_status_is_known_so_the_entry_point_builds_it_bodiless() {
    for s in [101, 204, 205, 304] {
        assert!(Resp::is_null_body_status(s), "{s}");
    }
    for s in [200, 201, 400, 404, 409, 500, 503] {
        assert!(!Resp::is_null_body_status(s), "{s}");
    }
}

#[test]
fn json_is_written_as_json_stringify_with_two_spaces_and_compact_without() {
    let v = json!({ "a": [1, {}], "b": [], "c": { "d": null }, "e": "é\u{2028}\n" });
    let pretty = Resp::json(&v, 200).body;
    // JSON.stringify(v, null, 2), byte for byte.
    assert_eq!(
        pretty,
        "{\n  \"a\": [\n    1,\n    {}\n  ],\n  \"b\": [],\n  \"c\": {\n    \"d\": null\n  },\n  \"e\": \"é\u{2028}\\n\"\n}"
    );
    assert_eq!(
        Resp::json_compact(&v, 200).body,
        "{\"a\":[1,{}],\"b\":[],\"c\":{\"d\":null},\"e\":\"é\u{2028}\\n\"}"
    );
    let r: Value = serde_json::from_str(&pretty).unwrap();
    assert_eq!(r, v);
}

#[test]
fn health_is_any_method_and_says_rust() {
    let w = World::new();
    for m in ["GET", "POST", "HEAD"] {
        let r = w.send(crate::http::Req {
            method: m.into(),
            url: "https://gw.example/health".into(),
            ..Default::default()
        });
        assert_eq!(r.status, 200, "{m}");
        let b = r.body_json();
        assert_eq!(b["runtime"], "rust");
        assert_eq!(b["version"], env!("CARGO_PKG_VERSION"));
    }
}
