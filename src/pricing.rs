//! Port of `shamwari/gateway/src/pricing.ts` (the estimated cost of one
//! answer) and of `overQuota` and `billingPeriod` from `index.ts` (the monthly
//! token cap).
//!
//! The cost is an ESTIMATE, labelled as one everywhere it appears: it is not
//! billing-grade and nothing should invoice from it. The table is versioned
//! and sourced, so a price change is a reviewed diff with a new version.
//!
//! A model with no entry is `None`, never 0: "unpriced" and "free" must not
//! look alike in a report. Zero tokens in and out is also `None`: zero means
//! the count was not measured, and an unmeasured answer must not read as free.

use crate::auth::{console, Level};
use crate::config::Config;
use crate::keys::key_service;
use crate::platform::Platform;
use crate::types::AuthContext;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::OnceLock;

/// `shamwari/gateway/pricing.json`, verbatim. Keyed `provider/model`, the
/// same key `provenance` uses, so what is priced is what actually served.
pub const PRICING_JSON: &str = r#"{
  "version": "2026-10-29",
  "status": "estimate, not billing grade",
  "currency": "USD",
  "source": "https://developers.cloudflare.com/workers-ai/platform/pricing/",
  "retrieved": "2026-10-29",
  "notes": "Keyed provider/model, the same key provenance.ts uses, so what is priced is what actually served. Input is priced uncached (the upper bound; cached-input token counts are not recorded). Excludes the edge embedding of the question and any Cloudflare plan or neuron allowances. A model not listed here is unpriced and its cost is null, never zero.",
  "per_million_tokens": {
    "workers-ai/@cf/zai-org/glm-5.3-flash": { "input": 0.15, "output": 0.5 },
    "workers-ai/@cf/zai-org/glm-5.3": { "input": 1.4, "output": 4.4 }
  }
}
"#;

/// The table's `version`, carried on every usage event beside the estimate.
pub const PRICE_TABLE_VERSION: &str = "2026-10-29";

/// (input, output) USD per million tokens.
fn prices() -> &'static HashMap<String, (f64, f64)> {
    static PRICES: OnceLock<HashMap<String, (f64, f64)>> = OnceLock::new();
    PRICES.get_or_init(|| {
        let t: Value = serde_json::from_str(PRICING_JSON).unwrap_or(Value::Null);
        t["per_million_tokens"]
            .as_object()
            .map(|m| {
                m.iter()
                    .filter_map(|(k, v)| {
                        Some((k.clone(), (v["input"].as_f64()?, v["output"].as_f64()?)))
                    })
                    .collect()
            })
            .unwrap_or_default()
    })
}

/// `served_key` is `provider/model` as served, the key provenance uses.
pub fn estimate_cost_usd(
    served_key: Option<&str>,
    input_tokens: u64,
    output_tokens: u64,
) -> Option<f64> {
    let key = served_key.filter(|k| !k.is_empty())?;
    if input_tokens == 0 && output_tokens == 0 {
        return None;
    }
    // Same normalisation provenance applies, so what it accepts is what is priced.
    let (input, output) = *prices().get(&crate::auth::js_trim(key).to_lowercase())?;
    let usd = (input_tokens as f64 * input + output_tokens as f64 * output) / 1_000_000.0;
    // Fixed precision keeps the number stable across sums and out of float noise.
    Some((usd * 1e9).round() / 1e9)
}

/// The problems with a price table; empty means valid.
pub fn validate_pricing(t: &Value) -> Vec<String> {
    let mut errs = vec![];
    if !t["version"].as_str().is_some_and(|v| !v.is_empty()) {
        errs.push("version required".to_owned());
    }
    if t["currency"] != "USD" {
        errs.push("currency must be USD".to_owned());
    }
    if !t["source"]
        .as_str()
        .is_some_and(|s| s.starts_with("https://"))
    {
        errs.push("source required".to_owned());
    }
    if !t["retrieved"].is_string() {
        errs.push("retrieved date required".to_owned());
    }
    if let Some(table) = t["per_million_tokens"].as_object() {
        for (k, p) in table {
            let ok = |v: &Value| v.as_f64().is_some_and(|f| f.is_finite() && f >= 0.0);
            if !ok(&p["input"]) {
                errs.push(format!("{k}: input"));
            }
            if !ok(&p["output"]) {
                errs.push(format!("{k}: output"));
            }
            if !k.contains('/') {
                errs.push(format!("{k}: key must be provider/model"));
            }
            if *k != crate::auth::js_trim(k).to_lowercase() {
                errs.push(format!("{k}: key must be lowercase and trimmed"));
            }
        }
    }
    errs
}

/// The billing period (`YYYY-MM`, UTC) that `now_ms` falls in.
pub fn billing_period(now_ms: u64) -> String {
    // Civil date from days since the epoch (Howard Hinnant's algorithm).
    let days = (now_ms / 86_400_000) as i64;
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!("{year}-{month:02}")
}

/// Whether the caller has used more than its monthly cap this billing period.
///
/// Fails open: a rollup outage must not stop a paying customer. It also means
/// the cap is not enforced while it lasts, so it says so in the log (no
/// identifier, no content in the line).
pub async fn over_quota<P: Platform>(p: &P, cfg: &Config, auth: &AuthContext) -> bool {
    let period = billing_period(p.now_ms());
    match key_service(cfg)
        .usage_total(p, cfg, &auth.owner_entity_id, &period)
        .await
    {
        Ok(total) => total > auth.monthly_token_cap as f64,
        Err(e) => {
            console(
                Level::Warn,
                &format!("quota rollup failed, cap not enforced {e}"),
            );
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::take_logs;
    use crate::keys::USAGE_EVENT_FIELDS;
    use crate::testing::{block_on, FakePlatform};
    use serde_json::json;

    const FLASH: &str = "workers-ai/@cf/zai-org/glm-5.3-flash";
    const GLM: &str = "workers-ai/@cf/zai-org/glm-5.3";

    #[test]
    fn prices_a_served_model_from_the_versioned_table() {
        // 1M in + 1M out on glm-5.3-flash: 0.15 + 0.50
        assert_eq!(
            estimate_cost_usd(Some(FLASH), 1_000_000, 1_000_000),
            Some(0.65)
        );
        let c = estimate_cost_usd(Some(GLM), 1000, 500).unwrap();
        assert!((c - 0.0036).abs() < 1e-9);
    }

    #[test]
    fn an_unmeasured_answer_and_an_unpriced_model_are_none_never_zero() {
        assert_eq!(estimate_cost_usd(Some(FLASH), 0, 0), None);
        assert!(estimate_cost_usd(Some(FLASH), 0, 1).is_some());
        assert_eq!(
            estimate_cost_usd(Some("workers-ai/@cf/some/unlisted"), 1000, 1000),
            None
        );
        assert_eq!(estimate_cost_usd(None, 1000, 1000), None);
        assert_eq!(estimate_cost_usd(Some(""), 1, 1), None);
    }

    #[test]
    fn every_model_the_provenance_allowlist_serves_from_workers_ai_is_priced() {
        for k in [FLASH, GLM] {
            assert!(estimate_cost_usd(Some(k), 1, 1).is_some(), "{k}");
        }
    }

    #[test]
    fn matches_the_model_the_way_provenance_does_case_and_padding_do_not_hide_it() {
        let exact = estimate_cost_usd(Some(FLASH), 10, 10);
        assert_eq!(
            estimate_cost_usd(Some(" Workers-AI/@cf/zai-org/GLM-5.3-Flash "), 10, 10),
            exact
        );
    }

    #[test]
    fn the_table_is_valid_versioned_and_sourced() {
        let t: Value = serde_json::from_str(PRICING_JSON).unwrap();
        assert_eq!(validate_pricing(&t), Vec::<String>::new());
        assert_eq!(t["version"], PRICE_TABLE_VERSION);
        assert_eq!(prices().len(), 2);
    }

    #[test]
    fn a_malformed_table_is_caught() {
        let errs = validate_pricing(&json!({
            "version": "", "currency": "ZWL", "per_million_tokens": { "x": { "input": -1 } }
        }));
        assert!(errs.len() > 3, "{errs:?}");
        let errs = validate_pricing(&json!({
            "version": "v", "currency": "USD", "source": "https://s", "retrieved": "d",
            "per_million_tokens": { " A/b": { "input": 1, "output": 1 } }
        }));
        assert_eq!(errs, [" A/b: key must be lowercase and trimmed"]);
    }

    #[test]
    fn the_cost_fields_are_on_the_usage_event_allowlist() {
        assert!(USAGE_EVENT_FIELDS.contains(&"costUsdEstimate"));
        assert!(USAGE_EVENT_FIELDS.contains(&"priceTableVersion"));
    }

    #[test]
    fn the_billing_period_is_the_utc_year_and_month() {
        assert_eq!(billing_period(1_767_225_600_000), "2026-01"); // 2026-01-01T00:00:00Z
        assert_eq!(billing_period(1_767_225_599_999), "2025-12");
        assert_eq!(billing_period(1_709_164_800_000), "2024-02"); // 2024-02-29
        assert_eq!(billing_period(0), "1970-01");
    }

    fn ctx(cap: u64) -> AuthContext {
        AuthContext {
            api_key_id: "k1".into(),
            owner_entity_id: "e1".into(),
            owner_person_id: None,
            key_type: "shamwari".into(),
            tier: "community".into(),
            monthly_token_cap: cap,
            scopes: None,
            training_consent: None,
        }
    }

    fn core() -> Config {
        Config {
            core_url: Some("https://core.example".into()),
            shamwari_core_token: Some("t".into()),
            ..Default::default()
        }
    }

    #[test]
    fn over_quota_only_past_the_cap_for_this_billing_period() {
        let p = FakePlatform::new();
        p.on(
            "https://core.example/rollup",
            200,
            r#"{"total_tokens":100}"#,
        );
        assert!(!block_on(over_quota(&p, &core(), &ctx(100))));
        assert!(block_on(over_quota(&p, &core(), &ctx(99))));
        assert_eq!(
            p.requests.borrow()[0].url,
            "https://core.example/rollup?owner_entity_id=e1&billing_period=2026-01"
        );
    }

    #[test]
    fn over_quota_fails_open_on_a_rollup_outage_and_says_so() {
        take_logs();
        let p = FakePlatform::new();
        p.on("https://core.example/rollup", 500, "x");
        assert!(!block_on(over_quota(&p, &core(), &ctx(1))));
        let logs = take_logs();
        assert!(logs
            .iter()
            .any(|l| l.contains("quota rollup failed, cap not enforced")));
        assert!(!logs.iter().any(|l| l.contains("e1")));
    }

    #[test]
    fn over_quota_reads_the_servers_api_when_it_is_switched_on() {
        let p = FakePlatform::new();
        p.on(
            "https://servers.example/v1/shamwari/usage/rollup",
            200,
            r#"{"total_tokens":1000000000}"#,
        );
        let cfg = Config {
            servers_api_enabled: Some("true".into()),
            servers_api_url: Some("https://servers.example".into()),
            servers_api_token: Some("t".into()),
            ..core()
        };
        assert!(block_on(over_quota(&p, &cfg, &ctx(100_000))));
    }
}
