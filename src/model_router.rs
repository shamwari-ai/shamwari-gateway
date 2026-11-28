//! Port of `shamwari/gateway/src/router.ts`: model routing, tiers and the
//! reach ceilings, parameterised by `data/routing-policy.json`.
//!
//! Both live tiers run on Workers AI, billed through Cloudflare: there is no
//! provider API key and no custom provider. The dynamic route
//! (`dynamic/<name>` on the AI Gateway) chooses between them from the `tier`
//! metadata; what is recorded here is the intent, and [`crate::provenance`]
//! stamps what served.
//!
//! Provider identity and licence class stay here rather than in the policy
//! file. They are provenance-bearing: Core rejects restricted rows from the
//! Mind training path on the strength of what [`targets`] stamps, so it must
//! not be editable without a code review.

use crate::config::Config;
use crate::lang::js;
use crate::types::{LicenseClass, Message, Tier};
use serde_json::Value;
use std::fmt;
use std::sync::OnceLock;

/// The shipped policy, bundled into the Worker as `router.ts` bundles it.
pub const ROUTING_POLICY_JSON: &str = include_str!("../data/routing-policy.json");

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Target {
    pub tier: Tier,
    pub provider: String,
    pub model: String,
    pub license_class: LicenseClass,
}

/// `Record<Tier, Target>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Targets {
    pub economy: Target,
    pub standard: Target,
    pub premium: Target,
}

impl Targets {
    pub fn get(&self, tier: Tier) -> &Target {
        match tier {
            Tier::Economy => &self.economy,
            Tier::Standard => &self.standard,
            Tier::Premium => &self.premium,
        }
    }
}

pub fn tier_name(t: Tier) -> &'static str {
    match t {
        Tier::Economy => "economy",
        Tier::Standard => "standard",
        Tier::Premium => "premium",
    }
}

fn parse_tier(v: &Value) -> Option<Tier> {
    match v.as_str()? {
        "economy" => Some(Tier::Economy),
        "standard" => Some(Tier::Standard),
        "premium" => Some(Tier::Premium),
        _ => None,
    }
}

/// The model each tier asks for. Model ids come from `ECONOMY_MODEL` and
/// `STANDARD_MODEL`, never from code or the policy file.
pub fn targets(cfg: &Config) -> Targets {
    Targets {
        economy: Target {
            tier: Tier::Economy,
            provider: "workers-ai".into(),
            model: cfg.economy_model.clone().unwrap_or_default(),
            license_class: LicenseClass::OpenWeight,
        },
        standard: Target {
            tier: Tier::Standard,
            provider: "workers-ai".into(),
            model: cfg.standard_model.clone().unwrap_or_default(),
            license_class: LicenseClass::OpenWeight,
        },
        // Premium is deliberately unconfigured. When Claude or GPT are added,
        // the licence class MUST stay restricted: their terms bar using
        // outputs to train competing models, and Core rejects restricted rows
        // from the Mind training path.
        premium: Target {
            tier: Tier::Premium,
            provider: "unconfigured".into(),
            model: String::new(),
            license_class: LicenseClass::Restricted,
        },
    }
}

/// The most a generation may ask for, with or without a reach. A reach lowers
/// it and cannot raise it, so the policy validator caps `max_output_tokens`
/// here: a larger number would validate but never be honoured.
pub const DEFAULT_MAX_OUTPUT_TOKENS: u32 = 2048;

/// How a caller reaches Shamwari: r0 SMS/USSD, r1 chat app, r2 Lite PWA. R3
/// (on-device Mind) is deliberately absent: it does not use this gateway for
/// personal data, and a cloud request cannot claim it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Reach {
    R0,
    R1,
    R2,
}

pub const REACHES: [Reach; 3] = [Reach::R0, Reach::R1, Reach::R2];

impl Reach {
    pub fn as_str(self) -> &'static str {
        match self {
            Reach::R0 => "r0",
            Reach::R1 => "r1",
            Reach::R2 => "r2",
        }
    }

    pub fn parse(s: &str) -> Option<Reach> {
        REACHES.into_iter().find(|r| r.as_str() == s)
    }
}

/// `isReach`: a string naming exactly one of r0, r1, r2.
pub fn is_reach(v: &Value) -> Option<Reach> {
    v.as_str().and_then(Reach::parse)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReachLimits {
    /// The highest model tier this reach may be routed to. Never premium.
    pub max_tier: Tier,
    pub max_output_tokens: u32,
    pub extractive_max_chars: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Escalation {
    pub tier: Tier,
    pub max_total_chars: usize,
    pub max_messages: usize,
    /// By language, in file order.
    pub keywords: Vec<(String, Vec<String>)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoutingPolicy {
    pub version: u32,
    pub default_tier: Tier,
    /// r0, r1, r2, in that order.
    pub reach: Vec<(Reach, ReachLimits)>,
    /// In file order.
    pub model_aliases: Vec<(String, Tier)>,
    pub escalation: Escalation,
}

impl RoutingPolicy {
    pub fn reach(&self, r: Reach) -> &ReachLimits {
        &self
            .reach
            .iter()
            .find(|(k, _)| *k == r)
            .expect("a validated policy names every reach")
            .1
    }

    pub fn alias(&self, name: &str) -> Option<Tier> {
        self.model_aliases
            .iter()
            .find(|(a, _)| a == name)
            .map(|(_, t)| *t)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyError(pub String);

impl fmt::Display for PolicyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "routing-policy.json: {}", self.0)
    }
}

impl std::error::Error for PolicyError {}

fn err<T>(detail: String) -> Result<T, PolicyError> {
    Err(PolicyError(detail))
}

/// `JSON.stringify(v)` in a message, with `undefined` for a missing value.
fn shown(v: Option<&Value>) -> String {
    v.map_or_else(|| "undefined".to_owned(), Value::to_string)
}

/// `Object.entries` for anything `typeof` calls an object: an object's
/// entries, or an array's, keyed by index. `None` for null and primitives.
fn entries(v: Option<&Value>) -> Option<Vec<(String, &Value)>> {
    match v? {
        Value::Object(m) => Some(m.iter().map(|(k, v)| (k.clone(), v)).collect()),
        Value::Array(a) => Some(
            a.iter()
                .enumerate()
                .map(|(i, v)| (i.to_string(), v))
                .collect(),
        ),
        _ => None,
    }
}

fn field<'a>(v: &'a Value, k: &str) -> Option<&'a Value> {
    v.as_object().and_then(|o| o.get(k))
}

fn positive_int(v: Option<&Value>, field: &str, max: Option<u64>) -> Result<u64, PolicyError> {
    let n = v.and_then(Value::as_f64);
    let n = match n {
        Some(n) if n.fract() == 0.0 && n > 0.0 && n.is_finite() => n as u64,
        _ => {
            return err(format!(
                "{field} must be a positive integer, got {}",
                shown(v)
            ))
        }
    };
    if let Some(max) = max {
        if n > max {
            return err(format!("{field} must be at most {max}, got {n}"));
        }
    }
    Ok(n)
}

/// A reach block must name exactly r0, r1 and r2, each with sane ceilings.
fn validate_reach(input: Option<&Value>) -> Result<Vec<(Reach, ReachLimits)>, PolicyError> {
    let given = match input {
        Some(Value::Object(_)) => entries(input).unwrap_or_default(),
        _ => return err("reach must be an object keyed by r0, r1 and r2".into()),
    };
    let mut keys: Vec<String> = given.iter().map(|(k, _)| k.clone()).collect();
    keys.sort();
    let want: Vec<&str> = REACHES.iter().map(|r| r.as_str()).collect();
    if keys.join(",") != want.join(",") {
        return err(format!(
            "reach must name exactly {}, got {}",
            want.join(", "),
            keys.join(", ")
        ));
    }
    let mut out = Vec::new();
    for r in REACHES {
        let l = given
            .iter()
            .find(|(k, _)| k == r.as_str())
            .map(|(_, v)| *v)
            .filter(|v| v.is_object() || v.is_array());
        let Some(lim) = l else {
            return err(format!("reach.{} must be an object", r.as_str()));
        };
        // Premium is opt-in, outside the ladder. A reach ceiling of premium
        // would let the channel alone choose a model with different terms and cost.
        let max_tier = match field(lim, "max_tier").and_then(Value::as_str) {
            Some("economy") => Tier::Economy,
            Some("standard") => Tier::Standard,
            _ => {
                return err(format!(
                    "reach.{}.max_tier must be economy or standard, got {}",
                    r.as_str(),
                    shown(field(lim, "max_tier"))
                ))
            }
        };
        let max_output_tokens = positive_int(
            field(lim, "max_output_tokens"),
            &format!("reach.{}.max_output_tokens", r.as_str()),
            Some(DEFAULT_MAX_OUTPUT_TOKENS as u64),
        )? as u32;
        let extractive_max_chars = positive_int(
            field(lim, "extractive_max_chars"),
            &format!("reach.{}.extractive_max_chars", r.as_str()),
            Some(20_000),
        )? as usize;
        out.push((
            r,
            ReachLimits {
                max_tier,
                max_output_tokens,
                extractive_max_chars,
            },
        ));
    }
    Ok(out)
}

/// Validated at load, not at first use. A policy that parses but means
/// something different (an unknown tier name, a threshold of zero, a keyword
/// with a capital letter that the lowercased match can never hit) would
/// otherwise route every request to the default tier and look like it was
/// working.
pub fn validate_policy(input: &Value) -> Result<RoutingPolicy, PolicyError> {
    if !(input.is_object() || input.is_array()) {
        return err("not an object".into());
    }
    let p = |k: &str| field(input, k);

    if p("version").and_then(Value::as_f64) != Some(1.0) {
        return err(format!(
            "unsupported version {}, expected 1",
            shown(p("version"))
        ));
    }
    let Some(default_tier) = p("default_tier").and_then(parse_tier) else {
        return err(format!("unknown default_tier {}", shown(p("default_tier"))));
    };

    let reach = validate_reach(p("reach"))?;

    let Some(aliases_in) = entries(p("model_aliases")) else {
        return err("model_aliases must be an object".into());
    };
    let mut model_aliases = Vec::new();
    for (alias, tier) in aliases_in {
        let Some(t) = parse_tier(tier) else {
            return err(format!(
                "model_aliases[\"{alias}\"] names unknown tier {}",
                tier
            ));
        };
        model_aliases.push((alias, t));
    }

    let esc = match p("escalation") {
        Some(e @ (Value::Object(_) | Value::Array(_))) => e,
        _ => return err("escalation must be an object".into()),
    };
    let Some(esc_tier) = field(esc, "tier").and_then(parse_tier) else {
        return err(format!(
            "escalation.tier names unknown tier {}",
            shown(field(esc, "tier"))
        ));
    };

    let Some(keywords_in) = entries(field(esc, "keywords")) else {
        return err("escalation.keywords must be an object keyed by language".into());
    };
    let mut keywords = Vec::new();
    for (lang, list) in keywords_in {
        let Some(list) = list.as_array() else {
            return err(format!("escalation.keywords.{lang} must be an array"));
        };
        let mut words = Vec::new();
        for kw in list {
            let Some(kw) = kw.as_str().filter(|k| !k.is_empty()) else {
                return err(format!(
                    "escalation.keywords.{lang} contains a non-string or empty entry"
                ));
            };
            // Matching lowercases the request text, so an uppercase keyword
            // can never fire. Reject it rather than shipping a dead rule.
            if kw != kw.to_lowercase() {
                return err(format!(
                    "escalation.keywords.{lang} entry \"{kw}\" must be lowercase"
                ));
            }
            words.push(kw.to_owned());
        }
        keywords.push((lang, words));
    }

    let max_total_chars = positive_int(
        field(esc, "max_total_chars"),
        "escalation.max_total_chars",
        None,
    )? as usize;
    let max_messages =
        positive_int(field(esc, "max_messages"), "escalation.max_messages", None)? as usize;

    Ok(RoutingPolicy {
        version: 1,
        default_tier,
        reach,
        model_aliases,
        escalation: Escalation {
            tier: esc_tier,
            max_total_chars,
            max_messages,
            keywords,
        },
    })
}

/// The shipped policy. Validated on first use; a bad file is a panic there,
/// and the test that validates the shipped file catches it in CI first.
pub fn policy() -> &'static RoutingPolicy {
    static P: OnceLock<RoutingPolicy> = OnceLock::new();
    P.get_or_init(|| {
        let raw: Value =
            serde_json::from_str(ROUTING_POLICY_JSON).expect("routing-policy.json parses");
        validate_policy(&raw).unwrap_or_else(|e| panic!("{e}"))
    })
}

/// Heuristic v0, parameterised by routing-policy.json. Deliberately not an
/// LLM classifier: paying for a model call to decide which model to call
/// doubles latency and cost on every request.
///
/// Economy handles the bulk; that is where the margin comes from, since AI
/// Gateway passes provider pricing through at cost.
pub fn route(messages: &[Message], requested: Option<&str>, cfg: &Config) -> Target {
    route_with(policy(), messages, requested, cfg)
}

/// [`route`] against a given policy.
pub fn route_with(
    policy: &RoutingPolicy,
    messages: &[Message],
    requested: Option<&str>,
    cfg: &Config,
) -> Target {
    let t = targets(cfg);

    if let Some(tier) = requested.and_then(|r| policy.alias(r)) {
        return t.get(tier).clone();
    }

    let esc = &policy.escalation;
    let text = messages
        .iter()
        .map(|m| m.content.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    if js::len(&text) > esc.max_total_chars || messages.len() > esc.max_messages {
        return t.get(esc.tier).clone();
    }

    let lower = text.to_lowercase();
    if esc
        .keywords
        .iter()
        .flat_map(|(_, list)| list.iter())
        .any(|s| lower.contains(s.as_str()))
    {
        return t.get(esc.tier).clone();
    }

    t.get(policy.default_tier).clone()
}

fn rank(t: Tier) -> u8 {
    match t {
        Tier::Economy => 0,
        Tier::Standard => 1,
        Tier::Premium => 2,
    }
}

/// What [`apply_reach`] returns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reached {
    pub target: Target,
    pub clamped: bool,
    /// The tier routed to before the clamp, so clamping can be measured.
    pub routed_tier: Tier,
}

/// Clamp a routed target to the reach's ceiling. A ceiling, never a floor: a
/// target already at or below it is untouched. The result is a tier from
/// [`targets`], so provenance (licence class, provider) still comes from code.
pub fn apply_reach(target: Target, reach: Option<Reach>, cfg: &Config) -> Reached {
    let routed_tier = target.tier;
    let Some(reach) = reach else {
        return Reached {
            target,
            clamped: false,
            routed_tier,
        };
    };
    let ceiling = policy().reach(reach).max_tier;
    if rank(routed_tier) <= rank(ceiling) {
        return Reached {
            target,
            clamped: false,
            routed_tier,
        };
    }
    Reached {
        target: targets(cfg).get(ceiling).clone(),
        clamped: true,
        routed_tier,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provenance::{open_weight_models, resolve_license_class};
    use crate::types::Role;
    use serde_json::json;
    use std::collections::{HashMap, HashSet};

    fn cfg() -> Config {
        Config {
            economy_model: Some("glm-5.3-flash".into()),
            standard_model: Some("glm-5.3".into()),
            ..Default::default()
        }
    }

    fn user(content: &str) -> Message {
        Message {
            role: Role::User,
            content: content.into(),
        }
    }

    fn shipped() -> Value {
        serde_json::from_str(ROUTING_POLICY_JSON).unwrap()
    }

    fn rejects(v: &Value, needle: &str) {
        match validate_policy(v) {
            Ok(_) => panic!("accepted, wanted an error containing {needle:?}"),
            Err(e) => assert!(
                e.to_string().contains(needle),
                "{e} does not contain {needle:?}"
            ),
        }
    }

    // router.test.ts. Rule 2: only open_weight provenance may train Shamwari
    // Mind. The licence class is stamped here, at generation time, and never
    // inferred later.

    #[test]
    fn marks_both_live_tiers_open_weight_so_they_stay_mind_trainable() {
        let t = targets(&cfg());
        assert_eq!(t.economy.license_class, LicenseClass::OpenWeight);
        assert_eq!(t.standard.license_class, LicenseClass::OpenWeight);
    }

    #[test]
    fn keeps_premium_restricted_and_unconfigured() {
        let t = targets(&cfg());
        assert_eq!(t.premium.license_class, LicenseClass::Restricted);
        assert_eq!(t.premium.provider, "unconfigured");
    }

    #[test]
    fn honours_an_explicit_model_request() {
        assert_eq!(
            route(&[user("hi")], Some("shamwari-standard"), &cfg()).tier,
            Tier::Standard
        );
        assert_eq!(
            route(
                &[user("calculate my PAYE")],
                Some("shamwari-economy"),
                &cfg()
            )
            .tier,
            Tier::Economy
        );
    }

    #[test]
    fn defaults_to_economy_for_short_simple_turns() {
        assert_eq!(
            route(&[user("Mhoro, urikuita sei?")], None, &cfg()).tier,
            Tier::Economy
        );
    }

    #[test]
    fn escalates_on_hard_task_keywords_in_english_and_in_shona() {
        assert_eq!(
            route(&[user("Please calculate my PAYE for August")], None, &cfg()).tier,
            Tier::Standard
        );
        assert_eq!(
            route(&[user("Tsanangura mutero wangu")], None, &cfg()).tier,
            Tier::Standard
        );
    }

    #[test]
    fn escalates_on_long_input_or_long_conversations() {
        assert_eq!(
            route(&[user(&"a".repeat(6001))], None, &cfg()).tier,
            Tier::Standard
        );
        let many: Vec<Message> = (0..13).map(|_| user("hi")).collect();
        assert_eq!(route(&many, None, &cfg()).tier, Tier::Standard);
    }

    #[test]
    fn reads_model_ids_from_env_rather_than_hardcoding_them() {
        assert_eq!(route(&[user("hi")], None, &cfg()).model, "glm-5.3-flash");
        assert_eq!(
            route(&[user("hi")], Some("shamwari-standard"), &cfg()).model,
            "glm-5.3"
        );
    }

    // policy.test.ts: the shipped file, and every rejection path, so the
    // validator itself cannot rot into a no-op.

    #[test]
    fn the_shipped_policy_validates() {
        validate_policy(&shipped()).unwrap();
    }

    #[test]
    fn the_shipped_policy_is_what_the_router_actually_loaded() {
        let p = policy();
        assert_eq!(p.default_tier, Tier::Economy);
        assert_eq!(p.escalation.tier, Tier::Standard);
        assert_eq!(p.escalation.max_total_chars, 6000);
        assert_eq!(p.escalation.max_messages, 12);
    }

    #[test]
    fn maps_both_public_model_names() {
        assert_eq!(
            policy().model_aliases,
            vec![
                ("shamwari-economy".to_owned(), Tier::Economy),
                ("shamwari-standard".to_owned(), Tier::Standard),
            ]
        );
    }

    #[test]
    fn carries_no_provenance_fields_those_belong_in_code() {
        fn keys(v: &Value, out: &mut HashSet<String>) {
            match v {
                Value::Array(a) => a.iter().for_each(|c| keys(c, out)),
                Value::Object(o) => {
                    for (k, c) in o {
                        out.insert(k.to_lowercase());
                        keys(c, out);
                    }
                }
                _ => {}
            }
        }
        let mut present = HashSet::new();
        keys(&shipped(), &mut present);
        // Key names, not raw text: the notes field explains why these are absent.
        for forbidden in [
            "licenseclass",
            "license_class",
            "provider",
            "direct_url",
            "directurl",
            "api_key",
            "apikey",
        ] {
            assert!(!present.contains(forbidden), "{forbidden}");
        }
    }

    #[test]
    fn keeps_a_keyword_bucket_per_language_so_gaps_are_visible() {
        let kw: HashMap<&str, &Vec<String>> = policy()
            .escalation
            .keywords
            .iter()
            .map(|(l, w)| (l.as_str(), w))
            .collect();
        let mut langs: Vec<&str> = kw.keys().copied().collect();
        langs.sort();
        assert_eq!(langs, ["en", "nd", "sn"]);
        // Ndebele is empty today. A known gap, not an oversight: this fails
        // loudly if someone fills it and forgets the escalation tests.
        assert!(kw["nd"].is_empty());
        assert!(kw["sn"].contains(&"tsanangura".to_owned()));
    }

    #[test]
    fn rejects_a_non_object() {
        for bad in [json!(null), json!("policy"), json!(42), json!([])] {
            assert!(validate_policy(&bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn rejects_an_unsupported_version() {
        let mut p = shipped();
        p["version"] = json!(2);
        rejects(&p, "unsupported version");
        p.as_object_mut().unwrap().remove("version");
        rejects(&p, "unsupported version undefined");
    }

    #[test]
    fn rejects_a_tier_name_that_does_not_exist_in_code() {
        let mut p = shipped();
        p["default_tier"] = json!("cheap");
        rejects(&p, "unknown default_tier");
        let mut p = shipped();
        p["escalation"]["tier"] = json!("deluxe");
        rejects(&p, "escalation.tier names unknown tier");
    }

    #[test]
    fn rejects_an_alias_pointing_at_a_tier_that_does_not_exist() {
        let mut p = shipped();
        p["model_aliases"] = json!({ "shamwari-turbo": "turbo" });
        rejects(&p, "names unknown tier");
    }

    #[test]
    fn rejects_a_threshold_that_is_zero_negative_or_fractional() {
        for bad in [json!(0), json!(-1), json!(1.5), json!("6000"), json!(null)] {
            let mut p = shipped();
            p["escalation"]["max_total_chars"] = bad;
            rejects(&p, "max_total_chars must be a positive integer");
        }
    }

    #[test]
    fn rejects_an_uppercase_keyword_which_the_lowercased_match_could_never_fire_on() {
        let mut p = shipped();
        p["escalation"]["keywords"] = json!({ "en": ["Calculate"] });
        rejects(&p, "must be lowercase");
    }

    #[test]
    fn rejects_an_empty_or_non_string_keyword() {
        for bad in [json!(""), json!(42), json!(null)] {
            let mut p = shipped();
            p["escalation"]["keywords"] = json!({ "en": [bad] });
            rejects(&p, "non-string or empty entry");
        }
    }

    #[test]
    fn rejects_keywords_that_are_not_grouped_into_arrays() {
        let mut p = shipped();
        p["escalation"]["keywords"] = json!({ "en": "calculate" });
        rejects(&p, "must be an array");
    }

    #[test]
    fn the_reach_ceilings_ship_for_r0_r1_and_r2_and_never_name_premium() {
        let p = policy();
        let names: Vec<&str> = p.reach.iter().map(|(r, _)| r.as_str()).collect();
        assert_eq!(names, ["r0", "r1", "r2"]);
        assert_eq!(p.reach(Reach::R0).max_tier, Tier::Economy);
        for (_, r) in &p.reach {
            assert!(matches!(r.max_tier, Tier::Economy | Tier::Standard));
        }
    }

    #[test]
    fn the_reach_ceilings_get_looser_as_the_channel_gets_richer() {
        let p = policy();
        assert!(p.reach(Reach::R0).max_output_tokens < p.reach(Reach::R1).max_output_tokens);
        assert!(p.reach(Reach::R1).max_output_tokens < p.reach(Reach::R2).max_output_tokens);
        assert!(p.reach(Reach::R0).extractive_max_chars < p.reach(Reach::R1).extractive_max_chars);
    }

    #[test]
    fn the_reach_ceilings_are_rejected_when_missing_extra_or_pointing_at_premium() {
        let mut p = shipped();
        p.as_object_mut().unwrap().remove("reach");
        rejects(&p, "reach must be an object");

        let mut extra = shipped();
        extra["reach"]["r3"] = json!({ "max_tier": "standard" });
        rejects(&extra, "reach must name exactly");

        let mut prem = shipped();
        prem["reach"]["r1"]["max_tier"] = json!("premium");
        rejects(&prem, "max_tier must be economy or standard");
    }

    #[test]
    fn the_reach_ceilings_are_rejected_when_a_cap_is_not_a_sane_positive_integer() {
        for bad in [
            json!(0),
            json!(-5),
            json!(1.5),
            json!("200"),
            json!(null),
            json!(100000),
        ] {
            let mut p = shipped();
            p["reach"]["r0"]["max_output_tokens"] = bad;
            rejects(&p, "max_output_tokens");
        }
        let mut p = shipped();
        p["reach"]["r2"]["extractive_max_chars"] = json!(0);
        rejects(&p, "extractive_max_chars");
    }

    // reach.test.ts: reach is a ceiling. It clamps the model tier down and
    // caps generation, and it never raises anything.

    fn reach_cfg() -> Config {
        Config {
            economy_model: Some("e".into()),
            standard_model: Some("s".into()),
            ..Default::default()
        }
    }

    #[test]
    fn apply_reach_clamps_a_standard_route_down_to_economy_at_r0_and_says_so() {
        let c = reach_cfg();
        let r = apply_reach(targets(&c).standard, Some(Reach::R0), &c);
        assert_eq!(r.target.tier, Tier::Economy);
        assert!(r.clamped);
    }

    #[test]
    fn apply_reach_is_a_ceiling_never_a_floor() {
        let c = reach_cfg();
        let r = apply_reach(targets(&c).economy, Some(Reach::R2), &c);
        assert!(!r.clamped);
        assert_eq!(r.target.tier, Tier::Economy);
        assert!(!apply_reach(targets(&c).standard, Some(Reach::R1), &c).clamped);
    }

    #[test]
    fn apply_reach_leaves_a_request_with_no_reach_untouched() {
        let c = reach_cfg();
        let t = targets(&c).standard;
        assert_eq!(
            apply_reach(t.clone(), None, &c),
            Reached {
                target: t,
                clamped: false,
                routed_tier: Tier::Standard
            }
        );
    }

    #[test]
    fn apply_reach_can_never_be_the_way_to_premium() {
        let c = reach_cfg();
        let r = apply_reach(targets(&c).premium, Some(Reach::R2), &c);
        assert_eq!(r.target.tier, Tier::Standard);
        assert!(r.clamped);
    }

    #[test]
    fn apply_reach_takes_provenance_from_code() {
        let c = reach_cfg();
        assert_eq!(
            apply_reach(targets(&c).standard, Some(Reach::R0), &c)
                .target
                .license_class,
            LicenseClass::OpenWeight
        );
    }

    #[test]
    fn apply_reach_keeps_the_tier_it_was_routed_to_before_the_clamp() {
        let c = reach_cfg();
        let r = apply_reach(targets(&c).standard, Some(Reach::R0), &c);
        assert_eq!(r.routed_tier, Tier::Standard);
        assert_eq!(r.target.tier, Tier::Economy);
    }

    #[test]
    fn does_not_accept_a_cap_the_code_would_never_honour() {
        let mut p = shipped();
        p["reach"]["r2"]["max_output_tokens"] = json!(DEFAULT_MAX_OUTPUT_TOKENS + 1);
        rejects(&p, "max_output_tokens must be at most");
    }

    #[test]
    fn is_reach_names_exactly_r0_r1_and_r2() {
        assert_eq!(is_reach(&json!("r1")), Some(Reach::R1));
        for bad in [
            json!("r3"),
            json!("R0"),
            json!("sms"),
            json!(5),
            json!(null),
        ] {
            assert_eq!(is_reach(&bad), None, "{bad}");
        }
    }

    // dynamic-route.test.ts: dynamic-route.json is posted to Cloudflare, not
    // bundled into the Worker, so nothing else would catch a hand-edit that
    // breaks it: graph integrity, and the rule-2 invariant that every path
    // through the route stays open-weight.

    struct El {
        id: String,
        kind: String,
        outputs: Vec<(String, String)>,
        props: Value,
    }

    fn elements_of(v: &Value) -> Vec<El> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|e| El {
                id: e["id"].as_str().unwrap().to_owned(),
                kind: e["type"].as_str().unwrap().to_owned(),
                outputs: e["outputs"]
                    .as_object()
                    .map(|o| {
                        o.iter()
                            .map(|(k, v)| (k.clone(), v["elementId"].as_str().unwrap().to_owned()))
                            .collect()
                    })
                    .unwrap_or_default(),
                props: e.get("properties").cloned().unwrap_or(Value::Null),
            })
            .collect()
    }

    fn route_json() -> Value {
        serde_json::from_str(include_str!("../data/dynamic-route.json")).unwrap()
    }

    fn route_elements() -> Vec<El> {
        elements_of(&route_json()["elements"])
    }

    fn of_type<'a>(els: &'a [El], t: &str) -> Vec<&'a El> {
        els.iter().filter(|e| e.kind == t).collect()
    }

    fn edges(els: &[El]) -> Vec<(String, String, String)> {
        els.iter()
            .flat_map(|e| {
                e.outputs
                    .iter()
                    .map(|(l, to)| (e.id.clone(), l.clone(), to.clone()))
            })
            .collect()
    }

    fn output_labels(e: &El) -> Vec<String> {
        let mut l: Vec<String> = e.outputs.iter().map(|(k, _)| k.clone()).collect();
        l.sort();
        l
    }

    #[test]
    fn the_dynamic_route_has_exactly_one_start_and_one_end() {
        let els = route_elements();
        assert_eq!(of_type(&els, "start").len(), 1);
        assert_eq!(of_type(&els, "end").len(), 1);
    }

    #[test]
    fn the_dynamic_route_has_no_duplicate_element_ids() {
        let els = route_elements();
        let ids: HashSet<&str> = els.iter().map(|e| e.id.as_str()).collect();
        assert_eq!(ids.len(), els.len());
    }

    #[test]
    fn the_dynamic_route_has_no_dangling_element_id() {
        let els = route_elements();
        let ids: HashSet<&str> = els.iter().map(|e| e.id.as_str()).collect();
        let dangling: Vec<_> = edges(&els)
            .into_iter()
            .filter(|(_, _, to)| !ids.contains(to.as_str()))
            .collect();
        assert!(dangling.is_empty(), "{dangling:?}");
    }

    #[test]
    fn the_dynamic_route_reaches_every_element_from_start() {
        let els = route_elements();
        let es = edges(&els);
        let mut seen = HashSet::new();
        let mut stack = vec![of_type(&els, "start")[0].id.clone()];
        while let Some(n) = stack.pop() {
            if !seen.insert(n.clone()) {
                continue;
            }
            for (from, _, to) in &es {
                if *from == n {
                    stack.push(to.clone());
                }
            }
        }
        let unreached: Vec<_> = els
            .iter()
            .filter(|e| !seen.contains(&e.id))
            .map(|e| &e.id)
            .collect();
        assert!(unreached.is_empty(), "{unreached:?}");
    }

    #[test]
    fn the_dynamic_route_gives_every_model_and_rate_node_a_success_and_a_fallback() {
        for e in route_elements()
            .iter()
            .filter(|e| e.kind == "model" || e.kind == "rate")
        {
            assert_eq!(output_labels(e), ["fallback", "success"], "{}", e.id);
        }
    }

    #[test]
    fn the_dynamic_route_gives_the_conditional_both_branches() {
        for e in of_type(&route_elements(), "conditional") {
            assert_eq!(output_labels(e), ["false", "true"]);
        }
    }

    #[test]
    fn rule_2_the_dynamic_route_routes_only_to_open_weight_providers() {
        // 'workers-ai' is what the dashboard serialises for Workers AI; 'zai'
        // would be a custom provider this account does not have.
        let open: HashSet<&str> = ["zai", "workers-ai"].into();
        let els = route_elements();
        let models = of_type(&els, "model");
        assert!(!models.is_empty());
        for e in models {
            assert!(
                open.contains(e.props["provider"].as_str().unwrap()),
                "{}",
                e.id
            );
        }
    }

    #[test]
    fn rule_2_the_dynamic_route_names_no_restricted_provider_anywhere_in_the_file() {
        let flat = route_json().to_string().to_lowercase();
        for restricted in ["anthropic", "claude", "openai", "gpt-", "azure"] {
            assert!(!flat.contains(restricted), "{restricted}");
        }
    }

    #[test]
    fn rule_2_every_model_node_names_a_provider_and_a_model() {
        for e in of_type(&route_elements(), "model") {
            assert!(e.props["provider"].is_string(), "{}", e.id);
            assert!(e.props["model"].is_string(), "{}", e.id);
        }
    }

    #[test]
    fn rule_2_every_model_node_resolves_to_open_weight() {
        for e in of_type(&route_elements(), "model") {
            let (p, m) = (
                e.props["provider"].as_str().unwrap(),
                e.props["model"].as_str().unwrap(),
            );
            assert_eq!(
                resolve_license_class(p, m),
                LicenseClass::OpenWeight,
                "{p}/{m}"
            );
        }
    }

    #[test]
    fn rule_2_every_model_node_is_a_pair_in_the_allowlist_itself() {
        let allowed: HashSet<&str> = open_weight_models().iter().copied().collect();
        for e in of_type(&route_elements(), "model") {
            let pair = format!(
                "{}/{}",
                e.props["provider"].as_str().unwrap(),
                e.props["model"].as_str().unwrap()
            )
            .to_lowercase();
            assert!(allowed.contains(pair.as_str()), "{pair}");
        }
    }

    #[test]
    fn the_route_offers_a_node_per_live_tier_plus_the_last_resort_fallback() {
        assert_eq!(of_type(&route_elements(), "model").len(), 3);
    }

    #[test]
    fn the_route_caps_spend_before_choosing_a_model() {
        let els = route_elements();
        let rate = of_type(&els, "rate");
        let rate = rate.first().expect("a rate node");
        assert_eq!(rate.props["limitType"], "cost");
        // Reached directly from start, so the cap cannot be bypassed by a branch.
        let start = &of_type(&els, "start")[0].id;
        let first = edges(&els)
            .into_iter()
            .find(|(f, _, _)| f == start)
            .map(|(_, _, t)| t);
        assert_eq!(first.as_deref(), Some(rate.id.as_str()));
    }

    #[test]
    fn the_route_degrades_to_economy_when_the_cap_is_hit_rather_than_failing() {
        let els = route_elements();
        let rate = of_type(&els, "rate")[0];
        let fallback = rate
            .outputs
            .iter()
            .find(|(l, _)| l == "fallback")
            .map(|(_, t)| t)
            .unwrap();
        let target = els.iter().find(|e| &e.id == fallback).unwrap();
        assert_eq!(target.props["provider"], "workers-ai");
        assert_eq!(target.props["model"], "@cf/zai-org/glm-5.3");
    }

    fn phase1() -> Vec<El> {
        elements_of(
            &serde_json::from_str(include_str!("../data/dynamic-route-phase1.json")).unwrap(),
        )
    }

    #[test]
    fn phase1_is_a_closed_graph_with_one_start_and_one_end() {
        let els = phase1();
        assert_eq!(of_type(&els, "start").len(), 1);
        assert_eq!(of_type(&els, "end").len(), 1);
        let ids: HashSet<&str> = els.iter().map(|e| e.id.as_str()).collect();
        assert!(edges(&els)
            .iter()
            .all(|(_, _, to)| ids.contains(to.as_str())));
    }

    #[test]
    fn phase1_needs_no_custom_provider_every_node_is_on_workers_ai() {
        for e in of_type(&phase1(), "model") {
            assert_eq!(e.props["provider"], "workers-ai");
        }
    }

    #[test]
    fn phase1_gives_every_node_a_real_timeout() {
        // 0 is the editor's default: a node then never times out, or times
        // out at once. Either way the chain below it is dead.
        for e in of_type(&phase1(), "model") {
            assert!(e.props["timeout"].as_f64().unwrap() > 0.0, "{}", e.id);
        }
    }

    #[test]
    fn phase1_retries_the_first_node_before_falling_back() {
        let els = phase1();
        let first = els.iter().find(|e| e.id == "economy").unwrap();
        assert!(first.props["retries"].as_f64().unwrap() > 0.0);
    }

    #[test]
    fn phase1_uses_cf_model_ids_which_is_what_workers_ai_accepts() {
        let re = regex_lite::Regex::new(r"^@cf/[a-z0-9._-]+/[a-zA-Z0-9._-]+$").unwrap();
        for e in of_type(&phase1(), "model") {
            assert!(re.is_match(e.props["model"].as_str().unwrap()), "{}", e.id);
        }
    }

    #[test]
    fn phase1_routes_to_no_code_specialised_model() {
        let re = regex_lite::Regex::new(r"coder|-code").unwrap();
        for e in of_type(&phase1(), "model") {
            assert!(!re.is_match(e.props["model"].as_str().unwrap()), "{}", e.id);
        }
    }
}
