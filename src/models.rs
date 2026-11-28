//! Port of `shamwari/gateway/src/models.ts`: `GET /v1/models`, the model ids a
//! caller may put in `model`, in the shape of OpenAI's model list, so
//! OpenAI-compatible clients can discover them.
//!
//! Derived, never listed by hand. The ids are the aliases the router honours
//! (`routing-policy.json` `model_aliases`), and a tier is offered only when
//! [`targets`] has a model configured for it and that model is open weight.
//! So the list cannot drift from routing, and premium (unconfigured, and
//! restricted by Rule 2) can never appear here, even if an alias for it is
//! added to the policy file.
//!
//! `created` is left out on purpose: these are aliases with no such date, and
//! a made-up one would be worse than none. `owned_by` is Shamwari because the
//! alias is ours.
//!
//! The route authenticates like the rest of `/v1` (not metered, not
//! quota-checked): [`crate::gateway::handle_models`] authenticates, then
//! answers [`list_response`].

use crate::config::Config;
use crate::http::Resp;
use crate::model_router::{policy, targets, RoutingPolicy};
use crate::types::{LicenseClass, Tier};
use serde_json::{json, Value};

/// The tiers that may be advertised: configured, open weight, and never premium.
fn offered(cfg: &Config, tier: Tier) -> bool {
    if tier == Tier::Premium {
        return false;
    }
    let t = targets(cfg);
    let t = t.get(tier);
    t.provider != "unconfigured"
        && !t.model.trim().is_empty()
        && t.license_class == LicenseClass::OpenWeight
}

pub fn public_model_ids(cfg: &Config) -> Vec<String> {
    public_model_ids_in(policy(), cfg)
}

/// [`public_model_ids`] for a given policy.
pub fn public_model_ids_in(policy: &RoutingPolicy, cfg: &Config) -> Vec<String> {
    let mut ids: Vec<String> = policy
        .model_aliases
        .iter()
        .filter(|(_, tier)| offered(cfg, *tier))
        .map(|(alias, _)| alias.clone())
        .collect();
    ids.sort();
    ids
}

/// `{ object: "list", data: [{ id, object: "model", owned_by: "shamwari" }] }`.
pub fn model_list(cfg: &Config) -> Value {
    let data: Vec<Value> = public_model_ids(cfg)
        .into_iter()
        .map(|id| json!({ "id": id, "object": "model", "owned_by": "shamwari" }))
        .collect();
    json!({ "object": "list", "data": data })
}

/// The 200 an authenticated caller gets. The wiring calls this only after
/// authentication passes; a failed one answers as the chat endpoint does.
pub fn list_response(cfg: &Config) -> Resp {
    Resp::json(&model_list(cfg), 200)
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_router::route;
    use crate::types::{Message, Role};

    fn cfg() -> Config {
        Config {
            economy_model: Some("@cf/zai-org/glm-5.3-flash".into()),
            standard_model: Some("@cf/zai-org/glm-5.3".into()),
            ..Default::default()
        }
    }

    #[test]
    fn is_exactly_the_policy_aliases_for_configured_open_weight_tiers() {
        let c = cfg();
        let t = targets(&c);
        let mut expected: Vec<String> = policy()
            .model_aliases
            .iter()
            .filter(|(_, tier)| {
                *tier != Tier::Premium
                    && t.get(*tier).license_class == LicenseClass::OpenWeight
                    && !t.get(*tier).model.is_empty()
            })
            .map(|(a, _)| a.clone())
            .collect();
        expected.sort();
        assert_eq!(public_model_ids(&c), expected);
        // Today's policy, pinned so a change to what is offered is a visible diff.
        assert_eq!(
            public_model_ids(&c),
            ["shamwari-economy", "shamwari-standard"]
        );
    }

    #[test]
    fn lists_only_ids_the_router_pins_to_their_own_tier() {
        let c = cfg();
        let hello = [Message {
            role: Role::User,
            content: "hello".into(),
        }];
        for id in public_model_ids(&c) {
            let target = route(&hello, Some(&id), &c);
            assert_eq!(Some(target.tier), policy().alias(&id), "{id}");
            assert_eq!(target.license_class, LicenseClass::OpenWeight, "{id}");
            assert_ne!(target.provider, "unconfigured", "{id}");
        }
    }

    #[test]
    fn never_lists_premium_or_a_restricted_tier_even_if_an_alias_for_it_is_added() {
        let mut p = policy().clone();
        p.model_aliases
            .push(("shamwari-premium".into(), Tier::Premium));
        let c = cfg();
        assert_eq!(targets(&c).premium.license_class, LicenseClass::Restricted);
        let ids = public_model_ids_in(&p, &c);
        assert!(!ids.contains(&"shamwari-premium".to_owned()));
        for id in ids {
            assert_ne!(p.alias(&id), Some(Tier::Premium));
        }
    }

    #[test]
    fn drops_a_tier_whose_model_is_not_configured() {
        let c = Config {
            standard_model: Some(String::new()),
            ..cfg()
        };
        assert_eq!(public_model_ids(&c), ["shamwari-economy"]);
        let c = Config {
            economy_model: Some(" ".into()),
            standard_model: Some(String::new()),
            ..cfg()
        };
        assert!(public_model_ids(&c).is_empty());
        assert!(public_model_ids(&Config::default()).is_empty());
    }

    #[test]
    fn has_the_openai_list_shape() {
        let list = model_list(&cfg());
        assert_eq!(list["object"], "list");
        for m in list["data"].as_array().unwrap() {
            assert_eq!(
                *m,
                json!({ "id": m["id"], "object": "model", "owned_by": "shamwari" })
            );
        }
    }

    #[test]
    fn the_response_is_json_with_the_list() {
        let r = list_response(&cfg());
        assert_eq!(r.status, 200);
        assert_eq!(r.header("content-type"), Some("application/json"));
        assert_eq!(r.body_json(), model_list(&cfg()));
    }
}
