//! Port of `shamwari/gateway/src/types.ts`: the types every module shares.
//!
//! Field names serialise exactly as the TypeScript objects do (camelCase where
//! TypeScript used it), because the same JSON crosses the KV cache, the queue
//! and the wire, and both Workers read it until the cutover.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Tier {
    Economy,
    Standard,
    Premium,
}

/// Shamwari's three layers of intelligence. Data scopes, not deployment tiers.
///
/// - `personal`: the user's own pod data, which never leaves the device.
/// - `community`: anonymised platform data, which may reach Cloud.
/// - `platform`: base Mukoko knowledge, which may reach Cloud.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Scope {
    Personal,
    Community,
    Platform,
}

impl Scope {
    pub fn as_str(self) -> &'static str {
        match self {
            Scope::Personal => "personal",
            Scope::Community => "community",
            Scope::Platform => "platform",
        }
    }
}

/// Gates Mind training eligibility. Stamped at generation, never inferred.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LicenseClass {
    OpenWeight,
    Restricted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GroundChunk {
    pub content: String,
    pub heading: Option<String>,
    pub title: String,
    pub authority: String,
    pub resource_type: String,
    pub source_url: Option<String>,
    pub effective_from: Option<String>,
    pub scope: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AuthContext {
    pub api_key_id: String,
    pub owner_entity_id: String,
    pub owner_person_id: Option<String>,
    pub key_type: String,
    pub tier: String,
    pub monthly_token_cap: u64,
    /// What this key may ask for. Absent (an older cached context, or the Core
    /// path, which carries none) means community and platform only: personal
    /// is never implied. It can only narrow a request, never widen what the
    /// scope gate allows.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scopes: Option<Vec<Scope>>,
    /// The caller agreed that their conversations may become Shamwari Mind
    /// training material. Only an explicit `true` counts: absent means nobody
    /// asked, and storing a conversation is not consent to train on it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub training_consent: Option<bool>,
}

impl AuthContext {
    pub fn consented_to_training(&self) -> bool {
        self.training_consent == Some(true)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SinkDatabase {
    Shamwari,
    Platform,
}

/// One message on the `shamwari-sink` queue.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SinkMessage {
    pub database: SinkDatabase,
    pub collection: String,
    pub doc: Map<String, Value>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn an_auth_context_reads_the_json_the_typescript_worker_caches() {
        let cached = json!({
            "apiKeyId": "k1", "ownerEntityId": "e1", "ownerPersonId": null,
            "keyType": "customer", "tier": "standard", "monthlyTokenCap": 1000,
            "scopes": ["community", "platform"]
        });
        let ctx: AuthContext = serde_json::from_value(cached.clone()).unwrap();
        assert_eq!(ctx.scopes, Some(vec![Scope::Community, Scope::Platform]));
        assert!(!ctx.consented_to_training());
        assert_eq!(serde_json::to_value(&ctx).unwrap(), cached);
    }

    #[test]
    fn only_an_explicit_true_is_consent_to_train() {
        let base = json!({"apiKeyId":"k","ownerEntityId":"e","ownerPersonId":"p","keyType":"workos","tier":"economy","monthlyTokenCap":1});
        for (v, want) in [(json!(true), true), (json!(false), false)] {
            let mut j = base.clone();
            j["trainingConsent"] = v;
            let ctx: AuthContext = serde_json::from_value(j).unwrap();
            assert_eq!(ctx.consented_to_training(), want);
        }
        let ctx: AuthContext = serde_json::from_value(base).unwrap();
        assert!(!ctx.consented_to_training());
    }

    #[test]
    fn scope_and_license_class_spell_as_typescript_does() {
        assert_eq!(
            serde_json::to_value(Scope::Personal).unwrap(),
            json!("personal")
        );
        assert_eq!(
            serde_json::to_value(LicenseClass::OpenWeight).unwrap(),
            json!("open_weight")
        );
        assert_eq!(
            serde_json::to_value(SinkDatabase::Shamwari).unwrap(),
            json!("shamwari")
        );
    }
}
