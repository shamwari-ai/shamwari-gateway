//! Port of `shamwari/gateway/src/provenance.ts`, and of `conversationTrainable`
//! from `index.ts`: RULE 2, resolved from what actually served the request.
//!
//! Only `open_weight` provenance may train Shamwari Mind. Anthropic's and
//! OpenAI's terms bar using their outputs to train a competing model. Z.ai's
//! GLM-5.3 licence explicitly grants the right to "fine-tune... and create
//! derivative works" (read directly from the LICENSE file at
//! <https://huggingface.co/zai-org/GLM-5.3/raw/main/LICENSE> before trusting
//! this comment). GLM-5.3-Flash ships under plain MIT.
//!
//! `model_router` stamps a licence class on the tier the Worker *asks for*.
//! That is the right place for intent and it is not sufficient on its own,
//! because the model that answers is not always the model that was asked for:
//!
//! - `infer` degrades to Workers AI when the AI Gateway is unreachable, and
//!   that path reports a provider of `workers-ai`.
//! - An AI Gateway **dynamic route** can pick a different model entirely.
//!   Editing a route is a dashboard action: no code review, no deploy.
//!
//! So provenance is resolved here, from the `provider/model` pair the response
//! says served the request, against an allowlist that lives in code, for the
//! same reason the licence class is not in `routing-policy.json`: provenance
//! must not be editable without review.
//!
//! Unknown pairs resolve to `restricted`. An unlisted model is a model whose
//! terms nobody has read. A lost training row is recoverable; a corpus that
//! has to be thrown away is not.

use crate::failure::console;
use crate::types::{LicenseClass, Scope};

/// The allowlist. Every entry is lowercase, `provider/model`.
const OPEN_WEIGHT: &[&str] = &[
    // Z.ai GLM-5.3. Bespoke licence, explicitly permits fine-tuning and
    // derivative works; the only gate is a security review for
    // Model-as-a-Service operators above $10B revenue over any 12 months.
    // Read the LICENSE file before adding a new variant here.
    "zai/glm-5.3",
    // Z.ai GLM-5.3-Flash. Plain MIT.
    "zai/glm-5.3-flash",
    // Workers AI last-resort fallback. Same GLM-5.3-Flash weights,
    // Cloudflare-hosted.
    "workers-ai/@cf/zai-org/glm-5.3-flash",
    // The standard tier served from Workers AI, which is where
    // dynamic-route.json runs its GLM-5.3 node (Cloudflare-billed, no BYOK).
    // Same GLM-5.3 weights and licence as `zai/glm-5.3`; only the host differs.
    "workers-ai/@cf/zai-org/glm-5.3",
];

/// The Workers AI fallback model, named once, so usage events can say which
/// model answered when the gateway was down, and the allowlist has something
/// real to key on.
pub const WORKERS_AI_FALLBACK_MODEL: &str = "@cf/zai-org/glm-5.3";

/// Normalised so casing or padding from a provider header cannot miss.
fn key(provider: &str, model: &str) -> String {
    format!("{}/{}", provider.trim(), model.trim()).to_lowercase()
}

/// Resolve provenance from the model that actually generated the text.
///
/// Fails closed. Callers must use this rather than `Target::license_class`
/// when stamping a row that the Mind training pipeline could read.
pub fn resolve_license_class(provider: &str, model: &str) -> LicenseClass {
    let k = key(provider, model);
    if OPEN_WEIGHT.contains(&k.as_str()) {
        return LicenseClass::OpenWeight;
    }
    // Visible on purpose: an unlisted model quietly losing its training rows
    // is the failure this whole module exists to make loud.
    console::warn(&format!(
        "provenance: unlisted model, stamping restricted {k}"
    ));
    LicenseClass::Restricted
}

/// Exposed for the test that pins the allowlist.
pub fn open_weight_models() -> &'static [&'static str] {
    OPEN_WEIGHT
}

/// The licence class as it is written on a row.
pub fn license_class_name(c: LicenseClass) -> &'static str {
    match c {
        LicenseClass::OpenWeight => "open_weight",
        LicenseClass::Restricted => "restricted",
    }
}

/// Whether a stored conversation may ever reach the Mind training pipeline
/// (`conversationTrainable` in `index.ts`).
///
/// Every term is a way to say no; there is no way to say yes except being an
/// open_weight answer to a caller who was asked and said yes (`consented`).
/// Storing a conversation is not consent to train on it, so a caller with no
/// recorded consent is not trainable, whatever kind of key it holds. Channel
/// traffic and signed-in people (WorkOS) stay excluded even with a flag, until
/// the owner decides how their consent is asked for.
///
/// `looks_personal`: the system judged that the question needs the person's
/// own information (#81). Their text is then probably about them, and it is
/// exactly the traffic that must not become Mind training material, whatever
/// scope it travelled in.
pub fn conversation_trainable(
    scope: Scope,
    license_class: LicenseClass,
    via_channel: bool,
    key_type: &str,
    looks_personal: bool,
    consented: bool,
) -> bool {
    consented
        && scope != Scope::Personal
        && license_class == LicenseClass::OpenWeight
        && !via_channel
        && key_type != "workos"
        && !looks_personal
}

#[cfg(test)]
mod tests {
    use super::*;
    use LicenseClass::{OpenWeight, Restricted};

    // Rule 2: only open_weight provenance may train Shamwari Mind. The router
    // tests cover the class stamped on the tier that was *requested*; these
    // cover the one resolved from the model that actually *served*, which is
    // the only one a training row may carry.

    #[test]
    fn admits_the_configured_glm_5_3_and_glm_5_3_flash_models() {
        assert_eq!(resolve_license_class("zai", "glm-5.3"), OpenWeight);
        assert_eq!(resolve_license_class("zai", "glm-5.3-flash"), OpenWeight);
    }

    #[test]
    fn admits_the_workers_ai_fallback_under_its_real_model_id() {
        assert_eq!(
            resolve_license_class("workers-ai", WORKERS_AI_FALLBACK_MODEL),
            OpenWeight
        );
    }

    #[test]
    fn admits_glm_5_3_when_workers_ai_serves_it_and_only_under_its_real_id() {
        assert_eq!(
            resolve_license_class("workers-ai", "@cf/zai-org/glm-5.3"),
            OpenWeight
        );
        // The neighbours on the same host stay unlisted until someone reads their terms.
        assert_eq!(
            resolve_license_class("workers-ai", "@cf/zai-org/glm-5.2"),
            Restricted
        );
        assert_eq!(resolve_license_class("workers-ai", "glm-5.3"), Restricted);
        // Header values arrive with whatever casing and padding the provider sent.
        assert_eq!(
            resolve_license_class(" Workers-AI ", " @CF/zai-org/GLM-5.3 "),
            OpenWeight
        );
    }

    #[test]
    fn fails_closed_for_a_model_nobody_has_read_the_terms_of() {
        assert_eq!(resolve_license_class("zai", "glm-9-unreleased"), Restricted);
        assert_eq!(
            resolve_license_class("zai", "glm-some-new-thing"),
            Restricted
        );
    }

    #[test]
    fn fails_closed_for_the_providers_whose_terms_bar_training_outright() {
        assert_eq!(
            resolve_license_class("anthropic", "claude-opus-5"),
            Restricted
        );
        assert_eq!(resolve_license_class("openai", "gpt-5"), Restricted);
    }

    #[test]
    fn warns_when_it_stamps_restricted_so_a_silent_corpus_loss_is_visible() {
        console::take();
        resolve_license_class("openai", "gpt-5");
        let logged = console::take();
        assert_eq!(logged.len(), 1);
        assert!(logged[0].starts_with("warn: provenance: unlisted model"));
        resolve_license_class("zai", "glm-5.3");
        assert!(console::take().is_empty());
    }

    #[test]
    fn is_insensitive_to_casing_and_padding_from_a_provider_header() {
        assert_eq!(resolve_license_class(" Zai ", " GLM-5.3 "), OpenWeight);
    }

    #[test]
    fn never_admits_a_bare_provider_or_a_bare_model() {
        assert_eq!(resolve_license_class("zai", ""), Restricted);
        assert_eq!(resolve_license_class("", "glm-5.3"), Restricted);
    }

    #[test]
    fn keeps_every_allowlist_entry_lowercase_or_lookups_silently_miss() {
        for entry in open_weight_models() {
            assert_eq!(*entry, entry.to_lowercase());
        }
    }

    // conversationTrainable (scope-decision.test.ts, workos.test.ts).

    #[test]
    fn conversation_trainable_says_no_when_the_question_looked_personal() {
        assert!(!conversation_trainable(
            Scope::Platform,
            OpenWeight,
            false,
            "shamwari",
            true,
            true
        ));
        assert!(conversation_trainable(
            Scope::Platform,
            OpenWeight,
            false,
            "shamwari",
            false,
            true
        ));
    }

    #[test]
    fn a_caller_with_no_recorded_consent_is_never_trainable_whatever_else_is_true() {
        // Every other term says yes; only consent is missing.
        assert!(!conversation_trainable(
            Scope::Platform,
            OpenWeight,
            false,
            "shamwari",
            false,
            false
        ));
        assert!(!conversation_trainable(
            Scope::Community,
            OpenWeight,
            false,
            "standard",
            false,
            false
        ));
        assert!(conversation_trainable(
            Scope::Platform,
            OpenWeight,
            false,
            "shamwari",
            false,
            true
        ));
    }

    #[test]
    fn keeps_a_workos_caller_out_of_training_even_for_an_open_weight_community_answer() {
        // The baseline, consented.
        assert!(conversation_trainable(
            Scope::Community,
            OpenWeight,
            false,
            "shamwari",
            false,
            true
        ));
        assert!(!conversation_trainable(
            Scope::Community,
            OpenWeight,
            false,
            "workos",
            false,
            true
        ));
        assert!(!conversation_trainable(
            Scope::Platform,
            OpenWeight,
            false,
            "workos",
            false,
            true
        ));
    }

    #[test]
    fn restricted_output_personal_scope_and_channel_traffic_are_never_trainable() {
        assert!(!conversation_trainable(
            Scope::Platform,
            Restricted,
            false,
            "shamwari",
            false,
            true
        ));
        assert!(!conversation_trainable(
            Scope::Personal,
            OpenWeight,
            false,
            "shamwari",
            false,
            true
        ));
        assert!(!conversation_trainable(
            Scope::Platform,
            OpenWeight,
            true,
            "shamwari",
            false,
            true
        ));
    }
}
