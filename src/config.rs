//! The Worker's `[vars]` and secrets, as the TypeScript gateway's `Env`
//! (`shamwari/gateway/src/types.ts`). Every value is a string or absent; each
//! module parses what it reads and fails closed on a malformed value.

#[derive(Debug, Clone, Default)]
pub struct Config {
    pub cf_account_id: Option<String>,
    pub cf_gateway_id: Option<String>,
    pub cf_route_name: Option<String>,
    pub core_url: Option<String>,
    /// Secret.
    pub shamwari_core_token: Option<String>,
    pub own_sites: Option<String>,
    pub nyuchi_api_url: Option<String>,
    pub servers_api_enabled: Option<String>,
    pub servers_api_url: Option<String>,
    /// Secret.
    pub servers_api_token: Option<String>,
    /// Secret.
    pub servers_api_internal_key: Option<String>,
    pub workos_client_id: Option<String>,
    pub workos_issuer: Option<String>,
    pub workos_audience: Option<String>,
    pub workos_jwks_url: Option<String>,
    pub workos_monthly_token_cap: Option<String>,
    pub workos_max_token_age_seconds: Option<String>,
    pub cors_origins: Option<String>,
    pub embedding_model: Option<String>,
    pub ground_top_k: Option<String>,
    pub auth_cache_ttl: Option<String>,
    pub economy_model: Option<String>,
    pub standard_model: Option<String>,
    pub surface: Option<String>,
    pub mind_available: Option<String>,
    /// Secret.
    pub channel_webhook_secret: Option<String>,
    /// Secret.
    pub channel_api_key: Option<String>,
    /// "false" keeps this Worker from writing to the `shamwari-sink` queue.
    /// Off until the cutover, so parity traffic never double-writes Core.
    pub sink_enabled: Option<String>,
}

impl Config {
    /// The value, with an empty string read as absent.
    pub fn get<'a>(&self, v: &'a Option<String>) -> Option<&'a str> {
        v.as_deref().filter(|s| !s.is_empty())
    }

    pub fn mind_available(&self) -> bool {
        self.mind_available.as_deref() == Some("true")
    }

    pub fn sink_enabled(&self) -> bool {
        self.sink_enabled.as_deref() != Some("false")
    }
}
