//! The Shamwari edge gateway (`api.shamwari.ai`): routed inference grounded in
//! Zimbabwean law and policy, on Cloudflare Workers, in Rust.
//!
//! This is the Rust port of `shamwari-ai/shamwari/gateway` (TypeScript). Each
//! module names the TypeScript file it ports. Everything that decides
//! behaviour is plain Rust over [`platform::Platform`], so it runs under
//! `cargo test` on the host; only `worker_entry` (wasm32 alone) touches the
//! Workers runtime.
//!
//! The two rules hold here as they do in TypeScript, and the tests that guard
//! them port with the code:
//!
//! 1. Personal content never reaches third-party inference ([`scope`]).
//! 2. Only open-weight output with explicit consent may train Shamwari Mind
//!    ([`provenance`]).

pub mod ai_gateway;
pub mod auth;
pub mod channels;
pub mod config;
pub mod core;
pub mod cors;
pub mod envelope;
pub mod extractive;
pub mod failure;
pub mod gateway;
pub mod ground;
pub mod ground_store;
pub mod http;
pub mod idempotency;
pub mod keys;
pub mod lang;
pub mod model_router;
pub mod models;
pub mod own_sites;
pub mod personal;
pub mod platform;
pub mod pricing;
pub mod probe;
pub mod provenance;
pub mod responses;
pub mod router;
pub mod scope;
pub mod sink;
pub mod skills;
pub mod types;
pub mod workos;

#[cfg(test)]
pub mod testing;

#[cfg(target_arch = "wasm32")]
mod worker_entry;

#[cfg(test)]
mod tests;
