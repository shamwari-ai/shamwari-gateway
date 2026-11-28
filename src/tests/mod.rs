//! The handler tests: the TypeScript tests that go through `worker.fetch`,
//! run against the real router over the shared fake world ([`world`]).
mod channels;
mod chat;
mod cors;
mod ground_context;
mod handlers;
mod idempotency;
mod models;
mod responses;
mod router;
mod world;
