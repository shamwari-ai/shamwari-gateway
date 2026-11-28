# The Rust port

The gateway that serves `api.shamwari.ai` is being rebuilt in Rust
(owner, 2026-11-28: "anything backend is Rust"). The TypeScript Worker in
[`shamwari-ai/shamwari/gateway`](https://github.com/shamwari-ai/shamwari/tree/main/gateway)
keeps serving traffic until every route here passes parity, then the
cutover moves the route, the cron and the queue consumer to this Worker
in one change.

## How it is built

- `workers-rs`, compiled to `wasm32-unknown-unknown` and bundled by
  `worker-build` (`scripts/build-worker.sh`), as in
  [`shamwari-ai/github-app`](https://github.com/shamwari-ai/github-app).
- Every rule is plain Rust over [`Platform`](../src/platform.rs) (HTTP,
  Workers AI, KV, the queue, the Secrets Store, the clock), so it runs under
  `cargo test` on the host. Only `src/worker_entry.rs` touches the runtime.
- Tests use the one fake, [`src/testing.rs`](../src/testing.rs).
- Each module names the TypeScript file it ports. A route whose port has
  not landed answers `501 {"error":"not_implemented"}`.

## Until the cutover

This Worker deploys as `shamwari-gateway-rs` on workers.dev only:

|                           | TypeScript `shamwari-gateway` | Rust `shamwari-gateway-rs` |
| ------------------------- | ----------------------------- | -------------------------- |
| `api.shamwari.ai`         | yes                           | no                         |
| Weekly probe cron         | yes                           | no                         |
| `shamwari-sink` consumer  | yes                           | no                         |
| Writes to `shamwari-sink` | yes                           | no (`SINK_ENABLED=false`)  |

## Modules

| Rust                                                            | TypeScript                                                      | Port             |
| --------------------------------------------------------------- | --------------------------------------------------------------- | ---------------- |
| `router.rs`                                                     | `index.ts` (dispatch), `GET /health`                            | done             |
| `auth.rs`, `keys.rs`, `workos.rs`                               | `auth.ts`, `keys.ts`, `workos.ts`                               | A                |
| `cors.rs`, `idempotency.rs`, `pricing.rs`                       | `cors.ts`, `idempotency.ts`, `pricing.ts`                       | A                |
| `scope.rs`, `personal.rs`, `lang.rs`                            | `scope.ts`, `personal.ts`, `lang.ts`                            | B                |
| `ground.rs`, `ground_store.rs`, `own_sites.rs`, `extractive.rs` | `ground.ts`, `ground-store.ts`, `own-sites.ts`, `extractive.ts` | B                |
| `models.rs`, `model_router.rs`, `gateway.rs`                    | `models.ts`, `router.ts`, `gateway.ts`                          | C                |
| `envelope.rs`, `failure.rs`, `provenance.rs`, `skills.rs`       | `envelope.ts`, `failure.ts`, `provenance.ts`, `skills.ts`       | C                |
| `core.rs`, `sink.rs`, `probe.rs`                                | `core.ts`, `sink.ts`, `probe.ts`                                | C                |
| `responses.rs`, `channels/`                                     | `responses.ts`, `channels/`                                     | C                |
| `gateway.rs` (`admit`, the chat and ground handlers)            | `index.ts`                                                      | after A, B and C |

## The two rules

The TypeScript tests that guard them port with the code and must pass
unchanged in meaning:

1. Personal content never reaches third-party inference (`scope.rs`,
   `personal.rs`).
2. Only open-weight output with explicit consent may train Shamwari Mind
   (`provenance.rs`, `sink.rs`).

## Parity

Before the cutover every route is called on both Workers with the same
requests and the answers compared (status, the headers that matter, the
body with volatile fields normalised). Intended differences are listed with
their reason, as `mzizi-api`'s `scripts/parity.mjs` does.
