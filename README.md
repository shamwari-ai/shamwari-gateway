# Shamwari Gateway

> The Shamwari Cloud edge gateway — routed inference, grounded in Zimbabwean law and policy. Rust on Cloudflare Workers (workers-rs).

[![Lint](https://github.com/shamwari-ai/shamwari-gateway/actions/workflows/lint.yml/badge.svg)](https://github.com/shamwari-ai/shamwari-gateway/actions/workflows/lint.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
![Rust](https://img.shields.io/badge/Rust-workers--rs-000000?style=flat-square&logo=rust&logoColor=white)
![Cloudflare Workers](https://img.shields.io/badge/Cloudflare_Workers-F38020?style=flat-square&logo=cloudflare&logoColor=white)

**Default branch:** `main` | **Code still lives in:** [`shamwari/gateway`](https://github.com/shamwari-ai/shamwari/tree/main/gateway) | **Docs:** [docs.shamwari.ai/architecture](https://docs.shamwari.ai/architecture)

---

## What it is

This repository is the Shamwari edge gateway, in Rust. It is being ported from
the TypeScript Worker in
[`shamwari/gateway`](https://github.com/shamwari-ai/shamwari/tree/main/gateway),
which keeps serving `api.shamwari.ai` until every route here passes parity.
See [docs/rust-port.md](docs/rust-port.md) for the plan, the module table and
what each route needs before the cutover.

```bash
cargo test                      # every rule, on the host
cargo clippy --all-targets -- -D warnings
bash scripts/build-worker.sh    # the wasm32 Worker, as Cloudflare builds it
npx wrangler dev                # run it locally
```

## What it serves

One `export default` with three handlers: `fetch`, `queue` and `scheduled`.
The fetch half answers three routes and nothing else — anything unmatched is
a `404`.

| Route                       |                           |
| --------------------------- | ------------------------- |
| `GET /health`               | liveness                  |
| `POST /v1/chat/completions` | inference                 |
| `POST /v1/ground/context`   | grounding, retrieval only |

The queue handler drains the `shamwari-sink` queue into Core; the cron runs
weekly (`0 6 * * 1`).

## Rust, not Hono

The earlier plan to move the TypeScript Worker to Hono is superseded: the
gateway is Rust (owner, 2026-11-28: anything backend is Rust). The route table
lives in `src/router.rs`, and the `queue` and `scheduled` handlers in
`src/worker_entry.rs`.

## Rule 1 lives here, twice

`src/scope.rs` (ported from `src/scope.ts`) is the fail-fast half of the scope gate: personal-scope content
never reaches a third-party inference provider, and a request that tries gets
a `409` without burning a Core round trip. The authoritative half is
`main.py::resolve_scope` in
[`shamwari-core`](https://github.com/shamwari-ai/shamwari-core), which the
Worker cannot override.

The gate distinguishes `cloud_inference` from `retrieval_only`, because a gate
that only knows the scope cannot tell "send this to a provider" from "hand
this back for the device to answer". That distinction is the fix for a real
leak, not a refinement.

The duplication is **deliberate and must not be removed.** A shared package
would collapse two independent checks into one, and one check is exactly what
the design guards against. What is needed instead is a contract test vendored
into both repos, asserting identical `(scope, destination)` behaviour.

## Ecosystem

- [`shamwari`](https://github.com/shamwari-ai/shamwari) — the umbrella;
  `CLAUDE.md` has the line-level reasoning behind both rules
- [`shamwari-core`](https://github.com/shamwari-ai/shamwari-core) — the
  service this Worker authenticates and retrieves against
- [`shamwari-sandbox`](https://github.com/shamwari-ai/shamwari-sandbox) — the
  execution host this gateway will route personal-scope artefacts to
- [Org standards](https://github.com/shamwari-ai/.github/blob/main/ORG_STANDARDS.md)

## Contributing

See the org's
[CONTRIBUTING.md](https://github.com/shamwari-ai/.github/blob/main/CONTRIBUTING.md),
[SECURITY.md](https://github.com/shamwari-ai/.github/blob/main/SECURITY.md) and
[CODE_OF_CONDUCT.md](https://github.com/shamwari-ai/.github/blob/main/CODE_OF_CONDUCT.md).

## Licence

Licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
— see `LICENSE` and `NOTICE`.

© Bundu Foundation. Shamwari is Bundu Foundation IP, sold commercially under
Nyuchi Africa.
