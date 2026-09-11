# Shamwari Gateway

> The Shamwari Cloud edge gateway — routed inference, grounded in Zimbabwean law and policy. TypeScript on Cloudflare Workers.

[![Lint](https://github.com/shamwari-ai/shamwari-gateway/actions/workflows/lint.yml/badge.svg)](https://github.com/shamwari-ai/shamwari-gateway/actions/workflows/lint.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
![TypeScript](https://img.shields.io/badge/TypeScript-5-3178C6?style=flat-square&logo=typescript&logoColor=white)
![Cloudflare Workers](https://img.shields.io/badge/Cloudflare_Workers-F38020?style=flat-square&logo=cloudflare&logoColor=white)

**Default branch:** `scaffold` | **Code still lives in:** [`shamwari/gateway`](https://github.com/shamwari-ai/shamwari/tree/main/gateway) | **Docs:** [docs.shamwari.ai/architecture](https://docs.shamwari.ai/architecture)

---

## What it is

This repository will own the Shamwari edge gateway. It does not own it yet.

`scaffold` — the default branch, and the one you are reading — holds a
licence, CI wiring and this file. `main` is deliberately empty so the
extraction can `git subtree split -P gateway` out of
[`shamwari`](https://github.com/shamwari-ai/shamwari) and push history
straight in without a merge or a force. Merge `scaffold` **after** that
import lands.

The Worker itself is written, typechecked and tested in the monorepo, and is
**not deployed**.

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

## Hono is the plan, not the present

None of these routes render a page, which is why the plan is **Hono** and not
Astro. Hono is Workers-native, adds no meaningful cold-start cost for three
routes, and replaces the hand-rolled `if (url.pathname === ...)` chain with a
route table that survives voice and image endpoints being added. It wraps only
the `fetch` export; `queue` and `scheduled` are untouched by it.

That migration has **not happened yet** — `gateway/src/index.ts` in the
monorepo is still the pathname chain, and `hono` is not in its
`package.json`. Per the suggested order in `docs/repo-split.md` the migration
happens in the monorepo first, so a routing mistake is caught by the existing
suite rather than surfacing as this repo's first bug.

**Do not add Astro here.** See "What not to do" in the repo-split plan.

## Rule 1 lives here, twice

`src/scope.ts` is the fail-fast half of the scope gate: personal-scope content
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
