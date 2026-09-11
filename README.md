# shamwari-gateway

The edge API gateway. TypeScript on Cloudflare Workers.

> **This branch is scaffolding.** `main` is deliberately empty so the
> extraction can `git subtree split -P gateway` out of
> [`shamwari`](https://github.com/shamwari-ai/shamwari) and push history
> straight in without a merge or a force. Merge this branch **after** that
> import lands.

## What it serves

Three routes, plus a queue consumer and a cron handler:

| Route                       |           |
| --------------------------- | --------- |
| `GET /health`               | liveness  |
| `POST /v1/chat/completions` | inference |
| `POST /v1/ground/context`   | grounding |

None of them render a page — which is why this repo uses **Hono** and not
Astro. Hono is Workers-native, adds no meaningful cold-start cost for three
routes, and replaces the hand-rolled `if (url.pathname === ...)` chain with
a route table that survives voice and image endpoints being added. Hono
wraps only the `fetch` export; the `queue` and `scheduled` exports are
untouched by it.

**Do not add Astro here.** See "What not to do" in the repo-split plan.

## Rule 1 lives here

`src/scope.ts` is the fail-fast half of the scope gate. The authoritative
half is `core/main.py::resolve_scope` in
[`shamwari-core`](https://github.com/shamwari-ai/shamwari-core).

The duplication is **deliberate and must not be removed.** A shared package
would collapse two independent checks into one, and one check is exactly
what the design guards against. What is needed instead is a contract test
vendored into both repos, asserting identical `(scope, destination)`
behaviour — see the repo-split plan.

## Before the extraction

Per the suggested order, the Hono migration happens **in the monorepo
first**, so a routing mistake is caught by the existing suite rather than
surfacing as this repo's first bug.

## Related

- [`shamwari`](https://github.com/shamwari-ai/shamwari) — umbrella
- [Org standards](https://github.com/shamwari-ai/.github/blob/main/ORG_STANDARDS.md)
