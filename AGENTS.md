# AGENTS.md — shamwari-gateway

Rules for AI agents and people working in this repository. They layer on the
org-wide guide, [nyuchi/.github AGENTS.md](https://github.com/nyuchi/.github/blob/main/AGENTS.md).
Where the two conflict, this file wins. Read `README.md` first.

## What this repo is

The planned home of the Shamwari edge gateway: routed inference grounded in
Zimbabwean law and policy, written in TypeScript for Cloudflare Workers. It
does not own the code yet. `main` holds a licence, CI wiring, lint config and
the README. The Worker is written and tested in
[`shamwari-ai/shamwari`](https://github.com/shamwari-ai/shamwari) under
`gateway/`, and code changes land there until the extraction into this repo is
done. The monorepo's
[`AGENTS.md`](https://github.com/shamwari-ai/shamwari/blob/main/AGENTS.md)
holds the project rules.

Once imported, the Worker has one `export default` with `fetch`, `queue` and
`scheduled` handlers. The fetch half answers `GET /health`,
`POST /v1/chat/completions` and `POST /v1/ground/context`, and returns `404`
for anything else (README).

## The two rules

Do not weaken either without the owner's written sign-off.

1. **Personal-scope content never reaches a third-party inference provider.**
   `src/scope.ts` is the fail-fast half of the gate. The authoritative half is
   `main.py::resolve_scope` in
   [`shamwari-core`](https://github.com/shamwari-ai/shamwari-core), which the
   Worker cannot override. The duplication is deliberate: do not factor the two
   into a shared package. The gate separates `cloud_inference` from
   `retrieval_only`; keep that distinction.
2. **Only open-weight output may train Shamwari Mind, and storing is not
   consent.**

## Checks

CI (`.github/workflows/ci.yml`) calls the org's reusable workflows:
`reusable-ci-node-npm.yml` with `typecheck-script: typecheck` and
`test-script: test`, `reusable-gitleaks.yml`, and `reusable-codeql.yml` for
`javascript-typescript`. The PR title is linted by `pr-title-lint.yml`. When
the code is imported, the commands are `npm ci`, `npm run typecheck` and
`npm test`. There is no `package.json` on `main` yet, so there is nothing to
run locally today.

## Conventions

- **Hono is the plan, not the present.** The migration happens in the monorepo
  first, then moves here. It wraps only the `fetch` export. **Do not add
  Astro.**
- **Conventional Commits** on every commit and PR title; the PR title becomes
  the squash-commit subject.
- **Trunk-based.** `main` is the only target; open every pull request against
  it and merge once CI is green.
- **Pin actions to full commit SHAs** when you add or change a workflow that
  uses third-party actions.
- **This repository is public.** No secrets, keys, tokens or internal
  hostnames in code, commits, PRs or issues. The gateway holds no database
  credentials; Core does.
- Say "open weights", not "fully open source".

## Out of bounds

- Do not deploy the Worker, change routes, bindings, queues or the cron
  schedule, or touch Cloudflare or DNS configuration from here.
- Do not import history or restructure branches. The extraction is the
  owner's call.
- Do not edit `LICENSE` or `NOTICE`.
