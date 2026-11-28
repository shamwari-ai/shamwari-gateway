# The Rust port

The gateway that serves `api.shamwari.ai` is being rebuilt in Rust
(owner, 2026-11-28: "anything backend is Rust"). The TypeScript Worker in
[`shamwari-ai/shamwari/gateway`](https://github.com/shamwari-ai/shamwari/tree/main/gateway)
keeps serving traffic until every route here passes parity, then the
cutover moves the route, the cron and the queue consumer to this Worker
in one change.

Every module is ported and every route is wired. What is left is parity on
a deployed Worker and the cutover ([below](#the-cutover)).

## How it is built

- `workers-rs`, compiled to `wasm32-unknown-unknown` and bundled by
  `worker-build` (`scripts/build-worker.sh`), as in
  [`shamwari-ai/github-app`](https://github.com/shamwari-ai/github-app).
- Every rule is plain Rust over [`Platform`](../src/platform.rs) (HTTP,
  Workers AI, KV, the queue, the Secrets Store, the clock), so it runs under
  `cargo test` on the host. Only `src/worker_entry.rs` touches the runtime.
- Tests use the one fake, [`src/testing.rs`](../src/testing.rs). The tests
  that go through the Worker's fetch handler live in
  [`src/tests/`](../src/tests), over one shared world
  ([`src/tests/world.rs`](../src/tests/world.rs)).
- Each module names the TypeScript file it ports.

## Until the cutover

This Worker deploys as `shamwari-gateway-rs` on workers.dev only:

|                           | TypeScript `shamwari-gateway` | Rust `shamwari-gateway-rs`                     |
| ------------------------- | ----------------------------- | ---------------------------------------------- |
| `api.shamwari.ai`         | yes                           | no                                             |
| Weekly probe cron         | yes                           | no (handler built, no trigger)                 |
| `shamwari-sink` consumer  | yes                           | no (handler built, not bound)                  |
| Writes to `shamwari-sink` | yes                           | no (`SINK_ENABLED=false`)                      |

## Routes

| Route                       | Handler                                   | TypeScript                       | State |
| --------------------------- | ----------------------------------------- | -------------------------------- | ----- |
| `GET /health` (any method)  | `router.rs`                               | `index.ts`                       | done  |
| `GET /v1/models`            | `gateway::handle_models`                  | `index.ts`, `models.ts`          | done  |
| `POST /v1/ground/context`   | `gateway::handle_ground_context`          | `groundContext` in `index.ts`    | done  |
| `POST /v1/chat/completions` | `gateway::handle_chat`                    | `worker.fetch` in `index.ts`     | done  |
| `POST /v1/responses`        | `gateway::handle_open_responses`          | `responses.ts`                   | done  |
| `POST /v1/channels/sms`     | `gateway::handle_sms`                     | `channels/channel.ts`, `sms.ts`  | done  |
| `POST /v1/channels/ussd`    | `gateway::handle_ussd`                    | `channels/channel.ts`, `ussd.ts` | done  |
| queue `shamwari-sink`       | `worker_entry::queue` → `sink::drain`     | `worker.queue`, `sink.ts`        | done  |
| cron (weekly probe)         | `worker_entry::scheduled` → `probe::scheduled` | `worker.scheduled`, `probe.ts` | done  |

`router::dispatch` is the TypeScript `worker.fetch` (no CORS); `/v1/responses`
and both channels call it again in-process for the chat path, as TypeScript
does. `router::route` answers an error that escaped a handler with
`500 internal_error` and the cleaned reason; `router::handle` answers a CORS
preflight and wraps every other response in CORS from outside, the 500
included.

## Modules

| Rust                                                            | TypeScript                                                      | Port |
| --------------------------------------------------------------- | --------------------------------------------------------------- | ---- |
| `router.rs`                                                     | `index.ts` (dispatch, the top-level `catch`), `GET /health`     | done |
| `gateway.rs`                                                    | `index.ts` (`admit`, the chat and ground handlers, `citationsOf`, `ownInformation`) | done |
| `auth.rs`, `keys.rs`, `workos.rs`                               | `auth.ts`, `keys.ts`, `workos.ts`                               | done |
| `cors.rs`, `idempotency.rs`, `pricing.rs`                       | `cors.ts`, `idempotency.ts`, `pricing.ts`                       | done |
| `scope.rs`, `personal.rs`, `lang.rs`                            | `scope.ts`, `personal.ts`, `lang.ts`                            | done |
| `ground.rs`, `ground_store.rs`, `own_sites.rs`, `extractive.rs` | `ground.ts`, `ground-store.ts`, `own-sites.ts`, `extractive.ts` | done |
| `models.rs`, `model_router.rs`, `ai_gateway.rs`                 | `models.ts`, `router.ts`, `gateway.ts`                          | done |
| `envelope.rs`, `failure.rs`, `provenance.rs`, `skills.rs`       | `envelope.ts`, `failure.ts`, `provenance.ts`, `skills.ts`       | done |
| `core.rs`, `sink.rs`, `probe.rs`                                | `core.ts`, `sink.ts`, `probe.ts`                                | done |
| `responses.rs`, `channels/`                                     | `responses.ts`, `channels/`                                     | done |
| `worker_entry.rs`                                               | `export default` (`fetch`, `queue`, `scheduled`)                | done |

## The two rules

The TypeScript tests that guard them port with the code and pass unchanged
in meaning, stated as counts of provider calls where TypeScript states them
that way (`src/tests/chat.rs`, `ground_context.rs`, `responses.rs`,
`channels.rs`):

1. Personal content never reaches third-party inference (`scope.rs`,
   `personal.rs`, `gateway.rs`). An explicit personal scope is a 409 before
   Ground, an embedding or a model is reached; a question the gateway's own
   rules (or Core) say needs the person's own information gets only the cited
   Ground passage.
2. Only open-weight output with explicit consent may train Shamwari Mind
   (`provenance.rs`, `sink.rs`). An extractive answer writes no conversation
   row; channel traffic, a WorkOS sign-in and a personal-looking question are
   never trainable.

The person's WorkOS token is read for the one request and goes only to the
Nyuchi API (`ground_store.rs`). It is never logged, cached or stored.

## JSON

`Resp::json` writes `JSON.stringify(body, null, 2)` byte for byte (two-space
indentation, the same key order), as TypeScript's `json()` does. The places
whose TypeScript writes unindented JSON use `Resp::json_compact`: the compact
envelope, `/v1/responses` and the channel replies. Two tests compare whole
response bodies with the bytes TypeScript writes.

## Known differences from TypeScript

Intended, or forced by the runtime. Parity lists the ones a request can see
in `EXPECTED`.

From the wiring (this change):

- **`/health`** reports `runtime: "rust"` and the crate version (0.1.5);
  TypeScript reports `version: "1.0.0"` and no runtime.
- **No `ctx.waitUntil`.** The usage meter, the sink enqueue and the
  idempotency store are awaited before the response is returned, so their
  latency (a few ms to the queue; up to the 2 s Servers API timeout when that
  meter is on) is added to the response. What is written, and what a failure
  does, is unchanged.
- **Malformed `messages`.** A `messages` value that is not a list of
  `{role: system|user|assistant, content: string}` is a
  `400 {"error":"invalid_messages"}`. TypeScript trusts the shape: it answers
  500, or passes an unknown role on to the provider. An empty or missing list
  is still `400 messages_required`, and a body of JSON `null` is still the
  500 `internal_error`.
- **The queue consumer** acknowledges a message whose body cannot be read as a
  sink message and logs `sink parked a document: status=400
  error=unreadable_message` (no identifiers). TypeScript would send it to
  Core, which refuses it as a 400, and park it the same way.
- **String handling.** The trim in `auth.rs` (the bearer token, the price
  keys) and in the channels' input now uses JavaScript's whitespace set, so
  U+0085 is no longer trimmed and U+FEFF is, as in TypeScript.

From agent A (#24, auth):

- **A KV read error in `authenticate`** answers `503 auth_unavailable`.
  TypeScript throws there and the Worker answers `500 internal_error`.
- **The key path now shares Core's breaker** (this change switched `keys.rs`
  to `crate::core`), as TypeScript's `keys.ts` does: while Core is down a key
  check fails at once as `Unavailable`.

From agent B (#23, ground):

- **Own-sites cache in KV.** The hour-long edge cache lives in `AUTH_CACHE`
  under `own-sites:v1:<sha256>`; a KV error is a miss.
- **The own-sites deadline is a clock.** A slow site's request is aborted at
  the deadline instead of finishing in the background, so its answer is not
  cached for the next question.
- **No logging in `ground.rs` and `own_sites.rs`**: their `console.warn` /
  `console.error` lines are not ported.
- **Stricter chunk parsing.** A Core chunk missing a required field makes the
  answer ungrounded where TypeScript would use it as it came.

From agent C (#25, inference):

- **The sink enqueue sends one message at a time** (`queue_send`) rather than
  one `sendBatch`; a failure part-way leaves the earlier messages sent. It is
  logged and swallowed as before.
- **`lang::js::slice` and `lang::js::prefix`.** A cut through a surrogate
  pair leaves U+FFFD (`slice`) or stops before the pair (`prefix`, the failure
  reasons and the channels' input); JavaScript leaves a lone surrogate.

## Parity

[`scripts/parity.mjs`](../scripts/parity.mjs) sends the same requests to the
TypeScript Worker (baseline) and this Worker (candidate) and compares the
status, the content type, the CORS headers (and `Retry-After`,
`Idempotent-Replayed`) and the body as parsed JSON, with volatile fields
(ids, timestamps, `latency_ms`) normalised. Differences made on purpose are
in its `EXPECTED` map with the reason; anything else fails the run (exit 1).
Node 22, no dependencies.

```bash
# No key: health, 404s, method errors, CORS preflights, the 401s.
node scripts/parity.mjs \
  --baseline https://shamwari-gateway.<sub>.workers.dev \
  --candidate https://shamwari-gateway-rs.<sub>.workers.dev

# With a test key, from an environment variable (never on the command line):
export PARITY_API_KEY=...   # a test key, not a customer's
node scripts/parity.mjs --baseline ... --candidate ... --key PARITY_API_KEY
```

With a key it also sends `GET /v1/models`, `POST /v1/ground/context`, and
chat completions that cost no inference: `mode: "extractive"` at `reach: r0`
and in the compact envelope, and the 400s and the personal-scope 409. The
TypeScript Worker meters the extractive ones (zero-token usage events) as it
would any request with that key; the Rust Worker writes nothing while
`SINK_ENABLED` is `"false"`.

`--key` names the variable; a value that is not a variable name is refused.
The script refuses to send a key to `api.shamwari.ai`. Run it against the two
workers.dev URLs, never against production with a credential.

## The cutover

When parity is clean on the two workers.dev Workers, in one pull request per
repository, merged together:

1. **Move the route.** Add `routes = [{ pattern = "api.shamwari.ai",
   custom_domain = true }]` to `wrangler.toml` here, and remove the `routes`
   entry from `shamwari/gateway/wrangler.jsonc`.
2. **Move the cron.** Add `[triggers] crons = ["0 6 * * 1"]` here and remove
   `triggers` from `wrangler.jsonc`, so the probe runs once a week, not twice.
3. **Move the queue consumer.** Add the `[[queues.consumers]]` block here
   (`shamwari-sink`, `max_batch_size = 100`, `max_batch_timeout = 10`,
   `max_retries = 5`, `dead_letter_queue = "shamwari-sink-dlq"`) and remove
   `queues.consumers` from `wrangler.jsonc`. A queue has one consumer.
4. **Turn the sink on.** Set `SINK_ENABLED = "true"` (or remove it) here.
5. **Set the secrets** on `shamwari-gateway-rs` that the TypeScript Worker
   holds: `SHAMWARI_CORE_TOKEN`, `CHANNEL_WEBHOOK_SECRET`, `CHANNEL_API_KEY`,
   and `SERVERS_API_TOKEN` / `SERVERS_API_INTERNAL_KEY` if the Servers API is
   on.
6. **Soak.** Watch Workers Logs, the `shamwari-sink` backlog and DLQ, and
   `platform.serviceHealth` for at least a week, with the TypeScript Worker
   still deployed (no route) so a rollback is moving the route back.
7. **Delete the TypeScript gateway** (`shamwari/gateway`) and its Worker once
   the soak is clean.
