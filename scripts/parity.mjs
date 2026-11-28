#!/usr/bin/env node
/**
 * Parity: the TypeScript gateway (baseline) and this Rust Worker (candidate)
 * answer the same requests the same way.
 *
 *   node scripts/parity.mjs --baseline <url> --candidate <url> [--key NAME] [--json]
 *
 *   --baseline   the TypeScript Worker, e.g. https://shamwari-gateway.<sub>.workers.dev
 *   --candidate  the Rust Worker,       e.g. https://shamwari-gateway-rs.<sub>.workers.dev
 *   --key        the NAME of the environment variable holding a test API key
 *                (default PARITY_API_KEY; --key-env is the same flag). The key
 *                itself is never an argument, and a value that is not a
 *                variable name is refused:
 *                a value passed on the command line ends up in shell history
 *                and process listings. Without a key only the requests that
 *                need none are sent.
 *   --json       print the results as JSON instead of a table.
 *
 * No dependencies: Node 22's fetch. Every request is sent to both targets and
 * these are compared:
 *
 *   status · content-type · the CORS headers (and Retry-After,
 *   Idempotent-Replayed) · the body, parsed as JSON, with volatile fields
 *   (ids, timestamps, latency) normalised; text when it is not JSON.
 *
 * Differences made on purpose are listed in EXPECTED with the reason; anything
 * else fails the run (exit 1).
 *
 * Never point it at api.shamwari.ai with a key: it refuses. With a key the
 * extractive chat completion is metered on the TypeScript Worker (a zero-token
 * usage event), as any request with that key would be; the Rust Worker writes
 * nothing while SINK_ENABLED is "false".
 */

const args = Object.fromEntries(
  process.argv.slice(2).reduce((acc, a, i, all) => {
    if (a.startsWith("--")) acc.push([a.slice(2), all[i + 1] === undefined || all[i + 1].startsWith("--") ? true : all[i + 1]]);
    return acc;
  }, []),
);

function fail(msg) {
  console.error(`parity: ${msg}`);
  process.exit(2);
}

if (typeof args.baseline !== "string" || typeof args.candidate !== "string") {
  fail("usage: node scripts/parity.mjs --baseline <url> --candidate <url> [--key NAME] [--json]");
}
const BASELINE = args.baseline.replace(/\/+$/, "");
const CANDIDATE = args.candidate.replace(/\/+$/, "");
// `--key NAME` is the same as `--key-env NAME`: it names a variable, never holds a key.
const KEY_ARG = args["key-env"] ?? args.key;
const KEY_ENV = typeof KEY_ARG === "string" ? KEY_ARG : "PARITY_API_KEY";
if (!/^[A-Z_][A-Z0-9_]*$/.test(KEY_ENV)) fail("--key takes the NAME of an environment variable (e.g. PARITY_API_KEY), never a key");
const KEY = process.env[KEY_ENV] || null;

const PRODUCTION = new Set(["api.shamwari.ai"]);
for (const u of [BASELINE, CANDIDATE]) {
  let host;
  try {
    host = new URL(u).hostname;
  } catch {
    fail(`not a URL: ${u}`);
  }
  if (KEY && PRODUCTION.has(host)) fail(`refusing to send a credential to production (${host})`);
}

const SITE = "https://shamwari.ai";
const OTHER_ORIGIN = "https://parity.invalid";
const PLAIN_Q = "What does section 48 of the Constitution say?";
// A key-shaped value that no key service holds, drawn per run.
const BAD_KEY = `sk_shamwari_${[...crypto.getRandomValues(new Uint8Array(16))].map((b) => b.toString(16).padStart(2, "0")).join("")}`;

const HEADERS = [
  "content-type",
  "access-control-allow-origin",
  "access-control-allow-methods",
  "access-control-allow-headers",
  "access-control-allow-credentials",
  "access-control-expose-headers",
  "access-control-max-age",
  "vary",
  "retry-after",
  "idempotent-replayed",
];

/** Values that differ on every request, by key name, anywhere in a body. */
const VOLATILE = new Set(["id", "request_id", "latency_ms", "created_at", "completed_at", "createdAt", "startedAt"]);
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

function normalise(v) {
  if (Array.isArray(v)) return v.map(normalise);
  if (v && typeof v === "object") {
    const out = {};
    for (const [k, x] of Object.entries(v)) out[k] = VOLATILE.has(k) ? "<volatile>" : normalise(x);
    return out;
  }
  if (typeof v === "string" && UUID.test(v)) return "<uuid>";
  return v;
}

/** Drop the dotted paths an EXPECTED entry excuses (`shamwari.latency_ms`). */
function without(v, paths) {
  const copy = structuredClone(v);
  for (const p of paths) {
    const parts = p.split(".");
    let o = copy;
    for (const k of parts.slice(0, -1)) o = o?.[k];
    if (o && typeof o === "object") delete o[parts.at(-1)];
  }
  return copy;
}

const chat = (body) => ({ messages: [{ role: "user", content: PLAIN_Q }], ...body });
const auth = (key) => ({ Authorization: `Bearer ${key}` });
const json = { "Content-Type": "application/json" };
const preflight = (origin) => ({
  ...(origin ? { Origin: origin } : {}),
  "Access-Control-Request-Method": "POST",
  "Access-Control-Request-Headers": "authorization, content-type, idempotency-key, x-shamwari-channel",
});

/** Requests that need no key: they reach no Core, no model and no queue. */
const OPEN = [
  { name: "health", method: "GET", path: "/health" },
  { name: "health, any method", method: "POST", path: "/health" },
  { name: "unknown route", method: "GET", path: "/nope" },
  { name: "root", method: "GET", path: "/" },
  { name: "chat with GET", method: "GET", path: "/v1/chat/completions" },
  { name: "models with PUT", method: "PUT", path: "/v1/models" },
  { name: "models with POST", method: "POST", path: "/v1/models", headers: json, body: "{}" },
  { name: "responses with GET", method: "GET", path: "/v1/responses" },
  { name: "ground with GET", method: "GET", path: "/v1/ground/context" },
  { name: "preflight, chat, app origin", method: "OPTIONS", path: "/v1/chat/completions", headers: preflight(SITE) },
  { name: "preflight, responses, app origin", method: "OPTIONS", path: "/v1/responses", headers: preflight(SITE) },
  { name: "preflight, chat, other origin", method: "OPTIONS", path: "/v1/chat/completions", headers: preflight(OTHER_ORIGIN) },
  { name: "preflight, chat, no origin", method: "OPTIONS", path: "/v1/chat/completions", headers: preflight(null) },
  { name: "preflight, models (not a CORS path)", method: "OPTIONS", path: "/v1/models", headers: preflight(SITE) },
  { name: "OPTIONS that is not a preflight", method: "OPTIONS", path: "/v1/chat/completions" },
  { name: "chat, no key, app origin", method: "POST", path: "/v1/chat/completions", headers: { ...json, Origin: SITE }, body: JSON.stringify(chat({})) },
  { name: "chat, no key, other origin", method: "POST", path: "/v1/chat/completions", headers: { ...json, Origin: OTHER_ORIGIN }, body: JSON.stringify(chat({})) },
  { name: "chat, not a key", method: "POST", path: "/v1/chat/completions", headers: { ...json, ...auth("not-a-key") }, body: JSON.stringify(chat({})) },
  { name: "chat, unknown key", method: "POST", path: "/v1/chat/completions", headers: { ...json, ...auth(BAD_KEY), Origin: SITE }, body: JSON.stringify(chat({})) },
  { name: "models, no key", method: "GET", path: "/v1/models" },
  { name: "models, unknown key", method: "GET", path: "/v1/models", headers: auth(BAD_KEY) },
  { name: "ground, no key", method: "POST", path: "/v1/ground/context", headers: json, body: JSON.stringify(chat({})) },
  { name: "responses, no key", method: "POST", path: "/v1/responses", headers: { ...json, Origin: SITE }, body: JSON.stringify({ input: PLAIN_Q }) },
  { name: "responses, unknown key", method: "POST", path: "/v1/responses", headers: { ...json, ...auth(BAD_KEY) }, body: JSON.stringify({ input: PLAIN_Q }) },
];

/** Requests that need a key: still no inference. */
const KEYED = KEY
  ? [
      { name: "models", method: "GET", path: "/v1/models", headers: auth(KEY) },
      { name: "ground/context", method: "POST", path: "/v1/ground/context", headers: { ...json, ...auth(KEY) }, body: JSON.stringify(chat({})) },
      {
        name: "chat, extractive at r0 (no inference)",
        method: "POST",
        path: "/v1/chat/completions",
        headers: { ...json, ...auth(KEY), Origin: SITE },
        body: JSON.stringify(chat({ mode: "extractive", reach: "r0" })),
      },
      {
        name: "chat, extractive compact (no inference)",
        method: "POST",
        path: "/v1/chat/completions",
        headers: { ...json, ...auth(KEY) },
        body: JSON.stringify(chat({ mode: "extractive", format: "compact" })),
      },
      { name: "chat, invalid mode", method: "POST", path: "/v1/chat/completions", headers: { ...json, ...auth(KEY) }, body: JSON.stringify(chat({ mode: "turbo" })) },
      { name: "chat, invalid reach", method: "POST", path: "/v1/chat/completions", headers: { ...json, ...auth(KEY) }, body: JSON.stringify(chat({ reach: "r9" })) },
      { name: "chat, explicit personal (409)", method: "POST", path: "/v1/chat/completions", headers: { ...json, ...auth(KEY), Origin: SITE }, body: JSON.stringify(chat({ scope: "personal" })) },
      { name: "chat, bad idempotency key", method: "POST", path: "/v1/chat/completions", headers: { ...json, ...auth(KEY), "Idempotency-Key": "has space" }, body: JSON.stringify(chat({})) },
      { name: "chat, no messages", method: "POST", path: "/v1/chat/completions", headers: { ...json, ...auth(KEY) }, body: "{}" },
      { name: "chat, not JSON", method: "POST", path: "/v1/chat/completions", headers: { ...json, ...auth(KEY) }, body: "{" },
    ]
  : [];

/**
 * Differences made on purpose. `allow` names what may differ: `status`,
 * `header:<name>`, or `body:<dotted path>` (that field is dropped from both
 * bodies before they are compared). Anything else on the same request still
 * fails the run.
 */
const EXPECTED = {
  health: {
    allow: ["body:version", "body:runtime"],
    reason: "/health names the runtime (\"rust\") and the crate version; the TypeScript Worker reports version 1.0.0 and no runtime.",
  },
  "health, any method": {
    allow: ["body:version", "body:runtime"],
    reason: "As /health.",
  },
};

async function send(base, r) {
  const started = Date.now();
  try {
    const res = await fetch(base + r.path, { method: r.method, headers: r.headers ?? {}, body: r.body, redirect: "manual" });
    const text = await res.text();
    let body;
    try {
      body = { json: JSON.parse(text) };
    } catch {
      body = { text };
    }
    const headers = Object.fromEntries(HEADERS.map((h) => [h, res.headers.get(h)]));
    return { status: res.status, headers, body, ms: Date.now() - started };
  } catch (e) {
    return { status: 0, headers: {}, body: { text: `fetch failed: ${String(e?.cause ?? e)}` }, ms: Date.now() - started };
  }
}

function compare(r, a, b) {
  const exp = EXPECTED[r.name];
  const allow = new Set(exp?.allow ?? []);
  const diffs = [];
  const excused = [];
  const note = (kind, what) => (allow.has(kind) ? excused : diffs).push(what);

  if (a.status !== b.status) note("status", `status ${a.status} != ${b.status}`);
  for (const h of HEADERS) {
    if ((a.headers[h] ?? null) !== (b.headers[h] ?? null)) note(`header:${h}`, `${h}: ${JSON.stringify(a.headers[h])} != ${JSON.stringify(b.headers[h])}`);
  }
  const paths = [...allow].filter((x) => x.startsWith("body:")).map((x) => x.slice(5));
  if ("json" in a.body && "json" in b.body) {
    const [x, y] = [normalise(a.body.json), normalise(b.body.json)];
    for (const p of paths) if (!deepEqual(pick(x, p), pick(y, p))) excused.push(`body ${p}`);
    if (!deepEqual(without(x, paths), without(y, paths))) diffs.push(`body ${firstDiff(without(x, paths), without(y, paths))}`);
  } else if (JSON.stringify(a.body) !== JSON.stringify(b.body)) {
    diffs.push(`body ${JSON.stringify(a.body).slice(0, 120)} != ${JSON.stringify(b.body).slice(0, 120)}`);
  }
  return { diffs, excused, reason: exp?.reason ?? null };
}

function pick(v, p) {
  return p.split(".").reduce((o, k) => o?.[k], v);
}

function deepEqual(x, y) {
  if (x === y) return true;
  if (typeof x !== typeof y || x === null || y === null || typeof x !== "object") return false;
  if (Array.isArray(x) !== Array.isArray(y)) return false;
  const kx = Object.keys(x);
  const ky = Object.keys(y);
  if (kx.length !== ky.length) return false;
  return kx.every((k) => Object.hasOwn(y, k) && deepEqual(x[k], y[k]));
}

function firstDiff(x, y, at = "$") {
  if (deepEqual(x, y)) return null;
  if (x && y && typeof x === "object" && typeof y === "object" && Array.isArray(x) === Array.isArray(y)) {
    for (const k of new Set([...Object.keys(x), ...Object.keys(y)])) {
      const d = firstDiff(x[k], y[k], `${at}.${k}`);
      if (d) return d;
    }
  }
  return `${at}: ${JSON.stringify(x)?.slice(0, 100)} != ${JSON.stringify(y)?.slice(0, 100)}`;
}

const requests = [...OPEN, ...KEYED];
const results = [];
for (const r of requests) {
  // One at a time: the same request on both, then the next.
  const [a, b] = await Promise.all([send(BASELINE, r), send(CANDIDATE, r)]);
  const c = compare(r, a, b);
  results.push({ name: r.name, request: `${r.method} ${r.path}`, baseline: a.status, candidate: b.status, ms: [a.ms, b.ms], ...c });
}

const failed = results.filter((r) => r.diffs.length > 0);
if (args.json) {
  console.log(JSON.stringify({ baseline: BASELINE, candidate: CANDIDATE, keyed: Boolean(KEY), results }, null, 2));
} else {
  console.log(`baseline  ${BASELINE}\ncandidate ${CANDIDATE}\nkey       ${KEY ? `from $${KEY_ENV}` : "none (keyed requests skipped)"}\n`);
  for (const r of results) {
    const mark = r.diffs.length ? "FAIL" : r.excused.length ? "EXPECTED" : "ok";
    console.log(`${mark.padEnd(8)} ${String(r.baseline).padEnd(3)} ${String(r.candidate).padEnd(3)} ${r.request.padEnd(34)} ${r.name}`);
    for (const d of r.diffs) console.log(`           ${d}`);
    if (r.excused.length) console.log(`           expected: ${r.excused.join("; ")} (${r.reason})`);
  }
  console.log(`\n${results.length - failed.length}/${results.length} match${failed.length ? `, ${failed.length} unexplained` : ""}`);
}
process.exit(failed.length ? 1 : 0);
