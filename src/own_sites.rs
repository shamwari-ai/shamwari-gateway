//! Port of `shamwari/gateway/src/own-sites.ts`: our own sites, read live at
//! question time (owner, 2026-11-23: "travel info gets pulled from the site
//! directly; we own the site, so don't duplicate").
//!
//! Nothing is copied into Ground. Each site's own search (the MCP endpoint its
//! docs host serves) is asked the question, and the passages it returns are
//! cited with a link to the page, so an answer is always as current as the
//! site. The search sees the question text, so `ground()` calls this only
//! where a cloud provider may see it: platform scope, a cloud destination, and
//! a question that does not need the person's own information.
//!
//! Getting to an answer never waits on a site: one short limit for all of
//! them. A slow site only misses that one answer (slow is not down); an error
//! opens that site's breaker. What a site returns is kept for an hour, keyed
//! on the question, so a repeat question is answered at once and stays
//! current.
//!
//! # Where the hour is kept
//!
//! TypeScript used the Workers edge cache (`caches.default`). The Rust port
//! keeps it in KV through [`Platform::kv_put`] with the same one-hour TTL,
//! under `own-sites:v1:<sha-256 of the TypeScript cache key>`: a KV key has a
//! 512-byte limit a long question would pass, and the hash keeps the
//! question's words out of the key. A KV error is a cache miss, as a cache
//! error was.
//!
//! # The shared limit
//!
//! [`Platform`] has no timer, so the one limit for all sites is a deadline on
//! the clock: each site's request gets `timeout_ms` set to what is left of it,
//! and a site that answers, fails or times out after the deadline has passed
//! is slow, not down. Unlike TypeScript, a slow site's request is aborted at
//! the deadline rather than left to finish in the background, so its answer
//! is not cached for the next question.

use crate::lang::js;
use crate::platform::{HttpRequest, Platform};
use crate::types::GroundChunk;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::sync::OnceLock;
use unicode_normalization::UnicodeNormalization;

/// How an MCP search tool lays out its hits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HitFormat {
    /// One text item per hit ("Title: ...").
    Titled,
    /// One numbered list.
    Numbered,
}

/// A site with a search we can call: the MCP endpoint its docs host serves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct McpSite {
    /// Shown as the source's authority, and used in the citation.
    pub name: &'static str,
    pub mcp: &'static str,
    pub tool: &'static str,
    pub format: HitFormat,
}

/// A site with no search to call that publishes its full text as
/// `llms-full.txt`: searched here, by section.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FullTextSite {
    pub name: &'static str,
    pub url: &'static str,
    /// Pages under these links are left out: a copy of another site of ours,
    /// which is read from its source.
    pub skip: &'static [&'static str],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OwnSite {
    Mcp(McpSite),
    LlmsFull(FullTextSite),
}

impl OwnSite {
    pub fn name(&self) -> &'static str {
        match self {
            OwnSite::Mcp(s) => s.name,
            OwnSite::LlmsFull(s) => s.name,
        }
    }

    fn id(&self) -> &'static str {
        match self {
            OwnSite::Mcp(s) => s.mcp,
            OwnSite::LlmsFull(s) => s.url,
        }
    }
}

const fn mcp(
    name: &'static str,
    url: &'static str,
    tool: &'static str,
    format: HitFormat,
) -> OwnSite {
    OwnSite::Mcp(McpSite {
        name,
        mcp: url,
        tool,
        format,
    })
}

/// Everything of ours that a machine can read. A site that offers neither is
/// not here (owner: add one).
pub const OWN_SITES: &[OwnSite] = &[
    mcp(
        "Zimbabwe Travel Information",
        "https://www.travel-info.co.zw/mcp",
        "search_zimbabwe_travel_information",
        HitFormat::Titled,
    ),
    mcp(
        "Nyuchi Docs",
        "https://docs.nyuchi.com/mcp",
        "search_docs",
        HitFormat::Numbered,
    ),
    mcp(
        "Shamwari Docs",
        "https://docs.shamwari.ai/mcp",
        "search_shamwari",
        HitFormat::Titled,
    ),
    mcp(
        "Mzizi",
        "https://docs.mzizi.dev/mcp",
        "search_mzizi",
        HitFormat::Titled,
    ),
    mcp(
        "SiafuDB",
        "https://siafudb.org/mcp",
        "search_siafu_db",
        HitFormat::Titled,
    ),
    mcp(
        "Open Neural Transfer Layer",
        "https://openntl.org/mcp",
        "search_neural_transfer_layer",
        HitFormat::Titled,
    ),
    // Bundu Docs no longer carries a copy of the Mzizi docs (bundu-labs/bundu-docs#53): its old /mzizi/ pages
    // redirect to docs.mzizi.dev, and Mzizi is read from Mzizi. `skip` stays for any future copy.
    OwnSite::LlmsFull(FullTextSite {
        name: "Bundu Docs",
        url: "https://docs.bundu.org/llms-full.txt",
        skip: &[],
    }),
];

/// Sites answer in 0.4 to 3 s (a repeat question at once, from the cache).
/// Past this the answer goes on without them; with the ranking after (about
/// 0.3 s) the whole step stays near Ground's own 1.5 s.
pub const OWN_SITE_TIMEOUT_MS: u64 = 1200;
pub const OWN_SITE_COOLDOWN_MS: u64 = 30_000;
/// How long a site's answer to one question is kept.
pub const OWN_SITE_CACHE_SECONDS: u64 = 3600;
/// At most this many passages per site, each cut to this many characters.
pub const OWN_SITE_TOP_K: usize = 3;
pub const OWN_SITE_MAX_CHARS: usize = 1500;
/// Sections of a full text passed on to the ranking by meaning.
pub const OWN_SITE_FULL_TEXT_CANDIDATES: usize = 8;
/// Passages scoring at least this against the question (bge-m3 cosine) are
/// kept. Measured on our sites, 2026-11-23: passages that answer score 0.58
/// to 0.76, unrelated ones 0.30 to 0.45, "hello" at most 0.56.
pub const OWN_SITE_MIN_SIMILARITY: f64 = 0.58;
/// At most this many passages from all our sites together.
pub const OWN_SITE_KEEP: usize = 4;

thread_local! {
    /// The TypeScript module's `downUntil` map, by site id.
    static DOWN_UNTIL: RefCell<HashMap<&'static str, u64>> = RefCell::new(HashMap::new());
}

/// For tests: close every site's breaker.
pub fn reset_own_site_breakers() {
    DOWN_UNTIL.with(|d| d.borrow_mut().clear());
}

pub fn own_site_down(site: &OwnSite, now_ms: u64) -> bool {
    DOWN_UNTIL.with(|d| now_ms < d.borrow().get(site.id()).copied().unwrap_or(0))
}

const STOP: &str = "what when where which who whom whose why how does do did is are was were the and for with from that this there their about into your have has had can could would should will shall may might must tell me please much many some any best good";

fn stop_words() -> &'static HashSet<&'static str> {
    static S: OnceLock<HashSet<&'static str>> = OnceLock::new();
    S.get_or_init(|| STOP.split(' ').collect())
}

fn re(pattern: &str) -> Regex {
    Regex::new(pattern).expect("an own-sites pattern compiles")
}

macro_rules! regex {
    ($pattern:expr) => {{
        static R: OnceLock<Regex> = OnceLock::new();
        R.get_or_init(|| re(&$pattern))
    }};
}

/// A rough stem, the same on both sides: plurals, and British "-our" against
/// American "-or" (colour, color).
pub fn stem(w: &str) -> String {
    // `w.replace(/our(s|ed|ing)?$/, 'or$1')`: the leftmost "our" whose rest is
    // one of those endings.
    let mut s = w.to_owned();
    if let Some(i) = w
        .char_indices()
        .map(|(i, _)| i)
        .find(|&i| w[i..].starts_with("our") && matches!(&w[i + 3..], "" | "s" | "ed" | "ing"))
    {
        s = format!("{}or{}", &w[..i], &w[i + 3..]);
    }
    let long = js::len(&s) > 4;
    if long && s.ends_with("ies") {
        s.truncate(s.len() - 3);
        s.push('y');
    } else if long
        && s.ends_with("es")
        && ["ches", "shes", "xes", "sses"]
            .iter()
            .any(|e| s.ends_with(e))
    {
        s.truncate(s.len() - 2);
    } else if long && s.ends_with('s') && !s.ends_with("ss") {
        s.truncate(s.len() - 1);
    }
    s
}

/// Words of four letters or more that carry meaning: used to drop passages
/// that share nothing with the question.
pub fn terms(text: &str) -> HashSet<String> {
    let folded: String = text.to_lowercase().nfkd().collect();
    regex!(r"[^\p{L}\p{N}]+")
        .split(&folded)
        .filter(|w| js::len(w) >= 4 && !stop_words().contains(w))
        .map(stem)
        .collect()
}

/// MDX to readable text: keep a component's title, drop tags, keep link text.
/// (The TypeScript patterns have no `u` flag, so `\b` is an ASCII boundary.)
pub fn plain(md: &str) -> String {
    let s = regex!(
        r#"<[A-Z][A-Za-z]{0,40}(?-u:\b)[^>]{0,300}?(?-u:\b)title="([^"]{1,200})"[^>]{0,300}>"#
    )
    .replace_all(md, "\n${1}\n");
    let s = regex!(r"</?[A-Z][A-Za-z]{0,40}(?-u:\b)[^>]{0,500}>").replace_all(&s, "");
    let s = regex!(r"!\[[^\]]{0,300}\]\([^)]{0,500}\)").replace_all(&s, "");
    let s = regex!(r"\[([^\]]{1,300})\]\([^)]{0,500}\)").replace_all(&s, "${1}");
    let s = regex!(r"[ \t]+\n").replace_all(&s, "\n");
    let s = regex!(r"\n{3,}").replace_all(&s, "\n\n");
    js::trim(&s).to_owned()
}

/// One passage a site returned.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Hit {
    pub title: String,
    pub link: Option<String>,
    pub content: String,
}

/// Lines as a JavaScript `m` pattern's `^` and `$` see them.
fn js_lines(t: &str) -> impl Iterator<Item = &str> {
    t.split(['\n', '\r', '\u{2028}', '\u{2029}'])
}

/// The search tool answers with one text item per hit:
/// "Title: ...\nLink: ...\nPage: ...\nContent: ...".
pub fn parse_hits(texts: &[String]) -> Vec<Hit> {
    let mut hits = Vec::new();
    for t in texts {
        // `/^Title: (.*)$/m`, then `.trim()`.
        let title = js_lines(t)
            .find_map(|l| l.strip_prefix("Title: "))
            .map(|v| js::trim(v).to_owned());
        // `/^Link: (https:\/\/\S+)$/m`.
        let link = js_lines(t)
            .filter_map(|l| l.strip_prefix("Link: "))
            .find(|v| {
                v.strip_prefix("https://")
                    .is_some_and(|rest| !rest.is_empty() && !rest.chars().any(js::is_space))
            })
            .map(str::to_owned);
        let Some(title) = title.filter(|t| !t.is_empty()) else {
            continue;
        };
        let Some(i) = t.find("\nContent: ") else {
            continue;
        };
        hits.push(Hit {
            title,
            link,
            content: plain(&t[i + "\nContent: ".len()..]),
        });
    }
    hits
}

/// `s.split(/\n(?=<lookahead>)/)`: cut at each newline the rest after it
/// starts with `lookahead`.
fn split_before<'a>(s: &'a str, lookahead: &Regex) -> Vec<&'a str> {
    let mut out = Vec::new();
    let mut from = 0;
    for (i, _) in s.match_indices('\n') {
        if lookahead.is_match(&s[i + 1..]) {
            out.push(&s[from..i]);
            from = i + 1;
        }
    }
    out.push(&s[from..]);
    out
}

/// Nyuchi Docs' layout: one text holding
/// "1. Title\n   https://...\n   content", "2. ...".
pub fn parse_numbered(texts: &[String]) -> Vec<Hit> {
    let next = regex!(r"^[0-9]{1,2}\. ");
    let block_re = regex!(format!(
        r"^[0-9]{{1,2}}\. ([^\n]{{1,300}})\n[{ws}]*(https://[^{ws}]{{1,500}})\n((?s:.*))$",
        ws = js::SPACE
    ));
    let skip = regex!(r"\[Skip to content\]\([^)]{0,100}\)");
    let section = regex!(r"\[Section titled[^\]]{0,200}\]\([^)]{0,200}\)");
    let mut hits = Vec::new();
    for t in texts {
        for block in split_before(t, next) {
            let Some(m) = block_re.captures(js::trim(block)) else {
                continue;
            };
            let body = skip.replace_all(&m[3], "");
            let body = section.replace_all(&body, "");
            hits.push(Hit {
                title: js::trim(&m[1]).to_owned(),
                link: Some(m[2].to_owned()),
                content: plain(&body),
            });
        }
    }
    hits
}

/// An `llms-full.txt` split into sections: each page ("# Title", then
/// "Source: <url>") at its ##/### headings.
pub fn full_text_sections(text: &str) -> Vec<Hit> {
    let ws = js::SPACE;
    let page_start = regex!(format!(r"^# [^\n]{{1,300}}\n[{ws}]*\nSource: https://"));
    let page_re = regex!(format!(
        r"^# ([^\n]{{1,300}})\n[{ws}]*\nSource: (https://[^{ws}]{{1,500}})\n((?s:.*))$"
    ));
    let heading_start = regex!(r"^#{2,3} ");
    let heading = regex!(r"^#{2,3} ([^\n]{1,300})\n");
    let mut out = Vec::new();
    for page in split_before(text, page_start) {
        let Some(m) = page_re.captures(js::trim(page)) else {
            continue;
        };
        let (title, link, body) = (&m[1], &m[2], m.get(3).map_or("", |b| b.as_str()));
        for part in split_before(body, heading_start) {
            let h = heading.captures(part);
            let rest = match &h {
                Some(h) => &part[h[0].len()..],
                None => part,
            };
            let content = js::slice(&plain(rest), OWN_SITE_MAX_CHARS);
            // `content.split(/\s+/).length < 8`.
            if js_split_count(&content) < 8 {
                continue;
            }
            out.push(Hit {
                title: match &h {
                    Some(h) => format!("{title} — {}", js::trim(&h[1])),
                    None => title.to_owned(),
                },
                link: Some(link.to_owned()),
                content,
            });
        }
    }
    out
}

/// `s.split(/\s+/).length`: the runs of whitespace, plus one.
fn js_split_count(s: &str) -> usize {
    let mut n = 1;
    let mut in_space = false;
    for c in s.chars() {
        let sp = js::is_space(c);
        if sp && !in_space {
            n += 1;
        }
        in_space = sp;
    }
    n
}

/// Sections of a full text that share the most words with the question (the
/// ranking by meaning comes after).
pub fn best_sections(sections: &[Hit], query: &str, k: usize) -> Vec<Hit> {
    let want = terms(query);
    if want.is_empty() {
        return vec![];
    }
    // Words in the order JavaScript's Set iterates them does not matter here:
    // only the count is used.
    let mut scored: Vec<(&Hit, usize)> = sections
        .iter()
        .map(|h| {
            let have = terms(&format!("{} {} {}", h.title, h.title, h.content));
            (h, want.iter().filter(|w| have.contains(*w)).count())
        })
        .filter(|(_, n)| *n > 0)
        .collect();
    scored.sort_by_key(|x| std::cmp::Reverse(x.1));
    scored.into_iter().take(k).map(|(h, _)| h.clone()).collect()
}

/// An MCP reply arrives as JSON or as one server-sent event; either way, the
/// tool result's text items.
fn texts(body: &str) -> Result<Vec<String>, String> {
    let json = if js::trim_start(body).starts_with('{') {
        body.to_owned()
    } else {
        body.split('\n')
            .filter_map(|l| l.strip_prefix("data: "))
            .collect::<String>()
    };
    let j: Value = serde_json::from_str(&json).map_err(|e| format!("SyntaxError: {e}"))?;
    if j.is_null() {
        return Err("TypeError: null reply".into());
    }
    let content = match j.get("result").and_then(|r| r.get("content")) {
        None | Some(Value::Null) => return Ok(vec![]),
        Some(Value::Array(c)) => c,
        Some(_) => return Err("TypeError: content is not a list".into()),
    };
    Ok(content
        .iter()
        .filter(|c| c.get("type").and_then(Value::as_str) == Some("text"))
        .filter_map(|c| c.get("text").and_then(Value::as_str).map(str::to_owned))
        .collect())
}

/// The TypeScript edge cache key: the site and the question's terms, sorted.
fn cache_key(site: &OwnSite, query: &str) -> String {
    let mut words: Vec<String> = terms(query).into_iter().collect();
    words.sort_by(|a, b| js::cmp(a, b));
    let joined = words.join(" ");
    let q = if joined.is_empty() {
        js::trim(query).to_lowercase()
    } else {
        joined
    };
    format!("{}?q={}", site.id(), js::encode_uri_component(&q))
}

/// The KV key a cache key is kept under.
fn kv_key(key: &str) -> String {
    let digest = Sha256::digest(key.as_bytes());
    let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
    format!("own-sites:v1:{hex}")
}

async fn cached<P, F, Fut>(p: &P, key: &str, load: F) -> Result<Vec<Hit>, String>
where
    P: Platform,
    F: FnOnce() -> Fut,
    Fut: Future<Output = Result<Vec<Hit>, String>>,
{
    let k = kv_key(key);
    if let Ok(Some(hit)) = p.kv_get(&k).await {
        if let Ok(v) = serde_json::from_str::<Vec<Hit>>(&hit) {
            return Ok(v);
        }
    }
    let value = load().await?;
    if let Ok(s) = serde_json::to_string(&value) {
        let _ = p.kv_put(&k, &s, Some(OWN_SITE_CACHE_SECONDS)).await;
    }
    Ok(value)
}

/// What a site's request may still take before the shared deadline.
fn remaining(p: &impl Platform, deadline: u64) -> Option<u32> {
    let left = deadline.saturating_sub(p.now_ms());
    (left > 0).then(|| left.min(u32::MAX as u64) as u32)
}

/// The outcome of asking one site.
enum Asked {
    Hits(Vec<Hit>),
    Slow,
    Failed,
}

async fn fetch_hits<P: Platform>(p: &P, site: &OwnSite, query: &str, deadline: u64) -> Asked {
    if remaining(p, deadline).is_none() {
        return Asked::Slow;
    }
    // What is left when the request is made (a cache read comes first).
    let limit = || remaining(p, deadline).ok_or_else(|| "timeout".to_owned());
    let result = match site {
        OwnSite::LlmsFull(s) => {
            // The whole text is kept for an hour, not per question; it is split and searched here.
            let sections = cached(p, s.url, || async {
                let res = p
                    .fetch(HttpRequest {
                        method: "GET".into(),
                        url: s.url.into(),
                        timeout_ms: Some(limit()?),
                        ..Default::default()
                    })
                    .await?;
                if !res.ok() {
                    return Err(format!("http_{}", res.status));
                }
                Ok(full_text_sections(&res.body))
            })
            .await;
            sections.map(|sections| {
                let own: Vec<Hit> = sections
                    .into_iter()
                    .filter(|h| {
                        !s.skip
                            .iter()
                            .any(|prefix| h.link.as_deref().is_some_and(|l| l.starts_with(prefix)))
                    })
                    .collect();
                // More candidates than a search site gives: the ranking by meaning picks among them.
                best_sections(&own, query, OWN_SITE_FULL_TEXT_CANDIDATES)
            })
        }
        OwnSite::Mcp(s) => {
            cached(p, &cache_key(site, query), || async {
                let body = json!({
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": { "name": s.tool, "arguments": { "query": query } },
                });
                let res = p
                    .fetch(HttpRequest {
                        method: "POST".into(),
                        url: s.mcp.into(),
                        headers: vec![
                            ("Content-Type".into(), "application/json".into()),
                            (
                                "Accept".into(),
                                "application/json, text/event-stream".into(),
                            ),
                        ],
                        body: Some(body.to_string()),
                        timeout_ms: Some(limit()?),
                    })
                    .await?;
                if !res.ok() {
                    return Err(format!("http_{}", res.status));
                }
                let t = texts(&res.body)?;
                let mut hits = match s.format {
                    HitFormat::Numbered => parse_numbered(&t),
                    HitFormat::Titled => parse_hits(&t),
                };
                hits.truncate(OWN_SITE_TOP_K);
                Ok(hits)
            })
            .await
        }
    };
    // Past the deadline the answer has gone on without this site, whatever it
    // said: slow, never down. "timeout" is the runtime aborting the request at
    // the deadline it was given.
    if p.now_ms() >= deadline || result.as_ref().is_err_and(|e| e == "timeout") {
        return Asked::Slow;
    }
    match result {
        Ok(h) => Asked::Hits(h),
        Err(_) => Asked::Failed,
    }
}

async fn search_site<P: Platform>(
    p: &P,
    site: &OwnSite,
    query: &str,
    deadline: u64,
) -> Vec<GroundChunk> {
    if own_site_down(site, p.now_ms()) {
        return vec![];
    }
    match fetch_hits(p, site, query, deadline).await {
        // Slow is not down: this answer goes on without the site, and the breaker stays shut.
        Asked::Slow => vec![],
        Asked::Hits(hits) => {
            DOWN_UNTIL.with(|d| d.borrow_mut().remove(site.id()));
            hits.into_iter()
                .map(|h| GroundChunk {
                    content: js::slice(&h.content, OWN_SITE_MAX_CHARS),
                    heading: None,
                    title: h.title,
                    authority: site.name().to_owned(),
                    resource_type: "web page".into(),
                    source_url: h.link,
                    effective_from: None,
                    scope: "platform".into(),
                })
                .collect()
        }
        Asked::Failed => {
            let until = p.now_ms() + OWN_SITE_COOLDOWN_MS;
            DOWN_UNTIL.with(|d| d.borrow_mut().insert(site.id(), until));
            vec![]
        }
    }
}

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let (mut d, mut x, mut y) = (0.0, 0.0, 0.0);
    for (ai, bi) in a.iter().zip(b) {
        d += ai * bi;
        x += ai * ai;
        y += bi * bi;
    }
    // `x && y ? ... : 0`: zero and NaN are false.
    let truthy = |v: f64| v != 0.0 && !v.is_nan();
    if truthy(x) && truthy(y) {
        d / (x * y).sqrt()
    } else {
        0.0
    }
}

/// Keep the passages that answer the question. With the question's
/// embedding, rank every passage by meaning in one embedding call; without it
/// (or if that call fails), keep those that share a word with the question.
///
/// `embed` is the provider call for the passages; `ground()` checks the scope
/// decision inside it, as TypeScript does.
pub async fn rank_own_sites<F, Fut>(
    query: &str,
    chunks: Vec<GroundChunk>,
    question_vector: Option<&[f64]>,
    embed: F,
) -> Vec<GroundChunk>
where
    F: FnOnce(Vec<String>) -> Fut,
    Fut: Future<Output = Result<Vec<Vec<f64>>, String>>,
{
    if chunks.is_empty() {
        return vec![];
    }
    if let Some(qv) = question_vector {
        let inputs = chunks
            .iter()
            .map(|c| format!("{}\n{}", c.title, js::slice(&c.content, 600)))
            .collect();
        if let Ok(vs) = embed(inputs).await {
            if vs.len() == chunks.len() {
                let mut scored: Vec<(GroundChunk, f64)> = chunks
                    .into_iter()
                    .zip(vs.iter().map(|v| cosine(qv, v)))
                    .filter(|(_, s)| *s >= OWN_SITE_MIN_SIMILARITY)
                    .collect();
                scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                return scored
                    .into_iter()
                    .take(OWN_SITE_KEEP)
                    .map(|(c, _)| c)
                    .collect();
            }
        }
        // The ranking is unavailable: matching words.
    }
    let want = terms(query);
    chunks
        .into_iter()
        .filter(|c| {
            terms(&format!("{} {}", c.title, c.content))
                .iter()
                .any(|w| want.contains(w))
        })
        .take(OWN_SITE_KEEP)
        .collect()
}

/// Ask every site at once, under one limit. A site that fails or is slow adds
/// nothing; this never fails.
pub async fn search_own_sites<P: Platform>(p: &P, query: &str) -> Vec<GroundChunk> {
    search_own_sites_with(p, query, OWN_SITES, OWN_SITE_TIMEOUT_MS).await
}

/// [`search_own_sites`] over these sites, with this limit.
pub async fn search_own_sites_with<P: Platform>(
    p: &P,
    query: &str,
    sites: &[OwnSite],
    timeout_ms: u64,
) -> Vec<GroundChunk> {
    let deadline = p.now_ms() + timeout_ms;
    futures_util::future::join_all(sites.iter().map(|s| search_site(p, s, query, deadline)))
        .await
        .into_iter()
        .flatten()
        .collect()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::config::Config;
    use crate::ground::ground;
    use crate::platform::HttpResponse;
    use crate::scope::{Destination, ScopeResolution};
    use crate::testing::{block_on, FakePlatform};
    use crate::types::{Message, Role, Scope};

    pub(crate) const HIT: &str = "Title: When is the best time to visit Zimbabwe?\nLink: https://www.travel-info.co.zw/faq#when-is-the-best-time-to-visit-zimbabwe\nPage: faq\nContent: ## Climate\n\n<Accordion title=\"When is the best time to visit Zimbabwe?\">\n  Victoria Falls is at high water from about February to May. See [weather by month](/essentials/weather-by-month).\n</Accordion>";
    const OFF_TOPIC: &str = "Title: Packing guide\nLink: https://www.travel-info.co.zw/essentials/packing-guide\nPage: x\nContent: Bring a hat.";

    pub(crate) fn sse(texts: &[&str]) -> String {
        let content: Vec<Value> = texts
            .iter()
            .map(|t| json!({ "type": "text", "text": t }))
            .collect();
        format!(
            "event: message\ndata: {}\n\n",
            json!({ "result": { "content": content } })
        )
    }

    fn env() -> Config {
        Config {
            own_sites: Some("on".into()),
            core_url: Some("https://core.invalid".into()),
            shamwari_core_token: Some("t".into()),
            ground_top_k: Some("6".into()),
            embedding_model: Some("@cf/baai/bge-m3".into()),
            ..Default::default()
        }
    }

    const TRAVEL: OwnSite = OWN_SITES[0];
    const ONE: &[OwnSite] = &[TRAVEL];

    fn platform() -> FakePlatform {
        reset_own_site_breakers();
        crate::ground_store::reset_ground_store_breaker();
        let p = FakePlatform::new();
        p.on_ai(|_, _| Some(Ok(json!({ "data": [[0.5, 0.25]] }))));
        p
    }

    fn resolution(scope: Scope, destination: Destination, without: bool) -> ScopeResolution {
        ScopeResolution {
            scope,
            destination,
            grants: vec![scope],
            without_own_information: without,
        }
    }

    fn ask(q: &str) -> Vec<Message> {
        vec![Message {
            role: Role::User,
            content: q.into(),
        }]
    }

    fn host(url: &str) -> String {
        url::Url::parse(url).unwrap().host_str().unwrap().to_owned()
    }

    fn ok(body: &str) -> Option<Result<HttpResponse, String>> {
        Some(Ok(HttpResponse {
            status: 200,
            headers: vec![],
            body: body.into(),
        }))
    }

    fn chunk(title: &str, content: &str) -> GroundChunk {
        GroundChunk {
            content: content.into(),
            heading: None,
            title: title.into(),
            authority: "a".into(),
            resource_type: "web page".into(),
            source_url: None,
            effective_from: None,
            scope: "platform".into(),
        }
    }

    fn set(words: &[&str]) -> HashSet<String> {
        words.iter().map(|w| w.to_string()).collect()
    }

    async fn no_embed(_: Vec<String>) -> Result<Vec<Vec<f64>>, String> {
        Ok(vec![])
    }

    async fn core_via(p: &FakePlatform, body: Value) -> Result<Value, String> {
        let res = p
            .fetch(HttpRequest {
                method: "POST".into(),
                url: "https://core.invalid/ground/search".into(),
                body: Some(body.to_string()),
                ..Default::default()
            })
            .await?;
        if !res.ok() {
            return Err(format!("core {}", res.status));
        }
        serde_json::from_str(&res.body).map_err(|e| e.to_string())
    }

    // reading our own sites

    #[test]
    fn turns_the_mdx_a_site_returns_into_plain_text_with_the_link_text_kept() {
        let h = &parse_hits(&[HIT.to_owned()])[0];
        assert_eq!(h.title, "When is the best time to visit Zimbabwe?");
        assert_eq!(
            h.link.as_deref(),
            Some("https://www.travel-info.co.zw/faq#when-is-the-best-time-to-visit-zimbabwe")
        );
        assert!(h.content.contains("Victoria Falls is at high water"));
        assert!(h.content.contains("weather by month"));
        assert!(!h.content.contains("<Accordion") && !h.content.contains("]("));
        assert_eq!(
            plain("<Card title=\"Hwange\">\nBig park\n</Card>"),
            "Hwange\n\nBig park"
        );
    }

    #[test]
    fn without_a_question_vector_keeps_only_passages_that_share_a_word_with_the_question() {
        let p = platform();
        let body = sse(&[HIT, OFF_TOPIC]);
        p.on_fn(move |_| ok(&body));
        let q = "When does Victoria Falls have the most water?";
        let found = block_on(search_own_sites_with(&p, q, ONE, OWN_SITE_TIMEOUT_MS));
        let out = block_on(rank_own_sites(q, found, None, no_embed));
        let titles: Vec<&str> = out.iter().map(|c| c.title.as_str()).collect();
        assert_eq!(titles, ["When is the best time to visit Zimbabwe?"]);
        assert_eq!(out[0].authority, "Zimbabwe Travel Information");
        assert_eq!(out[0].scope, "platform");
        assert!(out[0]
            .source_url
            .as_deref()
            .unwrap()
            .starts_with("https://"));
        assert_eq!(terms("hello there"), set(&["hello"]));
        assert_eq!(terms("colours"), terms("color"));
        assert_eq!(terms("categories"), terms("category"));
    }

    #[test]
    fn with_a_question_vector_ranks_every_passage_by_meaning_and_keeps_only_those_above_the_bar() {
        let chunks = vec![chunk("weak", "x"), chunk("best", "x"), chunk("good", "x")];
        let vec = |s: f64| vec![s, (1.0 - s * s).sqrt()];
        let vs = vec![vec(0.4), vec(0.75), vec(OWN_SITE_MIN_SIMILARITY + 0.01)];
        let out = block_on(rank_own_sites(
            "q",
            chunks,
            Some(&[1.0, 0.0]),
            |_| async move { Ok(vs) },
        ));
        let titles: Vec<&str> = out.iter().map(|c| c.title.as_str()).collect();
        assert_eq!(titles, ["best", "good"]);
    }

    #[test]
    fn keeps_at_most_four_passages_from_all_sites_together() {
        let chunks: Vec<GroundChunk> = (0..6).map(|i| chunk(&format!("t{i}"), "x")).collect();
        let vs: Vec<Vec<f64>> = (0..6).map(|i| vec![1.0, 0.01 * i as f64]).collect();
        let out = block_on(rank_own_sites(
            "q",
            chunks,
            Some(&[1.0, 0.0]),
            |_| async move { Ok(vs) },
        ));
        let titles: Vec<&str> = out.iter().map(|c| c.title.as_str()).collect();
        assert_eq!(titles, ["t0", "t1", "t2", "t3"]);
        // The passages are embedded as title, then the first 600 characters.
        let seen = RefCell::new(vec![]);
        let long = chunk("T", &"y".repeat(900));
        block_on(rank_own_sites("q", vec![long], Some(&[1.0]), |t| {
            *seen.borrow_mut() = t;
            async { Ok(vec![]) }
        }));
        assert_eq!(seen.borrow()[0], format!("T\n{}", "y".repeat(600)));
        let lexical: Vec<GroundChunk> = (0..6)
            .map(|i| chunk(&format!("victoria {i}"), "x"))
            .collect();
        assert_eq!(
            block_on(rank_own_sites("Victoria", lexical, None, no_embed)).len(),
            OWN_SITE_KEEP
        );
    }

    #[test]
    fn falls_back_to_matching_words_if_the_ranking_call_fails() {
        let c = chunk("t", "Victoria Falls water");
        let out = block_on(rank_own_sites(
            "Victoria Falls",
            vec![c.clone()],
            Some(&[1.0, 0.0]),
            |_| async { Err("down".to_owned()) },
        ));
        assert_eq!(out.len(), 1);
        // And if it answers for the wrong number of passages.
        let out = block_on(rank_own_sites(
            "Victoria Falls",
            vec![c],
            Some(&[1.0, 0.0]),
            no_embed,
        ));
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn reads_nyuchi_docs_numbered_layout() {
        let t = "1. API overview\n   https://docs.nyuchi.com/api/overview/\n   [Skip to content](#%5Ftop)\n\n# API overview\n\nThe Nyuchi API is one gateway.\n\n2. Identity\n   https://docs.nyuchi.com/identity/\n   How Nyuchi signs people in.";
        let hits = parse_numbered(&[t.to_owned()]);
        let got: Vec<(&str, &str)> = hits
            .iter()
            .map(|h| (h.title.as_str(), h.link.as_deref().unwrap()))
            .collect();
        assert_eq!(
            got,
            [
                ("API overview", "https://docs.nyuchi.com/api/overview/"),
                ("Identity", "https://docs.nyuchi.com/identity/"),
            ]
        );
        assert!(hits[0].content.contains("The Nyuchi API is one gateway."));
        assert!(!hits[0].content.contains("Skip to content"));
    }

    #[test]
    fn splits_an_llms_full_txt_into_sections_with_links_and_picks_the_ones_sharing_words() {
        let full = [
            "# Bundu Docs — full text", "", "---", "",
            "# Brand System", "", "Source: https://docs.bundu.org/bundu/brand/overview/", "",
            "Brand guidelines for every Bundu product, five minerals in all of them.", "",
            "## The Five African Minerals", "", "Our colour palette is drawn from the mineral wealth of Africa and each colour has meaning.", "",
            "# Engineering overview", "", "Source: https://docs.bundu.org/bundu/engineering/overview/", "",
            "How we build software across the foundation, with the same rules for every repository we own.",
        ]
        .join("\n");
        let sections = full_text_sections(&full);
        let titles: Vec<&str> = sections.iter().map(|h| h.title.as_str()).collect();
        assert_eq!(
            titles,
            [
                "Brand System",
                "Brand System — The Five African Minerals",
                "Engineering overview"
            ]
        );
        assert_eq!(
            sections[1].link.as_deref(),
            Some("https://docs.bundu.org/bundu/brand/overview/")
        );
        let best = best_sections(
            &sections,
            "What colour palette do the minerals give?",
            OWN_SITE_TOP_K,
        );
        assert_eq!(best[0].title, "Brand System — The Five African Minerals");
        assert_eq!(best_sections(&sections, "hi", OWN_SITE_TOP_K), vec![]);
    }

    #[test]
    fn leaves_out_a_full_text_sites_copy_of_another_of_our_sites() {
        let full = [
            "# Bundu Docs — full text", "",
            "# Design tokens", "", "Source: https://docs.bundu.org/mzizi/foundations/tokens/", "",
            "The Five African Minerals palette for every product, in five colours that each carry meaning.", "",
            "# Brand System", "", "Source: https://docs.bundu.org/bundu/brand/overview/", "",
            "The minerals palette carries the brand across every product we make for the foundation.",
        ]
        .join("\n");
        let p = platform();
        p.on_fn(move |_| ok(&full));
        let bundu = OwnSite::LlmsFull(FullTextSite {
            name: "Bundu Docs",
            url: "https://docs.bundu.org/llms-full.txt",
            skip: &["https://docs.bundu.org/mzizi/"],
        });
        let out = block_on(search_own_sites_with(
            &p,
            "minerals palette",
            &[bundu],
            OWN_SITE_TIMEOUT_MS,
        ));
        let links: Vec<&str> = out
            .iter()
            .map(|c| c.source_url.as_deref().unwrap())
            .collect();
        assert_eq!(links, ["https://docs.bundu.org/bundu/brand/overview/"]);
    }

    #[test]
    fn asks_the_site_its_own_search_tool_with_the_question() {
        let p = platform();
        let body = sse(&[HIT]);
        p.on_fn(move |_| ok(&body));
        block_on(search_own_sites_with(
            &p,
            "Victoria Falls water",
            ONE,
            OWN_SITE_TIMEOUT_MS,
        ));
        let req = p.requests.borrow()[0].clone();
        let OwnSite::Mcp(travel) = TRAVEL else {
            unreachable!()
        };
        assert_eq!(req.url, travel.mcp);
        assert_eq!(req.method, "POST");
        assert!(req.headers.contains(&(
            "Accept".into(),
            "application/json, text/event-stream".into()
        )));
        let sent: Value = serde_json::from_str(req.body.as_deref().unwrap()).unwrap();
        assert_eq!(sent["method"], "tools/call");
        assert_eq!(sent["params"]["name"], travel.tool);
        assert_eq!(
            sent["params"]["arguments"],
            json!({ "query": "Victoria Falls water" })
        );
        // Under the one shared limit.
        assert_eq!(req.timeout_ms, Some(OWN_SITE_TIMEOUT_MS as u32));
    }

    #[test]
    fn reads_a_plain_json_reply_too() {
        let p = platform();
        p.on_fn(|_| ok(&json!({ "result": { "content": [{ "type": "text", "text": HIT }, { "type": "image" }] } }).to_string()));
        let out = block_on(search_own_sites_with(
            &p,
            "Victoria Falls",
            ONE,
            OWN_SITE_TIMEOUT_MS,
        ));
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn a_failing_site_adds_nothing_never_fails_and_is_skipped_for_a_while() {
        let p = platform();
        p.on("https://", 502, "down");
        assert_eq!(
            block_on(search_own_sites_with(
                &p,
                "Victoria Falls",
                ONE,
                OWN_SITE_TIMEOUT_MS
            )),
            vec![]
        );
        assert!(own_site_down(&TRAVEL, p.now_ms()));
        assert_eq!(
            block_on(search_own_sites_with(
                &p,
                "Victoria Falls",
                ONE,
                OWN_SITE_TIMEOUT_MS
            )),
            vec![]
        );
        assert_eq!(p.requests.borrow().len(), 1);
        const { assert!(OWN_SITE_TIMEOUT_MS <= 1500) };
        *p.now.borrow_mut() += OWN_SITE_COOLDOWN_MS;
        assert!(!own_site_down(&TRAVEL, p.now_ms()));
    }

    /// A site that answers after the deadline: the clock moves on past it.
    struct Slow(FakePlatform, u64);

    impl Platform for Slow {
        async fn fetch(&self, req: HttpRequest) -> Result<HttpResponse, String> {
            let out = self.0.fetch(req).await;
            *self.0.now.borrow_mut() += self.1;
            out
        }
        async fn ai_run(&self, m: &str, i: &Value, o: Option<&Value>) -> Result<Value, String> {
            self.0.ai_run(m, i, o).await
        }
        async fn kv_get(&self, k: &str) -> Result<Option<String>, String> {
            self.0.kv_get(k).await
        }
        async fn kv_put(&self, k: &str, v: &str, t: Option<u64>) -> Result<(), String> {
            self.0.kv_put(k, v, t).await
        }
        async fn kv_delete(&self, k: &str) -> Result<(), String> {
            self.0.kv_delete(k).await
        }
        async fn queue_send(&self, m: &Value) -> Result<(), String> {
            self.0.queue_send(m).await
        }
        async fn secret_store_get(&self, b: &str) -> Result<Option<String>, String> {
            self.0.secret_store_get(b).await
        }
        fn now_ms(&self) -> u64 {
            self.0.now_ms()
        }
        fn random_bytes(&self, n: usize) -> Vec<u8> {
            self.0.random_bytes(n)
        }
    }

    #[test]
    fn a_slow_site_misses_this_answer_only_the_breaker_stays_shut() {
        // It answers, but after the limit.
        let slow = Slow(platform(), 200);
        let body = sse(&[HIT]);
        slow.0.on_fn(move |_| ok(&body));
        assert_eq!(
            block_on(search_own_sites_with(&slow, "Victoria Falls", ONE, 20)),
            vec![]
        );
        assert!(!own_site_down(&TRAVEL, slow.now_ms()));
        // It fails, but after the limit: still slow, not down.
        let slow = Slow(platform(), 200);
        slow.0.on("https://", 502, "down");
        assert_eq!(
            block_on(search_own_sites_with(&slow, "Victoria Falls", ONE, 20)),
            vec![]
        );
        assert!(!own_site_down(&TRAVEL, slow.now_ms()));
        // The runtime aborted it at the limit it was given.
        let p = platform();
        p.on_fn(|_| Some(Err("timeout".into())));
        assert_eq!(
            block_on(search_own_sites_with(&p, "Victoria Falls", ONE, 20)),
            vec![]
        );
        assert!(!own_site_down(&TRAVEL, p.now_ms()));
        assert_eq!(p.requests.borrow()[0].timeout_ms, Some(20));
    }

    #[test]
    fn every_site_shares_one_deadline() {
        // The first site to answer takes 1000 ms of the 1200: the next gets what is left.
        let slow = Slow(platform(), 1000);
        let body = sse(&[HIT]);
        slow.0.on_fn(move |_| ok(&body));
        let two = [OWN_SITES[0], OWN_SITES[2]];
        block_on(search_own_sites_with(
            &slow,
            "Victoria Falls",
            &two,
            OWN_SITE_TIMEOUT_MS,
        ));
        let timeouts: Vec<Option<u32>> = slow
            .0
            .requests
            .borrow()
            .iter()
            .map(|r| r.timeout_ms)
            .collect();
        assert_eq!(timeouts, [Some(1200), Some(200)]);
    }

    #[test]
    fn a_repeat_question_is_answered_from_the_hour_kept_in_kv() {
        let p = platform();
        let body = sse(&[HIT]);
        p.on_fn(move |_| ok(&body));
        let first = block_on(search_own_sites_with(
            &p,
            "Victoria Falls water",
            ONE,
            OWN_SITE_TIMEOUT_MS,
        ));
        // The same words in another order and case: the same key.
        let again = block_on(search_own_sites_with(
            &p,
            "water, victoria FALLS?",
            ONE,
            OWN_SITE_TIMEOUT_MS,
        ));
        assert_eq!(first, again);
        assert_eq!(first.len(), 1);
        assert_eq!(p.requests.borrow().len(), 1);
        let kv = p.kv.borrow();
        let keys: Vec<&String> = kv.keys().collect();
        assert_eq!(keys.len(), 1);
        assert!(keys[0].starts_with("own-sites:v1:"));
        assert!(keys[0].len() < 512);
        // The key carries no word of the question.
        assert!(!keys[0].contains("victoria"));
        assert_eq!(
            cache_key(&TRAVEL, "Victoria Falls water"),
            "https://www.travel-info.co.zw/mcp?q=fall%20victoria%20water"
        );
        assert_eq!(
            cache_key(&TRAVEL, " Hi "),
            "https://www.travel-info.co.zw/mcp?q=hi"
        );
    }

    // ground() and our own sites

    #[test]
    fn cites_the_page_with_its_link_beside_whatever_ground_found() {
        let p = platform();
        let body = sse(&[HIT]);
        p.on_fn(move |r| {
            if host(&r.url) == "core.invalid" {
                return Some(Ok(HttpResponse {
                    status: 503,
                    headers: vec![],
                    body: "down".into(),
                }));
            }
            ok(&body)
        });
        let cfg = env();
        let cloud = resolution(Scope::Platform, Destination::Cloud, false);
        // bge-m3 says the travel passage answers the question.
        p.ai.borrow_mut().clear();
        p.on_ai(|_, input| {
            let n = input["text"].as_array().unwrap().len();
            Some(Ok(json!({ "data": vec![vec![1.0, 0.0]; n] })))
        });
        let r = block_on(ground(
            &p,
            &cfg,
            |b| core_via(&p, b),
            &ask("When is Victoria Falls fullest?"),
            "e-1",
            &cloud,
            Some("en"),
            None,
        ))
        .unwrap();
        assert!(r.hit);
        assert!(r.system_prompt.contains("Zimbabwe Travel Information"));
        assert!(r
            .system_prompt
            .contains("Link: https://www.travel-info.co.zw/faq"));
        // Ground could not be asked, but our sites answered: a hit, not "unavailable".
        assert!(!r.unavailable);
    }

    #[test]
    fn rule_1_a_question_that_needs_the_persons_own_information_never_reaches_a_site() {
        let p = platform();
        p.on_fn(|r| {
            assert_eq!(host(&r.url), "core.invalid", "a site must not be asked");
            ok(r#"{"scope":"platform","hit":false,"chunks":[]}"#)
        });
        let cfg = env();
        let personal = resolution(Scope::Platform, Destination::Cloud, true);
        block_on(ground(
            &p,
            &cfg,
            |b| core_via(&p, b),
            &ask("What is my tax at Victoria Falls?"),
            "e-1",
            &personal,
            Some("en"),
            None,
        ))
        .unwrap();
        assert!(p
            .requests
            .borrow()
            .iter()
            .all(|r| host(&r.url) == "core.invalid"));
        assert!(p.ai_calls.borrow().is_empty());
    }

    #[test]
    fn rule_1_a_mind_destination_or_a_non_platform_scope_never_reaches_a_site() {
        let p = platform();
        p.on_fn(|_| ok(r#"{"scope":"community","hit":false,"chunks":[]}"#));
        let cfg = env();
        let mind = resolution(Scope::Platform, Destination::Mind, false);
        let community = resolution(Scope::Community, Destination::Cloud, false);
        block_on(ground(
            &p,
            &cfg,
            |b| core_via(&p, b),
            &ask("Victoria Falls"),
            "e-1",
            &mind,
            Some("en"),
            None,
        ))
        .unwrap();
        block_on(ground(
            &p,
            &cfg,
            |b| core_via(&p, b),
            &ask("Victoria Falls"),
            "e-1",
            &community,
            Some("en"),
            None,
        ))
        .unwrap();
        assert_eq!(p.requests.borrow().len(), 2);
        assert!(p
            .requests
            .borrow()
            .iter()
            .all(|r| host(&r.url) == "core.invalid"));
    }

    #[test]
    fn is_off_unless_the_setting_says_on() {
        for setting in [None, Some("off"), Some("true"), Some("")] {
            let p = platform();
            p.on_fn(|_| ok(r#"{"scope":"platform","hit":false,"chunks":[]}"#));
            let cfg = Config {
                own_sites: setting.map(str::to_owned),
                ..env()
            };
            let cloud = resolution(Scope::Platform, Destination::Cloud, false);
            block_on(ground(
                &p,
                &cfg,
                |b| core_via(&p, b),
                &ask("Victoria Falls"),
                "e-1",
                &cloud,
                Some("en"),
                None,
            ))
            .unwrap();
            assert!(
                p.requests
                    .borrow()
                    .iter()
                    .all(|r| host(&r.url) == "core.invalid"),
                "{setting:?}"
            );
        }
    }
}
