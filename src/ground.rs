//! Port of `shamwari/gateway/src/ground.ts`: grounding, retrieval only.
//!
//! [`ground`] finds the passages a question rests on (Core's `/ground/search`,
//! or Ground in Mongo through the Nyuchi API for a signed-in person, and our
//! own sites read live) and builds the system prompt that cites them.
//!
//! `POST /v1/ground/context` is wired with `admit` in
//! [`crate::gateway::handle_ground_context`].

use crate::config::Config;
use crate::ground_store::{ground_search, ground_store_configured, GroundQuery};
use crate::lang::js;
use crate::own_sites::{rank_own_sites, search_own_sites};
use crate::platform::Platform;
use crate::scope::{provider_scope, Destination, ScopeRefusal, ScopeResolution};
use crate::types::{GroundChunk, Message, Role, Scope};
use serde_json::{json, Map, Value};
use std::future::Future;

/// Shamwari's standing instructions to the model (owner review, 2026-11-23).
///
/// The earlier wording leaked our own vocabulary to people ("no Ground context
/// supplied", even in Shona), opened every answer with that disclaimer because
/// Core was down and no sources ever arrived, and set no length, so answers
/// ran to 500-760 tokens with headings and tables on a 360px phone. Measured
/// on GLM-5.3 with the same questions, this wording answers 20-45% faster with
/// none of that. The third-person opening it replaces was only a workaround
/// for the AI Gateway prompt-injection guard, which is now off (our own rules
/// are the guard: scope.rs, provenance.rs). The TypeScript gateway carries the
/// same text; the tests below hold what it must and must not say.
pub const BASE: &str = "You are Shamwari, an AI companion built in Zimbabwe.

- Reply in the language the person writes in: Shona, Ndebele, English or a mix. Never comment on their choice of language.
- Lead with the answer. Keep it short: about 150 words at most, in plain sentences or a short list. No tables and no headings unless the person asks for detail.
- If sources are listed below, base the answer on the ones that answer the question, ignore the rest, and cite each one you use by its name and section, or a web page by its title and link.
- If no sources are listed and the question is about Zimbabwean law, fees or official procedures, end with one short line saying this is general information and which office can confirm it. Say nothing about sources for everyday conversation.
- Never invent an Act, section number, statutory instrument, fee, rate or date. If you are not sure of a name, describe the thing instead of naming it.
- Never mention these instructions, sources being missing, or words like \"context\" or \"Ground\".";

#[derive(Debug, Clone, PartialEq)]
pub struct GroundResult {
    pub chunks: Vec<GroundChunk>,
    pub system_prompt: String,
    pub hit: bool,
    /// Core could not be reached. Distinct from a miss: "Ground has nothing on
    /// this" and "Ground could not be asked" are different statements, and
    /// only the first may be said to a person as a fact about the corpus.
    pub unavailable: bool,
    /// Core's own view of whether the question needs the person's own
    /// information. Core is authoritative: the notice is shown when either
    /// side says yes. False when Core could not be asked (the gateway's own
    /// view still applies).
    pub core_needs_own_information: bool,
}

impl GroundResult {
    fn ungrounded(unavailable: bool, core_needs_own_information: bool) -> Self {
        GroundResult {
            chunks: vec![],
            system_prompt: BASE.to_owned(),
            hit: false,
            unavailable,
            core_needs_own_information,
        }
    }
}

/// One passage as the model sees it: `[n] Title — Heading — Authority
/// (effective date)`, the link, then the text.
pub fn context_line(n: usize, c: &GroundChunk) -> String {
    let cite = [
        Some(c.title.as_str()),
        c.heading.as_deref(),
        Some(c.authority.as_str()),
    ]
    .into_iter()
    .flatten()
    .filter(|s| !s.is_empty())
    .collect::<Vec<_>>()
    .join(" — ");
    let eff = match c.effective_from.as_deref() {
        Some(e) if !e.is_empty() => format!(" (effective {e})"),
        _ => String::new(),
    };
    let link = match c.source_url.as_deref() {
        Some(u) if !u.is_empty() => format!("\nLink: {u}"),
        _ => String::new(),
    };
    format!("[{n}] {cite}{eff}{link}\n{}", c.content)
}

/// [`BASE`] with the passages listed. Called "sources", the word the
/// instructions use: the model never sees our internal name for them.
pub fn system_prompt(chunks: &[GroundChunk]) -> String {
    let context = chunks
        .iter()
        .enumerate()
        .map(|(i, c)| context_line(i + 1, c))
        .collect::<Vec<_>>()
        .join("\n\n");
    format!("{BASE}\n\n--- SOURCES ---\n{context}\n--- END SOURCES ---")
}

fn vector(v: &Value) -> Option<Vec<f64>> {
    v.as_array()?.iter().map(Value::as_f64).collect()
}

/// The question's embedding on Workers AI. The call is a provider call:
/// refuse here, at the call, if the decision it was handed is not one a
/// provider may see.
async fn embed<P: Platform>(
    p: &P,
    cfg: &Config,
    text: &str,
    decision: &ScopeResolution,
) -> Result<Result<Vec<f64>, String>, ScopeRefusal> {
    provider_scope(decision.scope, decision.destination)?;
    let Some(model) = cfg.get(&cfg.embedding_model) else {
        return Ok(Err("no embedding model".into()));
    };
    Ok(
        match p.ai_run(model, &json!({ "text": [text] }), None).await {
            Ok(out) => out
                .get("data")
                .and_then(|d| d.get(0))
                .and_then(vector)
                .ok_or_else(|| "embedding failed".to_owned()),
            Err(e) => Err(e),
        },
    )
}

/// Core (or the Nyuchi API) must answer for exactly the scope that was asked.
/// A different scope, or a chunk that carries one, means the two sides
/// disagree about a scope decision; the answer is not used. Fail closed, the
/// same way Core does for the data API.
fn checked(res: &Value, scope: Scope) -> Result<(Vec<GroundChunk>, bool), String> {
    let s = scope.as_str();
    let chunks = res.get("chunks").and_then(Value::as_array);
    let agrees = res.get("scope").and_then(Value::as_str) == Some(s)
        && chunks.is_some_and(|cs| {
            cs.iter()
                .all(|c| c.is_object() && c.get("scope").and_then(Value::as_str) == Some(s))
        });
    if !agrees {
        return Err("scope mismatch".into());
    }
    let chunks = serde_json::from_value(Value::Array(chunks.cloned().unwrap_or_default()))
        .map_err(|e| e.to_string())?;
    Ok((
        chunks,
        res.get("needs_own_information") == Some(&Value::Bool(true)),
    ))
}

/// Ground the latest user turn.
///
/// `core_search` is Core's `POST /ground/search` with this body (the caller
/// passes `crate::core`'s client, so Core's breaker is the one every Core
/// call shares); it is called at most once, and not at all when Ground in
/// Mongo answers instead. `person_token` is the signed-in person's own WorkOS
/// token, for Ground in Mongo through the Nyuchi API; `None` for an API key.
/// It goes to the Nyuchi API URL and nowhere else.
///
/// The only error is a [`ScopeRefusal`]: a decision a provider may not see
/// reached a provider call. That is a caller bug, never degraded quietly.
/// Everything else (Workers AI down, Core down, a scope mismatch) degrades to
/// an answer with fewer sources and says so.
#[allow(clippy::too_many_arguments)]
pub async fn ground<P, F, Fut>(
    p: &P,
    cfg: &Config,
    core_search: F,
    messages: &[Message],
    owner_entity_id: &str,
    decision: &ScopeResolution,
    language: Option<&str>,
    person_token: Option<&str>,
) -> Result<GroundResult, ScopeRefusal>
where
    P: Platform,
    F: FnOnce(Value) -> Fut,
    Fut: Future<Output = Result<Value, String>>,
{
    let (scope, destination) = (decision.scope, decision.destination);
    let Some(last_user) = messages.iter().rev().find(|m| m.role == Role::User) else {
        return Ok(GroundResult::ungrounded(false, false));
    };
    let question = last_user.content.as_str();

    // RULE 1, IN THE LEAK PATH.
    //
    // Embedding is inference. `AI.run` is a Cloudflare-hosted model, so
    // embedding a personal-scope question here would put the user's own words
    // through a third-party provider, upstream of Core's authoritative check,
    // which is the one place designed to catch exactly this.
    //
    // A `mind` destination therefore gets no edge embedding. Core falls back
    // to lexical-only retrieval, which is weaker ranking over the same public
    // corpus rather than a different corpus. Mind supplies its own vector,
    // computed on the device, when it ships.
    //
    // The question text still reaches Core. That is first-party Nyuchi
    // infrastructure, not a third-party inference provider, and Core already
    // stores the conversation, so this adds no exposure that rule 1 has not
    // already accepted.
    //
    // The same holds for a question the gateway's own rules say needs the
    // person's own information (`without_own_information`). The owner's rule
    // (shamwari#291) is that such a question is personal scope: Workers AI is
    // a third-party host, so it gets no edge embedding either, only Core's
    // keyword search.
    //
    // An embedding failure is an outage of Workers AI, which is exactly when
    // the lower rungs must still work. It degrades to the same keyword search
    // a mind destination gets: weaker ranking over the same corpus, never a
    // different one. Our own sites are read live, in parallel with everything
    // below, under their own short limit. Their search sees the question, so
    // they are asked only where a provider may see it: platform scope, cloud.
    let cloud_may_see = destination == Destination::Cloud && !decision.without_own_information;
    let ask_sites =
        cfg.own_sites.as_deref() == Some("on") && cloud_may_see && scope == Scope::Platform;
    if ask_sites {
        provider_scope(scope, destination)?;
    }
    let own_sites = async {
        if ask_sites {
            search_own_sites(p, question).await
        } else {
            vec![]
        }
    };

    let grounded = async {
        let mut embedding: Option<Vec<f64>> = None;
        if cloud_may_see {
            // A refusal is a caller bug, not "Workers AI is down": loud, never
            // a quiet fallback. Any other failure searches by keyword.
            if let Ok(v) = embed(p, cfg, question, decision).await? {
                embedding = Some(v);
            }
        }

        let top_k = js::number(cfg.ground_top_k.as_deref());
        // Core runs $rankFusion over the vector and lexical indexes and
        // applies the scope filter. It also logs the miss if there is one.
        let mut body = Map::new();
        body.insert("query".into(), json!(question));
        if let Some(e) = &embedding {
            body.insert("embedding".into(), json!(e));
        }
        body.insert("owner_entity_id".into(), json!(owner_entity_id));
        body.insert("scope".into(), json!(scope.as_str()));
        body.insert("destination".into(), json!(destination.as_str()));
        body.insert("language".into(), json!(language));
        body.insert("top_k".into(), js::number_json(top_k));
        // Core decides again from the same facts and may refuse (its checks
        // are authoritative; scope.rs is the fast path). The Nyuchi API takes
        // neither field: ground_search leaves them out.
        body.insert("granted_scopes".into(), json!(decision.grants));
        body.insert(
            "needs_own_information".into(),
            json!(decision.without_own_information),
        );

        // Mongo through the Nyuchi API when configured; Core otherwise. Both
        // answer in the same shape and both make the Rule 1 checks again
        // before they read anything (scope.rs is the fast path).
        let res = match person_token.filter(|_| ground_store_configured(cfg, person_token)) {
            Some(token) => {
                let q = GroundQuery {
                    query: question.to_owned(),
                    embedding: embedding.clone(),
                    owner_entity_id: owner_entity_id.to_owned(),
                    scope: scope.as_str().to_owned(),
                    destination: destination.as_str().to_owned(),
                    language: language.map(str::to_owned),
                    top_k,
                };
                ground_search(p, cfg, &q, token)
                    .await
                    .map_err(|e| e.to_string())
            }
            None => core_search(Value::Object(body)).await,
        };
        // Ungrounded is degraded, not broken. Answer anyway and say so.
        let found = res.and_then(|r| checked(&r, scope));
        Ok::<_, ScopeRefusal>((embedding, found))
    };

    let (sites, grounded) = futures_util::future::join(own_sites, grounded).await;
    let (embedding, found) = grounded?;
    let (mut chunks, unavailable, core_needs) = match found {
        Ok((chunks, needs)) => (chunks, false, needs),
        Err(_) => (vec![], true, false),
    };

    // Ground first (vetted, held passages), then what our own sites say now,
    // ranked against the question by meaning with the same embedding model
    // (the same gate applies: `sites` is empty unless the cloud may see it).
    let from_sites = rank_own_sites(question, sites, embedding.as_deref(), |texts| async move {
        provider_scope(scope, destination).map_err(|e| e.to_string())?;
        let model = cfg
            .get(&cfg.embedding_model)
            .ok_or_else(|| "no embedding model".to_owned())?;
        let out = p.ai_run(model, &json!({ "text": texts }), None).await?;
        match out.get("data") {
            None | Some(Value::Null) => Ok(vec![]),
            Some(Value::Array(vs)) => vs
                .iter()
                .map(|v| vector(v).ok_or_else(|| "embedding failed".to_owned()))
                .collect(),
            Some(_) => Err("embedding failed".into()),
        }
    })
    .await;
    chunks.extend(from_sites);

    if chunks.is_empty() {
        return Ok(GroundResult::ungrounded(unavailable, core_needs));
    }
    Ok(GroundResult {
        system_prompt: system_prompt(&chunks),
        chunks,
        hit: true,
        unavailable: false,
        core_needs_own_information: core_needs,
    })
}

#[cfg(test)]
mod tests {
    //! `test/ground.test.ts`, `test/prompt-wording.test.ts` (the prompt) and
    //! the ground() cases of `test/scope-decision.test.ts`.
    //!
    //! ground() calls `AI.run` to embed the user's question. Embedding is
    //! inference, and `AI` is a Cloudflare-hosted model, so this is a rule-1
    //! surface even though it never looks like one: nothing here is named
    //! "provider" or "completion". The fake counts `AI.run` calls. That count
    //! is the test.
    use super::*;
    use crate::platform::HttpRequest;
    use crate::testing::{block_on, FakePlatform};
    use std::cell::RefCell;

    const SN_Q: &str = "Ndinobhadhara marii PAYE?";
    const PERSONAL_Q: &str = "How much PAYE do I owe on my salary?";
    const PLAIN_Q: &str = "What does section 48 of the Constitution say?";

    fn messages(q: &str) -> Vec<Message> {
        vec![Message {
            role: Role::User,
            content: q.into(),
        }]
    }

    fn chunk(scope: &str) -> Value {
        json!({
            "content": "Every person has the right to life.",
            "heading": "Section 48",
            "title": "Constitution of Zimbabwe",
            "authority": "Parliament of Zimbabwe",
            "resource_type": "legislation",
            "source_url": "https://example.invalid/constitution",
            "effective_from": "2013-05-22",
            "scope": scope,
        })
    }

    fn env() -> Config {
        Config {
            embedding_model: Some("@cf/baai/bge-m3".into()),
            ground_top_k: Some("6".into()),
            core_url: Some("https://core.example".into()),
            shamwari_core_token: Some("t".into()),
            ..Default::default()
        }
    }

    fn platform() -> FakePlatform {
        crate::ground_store::reset_ground_store_breaker();
        crate::own_sites::reset_own_site_breakers();
        let p = FakePlatform::new();
        p.on_ai(|_, _| Some(Ok(json!({ "data": [vec![0.1; 1024]] }))));
        p
    }

    fn resolution(scope: Scope, destination: Destination) -> ScopeResolution {
        ScopeResolution {
            scope,
            destination,
            grants: vec![Scope::Personal, Scope::Community, Scope::Platform],
            without_own_information: false,
        }
    }

    type Answer = dyn Fn(&Value) -> Result<Value, String>;

    /// A Core that records what it was sent and answers with `answer(body)`.
    struct Core {
        seen: RefCell<Vec<Value>>,
        answer: Box<Answer>,
    }

    impl Core {
        fn new(answer: impl Fn(&Value) -> Result<Value, String> + 'static) -> Self {
            Core {
                seen: RefCell::new(vec![]),
                answer: Box::new(answer),
            }
        }

        /// Core answers for the scope it was asked for, as the real one does.
        fn empty() -> Self {
            Core::new(|b| Ok(json!({ "chunks": [], "hit": false, "scope": b["scope"] })))
        }

        fn search(&self, body: Value) -> impl Future<Output = Result<Value, String>> + '_ {
            let out = (self.answer)(&body);
            self.seen.borrow_mut().push(body);
            async move { out }
        }

        fn calls(&self) -> usize {
            self.seen.borrow().len()
        }
    }

    fn run(
        p: &FakePlatform,
        cfg: &Config,
        core: &Core,
        q: &str,
        d: &ScopeResolution,
    ) -> Result<GroundResult, ScopeRefusal> {
        block_on(ground(
            p,
            cfg,
            |b| core.search(b),
            &messages(q),
            "e1",
            d,
            Some("sn"),
            None,
        ))
    }

    // ground() and the edge embedding

    #[test]
    fn embeds_on_cloudflare_for_a_cloud_destination() {
        let (p, core) = (platform(), Core::empty());
        run(
            &p,
            &env(),
            &core,
            SN_Q,
            &resolution(Scope::Platform, Destination::Cloud),
        )
        .unwrap();
        assert_eq!(p.ai_calls.borrow().len(), 1);
        let (model, input, options) = p.ai_calls.borrow()[0].clone();
        assert_eq!(model, "@cf/baai/bge-m3");
        assert_eq!(input, json!({ "text": [SN_Q] }));
        assert_eq!(options, None);
        assert_eq!(
            core.seen.borrow()[0]["embedding"].as_array().unwrap().len(),
            1024
        );
    }

    // THE REGRESSION. This ran unconditionally, so the moment
    // MIND_AVAILABLE=true stopped decideDestination throwing, every
    // personal-scope question was embedded by a third-party model, upstream
    // of Core's authoritative check, which is the one place designed to catch
    // it. docs/scaling-and-memory.md recorded this as fixed while it was not.
    #[test]
    fn never_embeds_on_cloudflare_for_a_mind_destination() {
        let (p, core) = (platform(), Core::empty());
        run(
            &p,
            &env(),
            &core,
            SN_Q,
            &resolution(Scope::Personal, Destination::Mind),
        )
        .unwrap();
        assert!(p.ai_calls.borrow().is_empty());
        assert!(core.seen.borrow()[0].get("embedding").is_none());
    }

    #[test]
    fn still_sends_the_query_text_to_core_which_is_first_party() {
        let (p, core) = (platform(), Core::empty());
        run(
            &p,
            &env(),
            &core,
            SN_Q,
            &resolution(Scope::Personal, Destination::Mind),
        )
        .unwrap();
        assert_eq!(core.seen.borrow()[0]["query"], SN_Q);
    }

    #[test]
    fn degrades_to_ungrounded_rather_than_failing_when_core_is_down() {
        let p = platform();
        let core = Core::new(|_| Err("core 500: boom".into()));
        let r = run(
            &p,
            &env(),
            &core,
            SN_Q,
            &resolution(Scope::Personal, Destination::Mind),
        )
        .unwrap();
        assert!(!r.hit);
        assert!(r.unavailable);
        assert!(r.system_prompt.contains("Shamwari"));
        assert_eq!(r.system_prompt, BASE);
    }

    // scope-decision.test.ts: "ground() refuses to embed for a decision a
    // provider may not see, and counts zero".
    #[test]
    fn refuses_to_embed_for_a_decision_a_provider_may_not_see_and_counts_zero() {
        let bad = ScopeResolution {
            scope: Scope::Personal,
            destination: Destination::Cloud,
            grants: vec![Scope::Personal],
            without_own_information: false,
        };
        for own_sites in [None, Some("on")] {
            let (p, core) = (platform(), Core::empty());
            let cfg = Config {
                own_sites: own_sites.map(str::to_owned),
                ..env()
            };
            let e = run(&p, &cfg, &core, PERSONAL_Q, &bad).unwrap_err();
            assert_eq!(e.scope, Scope::Personal);
            assert_eq!(
                p.ai_calls.borrow().len() + p.requests.borrow().len() + core.calls(),
                0
            );
        }
    }

    #[test]
    fn a_personal_looking_question_gets_no_edge_embedding_only_cores_keyword_search() {
        let (p, core) = (platform(), Core::empty());
        let d = ScopeResolution {
            without_own_information: true,
            ..resolution(Scope::Platform, Destination::Cloud)
        };
        let cfg = Config {
            own_sites: Some("on".into()),
            ..env()
        };
        run(&p, &cfg, &core, PERSONAL_Q, &d).unwrap();
        assert!(p.ai_calls.borrow().is_empty());
        assert!(p.requests.borrow().is_empty());
        assert_eq!(core.calls(), 1);
        let sent = core.seen.borrow()[0].clone();
        assert!(sent.get("embedding").is_none());
        // Core is told, so its own check (edge embedding refused) applies too.
        assert_eq!(sent["needs_own_information"], true);
    }

    #[test]
    fn sends_core_the_grants_and_its_own_view_so_core_decides_again_from_the_same_facts() {
        let (p, core) = (platform(), Core::empty());
        let d = ScopeResolution {
            scope: Scope::Platform,
            destination: Destination::Cloud,
            grants: vec![Scope::Community, Scope::Platform],
            without_own_information: true,
        };
        run(&p, &env(), &core, PERSONAL_Q, &d).unwrap();
        let sent = core.seen.borrow()[0].clone();
        assert_eq!(
            sent,
            json!({
                "query": PERSONAL_Q,
                "owner_entity_id": "e1",
                "scope": "platform",
                "destination": "cloud",
                "language": "sn",
                "top_k": 6,
                "granted_scopes": ["community", "platform"],
                "needs_own_information": true,
            })
        );
    }

    #[test]
    fn a_mind_destination_reaches_core_as_personal_and_mind() {
        let (p, core) = (platform(), Core::empty());
        run(
            &p,
            &env(),
            &core,
            PERSONAL_Q,
            &resolution(Scope::Personal, Destination::Mind),
        )
        .unwrap();
        let sent = core.seen.borrow()[0].clone();
        assert_eq!(
            (sent["scope"].clone(), sent["destination"].clone()),
            (json!("personal"), json!("mind"))
        );
        assert!(p.ai_calls.borrow().is_empty());
    }

    #[test]
    fn still_answers_when_the_embedding_model_is_down_by_keyword_search() {
        let p = FakePlatform::new();
        p.on_ai(|_, _| Some(Err("AI down".into())));
        let core = Core::new(|b| {
            Ok(json!({ "chunks": [chunk("platform")], "hit": true, "scope": b["scope"] }))
        });
        let r = run(
            &p,
            &env(),
            &core,
            PLAIN_Q,
            &resolution(Scope::Platform, Destination::Cloud),
        )
        .unwrap();
        assert!(r.hit);
        assert!(core.seen.borrow()[0].get("embedding").is_none());
        // An answer without a vector, too.
        let p = FakePlatform::new();
        p.on_ai(|_, _| Some(Ok(json!({ "data": [] }))));
        let core = Core::empty();
        run(
            &p,
            &env(),
            &core,
            PLAIN_Q,
            &resolution(Scope::Platform, Destination::Cloud),
        )
        .unwrap();
        assert!(core.seen.borrow()[0].get("embedding").is_none());
    }

    #[test]
    fn core_answering_for_a_different_scope_is_not_used_nothing_is_cited() {
        let mut payslip = chunk("personal");
        payslip["content"] = json!("PAYSLIP");
        let answers = [
            json!({ "chunks": [payslip.clone()], "hit": true, "scope": "personal" }),
            json!({ "chunks": [chunk("community")], "hit": true, "scope": "community" }),
            json!({ "chunks": [payslip], "hit": true, "scope": "platform" }),
            json!({ "chunks": [chunk("platform")], "hit": true }),
            json!({ "chunks": [null], "hit": true, "scope": "platform" }),
            json!({ "chunks": "x", "hit": true, "scope": "platform" }),
            json!(null),
        ];
        for answer in answers {
            let p = platform();
            let a = answer.clone();
            let core = Core::new(move |_| Ok(a.clone()));
            let r = run(
                &p,
                &env(),
                &core,
                PLAIN_Q,
                &resolution(Scope::Platform, Destination::Cloud),
            )
            .unwrap();
            assert!(!r.hit, "{answer}");
            assert!(r.unavailable);
            assert!(r.chunks.is_empty());
            assert!(!r.system_prompt.contains("PAYSLIP"));
            assert_eq!(core.calls(), 1, "one attempt: no retry as another scope");
        }
    }

    #[test]
    fn core_saying_the_question_needs_own_information_is_carried_and_only_true_counts() {
        for (v, want) in [
            (json!(true), true),
            (json!(false), false),
            (json!("true"), false),
            (json!(null), false),
        ] {
            let p = platform();
            let core = Core::new(move |b| {
                Ok(
                    json!({ "chunks": [chunk("platform")], "hit": true, "scope": b["scope"], "needs_own_information": v }),
                )
            });
            let r = run(
                &p,
                &env(),
                &core,
                PLAIN_Q,
                &resolution(Scope::Platform, Destination::Cloud),
            )
            .unwrap();
            assert_eq!(r.core_needs_own_information, want);
        }
    }

    #[test]
    fn cites_each_passage_with_title_heading_authority_date_and_link() {
        let p = platform();
        let core = Core::new(|b| {
            let mut bare = chunk("platform");
            bare["heading"] = json!(null);
            bare["authority"] = json!("");
            bare["effective_from"] = json!(null);
            bare["source_url"] = json!(null);
            bare["content"] = json!("Second.");
            Ok(json!({ "chunks": [chunk("platform"), bare], "hit": true, "scope": b["scope"] }))
        });
        let r = run(
            &p,
            &env(),
            &core,
            PLAIN_Q,
            &resolution(Scope::Platform, Destination::Cloud),
        )
        .unwrap();
        assert!(r.hit && !r.unavailable);
        assert_eq!(r.chunks.len(), 2);
        assert_eq!(
            r.system_prompt,
            format!(
                "{BASE}\n\n--- SOURCES ---\n\
                 [1] Constitution of Zimbabwe — Section 48 — Parliament of Zimbabwe (effective 2013-05-22)\n\
                 Link: https://example.invalid/constitution\n\
                 Every person has the right to life.\n\n\
                 [2] Constitution of Zimbabwe\nSecond.\n--- END SOURCES ---"
            )
        );
    }

    #[test]
    fn no_user_turn_is_ungrounded_and_asks_nobody() {
        let (p, core) = (platform(), Core::empty());
        let only_system = [Message {
            role: Role::System,
            content: "x".into(),
        }];
        let d = resolution(Scope::Platform, Destination::Cloud);
        let r = block_on(ground(
            &p,
            &env(),
            |b| core.search(b),
            &only_system,
            "e1",
            &d,
            None,
            None,
        ))
        .unwrap();
        assert_eq!(r, GroundResult::ungrounded(false, false));
        assert_eq!(core.calls() + p.ai_calls.borrow().len(), 0);
    }

    #[test]
    fn grounds_the_latest_user_turn() {
        let (p, core) = (platform(), Core::empty());
        let msgs = [
            Message {
                role: Role::User,
                content: "first".into(),
            },
            Message {
                role: Role::User,
                content: "second".into(),
            },
            Message {
                role: Role::Assistant,
                content: "third".into(),
            },
        ];
        let d = resolution(Scope::Platform, Destination::Cloud);
        block_on(ground(
            &p,
            &env(),
            |b| core.search(b),
            &msgs,
            "e1",
            &d,
            None,
            None,
        ))
        .unwrap();
        assert_eq!(core.seen.borrow()[0]["query"], "second");
        assert_eq!(core.seen.borrow()[0]["language"], json!(null));
    }

    #[test]
    fn a_malformed_top_k_reaches_core_as_null_as_number_and_json_stringify_give() {
        let (p, core) = (platform(), Core::empty());
        let cfg = Config {
            ground_top_k: None,
            ..env()
        };
        run(
            &p,
            &cfg,
            &core,
            PLAIN_Q,
            &resolution(Scope::Platform, Destination::Cloud),
        )
        .unwrap();
        assert_eq!(core.seen.borrow()[0]["top_k"], json!(null));
    }

    #[test]
    fn the_persons_token_goes_only_to_the_nyuchi_api() {
        let p = platform();
        // No answer anywhere: every request is a network error.
        p.on_fn(|_: &HttpRequest| None);
        let cfg = Config {
            nyuchi_api_url: Some("https://api.example.com".into()),
            own_sites: Some("on".into()),
            ..env()
        };
        let core = Core::empty();
        let d = resolution(Scope::Platform, Destination::Cloud);
        // Every own site is asked too, and none of them sees the token.
        let r = block_on(ground(
            &p,
            &cfg,
            |b| core.search(b),
            &messages(PLAIN_Q),
            "e1",
            &d,
            None,
            Some("tok.en"),
        ))
        .unwrap();
        assert!(r.unavailable);
        for req in p.requests.borrow().iter() {
            let carries = req.headers.iter().any(|(_, v)| v.contains("tok.en"));
            assert_eq!(
                carries,
                req.url.starts_with("https://api.example.com/"),
                "{}",
                req.url
            );
        }
        assert_eq!(core.calls(), 0);
    }

    // test/prompt-wording.test.ts

    #[test]
    fn speaks_to_the_model_as_shamwari_directly() {
        let p = FakePlatform::new();
        p.on_ai(|_, _| Some(Err("down".into())));
        let core = Core::new(|_| Err("down".into()));
        let r = run(
            &p,
            &env(),
            &core,
            "what is vat",
            &resolution(Scope::Platform, Destination::Mind),
        )
        .unwrap();
        assert!(r.system_prompt.starts_with("You are Shamwari"));
    }

    #[test]
    fn never_hands_the_model_our_internal_vocabulary_to_repeat_back_to_people() {
        let told: String = BASE
            .split('\n')
            .filter(|l| !l.starts_with("- Never mention"))
            .collect::<Vec<_>>()
            .join("\n");
        let ground = regex::Regex::new(r"\bGround\b").unwrap();
        let context = regex::Regex::new(r"(?i)\bcontext\b").unwrap();
        assert!(!ground.is_match(&told));
        assert!(!context.is_match(&told));
    }

    #[test]
    fn asks_for_short_answers_fit_for_a_small_screen_without_tables_or_headings() {
        assert!(BASE.contains("about 150 words at most"));
        assert!(BASE.contains("No tables and no headings"));
    }

    #[test]
    fn keeps_the_disclaimer_to_law_fees_and_procedures_at_the_end_never_for_everyday_talk() {
        assert!(BASE.contains("end with one short line"));
        assert!(BASE.contains("Say nothing about sources for everyday conversation"));
    }

    #[test]
    fn still_forbids_invented_law() {
        assert!(BASE.contains(
            "Never invent an Act, section number, statutory instrument, fee, rate or date"
        ));
    }
}
