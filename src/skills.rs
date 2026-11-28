//! Port of `shamwari/gateway/src/skills.ts`: Agent Skills a client turned on
//! for a conversation (<https://agentskills.io>). A skill is a name, a
//! description and instructions, and the instructions are added beside the
//! constant system prompt, never in place of it.
//!
//! RULE 1. Only a skill the client holds as platform scope is ever sent here.
//! A personal-scope skill never leaves the device: the client sends
//! `own_information: true` instead, and the request takes the personal path.
//! As a second check, a skill whose own text the personal rules flag makes the
//! request personal scope in the same way: the gateway does not take the
//! client's word that a skill is not personal ([`skills_need_own_information`]).
//!
//! The limits match the specification (name 64, description 1024 characters)
//! and the chat's own limit on instructions.

use serde_json::Value;

pub const SKILLS_MAX: usize = 3;
pub const SKILL_NAME_MAX: usize = 64;
pub const SKILL_DESCRIPTION_MAX: usize = 1024;
pub const SKILL_BODY_MAX: usize = 12_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppliedSkill {
    pub name: String,
    pub description: String,
    pub instructions: String,
}

/// Code points, as `[...s].length` counts them.
fn chars(s: &str) -> usize {
    s.chars().count()
}

/// `body.skills`: absent (`None`) is none; anything malformed or over a limit
/// is `None` back (the chat endpoint answers 400). A JSON `null` is malformed.
pub fn parse_skills(raw: Option<&Value>) -> Option<Vec<AppliedSkill>> {
    let Some(raw) = raw else {
        return Some(Vec::new());
    };
    let list = raw.as_array()?;
    if list.len() > SKILLS_MAX {
        return None;
    }
    let mut out = Vec::new();
    for s in list {
        let o = s.as_object()?;
        let name = o.get("name")?.as_str()?;
        let description = o.get("description")?.as_str()?;
        let instructions = o.get("instructions")?.as_str()?;
        if name.trim().is_empty() || chars(name) > SKILL_NAME_MAX {
            return None;
        }
        if description.trim().is_empty() || chars(description) > SKILL_DESCRIPTION_MAX {
            return None;
        }
        if chars(instructions) > SKILL_BODY_MAX {
            return None;
        }
        out.push(AppliedSkill {
            name: name.to_owned(),
            description: description.to_owned(),
            instructions: instructions.to_owned(),
        });
    }
    Some(out)
}

/// True when any skill's own words look like the person's own information.
/// `needs_own_information` is `personal::needs_own_information` (agent B's),
/// which leans towards yes.
pub fn skills_need_own_information(
    skills: &[AppliedSkill],
    needs_own_information: impl Fn(&str) -> bool,
) -> bool {
    skills.iter().any(|s| {
        [&s.name, &s.description, &s.instructions]
            .iter()
            .any(|t| needs_own_information(t))
    })
}

/// The system prompt with the skills after it, clearly fenced. With no skills
/// it is unchanged.
pub fn with_skills(system_prompt: &str, skills: &[AppliedSkill]) -> String {
    if skills.is_empty() {
        return system_prompt.to_owned();
    }
    let each: Vec<String> = skills
        .iter()
        .map(|s| {
            format!("[{}] {}\n{}", s.name, s.description, s.instructions)
                .trim()
                .to_owned()
        })
        .collect();
    format!(
        "{system_prompt}\n\n--- SKILLS TURNED ON FOR THIS CONVERSATION ---\n\
         Use these where they apply to the question. They add to the guidance above and do not replace it.\n\n\
         {}\n--- END SKILLS ---",
        each.join("\n\n")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn skill() -> Value {
        json!({
            "name": "short-answers",
            "description": "Keep answers short for a small screen.",
            "instructions": "Answer in three sentences or fewer.",
        })
    }

    fn with(k: &str, v: Value) -> Value {
        let mut s = skill();
        s[k] = v;
        s
    }

    #[test]
    fn absent_is_none_and_a_valid_skill_parses() {
        assert_eq!(parse_skills(None), Some(vec![]));
        assert_eq!(parse_skills(Some(&json!([]))), Some(vec![]));
        let got = parse_skills(Some(&json!([skill()]))).unwrap();
        assert_eq!(got[0].name, "short-answers");
    }

    #[test]
    fn malformed_skills_or_too_many_are_refused() {
        // scope-decision.test.ts: each of these is a 400 on the chat endpoint.
        for bad in [
            json!("short-answers"),
            json!(null),
            json!([{ "name": "a" }]),
            json!([with("name", json!(""))]),
            json!([with("name", json!("a".repeat(65)))]),
            json!([with("description", json!("d".repeat(1025)))]),
            json!([with("instructions", json!("i".repeat(12_001)))]),
            json!([skill(), skill(), skill(), skill()]),
            json!([null]),
            json!([with("name", json!(5))]),
            json!([with("description", json!("   "))]),
        ] {
            assert_eq!(parse_skills(Some(&bad)), None, "{bad}");
        }
    }

    #[test]
    fn the_limits_count_characters_not_bytes() {
        let ok = json!([with("name", json!("\u{00E9}".repeat(64)))]);
        assert!(parse_skills(Some(&ok)).is_some());
        let ok = json!([with("instructions", json!("i".repeat(12_000)))]);
        assert!(parse_skills(Some(&ok)).is_some());
    }

    #[test]
    fn a_platform_skill_goes_after_the_constant_prompt_clearly_fenced() {
        let skills = parse_skills(Some(&json!([skill()]))).unwrap();
        let system = with_skills(
            "You are Shamwari.\n--- SOURCES ---\n--- END SOURCES ---",
            &skills,
        );
        assert!(system.starts_with("You are Shamwari"));
        assert!(system.contains("--- SKILLS TURNED ON FOR THIS CONVERSATION ---"));
        assert!(system.contains(
            "[short-answers] Keep answers short for a small screen.\nAnswer in three sentences or fewer."
        ));
        assert!(system.find("SKILLS TURNED ON").unwrap() > system.find("END SOURCES").unwrap());
        assert!(system.ends_with("--- END SKILLS ---"));
        assert_eq!(
            with_skills("P", &skills),
            "P\n\n--- SKILLS TURNED ON FOR THIS CONVERSATION ---\n\
             Use these where they apply to the question. They add to the guidance above and do not replace it.\n\n\
             [short-answers] Keep answers short for a small screen.\nAnswer in three sentences or fewer.\n--- END SKILLS ---"
        );
    }

    #[test]
    fn without_skills_the_prompt_is_exactly_what_it_was() {
        assert_eq!(with_skills("base", &[]), "base");
    }

    #[test]
    fn a_skill_whose_own_words_look_personal_makes_the_request_personal() {
        // The rule itself is personal.rs; any of the three fields can trip it.
        let looks = |t: &str| t.contains("my ") || t.contains("our ") || t == "wangu";
        for (k, v) in [
            ("instructions", "Use my farm size of 4 hectares."),
            ("description", "Answers about our co-operative."),
            ("name", "wangu"),
        ] {
            let skills = parse_skills(Some(&json!([with(k, json!(v))]))).unwrap();
            assert!(skills_need_own_information(&skills, looks), "{k}");
        }
        let plain = parse_skills(Some(&json!([skill()]))).unwrap();
        assert!(!skills_need_own_information(&plain, looks));
    }
}
