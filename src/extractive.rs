//! Port of `shamwari/gateway/src/extractive.ts`: the extractive rung of the
//! degradation ladder (docs/tiered-access-design.md).
//!
//! Retrieval alone answers: the cited passage, with no generation. It costs
//! no inference, so it is the cheapest rung, and it is what remains when
//! every inference path is down. It cannot invent a citation, because nothing
//! here writes text of its own beyond a fixed label: every word of the answer
//! is a field of a chunk Ground returned.
//!
//! It does not judge relevance. Ground returns the nearest passages whether
//! or not they answer the question, so the citation line says "Closest
//! source", which is true, rather than "Source", which would claim the
//! passage answers it. A relevance threshold needs a score from Ground and an
//! eval set to calibrate it; see docs/tiered-access-design.md.
//!
//! Because no model wrote it, this text has no teacher model and no licence
//! class, and Ground chunks carry their own `mind_eligible` terms that the
//! gateway does not see here. So an extractive answer is never stored as a
//! conversation and never marked trainable: it is metered and nothing more.
//!
//! Lengths are JavaScript's (UTF-16 code units), so a cap means what it means
//! in the TypeScript Worker, and an SMS answer is cut at the same place.

use crate::lang::js;
use crate::types::GroundChunk;
use regex::Regex;
use std::sync::OnceLock;

/// Under three GSM-7 SMS segments (3 x 153 = 459). Per-channel caps come later.
pub const EXTRACTIVE_MAX_CHARS: usize = 450;

/// Said plainly, as the system prompt requires, when Ground has nothing.
pub const NO_SOURCE_MESSAGE: &str =
    "I could not find this in the sources I hold, so I will not guess. Please check with the relevant authority.";

#[derive(Debug, Clone, PartialEq)]
pub struct ExtractiveAnswer {
    pub text: String,
    /// The chunks the text was built from, in order.
    pub used: Vec<GroundChunk>,
    /// A passage was cut short. The full text is at the citation's source.
    pub truncated: bool,
}

/// Core fills this in when a field is missing. It names nothing, so it is not
/// cited.
const PLACEHOLDER_AUTHORITY: &str = "Unknown";

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ExtractiveOptions {
    /// Replace typographic characters (curly quotes, en and em dashes, the
    /// ellipsis, non-breaking spaces) with their plain ASCII forms. One such
    /// character forces an SMS into UCS-2, at 70 characters a segment instead
    /// of 160, so a "three segment" answer would become seven. This does not
    /// make text GSM-7 clean (an accented letter still is not); it removes the
    /// characters this module and typeset sources introduce. The words are
    /// never changed.
    pub plain_text: bool,
}

pub fn plain(text: &str) -> String {
    text.chars()
        .map(|c| match c {
            '\u{2018}' | '\u{2019}' | '\u{201B}' => "'".to_owned(),
            '\u{201C}' | '\u{201D}' => "\"".to_owned(),
            '\u{2013}' | '\u{2014}' => "-".to_owned(),
            '\u{2026}' => "...".to_owned(),
            '\u{00A0}' => " ".to_owned(),
            c => c.to_string(),
        })
        .collect()
}

fn ellipsis(opts: ExtractiveOptions) -> &'static str {
    if opts.plain_text {
        "..."
    } else {
        "\u{2026}"
    }
}

fn citation(c: &GroundChunk, max: usize, opts: ExtractiveOptions) -> String {
    let where_ = [Some(c.title.as_str()), c.heading.as_deref()]
        .into_iter()
        .flatten()
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(" — ");
    let by = if !c.authority.is_empty() && c.authority != PLACEHOLDER_AUTHORITY {
        format!(" ({})", c.authority)
    } else {
        String::new()
    };
    let eff = match c.effective_from.as_deref() {
        Some(e) if !e.is_empty() => format!(", effective {e}"),
        _ => String::new(),
    };
    let text = format!("Closest source: {where_}{by}{eff}");
    let text = if opts.plain_text { plain(&text) } else { text };
    clip(&text, max, ellipsis(opts)).0
}

/// Cut on a word boundary, never mid-word, and say so with an ellipsis. Line
/// breaks are kept: they carry the numbering of subsections, and losing them
/// can merge a rule into its exception. The answer and whether it was cut.
pub fn clip(text: &str, max: usize, ellipsis: &str) -> (String, bool) {
    static SPACES: OnceLock<Regex> = OnceLock::new();
    static BREAKS: OnceLock<Regex> = OnceLock::new();
    let spaces = SPACES.get_or_init(|| Regex::new(r"[ \t]+").unwrap());
    let breaks = BREAKS.get_or_init(|| Regex::new(r"\n{3,}").unwrap());
    let t = spaces.replace_all(js::trim(text), " ");
    let t = breaks.replace_all(&t, "\n\n");
    if js::len(&t) <= max {
        return (t.into_owned(), false);
    }
    let units: Vec<u16> = t.encode_utf16().collect();
    let cut = &units[..max.saturating_sub(js::len(ellipsis)).min(units.len())];
    let last = |u: u16| cut.iter().rposition(|&c| c == u);
    let at = last(b' ' as u16).max(last(b'\n' as u16));
    let kept = match at {
        Some(at) if at as f64 > max as f64 * 0.5 => &cut[..at],
        _ => cut,
    };
    let kept = String::from_utf16_lossy(kept);
    (format!("{}{ellipsis}", js::trim_end(&kept)), true)
}

/// Build an answer from the best chunk, adding further chunks only while they
/// fit. The first chunk is always included, clipped if it must be, together
/// with its citation: an answer with no source is never produced. The whole
/// answer, citation included, stays within `max_chars`
/// ([`EXTRACTIVE_MAX_CHARS`] unless a channel says otherwise).
pub fn extractive_answer(
    chunks: &[GroundChunk],
    max_chars: usize,
    opts: ExtractiveOptions,
) -> Option<ExtractiveAnswer> {
    let first = chunks.first()?;
    let ellipsis = ellipsis(opts);

    // The citation may take at most 40% of the budget, so a very long title
    // cannot push the answer over the limit or crowd out the passage.
    let cite_max = (max_chars as f64 * 0.4).floor() as usize;
    let cite = citation(first, cite_max, opts);
    let room = 20.max(max_chars as i64 - js::len(&cite) as i64 - 1) as usize;
    let body = |c: &GroundChunk| {
        let content = if opts.plain_text {
            plain(&c.content)
        } else {
            c.content.clone()
        };
        clip(&content, room, ellipsis)
    };

    let (head, mut truncated) = body(first);
    let mut parts = vec![format!("{head}\n{cite}")];
    let mut used = vec![first.clone()];

    for c in &chunks[1..] {
        let (b, cut) = body(c);
        let next = format!("{b}\n{}", citation(c, cite_max, opts));
        let joined_len = parts.iter().map(|p| js::len(p) + 2).sum::<usize>() + js::len(&next);
        if joined_len > max_chars {
            break;
        }
        parts.push(next);
        used.push(c.clone());
        truncated = truncated || cut;
    }

    Some(ExtractiveAnswer {
        text: parts.join("\n\n"),
        used,
        truncated,
    })
}

#[cfg(test)]
mod tests {
    //! `test/extractive.test.ts`. The property that matters is that nothing
    //! in the answer is written by the gateway beyond a fixed label: every
    //! word is a field of a chunk Ground returned, so it cannot invent a
    //! citation, a section number or a rate.
    use super::*;

    fn chunk() -> GroundChunk {
        GroundChunk {
            content: "Every person has the right to life.".into(),
            heading: Some("Section 48 — Right to life".into()),
            title: "Constitution of Zimbabwe Amendment (No. 20) Act 2013".into(),
            authority: "Parliament of Zimbabwe".into(),
            resource_type: "legislation".into(),
            source_url: Some("https://example.invalid/constitution".into()),
            effective_from: Some("2013-05-22".into()),
            scope: "platform".into(),
        }
    }

    fn with_content(content: &str) -> GroundChunk {
        GroundChunk {
            content: content.into(),
            ..chunk()
        }
    }

    fn answer(chunks: &[GroundChunk]) -> Option<ExtractiveAnswer> {
        extractive_answer(chunks, EXTRACTIVE_MAX_CHARS, ExtractiveOptions::default())
    }

    fn text(chunks: &[GroundChunk]) -> String {
        answer(chunks).map(|a| a.text).unwrap_or_default()
    }

    #[test]
    fn returns_nothing_when_ground_returned_nothing_rather_than_an_answer_with_no_source() {
        assert_eq!(answer(&[]), None);
    }

    #[test]
    fn is_the_passage_plus_its_citation_built_from_the_chunk_and_nothing_else() {
        let a = answer(&[chunk()]).unwrap();
        assert_eq!(
            a.text,
            "Every person has the right to life.\n\
             Closest source: Constitution of Zimbabwe Amendment (No. 20) Act 2013 — Section 48 — Right to life \
             (Parliament of Zimbabwe), effective 2013-05-22"
        );
        assert_eq!(a.used.len(), 1);
    }

    #[test]
    fn never_states_a_section_rate_or_date_that_is_not_in_the_chunk() {
        let bare = GroundChunk {
            title: "Income Tax Act".into(),
            heading: None,
            authority: String::new(),
            effective_from: None,
            ..chunk()
        };
        let t = text(std::slice::from_ref(&bare));
        assert!(!t.contains("Section") && !t.contains("effective") && !t.contains('('));
        assert!(t.contains(&bare.title));
    }

    #[test]
    fn keeps_the_first_chunk_and_its_citation_even_when_it_must_be_clipped() {
        let a = answer(&[with_content(&"word ".repeat(2000))]).unwrap();
        assert!(js::len(&a.text) <= EXTRACTIVE_MAX_CHARS);
        assert!(a.text.contains("Closest source: "));
        assert!(a.text.contains('\u{2026}'));
        assert!(a.truncated);
    }

    #[test]
    fn clips_on_a_word_boundary_never_mid_word() {
        let a = extractive_answer(
            &[with_content(&"alpha beta gamma delta ".repeat(100))],
            120,
            ExtractiveOptions::default(),
        )
        .unwrap();
        let body = a.text.split('\n').next().unwrap();
        assert!(body.ends_with('\u{2026}'));
        let last = body
            .trim_end_matches('\u{2026}')
            .split(' ')
            .next_back()
            .unwrap();
        assert!(
            ["alpha", "beta", "gamma", "delta"].contains(&last),
            "{last:?}"
        );
    }

    #[test]
    fn adds_a_further_chunk_only_while_it_fits_and_reports_which_it_used() {
        let three = [chunk(), with_content("Second."), with_content("Third.")];
        let a = extractive_answer(&three, 10_000, ExtractiveOptions::default()).unwrap();
        assert_eq!(a.used.len(), 3);
        let one = answer(&[chunk()]).unwrap();
        let tight = extractive_answer(
            &[chunk(), with_content("Second.")],
            js::len(&one.text) + 5,
            ExtractiveOptions::default(),
        )
        .unwrap();
        assert_eq!(tight.used.len(), 1);
        assert!(!tight.text.contains("Second."));
    }

    #[test]
    fn stays_within_the_cap_even_when_the_title_is_enormous() {
        let huge = GroundChunk {
            title: "T".repeat(2000),
            heading: Some("H".repeat(2000)),
            ..chunk()
        };
        let a = answer(&[huge]).unwrap();
        assert!(js::len(&a.text) <= EXTRACTIVE_MAX_CHARS);
        assert!(a.text.contains("Closest source: "));
    }

    #[test]
    fn does_not_cite_the_placeholder_core_fills_in_for_a_missing_authority() {
        let t = text(&[GroundChunk {
            authority: "Unknown".into(),
            ..chunk()
        }]);
        assert!(!t.contains("Unknown"));
    }

    #[test]
    fn says_whether_a_passage_was_cut_and_keeps_line_breaks_that_number_subsections() {
        let short = answer(&[with_content("(1) A person may.\n(2) Except where barred.")]).unwrap();
        assert!(!short.truncated);
        assert!(short
            .text
            .contains("(1) A person may.\n(2) Except where barred."));
    }

    #[test]
    fn labels_the_source_as_the_closest_one_not_as_proof_it_answers_the_question() {
        let t = text(&[chunk()]);
        assert!(t.contains("Closest source:"));
        assert!(!t.starts_with("Source:") && !t.contains("\nSource:"));
    }

    #[test]
    fn can_keep_its_own_marker_plain_too_and_still_within_the_cap() {
        let a = extractive_answer(
            &[with_content(&"alpha beta gamma delta ".repeat(100))],
            120,
            ExtractiveOptions { plain_text: true },
        )
        .unwrap();
        assert!(js::len(&a.text) <= 120);
        assert!(a.text.contains("..."));
        assert!(!a.text.contains('\u{2026}'));
        // The citation's own dashes are plain too.
        assert!(!a.text.contains('—'));
    }

    #[test]
    fn counts_length_as_javascript_does_in_utf_16_code_units() {
        // Each emoji is two UTF-16 units: a cap of 450 holds fewer of them.
        let a = answer(&[with_content(&"\u{1F600} ".repeat(400))]).unwrap();
        assert!(js::len(&a.text) <= EXTRACTIVE_MAX_CHARS);
        assert!(a.truncated);
        assert_eq!(
            clip("  a \t b\n\n\n\nc ", 99, "\u{2026}"),
            ("a b\n\nc".to_owned(), false)
        );
        assert_eq!(clip("abcdef", 4, "..."), ("a...".to_owned(), true));
    }

    #[test]
    fn plain_replaces_only_the_typographic_characters() {
        assert_eq!(
            plain("\u{2018}a\u{2019} \u{201C}b\u{201D} c\u{2013}d\u{2014}e\u{2026}\u{00A0}é"),
            "'a' \"b\" c-d-e... é"
        );
    }
}
