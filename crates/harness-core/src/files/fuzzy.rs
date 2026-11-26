//! Fuzzy ranking of workspace paths, built on `nucleo-matcher`.

use std::{cmp::Reverse, ops::Range};

use nucleo_matcher::{
    Config, Matcher, Utf32Str,
    pattern::{Atom, AtomKind, CaseMatching, Normalization},
};

use super::WorkspacePath;

/// One path that matched a query, with the bytes that matched.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FileMatch {
    path: WorkspacePath,
    highlights: Vec<Range<usize>>,
}

impl FileMatch {
    /// The matched path.
    pub fn path(&self) -> &WorkspacePath {
        &self.path
    }

    /// Byte ranges of [`Self::path`] that matched the query, sorted and
    /// non-overlapping. Empty for an empty query.
    pub fn highlights(&self) -> &[Range<usize>] {
        &self.highlights
    }
}

/// Ranks `files` against `query` and returns the best `limit` matches.
///
/// Matching is fuzzy (query characters must appear in order, with gaps
/// allowed), case-insensitive unless the query has an uppercase letter, and
/// favours matches at path-segment boundaries. Ties go to the shorter path,
/// then to the alphabetically first one.
///
/// An empty query matches every file, shallowest first, so a bare `@` offers
/// the workspace's top-level files.
///
/// This scores every file, which takes a few milliseconds for tens of
/// thousands of paths; call it off the UI thread.
///
/// # Examples
///
/// ```
/// use harness_core::files::{WorkspacePath, match_files};
///
/// let files = ["Cargo.toml", "src/agent.rs", "src/lib.rs"].map(|p| WorkspacePath::new(p).unwrap());
/// let best = match_files(&files, "lib", 5);
/// assert_eq!(best[0].path().as_str(), "src/lib.rs");
/// assert_eq!(best[0].highlights(), [4..7]);
/// ```
pub fn match_files(files: &[WorkspacePath], query: &str, limit: usize) -> Vec<FileMatch> {
    if query.is_empty() {
        let mut shallowest: Vec<&WorkspacePath> = files.iter().collect();
        shallowest.sort_by_key(|path| (path.depth(), *path));
        return shallowest
            .into_iter()
            .take(limit)
            .map(|path| FileMatch {
                path: path.clone(),
                highlights: Vec::new(),
            })
            .collect();
    }

    let atom = Atom::new(
        query,
        CaseMatching::Smart,
        Normalization::Smart,
        AtomKind::Fuzzy,
        false,
    );
    let mut matcher = Matcher::new(Config::DEFAULT.match_paths());
    let mut chars = Vec::new();
    let mut scored: Vec<(u16, &WorkspacePath)> = files
        .iter()
        .filter_map(|path| {
            let score = atom.score(haystack(path.as_str(), &mut chars), &mut matcher)?;
            Some((score, path))
        })
        .collect();
    scored.sort_by_key(|&(score, path)| (Reverse(score), path.as_str().len(), path));
    scored.truncate(limit);

    let mut indices = Vec::new();
    scored
        .into_iter()
        .map(|(_, path)| {
            indices.clear();
            atom.indices(
                haystack(path.as_str(), &mut chars),
                &mut matcher,
                &mut indices,
            );
            FileMatch {
                path: path.clone(),
                highlights: byte_ranges(path.as_str(), &mut indices),
            }
        })
        .collect()
}

/// Views `text` as one matcher unit per `char`, so match indices are char
/// indices. `Utf32Str::new` would group graphemes instead, which would not
/// map back to byte offsets one to one.
fn haystack<'a>(text: &'a str, chars: &'a mut Vec<char>) -> Utf32Str<'a> {
    if text.is_ascii() {
        Utf32Str::Ascii(text.as_bytes())
    } else {
        chars.clear();
        chars.extend(text.chars());
        Utf32Str::Unicode(chars)
    }
}

/// Converts matched char indices into merged byte ranges of `text`.
fn byte_ranges(text: &str, indices: &mut [u32]) -> Vec<Range<usize>> {
    indices.sort_unstable();
    let mut wanted = indices.iter().peekable();
    let mut ranges: Vec<Range<usize>> = Vec::new();
    for (char_index, (start, ch)) in text.char_indices().enumerate() {
        let Some(&&next) = wanted.peek() else { break };
        if usize::try_from(next).ok() != Some(char_index) {
            continue;
        }
        while wanted.next_if_eq(&&next).is_some() {}
        let end = start + ch.len_utf8();
        match ranges.last_mut() {
            Some(last) if last.end == start => last.end = end,
            Some(_) | None => ranges.push(start..end),
        }
    }
    ranges
}

#[cfg(test)]
mod tests {
    use super::*;

    fn paths(list: &[&str]) -> Vec<WorkspacePath> {
        list.iter()
            .map(|p| WorkspacePath::new(*p).unwrap())
            .collect()
    }

    fn spans(found: &FileMatch) -> Vec<(usize, usize)> {
        found
            .highlights()
            .iter()
            .map(|r| (r.start, r.end))
            .collect()
    }

    fn ranked(files: &[WorkspacePath], query: &str) -> Vec<String> {
        match_files(files, query, 10)
            .into_iter()
            .map(|m| m.path().as_str().to_owned())
            .collect()
    }

    #[test]
    fn matches_characters_in_order_with_gaps() {
        let files = paths(&["crates/harness-app/src/chat.rs", "README.md"]);
        assert_eq!(ranked(&files, "hach"), ["crates/harness-app/src/chat.rs"]);
    }

    #[test]
    fn excludes_files_missing_a_query_character() {
        let files = paths(&["src/lib.rs", "src/agent.rs"]);
        assert_eq!(ranked(&files, "libz"), Vec::<String>::new());
    }

    #[test]
    fn ranks_file_name_match_above_scattered_match() {
        let files = paths(&["crates/harness-core/src/model/openai.rs", "src/model.rs"]);
        assert_eq!(
            ranked(&files, "model"),
            ["src/model.rs", "crates/harness-core/src/model/openai.rs"]
        );
    }

    #[test]
    fn prefers_segment_boundaries() {
        let files = paths(&["src/toolchain_lib_helper.rs", "lib/tool.rs"]);
        assert_eq!(ranked(&files, "lt")[0], "lib/tool.rs");
    }

    #[test]
    fn breaks_ties_by_length_then_name() {
        let files = paths(&["b/x.rs", "a/x.rs", "aa/x.rs"]);
        assert_eq!(ranked(&files, "x.rs"), ["a/x.rs", "b/x.rs", "aa/x.rs"]);
    }

    #[test]
    fn ignores_case_for_lowercase_query() {
        let files = paths(&["README.md"]);
        assert_eq!(ranked(&files, "readme"), ["README.md"]);
    }

    #[test]
    fn uppercase_query_is_case_sensitive() {
        let files = paths(&["readme.md", "README.md"]);
        assert_eq!(ranked(&files, "READ"), ["README.md"]);
    }

    #[test]
    fn empty_query_lists_shallowest_files_first() {
        let files = paths(&["a/b/c.rs", "z.rs", "a/b.rs", "Cargo.toml"]);
        assert_eq!(
            ranked(&files, ""),
            ["Cargo.toml", "z.rs", "a/b.rs", "a/b/c.rs"]
        );
    }

    #[test]
    fn respects_limit() {
        let files = paths(&["a1", "a2", "a3", "a4"]);
        assert_eq!(match_files(&files, "a", 2).len(), 2);
        assert_eq!(match_files(&files, "", 3).len(), 3);
    }

    #[test]
    fn highlights_matched_bytes_as_merged_ranges() {
        let files = paths(&["src/chat.rs"]);
        let best = match_files(&files, "chat", 1);
        assert_eq!(spans(&best[0]), [(4, 8)]);
    }

    #[test]
    fn highlights_split_matches_separately() {
        let files = paths(&["ab/cd"]);
        let best = match_files(&files, "ad", 1);
        assert_eq!(spans(&best[0]), [(0, 1), (4, 5)]);
    }

    #[test]
    fn highlights_use_byte_offsets_for_non_ascii_paths() {
        let files = paths(&["docs/café/menu.md"]);
        let best = match_files(&files, "émenu", 1);
        let path = best[0].path().as_str();
        let matched: String = best[0]
            .highlights()
            .iter()
            .map(|range| &path[range.clone()])
            .collect();
        assert_eq!(matched, "émenu");
    }
}
