//! State of the `@` file-mention popup above the composer.
//!
//! This is the UI's view of one mention: which token it is for, the ranked
//! matches for that token's query, and which row is highlighted. It holds no
//! GPUI types so its transitions can be tested directly; the chat view feeds
//! it cursor changes and search results and renders what it holds.

use harness_core::{FileMatch, FileMention, MentionEdit};

/// Which way the highlighted row moves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Step {
    Up,
    Down,
}

/// What a cursor or text change did to the popup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Transition {
    /// Nothing visible changed.
    Unchanged,
    /// A popup opened for a new token; the file list may be stale.
    Opened,
    /// The open token's query changed; its matches need recomputing.
    Requeried,
    /// The popup went away.
    Closed,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) enum FilePopup {
    /// The cursor is not in an `@` token.
    #[default]
    Closed,
    /// The user dismissed the popup for the token starting at `start`; it
    /// stays hidden until the cursor leaves that token.
    Dismissed { start: usize },
    /// Showing matches for `mention`. `matches` may still be for the
    /// previous query until the new results arrive.
    Open {
        mention: FileMention,
        matches: Vec<FileMatch>,
        selected: usize,
    },
}

impl FilePopup {
    /// Follows the token under the cursor, if any.
    pub(crate) fn on_mention(&mut self, next: Option<FileMention>) -> Transition {
        let Some(next) = next else {
            return match std::mem::take(self) {
                Self::Closed => Transition::Unchanged,
                Self::Dismissed { .. } | Self::Open { .. } => Transition::Closed,
            };
        };
        match self {
            Self::Dismissed { start } if *start == next.start() => Transition::Unchanged,
            Self::Open {
                mention, selected, ..
            } if mention.start() == next.start() => {
                let requeried = mention.query() != next.query();
                *mention = next;
                if requeried {
                    *selected = 0;
                    Transition::Requeried
                } else {
                    Transition::Unchanged
                }
            }
            Self::Closed | Self::Dismissed { .. } | Self::Open { .. } => {
                *self = Self::Open {
                    mention: next,
                    matches: Vec::new(),
                    selected: 0,
                };
                Transition::Opened
            }
        }
    }

    /// Shows `found` if it was computed for the current query. Returns
    /// whether anything changed.
    pub(crate) fn on_matches(&mut self, query: &str, found: Vec<FileMatch>) -> bool {
        match self {
            Self::Open {
                mention,
                matches,
                selected,
            } if mention.query() == query => {
                *matches = found;
                *selected = 0;
                true
            }
            Self::Closed | Self::Dismissed { .. } | Self::Open { .. } => false,
        }
    }

    /// Moves the highlight one row, wrapping at either end. Returns whether
    /// there was a row to move to.
    pub(crate) fn step(&mut self, step: Step) -> bool {
        let Self::Open {
            matches, selected, ..
        } = self
        else {
            return false;
        };
        let count = matches.len();
        if count == 0 {
            return false;
        }
        *selected = match step {
            Step::Down => (*selected + 1) % count,
            Step::Up => (*selected + count - 1) % count,
        };
        true
    }

    /// Highlights row `ix`, as when the pointer moves over it. Returns
    /// whether the highlight moved.
    pub(crate) fn hover(&mut self, ix: usize) -> bool {
        match self {
            Self::Open {
                matches, selected, ..
            } if ix < matches.len() && *selected != ix => {
                *selected = ix;
                true
            }
            Self::Closed | Self::Dismissed { .. } | Self::Open { .. } => false,
        }
    }

    /// Hides the popup until the cursor leaves the current token. Returns
    /// whether it was open.
    pub(crate) fn dismiss(&mut self) -> bool {
        match self {
            Self::Open { mention, .. } => {
                *self = Self::Dismissed {
                    start: mention.start(),
                };
                true
            }
            Self::Closed | Self::Dismissed { .. } => false,
        }
    }

    /// Accepts row `ix`, or the highlighted row, and closes the popup.
    /// Returns the edit to apply, or `None` (leaving the popup as it was)
    /// when there is no such row.
    pub(crate) fn accept(&mut self, ix: Option<usize>) -> Option<MentionEdit> {
        let Self::Open {
            mention,
            matches,
            selected,
        } = self
        else {
            return None;
        };
        let edit = mention.accept(matches.get(ix.unwrap_or(*selected))?.path());
        *self = Self::Closed;
        Some(edit)
    }

    pub(crate) fn is_open(&self) -> bool {
        matches!(self, Self::Open { .. })
    }

    /// Whether there is a row that Enter or Tab would accept.
    pub(crate) fn has_matches(&self) -> bool {
        match self {
            Self::Open { matches, .. } => !matches.is_empty(),
            Self::Closed | Self::Dismissed { .. } => false,
        }
    }

    /// The query of the open token.
    pub(crate) fn query(&self) -> Option<&str> {
        match self {
            Self::Open { mention, .. } => Some(mention.query()),
            Self::Closed | Self::Dismissed { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use harness_core::{WorkspacePath, match_files};

    use super::*;

    fn mention(text: &str) -> Option<FileMention> {
        FileMention::at_cursor(text, text.len())
    }

    fn found(query: &str) -> Vec<FileMatch> {
        let files = ["a.rs", "b.rs", "c.rs"].map(|p| WorkspacePath::new(p).unwrap());
        match_files(&files, query, 10)
    }

    fn open_with_matches(text: &str) -> FilePopup {
        let mut popup = FilePopup::Closed;
        popup.on_mention(mention(text));
        let query = popup.query().unwrap().to_owned();
        popup.on_matches(&query, found(&query));
        popup
    }

    fn selected(popup: &FilePopup) -> usize {
        match popup {
            FilePopup::Open { selected, .. } => *selected,
            FilePopup::Closed | FilePopup::Dismissed { .. } => panic!("popup is not open"),
        }
    }

    #[test]
    fn closed_opens_on_a_token() {
        let mut popup = FilePopup::Closed;
        assert_eq!(popup.on_mention(mention("@")), Transition::Opened);
        assert_eq!(popup.query(), Some(""));
        assert!(!popup.has_matches());
    }

    #[test]
    fn closed_ignores_text_without_a_token() {
        let mut popup = FilePopup::Closed;
        assert_eq!(popup.on_mention(None), Transition::Unchanged);
        assert_eq!(popup, FilePopup::Closed);
    }

    #[test]
    fn open_requeries_when_the_query_changes() {
        let mut popup = open_with_matches("@");
        popup.step(Step::Down);
        assert_eq!(popup.on_mention(mention("@a")), Transition::Requeried);
        assert_eq!(popup.query(), Some("a"));
        assert_eq!(selected(&popup), 0);
        assert!(popup.has_matches(), "keeps old rows until new ones arrive");
    }

    #[test]
    fn open_is_unchanged_when_the_query_is_the_same() {
        let mut popup = open_with_matches("@a");
        assert_eq!(popup.on_mention(mention("@a")), Transition::Unchanged);
    }

    #[test]
    fn open_reopens_for_a_different_token() {
        let mut popup = open_with_matches("@a");
        assert_eq!(popup.on_mention(mention("@a @")), Transition::Opened);
        assert!(!popup.has_matches());
    }

    #[test]
    fn open_closes_when_the_cursor_leaves_the_token() {
        let mut popup = open_with_matches("@a");
        assert_eq!(popup.on_mention(None), Transition::Closed);
        assert_eq!(popup, FilePopup::Closed);
    }

    #[test]
    fn open_drops_results_for_a_stale_query() {
        let mut popup = FilePopup::Closed;
        popup.on_mention(mention("@b"));
        assert!(!popup.on_matches("", found("")));
        assert!(!popup.has_matches());
    }

    #[test]
    fn step_wraps_in_both_directions() {
        let mut popup = open_with_matches("@");
        assert!(popup.step(Step::Up));
        assert_eq!(selected(&popup), 2);
        assert!(popup.step(Step::Down));
        assert_eq!(selected(&popup), 0);
    }

    #[test]
    fn step_without_matches_does_nothing() {
        let mut popup = FilePopup::Closed;
        assert!(!popup.step(Step::Down));
        popup.on_mention(mention("@"));
        assert!(!popup.step(Step::Down));
    }

    #[test]
    fn hover_moves_the_highlight() {
        let mut popup = open_with_matches("@");
        assert!(popup.hover(1));
        assert!(!popup.hover(1), "already highlighted");
        assert!(!popup.hover(9), "no such row");
        assert_eq!(selected(&popup), 1);
    }

    #[test]
    fn accept_returns_edit_for_highlighted_row_and_closes() {
        let mut popup = open_with_matches("see @");
        popup.step(Step::Down);
        let edit = popup.accept(None).unwrap();
        assert_eq!(edit.range, 4..5);
        assert_eq!(edit.text, "b.rs ");
        assert_eq!(popup, FilePopup::Closed);
    }

    #[test]
    fn accept_clicked_row() {
        let mut popup = open_with_matches("@");
        assert_eq!(popup.accept(Some(2)).unwrap().text, "c.rs ");
    }

    #[test]
    fn accept_without_a_row_keeps_the_popup() {
        let mut popup = FilePopup::Closed;
        popup.on_mention(mention("@zzz"));
        assert_eq!(popup.accept(None), None);
        assert!(popup.is_open());
        assert_eq!(FilePopup::Closed.accept(None), None);
    }

    #[test]
    fn dismissed_stays_hidden_while_editing_the_same_token() {
        let mut popup = open_with_matches("@a");
        assert!(popup.dismiss());
        assert_eq!(popup.on_mention(mention("@ab")), Transition::Unchanged);
        assert!(!popup.is_open());
    }

    #[test]
    fn dismissed_reopens_for_a_new_token() {
        let mut popup = open_with_matches("@a");
        popup.dismiss();
        assert_eq!(popup.on_mention(mention("@a @")), Transition::Opened);
    }

    #[test]
    fn dismissed_closes_when_the_token_ends() {
        let mut popup = open_with_matches("@a");
        popup.dismiss();
        assert_eq!(popup.on_mention(None), Transition::Closed);
        assert!(!popup.dismiss(), "nothing left to dismiss");
    }
}
