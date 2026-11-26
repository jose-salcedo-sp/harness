//! Finding the `@query` token at the composer's cursor.

use std::ops::Range;

use super::WorkspacePath;

/// An `@query` token the cursor is inside: the user is asking for a file.
///
/// A token starts with `@` at the start of the text or after whitespace, so
/// `name@example.com` is not one, and runs to the next whitespace.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FileMention {
    range: Range<usize>,
    query: String,
    tail: Tail,
}

/// What follows the token, which decides whether accepting adds a space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Tail {
    EndOfText,
    Whitespace,
}

/// A text replacement: put [`Self::text`] in place of the bytes in
/// [`Self::range`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MentionEdit {
    /// Byte range of the text to replace.
    pub range: Range<usize>,
    /// The replacement text.
    pub text: String,
}

impl FileMention {
    /// Finds the mention token around byte offset `cursor` in `text`.
    ///
    /// Returns `None` when the word before the cursor does not start with
    /// `@`, or when `cursor` is past the end or not on a character boundary.
    ///
    /// # Examples
    ///
    /// ```
    /// use harness_core::files::FileMention;
    ///
    /// let text = "read @src/li";
    /// let mention = FileMention::at_cursor(text, text.len()).unwrap();
    /// assert_eq!(mention.query(), "src/li");
    /// assert_eq!(mention.range(), 5..12);
    /// assert!(FileMention::at_cursor("mail me@host", 12).is_none());
    /// ```
    pub fn at_cursor(text: &str, cursor: usize) -> Option<Self> {
        let before = text.get(..cursor)?;
        let after = text.get(cursor..)?;
        let start = before
            .char_indices()
            .rev()
            .find(|(_, ch)| ch.is_whitespace())
            .map_or(0, |(ix, ch)| ix + ch.len_utf8());
        let query = before.get(start..)?.strip_prefix('@')?;
        let (end, tail) = after
            .find(char::is_whitespace)
            .map_or((text.len(), Tail::EndOfText), |len| {
                (cursor + len, Tail::Whitespace)
            });
        Some(Self {
            range: start..end,
            query: query.to_owned(),
            tail,
        })
    }

    /// The text between `@` and the cursor, used to search for files.
    pub fn query(&self) -> &str {
        &self.query
    }

    /// Byte range of the whole token, from `@` to the next whitespace.
    pub fn range(&self) -> Range<usize> {
        self.range.clone()
    }

    /// Byte offset of the token's `@`, which identifies the token while the
    /// user edits its query.
    pub fn start(&self) -> usize {
        self.range.start
    }

    /// The edit that replaces the whole token with `path`.
    ///
    /// At the end of the text a space follows the path so the user can keep
    /// typing; before existing whitespace nothing is added.
    pub fn accept(&self, path: &WorkspacePath) -> MentionEdit {
        let mut text = path.as_str().to_owned();
        match self.tail {
            Tail::EndOfText => text.push(' '),
            Tail::Whitespace => {}
        }
        MentionEdit {
            range: self.range(),
            text,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at_end(text: &str) -> Option<FileMention> {
        FileMention::at_cursor(text, text.len())
    }

    fn apply(text: &str, edit: &MentionEdit) -> String {
        let mut out = text.to_owned();
        out.replace_range(edit.range.clone(), &edit.text);
        out
    }

    #[test]
    fn bare_at_sign_is_an_empty_query() {
        let mention = at_end("@").unwrap();
        assert_eq!(mention.query(), "");
        assert_eq!(mention.range(), 0..1);
    }

    #[test]
    fn finds_token_after_whitespace() {
        let mention = at_end("look at @crates/core").unwrap();
        assert_eq!(mention.query(), "crates/core");
        assert_eq!(mention.range(), 8..20);
        assert_eq!(mention.start(), 8);
    }

    #[test]
    fn finds_token_after_newline() {
        assert_eq!(at_end("first line\n@lib").unwrap().query(), "lib");
    }

    #[test]
    fn ignores_at_sign_inside_a_word() {
        assert_eq!(at_end("mail me@example.com"), None);
    }

    #[test]
    fn ignores_word_without_at_sign() {
        assert_eq!(at_end("plain words"), None);
    }

    #[test]
    fn ends_once_whitespace_follows_the_token() {
        assert_eq!(at_end("@src/lib.rs "), None);
    }

    #[test]
    fn query_stops_at_cursor_but_range_covers_whole_token() {
        let mention = FileMention::at_cursor("@chatview rest", 5).unwrap();
        assert_eq!(mention.query(), "chat");
        assert_eq!(mention.range(), 0..9);
    }

    #[test]
    fn uses_the_token_the_cursor_is_in() {
        let text = "@one @two";
        assert_eq!(FileMention::at_cursor(text, 4).unwrap().query(), "one");
        assert_eq!(FileMention::at_cursor(text, 9).unwrap().query(), "two");
    }

    #[test]
    fn keeps_later_at_signs_in_the_query() {
        assert_eq!(at_end("@@scope/pkg").unwrap().query(), "@scope/pkg");
    }

    #[test]
    fn handles_multibyte_text() {
        let text = "café @é";
        let mention = at_end(text).unwrap();
        assert_eq!(mention.query(), "é");
        assert_eq!(&text[mention.range()], "@é");
    }

    #[test]
    fn rejects_cursor_off_a_char_boundary() {
        assert_eq!(FileMention::at_cursor("@é", 2), None);
    }

    #[test]
    fn rejects_cursor_past_the_end() {
        assert_eq!(FileMention::at_cursor("@a", 3), None);
    }

    #[test]
    fn accept_at_end_of_text_appends_a_space() {
        let text = "read @chat";
        let path = WorkspacePath::new("crates/harness-app/src/chat.rs").unwrap();
        let edit = at_end(text).unwrap().accept(&path);
        assert_eq!(apply(text, &edit), "read crates/harness-app/src/chat.rs ");
    }

    #[test]
    fn accept_before_whitespace_adds_nothing() {
        let text = "read @cha and summarize";
        let path = WorkspacePath::new("src/chat.rs").unwrap();
        let edit = FileMention::at_cursor(text, 8).unwrap().accept(&path);
        assert_eq!(apply(text, &edit), "read src/chat.rs and summarize");
    }
}
