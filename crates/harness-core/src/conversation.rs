//! Conversation domain types: what the user said, what the model replied, and
//! what tools returned.

use std::fmt;

use crate::tool::{ToolCall, ToolCallId};

/// Text the user submitted: trimmed and guaranteed non-empty.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct UserText(String);

/// Returned when user input is empty after trimming whitespace.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("message is empty")]
pub struct EmptyMessage;

impl UserText {
    /// Parses raw input from a text box.
    ///
    /// # Errors
    ///
    /// Returns [`EmptyMessage`] when `raw` is empty or only whitespace.
    pub fn parse(raw: &str) -> Result<Self, EmptyMessage> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            Err(EmptyMessage)
        } else {
            Ok(Self(trimmed.to_owned()))
        }
    }

    /// The message text.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for UserText {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// The instructions sent ahead of every conversation.
///
/// Every model request starts with one, so a request is never empty.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SystemPrompt(String);

impl SystemPrompt {
    /// Wraps custom instructions.
    pub fn new(text: impl Into<String>) -> Self {
        Self(text.into())
    }

    /// The instructions text.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for SystemPrompt {
    fn default() -> Self {
        Self::new(
            "You are a helpful assistant running inside a desktop chat app. \
             When the answer depends on a file, call the `read_file` tool with a path \
             relative to the workspace, then answer from its contents. \
             Keep answers short.",
        )
    }
}

/// One entry in the conversation history.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Message {
    /// A message the user typed.
    User(UserText),
    /// A model reply: streamed text plus any tools it asked to run.
    Assistant {
        /// The visible reply text. May be empty when the model only called tools.
        text: String,
        /// Tool calls requested in this reply, in the order the model sent them.
        tool_calls: Vec<ToolCall>,
    },
    /// The result of one tool call, fed back to the model.
    Tool {
        /// The call this result answers.
        call_id: ToolCallId,
        /// Tool output, or a description of why the tool failed.
        content: String,
    },
}

impl Message {
    /// Approximate size of this message in characters, used to estimate tokens.
    pub(crate) fn char_len(&self) -> usize {
        match self {
            Self::User(text) => text.as_str().len(),
            Self::Assistant { text, tool_calls } => {
                text.len()
                    + tool_calls
                        .iter()
                        .map(|call| call.name.as_str().len() + call.arguments.as_str().len())
                        .sum::<usize>()
            }
            Self::Tool { content, .. } => content.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_trims_surrounding_whitespace() {
        assert_eq!(
            UserText::parse("  hi there \n").unwrap().as_str(),
            "hi there"
        );
    }

    #[test]
    fn rejects_empty_message() {
        assert_eq!(UserText::parse(""), Err(EmptyMessage));
    }

    #[test]
    fn rejects_whitespace_only_message() {
        assert_eq!(UserText::parse(" \t\n "), Err(EmptyMessage));
    }
}
