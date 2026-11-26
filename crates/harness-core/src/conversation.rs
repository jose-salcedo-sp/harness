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

/// The model's thinking trace for one response: trimmed and guaranteed non-empty.
///
/// It is kept apart from the answer text so the UI can show it separately and
/// so the agent loop can decide when to send it back to the model.
///
/// # Sent back within a turn only
///
/// Every reasoning trace stays in the [`Message`] history for the UI, but a
/// model request carries only the traces of the turn in progress: the assistant
/// messages after the latest user message, that is, the current tool-calling
/// chain. Earlier turns are sent without their reasoning.
///
/// - Thinking models with Qwen3- or DeepSeek-style chat templates are trained
///   to see their own earlier thoughts while they call tools to answer one
///   question. Dropping them mid-turn makes the model re-derive its plan after
///   every tool result.
/// - Those same templates strip thoughts from finished turns, so sending them
///   would only cost context-window space and invalidate the server's prompt
///   cache for nothing. Old thoughts also tend to anchor the model on stale
///   plans once the user has moved on.
///
/// Servers whose template cannot render reasoning ignore it (llama.cpp does),
/// so sending it within the turn is safe.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Reasoning(String);

/// Returned when a reasoning trace is empty after trimming whitespace.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("reasoning is empty")]
pub struct EmptyReasoning;

impl Reasoning {
    /// Wraps the reasoning text streamed for one response.
    ///
    /// # Errors
    ///
    /// Returns [`EmptyReasoning`] when `text` is empty or only whitespace.
    pub fn new(text: &str) -> Result<Self, EmptyReasoning> {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            Err(EmptyReasoning)
        } else {
            Ok(Self(trimmed.to_owned()))
        }
    }

    /// The reasoning text.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// One entry in the conversation history.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Message {
    /// A message the user typed.
    User(UserText),
    /// A model reply: streamed text plus any tools it asked to run.
    Assistant {
        /// What the model thought before replying, if it streamed any. See
        /// [`Reasoning`] for when it is sent back to the model.
        reasoning: Option<Reasoning>,
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
            Self::Assistant {
                reasoning,
                text,
                tool_calls,
            } => {
                reasoning.as_ref().map_or(0, |r| r.as_str().len())
                    + text.len()
                    + tool_calls
                        .iter()
                        .map(|call| call.name.as_str().len() + call.arguments.as_str().len())
                        .sum::<usize>()
            }
            Self::Tool { content, .. } => content.len(),
        }
    }
}

/// The history to send with the next model request: `messages` with the
/// reasoning of every turn before the latest user message removed. See
/// [`Reasoning`] for why.
pub(crate) fn request_history(messages: &[Message]) -> Vec<Message> {
    let turn_start = messages
        .iter()
        .rposition(|message| matches!(message, Message::User(_)))
        .unwrap_or(0);
    messages
        .iter()
        .enumerate()
        .map(|(ix, message)| match message {
            Message::Assistant {
                text, tool_calls, ..
            } if ix < turn_start => Message::Assistant {
                reasoning: None,
                text: text.clone(),
                tool_calls: tool_calls.clone(),
            },
            Message::User(_) | Message::Assistant { .. } | Message::Tool { .. } => message.clone(),
        })
        .collect()
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

    #[test]
    fn reasoning_is_trimmed_and_non_empty() {
        assert_eq!(Reasoning::new("\n plan \n").unwrap().as_str(), "plan");
        assert_eq!(Reasoning::new(" \n"), Err(EmptyReasoning));
    }

    fn assistant(reasoning: &str, text: &str) -> Message {
        Message::Assistant {
            reasoning: Some(Reasoning::new(reasoning).unwrap()),
            text: text.to_owned(),
            tool_calls: Vec::new(),
        }
    }

    fn reasoning_of(message: &Message) -> Option<&str> {
        match message {
            Message::Assistant { reasoning, .. } => reasoning.as_ref().map(Reasoning::as_str),
            Message::User(_) | Message::Tool { .. } => None,
        }
    }

    #[test]
    fn request_history_keeps_reasoning_only_for_the_current_turn() {
        let history = request_history(&[
            Message::User(UserText::parse("first").unwrap()),
            assistant("old plan", "one"),
            Message::User(UserText::parse("second").unwrap()),
            assistant("new plan", ""),
            Message::Tool {
                call_id: ToolCallId::new("c1").unwrap(),
                content: "result".to_owned(),
            },
        ]);
        let kept: Vec<_> = history.iter().map(reasoning_of).collect();
        assert_eq!(kept, [None, None, None, Some("new plan"), None]);
        assert!(matches!(
            history.get(1),
            Some(Message::Assistant { text, .. }) if text == "one"
        ));
    }
}
