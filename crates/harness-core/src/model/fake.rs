//! Deterministic [`ModelProvider`] for tests and running the app offline.

use std::{
    collections::VecDeque,
    sync::{Mutex, PoisonError},
};

use futures::stream;

use super::{ModelChunk, ModelError, ModelProvider, ModelRequest, ModelStream, TokenUsage};
use crate::{
    conversation::Message,
    tool::{ToolArguments, ToolCall, ToolCallId, ToolName},
};

/// One scripted model response: the items its stream yields, in order.
#[derive(Debug, Default)]
pub struct FakeReply(Vec<Result<ModelChunk, ModelError>>);

impl FakeReply {
    /// An empty response.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a text chunk.
    #[must_use]
    pub fn text(mut self, text: impl Into<String>) -> Self {
        self.0.push(Ok(ModelChunk::Text(text.into())));
        self
    }

    /// Adds a tool call.
    #[must_use]
    pub fn tool_call(mut self, call: ToolCall) -> Self {
        self.0.push(Ok(ModelChunk::ToolCall(call)));
        self
    }

    /// Adds a usage report.
    #[must_use]
    pub fn usage(mut self, usage: TokenUsage) -> Self {
        self.0.push(Ok(ModelChunk::Usage(usage)));
        self
    }

    /// Ends the response with `error` after the chunks added so far.
    #[must_use]
    pub fn fail(mut self, error: ModelError) -> Self {
        self.0.push(Err(error));
        self
    }
}

#[derive(Debug)]
enum Behavior {
    Scripted(Mutex<VecDeque<FakeReply>>),
    Echo,
}

/// A model provider that never touches the network.
///
/// - [`FakeProvider::scripted`] replays fixed responses, one per request.
/// - [`FakeProvider::echo`] follows simple rules, so the app can be driven by
///   hand: a message containing `read <path>` calls the `read_file` tool, a
///   tool result is summarized, and anything else is echoed back.
#[derive(Debug)]
pub struct FakeProvider {
    behavior: Behavior,
    requests: Mutex<Vec<ModelRequest>>,
}

impl FakeProvider {
    /// Replays `replies` in order. Once they run out, every response is empty.
    pub fn scripted(replies: impl IntoIterator<Item = FakeReply>) -> Self {
        Self::with_behavior(Behavior::Scripted(Mutex::new(
            replies.into_iter().collect(),
        )))
    }

    /// Answers by rule; see the type docs.
    pub fn echo() -> Self {
        Self::with_behavior(Behavior::Echo)
    }

    fn with_behavior(behavior: Behavior) -> Self {
        Self {
            behavior,
            requests: Mutex::new(Vec::new()),
        }
    }

    /// Every request received so far, oldest first.
    pub fn requests(&self) -> Vec<ModelRequest> {
        self.requests
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
}

impl ModelProvider for FakeProvider {
    fn stream(&self, request: ModelRequest) -> ModelStream {
        let items = match &self.behavior {
            Behavior::Scripted(replies) => {
                replies
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .pop_front()
                    .unwrap_or_default()
                    .0
            }
            Behavior::Echo => echo_reply(&request).into_iter().map(Ok).collect(),
        };
        self.requests
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(request);
        Box::pin(stream::iter(items))
    }
}

fn echo_reply(request: &ModelRequest) -> Vec<ModelChunk> {
    let prompt_chars = request.system.as_str().len()
        + request
            .messages
            .iter()
            .map(Message::char_len)
            .sum::<usize>();
    let reply = match request.messages.last() {
        Some(Message::User(text)) => match requested_path(text.as_str()) {
            Some(path) => EchoReply::ReadFile(path),
            None => EchoReply::Text(format!(
                "You said: \"{text}\"\n\nI'm the offline fake provider, so I only echo. \
                 Ask me to `read <path>` to watch the loop call the read_file tool."
            )),
        },
        Some(Message::Tool { content, .. }) => EchoReply::Text(summarize_file(content)),
        Some(Message::Assistant { .. }) | None => EchoReply::Text("Hello!".to_owned()),
    };

    let mut chunks = Vec::new();
    let completion_chars = match reply {
        EchoReply::Text(text) => {
            chunks.extend(
                text.split_inclusive(' ')
                    .map(|word| ModelChunk::Text(word.to_owned())),
            );
            text.len()
        }
        EchoReply::ReadFile(path) => {
            let arguments = serde_json::json!({ "path": path }).to_string();
            let len = arguments.len();
            let id = ToolCallId::new(format!("call_{}", request.messages.len()));
            let name = ToolName::new("read_file");
            if let (Ok(id), Ok(name)) = (id, name) {
                chunks.push(ModelChunk::Text(format!("Reading `{path}`…")));
                chunks.push(ModelChunk::ToolCall(ToolCall {
                    id,
                    name,
                    arguments: ToolArguments::new(arguments),
                }));
            }
            len
        }
    };
    chunks.push(ModelChunk::Usage(TokenUsage {
        prompt_tokens: super::estimate_tokens(prompt_chars),
        completion_tokens: super::estimate_tokens(completion_chars),
    }));
    chunks
}

enum EchoReply {
    Text(String),
    ReadFile(String),
}

fn requested_path(text: &str) -> Option<String> {
    let mut words = text.split_whitespace();
    words.find(|word| word.eq_ignore_ascii_case("read"))?;
    let path = words
        .next()?
        .trim_matches(|c: char| matches!(c, '`' | '"' | '\'' | '?' | '!' | ',' | ';'))
        .trim_end_matches('.');
    (!path.is_empty()).then(|| path.to_owned())
}

fn summarize_file(content: &str) -> String {
    if content.starts_with("error:") {
        return format!("I couldn't read that file: {content}");
    }
    let preview: Vec<&str> = content.lines().take(6).collect();
    format!(
        "The file has {} lines ({} bytes). It starts with:\n\n{}",
        content.lines().count(),
        content.len(),
        preview.join("\n")
    )
}

#[cfg(test)]
mod tests {
    use futures::{StreamExt, executor::block_on};

    use super::*;
    use crate::conversation::{SystemPrompt, UserText};

    fn request(messages: Vec<Message>) -> ModelRequest {
        ModelRequest {
            model: super::super::ModelId::new("fake").unwrap(),
            system: SystemPrompt::default(),
            messages,
            tools: Vec::new(),
        }
    }

    fn collect(provider: &FakeProvider, request: ModelRequest) -> Vec<ModelChunk> {
        block_on(provider.stream(request).map(Result::unwrap).collect())
    }

    #[test]
    fn echo_calls_read_file_when_asked_to_read() {
        let chunks = collect(
            &FakeProvider::echo(),
            request(vec![Message::User(
                UserText::parse("please read `Cargo.toml`?").unwrap(),
            )]),
        );
        let call = chunks.iter().find_map(|chunk| match chunk {
            ModelChunk::ToolCall(call) => Some(call),
            ModelChunk::Text(_) | ModelChunk::Usage(_) => None,
        });
        assert_eq!(call.unwrap().arguments.as_str(), r#"{"path":"Cargo.toml"}"#);
    }

    #[test]
    fn echo_is_deterministic() {
        let provider = FakeProvider::echo();
        let messages = vec![Message::User(UserText::parse("hello").unwrap())];
        assert_eq!(
            collect(&provider, request(messages.clone())),
            collect(&provider, request(messages))
        );
    }

    #[test]
    fn scripted_replies_run_out_to_empty_responses() {
        let provider = FakeProvider::scripted([FakeReply::new().text("one")]);
        assert_eq!(
            collect(&provider, request(Vec::new())),
            [ModelChunk::Text("one".to_owned())]
        );
        assert_eq!(collect(&provider, request(Vec::new())), Vec::new());
    }
}
