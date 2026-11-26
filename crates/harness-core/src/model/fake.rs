//! Deterministic [`ModelProvider`] for tests and running the app offline.

use std::{
    collections::VecDeque,
    sync::{Mutex, PoisonError},
    time::Duration,
};

use futures::{channel::mpsc, stream};

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

    /// Adds a reasoning chunk.
    #[must_use]
    pub fn reasoning(mut self, text: impl Into<String>) -> Self {
        self.0.push(Ok(ModelChunk::Reasoning(text.into())));
        self
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
///   hand: a message starting with `write <path> <text>` calls the
///   `write_file` tool with `<text>` as the contents (`\n` becomes a line
///   break), a message containing `read <path>` calls the `read_file` tool, a
///   tool result is summarized, and anything else is echoed back. Every echo
///   response starts with a short reasoning trace explaining the rule it took.
///
/// Responses stream instantly unless [`FakeProvider::chunk_delay`] is set.
#[derive(Debug)]
pub struct FakeProvider {
    behavior: Behavior,
    chunk_delay: Duration,
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
            chunk_delay: Duration::ZERO,
            requests: Mutex::new(Vec::new()),
        }
    }

    /// Waits `delay` before each stream item, like a model generating tokens,
    /// so the UI can be watched while it streams. Each paced response runs on
    /// its own OS thread; dropping the stream stops it.
    #[must_use]
    pub fn chunk_delay(mut self, delay: Duration) -> Self {
        self.chunk_delay = delay;
        self
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
        if self.chunk_delay.is_zero() {
            Box::pin(stream::iter(items))
        } else {
            paced(items, self.chunk_delay)
        }
    }
}

fn paced(items: Vec<Result<ModelChunk, ModelError>>, delay: Duration) -> ModelStream {
    let (sender, receiver) = mpsc::unbounded();
    let spawned = std::thread::Builder::new()
        .name("harness-fake-model".to_owned())
        .spawn(move || {
            for item in items {
                std::thread::sleep(delay);
                if sender.unbounded_send(item).is_err() {
                    return;
                }
            }
        });
    match spawned {
        Ok(_) => Box::pin(receiver),
        Err(error) => Box::pin(stream::iter([Err(ModelError::Spawn(error))])),
    }
}

fn echo_reply(request: &ModelRequest) -> Vec<ModelChunk> {
    let prompt_chars = request.system.as_str().len()
        + request
            .messages
            .iter()
            .map(Message::char_len)
            .sum::<usize>();
    let (reasoning, reply) = match request.messages.last() {
        Some(Message::User(text)) => {
            if let Some((path, contents)) = requested_write(text.as_str()) {
                (
                    format!(
                        "The user wants `{path}` to hold the text they gave. write_file \
                         replaces the whole file, so I'll send that text as the complete \
                         contents."
                    ),
                    EchoReply::WriteFile { path, contents },
                )
            } else if let Some(path) = requested_path(text.as_str()) {
                (
                    format!(
                        "The user wants to know what is in `{path}`. I can't answer from \
                         memory, so I'll call read_file with that path and summarize the \
                         result."
                    ),
                    EchoReply::ReadFile(path),
                )
            } else {
                (
                    "There is no file to read or write in this message, so I'll echo it \
                     back and explain how to try the file tools."
                        .to_owned(),
                    EchoReply::Text(format!(
                        "You said: \"{text}\"\n\nI'm the offline fake provider, so I only \
                         echo. Ask me to `read <path>` or `write <path> <text>` to watch the \
                         loop call the read_file or write_file tool."
                    )),
                )
            }
        }
        Some(Message::Tool { call_id, content }) => {
            if tool_name(&request.messages, call_id) == Some(WRITE_FILE) {
                (
                    write_result_reasoning(content),
                    EchoReply::Text(summarize_write(content)),
                )
            } else {
                (
                    tool_result_reasoning(content),
                    EchoReply::Text(summarize_file(content)),
                )
            }
        }
        Some(Message::Assistant { .. }) | None => (
            "Nothing to respond to yet, so I'll just say hello.".to_owned(),
            EchoReply::Text("Hello!".to_owned()),
        ),
    };

    let mut chunks: Vec<ModelChunk> = reasoning
        .split_inclusive(' ')
        .map(|word| ModelChunk::Reasoning(word.to_owned()))
        .collect();
    let completion_chars = reasoning.len()
        + match reply {
            EchoReply::Text(text) => {
                chunks.extend(
                    text.split_inclusive(' ')
                        .map(|word| ModelChunk::Text(word.to_owned())),
                );
                text.len()
            }
            EchoReply::ReadFile(path) => {
                let arguments = serde_json::json!({ "path": path });
                push_tool_call(&mut chunks, request, READ_FILE, &arguments)
            }
            EchoReply::WriteFile { path, contents } => {
                let arguments = serde_json::json!({ "path": path, "contents": contents });
                push_tool_call(&mut chunks, request, WRITE_FILE, &arguments)
            }
        };
    chunks.push(ModelChunk::Usage(TokenUsage {
        prompt_tokens: super::estimate_tokens(prompt_chars),
        completion_tokens: super::estimate_tokens(completion_chars),
    }));
    chunks
}

const READ_FILE: &str = "read_file";
const WRITE_FILE: &str = "write_file";

/// Streams a call to `tool`; returns the characters "generated".
fn push_tool_call(
    chunks: &mut Vec<ModelChunk>,
    request: &ModelRequest,
    tool: &str,
    arguments: &serde_json::Value,
) -> usize {
    let arguments = arguments.to_string();
    let len = arguments.len();
    let id = ToolCallId::new(format!("call_{}", request.messages.len()));
    let name = ToolName::new(tool);
    if let (Ok(id), Ok(name)) = (id, name) {
        chunks.push(ModelChunk::ToolCall(ToolCall {
            id,
            name,
            arguments: ToolArguments::new(arguments),
        }));
    }
    len
}

enum EchoReply {
    Text(String),
    ReadFile(String),
    WriteFile { path: String, contents: String },
}

/// Which tool produced the result for `call_id`, from the assistant turn that
/// requested it.
fn tool_name<'a>(messages: &'a [Message], call_id: &ToolCallId) -> Option<&'a str> {
    messages.iter().rev().find_map(|message| match message {
        Message::Assistant { tool_calls, .. } => tool_calls
            .iter()
            .find(|call| call.id == *call_id)
            .map(|call| call.name.as_str()),
        Message::User(_) | Message::Tool { .. } => None,
    })
}

/// Parses `write <path> <text>`: `write` must be the first word, and the text
/// after the path is the file contents, with `\n` turned into line breaks.
fn requested_write(text: &str) -> Option<(String, String)> {
    let (command, rest) = text.split_once(char::is_whitespace)?;
    if !command.eq_ignore_ascii_case("write") {
        return None;
    }
    let rest = rest.trim_start();
    let (path, contents) = rest.split_once(char::is_whitespace).unwrap_or((rest, ""));
    let path = path.trim_matches(|c: char| matches!(c, '`' | '"' | '\''));
    if path.is_empty() {
        return None;
    }
    let mut contents = contents.trim().replace("\\n", "\n");
    if !contents.is_empty() && !contents.ends_with('\n') {
        contents.push('\n');
    }
    Some((path.to_owned(), contents))
}

fn write_result_reasoning(content: &str) -> String {
    if content.starts_with("error:") {
        "The write failed, so the file is unchanged. I'll pass the error on.".to_owned()
    } else {
        "The write went through. I'll confirm what changed.".to_owned()
    }
}

fn summarize_write(content: &str) -> String {
    if content.starts_with("error:") {
        return format!("I couldn't write that file: {content}");
    }
    format!("Done: {content}.")
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

fn tool_result_reasoning(content: &str) -> String {
    if content.starts_with("error:") {
        "The read failed, so there is nothing to summarize. I'll pass the error on.".to_owned()
    } else {
        format!(
            "The tool returned {} lines. A preview of the first few should be enough to \
             describe the file.",
            content.lines().count()
        )
    }
}

fn summarize_file(content: &str) -> String {
    if content.starts_with("error:") {
        return format!("I couldn't read that file: {content}");
    }
    let preview: Vec<&str> = content.lines().take(6).collect();
    format!(
        "The file has {} lines ({} bytes). It starts with:\n\n```\n{}\n```",
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
            ModelChunk::Reasoning(_) | ModelChunk::Text(_) | ModelChunk::Usage(_) => None,
        });
        assert_eq!(call.unwrap().arguments.as_str(), r#"{"path":"Cargo.toml"}"#);
    }

    fn tool_call(chunks: &[ModelChunk]) -> &ToolCall {
        chunks
            .iter()
            .find_map(|chunk| match chunk {
                ModelChunk::ToolCall(call) => Some(call),
                ModelChunk::Reasoning(_) | ModelChunk::Text(_) | ModelChunk::Usage(_) => None,
            })
            .unwrap()
    }

    #[test]
    fn echo_calls_write_file_when_asked_to_write() {
        let chunks = collect(
            &FakeProvider::echo(),
            request(vec![Message::User(
                UserText::parse("write `notes/todo.md` # Todo\\n- milk").unwrap(),
            )]),
        );
        let call = tool_call(&chunks);
        assert_eq!(call.name.as_str(), "write_file");
        let args: serde_json::Value = serde_json::from_str(call.arguments.as_str()).unwrap();
        assert_eq!(
            args,
            serde_json::json!({ "path": "notes/todo.md", "contents": "# Todo\n- milk\n" })
        );
    }

    #[test]
    fn echo_does_not_write_unless_write_is_the_first_word() {
        let chunks = collect(
            &FakeProvider::echo(),
            request(vec![Message::User(
                UserText::parse("how do I write tests?").unwrap(),
            )]),
        );
        assert!(
            chunks
                .iter()
                .all(|chunk| !matches!(chunk, ModelChunk::ToolCall(_)))
        );
    }

    #[test]
    fn echo_summarizes_write_results_as_writes() {
        let call = ToolCall {
            id: ToolCallId::new("call_1").unwrap(),
            name: ToolName::new("write_file").unwrap(),
            arguments: ToolArguments::new("{}"),
        };
        let chunks = collect(
            &FakeProvider::echo(),
            request(vec![
                Message::Assistant {
                    reasoning: None,
                    text: String::new(),
                    tool_calls: vec![call],
                },
                Message::Tool {
                    call_id: ToolCallId::new("call_1").unwrap(),
                    content: "created `a.txt`: 3 bytes, 1 line".to_owned(),
                },
            ]),
        );
        let text: String = chunks
            .iter()
            .filter_map(|chunk| match chunk {
                ModelChunk::Text(text) => Some(text.as_str()),
                ModelChunk::Reasoning(_) | ModelChunk::ToolCall(_) | ModelChunk::Usage(_) => None,
            })
            .collect();
        assert_eq!(text, "Done: created `a.txt`: 3 bytes, 1 line.");
    }

    #[test]
    fn echo_thinks_before_replying() {
        let chunks = collect(
            &FakeProvider::echo(),
            request(vec![Message::User(UserText::parse("hello").unwrap())]),
        );
        let first_text = chunks
            .iter()
            .position(|chunk| matches!(chunk, ModelChunk::Text(_)))
            .unwrap();
        assert!(matches!(chunks.first(), Some(ModelChunk::Reasoning(_))));
        assert!(
            chunks
                .iter()
                .skip(first_text)
                .all(|chunk| !matches!(chunk, ModelChunk::Reasoning(_)))
        );
    }

    #[test]
    fn scripted_replies_stream_reasoning() {
        let provider = FakeProvider::scripted([FakeReply::new().reasoning("hm").text("ok")]);
        assert_eq!(
            collect(&provider, request(Vec::new())),
            [
                ModelChunk::Reasoning("hm".to_owned()),
                ModelChunk::Text("ok".to_owned())
            ]
        );
    }

    #[test]
    fn paced_replies_yield_the_same_chunks() {
        let messages = vec![Message::User(UserText::parse("hello").unwrap())];
        let paced = FakeProvider::echo().chunk_delay(Duration::from_millis(1));
        assert_eq!(
            collect(&paced, request(messages.clone())),
            collect(&FakeProvider::echo(), request(messages))
        );
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
