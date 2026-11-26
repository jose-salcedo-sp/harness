//! HTTP adapter for OpenAI-compatible chat-completions servers.
//!
//! This covers `llama-server` from llama.cpp (how the Bonsai 2 models run
//! locally), as well as hosted OpenAI-compatible APIs.

mod sse;

use std::{
    fmt,
    io::{BufRead, BufReader},
    time::Duration,
};

use futures::{channel::mpsc, stream};
use serde_json::{Value, json};

use super::{ModelError, ModelProvider, ModelRequest, ModelStream};
use crate::{conversation::Message, tool::ToolSpec};
use sse::SseDecoder;

/// Environment variable holding the server's base URL.
pub const BASE_URL_VAR: &str = "HARNESS_BASE_URL";
/// Environment variable holding the API key.
pub const API_KEY_VAR: &str = "HARNESS_API_KEY";
/// Base URL used when [`BASE_URL_VAR`] is unset: `llama-server`'s default address.
pub const DEFAULT_BASE_URL: &str = "http://127.0.0.1:8080/v1";
/// Environment variable selecting the [`ReasoningRequest`]: `llama-cpp` or `none`.
pub const REASONING_VAR: &str = "HARNESS_REASONING";

/// Which request fields ask the server to think and to stream the thoughts
/// separately.
///
/// Whatever is chosen, `reasoning_content` deltas and inline `<think>` tags in
/// the reply are both parsed into [`ModelChunk::Reasoning`](super::ModelChunk::Reasoning).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ReasoningRequest {
    /// Sends `llama-server`'s fields: `reasoning_format: "deepseek"`, so thoughts
    /// stream as `reasoning_content`, and `chat_template_kwargs.enable_thinking`,
    /// which turns thinking on in Qwen3-style chat templates.
    #[default]
    LlamaCpp,
    /// Sends no reasoning fields, for hosted APIs that reject unknown ones.
    None,
}

/// Why an environment value could not be parsed into a [`ReasoningRequest`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("reasoning request `{value}` must be `llama-cpp` or `none`")]
pub struct UnknownReasoningRequest {
    /// The rejected value.
    pub value: String,
}

impl std::str::FromStr for ReasoningRequest {
    type Err = UnknownReasoningRequest;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim() {
            "llama-cpp" => Ok(Self::LlamaCpp),
            "none" => Ok(Self::None),
            other => Err(UnknownReasoningRequest {
                value: other.to_owned(),
            }),
        }
    }
}

/// A bearer token. Its `Debug` output is redacted.
#[derive(Clone, PartialEq, Eq)]
pub struct ApiKey(String);

impl ApiKey {
    /// Wraps a key.
    pub fn new(key: impl Into<String>) -> Self {
        Self(key.into())
    }
}

impl fmt::Debug for ApiKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ApiKey(<redacted>)")
    }
}

/// Why an [`OpenAiConfig`] is invalid.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ConfigError {
    /// The base URL is not an `http://` or `https://` URL.
    #[error("base URL `{url}` must start with http:// or https://")]
    InvalidBaseUrl {
        /// The rejected URL.
        url: String,
    },
    /// [`REASONING_VAR`] holds an unknown value.
    #[error("invalid {REASONING_VAR}")]
    InvalidReasoning(#[from] UnknownReasoningRequest),
}

/// Where and how to reach the server.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenAiConfig {
    base_url: String,
    api_key: Option<ApiKey>,
    reasoning: ReasoningRequest,
    connect_timeout: Duration,
    response_timeout: Duration,
    read_timeout: Duration,
}

impl OpenAiConfig {
    /// A config for the server at `base_url`, such as `http://127.0.0.1:8080/v1`.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError::InvalidBaseUrl`] when `base_url` is not HTTP(S).
    pub fn new(base_url: &str) -> Result<Self, ConfigError> {
        let base_url = base_url.trim().trim_end_matches('/');
        if !(base_url.starts_with("http://") || base_url.starts_with("https://")) {
            return Err(ConfigError::InvalidBaseUrl {
                url: base_url.to_owned(),
            });
        }
        Ok(Self {
            base_url: base_url.to_owned(),
            api_key: None,
            reasoning: ReasoningRequest::default(),
            connect_timeout: Duration::from_secs(10),
            // A local model may spend a while on prompt processing before the
            // first byte, so the wait for headers is generous.
            response_timeout: Duration::from_secs(300),
            read_timeout: Duration::from_secs(120),
        })
    }

    /// Reads [`BASE_URL_VAR`], [`API_KEY_VAR`], and [`REASONING_VAR`] through
    /// `lookup`.
    ///
    /// Pass `|name| std::env::var(name).ok()` to read the process environment.
    /// The base URL defaults to [`DEFAULT_BASE_URL`]. The API key is optional,
    /// because a local `llama-server` does not require one. The reasoning
    /// request defaults to [`ReasoningRequest::LlamaCpp`].
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError::InvalidBaseUrl`] when the base URL is not HTTP(S),
    /// and [`ConfigError::InvalidReasoning`] when [`REASONING_VAR`] is neither
    /// `llama-cpp` nor `none`.
    pub fn from_env(lookup: impl Fn(&str) -> Option<String>) -> Result<Self, ConfigError> {
        let base_url = lookup(BASE_URL_VAR).unwrap_or_else(|| DEFAULT_BASE_URL.to_owned());
        let mut config = Self::new(&base_url)?;
        if let Some(key) = lookup(API_KEY_VAR).filter(|key| !key.trim().is_empty()) {
            config = config.api_key(ApiKey::new(key.trim()));
        }
        if let Some(reasoning) = lookup(REASONING_VAR).filter(|value| !value.trim().is_empty()) {
            config = config.reasoning(reasoning.parse()?);
        }
        Ok(config)
    }

    /// Chooses which reasoning fields requests carry.
    #[must_use]
    pub fn reasoning(mut self, reasoning: ReasoningRequest) -> Self {
        self.reasoning = reasoning;
        self
    }

    /// Sends `key` as a bearer token.
    #[must_use]
    pub fn api_key(mut self, key: ApiKey) -> Self {
        self.api_key = Some(key);
        self
    }

    /// The server's base URL, without a trailing slash.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }
}

/// Streams chat completions from an OpenAI-compatible server.
///
/// Each request runs on its own OS thread with blocking IO, so this adapter
/// needs no async runtime. Dropping the returned stream cancels the request:
/// the thread stops at the next chunk it receives, or when a timeout fires.
#[derive(Debug, Clone)]
pub struct OpenAiProvider {
    agent: ureq::Agent,
    config: OpenAiConfig,
}

impl OpenAiProvider {
    /// Creates a provider for `config`.
    pub fn new(config: OpenAiConfig) -> Self {
        let agent = ureq::Agent::config_builder()
            .timeout_connect(Some(config.connect_timeout))
            .timeout_recv_response(Some(config.response_timeout))
            .timeout_per_read(Some(config.read_timeout))
            .build()
            .into();
        Self { agent, config }
    }
}

type ChunkSender = mpsc::UnboundedSender<Result<super::ModelChunk, ModelError>>;

impl ModelProvider for OpenAiProvider {
    fn stream(&self, request: ModelRequest) -> ModelStream {
        let (sender, receiver) = mpsc::unbounded();
        let agent = self.agent.clone();
        let config = self.config.clone();
        let body = request_body(&request, self.config.reasoning).to_string();
        let spawned = std::thread::Builder::new()
            .name("harness-model-request".to_owned())
            .spawn(move || {
                if let Err(error) = run_request(&agent, &config, body, &sender) {
                    // A closed receiver means the caller cancelled; nobody
                    // is left to report the error to.
                    let _ = sender.unbounded_send(Err(error));
                }
            });
        match spawned {
            Ok(_) => Box::pin(receiver),
            Err(error) => Box::pin(stream::iter([Err(ModelError::Spawn(error))])),
        }
    }
}

fn run_request(
    agent: &ureq::Agent,
    config: &OpenAiConfig,
    body: String,
    sender: &ChunkSender,
) -> Result<(), ModelError> {
    let mut http = agent
        .post(format!("{}/chat/completions", config.base_url))
        .header("Content-Type", "application/json")
        .header("Accept", "text/event-stream");
    if let Some(ApiKey(key)) = &config.api_key {
        http = http.header("Authorization", format!("Bearer {key}"));
    }
    let response = http.send(body).map_err(|error| match error {
        ureq::Error::StatusCode(status) => ModelError::Status { status },
        other => ModelError::Transport(other),
    })?;

    let mut decoder = SseDecoder::default();
    let reader = BufReader::new(response.into_body().into_reader());
    for line in reader.lines() {
        let line = line.map_err(ModelError::Stream)?;
        for chunk in decoder.push_line(&line)? {
            if sender.unbounded_send(Ok(chunk)).is_err() {
                return Ok(());
            }
        }
        if decoder.is_done() {
            break;
        }
    }
    for chunk in decoder.finish()? {
        if sender.unbounded_send(Ok(chunk)).is_err() {
            return Ok(());
        }
    }
    Ok(())
}

fn request_body(request: &ModelRequest, reasoning: ReasoningRequest) -> Value {
    let mut messages = vec![json!({ "role": "system", "content": request.system.as_str() })];
    messages.extend(request.messages.iter().map(message_json));
    let mut body = serde_json::Map::new();
    body.insert("model".to_owned(), request.model.as_str().into());
    body.insert("stream".to_owned(), true.into());
    body.insert(
        "stream_options".to_owned(),
        json!({ "include_usage": true }),
    );
    body.insert("messages".to_owned(), messages.into());
    match reasoning {
        ReasoningRequest::LlamaCpp => {
            body.insert("reasoning_format".to_owned(), "deepseek".into());
            body.insert(
                "chat_template_kwargs".to_owned(),
                json!({ "enable_thinking": true }),
            );
        }
        ReasoningRequest::None => {}
    }
    if !request.tools.is_empty() {
        body.insert(
            "tools".to_owned(),
            request.tools.iter().map(tool_json).collect(),
        );
    }
    Value::Object(body)
}

fn message_json(message: &Message) -> Value {
    match message {
        Message::User(text) => json!({ "role": "user", "content": text.as_str() }),
        Message::Assistant {
            reasoning,
            text,
            tool_calls,
        } => {
            let mut json = if tool_calls.is_empty() {
                json!({ "role": "assistant", "content": text })
            } else {
                json!({
                    "role": "assistant",
                    "content": if text.is_empty() { Value::Null } else { Value::from(text.as_str()) },
                    "tool_calls": tool_calls.iter().map(|call| json!({
                        "id": call.id.as_str(),
                        "type": "function",
                        "function": { "name": call.name.as_str(), "arguments": call.arguments.as_str() },
                    })).collect::<Vec<_>>(),
                })
            };
            if let (Some(reasoning), Some(object)) = (reasoning, json.as_object_mut()) {
                object.insert("reasoning_content".to_owned(), reasoning.as_str().into());
            }
            json
        }
        Message::Tool { call_id, content } => json!({
            "role": "tool",
            "tool_call_id": call_id.as_str(),
            "content": content,
        }),
    }
}

fn tool_json(spec: &ToolSpec) -> Value {
    json!({
        "type": "function",
        "function": {
            "name": spec.name.as_str(),
            "description": spec.description,
            "parameters": spec.parameters,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        conversation::{Reasoning, SystemPrompt, UserText},
        model::ModelId,
        tool::{ToolArguments, ToolCall, ToolCallId, ToolName},
    };

    fn env(vars: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let vars: Vec<(String, String)> = vars
            .iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect();
        move |name| vars.iter().find(|(k, _)| k == name).map(|(_, v)| v.clone())
    }

    #[test]
    fn defaults_to_local_llama_server() {
        let config = OpenAiConfig::from_env(env(&[])).unwrap();
        assert_eq!(config.base_url(), DEFAULT_BASE_URL);
        assert_eq!(config.api_key, None);
    }

    #[test]
    fn reads_key_and_url_from_env() {
        let config = OpenAiConfig::from_env(env(&[
            (BASE_URL_VAR, "https://api.example.com/v1/"),
            (API_KEY_VAR, "sk-test"),
        ]))
        .unwrap();
        assert_eq!(config.base_url(), "https://api.example.com/v1");
        assert_eq!(config.api_key, Some(ApiKey::new("sk-test")));
    }

    #[test]
    fn rejects_non_http_base_url() {
        assert!(matches!(
            OpenAiConfig::new("ftp://host"),
            Err(ConfigError::InvalidBaseUrl { .. })
        ));
    }

    #[test]
    fn api_key_debug_is_redacted() {
        assert_eq!(
            format!("{:?}", ApiKey::new("sk-secret")),
            "ApiKey(<redacted>)"
        );
    }

    #[test]
    fn serializes_tool_round_trip_messages() {
        let call = ToolCall {
            id: ToolCallId::new("call_1").unwrap(),
            name: ToolName::new("read_file").unwrap(),
            arguments: ToolArguments::new(r#"{"path":"a"}"#),
        };
        let body = request_body(
            &ModelRequest {
                model: ModelId::new("bonsai-2-9b").unwrap(),
                system: SystemPrompt::new("be brief"),
                messages: vec![
                    Message::User(UserText::parse("read a").unwrap()),
                    Message::Assistant {
                        reasoning: None,
                        text: String::new(),
                        tool_calls: vec![call],
                    },
                    Message::Tool {
                        call_id: ToolCallId::new("call_1").unwrap(),
                        content: "hi".to_owned(),
                    },
                ],
                tools: ToolSpec::builtin(),
            },
            ReasoningRequest::None,
        );
        assert!(body.get("reasoning_format").is_none());
        assert!(body.get("chat_template_kwargs").is_none());
        assert!(body["messages"][2].get("reasoning_content").is_none());
        assert_eq!(body["model"], "bonsai-2-9b");
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["messages"][2]["content"], Value::Null);
        assert_eq!(
            body["messages"][2]["tool_calls"][0]["function"]["name"],
            "read_file"
        );
        assert_eq!(body["messages"][3]["tool_call_id"], "call_1");
        assert_eq!(body["tools"][0]["function"]["name"], "read_file");
    }

    #[test]
    fn llama_cpp_requests_ask_for_separate_reasoning() {
        let body = request_body(
            &ModelRequest {
                model: ModelId::new("bonsai-2-9b").unwrap(),
                system: SystemPrompt::new("be brief"),
                messages: vec![
                    Message::User(UserText::parse("hi").unwrap()),
                    Message::Assistant {
                        reasoning: Some(Reasoning::new("greet back").unwrap()),
                        text: "Hello!".to_owned(),
                        tool_calls: Vec::new(),
                    },
                ],
                tools: Vec::new(),
            },
            ReasoningRequest::default(),
        );
        assert_eq!(body["reasoning_format"], "deepseek");
        assert_eq!(body["chat_template_kwargs"]["enable_thinking"], true);
        assert_eq!(body["messages"][2]["reasoning_content"], "greet back");
        assert_eq!(body["messages"][2]["content"], "Hello!");
    }

    #[test]
    fn reads_reasoning_request_from_env() {
        let config = OpenAiConfig::from_env(env(&[(REASONING_VAR, "none")])).unwrap();
        assert_eq!(config.reasoning, ReasoningRequest::None);
        assert_eq!(
            OpenAiConfig::from_env(env(&[])).unwrap().reasoning,
            ReasoningRequest::LlamaCpp
        );
    }

    #[test]
    fn rejects_unknown_reasoning_request() {
        assert!(matches!(
            OpenAiConfig::from_env(env(&[(REASONING_VAR, "maybe")])),
            Err(ConfigError::InvalidReasoning(_))
        ));
    }
}
