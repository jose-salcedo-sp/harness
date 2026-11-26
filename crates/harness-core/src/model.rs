//! Models, token accounting, and the [`ModelProvider`] port.
//!
//! Adapters: [`openai::OpenAiProvider`] talks to any OpenAI-compatible
//! chat-completions server over HTTP, and [`fake::FakeProvider`] replies
//! deterministically for tests and offline demos.

pub mod fake;
pub mod openai;
mod think_tags;

use std::{fmt, num::NonZeroU32, str::FromStr};

use futures::stream::BoxStream;

use crate::{
    conversation::{Message, SystemPrompt},
    tool::{ToolCall, ToolSpec},
};

/// Identifier a provider uses to select a model, such as `bonsai-2-9b`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ModelId(String);

impl ModelId {
    /// Wraps a model identifier.
    ///
    /// # Errors
    ///
    /// Returns [`CatalogError::EmptyId`] when `id` is empty or whitespace.
    pub fn new(id: impl Into<String>) -> Result<Self, CatalogError> {
        let id = id.into();
        if id.trim().is_empty() {
            Err(CatalogError::EmptyId)
        } else {
            Ok(Self(id))
        }
    }

    /// The identifier string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ModelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// How many tokens a model can attend to at once.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ContextWindow(NonZeroU32);

impl ContextWindow {
    /// A window of `tokens` tokens.
    pub const fn new(tokens: NonZeroU32) -> Self {
        Self(tokens)
    }

    /// Size in tokens.
    pub const fn tokens(self) -> u32 {
        self.0.get()
    }
}

/// A model the user can pick.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ModelInfo {
    /// Identifier sent to the provider.
    pub id: ModelId,
    /// Human-readable name for the UI.
    pub label: String,
    /// Context window size.
    pub context_window: ContextWindow,
}

/// Why a model catalog could not be parsed.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CatalogError {
    /// A model identifier was empty.
    #[error("model id is empty")]
    EmptyId,
    /// The catalog listed no models.
    #[error("model catalog is empty")]
    Empty,
    /// An entry was not `id=tokens`.
    #[error("model entry `{entry}` is not `id=context_tokens`")]
    MalformedEntry {
        /// The offending entry.
        entry: String,
    },
    /// The context window was not a positive integer.
    #[error("model entry `{entry}` has an invalid context window")]
    InvalidContextWindow {
        /// The offending entry.
        entry: String,
        /// The parse failure.
        #[source]
        source: std::num::ParseIntError,
    },
}

/// The models the user can switch between. Never empty.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelCatalog {
    first: ModelInfo,
    rest: Vec<ModelInfo>,
}

impl ModelCatalog {
    /// A catalog with `first` as the default model.
    pub fn new(first: ModelInfo, rest: Vec<ModelInfo>) -> Self {
        Self { first, rest }
    }

    /// The model selected at startup.
    pub fn default_model(&self) -> &ModelInfo {
        &self.first
    }

    /// Every model, default first.
    pub fn iter(&self) -> impl Iterator<Item = &ModelInfo> {
        std::iter::once(&self.first).chain(&self.rest)
    }

    /// Looks a model up by identifier.
    pub fn get(&self, id: &ModelId) -> Option<&ModelInfo> {
        self.iter().find(|model| &model.id == id)
    }
}

impl FromStr for ModelCatalog {
    type Err = CatalogError;

    /// Parses a comma-separated list of `id=context_tokens` entries, such as
    /// `bonsai-2-9b=32768,bonsai-2-4b=32768`. The first entry is the default.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut models = s
            .split(',')
            .map(str::trim)
            .filter(|entry| !entry.is_empty())
            .map(parse_catalog_entry);
        let first = models.next().ok_or(CatalogError::Empty)??;
        let rest = models.collect::<Result<Vec<_>, _>>()?;
        Ok(Self::new(first, rest))
    }
}

fn parse_catalog_entry(entry: &str) -> Result<ModelInfo, CatalogError> {
    let (id, tokens) = entry
        .split_once('=')
        .ok_or_else(|| CatalogError::MalformedEntry {
            entry: entry.to_owned(),
        })?;
    let tokens = tokens.trim().parse::<NonZeroU32>().map_err(|source| {
        CatalogError::InvalidContextWindow {
            entry: entry.to_owned(),
            source,
        }
    })?;
    let id = ModelId::new(id.trim())?;
    Ok(ModelInfo {
        label: id.as_str().to_owned(),
        id,
        context_window: ContextWindow::new(tokens),
    })
}

/// Rough token count for `chars` characters of text (about four per token),
/// used when a provider does not report usage.
pub(crate) fn estimate_tokens(chars: usize) -> u32 {
    u32::try_from(chars.div_ceil(4)).unwrap_or(u32::MAX)
}

/// Token counts for one model response, as the provider reported them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct TokenUsage {
    /// Tokens in the request (the whole conversation so far).
    pub prompt_tokens: u32,
    /// Tokens the model generated.
    pub completion_tokens: u32,
}

impl TokenUsage {
    /// Tokens the conversation occupies after this response.
    pub fn total(self) -> u32 {
        self.prompt_tokens.saturating_add(self.completion_tokens)
    }
}

/// How much of the selected model's context window the conversation uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ContextUsage {
    /// Tokens used.
    pub used: u32,
    /// The window they count against.
    pub window: ContextWindow,
}

impl ContextUsage {
    /// Share of the window in use, as a whole percentage clamped to 100.
    pub fn percent(self) -> u8 {
        let percent = u64::from(self.used) * 100 / u64::from(self.window.tokens());
        u8::try_from(percent.min(100)).unwrap_or(100)
    }
}

/// Everything a provider needs to produce the next model response.
///
/// The system prompt is required, so a request always has at least one message.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelRequest {
    /// Which model to run.
    pub model: ModelId,
    /// Instructions sent first.
    pub system: SystemPrompt,
    /// The conversation so far, oldest first.
    pub messages: Vec<Message>,
    /// Tools the model may call.
    pub tools: Vec<ToolSpec>,
}

/// One piece of a streamed model response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelChunk {
    /// More of the model's thinking trace, kept apart from the reply text.
    Reasoning(String),
    /// More reply text.
    Text(String),
    /// A complete tool call.
    ToolCall(ToolCall),
    /// Token counts for this response.
    Usage(TokenUsage),
}

/// Why a model request failed.
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    /// The HTTP request could not be sent or the connection failed.
    #[error("could not reach the model server")]
    Transport(#[source] ureq::Error),
    /// The server answered with an error status.
    #[error("model server returned HTTP {status}")]
    Status {
        /// The HTTP status code.
        status: u16,
    },
    /// Reading the streamed body failed partway through.
    #[error("model stream was interrupted")]
    Stream(#[source] std::io::Error),
    /// A streamed event was not the JSON the protocol defines.
    #[error("model sent a malformed stream event")]
    Decode(#[source] serde_json::Error),
    /// A tool call finished without an id or a name.
    #[error("model sent tool call #{index} without an id or name")]
    IncompleteToolCall {
        /// Position of the call in the response.
        index: u64,
    },
    /// The background thread that reads the response could not start.
    #[error("could not start the model request thread")]
    Spawn(#[source] std::io::Error),
}

/// A streamed model response. Dropping it cancels the request.
pub type ModelStream = BoxStream<'static, Result<ModelChunk, ModelError>>;

/// Port: produces streamed model responses.
pub trait ModelProvider: Send + Sync {
    /// Starts a response to `request`.
    ///
    /// The stream yields chunks as they arrive and ends when the response is
    /// complete. A [`ModelError`] item means the response failed; the stream
    /// yields nothing after it.
    fn stream(&self, request: ModelRequest) -> ModelStream;
}

impl<P: ModelProvider + ?Sized> ModelProvider for Box<P> {
    fn stream(&self, request: ModelRequest) -> ModelStream {
        (**self).stream(request)
    }
}

impl<P: ModelProvider + ?Sized> ModelProvider for std::sync::Arc<P> {
    fn stream(&self, request: ModelRequest) -> ModelStream {
        (**self).stream(request)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn window(tokens: u32) -> ContextWindow {
        ContextWindow::new(NonZeroU32::new(tokens).unwrap())
    }

    #[test]
    fn parses_catalog_with_default_first() {
        let catalog: ModelCatalog = "a=100, b=200".parse().unwrap();
        assert_eq!(catalog.default_model().id.as_str(), "a");
        let windows: Vec<_> = catalog.iter().map(|m| m.context_window.tokens()).collect();
        assert_eq!(windows, [100, 200]);
    }

    #[test]
    fn rejects_empty_catalog() {
        assert_eq!(" , ".parse::<ModelCatalog>(), Err(CatalogError::Empty));
    }

    #[test]
    fn rejects_entry_without_window() {
        assert!(matches!(
            "a".parse::<ModelCatalog>(),
            Err(CatalogError::MalformedEntry { .. })
        ));
    }

    #[test]
    fn rejects_zero_context_window() {
        assert!(matches!(
            "a=0".parse::<ModelCatalog>(),
            Err(CatalogError::InvalidContextWindow { .. })
        ));
    }

    #[test]
    fn rejects_blank_model_id() {
        assert_eq!(" =10".parse::<ModelCatalog>(), Err(CatalogError::EmptyId));
    }

    #[test]
    fn percent_is_clamped() {
        let usage = ContextUsage {
            used: 300,
            window: window(200),
        };
        assert_eq!(usage.percent(), 100);
    }

    #[test]
    fn percent_rounds_down() {
        let usage = ContextUsage {
            used: 1,
            window: window(3),
        };
        assert_eq!(usage.percent(), 33);
    }
}
