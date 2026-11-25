//! UI-free backend for the harness chatbot: domain types, the agent loop, and
//! the ports (traits) through which it reaches models, tools, and storage.
//!
//! This crate must not depend on GPUI or any other UI toolkit. Adapters that
//! perform real IO implement the ports defined here; tests use in-memory fakes.
//! See `CODING_STANDARDS.md` at the repository root for the rules this crate
//! follows.
//!
//! - [`agent`]: the [`AgentLoop`] state machine and the [`Session`] driver.
//! - [`model`]: the [`ModelProvider`] port, with [`model::openai`] (HTTP) and
//!   [`model::fake`] adapters.
//! - [`tool`]: the [`ToolExecutor`] port, with [`FsTools`] and [`MemoryTools`].
//! - [`conversation`]: messages exchanged with the model.

pub mod agent;
pub mod conversation;
pub mod model;
pub mod tool;

pub use agent::{
    AgentEvent, AgentLoop, Command, Effect, Input, Session, Status, ToolOutcome, TransitionError,
    TurnFailure,
};
pub use conversation::{EmptyMessage, Message, SystemPrompt, UserText};
pub use model::{
    CatalogError, ContextUsage, ContextWindow, ModelCatalog, ModelChunk, ModelError, ModelId,
    ModelInfo, ModelProvider, ModelRequest, ModelStream, TokenUsage,
};
pub use tool::{
    FsTools, MemoryTools, ToolArguments, ToolCall, ToolCallId, ToolError, ToolExecutor, ToolName,
    ToolOutput, ToolRequest, ToolSpec,
};
