//! The agent loop: alternate model inference and tool execution until the
//! model answers without asking for a tool.
//!
//! [`AgentLoop`] is a pure state machine. [`AgentLoop::handle`] takes an
//! [`Input`] and returns the [`Effect`]s to perform; it never does IO itself.
//! [`Session`] is the driver that performs those effects through the
//! [`ModelProvider`](crate::model::ModelProvider) and
//! [`ToolExecutor`](crate::tool::ToolExecutor) ports and feeds the results back.
//!
//! ```text
//!            Submit                      response has tool calls
//!   Idle ─────────────▶ Inferring ─────────────────────────────▶ RunningTools
//!    ▲                   │   ▲                                        │
//!    │  response without │   └──────── last tool result arrives ──────┘
//!    └─── tool calls, ───┘             (Idle instead at the round limit)
//!         failure, or Cancel
//! ```

mod session;

use std::{num::NonZeroU32, sync::Arc};

pub use session::{Command, Session};

use crate::{
    conversation::{Message, Reasoning, SystemPrompt, UserText, request_history},
    model::{
        ContextUsage, ModelChunk, ModelError, ModelInfo, ModelRequest, TokenUsage, estimate_tokens,
    },
    tool::{ToolCall, ToolCallId, ToolError, ToolOutput, ToolSpec},
};

/// How many model responses one turn may take before the loop gives up.
pub const DEFAULT_MAX_ROUNDS: NonZeroU32 = match NonZeroU32::new(8) {
    Some(rounds) => rounds,
    None => NonZeroU32::MIN,
};

/// Tool result recorded for calls the user cancelled.
const CANCELLED_TOOL_RESULT: &str = "error: cancelled by the user";

/// Which step of a turn the loop is in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Status {
    /// Waiting for the user.
    Idle,
    /// Streaming a model response.
    Inferring,
    /// Waiting for tool results.
    RunningTools,
}

/// The model response being streamed.
#[derive(Debug, Default)]
struct Draft {
    reasoning: String,
    /// Whether the last content chunk was reasoning, so the next other chunk
    /// (or the end of the response) closes the reasoning block.
    thinking: bool,
    text: String,
    tool_calls: Vec<ToolCall>,
    usage: Option<TokenUsage>,
}

impl Draft {
    /// Emits [`AgentEvent::ReasoningFinished`] if a reasoning block is open.
    fn close_reasoning(&mut self, effects: &mut Vec<Effect>) {
        if std::mem::take(&mut self.thinking) {
            effects.push(Effect::Emit(AgentEvent::ReasoningFinished));
        }
    }

    fn reasoning(&self) -> Option<Reasoning> {
        // An empty trace just means the model did not think.
        Reasoning::new(&self.reasoning).ok()
    }

    /// The assistant message for what was streamed, or `None` when the
    /// response was empty.
    fn into_message(self) -> Option<Message> {
        let reasoning = self.reasoning();
        let empty = reasoning.is_none() && self.text.is_empty() && self.tool_calls.is_empty();
        (!empty).then_some(Message::Assistant {
            reasoning,
            text: self.text,
            tool_calls: self.tool_calls,
        })
    }
}

#[derive(Debug)]
enum Phase {
    Idle,
    Inferring {
        round: NonZeroU32,
        draft: Draft,
    },
    RunningTools {
        round: NonZeroU32,
        pending: Vec<ToolCallId>,
    },
}

/// Something that happened, fed into [`AgentLoop::handle`].
#[derive(Debug)]
pub enum Input {
    /// The user sent a message.
    Submit(UserText),
    /// The user picked another model.
    SelectModel(ModelInfo),
    /// The user asked to stop the current turn.
    Cancel,
    /// The model streamed a chunk.
    ModelChunk(ModelChunk),
    /// The model response ended normally.
    ModelFinished,
    /// The model response failed.
    ModelFailed(ModelError),
    /// A tool call completed.
    ToolFinished {
        /// The call that completed.
        id: ToolCallId,
        /// What it returned.
        result: Result<ToolOutput, ToolError>,
    },
}

/// Work the driver must do after a transition, in order.
#[derive(Debug)]
pub enum Effect {
    /// Tell the UI.
    Emit(AgentEvent),
    /// Start a model response; feed its chunks back as [`Input::ModelChunk`].
    Infer(ModelRequest),
    /// Run a tool; feed its result back as [`Input::ToolFinished`].
    RunTool(ToolCall),
}

/// How a tool call ended, as shown to the user.
#[derive(Debug, Clone)]
pub enum ToolOutcome {
    /// The tool returned output.
    Succeeded {
        /// The output.
        output: ToolOutput,
    },
    /// The tool failed; the error was reported to the model.
    Failed {
        /// The failure.
        error: Arc<ToolError>,
    },
    /// The user cancelled the turn before the tool finished.
    Cancelled,
}

/// Why a turn ended without an answer.
#[derive(Debug, Clone)]
pub enum TurnFailure {
    /// The model request failed.
    Model(Arc<ModelError>),
    /// The model kept calling tools past the round limit.
    RoundLimit {
        /// The limit that was hit.
        limit: NonZeroU32,
    },
}

/// Updates the UI renders, in the order they happen.
#[derive(Debug, Clone)]
pub enum AgentEvent {
    /// The user's message was accepted and a turn started.
    UserMessage(UserText),
    /// The model streamed more of its thinking trace. The first delta after
    /// any other event opens a new reasoning block.
    ReasoningDelta(String),
    /// The open reasoning block ended: the model started replying or calling
    /// tools, or the response ended. Always follows a [`Self::ReasoningDelta`]
    /// and precedes the events of whatever came next.
    ReasoningFinished,
    /// The model streamed more reply text.
    TextDelta(String),
    /// A tool call is about to run.
    ToolCallStarted(ToolCall),
    /// A tool call ended.
    ToolCallFinished {
        /// The call that ended.
        id: ToolCallId,
        /// How it ended.
        outcome: ToolOutcome,
    },
    /// Context-window usage changed.
    ContextUsage(ContextUsage),
    /// The selected model changed.
    ModelChanged(ModelInfo),
    /// The model answered; the loop is idle again.
    TurnFinished,
    /// The turn failed; the loop is idle again.
    TurnFailed(TurnFailure),
    /// The user cancelled the turn; the loop is idle again.
    TurnCancelled,
    /// A command was ignored because it is not valid right now.
    Rejected(TransitionError),
}

/// An [`Input`] that is not valid in the current [`Status`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TransitionError {
    /// The user acted while a turn was in progress.
    #[error("wait for the current reply to finish")]
    Busy,
    /// Model output arrived while no response was in progress.
    #[error("no model response is in progress")]
    NotInferring,
    /// A tool result arrived while no tools were running.
    #[error("no tool calls are running")]
    NotRunningTools,
    /// A tool result arrived for a call that is not pending.
    #[error("tool call `{id}` is not pending")]
    UnknownToolCall {
        /// The unexpected call id.
        id: ToolCallId,
    },
}

/// The agent loop state machine. See the [module docs](self).
#[derive(Debug)]
pub struct AgentLoop {
    model: ModelInfo,
    system: SystemPrompt,
    tools: Vec<ToolSpec>,
    max_rounds: NonZeroU32,
    messages: Vec<Message>,
    context_used: u32,
    phase: Phase,
}

impl AgentLoop {
    /// An idle loop with an empty conversation, the default system prompt,
    /// and the built-in tools.
    ///
    /// # Examples
    ///
    /// ```
    /// use harness_core::{AgentLoop, Effect, Input, ModelCatalog, Status, UserText};
    ///
    /// let catalog: ModelCatalog = "fake=4096".parse()?;
    /// let mut agent = AgentLoop::new(catalog.default_model().clone());
    /// let effects = agent.handle(Input::Submit(UserText::parse("hi")?))?;
    /// assert_eq!(agent.status(), Status::Inferring);
    /// assert!(effects.iter().any(|e| matches!(e, Effect::Infer(_))));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(model: ModelInfo) -> Self {
        Self {
            model,
            system: SystemPrompt::default(),
            tools: ToolSpec::builtin(),
            max_rounds: DEFAULT_MAX_ROUNDS,
            messages: Vec::new(),
            context_used: 0,
            phase: Phase::Idle,
        }
    }

    /// Replaces the system prompt.
    #[must_use]
    pub fn system_prompt(mut self, system: SystemPrompt) -> Self {
        self.system = system;
        self
    }

    /// Replaces the tools offered to the model.
    #[must_use]
    pub fn tools(mut self, tools: Vec<ToolSpec>) -> Self {
        self.tools = tools;
        self
    }

    /// Sets how many model responses one turn may take.
    #[must_use]
    pub fn max_rounds(mut self, max_rounds: NonZeroU32) -> Self {
        self.max_rounds = max_rounds;
        self
    }

    /// The current step.
    pub fn status(&self) -> Status {
        match self.phase {
            Phase::Idle => Status::Idle,
            Phase::Inferring { .. } => Status::Inferring,
            Phase::RunningTools { .. } => Status::RunningTools,
        }
    }

    /// The selected model.
    pub fn model(&self) -> &ModelInfo {
        &self.model
    }

    /// The conversation so far, oldest first.
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// How much of the model's context window the conversation uses.
    pub fn context_usage(&self) -> ContextUsage {
        ContextUsage {
            used: self.context_used,
            window: self.model.context_window,
        }
    }

    /// Applies `input` and returns the effects to perform, in order.
    ///
    /// [`Input::Cancel`] while idle is a no-op, because a cancel can race with
    /// the end of a turn.
    ///
    /// # Errors
    ///
    /// Returns a [`TransitionError`] when `input` is not valid in the current
    /// [`Status`]. The state is unchanged in that case.
    pub fn handle(&mut self, input: Input) -> Result<Vec<Effect>, TransitionError> {
        let phase = std::mem::replace(&mut self.phase, Phase::Idle);
        let (next, result) = self.transition(phase, input);
        self.phase = next;
        result
    }

    fn transition(
        &mut self,
        phase: Phase,
        input: Input,
    ) -> (Phase, Result<Vec<Effect>, TransitionError>) {
        match (phase, input) {
            (Phase::Idle, Input::Submit(text)) => {
                self.messages.push(Message::User(text.clone()));
                let effects = vec![
                    Effect::Emit(AgentEvent::UserMessage(text)),
                    Effect::Infer(self.request()),
                ];
                (Self::inferring(NonZeroU32::MIN), Ok(effects))
            }
            (Phase::Idle, Input::SelectModel(model)) => {
                self.model = model.clone();
                let effects = vec![
                    Effect::Emit(AgentEvent::ModelChanged(model)),
                    Effect::Emit(AgentEvent::ContextUsage(self.context_usage())),
                ];
                (Phase::Idle, Ok(effects))
            }
            (Phase::Idle, Input::Cancel) => (Phase::Idle, Ok(Vec::new())),
            (
                phase @ (Phase::Inferring { .. } | Phase::RunningTools { .. }),
                Input::Submit(_) | Input::SelectModel(_),
            ) => (phase, Err(TransitionError::Busy)),
            (Phase::Inferring { round, mut draft }, Input::ModelChunk(chunk)) => {
                let effects = self.on_chunk(&mut draft, chunk);
                (Phase::Inferring { round, draft }, Ok(effects))
            }
            (Phase::Inferring { round, draft }, Input::ModelFinished) => {
                self.finish_response(round, draft)
            }
            (Phase::Inferring { draft, .. }, Input::ModelFailed(error)) => {
                let failure = TurnFailure::Model(Arc::new(error));
                let effects = self.keep_partial_reply(draft, AgentEvent::TurnFailed(failure));
                (Phase::Idle, Ok(effects))
            }
            (Phase::Inferring { draft, .. }, Input::Cancel) => {
                let effects = self.keep_partial_reply(draft, AgentEvent::TurnCancelled);
                (Phase::Idle, Ok(effects))
            }
            (Phase::RunningTools { round, pending }, Input::ToolFinished { id, result }) => {
                self.tool_finished(round, pending, id, result)
            }
            (Phase::RunningTools { pending, .. }, Input::Cancel) => {
                (Phase::Idle, Ok(self.cancel_tools(pending)))
            }
            (
                phase @ (Phase::Idle | Phase::RunningTools { .. }),
                Input::ModelChunk(_) | Input::ModelFinished | Input::ModelFailed(_),
            ) => (phase, Err(TransitionError::NotInferring)),
            (phase @ (Phase::Idle | Phase::Inferring { .. }), Input::ToolFinished { .. }) => {
                (phase, Err(TransitionError::NotRunningTools))
            }
        }
    }

    fn on_chunk(&mut self, draft: &mut Draft, chunk: ModelChunk) -> Vec<Effect> {
        match chunk {
            ModelChunk::Reasoning(delta) => {
                if !draft.thinking && !draft.reasoning.is_empty() {
                    draft.reasoning.push_str("\n\n");
                }
                draft.thinking = true;
                draft.reasoning.push_str(&delta);
                vec![Effect::Emit(AgentEvent::ReasoningDelta(delta))]
            }
            ModelChunk::Text(delta) => {
                let mut effects = Vec::new();
                draft.close_reasoning(&mut effects);
                draft.text.push_str(&delta);
                effects.push(Effect::Emit(AgentEvent::TextDelta(delta)));
                effects
            }
            ModelChunk::ToolCall(call) => {
                let mut effects = Vec::new();
                draft.close_reasoning(&mut effects);
                draft.tool_calls.push(call);
                effects
            }
            ModelChunk::Usage(reported) => {
                draft.usage = Some(reported);
                self.context_used = reported.total();
                vec![Effect::Emit(AgentEvent::ContextUsage(self.context_usage()))]
            }
        }
    }

    fn cancel_tools(&mut self, pending: Vec<ToolCallId>) -> Vec<Effect> {
        let mut effects = Vec::with_capacity(pending.len() + 1);
        for id in pending {
            self.messages.push(Message::Tool {
                call_id: id.clone(),
                content: CANCELLED_TOOL_RESULT.to_owned(),
            });
            effects.push(Effect::Emit(AgentEvent::ToolCallFinished {
                id,
                outcome: ToolOutcome::Cancelled,
            }));
        }
        effects.push(Effect::Emit(AgentEvent::TurnCancelled));
        effects
    }

    fn inferring(round: NonZeroU32) -> Phase {
        Phase::Inferring {
            round,
            draft: Draft::default(),
        }
    }

    fn request(&self) -> ModelRequest {
        ModelRequest {
            model: self.model.id.clone(),
            system: self.system.clone(),
            messages: request_history(&self.messages),
            tools: self.tools.clone(),
        }
    }

    /// Records what streamed before a failure or cancel, then emits `ending`.
    /// Tool calls are dropped because they will never run.
    fn keep_partial_reply(&mut self, mut draft: Draft, ending: AgentEvent) -> Vec<Effect> {
        let mut effects = Vec::new();
        draft.close_reasoning(&mut effects);
        draft.tool_calls.clear();
        self.messages.extend(draft.into_message());
        effects.push(Effect::Emit(ending));
        effects
    }

    fn estimate_context(&self) -> u32 {
        let chars =
            self.system.as_str().len() + self.messages.iter().map(Message::char_len).sum::<usize>();
        estimate_tokens(chars)
    }

    fn finish_response(
        &mut self,
        round: NonZeroU32,
        mut draft: Draft,
    ) -> (Phase, Result<Vec<Effect>, TransitionError>) {
        let mut effects = Vec::new();
        draft.close_reasoning(&mut effects);
        let usage = draft.usage;
        let tool_calls = draft.tool_calls.clone();
        self.messages.push(Message::Assistant {
            reasoning: draft.reasoning(),
            text: draft.text,
            tool_calls: draft.tool_calls,
        });
        if usage.is_none() {
            self.context_used = self.estimate_context();
            effects.push(Effect::Emit(AgentEvent::ContextUsage(self.context_usage())));
        }
        if tool_calls.is_empty() {
            effects.push(Effect::Emit(AgentEvent::TurnFinished));
            return (Phase::Idle, Ok(effects));
        }
        let pending = tool_calls.iter().map(|call| call.id.clone()).collect();
        for call in tool_calls {
            effects.push(Effect::Emit(AgentEvent::ToolCallStarted(call.clone())));
            effects.push(Effect::RunTool(call));
        }
        (Phase::RunningTools { round, pending }, Ok(effects))
    }

    fn tool_finished(
        &mut self,
        round: NonZeroU32,
        mut pending: Vec<ToolCallId>,
        id: ToolCallId,
        result: Result<ToolOutput, ToolError>,
    ) -> (Phase, Result<Vec<Effect>, TransitionError>) {
        let Some(position) = pending.iter().position(|pending_id| *pending_id == id) else {
            return (
                Phase::RunningTools { round, pending },
                Err(TransitionError::UnknownToolCall { id }),
            );
        };
        pending.remove(position);

        let (content, outcome) = match result {
            Ok(output) => (
                output.as_str().to_owned(),
                ToolOutcome::Succeeded { output },
            ),
            Err(error) => (
                error.to_model_text(),
                ToolOutcome::Failed {
                    error: Arc::new(error),
                },
            ),
        };
        self.messages.push(Message::Tool {
            call_id: id.clone(),
            content,
        });
        let mut effects = vec![Effect::Emit(AgentEvent::ToolCallFinished { id, outcome })];

        if !pending.is_empty() {
            return (Phase::RunningTools { round, pending }, Ok(effects));
        }
        if let Some(next) = round.checked_add(1).filter(|next| *next <= self.max_rounds) {
            effects.push(Effect::Infer(self.request()));
            return (Self::inferring(next), Ok(effects));
        }
        let failure = TurnFailure::RoundLimit {
            limit: self.max_rounds,
        };
        effects.push(Effect::Emit(AgentEvent::TurnFailed(failure)));
        (Phase::Idle, Ok(effects))
    }
}

impl From<Command> for Input {
    fn from(command: Command) -> Self {
        match command {
            Command::Submit(text) => Self::Submit(text),
            Command::SelectModel(model) => Self::SelectModel(model),
            Command::Cancel => Self::Cancel,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        model::{ContextWindow, ModelId},
        tool::{ToolArguments, ToolName},
    };

    fn model(id: &str, window: u32) -> ModelInfo {
        ModelInfo {
            id: ModelId::new(id).unwrap(),
            label: id.to_owned(),
            context_window: ContextWindow::new(NonZeroU32::new(window).unwrap()),
        }
    }

    fn agent() -> AgentLoop {
        AgentLoop::new(model("m", 1000))
    }

    fn submit(text: &str) -> Input {
        Input::Submit(UserText::parse(text).unwrap())
    }

    fn call(id: &str) -> ToolCall {
        ToolCall {
            id: ToolCallId::new(id).unwrap(),
            name: ToolName::new("read_file").unwrap(),
            arguments: ToolArguments::new(r#"{"path":"a"}"#),
        }
    }

    fn tool_ok(id: &str) -> Input {
        Input::ToolFinished {
            id: ToolCallId::new(id).unwrap(),
            result: Ok(ToolOutput::new("contents")),
        }
    }

    fn inferring_agent() -> AgentLoop {
        let mut agent = agent();
        agent.handle(submit("hi")).unwrap();
        agent
    }

    fn running_tools_agent(ids: &[&str]) -> AgentLoop {
        let mut agent = inferring_agent();
        for id in ids {
            agent
                .handle(Input::ModelChunk(ModelChunk::ToolCall(call(id))))
                .unwrap();
        }
        agent.handle(Input::ModelFinished).unwrap();
        assert_eq!(agent.status(), Status::RunningTools);
        agent
    }

    fn infer_count(effects: &[Effect]) -> usize {
        effects
            .iter()
            .filter(|effect| matches!(effect, Effect::Infer(_)))
            .count()
    }

    fn has_event(effects: &[Effect], predicate: impl Fn(&AgentEvent) -> bool) -> bool {
        effects
            .iter()
            .any(|effect| matches!(effect, Effect::Emit(event) if predicate(event)))
    }

    // Idle

    #[test]
    fn submit_starts_inference_with_user_message() {
        let mut agent = agent();
        let effects = agent.handle(submit("hello")).unwrap();
        assert_eq!(agent.status(), Status::Inferring);
        assert!(has_event(&effects, |e| matches!(
            e,
            AgentEvent::UserMessage(_)
        )));
        let Some(Effect::Infer(request)) = effects.last() else {
            panic!("expected an Infer effect");
        };
        assert_eq!(request.messages, agent.messages());
        assert_eq!(request.tools, ToolSpec::builtin());
    }

    #[test]
    fn select_model_while_idle_switches_window() {
        let mut agent = agent();
        let effects = agent
            .handle(Input::SelectModel(model("big", 9000)))
            .unwrap();
        assert_eq!(agent.model().id.as_str(), "big");
        assert!(has_event(&effects, |e| matches!(
            e,
            AgentEvent::ContextUsage(usage) if usage.window.tokens() == 9000
        )));
    }

    #[test]
    fn cancel_while_idle_is_a_no_op() {
        let mut agent = agent();
        assert_eq!(agent.handle(Input::Cancel).unwrap().len(), 0);
        assert_eq!(agent.status(), Status::Idle);
    }

    #[test]
    fn idle_rejects_model_output() {
        let mut agent = agent();
        assert_eq!(
            agent.handle(Input::ModelFinished).unwrap_err(),
            TransitionError::NotInferring
        );
    }

    #[test]
    fn idle_rejects_tool_results() {
        let mut agent = agent();
        assert_eq!(
            agent.handle(tool_ok("x")).unwrap_err(),
            TransitionError::NotRunningTools
        );
    }

    // Inferring

    #[test]
    fn text_chunks_accumulate_into_reply() {
        let mut agent = inferring_agent();
        for delta in ["Hel", "lo"] {
            let effects = agent
                .handle(Input::ModelChunk(ModelChunk::Text(delta.to_owned())))
                .unwrap();
            assert!(has_event(
                &effects,
                |e| matches!(e, AgentEvent::TextDelta(d) if d == delta)
            ));
        }
        let effects = agent.handle(Input::ModelFinished).unwrap();
        assert_eq!(agent.status(), Status::Idle);
        assert!(has_event(&effects, |e| matches!(
            e,
            AgentEvent::TurnFinished
        )));
        assert_eq!(
            agent.messages().last(),
            Some(&Message::Assistant {
                reasoning: None,
                text: "Hello".to_owned(),
                tool_calls: Vec::new()
            })
        );
    }

    fn reasoning_chunk(delta: &str) -> Input {
        Input::ModelChunk(ModelChunk::Reasoning(delta.to_owned()))
    }

    fn events(effects: Vec<Effect>) -> Vec<AgentEvent> {
        effects
            .into_iter()
            .filter_map(|effect| match effect {
                Effect::Emit(event) => Some(event),
                Effect::Infer(_) | Effect::RunTool(_) => None,
            })
            .collect()
    }

    #[test]
    fn reasoning_streams_then_closes_before_text() {
        let mut agent = inferring_agent();
        let mut seen = Vec::new();
        for input in [
            reasoning_chunk("Let me "),
            reasoning_chunk("think."),
            Input::ModelChunk(ModelChunk::Text("Hi".to_owned())),
            Input::ModelFinished,
        ] {
            seen.extend(events(agent.handle(input).unwrap()));
        }
        assert!(matches!(
            seen.as_slice(),
            [
                AgentEvent::ReasoningDelta(a),
                AgentEvent::ReasoningDelta(b),
                AgentEvent::ReasoningFinished,
                AgentEvent::TextDelta(c),
                AgentEvent::ContextUsage(_),
                AgentEvent::TurnFinished,
            ] if a == "Let me " && b == "think." && c == "Hi"
        ));
        assert_eq!(
            agent.messages().last(),
            Some(&Message::Assistant {
                reasoning: Some(Reasoning::new("Let me think.").unwrap()),
                text: "Hi".to_owned(),
                tool_calls: Vec::new()
            })
        );
    }

    #[test]
    fn reasoning_closes_before_tool_calls_start() {
        let mut agent = inferring_agent();
        agent.handle(reasoning_chunk("need a file")).unwrap();
        let on_call = events(
            agent
                .handle(Input::ModelChunk(ModelChunk::ToolCall(call("c1"))))
                .unwrap(),
        );
        assert!(matches!(
            on_call.as_slice(),
            [AgentEvent::ReasoningFinished]
        ));
        let on_finish = events(agent.handle(Input::ModelFinished).unwrap());
        assert!(
            !on_finish
                .iter()
                .any(|e| matches!(e, AgentEvent::ReasoningFinished))
        );
        assert!(matches!(
            on_finish.last(),
            Some(AgentEvent::ToolCallStarted(_))
        ));
    }

    #[test]
    fn reasoning_only_response_closes_when_it_ends() {
        let mut agent = inferring_agent();
        agent.handle(reasoning_chunk("hmm")).unwrap();
        let seen = events(agent.handle(Input::ModelFinished).unwrap());
        assert!(matches!(seen.first(), Some(AgentEvent::ReasoningFinished)));
        assert!(matches!(seen.last(), Some(AgentEvent::TurnFinished)));
    }

    #[test]
    fn reasoning_resumed_after_text_is_a_new_block() {
        let mut agent = inferring_agent();
        agent.handle(reasoning_chunk("one")).unwrap();
        agent
            .handle(Input::ModelChunk(ModelChunk::Text("a".to_owned())))
            .unwrap();
        agent.handle(reasoning_chunk("two")).unwrap();
        let seen = events(agent.handle(Input::ModelFinished).unwrap());
        assert!(matches!(seen.first(), Some(AgentEvent::ReasoningFinished)));
        assert!(matches!(
            agent.messages().last(),
            Some(Message::Assistant { reasoning: Some(r), .. }) if r.as_str() == "one\n\ntwo"
        ));
    }

    #[test]
    fn cancel_while_thinking_closes_and_keeps_reasoning() {
        let mut agent = inferring_agent();
        agent.handle(reasoning_chunk("halfway")).unwrap();
        let seen = events(agent.handle(Input::Cancel).unwrap());
        assert!(matches!(
            seen.as_slice(),
            [AgentEvent::ReasoningFinished, AgentEvent::TurnCancelled]
        ));
        assert!(matches!(
            agent.messages().last(),
            Some(Message::Assistant { reasoning: Some(r), text, .. })
                if r.as_str() == "halfway" && text.is_empty()
        ));
    }

    #[test]
    fn failure_while_thinking_closes_reasoning_first() {
        let mut agent = inferring_agent();
        agent.handle(reasoning_chunk("halfway")).unwrap();
        let seen = events(
            agent
                .handle(Input::ModelFailed(ModelError::Status { status: 500 }))
                .unwrap(),
        );
        assert!(matches!(
            seen.as_slice(),
            [AgentEvent::ReasoningFinished, AgentEvent::TurnFailed(_)]
        ));
    }

    #[test]
    fn reasoning_is_sent_back_within_the_turn_only() {
        let mut agent = inferring_agent();
        agent.handle(reasoning_chunk("plan")).unwrap();
        agent
            .handle(Input::ModelChunk(ModelChunk::ToolCall(call("c1"))))
            .unwrap();
        agent.handle(Input::ModelFinished).unwrap();
        let effects = agent.handle(tool_ok("c1")).unwrap();
        let Some(Effect::Infer(request)) = effects.last() else {
            panic!("expected an Infer effect");
        };
        assert!(request.messages.iter().any(|m| matches!(
            m,
            Message::Assistant { reasoning: Some(r), .. } if r.as_str() == "plan"
        )));

        agent.handle(Input::ModelFinished).unwrap();
        let effects = agent.handle(submit("next question")).unwrap();
        let Some(Effect::Infer(request)) = effects.last() else {
            panic!("expected an Infer effect");
        };
        assert!(!request.messages.iter().any(|m| matches!(
            m,
            Message::Assistant {
                reasoning: Some(_),
                ..
            }
        )));
        assert!(agent.messages().iter().any(|m| matches!(
            m,
            Message::Assistant {
                reasoning: Some(_),
                ..
            }
        )));
    }

    #[test]
    fn reported_usage_sets_context_used() {
        let mut agent = inferring_agent();
        agent
            .handle(Input::ModelChunk(ModelChunk::Usage(TokenUsage {
                prompt_tokens: 40,
                completion_tokens: 2,
            })))
            .unwrap();
        let effects = agent.handle(Input::ModelFinished).unwrap();
        assert_eq!(agent.context_usage().used, 42);
        assert!(!has_event(&effects, |e| matches!(
            e,
            AgentEvent::ContextUsage(_)
        )));
    }

    #[test]
    fn missing_usage_is_estimated() {
        let mut agent = inferring_agent();
        let effects = agent.handle(Input::ModelFinished).unwrap();
        assert!(agent.context_usage().used > 0);
        assert!(has_event(&effects, |e| matches!(
            e,
            AgentEvent::ContextUsage(_)
        )));
    }

    #[test]
    fn tool_calls_start_running_tools() {
        let mut agent = inferring_agent();
        agent
            .handle(Input::ModelChunk(ModelChunk::ToolCall(call("c1"))))
            .unwrap();
        let effects = agent.handle(Input::ModelFinished).unwrap();
        assert_eq!(agent.status(), Status::RunningTools);
        assert!(
            effects
                .iter()
                .any(|e| matches!(e, Effect::RunTool(c) if c.id.as_str() == "c1"))
        );
        assert!(has_event(&effects, |e| matches!(
            e,
            AgentEvent::ToolCallStarted(_)
        )));
    }

    #[test]
    fn model_failure_returns_to_idle() {
        let mut agent = inferring_agent();
        let effects = agent
            .handle(Input::ModelFailed(ModelError::Status { status: 500 }))
            .unwrap();
        assert_eq!(agent.status(), Status::Idle);
        assert!(has_event(&effects, |e| matches!(
            e,
            AgentEvent::TurnFailed(TurnFailure::Model(_))
        )));
    }

    #[test]
    fn cancel_while_inferring_keeps_partial_text() {
        let mut agent = inferring_agent();
        agent
            .handle(Input::ModelChunk(ModelChunk::Text("partial".to_owned())))
            .unwrap();
        let effects = agent.handle(Input::Cancel).unwrap();
        assert_eq!(agent.status(), Status::Idle);
        assert!(has_event(&effects, |e| matches!(
            e,
            AgentEvent::TurnCancelled
        )));
        assert!(matches!(
            agent.messages().last(),
            Some(Message::Assistant { text, .. }) if text == "partial"
        ));
    }

    #[test]
    fn inferring_rejects_submit() {
        let mut agent = inferring_agent();
        assert_eq!(
            agent.handle(submit("again")).unwrap_err(),
            TransitionError::Busy
        );
        assert_eq!(agent.status(), Status::Inferring);
    }

    #[test]
    fn inferring_rejects_model_switch() {
        let mut agent = inferring_agent();
        assert_eq!(
            agent
                .handle(Input::SelectModel(model("other", 10)))
                .unwrap_err(),
            TransitionError::Busy
        );
        assert_eq!(agent.model().id.as_str(), "m");
    }

    #[test]
    fn inferring_rejects_tool_results() {
        let mut agent = inferring_agent();
        assert_eq!(
            agent.handle(tool_ok("c1")).unwrap_err(),
            TransitionError::NotRunningTools
        );
    }

    // RunningTools

    #[test]
    fn waits_for_every_tool_before_inferring_again() {
        let mut agent = running_tools_agent(&["c1", "c2"]);
        let effects = agent.handle(tool_ok("c2")).unwrap();
        assert_eq!(agent.status(), Status::RunningTools);
        assert_eq!(infer_count(&effects), 0);

        let effects = agent.handle(tool_ok("c1")).unwrap();
        assert_eq!(agent.status(), Status::Inferring);
        assert_eq!(infer_count(&effects), 1);
    }

    #[test]
    fn tool_failure_is_reported_to_the_model() {
        let mut agent = running_tools_agent(&["c1"]);
        agent
            .handle(Input::ToolFinished {
                id: ToolCallId::new("c1").unwrap(),
                result: Err(ToolError::EmptyPath),
            })
            .unwrap();
        assert_eq!(agent.status(), Status::Inferring);
        assert!(matches!(
            agent.messages().last(),
            Some(Message::Tool { content, .. }) if content == "error: path is empty"
        ));
    }

    #[test]
    fn stops_at_round_limit() {
        let mut agent = AgentLoop::new(model("m", 1000)).max_rounds(NonZeroU32::MIN);
        agent.handle(submit("hi")).unwrap();
        agent
            .handle(Input::ModelChunk(ModelChunk::ToolCall(call("c1"))))
            .unwrap();
        agent.handle(Input::ModelFinished).unwrap();
        let effects = agent.handle(tool_ok("c1")).unwrap();
        assert_eq!(agent.status(), Status::Idle);
        assert_eq!(infer_count(&effects), 0);
        assert!(has_event(&effects, |e| matches!(
            e,
            AgentEvent::TurnFailed(TurnFailure::RoundLimit { .. })
        )));
    }

    #[test]
    fn cancel_while_running_tools_records_cancelled_results() {
        let mut agent = running_tools_agent(&["c1"]);
        let effects = agent.handle(Input::Cancel).unwrap();
        assert_eq!(agent.status(), Status::Idle);
        assert!(has_event(&effects, |e| matches!(
            e,
            AgentEvent::ToolCallFinished {
                outcome: ToolOutcome::Cancelled,
                ..
            }
        )));
        assert!(matches!(
            agent.messages().last(),
            Some(Message::Tool { content, .. }) if content == CANCELLED_TOOL_RESULT
        ));
    }

    #[test]
    fn running_tools_rejects_unknown_tool_call() {
        let mut agent = running_tools_agent(&["c1"]);
        assert_eq!(
            agent.handle(tool_ok("zzz")).unwrap_err(),
            TransitionError::UnknownToolCall {
                id: ToolCallId::new("zzz").unwrap()
            }
        );
        assert_eq!(agent.status(), Status::RunningTools);
    }

    #[test]
    fn running_tools_rejects_model_output() {
        let mut agent = running_tools_agent(&["c1"]);
        assert_eq!(
            agent
                .handle(Input::ModelChunk(ModelChunk::Text("x".to_owned())))
                .unwrap_err(),
            TransitionError::NotInferring
        );
    }

    #[test]
    fn running_tools_rejects_reasoning() {
        let mut agent = running_tools_agent(&["c1"]);
        assert_eq!(
            agent.handle(reasoning_chunk("late")).unwrap_err(),
            TransitionError::NotInferring
        );
    }

    #[test]
    fn idle_rejects_reasoning() {
        let mut agent = agent();
        assert_eq!(
            agent.handle(reasoning_chunk("x")).unwrap_err(),
            TransitionError::NotInferring
        );
    }

    #[test]
    fn running_tools_rejects_submit() {
        let mut agent = running_tools_agent(&["c1"]);
        assert_eq!(
            agent.handle(submit("more")).unwrap_err(),
            TransitionError::Busy
        );
    }
}
