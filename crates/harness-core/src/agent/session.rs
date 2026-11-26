//! Drives an [`AgentLoop`] by performing its effects through the ports.

use std::{collections::VecDeque, future::Future};

use futures::{
    Stream, StreamExt as _,
    future::{self, Either},
};

use super::{AgentEvent, AgentLoop, Effect, Input};
use crate::{
    conversation::UserText,
    model::{ModelInfo, ModelProvider},
    tool::{ToolExecutor, ToolRequest},
};

/// What the UI can ask the session to do.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    /// Send a message and start a turn.
    Submit(UserText),
    /// Switch models. Only accepted between turns.
    SelectModel(ModelInfo),
    /// Stop the current turn, dropping any model request or tool run in flight.
    Cancel,
}

/// Runs an [`AgentLoop`] against a model provider and a tool executor.
#[derive(Debug)]
pub struct Session<P, T> {
    agent: AgentLoop,
    provider: P,
    tools: T,
}

enum Raced<T> {
    Command(Command),
    Done(T),
}

impl<P: ModelProvider, T: ToolExecutor> Session<P, T> {
    /// A session that drives `agent` with `provider` and `tools`.
    pub fn new(agent: AgentLoop, provider: P, tools: T) -> Self {
        Self {
            agent,
            provider,
            tools,
        }
    }

    /// Processes `commands` until the stream ends, reporting every change to
    /// `emit` as it happens.
    ///
    /// Commands are read while a turn is in progress, so [`Command::Cancel`]
    /// takes effect immediately. When `commands` ends, the turn in progress
    /// runs to completion and the future resolves. The future is `Send` when
    /// the arguments are, so it can run on a background thread pool.
    pub async fn run<C, E>(mut self, mut commands: C, mut emit: E)
    where
        C: Stream<Item = Command> + Unpin,
        E: FnMut(AgentEvent),
    {
        let mut open = true;
        while let Some(command) = commands.next().await {
            let mut queue = VecDeque::new();
            self.feed(command.into(), &mut queue, &mut emit);
            self.drain(&mut queue, &mut commands, &mut open, &mut emit)
                .await;
        }
    }

    /// Applies `input`. Events are emitted immediately so the UI streams;
    /// IO effects are queued for [`Self::drain`].
    fn feed(
        &mut self,
        input: Input,
        queue: &mut VecDeque<Effect>,
        emit: &mut impl FnMut(AgentEvent),
    ) {
        match self.agent.handle(input) {
            Ok(effects) => {
                for effect in effects {
                    match effect {
                        Effect::Emit(event) => emit(event),
                        work @ (Effect::Infer(_) | Effect::RunTool(_)) => queue.push_back(work),
                    }
                }
            }
            Err(error) => emit(AgentEvent::Rejected(error)),
        }
    }

    /// Handles a command that arrived mid-turn. Returns `true` if it ended the turn.
    fn interrupt(
        &mut self,
        command: Command,
        queue: &mut VecDeque<Effect>,
        emit: &mut impl FnMut(AgentEvent),
    ) -> bool {
        let cancel = matches!(command, Command::Cancel);
        if cancel {
            queue.clear();
        }
        self.feed(command.into(), queue, emit);
        cancel
    }

    async fn drain<C, E>(
        &mut self,
        queue: &mut VecDeque<Effect>,
        commands: &mut C,
        open: &mut bool,
        emit: &mut E,
    ) where
        C: Stream<Item = Command> + Unpin,
        E: FnMut(AgentEvent),
    {
        while let Some(effect) = queue.pop_front() {
            match effect {
                Effect::Emit(event) => emit(event),
                Effect::Infer(request) => {
                    let mut stream = self.provider.stream(request);
                    loop {
                        match race(commands, open, stream.next()).await {
                            Raced::Command(command) => {
                                if self.interrupt(command, queue, emit) {
                                    break;
                                }
                            }
                            Raced::Done(Some(Ok(chunk))) => {
                                self.feed(Input::ModelChunk(chunk), queue, emit);
                            }
                            Raced::Done(Some(Err(error))) => {
                                self.feed(Input::ModelFailed(error), queue, emit);
                                break;
                            }
                            Raced::Done(None) => {
                                self.feed(Input::ModelFinished, queue, emit);
                                break;
                            }
                        }
                    }
                }
                Effect::RunTool(call) => {
                    let mut run = match ToolRequest::parse(&call) {
                        Ok(request) => self.tools.execute(request),
                        Err(error) => Box::pin(future::ready(Err(error))),
                    };
                    loop {
                        match race(commands, open, &mut run).await {
                            Raced::Command(command) => {
                                if self.interrupt(command, queue, emit) {
                                    break;
                                }
                            }
                            Raced::Done(result) => {
                                let id = call.id.clone();
                                self.feed(Input::ToolFinished { id, result }, queue, emit);
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Waits for `work`, returning early if a command arrives first.
///
/// Once `commands` ends, `open` is cleared and only `work` is awaited.
async fn race<C, F>(commands: &mut C, open: &mut bool, work: F) -> Raced<F::Output>
where
    C: Stream<Item = Command> + Unpin,
    F: Future + Unpin,
{
    if !*open {
        return Raced::Done(work.await);
    }
    match future::select(commands.next(), work).await {
        Either::Left((Some(command), _)) => Raced::Command(command),
        Either::Left((None, work)) => {
            *open = false;
            Raced::Done(work.await)
        }
        Either::Right((output, _)) => Raced::Done(output),
    }
}

#[cfg(test)]
mod tests {
    use std::{num::NonZeroU32, sync::Arc};

    use futures::{channel::mpsc, executor::block_on, stream};

    use super::*;
    use crate::{
        agent::{ToolOutcome, TurnFailure},
        conversation::Message,
        model::{
            ContextWindow, ModelChunk, ModelError, ModelId, ModelStream, TokenUsage,
            fake::{FakeProvider, FakeReply},
        },
        tool::{MemoryTools, ToolArguments, ToolCall, ToolCallId, ToolName},
    };

    fn model() -> ModelInfo {
        ModelInfo {
            id: ModelId::new("fake").unwrap(),
            label: "Fake".to_owned(),
            context_window: ContextWindow::new(NonZeroU32::new(4096).unwrap()),
        }
    }

    fn read_call(id: &str, path: &str) -> ToolCall {
        ToolCall {
            id: ToolCallId::new(id).unwrap(),
            name: ToolName::new("read_file").unwrap(),
            arguments: ToolArguments::new(format!(r#"{{"path":"{path}"}}"#)),
        }
    }

    fn submit(text: &str) -> Command {
        Command::Submit(UserText::parse(text).unwrap())
    }

    fn run<P: ModelProvider>(provider: P, commands: Vec<Command>) -> Vec<AgentEvent> {
        let tools = MemoryTools::new([("notes.txt", "remember the milk")]);
        let session = Session::new(AgentLoop::new(model()), provider, tools);
        let mut events = Vec::new();
        block_on(session.run(stream::iter(commands), |event| events.push(event)));
        events
    }

    fn reply_text(events: &[AgentEvent]) -> String {
        events
            .iter()
            .filter_map(|event| {
                if let AgentEvent::TextDelta(delta) = event {
                    Some(delta.as_str())
                } else {
                    None
                }
            })
            .collect()
    }

    #[test]
    fn reads_file_then_answers() {
        let provider = Arc::new(FakeProvider::scripted([
            FakeReply::new().tool_call(read_call("c1", "notes.txt")),
            FakeReply::new()
                .text("You need ")
                .text("milk.")
                .usage(TokenUsage {
                    prompt_tokens: 30,
                    completion_tokens: 4,
                }),
        ]));
        let events = run(Arc::clone(&provider), vec![submit("what do I need?")]);

        assert_eq!(reply_text(&events), "You need milk.");
        assert!(events.iter().any(|e| matches!(
            e,
            AgentEvent::ToolCallFinished { outcome: ToolOutcome::Succeeded { output }, .. }
                if output.as_str() == "remember the milk"
        )));
        assert!(matches!(events.last(), Some(AgentEvent::TurnFinished)));
        assert!(events.iter().any(|e| matches!(
            e,
            AgentEvent::ContextUsage(usage) if usage.used == 34
        )));

        let requests = provider.requests();
        assert_eq!(requests.len(), 2);
        assert!(matches!(
            requests[1].messages.last(),
            Some(Message::Tool { content, .. }) if content == "remember the milk"
        ));
    }

    /// The turn-shaping events, with deltas and context updates left out.
    fn outline(events: &[AgentEvent]) -> Vec<&'static str> {
        events
            .iter()
            .filter_map(|event| match event {
                AgentEvent::ReasoningDelta(_) => Some("think"),
                AgentEvent::ReasoningFinished => Some("thought"),
                AgentEvent::TextDelta(_) => Some("text"),
                AgentEvent::ToolCallStarted(_) => Some("tool"),
                AgentEvent::ToolCallFinished { .. } => Some("tool done"),
                AgentEvent::TurnFinished => Some("finished"),
                AgentEvent::UserMessage(_)
                | AgentEvent::ContextUsage(_)
                | AgentEvent::ModelChanged(_)
                | AgentEvent::TurnFailed(_)
                | AgentEvent::TurnCancelled
                | AgentEvent::Rejected(_) => None,
            })
            .fold(Vec::new(), |mut outline, step| {
                if outline.last() != Some(&step) {
                    outline.push(step);
                }
                outline
            })
    }

    #[test]
    fn multi_round_turn_reads_think_tool_think_answer() {
        let provider = FakeProvider::scripted([
            FakeReply::new()
                .reasoning("I should ")
                .reasoning("read the notes.")
                .tool_call(read_call("c1", "notes.txt")),
            FakeReply::new()
                .reasoning("They mention milk.")
                .text("You need ")
                .text("milk."),
        ]);
        let events = run(provider, vec![submit("what do I need?")]);
        assert_eq!(
            outline(&events),
            [
                "think",
                "thought",
                "tool",
                "tool done",
                "think",
                "thought",
                "text",
                "finished"
            ]
        );
        assert_eq!(reply_text(&events), "You need milk.");
    }

    #[test]
    fn echo_provider_thinks_in_every_round() {
        let events = run(FakeProvider::echo(), vec![submit("read notes.txt")]);
        assert_eq!(
            outline(&events),
            [
                "think",
                "thought",
                "tool",
                "tool done",
                "think",
                "thought",
                "text",
                "finished"
            ]
        );
    }

    #[test]
    fn missing_file_is_reported_to_model_not_user() {
        let provider = FakeProvider::scripted([
            FakeReply::new().tool_call(read_call("c1", "missing.txt")),
            FakeReply::new().text("No such file."),
        ]);
        let events = run(provider, vec![submit("read missing.txt")]);
        assert!(events.iter().any(|e| matches!(
            e,
            AgentEvent::ToolCallFinished {
                outcome: ToolOutcome::Failed { .. },
                ..
            }
        )));
        assert!(matches!(events.last(), Some(AgentEvent::TurnFinished)));
    }

    #[test]
    fn model_error_fails_the_turn() {
        let provider = FakeProvider::scripted([FakeReply::new()
            .text("par")
            .fail(ModelError::Status { status: 503 })]);
        let events = run(provider, vec![submit("hi")]);
        assert!(matches!(
            events.last(),
            Some(AgentEvent::TurnFailed(TurnFailure::Model(_)))
        ));
    }

    #[test]
    fn session_accepts_next_turn_after_finishing() {
        let provider =
            FakeProvider::scripted([FakeReply::new().text("one"), FakeReply::new().text("two")]);
        let session = Session::new(AgentLoop::new(model()), provider, MemoryTools::default());
        let (commands_tx, commands_rx) = mpsc::unbounded();
        commands_tx.unbounded_send(submit("a")).unwrap();
        let mut events = Vec::new();
        block_on(session.run(commands_rx, |event| {
            if matches!(event, AgentEvent::TurnFinished) {
                if reply_text(&events) == "one" {
                    commands_tx.unbounded_send(submit("b")).unwrap();
                } else {
                    commands_tx.close_channel();
                }
            }
            events.push(event);
        }));
        let finished = events
            .iter()
            .filter(|e| matches!(e, AgentEvent::TurnFinished))
            .count();
        assert_eq!(finished, 2);
        assert_eq!(reply_text(&events), "onetwo");
    }

    #[test]
    fn text_is_emitted_while_the_stream_is_still_open() {
        let (chunks_tx, chunks_rx) = mpsc::unbounded();
        let provider = ChannelProvider(std::sync::Mutex::new(Some(chunks_rx)));
        let session = Session::new(AgentLoop::new(model()), provider, MemoryTools::default());
        let mut chunks_tx = Some(chunks_tx);
        let mut events = Vec::new();
        block_on(session.run(stream::iter([submit("hi")]), |event| {
            if matches!(event, AgentEvent::UserMessage(_))
                && let Some(tx) = &chunks_tx
            {
                tx.unbounded_send(Ok(ModelChunk::Text("live".to_owned())))
                    .unwrap();
            }
            // Only closing the stream after the delta is seen lets the turn
            // finish, so this test hangs if deltas are buffered.
            if matches!(event, AgentEvent::TextDelta(_)) {
                chunks_tx.take();
            }
            events.push(event);
        }));
        assert_eq!(reply_text(&events), "live");
        assert!(matches!(events.last(), Some(AgentEvent::TurnFinished)));
    }

    #[test]
    fn cancel_stops_a_stream_in_flight() {
        let (commands_tx, commands_rx) = mpsc::unbounded();
        let (chunks_tx, chunks_rx) = mpsc::unbounded();
        let provider = ChannelProvider(std::sync::Mutex::new(Some(chunks_rx)));
        let session = Session::new(AgentLoop::new(model()), provider, MemoryTools::default());

        commands_tx.unbounded_send(submit("hi")).unwrap();
        let mut events = Vec::new();
        let mut cancel_sent = false;
        block_on(session.run(commands_rx, |event| {
            if matches!(event, AgentEvent::UserMessage(_)) {
                chunks_tx
                    .unbounded_send(Ok(ModelChunk::Text("partial".to_owned())))
                    .unwrap();
            }
            if matches!(event, AgentEvent::TextDelta(_)) && !cancel_sent {
                cancel_sent = true;
                commands_tx.unbounded_send(Command::Cancel).unwrap();
                commands_tx.close_channel();
            }
            events.push(event);
        }));
        assert!(matches!(events.last(), Some(AgentEvent::TurnCancelled)));
    }

    #[test]
    fn rejects_submit_during_turn() {
        let (commands_tx, commands_rx) = mpsc::unbounded();
        let (chunks_tx, chunks_rx) = mpsc::unbounded::<Result<ModelChunk, ModelError>>();
        let provider = ChannelProvider(std::sync::Mutex::new(Some(chunks_rx)));
        let session = Session::new(AgentLoop::new(model()), provider, MemoryTools::default());

        commands_tx.unbounded_send(submit("first")).unwrap();
        let mut chunks_tx = Some(chunks_tx);
        let mut events = Vec::new();
        block_on(session.run(commands_rx, |event| {
            if matches!(event, AgentEvent::UserMessage(_)) {
                commands_tx.unbounded_send(submit("second")).unwrap();
                commands_tx.close_channel();
            }
            if matches!(event, AgentEvent::Rejected(_)) {
                // Ending the model stream lets the turn finish.
                chunks_tx.take();
            }
            events.push(event);
        }));
        assert!(events.iter().any(|e| matches!(e, AgentEvent::Rejected(_))));
        assert!(matches!(events.last(), Some(AgentEvent::TurnFinished)));
    }

    /// Streams whatever the test pushes into a channel, for one request.
    struct ChannelProvider(
        std::sync::Mutex<Option<mpsc::UnboundedReceiver<Result<ModelChunk, ModelError>>>>,
    );

    impl ModelProvider for ChannelProvider {
        fn stream(&self, _: crate::model::ModelRequest) -> ModelStream {
            match self.0.lock().unwrap().take() {
                Some(receiver) => Box::pin(receiver),
                None => Box::pin(stream::empty()),
            }
        }
    }
}
