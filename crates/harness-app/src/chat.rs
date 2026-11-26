//! The chat window: header with model picker and context meter, the streamed
//! transcript, and the message composer with its `@` file-mention popup.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use futures::{StreamExt as _, channel::mpsc};
use gpui_kit::{
    App, AppContext as _, ClickEvent, Context, Entity, FontWeight, HighlightStyle,
    InteractiveElement as _, IntoElement, ParentElement as _, Render, SharedString,
    StatefulInteractiveElement as _, Styled as _, StyledText, Subscription, Task, WeakEntity,
    Window,
    component::{
        ActiveTheme as _, Icon, IconName, IndexPath, Sizable as _, ThemeStyled as _,
        bubble::{Bubble, BubbleVariant},
        button::{Button, ButtonRounded, ButtonVariants as _},
        collapsible::Collapsible,
        h_flex,
        input::{
            Enter, Escape, IndentInline, InputEvent, MoveDown, MoveUp, Textarea, TextareaState,
        },
        list::ListItem,
        message::{Message, MessageAlignment, MessageAvatar, MessageContent},
        message_scroller::{MessageScroller, MessageScrollerState},
        progress::Progress,
        select::{Select, SelectEvent, SelectItem, SelectState},
        shimmer::ShimmerText,
        spinner::Spinner,
        text::TextView,
        v_flex,
    },
    div,
    prelude::FluentBuilder as _,
    px,
};
use harness_core::{
    AgentEvent, AgentLoop, Command, ContextUsage, FileIndex, FileMatch, FileMention, FsTools,
    ModelCatalog, ModelId, ModelInfo, ModelProvider, Session, ToolCall, ToolCallId, ToolOutcome,
    ToolRequest, TurnFailure, UserText, WorkspacePath, WriteKind, match_files,
};

use crate::file_popup::{FilePopup, Step, Transition};

/// Most rows the file popup shows.
const FILE_ROWS: usize = 8;

const SUGGESTIONS: [&str; 3] = [
    "read Cargo.toml",
    "read crates/harness-core/src/agent.rs",
    "What can you do?",
];

/// What the view needs to start a session.
pub(crate) struct Launch {
    pub(crate) catalog: ModelCatalog,
    pub(crate) provider: Box<dyn ModelProvider>,
    pub(crate) tools: FsTools,
    /// Lists the files `@` mentions offer; should share `tools`' root.
    pub(crate) files: Box<dyn FileIndex>,
    pub(crate) provider_label: String,
}

/// What the loop is doing, as far as the UI is concerned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Turn {
    Idle,
    Thinking,
    RunningTool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ToolState {
    Running,
    Succeeded { summary: String },
    Failed { message: String },
    Cancelled,
}

#[derive(Debug, Clone)]
struct ToolEntry {
    id: ToolCallId,
    name: String,
    target: Option<String>,
    state: ToolState,
}

/// Whether a finished thinking block shows its trace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Disclosure {
    Collapsed,
    Expanded,
}

/// The lifecycle of one thinking block: open while the model streams its
/// reasoning, then collapsed to a summary the user can expand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ThinkingState {
    Streaming {
        started: Instant,
    },
    Done {
        elapsed: Duration,
        disclosure: Disclosure,
    },
}

/// The reasoning of one model response.
#[derive(Debug, Clone)]
struct ThinkingEntry {
    text: String,
    state: ThinkingState,
}

impl ThinkingEntry {
    fn new(delta: String, now: Instant) -> Self {
        Self {
            text: delta,
            state: ThinkingState::Streaming { started: now },
        }
    }

    fn is_streaming(&self) -> bool {
        matches!(self.state, ThinkingState::Streaming { .. })
    }

    /// Appends streamed reasoning. A finished block ignores it.
    fn push(&mut self, delta: &str) {
        if self.is_streaming() {
            self.text.push_str(delta);
        }
    }

    /// Ends streaming and collapses the block. Finishing twice is a no-op.
    fn finish(&mut self, now: Instant) {
        if let ThinkingState::Streaming { started } = self.state {
            self.state = ThinkingState::Done {
                elapsed: now.saturating_duration_since(started),
                disclosure: Disclosure::Collapsed,
            };
        }
    }

    /// Expands or collapses a finished block. A streaming block stays open.
    fn toggle(&mut self) {
        if let ThinkingState::Done { disclosure, .. } = &mut self.state {
            *disclosure = match disclosure {
                Disclosure::Collapsed => Disclosure::Expanded,
                Disclosure::Expanded => Disclosure::Collapsed,
            };
        }
    }

    fn is_open(&self) -> bool {
        match self.state {
            ThinkingState::Streaming { .. } => true,
            ThinkingState::Done { disclosure, .. } => disclosure == Disclosure::Expanded,
        }
    }
}

#[derive(Debug, Clone)]
enum Entry {
    User(SharedString),
    Thinking(ThinkingEntry),
    Assistant(String),
    Tool(ToolEntry),
    Error(String),
    Notice(SharedString),
}

/// The workspace file list the popup searches.
#[derive(Debug, Clone)]
enum FileList {
    Unloaded,
    /// A listing is running; `stale` is the previous one, if any, which is
    /// searched in the meantime.
    Loading {
        stale: Option<Arc<[WorkspacePath]>>,
    },
    Ready(Arc<[WorkspacePath]>),
    Failed(SharedString),
}

impl FileList {
    fn paths(&self) -> Option<&Arc<[WorkspacePath]>> {
        match self {
            Self::Loading { stale } => stale.as_ref(),
            Self::Ready(paths) => Some(paths),
            Self::Unloaded | Self::Failed(_) => None,
        }
    }

    fn start_loading(&mut self) {
        let stale = self.paths().cloned();
        *self = Self::Loading { stale };
    }
}

#[derive(Debug, Clone)]
struct ModelOption(ModelInfo);

impl SelectItem for ModelOption {
    type Value = ModelId;

    fn title(&self) -> SharedString {
        self.0.label.clone().into()
    }

    fn value(&self) -> &Self::Value {
        &self.0.id
    }

    fn render(&self, _: &mut Window, cx: &mut App) -> impl IntoElement {
        h_flex()
            .w_full()
            .justify_between()
            .gap_4()
            .child(self.0.label.clone())
            .child(
                div()
                    .text_xs()
                    .text_color(cx.theme().muted_foreground)
                    .child(format!(
                        "{} ctx",
                        compact_tokens(self.0.context_window.tokens())
                    )),
            )
    }
}

/// The root view of the window.
pub(crate) struct ChatView {
    entries: Vec<Entry>,
    turn: Turn,
    model: ModelInfo,
    catalog: ModelCatalog,
    context: ContextUsage,
    provider_label: SharedString,
    input: Entity<TextareaState>,
    model_select: Entity<SelectState<Vec<ModelOption>>>,
    scroller: Entity<MessageScrollerState>,
    commands: mpsc::UnboundedSender<Command>,
    file_index: Box<dyn FileIndex>,
    files: FileList,
    file_popup: FilePopup,
    /// Dropping a task cancels it, so replacing these abandons stale work.
    file_listing: Task<()>,
    file_search: Task<()>,
    _subscriptions: Vec<Subscription>,
    _session: Task<()>,
    _events: Task<()>,
}

impl ChatView {
    pub(crate) fn new(launch: Launch, window: &mut Window, cx: &mut Context<'_, Self>) -> Self {
        let Launch {
            catalog,
            provider,
            tools,
            files,
            provider_label,
        } = launch;

        let agent = AgentLoop::new(catalog.default_model().clone());
        let model = agent.model().clone();
        let context = agent.context_usage();

        let (commands, commands_rx) = mpsc::unbounded();
        let (events_tx, mut events_rx) = mpsc::unbounded();
        let session = Session::new(agent, provider, tools);
        let session_task = cx
            .background_executor()
            .spawn(session.run(commands_rx, move |event| {
                // The receiver closes only when the window is gone, and then
                // nobody is left to show the event to.
                let _ = events_tx.unbounded_send(event);
            }));
        let events_task = cx.spawn(async move |this, cx| {
            while let Some(event) = events_rx.next().await {
                if this.update(cx, |view, cx| view.apply(event, cx)).is_err() {
                    break;
                }
            }
        });

        let input = cx.new(|cx| {
            TextareaState::new(window, cx)
                .placeholder("Message Harness…")
                .auto_grow(1, 8)
                .submit_on_enter(true)
        });
        input.update(cx, |state, cx| state.focus(window, cx));

        let options: Vec<ModelOption> = catalog.iter().cloned().map(ModelOption).collect();
        let model_select =
            cx.new(|cx| SelectState::new(options, Some(IndexPath::new(0)), window, cx));
        let scroller = cx.new(|cx| MessageScrollerState::new(0, cx));

        let subscriptions = vec![
            cx.subscribe_in(&input, window, |this, _, event: &InputEvent, window, cx| {
                if let InputEvent::PressEnter { shift: false, .. } = event {
                    this.submit(window, cx);
                }
            }),
            // Observing, not subscribing: moving the cursor emits no event
            // but can enter or leave an `@` token.
            cx.observe(&input, |this, _, cx| this.follow_mention(cx)),
            cx.subscribe_in(
                &model_select,
                window,
                |this, _, event: &SelectEvent<Vec<ModelOption>>, _, cx| {
                    let SelectEvent::Confirm(Some(id)) = event else {
                        return;
                    };
                    this.select_model(id, cx);
                },
            ),
        ];

        Self {
            entries: Vec::new(),
            turn: Turn::Idle,
            model,
            catalog,
            context,
            provider_label: provider_label.into(),
            input,
            model_select,
            scroller,
            commands,
            file_index: files,
            files: FileList::Unloaded,
            file_popup: FilePopup::Closed,
            file_listing: Task::ready(()),
            file_search: Task::ready(()),
            _subscriptions: subscriptions,
            _session: session_task,
            _events: events_task,
        }
    }

    fn send(&self, command: Command) {
        // The session lives as long as this view, so the channel is open.
        let _ = self.commands.unbounded_send(command);
    }

    fn submit(&mut self, window: &mut Window, cx: &mut Context<'_, Self>) {
        if self.turn != Turn::Idle {
            return;
        }
        let raw = self.input.read(cx).value();
        let Ok(text) = UserText::parse(&raw) else {
            return;
        };
        self.input
            .update(cx, |state, cx| state.set_value("", window, cx));
        self.turn = Turn::Thinking;
        self.send(Command::Submit(text));
        cx.notify();
    }

    fn submit_suggestion(&mut self, text: &str, window: &mut Window, cx: &mut Context<'_, Self>) {
        self.input
            .update(cx, |state, cx| state.set_value(text.to_owned(), window, cx));
        self.submit(window, cx);
    }

    /// Opens, updates, or closes the file popup to match the `@` token
    /// under the cursor.
    fn follow_mention(&mut self, cx: &mut Context<'_, Self>) {
        let mention = {
            let input = self.input.read(cx);
            FileMention::at_cursor(&input.value(), input.cursor())
        };
        match self.file_popup.on_mention(mention) {
            Transition::Unchanged => return,
            Transition::Opened => {
                self.list_files(cx);
                self.search_files(cx);
            }
            Transition::Requeried => self.search_files(cx),
            Transition::Closed => self.file_search = Task::ready(()),
        }
        cx.notify();
    }

    /// Re-lists the workspace in the background, so files created since the
    /// last mention show up.
    fn list_files(&mut self, cx: &mut Context<'_, Self>) {
        if matches!(self.files, FileList::Loading { .. }) {
            return;
        }
        self.files.start_loading();
        let listing = self.file_index.files();
        self.file_listing = cx.spawn(async move |this, cx| {
            let result = cx.background_executor().spawn(listing).await;
            // An error means the view is gone, and nobody needs the files.
            let _ = this.update(cx, |view, cx| {
                view.files = match result {
                    Ok(paths) => FileList::Ready(paths.into()),
                    Err(error) => FileList::Failed(error_chain(&error).into()),
                };
                view.search_files(cx);
                cx.notify();
            });
        });
    }

    /// Ranks the file list against the popup's query in the background.
    fn search_files(&mut self, cx: &mut Context<'_, Self>) {
        let (Some(query), Some(paths)) = (self.file_popup.query(), self.files.paths()) else {
            return;
        };
        let query = query.to_owned();
        let paths = Arc::clone(paths);
        self.file_search = cx.spawn(async move |this, cx| {
            let ranking = {
                let query = query.clone();
                cx.background_executor()
                    .spawn(async move { match_files(&paths, &query, FILE_ROWS) })
            };
            let found = ranking.await;
            // An error means the view is gone, and nobody needs the matches.
            let _ = this.update(cx, |view, cx| {
                if view.file_popup.on_matches(&query, found) {
                    cx.notify();
                }
            });
        });
    }

    /// Replaces the `@` token with the file at row `ix`, or the highlighted
    /// row.
    fn accept_file(&mut self, ix: Option<usize>, window: &mut Window, cx: &mut Context<'_, Self>) {
        let Some(edit) = self.file_popup.accept(ix) else {
            return;
        };
        self.file_search = Task::ready(());
        self.input.update(cx, |state, cx| {
            state.set_selected_range(edit.range, cx);
            state.replace(edit.text, window, cx);
            state.focus(window, cx);
        });
        cx.notify();
    }

    fn on_move_up(&mut self, _: &MoveUp, _: &mut Window, cx: &mut Context<'_, Self>) {
        self.step_file(Step::Up, cx);
    }

    fn on_move_down(&mut self, _: &MoveDown, _: &mut Window, cx: &mut Context<'_, Self>) {
        self.step_file(Step::Down, cx);
    }

    fn step_file(&mut self, step: Step, cx: &mut Context<'_, Self>) {
        if self.file_popup.step(step) {
            cx.stop_propagation();
            cx.notify();
        }
    }

    fn on_enter(&mut self, action: &Enter, window: &mut Window, cx: &mut Context<'_, Self>) {
        if !action.shift && !action.secondary {
            self.accept_highlighted(window, cx);
        }
    }

    fn on_tab(&mut self, _: &IndentInline, window: &mut Window, cx: &mut Context<'_, Self>) {
        self.accept_highlighted(window, cx);
    }

    fn accept_highlighted(&mut self, window: &mut Window, cx: &mut Context<'_, Self>) {
        if self.file_popup.has_matches() {
            cx.stop_propagation();
            self.accept_file(None, window, cx);
        }
    }

    fn on_escape(&mut self, _: &Escape, _: &mut Window, cx: &mut Context<'_, Self>) {
        if self.file_popup.dismiss() {
            cx.stop_propagation();
            self.file_search = Task::ready(());
            cx.notify();
        }
    }

    fn cancel(&mut self, _: &ClickEvent, _: &mut Window, _: &mut Context<'_, Self>) {
        self.send(Command::Cancel);
    }

    fn select_model(&mut self, id: &ModelId, cx: &mut Context<'_, Self>) {
        if let Some(model) = self.catalog.get(id) {
            self.send(Command::SelectModel(model.clone()));
        }
        cx.notify();
    }

    fn apply(&mut self, event: AgentEvent, cx: &mut Context<'_, Self>) {
        match event {
            AgentEvent::UserMessage(text) => {
                self.entries
                    .push(Entry::User(text.as_str().to_owned().into()));
                self.turn = Turn::Thinking;
            }
            AgentEvent::ReasoningDelta(delta) => {
                if let Some(Entry::Thinking(thinking)) = self.entries.last_mut()
                    && thinking.is_streaming()
                {
                    thinking.push(&delta);
                } else {
                    self.entries
                        .push(Entry::Thinking(ThinkingEntry::new(delta, Instant::now())));
                }
            }
            AgentEvent::ReasoningFinished => {
                if let Some(thinking) = self.streaming_thinking() {
                    thinking.finish(Instant::now());
                }
            }
            AgentEvent::TextDelta(delta) => {
                if let Some(Entry::Assistant(text)) = self.entries.last_mut() {
                    text.push_str(&delta);
                } else {
                    self.entries.push(Entry::Assistant(delta));
                }
            }
            AgentEvent::ToolCallStarted(call) => {
                self.entries.push(Entry::Tool(tool_entry(&call)));
                self.turn = Turn::RunningTool;
            }
            AgentEvent::ToolCallFinished { id, outcome } => {
                let entry = self.entries.iter_mut().rev().find_map(|entry| match entry {
                    Entry::Tool(tool) if tool.id == id => Some(tool),
                    Entry::Tool(_)
                    | Entry::User(_)
                    | Entry::Thinking(_)
                    | Entry::Assistant(_)
                    | Entry::Error(_)
                    | Entry::Notice(_) => None,
                });
                if let Some(tool) = entry {
                    tool.state = tool_state(&outcome);
                }
                self.turn = Turn::Thinking;
            }
            AgentEvent::ContextUsage(usage) => self.context = usage,
            AgentEvent::ModelChanged(model) => self.model = model,
            AgentEvent::TurnFinished => self.turn = Turn::Idle,
            AgentEvent::TurnFailed(failure) => {
                self.entries.push(Entry::Error(failure_text(&failure)));
                self.turn = Turn::Idle;
            }
            AgentEvent::TurnCancelled => {
                self.entries.push(Entry::Notice("Stopped".into()));
                self.turn = Turn::Idle;
            }
            AgentEvent::Rejected(error) => {
                self.entries.push(Entry::Notice(error.to_string().into()));
            }
        }
        self.sync_scroller(cx);
        cx.notify();
    }

    fn streaming_thinking(&mut self) -> Option<&mut ThinkingEntry> {
        match self.entries.last_mut() {
            Some(Entry::Thinking(thinking)) if thinking.is_streaming() => Some(thinking),
            Some(
                Entry::Thinking(_)
                | Entry::User(_)
                | Entry::Assistant(_)
                | Entry::Tool(_)
                | Entry::Error(_)
                | Entry::Notice(_),
            )
            | None => None,
        }
    }

    fn toggle_thinking(&mut self, ix: usize, cx: &mut Context<'_, Self>) {
        if let Some(Entry::Thinking(thinking)) = self.entries.get_mut(ix) {
            thinking.toggle();
            self.scroller.update(cx, |scroller, cx| {
                scroller.remeasure_items(ix..ix + 1, cx);
            });
            cx.notify();
        }
    }

    /// Whether the "Thinking…" placeholder row is shown: the model is busy but
    /// nothing it is streaming is visible yet.
    fn shows_typing(&self) -> bool {
        self.turn != Turn::Idle
            && !matches!(self.entries.last(), Some(Entry::Assistant(_)))
            && !matches!(self.entries.last(), Some(Entry::Thinking(t)) if t.is_streaming())
    }

    fn row_count(&self) -> usize {
        self.entries.len() + usize::from(self.shows_typing())
    }

    fn sync_scroller(&self, cx: &mut Context<'_, Self>) {
        let rows = self.row_count();
        self.scroller.update(cx, |scroller, cx| {
            let current = scroller.item_count();
            if rows > current {
                scroller.append(rows - current, cx);
            } else if rows < current {
                scroller.splice(rows..current, 0, cx);
            }
            // Streaming text and tool status change the height of recent rows.
            scroller.remeasure_items(rows.saturating_sub(3)..rows, cx);
        });
    }

    fn render_row(&self, ix: usize, view: &WeakEntity<Self>, cx: &App) -> gpui_kit::AnyElement {
        match self.entries.get(ix) {
            Some(Entry::User(text)) => Message::new()
                .id(("user", ix))
                .alignment(MessageAlignment::End)
                .content(MessageContent::new().bubble(Bubble::new().child(text.clone())))
                .into_any_element(),
            Some(Entry::Thinking(thinking)) => render_thinking(ix, thinking, view.clone(), cx),
            Some(Entry::Assistant(text)) => Message::new()
                .id(("assistant", ix))
                .avatar_slot(MessageAvatar::new().child(bot_avatar(cx)))
                .content(
                    MessageContent::new().bubble(
                        Bubble::new()
                            .with_variant(BubbleVariant::Secondary)
                            .child(TextView::markdown(("markdown", ix), text.clone())),
                    ),
                )
                .into_any_element(),
            Some(Entry::Tool(tool)) => render_tool(ix, tool, cx),
            Some(Entry::Error(message)) => Message::new()
                .id(("error", ix))
                .avatar_slot(MessageAvatar::new().child(bot_avatar(cx)))
                .content(
                    MessageContent::new().bubble(
                        Bubble::new()
                            .with_variant(BubbleVariant::Destructive)
                            .child(
                                h_flex()
                                    .gap_2()
                                    .child(Icon::new(IconName::CircleAlert).small())
                                    .child(message.clone()),
                            ),
                    ),
                )
                .into_any_element(),
            Some(Entry::Notice(text)) => h_flex()
                .w_full()
                .justify_center()
                .text_xs()
                .text_color(cx.theme().muted_foreground)
                .child(text.clone())
                .into_any_element(),
            None => Message::new()
                .id("typing")
                .avatar_slot(MessageAvatar::new().child(bot_avatar(cx)))
                .content(
                    MessageContent::new().child(
                        h_flex()
                            .gap_2()
                            .text_sm()
                            .text_color(cx.theme().muted_foreground)
                            .child(Spinner::new().small())
                            .child(match self.turn {
                                Turn::RunningTool => "Running tool…",
                                Turn::Thinking | Turn::Idle => "Thinking…",
                            }),
                    ),
                )
                .into_any_element(),
        }
    }

    fn render_header(&self, cx: &mut Context<'_, Self>) -> impl IntoElement {
        let theme = cx.theme();
        h_flex()
            .w_full()
            .justify_between()
            .gap_4()
            .px_5()
            .py_3()
            .border_b_1()
            .border_color(theme.border)
            .child(
                h_flex().gap_3().child(bot_avatar(cx)).child(
                    v_flex()
                        .child(
                            div()
                                .text_sm()
                                .font_weight(FontWeight::SEMIBOLD)
                                .child("Harness"),
                        )
                        .child(
                            div()
                                .text_xs()
                                .text_color(theme.muted_foreground)
                                .child(self.provider_label.clone()),
                        ),
                ),
            )
            .child(
                h_flex().gap_4().child(self.render_context_meter(cx)).child(
                    Select::new(&self.model_select)
                        .small()
                        .w(px(200.))
                        .disabled(self.turn != Turn::Idle),
                ),
            )
    }

    fn render_context_meter(&self, cx: &App) -> impl IntoElement {
        let theme = cx.theme();
        let percent = self.context.percent();
        let color = match percent {
            90.. => theme.danger,
            75..=89 => theme.warning,
            _ => theme.primary,
        };
        h_flex()
            .id("context-meter")
            .gap_2()
            .text_xs()
            .text_color(theme.muted_foreground)
            .child("Context")
            .child(
                Progress::new("context-progress")
                    .w(px(88.))
                    .color(color)
                    .value(f32::from(percent))
                    .accessibility_label("Context window used"),
            )
            .child(div().text_color(theme.foreground).child(format!(
                "{} / {}",
                compact_tokens(self.context.used),
                compact_tokens(self.context.window.tokens())
            )))
    }

    fn render_empty_state(&self, cx: &mut Context<'_, Self>) -> impl IntoElement {
        let theme = cx.theme();
        v_flex()
            .size_full()
            .items_center()
            .justify_center()
            .gap_3()
            .child(
                div()
                    .size(px(48.))
                    .rounded_full()
                    .bg(theme.secondary)
                    .flex()
                    .items_center()
                    .justify_center()
                    .child(Icon::new(IconName::Bot).with_size(px(24.))),
            )
            .child(
                div()
                    .text_xl()
                    .font_weight(FontWeight::SEMIBOLD)
                    .child("What can I help with?"),
            )
            .child(
                div()
                    .text_sm()
                    .text_color(theme.muted_foreground)
                    .child(format!(
                        "Ask {} anything, or have it read a file from the workspace.",
                        self.model.label
                    )),
            )
            .child(
                h_flex()
                    .pt_2()
                    .gap_2()
                    .children(SUGGESTIONS.iter().enumerate().map(|(ix, text)| {
                        Button::new(("suggestion", ix))
                            .outline()
                            .small()
                            .rounded(ButtonRounded::Large)
                            .label(*text)
                            .on_click(cx.listener(move |this, _, window, cx| {
                                this.submit_suggestion(text, window, cx);
                            }))
                    })),
            )
    }

    fn render_composer(&self, cx: &mut Context<'_, Self>) -> impl IntoElement {
        let theme = cx.theme();
        let busy = self.turn != Turn::Idle;
        let action = if busy {
            Button::new("stop")
                .icon(IconName::Square)
                .rounded(ButtonRounded::Large)
                .tooltip("Stop")
                .on_click(cx.listener(Self::cancel))
        } else {
            Button::new("send")
                .primary()
                .icon(IconName::ArrowUp)
                .rounded(ButtonRounded::Large)
                .tooltip("Send")
                .on_click(cx.listener(|this, _, window, cx| this.submit(window, cx)))
        };
        let hint = if self.file_popup.is_open() {
            "↑↓ to choose · Enter or Tab to insert · Esc to dismiss"
        } else {
            "Enter to send · Shift+Enter for a new line · @ to mention a file"
        };
        div().w_full().px_5().pb_5().pt_2().child(
            v_flex()
                .relative()
                .w_full()
                .max_w(px(820.))
                .mx_auto()
                .gap_2()
                .p_3()
                .rounded_xl()
                .border_1()
                .border_color(theme.border)
                .bg(theme.secondary)
                .shadow_md()
                // Capture runs before the textarea's own handlers, so these
                // can claim the keys while the popup is open; each one lets
                // the key through when there is nothing to act on.
                .capture_action(cx.listener(Self::on_move_up))
                .capture_action(cx.listener(Self::on_move_down))
                .capture_action(cx.listener(Self::on_enter))
                .capture_action(cx.listener(Self::on_tab))
                .capture_action(cx.listener(Self::on_escape))
                .child(Textarea::new(&self.input).appearance(false))
                .child(
                    h_flex()
                        .justify_between()
                        .child(
                            div()
                                .text_xs()
                                .text_color(theme.muted_foreground)
                                .child(hint),
                        )
                        .child(action),
                )
                .children(self.render_file_popup(cx)),
        )
    }

    fn render_file_popup(&self, cx: &mut Context<'_, Self>) -> Option<impl IntoElement> {
        let FilePopup::Open {
            mention,
            matches,
            selected,
        } = &self.file_popup
        else {
            return None;
        };
        let rows = if matches.is_empty() {
            vec![self.render_file_status(mention.query(), cx)]
        } else {
            matches
                .iter()
                .enumerate()
                .map(|(ix, found)| render_file_row(ix, found, ix == *selected, cx))
                .collect()
        };
        let muted = cx.theme().muted_foreground;
        Some(
            div()
                .absolute()
                .bottom_full()
                .left_0()
                .right_0()
                .mb_2()
                .child(
                    v_flex()
                        .id("file-popup")
                        .occlude()
                        .popover_style(cx)
                        .p_1()
                        .gap_0p5()
                        .child(
                            div()
                                .px_2()
                                .py_1()
                                .text_xs()
                                .font_weight(FontWeight::SEMIBOLD)
                                .text_color(muted)
                                .child("Workspace files"),
                        )
                        .children(rows),
                ),
        )
    }

    /// The row shown in place of matches: still listing, failed, or empty.
    fn render_file_status(&self, query: &str, cx: &App) -> gpui_kit::AnyElement {
        let theme = cx.theme();
        let (icon, text): (gpui_kit::AnyElement, SharedString) = match &self.files {
            FileList::Unloaded | FileList::Loading { stale: None } => (
                Spinner::new().xsmall().into_any_element(),
                "Indexing workspace…".into(),
            ),
            FileList::Failed(message) => (
                Icon::new(IconName::CircleAlert)
                    .xsmall()
                    .text_color(theme.danger)
                    .into_any_element(),
                message.clone(),
            ),
            FileList::Loading { stale: Some(_) } | FileList::Ready(_) => (
                Icon::new(IconName::File).xsmall().into_any_element(),
                if query.is_empty() {
                    "No files in the workspace".into()
                } else {
                    format!("No files match “{query}”").into()
                },
            ),
        };
        h_flex()
            .gap_2()
            .px_2()
            .py_1p5()
            .text_sm()
            .text_color(theme.muted_foreground)
            .child(icon)
            .child(text)
            .into_any_element()
    }
}

impl Render for ChatView {
    fn render(&mut self, _: &mut Window, cx: &mut Context<'_, Self>) -> impl IntoElement {
        let view = cx.entity().downgrade();
        let transcript = if self.entries.is_empty() && self.turn == Turn::Idle {
            self.render_empty_state(cx).into_any_element()
        } else {
            MessageScroller::new("transcript", self.scroller.clone(), move |ix, _, cx| {
                view.upgrade().map_or_else(
                    || div().into_any_element(),
                    |entity| entity.read(cx).render_row(ix, &view, cx),
                )
            })
            .size_full()
            .into_any_element()
        };

        v_flex()
            .size_full()
            .bg(cx.theme().background)
            .text_color(cx.theme().foreground)
            .child(self.render_header(cx))
            .child(
                div()
                    .flex_1()
                    .min_h_0()
                    .w_full()
                    .max_w(px(860.))
                    .mx_auto()
                    .child(transcript),
            )
            .child(self.render_composer(cx))
    }
}

fn bot_avatar(cx: &App) -> impl IntoElement {
    let theme = cx.theme();
    div()
        .size(px(28.))
        .flex_none()
        .rounded_full()
        .bg(theme.primary)
        .flex()
        .items_center()
        .justify_center()
        .child(
            Icon::new(IconName::Bot)
                .small()
                .text_color(theme.primary_foreground),
        )
}

fn render_tool(ix: usize, tool: &ToolEntry, cx: &App) -> gpui_kit::AnyElement {
    let theme = cx.theme();
    let (status_icon, status_text): (gpui_kit::AnyElement, String) = match &tool.state {
        ToolState::Running => (
            Spinner::new().xsmall().into_any_element(),
            "running".to_owned(),
        ),
        ToolState::Succeeded { summary } => (
            Icon::new(IconName::CircleCheck)
                .xsmall()
                .text_color(theme.success)
                .into_any_element(),
            summary.clone(),
        ),
        ToolState::Failed { message } => (
            Icon::new(IconName::CircleX)
                .xsmall()
                .text_color(theme.danger)
                .into_any_element(),
            message.clone(),
        ),
        ToolState::Cancelled => (
            Icon::new(IconName::Ban).xsmall().into_any_element(),
            "cancelled".to_owned(),
        ),
    };
    h_flex()
        .id(("tool", ix))
        .w_full()
        .pl(px(40.))
        .child(
            h_flex()
                .gap_2()
                .px_3()
                .py_1p5()
                .rounded_lg()
                .border_1()
                .border_color(theme.border)
                .bg(theme.muted)
                .text_xs()
                .child(Icon::new(IconName::FileText).xsmall())
                .child(
                    div()
                        .font_family(theme.mono_font_family.clone())
                        .text_color(theme.foreground)
                        .child(tool.name.clone()),
                )
                .when_some(tool.target.clone(), |this, target| {
                    this.child(
                        div()
                            .font_family(theme.mono_font_family.clone())
                            .text_color(theme.muted_foreground)
                            .child(target),
                    )
                })
                .child(div().w(px(1.)).h(px(12.)).bg(theme.border))
                .child(status_icon)
                .child(div().text_color(theme.muted_foreground).child(status_text)),
        )
        .into_any_element()
}

fn render_thinking(
    ix: usize,
    thinking: &ThinkingEntry,
    view: WeakEntity<ChatView>,
    cx: &App,
) -> gpui_kit::AnyElement {
    let theme = cx.theme();
    let open = thinking.is_open();
    let label = match thinking.state {
        ThinkingState::Streaming { .. } => ShimmerText::new("Thinking…")
            .id(("thinking-shimmer", ix))
            .into_any_element(),
        ThinkingState::Done { elapsed, .. } => thought_summary(elapsed).into_any_element(),
    };
    let header = h_flex()
        .id(("thinking-toggle", ix))
        .gap_1p5()
        .text_xs()
        .text_color(theme.muted_foreground)
        .cursor_pointer()
        .hover(|style| style.text_color(theme.foreground))
        .child(
            Icon::new(if open {
                IconName::ChevronDown
            } else {
                IconName::ChevronRight
            })
            .xsmall(),
        )
        .child(label)
        .on_click(move |_, _, cx| {
            // A dropped view has no block left to toggle.
            let _ = view.update(cx, |view, cx| view.toggle_thinking(ix, cx));
        });
    let trace = div()
        .mt_1()
        .ml(px(5.))
        .pl_3()
        .border_l_2()
        .border_color(theme.border)
        .text_xs()
        .line_height(px(18.))
        .text_color(theme.muted_foreground)
        .child(thinking.text.trim().to_owned());
    h_flex()
        .id(("thinking", ix))
        .w_full()
        .pl(px(40.))
        .pr_5()
        .child(
            Collapsible::new()
                .open(open)
                .w_full()
                .child(header)
                .content(trace),
        )
        .into_any_element()
}

fn render_file_row(
    ix: usize,
    found: &FileMatch,
    selected: bool,
    cx: &mut Context<'_, ChatView>,
) -> gpui_kit::AnyElement {
    let theme = cx.theme();
    let path = found.path().as_str();
    let highlight = HighlightStyle {
        color: Some(theme.blue),
        font_weight: Some(FontWeight::SEMIBOLD),
        ..HighlightStyle::default()
    };
    let highlights = found
        .highlights()
        .iter()
        .map(|range| (range.clone(), highlight));
    ListItem::new(("file", ix))
        .selected(selected)
        .accessibility_label(path.to_owned())
        .px_2()
        .py_1()
        .rounded(theme.radius)
        .text_sm()
        .on_click(cx.listener(move |this, _, window, cx| this.accept_file(Some(ix), window, cx)))
        .on_mouse_enter(cx.listener(move |this, _, _, cx| {
            if this.file_popup.hover(ix) {
                cx.notify();
            }
        }))
        .child(
            h_flex()
                .gap_2()
                .child(
                    Icon::new(IconName::File)
                        .xsmall()
                        .text_color(theme.muted_foreground),
                )
                .child(
                    div()
                        .font_family(theme.mono_font_family.clone())
                        .child(StyledText::new(path.to_owned()).with_highlights(highlights)),
                ),
        )
        .into_any_element()
}

/// The collapsed label of a finished thinking block, such as `Thought for 4s`.
fn thought_summary(elapsed: Duration) -> String {
    let seconds = (elapsed.as_millis() + 500) / 1_000;
    match seconds {
        0 => "Thought for <1s".to_owned(),
        1..60 => format!("Thought for {seconds}s"),
        _ => format!("Thought for {}m {}s", seconds / 60, seconds % 60),
    }
}

fn tool_entry(call: &ToolCall) -> ToolEntry {
    let target = match ToolRequest::parse(call) {
        Ok(ToolRequest::ReadFile { path } | ToolRequest::WriteFile { path, .. }) => {
            Some(path.display().to_string())
        }
        Err(_) => None,
    };
    ToolEntry {
        id: call.id.clone(),
        name: call.name.as_str().to_owned(),
        target,
        state: ToolState::Running,
    }
}

fn tool_state(outcome: &ToolOutcome) -> ToolState {
    match outcome {
        ToolOutcome::Succeeded { output } => {
            let summary = if let Some(write) = output.file_write() {
                let verb = match write.kind {
                    WriteKind::Created => "created",
                    WriteKind::Overwritten => "updated",
                };
                let bytes = usize::try_from(write.bytes).unwrap_or(usize::MAX);
                format!("{verb} · {}", compact_bytes(bytes))
            } else {
                let text = output.as_str();
                format!(
                    "{} lines · {}",
                    text.lines().count(),
                    compact_bytes(text.len())
                )
            };
            ToolState::Succeeded { summary }
        }
        ToolOutcome::Failed { error } => ToolState::Failed {
            message: error.to_string(),
        },
        ToolOutcome::Cancelled => ToolState::Cancelled,
    }
}

fn failure_text(failure: &TurnFailure) -> String {
    match failure {
        TurnFailure::Model(error) => error_chain(error.as_ref()),
        TurnFailure::RoundLimit { limit } => {
            format!("Stopped after {limit} model calls without a final answer.")
        }
    }
}

fn error_chain(error: &dyn std::error::Error) -> String {
    let mut text = error.to_string();
    let mut source = error.source();
    while let Some(cause) = source {
        text.push_str(": ");
        text.push_str(&cause.to_string());
        source = cause.source();
    }
    text
}

/// Formats a token count as `850`, `1.2k`, or `32k`, truncating like the
/// context sizes models are usually described by.
fn compact_tokens(tokens: u32) -> String {
    match tokens {
        0..1_000 => tokens.to_string(),
        1_000..10_000 => format!("{}.{}k", tokens / 1_000, tokens % 1_000 / 100),
        _ => format!("{}k", tokens / 1_000),
    }
}

/// Formats a byte count as `512 B`, `1.5 KB`, or `20 KB`.
fn compact_bytes(bytes: usize) -> String {
    match bytes {
        0..1_024 => format!("{bytes} B"),
        1_024..10_240 => format!("{}.{} KB", bytes / 1_024, bytes % 1_024 * 10 / 1_024),
        _ => format!("{} KB", bytes / 1_024),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compacts_token_counts() {
        assert_eq!(compact_tokens(850), "850");
        assert_eq!(compact_tokens(1_250), "1.2k");
        assert_eq!(compact_tokens(2_048), "2.0k");
        assert_eq!(compact_tokens(32_768), "32k");
    }

    #[test]
    fn summarizes_thinking_time() {
        assert_eq!(
            thought_summary(Duration::from_millis(300)),
            "Thought for <1s"
        );
        assert_eq!(
            thought_summary(Duration::from_millis(2_600)),
            "Thought for 3s"
        );
        assert_eq!(
            thought_summary(Duration::from_secs(75)),
            "Thought for 1m 15s"
        );
    }

    #[test]
    fn thinking_streams_open_then_collapses_with_elapsed_time() {
        let start = Instant::now();
        let mut thinking = ThinkingEntry::new("Let me ".to_owned(), start);
        thinking.push("think.");
        assert!(thinking.is_open());
        thinking.finish(start + Duration::from_secs(4));
        assert_eq!(thinking.text, "Let me think.");
        assert!(!thinking.is_open());
        assert_eq!(
            thinking.state,
            ThinkingState::Done {
                elapsed: Duration::from_secs(4),
                disclosure: Disclosure::Collapsed
            }
        );
    }

    #[test]
    fn finished_thinking_toggles_and_ignores_late_events() {
        let start = Instant::now();
        let mut thinking = ThinkingEntry::new("plan".to_owned(), start);
        thinking.finish(start + Duration::from_secs(1));
        thinking.toggle();
        assert!(thinking.is_open());
        thinking.toggle();
        assert!(!thinking.is_open());

        thinking.push(" more");
        thinking.finish(start + Duration::from_secs(9));
        assert_eq!(thinking.text, "plan");
        assert!(matches!(
            thinking.state,
            ThinkingState::Done { elapsed, .. } if elapsed == Duration::from_secs(1)
        ));
    }

    #[test]
    fn streaming_thinking_cannot_be_collapsed() {
        let mut thinking = ThinkingEntry::new("plan".to_owned(), Instant::now());
        thinking.toggle();
        assert!(thinking.is_open());
        assert!(thinking.is_streaming());
    }

    #[test]
    fn compacts_byte_counts() {
        assert_eq!(compact_bytes(512), "512 B");
        assert_eq!(compact_bytes(1_536), "1.5 KB");
        assert_eq!(compact_bytes(20_480), "20 KB");
    }

    #[test]
    fn write_chip_shows_path_kind_and_size() {
        use harness_core::{FileWrite, ToolArguments, ToolName, ToolOutput};

        let call = ToolCall {
            id: ToolCallId::new("c1").unwrap(),
            name: ToolName::new("write_file").unwrap(),
            arguments: ToolArguments::new(r#"{"path":"src/new.rs","contents":"x"}"#),
        };
        assert_eq!(tool_entry(&call).target.as_deref(), Some("src/new.rs"));

        let outcome = |kind| ToolOutcome::Succeeded {
            output: ToolOutput::written(FileWrite::new(
                "src/new.rs".into(),
                kind,
                &"a".repeat(1_536),
            )),
        };
        assert_eq!(
            tool_state(&outcome(WriteKind::Created)),
            ToolState::Succeeded {
                summary: "created · 1.5 KB".to_owned()
            }
        );
        assert_eq!(
            tool_state(&outcome(WriteKind::Overwritten)),
            ToolState::Succeeded {
                summary: "updated · 1.5 KB".to_owned()
            }
        );
    }
}
