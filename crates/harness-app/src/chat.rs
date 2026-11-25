//! The chat window: header with model picker and context meter, the streamed
//! transcript, and the message composer.

use futures::{StreamExt as _, channel::mpsc};
use gpui_kit::{
    App, AppContext as _, ClickEvent, Context, Entity, FontWeight, InteractiveElement as _,
    IntoElement, ParentElement as _, Render, SharedString, Styled as _, Subscription, Task, Window,
    component::{
        ActiveTheme as _, Icon, IconName, IndexPath, Sizable as _,
        bubble::{Bubble, BubbleVariant},
        button::{Button, ButtonRounded, ButtonVariants as _},
        h_flex,
        input::{InputEvent, Textarea, TextareaState},
        message::{Message, MessageAlignment, MessageAvatar, MessageContent},
        message_scroller::{MessageScroller, MessageScrollerState},
        progress::Progress,
        select::{Select, SelectEvent, SelectItem, SelectState},
        spinner::Spinner,
        text::TextView,
        v_flex,
    },
    div,
    prelude::FluentBuilder as _,
    px,
};
use harness_core::{
    AgentEvent, AgentLoop, Command, ContextUsage, FsTools, ModelCatalog, ModelId, ModelInfo,
    ModelProvider, Session, ToolCall, ToolCallId, ToolOutcome, ToolRequest, TurnFailure, UserText,
};

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

#[derive(Debug, Clone)]
enum Entry {
    User(SharedString),
    Assistant(String),
    Tool(ToolEntry),
    Error(String),
    Notice(SharedString),
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

    fn shows_typing(&self) -> bool {
        self.turn != Turn::Idle && !matches!(self.entries.last(), Some(Entry::Assistant(_)))
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

    fn render_row(&self, ix: usize, cx: &App) -> gpui_kit::AnyElement {
        match self.entries.get(ix) {
            Some(Entry::User(text)) => Message::new()
                .id(("user", ix))
                .alignment(MessageAlignment::End)
                .content(MessageContent::new().bubble(Bubble::new().child(text.clone())))
                .into_any_element(),
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
        div().w_full().px_5().pb_5().pt_2().child(
            v_flex()
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
                .child(Textarea::new(&self.input).appearance(false))
                .child(
                    h_flex()
                        .justify_between()
                        .child(
                            div()
                                .text_xs()
                                .text_color(theme.muted_foreground)
                                .child("Enter to send · Shift+Enter for a new line"),
                        )
                        .child(action),
                ),
        )
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
                    |view| view.read(cx).render_row(ix, cx),
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

fn tool_entry(call: &ToolCall) -> ToolEntry {
    let target = match ToolRequest::parse(call) {
        Ok(ToolRequest::ReadFile { path }) => Some(path.display().to_string()),
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
            let text = output.as_str();
            ToolState::Succeeded {
                summary: format!(
                    "{} lines · {}",
                    text.lines().count(),
                    compact_bytes(text.len())
                ),
            }
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
    fn compacts_byte_counts() {
        assert_eq!(compact_bytes(512), "512 B");
        assert_eq!(compact_bytes(1_536), "1.5 KB");
        assert_eq!(compact_bytes(20_480), "20 KB");
    }
}
