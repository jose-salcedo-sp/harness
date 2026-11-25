//! Desktop chatbot window built on GPUI and gpui-component.
//!
//! This crate owns rendering and input only. Agent behaviour lives in
//! `harness-core`; views hold core state and forward user intents to it.

mod chat;
mod config;

use anyhow::Context as _;
use gpui_kit::{
    App, AppContext as _, Bounds, TitlebarOptions, WindowBounds, WindowOptions,
    component::{Theme, ThemeMode},
    px, size,
};
use harness_core::{
    FsTools, ModelProvider,
    model::{fake::FakeProvider, openai::OpenAiProvider},
};

use crate::{
    chat::{ChatView, Launch},
    config::{AppConfig, ProviderChoice},
};

fn main() -> anyhow::Result<()> {
    let config = AppConfig::load(std::env::args().skip(1), |name| std::env::var(name).ok())?;
    let launch = launch(config)?;

    gpui_kit::application()
        .with_assets(gpui_kit::assets::Assets)
        .run(move |cx| {
            gpui_kit::init(cx);
            Theme::change(ThemeMode::Dark, None, cx);
            if let Err(error) = open_window(launch, cx) {
                eprintln!("harness: {error:#}");
                cx.quit();
            }
        });
    Ok(())
}

fn launch(config: AppConfig) -> anyhow::Result<Launch> {
    let provider_label = config.provider_label();
    let workspace = match config.workspace {
        Some(path) => path,
        None => std::env::current_dir().context("could not read the current directory")?,
    };
    let tools = FsTools::new(&workspace)?;
    let provider: Box<dyn ModelProvider> = match config.provider {
        ProviderChoice::Fake => Box::new(FakeProvider::echo()),
        ProviderChoice::OpenAi(openai) => Box::new(OpenAiProvider::new(openai)),
    };
    Ok(Launch {
        catalog: config.catalog,
        provider,
        tools,
        provider_label,
    })
}

fn open_window(launch: Launch, cx: &mut App) -> anyhow::Result<()> {
    let bounds = Bounds::centered(None, size(px(960.), px(720.)), cx);
    let options = WindowOptions {
        window_bounds: Some(WindowBounds::Windowed(bounds)),
        window_min_size: Some(size(px(560.), px(480.))),
        titlebar: Some(TitlebarOptions {
            title: Some("Harness".into()),
            ..TitlebarOptions::default()
        }),
        app_id: Some("harness".to_owned()),
        ..WindowOptions::default()
    };
    gpui_kit::open_window(options, cx, |window, cx| {
        cx.new(|cx| ChatView::new(launch, window, cx))
    })
    .context("could not open the window")?;
    cx.on_window_closed(|cx, _| cx.quit()).detach();
    Ok(())
}
