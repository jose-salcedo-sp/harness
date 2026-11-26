//! Startup configuration, read once from the command line and environment.

use std::{num::NonZeroU32, path::PathBuf};

use anyhow::Context as _;
use harness_core::{ContextWindow, ModelCatalog, ModelId, ModelInfo, model::openai::OpenAiConfig};

/// Selects the model provider: `openai` (default) or `fake`.
pub(crate) const PROVIDER_VAR: &str = "HARNESS_PROVIDER";
/// Overrides the model list: comma-separated `id=context_tokens` entries.
pub(crate) const MODELS_VAR: &str = "HARNESS_MODELS";
/// Directory the `read_file` and `write_file` tools are confined to. Defaults
/// to the current directory.
pub(crate) const WORKSPACE_VAR: &str = "HARNESS_WORKSPACE";
/// Command-line flag that selects the fake provider.
pub(crate) const FAKE_FLAG: &str = "--fake";

/// Which [`harness_core::ModelProvider`] adapter to run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ProviderChoice {
    /// The deterministic offline provider.
    Fake,
    /// An OpenAI-compatible HTTP server, such as `llama-server`.
    OpenAi(OpenAiConfig),
}

/// Everything `main` needs to open the window.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AppConfig {
    pub(crate) provider: ProviderChoice,
    pub(crate) catalog: ModelCatalog,
    pub(crate) workspace: Option<PathBuf>,
}

impl AppConfig {
    /// Reads the configuration from `args` (without the program name) and the
    /// environment, through `lookup`.
    pub(crate) fn load(
        args: impl IntoIterator<Item = String>,
        lookup: impl Fn(&str) -> Option<String>,
    ) -> anyhow::Result<Self> {
        let fake_flag = args.into_iter().any(|arg| arg == FAKE_FLAG);
        let provider = match lookup(PROVIDER_VAR).as_deref().map(str::trim) {
            _ if fake_flag => ProviderChoice::Fake,
            Some("fake") => ProviderChoice::Fake,
            None | Some("" | "openai") => ProviderChoice::OpenAi(
                OpenAiConfig::from_env(&lookup).context("invalid model server settings")?,
            ),
            Some(other) => {
                anyhow::bail!("{PROVIDER_VAR} must be `openai` or `fake`, not `{other}`")
            }
        };
        let catalog = match lookup(MODELS_VAR) {
            Some(list) => list
                .parse()
                .with_context(|| format!("invalid {MODELS_VAR}"))?,
            None => default_catalog(&provider)?,
        };
        let workspace = lookup(WORKSPACE_VAR).map(PathBuf::from);
        Ok(Self {
            provider,
            catalog,
            workspace,
        })
    }

    /// A short description of the provider for the window header.
    pub(crate) fn provider_label(&self) -> String {
        match &self.provider {
            ProviderChoice::Fake => "Fake provider (offline)".to_owned(),
            ProviderChoice::OpenAi(config) => config.base_url().to_owned(),
        }
    }
}

fn model(id: &str, label: &str, context_tokens: u32) -> anyhow::Result<ModelInfo> {
    let tokens = NonZeroU32::new(context_tokens).context("context window must be positive")?;
    Ok(ModelInfo {
        id: ModelId::new(id)?,
        label: label.to_owned(),
        context_window: ContextWindow::new(tokens),
    })
}

/// The models offered when [`MODELS_VAR`] is unset.
///
/// The HTTP default is the Bonsai 2 family served by `llama-server`, with the
/// context size `llama-server` is commonly started with.
fn default_catalog(provider: &ProviderChoice) -> anyhow::Result<ModelCatalog> {
    Ok(match provider {
        ProviderChoice::Fake => ModelCatalog::new(
            model("fake-echo", "Fake echo", 8_192)?,
            vec![model("fake-echo-small", "Fake echo (small)", 2_048)?],
        ),
        ProviderChoice::OpenAi(_) => ModelCatalog::new(
            model("bonsai-2-9b", "Bonsai 2 9B", 32_768)?,
            vec![
                model("bonsai-2-4b", "Bonsai 2 4B", 32_768)?,
                model("bonsai-2-2b", "Bonsai 2 2B", 32_768)?,
            ],
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env(vars: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let vars: Vec<(String, String)> = vars
            .iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect();
        move |name| vars.iter().find(|(k, _)| k == name).map(|(_, v)| v.clone())
    }

    #[test]
    fn defaults_to_bonsai_over_http() {
        let config = AppConfig::load([], env(&[])).unwrap();
        assert!(matches!(config.provider, ProviderChoice::OpenAi(_)));
        assert_eq!(config.catalog.default_model().id.as_str(), "bonsai-2-9b");
    }

    #[test]
    fn fake_flag_selects_fake_provider() {
        let config = AppConfig::load(["--fake".to_owned()], env(&[])).unwrap();
        assert_eq!(config.provider, ProviderChoice::Fake);
        assert_eq!(config.catalog.default_model().id.as_str(), "fake-echo");
    }

    #[test]
    fn env_selects_fake_provider() {
        let config = AppConfig::load([], env(&[(PROVIDER_VAR, "fake")])).unwrap();
        assert_eq!(config.provider, ProviderChoice::Fake);
    }

    #[test]
    fn rejects_unknown_provider() {
        assert!(AppConfig::load([], env(&[(PROVIDER_VAR, "nope")])).is_err());
    }

    #[test]
    fn rejects_unknown_reasoning_request() {
        let error = AppConfig::load([], env(&[("HARNESS_REASONING", "maybe")])).unwrap_err();
        assert!(format!("{error:#}").contains("HARNESS_REASONING"));
    }

    #[test]
    fn models_var_overrides_catalog() {
        let config = AppConfig::load([], env(&[(MODELS_VAR, "qwen=4096")])).unwrap();
        assert_eq!(config.catalog.default_model().context_window.tokens(), 4096);
    }
}
