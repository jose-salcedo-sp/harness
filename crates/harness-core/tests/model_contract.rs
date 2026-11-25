//! Contract tests every `ModelProvider` adapter must pass.
//!
//! The fake runs in CI. The HTTP adapter test is ignored because it needs a
//! running server; run it with
//! `HARNESS_BASE_URL=http://127.0.0.1:8080/v1 HARNESS_MODEL=bonsai-2-9b
//! cargo test -p harness-core --test model_contract -- --ignored`
//! (and `HARNESS_API_KEY` if the server requires one).

#[cfg(test)]
mod contract {
    use futures::{StreamExt as _, executor::block_on};
    use harness_core::{
        ModelChunk, ModelId, ModelProvider, ModelRequest, SystemPrompt, ToolSpec, UserText,
        conversation::Message,
        model::{
            fake::FakeProvider,
            openai::{OpenAiConfig, OpenAiProvider},
        },
    };

    fn request(model: &str, text: &str) -> ModelRequest {
        ModelRequest {
            model: ModelId::new(model).unwrap(),
            system: SystemPrompt::new("Reply in one short sentence."),
            messages: vec![Message::User(UserText::parse(text).unwrap())],
            tools: ToolSpec::builtin(),
        }
    }

    fn streams_a_text_reply(provider: &impl ModelProvider, model: &str) {
        let chunks: Vec<ModelChunk> = block_on(
            provider
                .stream(request(model, "Say hello."))
                .map(|item| item.expect("stream item"))
                .collect(),
        );
        let text: String = chunks
            .iter()
            .filter_map(|chunk| match chunk {
                ModelChunk::Text(text) => Some(text.as_str()),
                ModelChunk::ToolCall(_) | ModelChunk::Usage(_) => None,
            })
            .collect();
        assert!(
            !text.trim().is_empty(),
            "expected reply text, got {chunks:?}"
        );
    }

    #[test]
    fn fake_provider_streams_a_text_reply() {
        streams_a_text_reply(&FakeProvider::echo(), "fake-echo");
    }

    #[test]
    #[ignore = "needs an OpenAI-compatible server; see the module docs"]
    fn openai_provider_streams_a_text_reply() {
        let config = OpenAiConfig::from_env(|name| std::env::var(name).ok()).unwrap();
        let model = std::env::var("HARNESS_MODEL").unwrap_or_else(|_| "bonsai-2-9b".to_owned());
        streams_a_text_reply(&OpenAiProvider::new(config), &model);
    }
}
