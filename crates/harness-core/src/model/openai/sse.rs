//! Decodes the server-sent events of a streamed chat completion.

use std::collections::BTreeMap;

use serde::Deserialize;

use crate::{
    model::{ModelChunk, ModelError, TokenUsage},
    tool::{ToolArguments, ToolCall, ToolCallId, ToolName},
};

#[derive(Deserialize)]
struct StreamEvent {
    #[serde(default)]
    choices: Vec<Choice>,
    usage: Option<Usage>,
}

#[derive(Deserialize)]
struct Choice {
    #[serde(default)]
    delta: Delta,
}

#[derive(Deserialize, Default)]
struct Delta {
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<ToolCallDelta>,
}

#[derive(Deserialize)]
struct ToolCallDelta {
    index: u64,
    id: Option<String>,
    function: Option<FunctionDelta>,
}

#[derive(Deserialize)]
struct FunctionDelta {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Deserialize)]
struct Usage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

#[derive(Default)]
struct PartialCall {
    id: String,
    name: String,
    arguments: String,
}

/// Turns `data:` lines into [`ModelChunk`]s.
///
/// Text is emitted as soon as it arrives. Tool calls arrive in fragments, so
/// they are assembled and emitted by [`SseDecoder::finish`].
#[derive(Default)]
pub(super) struct SseDecoder {
    calls: BTreeMap<u64, PartialCall>,
    done: bool,
}

impl SseDecoder {
    /// Whether the server sent the `[DONE]` sentinel.
    pub(super) fn is_done(&self) -> bool {
        self.done
    }

    /// Decodes one line of the event stream.
    pub(super) fn push_line(&mut self, line: &str) -> Result<Vec<ModelChunk>, ModelError> {
        let Some(data) = line.strip_prefix("data:").map(str::trim) else {
            return Ok(Vec::new());
        };
        if data == "[DONE]" {
            self.done = true;
            return Ok(Vec::new());
        }
        let event: StreamEvent = serde_json::from_str(data).map_err(ModelError::Decode)?;
        let mut chunks = Vec::new();
        for delta in event.choices.into_iter().map(|choice| choice.delta) {
            if let Some(text) = delta.content.filter(|text| !text.is_empty()) {
                chunks.push(ModelChunk::Text(text));
            }
            for fragment in delta.tool_calls {
                let call = self.calls.entry(fragment.index).or_default();
                if let Some(id) = fragment.id {
                    call.id.push_str(&id);
                }
                if let Some(function) = fragment.function {
                    call.name
                        .push_str(function.name.as_deref().unwrap_or_default());
                    call.arguments
                        .push_str(function.arguments.as_deref().unwrap_or_default());
                }
            }
        }
        if let Some(usage) = event.usage {
            chunks.push(ModelChunk::Usage(TokenUsage {
                prompt_tokens: usage.prompt_tokens,
                completion_tokens: usage.completion_tokens,
            }));
        }
        Ok(chunks)
    }

    /// Emits the assembled tool calls once the stream has ended.
    pub(super) fn finish(self) -> Result<Vec<ModelChunk>, ModelError> {
        self.calls
            .into_iter()
            .map(|(index, call)| {
                let incomplete = |_| ModelError::IncompleteToolCall { index };
                Ok(ModelChunk::ToolCall(ToolCall {
                    id: ToolCallId::new(call.id).map_err(incomplete)?,
                    name: ToolName::new(call.name).map_err(incomplete)?,
                    arguments: ToolArguments::new(if call.arguments.is_empty() {
                        "{}".to_owned()
                    } else {
                        call.arguments
                    }),
                }))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode(lines: &[&str]) -> Result<Vec<ModelChunk>, ModelError> {
        let mut decoder = SseDecoder::default();
        let mut chunks = Vec::new();
        for line in lines {
            chunks.extend(decoder.push_line(line)?);
        }
        chunks.extend(decoder.finish()?);
        Ok(chunks)
    }

    #[test]
    fn decodes_text_deltas_in_order() {
        let chunks = decode(&[
            r#"data: {"choices":[{"delta":{"role":"assistant","content":"Hel"}}]}"#,
            "",
            r#"data: {"choices":[{"delta":{"content":"lo"}}]}"#,
            "data: [DONE]",
        ])
        .unwrap();
        assert_eq!(
            chunks,
            [
                ModelChunk::Text("Hel".to_owned()),
                ModelChunk::Text("lo".to_owned())
            ]
        );
    }

    #[test]
    fn assembles_fragmented_tool_call() {
        let chunks = decode(&[
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"read_file","arguments":"{\"pa"}}]}}]}"#,
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"th\":\"a.txt\"}"}}]}}]}"#,
        ])
        .unwrap();
        assert_eq!(
            chunks,
            [ModelChunk::ToolCall(ToolCall {
                id: ToolCallId::new("call_1").unwrap(),
                name: ToolName::new("read_file").unwrap(),
                arguments: ToolArguments::new(r#"{"path":"a.txt"}"#),
            })]
        );
    }

    #[test]
    fn decodes_usage_from_final_event() {
        let chunks = decode(&[
            r#"data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":3,"total_tokens":15}}"#,
        ])
        .unwrap();
        assert_eq!(
            chunks,
            [ModelChunk::Usage(TokenUsage {
                prompt_tokens: 12,
                completion_tokens: 3
            })]
        );
    }

    #[test]
    fn ignores_comments_and_other_fields() {
        assert_eq!(
            decode(&[": keep-alive", "event: ping"]).unwrap(),
            Vec::new()
        );
    }

    #[test]
    fn marks_done_on_sentinel() {
        let mut decoder = SseDecoder::default();
        decoder.push_line("data: [DONE]").unwrap();
        assert!(decoder.is_done());
    }

    #[test]
    fn rejects_malformed_event() {
        assert!(matches!(
            decode(&["data: {oops"]),
            Err(ModelError::Decode(_))
        ));
    }

    #[test]
    fn rejects_tool_call_without_name() {
        assert!(matches!(
            decode(&[r#"data: {"choices":[{"delta":{"tool_calls":[{"index":2,"id":"c"}]}}]}"#]),
            Err(ModelError::IncompleteToolCall { index: 2 })
        ));
    }
}
