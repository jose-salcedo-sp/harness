//! Splits inline `<think>…</think>` spans out of streamed reply text.
//!
//! Servers that do not parse reasoning themselves (for example `llama-server`
//! with `--reasoning-format none`, or a template it has no parser for) stream
//! the model's thoughts inside the content, wrapped in think tags. Tags can be
//! split across chunks, so the splitter holds back any suffix that might be the
//! start of a tag until the next chunk decides it.

use super::ModelChunk;

const OPEN: &str = "<think>";
const CLOSE: &str = "</think>";

/// Which part of the response the text currently belongs to.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
enum Section {
    #[default]
    Answer,
    Thinking,
}

impl Section {
    /// The tag that ends this section.
    fn closing_tag(self) -> &'static str {
        match self {
            Self::Answer => OPEN,
            Self::Thinking => CLOSE,
        }
    }

    fn toggled(self) -> Self {
        match self {
            Self::Answer => Self::Thinking,
            Self::Thinking => Self::Answer,
        }
    }

    fn chunk(self, text: &str) -> Option<ModelChunk> {
        (!text.is_empty()).then(|| match self {
            Self::Answer => ModelChunk::Text(text.to_owned()),
            Self::Thinking => ModelChunk::Reasoning(text.to_owned()),
        })
    }
}

/// Turns content deltas into [`ModelChunk::Text`] and [`ModelChunk::Reasoning`].
#[derive(Debug, Default)]
pub(super) struct ThinkTagSplitter {
    section: Section,
    /// Text held back because it may be the start of a tag.
    held: String,
}

impl ThinkTagSplitter {
    /// Splits one content delta. Text that might begin a tag is held back.
    pub(super) fn push(&mut self, delta: &str) -> Vec<ModelChunk> {
        let mut rest = std::mem::take(&mut self.held);
        rest.push_str(delta);
        let mut chunks = Vec::new();
        loop {
            let tag = self.section.closing_tag();
            if let Some((before, after)) = rest.split_once(tag) {
                chunks.extend(self.section.chunk(before));
                self.section = self.section.toggled();
                rest = after.to_owned();
                continue;
            }
            let cut = rest.len() - partial_tag_suffix(&rest, tag);
            let (emit, hold) = rest.split_at_checked(cut).unwrap_or((&rest, ""));
            chunks.extend(self.section.chunk(emit));
            hold.clone_into(&mut self.held);
            return chunks;
        }
    }

    /// Flushes held-back text once the stream has ended. An unclosed `<think>`
    /// keeps its text as reasoning.
    pub(super) fn finish(self) -> Option<ModelChunk> {
        self.section.chunk(&self.held)
    }
}

/// Length of the longest suffix of `text` that is a proper prefix of `tag`.
fn partial_tag_suffix(text: &str, tag: &str) -> usize {
    (1..tag.len())
        .rev()
        .find(|&len| tag.get(..len).is_some_and(|prefix| text.ends_with(prefix)))
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn split(deltas: &[&str]) -> Vec<ModelChunk> {
        let mut splitter = ThinkTagSplitter::default();
        let mut chunks: Vec<ModelChunk> = deltas
            .iter()
            .flat_map(|delta| splitter.push(delta))
            .collect();
        chunks.extend(splitter.finish());
        merge(chunks)
    }

    /// Joins adjacent chunks of the same kind, so tests assert on content
    /// rather than on where the splitter happened to cut.
    fn merge(chunks: Vec<ModelChunk>) -> Vec<ModelChunk> {
        let mut merged: Vec<ModelChunk> = Vec::new();
        for chunk in chunks {
            match (merged.last_mut(), chunk) {
                (Some(ModelChunk::Text(a)), ModelChunk::Text(b))
                | (Some(ModelChunk::Reasoning(a)), ModelChunk::Reasoning(b)) => a.push_str(&b),
                (_, chunk) => merged.push(chunk),
            }
        }
        merged
    }

    fn reasoning(text: &str) -> ModelChunk {
        ModelChunk::Reasoning(text.to_owned())
    }

    fn text(text: &str) -> ModelChunk {
        ModelChunk::Text(text.to_owned())
    }

    #[test]
    fn passes_plain_text_through() {
        assert_eq!(split(&["Hello ", "world"]), [text("Hello world")]);
    }

    #[test]
    fn splits_tags_within_one_chunk() {
        assert_eq!(
            split(&["<think>plan</think>answer"]),
            [reasoning("plan"), text("answer")]
        );
    }

    #[test]
    fn splits_tags_cut_across_chunks() {
        assert_eq!(
            split(&["<th", "ink>pl", "an</", "thi", "nk>ans", "wer"]),
            [reasoning("plan"), text("answer")]
        );
    }

    #[test]
    fn splits_tags_cut_at_every_byte() {
        let input = "<think>a < b</think>so <yes>";
        let deltas: Vec<String> = input.chars().map(String::from).collect();
        let deltas: Vec<&str> = deltas.iter().map(String::as_str).collect();
        assert_eq!(split(&deltas), [reasoning("a < b"), text("so <yes>")]);
    }

    #[test]
    fn emits_text_before_a_possible_tag_without_waiting() {
        let mut splitter = ThinkTagSplitter::default();
        assert_eq!(splitter.push("hi <thi"), [text("hi ")]);
        assert_eq!(
            splitter.push("s is not a tag"),
            [text("<this is not a tag")]
        );
    }

    #[test]
    fn unclosed_think_stays_reasoning() {
        assert_eq!(
            split(&["<think>still going</th"]),
            [reasoning("still going</th")]
        );
    }

    #[test]
    fn held_text_is_flushed_at_the_end() {
        assert_eq!(split(&["a <"]), [text("a <")]);
    }

    #[test]
    fn keeps_multibyte_text_intact() {
        assert_eq!(
            split(&["<think>café ", "→</think>", "ñ"]),
            [reasoning("café →"), text("ñ")]
        );
    }
}
