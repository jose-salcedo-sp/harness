//! In-memory fake for [`FileIndex`], for tests and offline demos.

use std::sync::Arc;

use futures::future::BoxFuture;

use super::{FileIndex, FileIndexError, InvalidWorkspacePath, WorkspacePath};

/// Serves a fixed list of workspace paths.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MemoryFileIndex {
    files: Arc<[WorkspacePath]>,
}

impl MemoryFileIndex {
    /// Creates an index holding `paths`, sorted and deduplicated.
    ///
    /// # Errors
    ///
    /// Returns [`InvalidWorkspacePath`] for the first entry that is not a
    /// relative workspace path.
    pub fn new(
        paths: impl IntoIterator<Item = impl Into<String>>,
    ) -> Result<Self, InvalidWorkspacePath> {
        let mut files = paths
            .into_iter()
            .map(WorkspacePath::new)
            .collect::<Result<Vec<_>, _>>()?;
        files.sort();
        files.dedup();
        Ok(Self {
            files: files.into(),
        })
    }
}

impl FileIndex for MemoryFileIndex {
    fn files(&self) -> BoxFuture<'static, Result<Vec<WorkspacePath>, FileIndexError>> {
        Box::pin(std::future::ready(Ok(self.files.to_vec())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_paths() {
        assert_eq!(
            MemoryFileIndex::new(["ok.rs", "../escape.rs"]),
            Err(InvalidWorkspacePath {
                path: "../escape.rs".to_owned()
            })
        );
    }
}
