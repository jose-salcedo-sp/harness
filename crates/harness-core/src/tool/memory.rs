//! In-memory fake for [`ToolExecutor`], for tests and offline demos.

use std::{collections::HashMap, path::PathBuf, sync::Arc};

use futures::future::BoxFuture;

use super::{ToolError, ToolExecutor, ToolOutput, ToolRequest};

/// Serves `read_file` from a fixed map of paths to contents.
#[derive(Debug, Clone, Default)]
pub struct MemoryTools {
    files: Arc<HashMap<PathBuf, String>>,
}

impl MemoryTools {
    /// Creates a fake file system holding `files`.
    pub fn new<P, C>(files: impl IntoIterator<Item = (P, C)>) -> Self
    where
        P: Into<PathBuf>,
        C: Into<String>,
    {
        Self {
            files: Arc::new(
                files
                    .into_iter()
                    .map(|(path, contents)| (path.into(), contents.into()))
                    .collect(),
            ),
        }
    }
}

impl ToolExecutor for MemoryTools {
    fn execute(&self, request: ToolRequest) -> BoxFuture<'static, Result<ToolOutput, ToolError>> {
        let result = match request {
            ToolRequest::ReadFile { path } => match self.files.get(&path) {
                Some(contents) => Ok(ToolOutput::new(contents.clone())),
                None => Err(ToolError::Read {
                    path,
                    source: std::io::ErrorKind::NotFound.into(),
                }),
            },
        };
        Box::pin(std::future::ready(result))
    }
}
