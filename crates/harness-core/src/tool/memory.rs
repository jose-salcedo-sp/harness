//! In-memory fake for [`ToolExecutor`], for tests and offline demos.

use std::{
    collections::HashMap,
    path::{Component, Path, PathBuf},
    sync::{Arc, Mutex, PoisonError},
};

use futures::future::BoxFuture;

use super::{
    DEFAULT_WRITE_LIMIT, FileWrite, ToolError, ToolExecutor, ToolOutput, ToolRequest, WriteKind,
    check_write_size,
};

/// Serves `read_file` and `write_file` from a map of paths to contents.
///
/// Paths are relative to an imaginary root: absolute paths and `..` that
/// climbs above it are rejected like [`FsTools`](super::FsTools) rejects
/// escapes. Clones share the same files, so a test can keep a clone to
/// inspect what a session wrote.
#[derive(Debug, Clone, Default)]
pub struct MemoryTools {
    files: Arc<Mutex<HashMap<PathBuf, String>>>,
}

impl MemoryTools {
    /// Creates a fake file system holding `files`.
    pub fn new<P, C>(files: impl IntoIterator<Item = (P, C)>) -> Self
    where
        P: Into<PathBuf>,
        C: Into<String>,
    {
        let files = files
            .into_iter()
            .map(|(path, contents)| {
                let path = path.into();
                (normalize(&path).unwrap_or(path), contents.into())
            })
            .collect();
        Self {
            files: Arc::new(Mutex::new(files)),
        }
    }

    /// The current contents of `path`, if the fake holds it.
    pub fn file(&self, path: impl AsRef<Path>) -> Option<String> {
        let path = normalize(path.as_ref())?;
        self.files
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(&path)
            .cloned()
    }

    fn run(&self, request: ToolRequest) -> Result<ToolOutput, ToolError> {
        match request {
            ToolRequest::ReadFile { path } => {
                let key = normalize(&path)
                    .ok_or_else(|| ToolError::OutsideRoot { path: path.clone() })?;
                self.file(&key)
                    .map(ToolOutput::new)
                    .ok_or_else(|| ToolError::Read {
                        path,
                        source: std::io::ErrorKind::NotFound.into(),
                    })
            }
            ToolRequest::WriteFile { path, contents } => {
                check_write_size(&path, &contents, DEFAULT_WRITE_LIMIT)?;
                let Some(key) = normalize(&path) else {
                    return Err(ToolError::OutsideRoot { path });
                };
                let previous = self
                    .files
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .insert(key, contents.clone());
                let kind = match previous {
                    Some(_) => WriteKind::Overwritten,
                    None => WriteKind::Created,
                };
                Ok(ToolOutput::written(FileWrite::new(path, kind, &contents)))
            }
        }
    }
}

/// Resolves `.` and `..` lexically; `None` when `path` is absolute, climbs
/// above the root, or names the root itself.
fn normalize(path: &Path) -> Option<PathBuf> {
    let mut normal = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Normal(name) => normal.push(name),
            Component::CurDir => {}
            Component::ParentDir => {
                if !normal.pop() {
                    return None;
                }
            }
            Component::RootDir | Component::Prefix(_) => return None,
        }
    }
    (!normal.as_os_str().is_empty()).then_some(normal)
}

impl ToolExecutor for MemoryTools {
    fn execute(&self, request: ToolRequest) -> BoxFuture<'static, Result<ToolOutput, ToolError>> {
        Box::pin(std::future::ready(self.run(request)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::contract;

    #[test]
    fn write_contract_creates_then_overwrites() {
        contract::creates_then_overwrites(&MemoryTools::default());
    }

    #[test]
    fn write_contract_creates_parent_directories() {
        contract::creates_parent_directories(&MemoryTools::default());
    }

    #[test]
    fn write_contract_rejects_escapes() {
        contract::rejects_escapes(&MemoryTools::default());
    }

    #[test]
    fn write_contract_rejects_contents_over_default_limit() {
        contract::rejects_contents_over_default_limit(&MemoryTools::default());
    }

    #[test]
    fn clones_share_written_files() {
        let tools = MemoryTools::default();
        let observer = tools.clone();
        tools
            .run(ToolRequest::WriteFile {
                path: PathBuf::from("./a/../b.txt"),
                contents: "hi".to_owned(),
            })
            .unwrap();
        assert_eq!(observer.file("b.txt").as_deref(), Some("hi"));
    }
}
