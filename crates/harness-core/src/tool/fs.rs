//! File-system adapter for [`ToolExecutor`].

use std::path::{Path, PathBuf};

use futures::future::BoxFuture;

use super::{ToolError, ToolExecutor, ToolOutput, ToolRequest};

/// Largest file `read_file` returns by default, in bytes.
pub const DEFAULT_READ_LIMIT: u64 = 256 * 1024;

/// Runs tools against the real file system, confined to one root directory.
///
/// File reads are blocking and happen when the returned future is polled, so
/// drive the session on a background executor, never on a UI thread.
#[derive(Debug, Clone)]
pub struct FsTools {
    root: PathBuf,
    read_limit: u64,
}

/// Returned when the tool root directory cannot be resolved.
#[derive(Debug, thiserror::Error)]
#[error("workspace root `{}` is not accessible", path.display())]
pub struct RootUnavailable {
    /// The root that was requested.
    pub path: PathBuf,
    /// The underlying IO error.
    #[source]
    pub source: std::io::Error,
}

impl FsTools {
    /// Confines tools to `root`, which is resolved to an absolute path now.
    ///
    /// # Errors
    ///
    /// Returns [`RootUnavailable`] when `root` does not exist or cannot be
    /// resolved.
    pub fn new(root: impl AsRef<Path>) -> Result<Self, RootUnavailable> {
        let root = root.as_ref();
        let root = root.canonicalize().map_err(|source| RootUnavailable {
            path: root.to_path_buf(),
            source,
        })?;
        Ok(Self {
            root,
            read_limit: DEFAULT_READ_LIMIT,
        })
    }

    /// Sets the largest file, in bytes, that `read_file` returns.
    #[must_use]
    pub fn read_limit(mut self, bytes: u64) -> Self {
        self.read_limit = bytes;
        self
    }

    /// The directory tools are confined to.
    pub fn root(&self) -> &Path {
        &self.root
    }

    fn read_file(&self, requested: &Path) -> Result<ToolOutput, ToolError> {
        let read_error = |source| ToolError::Read {
            path: requested.to_path_buf(),
            source,
        };
        let resolved = self
            .root
            .join(requested)
            .canonicalize()
            .map_err(read_error)?;
        if !resolved.starts_with(&self.root) {
            return Err(ToolError::OutsideRoot {
                path: requested.to_path_buf(),
            });
        }
        let size = std::fs::metadata(&resolved).map_err(read_error)?.len();
        if size > self.read_limit {
            return Err(ToolError::TooLarge {
                path: requested.to_path_buf(),
                size,
                limit: self.read_limit,
            });
        }
        std::fs::read_to_string(&resolved)
            .map(ToolOutput::new)
            .map_err(read_error)
    }
}

impl ToolExecutor for FsTools {
    fn execute(&self, request: ToolRequest) -> BoxFuture<'static, Result<ToolOutput, ToolError>> {
        let tools = self.clone();
        Box::pin(async move {
            match request {
                ToolRequest::ReadFile { path } => tools.read_file(&path),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use futures::executor::block_on;

    use super::*;

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(name: &str) -> Self {
            let dir =
                std::env::temp_dir().join(format!("harness-fs-{name}-{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            // Best-effort cleanup; a leftover temp dir does not affect results.
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn read(tools: &FsTools, path: &str) -> Result<ToolOutput, ToolError> {
        block_on(tools.execute(ToolRequest::ReadFile {
            path: PathBuf::from(path),
        }))
    }

    #[test]
    fn reads_file_inside_root() {
        let dir = TempDir::new("inside");
        std::fs::write(dir.0.join("notes.txt"), "hello").unwrap();
        let tools = FsTools::new(&dir.0).unwrap();
        assert_eq!(read(&tools, "notes.txt").unwrap().as_str(), "hello");
    }

    #[test]
    fn refuses_paths_outside_root() {
        let dir = TempDir::new("outside");
        let inner = dir.0.join("inner");
        std::fs::create_dir_all(&inner).unwrap();
        std::fs::write(dir.0.join("secret.txt"), "nope").unwrap();
        let tools = FsTools::new(&inner).unwrap();
        assert!(matches!(
            read(&tools, "../secret.txt"),
            Err(ToolError::OutsideRoot { .. })
        ));
    }

    #[test]
    fn refuses_files_over_limit() {
        let dir = TempDir::new("limit");
        std::fs::write(dir.0.join("big.txt"), "0123456789").unwrap();
        let tools = FsTools::new(&dir.0).unwrap().read_limit(4);
        assert!(matches!(
            read(&tools, "big.txt"),
            Err(ToolError::TooLarge {
                size: 10,
                limit: 4,
                ..
            })
        ));
    }

    #[test]
    fn reports_missing_file() {
        let dir = TempDir::new("missing");
        let tools = FsTools::new(&dir.0).unwrap();
        assert!(matches!(
            read(&tools, "nope.txt"),
            Err(ToolError::Read { .. })
        ));
    }
}
