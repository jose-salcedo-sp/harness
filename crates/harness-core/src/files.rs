//! Workspace files the user can mention in a message, and the [`FileIndex`]
//! port that lists them.
//!
//! The composer turns an `@query` token into a file path in three steps:
//! [`FileMention::at_cursor`] finds the token, [`match_files`] ranks the
//! indexed paths against its query, and [`FileMention::accept`] produces the
//! edit that swaps the token for the chosen path.
//!
//! Adapters: [`WorkspaceFiles`] walks the real workspace, honouring
//! `.gitignore`, and [`MemoryFileIndex`] serves a fixed list for tests.

mod fuzzy;
mod memory;
mod mention;
mod walk;

use std::{
    fmt,
    path::{Component, Path, PathBuf},
};

use futures::future::BoxFuture;

pub use fuzzy::{FileMatch, match_files};
pub use memory::MemoryFileIndex;
pub use mention::{FileMention, MentionEdit};
pub use walk::{DEFAULT_FILE_LIMIT, WorkspaceFiles};

/// A file's path relative to the workspace root, with `/` separators.
///
/// Never empty, never absolute, and free of `.` and `..` segments, so it
/// cannot name anything outside the workspace.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct WorkspacePath(String);

/// Returned when a string is not a relative, normalized workspace path.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("`{path}` is not a relative workspace path")]
pub struct InvalidWorkspacePath {
    /// The rejected input.
    pub path: String,
}

impl WorkspacePath {
    /// Validates a `/`-separated path relative to the workspace root.
    ///
    /// # Errors
    ///
    /// Returns [`InvalidWorkspacePath`] when `path` is empty, starts with
    /// `/`, or has an empty, `.`, or `..` segment.
    pub fn new(path: impl Into<String>) -> Result<Self, InvalidWorkspacePath> {
        let path = path.into();
        let valid = !path.is_empty()
            && path
                .split('/')
                .all(|segment| !matches!(segment, "" | "." | ".."));
        if valid {
            Ok(Self(path))
        } else {
            Err(InvalidWorkspacePath { path })
        }
    }

    /// Converts a relative file-system path, or returns `None` when it has a
    /// non-UTF-8 or non-normal component.
    fn from_relative(path: &Path) -> Option<Self> {
        let segments = path
            .components()
            .map(|component| match component {
                Component::Normal(segment) => segment.to_str(),
                Component::Prefix(_)
                | Component::RootDir
                | Component::CurDir
                | Component::ParentDir => None,
            })
            .collect::<Option<Vec<_>>>()?;
        Self::new(segments.join("/")).ok()
    }

    /// The path text.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// How many directories deep the file sits; `0` for the workspace root.
    pub fn depth(&self) -> usize {
        self.0.matches('/').count()
    }
}

impl fmt::Display for WorkspacePath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Why the workspace files could not be listed.
#[derive(Debug, thiserror::Error)]
pub enum FileIndexError {
    /// The workspace root could not be read.
    #[error("workspace root `{}` is not readable", path.display())]
    RootUnreadable {
        /// The root that was walked.
        path: PathBuf,
        /// The underlying IO error.
        #[source]
        source: std::io::Error,
    },
}

/// Port: lists the files in the workspace.
///
/// The returned future is `Send + 'static` so the UI can run it on a
/// background executor.
pub trait FileIndex: Send + Sync {
    /// Every file the user may mention, sorted by path.
    fn files(&self) -> BoxFuture<'static, Result<Vec<WorkspacePath>, FileIndexError>>;
}

/// Shared contract every [`FileIndex`] adapter must satisfy.
#[cfg(test)]
mod tests_support {
    use futures::executor::block_on;

    use super::FileIndex;

    /// Asserts that `index`, holding `Cargo.toml`, `README.md`, and
    /// `src/lib.rs`, lists exactly those paths, sorted and `/`-separated.
    pub(crate) fn assert_lists_sorted_files(index: &impl FileIndex) {
        let files: Vec<String> = block_on(index.files())
            .unwrap()
            .into_iter()
            .map(|path| path.as_str().to_owned())
            .collect();
        assert_eq!(files, ["Cargo.toml", "README.md", "src/lib.rs"]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_nested_relative_path() {
        let path = WorkspacePath::new("crates/harness-core/src/lib.rs").unwrap();
        assert_eq!(path.as_str(), "crates/harness-core/src/lib.rs");
        assert_eq!(path.depth(), 3);
    }

    #[test]
    fn rejects_paths_that_could_leave_the_workspace() {
        for bad in [
            "",
            "/etc/passwd",
            "../secret",
            "a/../b",
            "a//b",
            "./a",
            "a/",
        ] {
            assert_eq!(
                WorkspacePath::new(bad),
                Err(InvalidWorkspacePath {
                    path: bad.to_owned()
                }),
                "{bad:?} should be rejected"
            );
        }
    }

    #[test]
    fn converts_relative_file_system_path() {
        let path = WorkspacePath::from_relative(Path::new("src").join("main.rs").as_path());
        assert_eq!(path, Some(WorkspacePath::new("src/main.rs").unwrap()));
    }

    #[test]
    fn refuses_to_convert_parent_components() {
        assert_eq!(WorkspacePath::from_relative(Path::new("../x.rs")), None);
    }
}
