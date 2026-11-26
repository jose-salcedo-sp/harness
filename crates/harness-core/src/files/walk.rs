//! File-system adapter for [`FileIndex`], built on the `ignore` crate.

use std::path::{Path, PathBuf};

use futures::future::BoxFuture;
use ignore::WalkBuilder;

use super::{FileIndex, FileIndexError, WorkspacePath};
use crate::tool::FsTools;

/// Most files [`WorkspaceFiles`] lists by default.
pub const DEFAULT_FILE_LIMIT: usize = 20_000;

/// Lists the files under the directory [`FsTools`] reads from.
///
/// The walk honours `.gitignore` (also outside a git repository), `.ignore`,
/// and git's exclude files, skips `.git`, and does not follow symbolic links,
/// so every listed path is a regular file inside the root. Hidden files that
/// are not ignored, such as `.github/`, are listed.
///
/// The walk is blocking and happens when the returned future is polled, so
/// run it on a background executor, never on a UI thread.
#[derive(Debug, Clone)]
pub struct WorkspaceFiles {
    root: PathBuf,
    limit: usize,
}

impl WorkspaceFiles {
    /// Lists files under `tools`' root, so mentions and `read_file` agree on
    /// what the workspace is.
    pub fn new(tools: &FsTools) -> Self {
        Self {
            root: tools.root().to_path_buf(),
            limit: DEFAULT_FILE_LIMIT,
        }
    }

    /// Sets the most files a listing returns; the walk stops there.
    #[must_use]
    pub fn limit(mut self, files: usize) -> Self {
        self.limit = files;
        self
    }

    /// The directory that is walked.
    pub fn root(&self) -> &Path {
        &self.root
    }
}

impl FileIndex for WorkspaceFiles {
    fn files(&self) -> BoxFuture<'static, Result<Vec<WorkspacePath>, FileIndexError>> {
        let index = self.clone();
        Box::pin(async move { walk(&index.root, index.limit) })
    }
}

fn walk(root: &Path, limit: usize) -> Result<Vec<WorkspacePath>, FileIndexError> {
    std::fs::read_dir(root).map_err(|source| FileIndexError::RootUnreadable {
        path: root.to_path_buf(),
        source,
    })?;
    let walker = WalkBuilder::new(root)
        .hidden(false)
        .follow_links(false)
        .require_git(false)
        .sort_by_file_name(Ord::cmp)
        .filter_entry(|entry| entry.file_name() != ".git")
        .build();
    let mut files = Vec::new();
    for entry in walker {
        if files.len() >= limit {
            break;
        }
        // An unreadable directory or malformed ignore file hides only its own
        // entries; the rest of the workspace is still worth offering.
        let Ok(entry) = entry else { continue };
        if !entry.file_type().is_some_and(|kind| kind.is_file()) {
            continue;
        }
        let path = entry
            .path()
            .strip_prefix(root)
            .ok()
            .and_then(WorkspacePath::from_relative);
        files.extend(path);
    }
    files.sort();
    Ok(files)
}

#[cfg(test)]
mod tests {
    use futures::executor::block_on;

    use super::*;
    use crate::files::{MemoryFileIndex, tests_support::assert_lists_sorted_files};

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(name: &str) -> Self {
            let dir =
                std::env::temp_dir().join(format!("harness-walk-{name}-{}", std::process::id()));
            // A leftover directory from an earlier run would add stray files.
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }

        fn write(&self, path: &str, contents: &str) {
            let path = self.0.join(path);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, contents).unwrap();
        }

        fn index(&self) -> WorkspaceFiles {
            WorkspaceFiles::new(&FsTools::new(&self.0).unwrap())
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            // Best-effort cleanup; a leftover temp dir does not affect results.
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn listed(index: &WorkspaceFiles) -> Vec<String> {
        block_on(index.files())
            .unwrap()
            .into_iter()
            .map(|path| path.as_str().to_owned())
            .collect()
    }

    #[test]
    fn lists_nested_files_relative_to_root() {
        let dir = TempDir::new("nested");
        dir.write("b.txt", "");
        dir.write("src/a.rs", "");
        dir.write(".github/ci.yml", "");
        assert_eq!(
            listed(&dir.index()),
            [".github/ci.yml", "b.txt", "src/a.rs"]
        );
    }

    #[test]
    fn respects_gitignore_without_a_repository() {
        let dir = TempDir::new("gitignore");
        dir.write(".gitignore", "target/\n*.log\n");
        dir.write("keep.rs", "");
        dir.write("debug.log", "");
        dir.write("target/out.bin", "");
        assert_eq!(listed(&dir.index()), [".gitignore", "keep.rs"]);
    }

    #[test]
    fn skips_git_directory() {
        let dir = TempDir::new("dotgit");
        dir.write(".git/HEAD", "ref: refs/heads/main");
        dir.write("main.rs", "");
        assert_eq!(listed(&dir.index()), ["main.rs"]);
    }

    #[test]
    fn stays_inside_the_tools_root() {
        let dir = TempDir::new("confined");
        dir.write("outside.txt", "");
        dir.write("inner/inside.txt", "");
        let tools = FsTools::new(dir.0.join("inner")).unwrap();
        let index = WorkspaceFiles::new(&tools);
        assert_eq!(index.root(), tools.root());
        assert_eq!(listed(&index), ["inside.txt"]);
    }

    #[cfg(unix)]
    #[test]
    fn does_not_list_symlinks_that_escape_the_root() {
        let dir = TempDir::new("symlink");
        dir.write("secret.txt", "");
        dir.write("inner/real.txt", "");
        std::os::unix::fs::symlink(dir.0.join("secret.txt"), dir.0.join("inner/link.txt")).unwrap();
        std::os::unix::fs::symlink(&dir.0, dir.0.join("inner/up")).unwrap();
        let index = WorkspaceFiles::new(&FsTools::new(dir.0.join("inner")).unwrap());
        assert_eq!(listed(&index), ["real.txt"]);
    }

    #[test]
    fn stops_at_the_limit() {
        let dir = TempDir::new("limit");
        for name in ["a", "b", "c"] {
            dir.write(name, "");
        }
        assert_eq!(listed(&dir.index().limit(2)), ["a", "b"]);
    }

    #[test]
    fn reports_a_root_that_disappeared() {
        let dir = TempDir::new("vanished");
        let index = dir.index();
        std::fs::remove_dir_all(&dir.0).unwrap();
        assert!(matches!(
            block_on(index.files()),
            Err(FileIndexError::RootUnreadable { .. })
        ));
    }

    #[test]
    fn satisfies_the_file_index_contract() {
        let dir = TempDir::new("contract");
        dir.write("src/lib.rs", "");
        dir.write("README.md", "");
        dir.write("Cargo.toml", "");
        assert_lists_sorted_files(&dir.index());
    }

    #[test]
    fn fake_satisfies_the_file_index_contract() {
        let fake = MemoryFileIndex::new(["src/lib.rs", "README.md", "Cargo.toml"]).unwrap();
        assert_lists_sorted_files(&fake);
    }
}
