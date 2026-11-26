//! File-system adapter for [`ToolExecutor`].

use std::{
    fs::{self, OpenOptions},
    io::{self, Write as _},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use futures::future::BoxFuture;

use super::{
    FileWrite, ToolError, ToolExecutor, ToolOutput, ToolRequest, WriteKind, check_write_size,
};

/// Largest file `read_file` returns by default, in bytes.
pub const DEFAULT_READ_LIMIT: u64 = 256 * 1024;

/// Largest contents `write_file` writes by default, in bytes.
pub const DEFAULT_WRITE_LIMIT: u64 = 256 * 1024;

/// Distinguishes temp files from concurrent writes in one process.
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Runs tools against the real file system, confined to one root directory.
///
/// File IO is blocking and happens when the returned future is polled, so
/// drive the session on a background executor, never on a UI thread.
///
/// `write_file` writes the contents to a temporary file in the target's
/// directory and renames it over the target, so readers see either the old
/// file or the new one, never a partial write.
#[derive(Debug, Clone)]
pub struct FsTools {
    root: PathBuf,
    read_limit: u64,
    write_limit: u64,
}

/// Returned when the tool root directory cannot be resolved.
#[derive(Debug, thiserror::Error)]
#[error("workspace root `{}` is not accessible", path.display())]
pub struct RootUnavailable {
    /// The root that was requested.
    pub path: PathBuf,
    /// The underlying IO error.
    #[source]
    pub source: io::Error,
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
            write_limit: DEFAULT_WRITE_LIMIT,
        })
    }

    /// Sets the largest file, in bytes, that `read_file` returns.
    #[must_use]
    pub fn read_limit(mut self, bytes: u64) -> Self {
        self.read_limit = bytes;
        self
    }

    /// Sets the largest contents, in bytes, that `write_file` writes.
    #[must_use]
    pub fn write_limit(mut self, bytes: u64) -> Self {
        self.write_limit = bytes;
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
        let size = fs::metadata(&resolved).map_err(read_error)?.len();
        if size > self.read_limit {
            return Err(ToolError::TooLarge {
                path: requested.to_path_buf(),
                size,
                limit: self.read_limit,
            });
        }
        fs::read_to_string(&resolved)
            .map(ToolOutput::new)
            .map_err(read_error)
    }

    fn write_file(&self, requested: PathBuf, contents: &str) -> Result<ToolOutput, ToolError> {
        check_write_size(&requested, contents, self.write_limit)?;
        let write_error = |source| ToolError::Write {
            path: requested.clone(),
            source,
        };
        let Some(target) = self.write_target(&requested).map_err(write_error)? else {
            return Err(ToolError::OutsideRoot {
                path: requested.clone(),
            });
        };
        let is_directory = || ToolError::IsDirectory {
            path: requested.clone(),
        };
        let existing = match fs::metadata(&target) {
            Ok(metadata) if metadata.is_dir() => return Err(is_directory()),
            Ok(metadata) => Some(metadata),
            Err(error) if error.kind() == io::ErrorKind::NotFound => None,
            Err(error) => return Err(write_error(error)),
        };
        let (Some(parent), Some(name)) = (target.parent(), target.file_name()) else {
            return Err(is_directory());
        };
        fs::create_dir_all(parent).map_err(write_error)?;

        let temp = parent.join(format!(
            ".{}.harness-{}-{}.tmp",
            name.to_string_lossy(),
            std::process::id(),
            TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let permissions = existing.as_ref().map(fs::Metadata::permissions);
        if let Err(error) = write_and_rename(&temp, &target, contents, permissions) {
            // The temp file is ours alone; if it cannot be removed it is only litter.
            let _ = fs::remove_file(&temp);
            return Err(write_error(error));
        }
        let kind = match existing {
            Some(_) => WriteKind::Overwritten,
            None => WriteKind::Created,
        };
        Ok(ToolOutput::written(FileWrite::new(
            requested, kind, contents,
        )))
    }

    /// Resolves where a write to `requested` lands, or `None` when that is
    /// outside the root.
    ///
    /// The deepest existing ancestor is canonicalized, so symlinks cannot
    /// lead out of the root. The part that does not exist yet must consist of
    /// plain names, since `..` there cannot be resolved against the disk.
    fn write_target(&self, requested: &Path) -> io::Result<Option<PathBuf>> {
        let joined = self.root.join(requested);
        let mut missing = Vec::new();
        let mut existing = joined.as_path();
        let mut target = loop {
            match existing.canonicalize() {
                Ok(resolved) => break resolved,
                Err(error) if error.kind() == io::ErrorKind::NotFound => {
                    // `file_name` is `None` when the path ends in `..`.
                    let (Some(parent), Some(name)) = (existing.parent(), existing.file_name())
                    else {
                        return Ok(None);
                    };
                    missing.push(name);
                    existing = parent;
                }
                Err(error) => return Err(error),
            }
        };
        for name in missing.into_iter().rev() {
            target.push(name);
        }
        Ok(target.starts_with(&self.root).then_some(target))
    }
}

fn write_and_rename(
    temp: &Path,
    target: &Path,
    contents: &str,
    permissions: Option<fs::Permissions>,
) -> io::Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(temp)?;
    file.write_all(contents.as_bytes())?;
    if let Some(permissions) = permissions {
        file.set_permissions(permissions)?;
    }
    file.sync_all()?;
    drop(file);
    fs::rename(temp, target)
}

impl ToolExecutor for FsTools {
    fn execute(&self, request: ToolRequest) -> BoxFuture<'static, Result<ToolOutput, ToolError>> {
        let tools = self.clone();
        Box::pin(async move {
            match request {
                ToolRequest::ReadFile { path } => tools.read_file(&path),
                ToolRequest::WriteFile { path, contents } => tools.write_file(path, &contents),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use futures::executor::block_on;

    use super::*;
    use crate::tool::contract;

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(name: &str) -> Self {
            let dir =
                std::env::temp_dir().join(format!("harness-fs-{name}-{}", std::process::id()));
            // Start clean in case an earlier run left the directory behind.
            let _ = fs::remove_dir_all(&dir);
            fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            // Best-effort cleanup; a leftover temp dir does not affect results.
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn read(tools: &FsTools, path: &str) -> Result<ToolOutput, ToolError> {
        block_on(tools.execute(ToolRequest::ReadFile {
            path: PathBuf::from(path),
        }))
    }

    fn write(tools: &FsTools, path: &str, contents: &str) -> Result<ToolOutput, ToolError> {
        block_on(tools.execute(ToolRequest::WriteFile {
            path: PathBuf::from(path),
            contents: contents.to_owned(),
        }))
    }

    #[test]
    fn reads_file_inside_root() {
        let dir = TempDir::new("inside");
        fs::write(dir.0.join("notes.txt"), "hello").unwrap();
        let tools = FsTools::new(&dir.0).unwrap();
        assert_eq!(read(&tools, "notes.txt").unwrap().as_str(), "hello");
    }

    #[test]
    fn refuses_paths_outside_root() {
        let dir = TempDir::new("outside");
        let inner = dir.0.join("inner");
        fs::create_dir_all(&inner).unwrap();
        fs::write(dir.0.join("secret.txt"), "nope").unwrap();
        let tools = FsTools::new(&inner).unwrap();
        assert!(matches!(
            read(&tools, "../secret.txt"),
            Err(ToolError::OutsideRoot { .. })
        ));
    }

    #[test]
    fn refuses_files_over_limit() {
        let dir = TempDir::new("limit");
        fs::write(dir.0.join("big.txt"), "0123456789").unwrap();
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

    #[test]
    fn write_contract_creates_then_overwrites() {
        let dir = TempDir::new("contract-overwrite");
        contract::creates_then_overwrites(&FsTools::new(&dir.0).unwrap());
    }

    #[test]
    fn write_contract_creates_parent_directories() {
        let dir = TempDir::new("contract-parents");
        contract::creates_parent_directories(&FsTools::new(&dir.0).unwrap());
    }

    #[test]
    fn write_contract_rejects_escapes() {
        let dir = TempDir::new("contract-escape");
        let inner = dir.0.join("inner");
        fs::create_dir_all(&inner).unwrap();
        contract::rejects_escapes(&FsTools::new(&inner).unwrap());
        assert!(!dir.0.join("escaped.txt").exists());
    }

    #[test]
    fn write_contract_rejects_contents_over_default_limit() {
        let dir = TempDir::new("contract-limit");
        contract::rejects_contents_over_default_limit(&FsTools::new(&dir.0).unwrap());
    }

    #[test]
    fn writes_land_on_disk_without_leaving_temp_files() {
        let dir = TempDir::new("atomic");
        let tools = FsTools::new(&dir.0).unwrap();
        write(&tools, "a.txt", "old").unwrap();
        write(&tools, "a.txt", "new").unwrap();
        assert_eq!(fs::read_to_string(dir.0.join("a.txt")).unwrap(), "new");
        let names: Vec<_> = fs::read_dir(&dir.0)
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        assert_eq!(names, ["a.txt"]);
    }

    #[test]
    fn refuses_writes_over_configured_limit() {
        let dir = TempDir::new("write-limit");
        let tools = FsTools::new(&dir.0).unwrap().write_limit(4);
        assert!(matches!(
            write(&tools, "big.txt", "0123456789"),
            Err(ToolError::ContentsTooLarge {
                size: 10,
                limit: 4,
                ..
            })
        ));
        assert!(!dir.0.join("big.txt").exists());
    }

    #[test]
    fn refuses_to_replace_a_directory() {
        let dir = TempDir::new("is-dir");
        fs::create_dir_all(dir.0.join("sub")).unwrap();
        let tools = FsTools::new(&dir.0).unwrap();
        assert!(matches!(
            write(&tools, "sub", "x"),
            Err(ToolError::IsDirectory { .. })
        ));
        assert!(matches!(
            write(&tools, ".", "x"),
            Err(ToolError::IsDirectory { .. })
        ));
    }

    #[test]
    fn refuses_absolute_paths_outside_root() {
        let dir = TempDir::new("absolute");
        let inner = dir.0.join("inner");
        fs::create_dir_all(&inner).unwrap();
        let tools = FsTools::new(&inner).unwrap();
        let outside = dir.0.join("abs.txt");
        assert!(matches!(
            write(&tools, outside.to_str().unwrap(), "x"),
            Err(ToolError::OutsideRoot { .. })
        ));
        assert!(!outside.exists());
    }

    #[cfg(unix)]
    #[test]
    fn refuses_writes_through_symlinks_that_leave_root() {
        let dir = TempDir::new("symlink");
        let inner = dir.0.join("inner");
        let outside = dir.0.join("outside");
        fs::create_dir_all(&inner).unwrap();
        fs::create_dir_all(&outside).unwrap();
        std::os::unix::fs::symlink(&outside, inner.join("link")).unwrap();
        let tools = FsTools::new(&inner).unwrap();
        assert!(matches!(
            write(&tools, "link/new/x.txt", "x"),
            Err(ToolError::OutsideRoot { .. })
        ));
        assert!(!outside.join("new").exists());
    }

    #[cfg(unix)]
    #[test]
    fn overwrite_keeps_file_permissions() {
        use std::os::unix::fs::PermissionsExt as _;

        let dir = TempDir::new("perms");
        let path = dir.0.join("run.sh");
        fs::write(&path, "old").unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o750)).unwrap();
        let tools = FsTools::new(&dir.0).unwrap();
        write(&tools, "run.sh", "new").unwrap();
        let mode = fs::metadata(&path).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o750);
    }
}
