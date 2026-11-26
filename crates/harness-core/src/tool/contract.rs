//! Behavior every [`ToolExecutor`] must share, run against each adapter.

use std::path::PathBuf;

use futures::executor::block_on;

use super::{DEFAULT_WRITE_LIMIT, ToolError, ToolExecutor, ToolOutput, ToolRequest, WriteKind};

fn read(tools: &impl ToolExecutor, path: &str) -> Result<ToolOutput, ToolError> {
    block_on(tools.execute(ToolRequest::ReadFile {
        path: PathBuf::from(path),
    }))
}

fn write(tools: &impl ToolExecutor, path: &str, contents: &str) -> Result<ToolOutput, ToolError> {
    block_on(tools.execute(ToolRequest::WriteFile {
        path: PathBuf::from(path),
        contents: contents.to_owned(),
    }))
}

pub(crate) fn creates_then_overwrites(tools: &impl ToolExecutor) {
    let created = write(tools, "notes.txt", "one\ntwo\n").unwrap();
    let report = created.file_write().unwrap();
    assert_eq!(report.kind, WriteKind::Created);
    assert_eq!((report.bytes, report.lines), (8, 2));

    let overwritten = write(tools, "notes.txt", "three").unwrap();
    let report = overwritten.file_write().unwrap();
    assert_eq!(report.kind, WriteKind::Overwritten);
    assert_eq!((report.bytes, report.lines), (5, 1));
    assert_eq!(
        overwritten.as_str(),
        "overwrote `notes.txt`: 5 bytes, 1 line"
    );
    assert_eq!(read(tools, "notes.txt").unwrap().as_str(), "three");
}

pub(crate) fn creates_parent_directories(tools: &impl ToolExecutor) {
    write(tools, "deep/er/file.rs", "fn main() {}\n").unwrap();
    assert_eq!(
        read(tools, "deep/er/file.rs").unwrap().as_str(),
        "fn main() {}\n"
    );
}

pub(crate) fn rejects_escapes(tools: &impl ToolExecutor) {
    for path in ["../escaped.txt", "a/../../escaped.txt", "/tmp/escaped.txt"] {
        assert!(
            matches!(write(tools, path, "x"), Err(ToolError::OutsideRoot { .. })),
            "{path} was not rejected"
        );
    }
}

pub(crate) fn rejects_contents_over_default_limit(tools: &impl ToolExecutor) {
    let size = DEFAULT_WRITE_LIMIT + 1;
    let contents = "x".repeat(usize::try_from(size).unwrap());
    assert!(matches!(
        write(tools, "big.txt", &contents),
        Err(ToolError::ContentsTooLarge { size: s, limit: DEFAULT_WRITE_LIMIT, .. }) if s == size
    ));
    assert!(matches!(
        read(tools, "big.txt"),
        Err(ToolError::Read { .. })
    ));
}
