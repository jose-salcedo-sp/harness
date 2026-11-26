//! Tools the model can call, and the [`ToolExecutor`] port that runs them.
//!
//! A model asks for a tool with a [`ToolCall`]: a name plus raw JSON
//! arguments, kept verbatim so they can be echoed back to the model. Before
//! anything runs, [`ToolRequest::parse`] turns that call into a typed request.

#[cfg(test)]
mod contract;
mod fs;
mod memory;

use std::{
    fmt,
    path::{Path, PathBuf},
};

use futures::future::BoxFuture;
use serde::Deserialize;

pub use fs::{DEFAULT_READ_LIMIT, DEFAULT_WRITE_LIMIT, FsTools, RootUnavailable};
pub use memory::MemoryTools;

/// Identifier the model assigned to one tool call.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ToolCallId(String);

/// Name of a tool, as the model refers to it.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ToolName(String);

/// Raw JSON arguments the model sent with a tool call.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ToolArguments(String);

/// Returned when a tool identifier or name is empty.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("tool identifier is empty")]
pub struct EmptyToolIdentifier;

macro_rules! non_empty_string {
    ($ty:ident) => {
        impl $ty {
            /// Wraps a non-empty string.
            ///
            /// # Errors
            ///
            /// Returns [`EmptyToolIdentifier`] when `value` is empty.
            pub fn new(value: impl Into<String>) -> Result<Self, EmptyToolIdentifier> {
                let value = value.into();
                if value.is_empty() {
                    Err(EmptyToolIdentifier)
                } else {
                    Ok(Self(value))
                }
            }

            /// The underlying string.
            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl fmt::Display for $ty {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
            }
        }
    };
}

non_empty_string!(ToolCallId);
non_empty_string!(ToolName);

impl ToolArguments {
    /// Wraps the argument string exactly as the model sent it.
    pub fn new(raw: impl Into<String>) -> Self {
        Self(raw.into())
    }

    /// The raw JSON text.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// A tool invocation requested by the model.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ToolCall {
    /// Identifier used to match the result to this call.
    pub id: ToolCallId,
    /// Which tool to run.
    pub name: ToolName,
    /// Arguments, not yet validated.
    pub arguments: ToolArguments,
}

/// A validated tool request, ready to execute.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ToolRequest {
    /// Read a UTF-8 text file, relative to the tool root.
    ReadFile {
        /// Path as the model wrote it.
        path: PathBuf,
    },
    /// Replace a UTF-8 text file, relative to the tool root, with `contents`.
    ///
    /// Always a whole-file rewrite, never a patch, so a stale view of the file
    /// cannot produce a half-applied edit.
    WriteFile {
        /// Path as the model wrote it.
        path: PathBuf,
        /// The complete new file contents.
        contents: String,
    },
}

const READ_FILE: &str = "read_file";
const WRITE_FILE: &str = "write_file";

#[derive(Deserialize)]
struct ReadFileArgs {
    path: String,
}

#[derive(Deserialize)]
struct WriteFileArgs {
    path: String,
    contents: String,
}

fn parse_path(raw: String) -> Result<PathBuf, ToolError> {
    if raw.trim().is_empty() {
        Err(ToolError::EmptyPath)
    } else {
        Ok(PathBuf::from(raw))
    }
}

impl ToolRequest {
    /// Validates a model's tool call.
    ///
    /// # Errors
    ///
    /// - [`ToolError::UnknownTool`] when no tool has the call's name.
    /// - [`ToolError::InvalidArguments`] when the arguments are not the JSON
    ///   object the tool expects.
    /// - [`ToolError::EmptyPath`] when `read_file` or `write_file` gets an
    ///   empty path.
    pub fn parse(call: &ToolCall) -> Result<Self, ToolError> {
        match call.name.as_str() {
            READ_FILE => {
                let args: ReadFileArgs = serde_json::from_str(call.arguments.as_str())
                    .map_err(ToolError::InvalidArguments)?;
                Ok(Self::ReadFile {
                    path: parse_path(args.path)?,
                })
            }
            WRITE_FILE => {
                let args: WriteFileArgs = serde_json::from_str(call.arguments.as_str())
                    .map_err(ToolError::InvalidArguments)?;
                Ok(Self::WriteFile {
                    path: parse_path(args.path)?,
                    contents: args.contents,
                })
            }
            _ => Err(ToolError::UnknownTool {
                name: call.name.clone(),
            }),
        }
    }
}

/// Describes a tool to the model: its name, purpose, and argument schema.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolSpec {
    /// The name the model calls the tool by.
    pub name: ToolName,
    /// What the tool does, written for the model.
    pub description: String,
    /// JSON Schema of the arguments object.
    pub parameters: serde_json::Value,
}

impl ToolSpec {
    /// Every tool the agent loop offers.
    pub fn builtin() -> Vec<Self> {
        vec![Self::read_file(), Self::write_file()]
    }

    /// The `read_file` tool.
    pub fn read_file() -> Self {
        Self {
            name: ToolName(READ_FILE.to_owned()),
            description: "Read a UTF-8 text file from the workspace and return its contents."
                .to_owned(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to the workspace root."
                    }
                },
                "required": ["path"]
            }),
        }
    }

    /// The `write_file` tool.
    pub fn write_file() -> Self {
        Self {
            name: ToolName(WRITE_FILE.to_owned()),
            description: "Create or overwrite a UTF-8 text file in the workspace. \
                 `contents` replaces the whole file: always send the complete new file, \
                 never a diff, excerpt, or placeholder such as `...`, because anything \
                 left out is deleted. To change an existing file, read it first. \
                 Missing parent directories are created."
                .to_owned(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to the workspace root."
                    },
                    "contents": {
                        "type": "string",
                        "description": "The complete new contents of the file."
                    }
                },
                "required": ["path", "contents"]
            }),
        }
    }
}

/// What a successful tool run returned.
///
/// The text is what the model reads. Tools that change files also carry a
/// structured report so the UI can describe the change without parsing text.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ToolOutput {
    text: String,
    write: Option<FileWrite>,
}

impl ToolOutput {
    /// Wraps tool output text.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            write: None,
        }
    }

    /// Reports a completed `write_file`; the text is the report's
    /// [`Display`](fmt::Display) form.
    pub fn written(write: FileWrite) -> Self {
        Self {
            text: write.to_string(),
            write: Some(write),
        }
    }

    /// The output text.
    pub fn as_str(&self) -> &str {
        &self.text
    }

    /// The file change, when the tool wrote a file.
    pub fn file_write(&self) -> Option<&FileWrite> {
        self.write.as_ref()
    }
}

/// Whether `write_file` made a new file or replaced an existing one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WriteKind {
    /// The file did not exist before.
    Created,
    /// An existing file was replaced.
    Overwritten,
}

/// What one `write_file` call did.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FileWrite {
    /// Path as the model wrote it.
    pub path: PathBuf,
    /// Whether the file was created or overwritten.
    pub kind: WriteKind,
    /// Size of the new contents in bytes.
    pub bytes: u64,
    /// Number of lines in the new contents.
    pub lines: usize,
}

impl FileWrite {
    /// Describes writing `contents` to `path`.
    pub fn new(path: PathBuf, kind: WriteKind, contents: &str) -> Self {
        Self {
            path,
            kind,
            bytes: byte_len(contents),
            lines: contents.lines().count(),
        }
    }
}

impl fmt::Display for FileWrite {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let verb = match self.kind {
            WriteKind::Created => "created",
            WriteKind::Overwritten => "overwrote",
        };
        let line_word = if self.lines == 1 { "line" } else { "lines" };
        write!(
            f,
            "{verb} `{}`: {} bytes, {} {line_word}",
            self.path.display(),
            self.bytes,
            self.lines
        )
    }
}

fn byte_len(contents: &str) -> u64 {
    u64::try_from(contents.len()).unwrap_or(u64::MAX)
}

/// Rejects `contents` larger than `limit` bytes for a write to `path`.
fn check_write_size(path: &Path, contents: &str, limit: u64) -> Result<(), ToolError> {
    let size = byte_len(contents);
    if size > limit {
        Err(ToolError::ContentsTooLarge {
            path: path.to_path_buf(),
            size,
            limit,
        })
    } else {
        Ok(())
    }
}

/// Why a tool call failed.
#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    /// The model asked for a tool that does not exist.
    #[error("unknown tool `{name}`")]
    UnknownTool {
        /// The requested name.
        name: ToolName,
    },
    /// The arguments were not valid JSON for this tool.
    #[error("invalid tool arguments")]
    InvalidArguments(#[source] serde_json::Error),
    /// `read_file` was called without a path.
    #[error("path is empty")]
    EmptyPath,
    /// The path resolves outside the directory the tools may read.
    #[error("path `{}` is outside the workspace", path.display())]
    OutsideRoot {
        /// The requested path.
        path: PathBuf,
    },
    /// The file is larger than the tool will return.
    #[error("file `{}` is {size} bytes, over the {limit} byte limit", path.display())]
    TooLarge {
        /// The requested path.
        path: PathBuf,
        /// Actual size in bytes.
        size: u64,
        /// Largest size the tool returns.
        limit: u64,
    },
    /// Reading the file failed.
    #[error("could not read `{}`", path.display())]
    Read {
        /// The requested path.
        path: PathBuf,
        /// The underlying IO error.
        #[source]
        source: std::io::Error,
    },
    /// `write_file` was given more contents than the tool will write.
    #[error("contents for `{}` are {size} bytes, over the {limit} byte write limit", path.display())]
    ContentsTooLarge {
        /// The requested path.
        path: PathBuf,
        /// Size of the contents in bytes.
        size: u64,
        /// Largest size the tool writes.
        limit: u64,
    },
    /// `write_file` targeted a directory.
    #[error("`{}` is a directory", path.display())]
    IsDirectory {
        /// The requested path.
        path: PathBuf,
    },
    /// Writing the file, or creating its parent directories, failed.
    #[error("could not write `{}`", path.display())]
    Write {
        /// The requested path.
        path: PathBuf,
        /// The underlying IO error.
        #[source]
        source: std::io::Error,
    },
}

impl ToolError {
    /// Renders the error and its causes as one line for the model to read.
    pub fn to_model_text(&self) -> String {
        let mut text = format!("error: {self}");
        let mut source = std::error::Error::source(self);
        while let Some(cause) = source {
            text.push_str(": ");
            text.push_str(&cause.to_string());
            source = cause.source();
        }
        text
    }
}

/// Port: runs validated tool requests.
///
/// The returned future is `Send + 'static` so the session can run on any
/// background executor.
pub trait ToolExecutor: Send + Sync {
    /// Runs one request.
    ///
    /// The future resolves to a [`ToolError`] when the request cannot be
    /// carried out; the session reports that error back to the model.
    fn execute(&self, request: ToolRequest) -> BoxFuture<'static, Result<ToolOutput, ToolError>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(name: &str, args: &str) -> ToolCall {
        ToolCall {
            id: ToolCallId::new("call_1").unwrap(),
            name: ToolName::new(name).unwrap(),
            arguments: ToolArguments::new(args),
        }
    }

    #[test]
    fn parses_read_file_call() {
        let request = ToolRequest::parse(&call("read_file", r#"{"path":"src/lib.rs"}"#)).unwrap();
        assert_eq!(
            request,
            ToolRequest::ReadFile {
                path: PathBuf::from("src/lib.rs")
            }
        );
    }

    #[test]
    fn rejects_unknown_tool() {
        let err = ToolRequest::parse(&call("rm_rf", "{}")).unwrap_err();
        assert!(matches!(err, ToolError::UnknownTool { name } if name.as_str() == "rm_rf"));
    }

    #[test]
    fn rejects_malformed_arguments() {
        let err = ToolRequest::parse(&call("read_file", "{not json")).unwrap_err();
        assert!(matches!(err, ToolError::InvalidArguments(_)));
    }

    #[test]
    fn rejects_blank_path() {
        let err = ToolRequest::parse(&call("read_file", r#"{"path":"  "}"#)).unwrap_err();
        assert!(matches!(err, ToolError::EmptyPath));
    }

    #[test]
    fn parses_write_file_call() {
        let request = ToolRequest::parse(&call(
            "write_file",
            r#"{"path":"out/a.txt","contents":"one\ntwo\n"}"#,
        ))
        .unwrap();
        assert_eq!(
            request,
            ToolRequest::WriteFile {
                path: PathBuf::from("out/a.txt"),
                contents: "one\ntwo\n".to_owned(),
            }
        );
    }

    #[test]
    fn write_file_requires_contents() {
        let err = ToolRequest::parse(&call("write_file", r#"{"path":"a.txt"}"#)).unwrap_err();
        assert!(matches!(err, ToolError::InvalidArguments(_)));
    }

    #[test]
    fn write_file_rejects_blank_path() {
        let err =
            ToolRequest::parse(&call("write_file", r#"{"path":"","contents":"x"}"#)).unwrap_err();
        assert!(matches!(err, ToolError::EmptyPath));
    }

    #[test]
    fn builtin_tools_include_write_file_requiring_full_contents() {
        let spec = ToolSpec::builtin()
            .into_iter()
            .find(|spec| spec.name.as_str() == "write_file")
            .unwrap();
        assert_eq!(
            spec.parameters["required"],
            serde_json::json!(["path", "contents"])
        );
        assert!(spec.description.contains("complete new file"));
    }

    #[test]
    fn written_output_text_reports_kind_size_and_lines() {
        let output = ToolOutput::written(FileWrite::new(
            PathBuf::from("a.txt"),
            WriteKind::Created,
            "one\ntwo\n",
        ));
        assert_eq!(output.as_str(), "created `a.txt`: 8 bytes, 2 lines");
        assert_eq!(output.file_write().unwrap().kind, WriteKind::Created);
    }

    #[test]
    fn rejects_empty_tool_name() {
        assert_eq!(ToolName::new(""), Err(EmptyToolIdentifier));
    }

    #[test]
    fn model_text_includes_error_chain() {
        let err = ToolError::Read {
            path: PathBuf::from("a.txt"),
            source: std::io::Error::new(std::io::ErrorKind::NotFound, "no such file"),
        };
        assert_eq!(
            err.to_model_text(),
            "error: could not read `a.txt`: no such file"
        );
    }
}
