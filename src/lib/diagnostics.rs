//! User-facing diagnostics: errors resolved to source positions.

use std::fmt;
use std::ops::Range;
use std::path::PathBuf;

/// A user-visible error with its source position resolved, where known.
///
/// Produced from a `parser::Error` at the evaluator boundary. Only errors
/// from `open`ed modules carry a file (relative to the library root);
/// errors in the program being evaluated render as a bare `line:col`,
/// matching the editor's own display.
pub struct Diagnostic {
    /// The module file the error occurred in, relative to the library
    /// root. `None` for errors local to the program or its source file.
    pub file: Option<PathBuf>,
    /// 1-based (line, column), when known. Relative to the program's own
    /// text for program errors, to the module's text for module errors.
    pub position: Option<(usize, usize)>,
    /// The error's byte range in the program slot's text, for editor
    /// highlighting. `None` when the error originated elsewhere.
    pub program_range: Option<Range<usize>>,
    pub message: String,
}

impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (&self.file, &self.position) {
            (Some(file), Some((line, col))) => {
                write!(f, "{}:{}:{}: {}", file.display(), line, col, self.message)
            }
            (Some(file), None) => write!(f, "{}: {}", file.display(), self.message),
            (None, Some((line, col))) => write!(f, "{}:{}: {}", line, col, self.message),
            (None, None) => f.write_str(&self.message),
        }
    }
}
