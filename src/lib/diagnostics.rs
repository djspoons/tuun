//! User-facing diagnostics: errors resolved to source positions.

use std::fmt;
use std::ops::Range;
use std::path::PathBuf;

use crate::expr;

/// A user-visible error with its source position resolved, where known.
///
/// Produced from an `expr::Error` at the evaluator boundary. Only errors
/// from `open`ed modules carry a file (relative to the library root);
/// errors in the program being evaluated render as a bare `line:col`,
/// matching the editor's own display.
#[derive(Clone, Debug, PartialEq)]
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

impl Diagnostic {
    /// Builds a diagnostic that carries only a message, with no position
    /// information.
    pub fn message_only(message: String) -> Diagnostic {
        Diagnostic {
            file: None,
            position: None,
            program_range: None,
            message,
        }
    }

    /// Builds a diagnostic for an error at `range` of a program's own
    /// `text`: a bare `line:col` position matching the editor's display,
    /// with the range kept for editor highlighting.
    pub fn in_program(message: String, range: Range<usize>, text: &str) -> Diagnostic {
        Diagnostic {
            file: None,
            position: Some(expr::line_col(text, range.start)),
            program_range: Some(range),
            message,
        }
    }
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
