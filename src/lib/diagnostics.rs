//! User-facing diagnostics: errors resolved to source positions.

use std::fmt;
use std::ops::Range;
use std::path::PathBuf;

use crate::expr;

/// Identifies which of the app's texts a span's byte range indexes.
///
/// Passed to the parser when parsing each kind of text, and matched by
/// `Evaluator::diagnose` to resolve an error's range against that same
/// text.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Source {
    /// A program slot's own text, as shown in the editor.
    Program,
    /// The backing source file the program set was loaded from.
    File,
    /// The module at this index in the evaluator's module table.
    Module(u32),
}

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
    /// A multi-line source snippet locating the error (see
    /// [`render_snippet`]), when the source text was available at
    /// resolution time. Single-line display sites (e.g. the status line)
    /// should ignore it.
    pub snippet: Option<String>,
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
            snippet: None,
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
            snippet: Some(render_snippet(text, &range)),
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

/// Renders a rustc-style snippet locating `range` in `source`: the line
/// containing the range's start, with a caret underline beneath the range's
/// portion of that line.
///
/// The underline is clamped to the first line of the range and is always at
/// least one caret wide (so empty ranges and end-of-input positions still point
/// somewhere).
///
/// # Example
/// ```
/// let snippet = tuun::diagnostics::render_snippet("a = 1;\nb = nope;", &(11..15));
/// assert_eq!(snippet, "  |\n2 | b = nope;\n  |     ^^^^");
/// ```
pub fn render_snippet(source: &str, range: &Range<usize>) -> String {
    let start = range.start.min(source.len());
    let (line, col) = expr::line_col(source, start);
    // Line boundaries sit after `\n` bytes, so byte scanning stays on
    // char boundaries even if `start` itself is not on one.
    let line_start = source.as_bytes()[..start]
        .iter()
        .rposition(|&b| b == b'\n')
        .map(|p| p + 1)
        .unwrap_or(0);
    let line_end = source.as_bytes()[start..]
        .iter()
        .position(|&b| b == b'\n')
        .map(|p| start + p)
        .unwrap_or(source.len());
    let underline_end = range.end.clamp(start, line_end);
    let underline = source
        .get(start..underline_end)
        .map(|s| s.chars().count())
        .unwrap_or(underline_end - start)
        .max(1);
    let gutter = line.to_string();
    format!(
        "{blank:pad$} |\n{gutter} | {text}\n{blank:pad$} | {space}{carets}",
        blank = "",
        pad = gutter.len(),
        gutter = gutter,
        text = &source[line_start..line_end],
        space = " ".repeat(col - 1),
        carets = "^".repeat(underline),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_snippet() {
        // A range within one line underlines exactly that range.
        assert_eq!(
            render_snippet("a = 1;\nb = nope;", &(11..15)),
            "  |\n2 | b = nope;\n  |     ^^^^"
        );

        // A multi-line range is clamped to the first line.
        assert_eq!(
            render_snippet("first\nsecond", &(2..9)),
            "  |\n1 | first\n  |   ^^^"
        );

        // An empty range still gets one caret.
        assert_eq!(
            render_snippet("x = ;", &(4..4)),
            "  |\n1 | x = ;\n  |     ^"
        );

        // A range at/past end of input points one past the last line.
        assert_eq!(render_snippet("ab", &(5..9)), "  |\n1 | ab\n  |   ^");

        // Columns count chars, not bytes: 'é' is 2 bytes, so the '1' at
        // byte 5 is the 5th char and the caret sits 4 chars in.
        assert_eq!(
            render_snippet("é = 1", &(5..6)),
            "  |\n1 | é = 1\n  |     ^"
        );
    }
}
