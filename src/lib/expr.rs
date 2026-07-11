//! The tuun expression language: the AST data structures, their construction
//! helpers, and printing, both the canonical `Display` form and the
//! source-preserving printer.

use std::fmt;
use std::fmt::{Debug, Display};
use std::ops::Range;
use std::rc::Rc;

use crate::waveform;

/// Identifies which text a span's byte range indexes.
///
/// Assigned by whoever parsed the text: the parser stamps everything
/// `Local`, and callers that know a file identity restamp with
/// [`set_span_source`] (the program set stamps its bindings `File`; the
/// evaluator stamps each loaded module `Module(id)`).
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum SourceId {
    /// The text handed to the parser, as-is.
    Local,
    /// The backing source file the program set was loaded from.
    File,
    /// The module at this index in the evaluator's module table.
    Module(u32),
}

/// A byte range plus the identity of the text it indexes.
#[derive(Clone, Debug, PartialEq)]
pub struct Span {
    pub source: SourceId,
    pub range: Range<usize>,
}

impl Span {
    /// Builds a span into the locally-parsed text.
    pub fn local(range: Range<usize>) -> Span {
        Span {
            source: SourceId::Local,
            range,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Error {
    span: Option<Span>,
    message: String,
}

impl Error {
    pub fn new(message: String) -> Self {
        Self {
            span: None,
            message,
        }
    }
    /// Construct an `Error` located at the given span. Use when a non-parser
    /// caller (e.g. the evaluator) wants to attach a source location to an
    /// error it didn't itself produce.
    pub fn with_span(message: String, span: Option<Span>) -> Self {
        Self { span, message }
    }

    /// The error's byte range, without its source identity.
    pub fn range(&self) -> Option<Range<usize>> {
        self.span.as_ref().map(|span| span.range.clone())
    }

    /// The identity of the text this error's range indexes.
    pub fn source(&self) -> Option<SourceId> {
        self.span.as_ref().map(|span| span.source)
    }

    /// The error message, without any location rendering.
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Renders the error as `line:col: message` against `source`, the text
    /// this error's range indexes; falls back to the bare message when the
    /// error has no range.
    pub fn display_with_source(&self, source: &str) -> String {
        match &self.span {
            Some(span) => {
                let (line, col) = line_col(source, span.range.start);
                format!("{}:{}: {}", line, col, self.message)
            }
            None => self.message.clone(),
        }
    }
}

/// Returns the 1-based (line, column) of byte `offset` in `source`.
///
/// The column counts characters, not bytes. Offsets past the end of `source`
/// are clamped to the end.
///
/// # Example
/// ```
/// // Offset points to the `d` on the second line.
/// let result = tuun::expr::line_col("ab\ncd", 4);
/// assert_eq!(result, (2, 2));
/// ```
pub fn line_col(source: &str, offset: usize) -> (usize, usize) {
    let offset = offset.min(source.len());
    let prefix = &source.as_bytes()[..offset];
    let line = prefix.iter().filter(|&&b| b == b'\n').count() + 1;
    let line_start = prefix
        .iter()
        .rposition(|&b| b == b'\n')
        .map(|p| p + 1)
        .unwrap_or(0);
    // A caller-supplied offset may not land on a char boundary; fall back to
    // a byte count rather than panicking on the slice.
    let col = source
        .get(line_start..offset)
        .map(|s| s.chars().count())
        .unwrap_or(offset - line_start)
        + 1;
    (line, col)
}

impl Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.span {
            Some(span) => write!(
                f,
                "{} at {}..{}",
                self.message, span.range.start, span.range.end
            ),
            None => f.write_str(&self.message),
        }
    }
}

#[derive(Clone)]
pub struct BuiltInFn<M>(pub Rc<dyn Fn(Vec<Expr<M>>) -> Expr<M>>);

impl<M> Debug for BuiltInFn<M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BuiltInFn(...)")
    }
}

#[derive(Clone, Debug)]
pub enum Pattern {
    Identifier(String),
    Tuple(Vec<Pattern>),
}

#[derive(Clone, Debug)]
pub enum Expr<M> {
    // Values
    Bool(bool),
    Float(f32),
    String(String),
    Waveform(waveform::Waveform<M>),
    Function {
        pattern: Pattern,
        body: Box<SourceExpr<M>>,
    },
    BuiltIn {
        name: String,
        // Pure functions from a vector of values to a value
        function: BuiltInFn<M>,
    },
    // A sequence-able waveform. In value form, both offset and waveform are Expr::Waveform.
    Seq {
        offset: Box<SourceExpr<M>>,
        waveform: Box<SourceExpr<M>>,
    },
    // If-Then-Else expression
    IfThenElse {
        condition: Box<SourceExpr<M>>,
        then: Box<SourceExpr<M>>,
        else_: Box<SourceExpr<M>>,
    },
    // Function application
    Variable(String),
    Application {
        function: Box<SourceExpr<M>>,
        argument: Box<SourceExpr<M>>,
    },
    // Compound expressions
    Tuple(Vec<SourceExpr<M>>),
    List(Vec<SourceExpr<M>>),
    // Errors
    Error(String),
}

// TODO use this in 'expect'?
impl<M> Default for Expr<M> {
    fn default() -> Self {
        Expr::Error("_".to_string())
    }
}

#[derive(Clone, Debug)]
pub struct SourceExpr<M> {
    pub expr: Expr<M>,
    pub span: Option<Span>, // Some(_) for parser output; None for synthesized
}

impl<M> From<Expr<M>> for SourceExpr<M> {
    fn from(expr: Expr<M>) -> Self {
        SourceExpr { expr, span: None }
    }
}

impl<M> Default for SourceExpr<M> {
    fn default() -> Self {
        SourceExpr::from(Expr::default())
    }
}

/// Boxes a bare `Expr<M>` into a `Box<SourceExpr<M>>` with no span.
///
/// Useful for synthesized compound expressions in builtins and the evaluator.
pub fn boxed<M>(expr: Expr<M>) -> Box<SourceExpr<M>> {
    Box::new(SourceExpr::from(expr))
}

impl<M> SourceExpr<M> {
    /// Builds an expression spanning `range` of the locally-parsed text
    /// (the parser's stamp; see [`SourceId::Local`]).
    pub fn with_span(expr: Expr<M>, range: Range<usize>) -> SourceExpr<M> {
        SourceExpr {
            expr,
            span: Some(Span::local(range)),
        }
    }
    pub fn bool(v: bool) -> SourceExpr<M> {
        SourceExpr::from(Expr::Bool(v))
    }
    pub fn float(v: f32) -> SourceExpr<M> {
        SourceExpr::from(Expr::Float(v))
    }
    pub fn string(v: String) -> SourceExpr<M> {
        SourceExpr::from(Expr::String(v))
    }
    pub fn variable(name: String) -> SourceExpr<M> {
        SourceExpr::from(Expr::Variable(name))
    }
    pub fn error(msg: String) -> SourceExpr<M> {
        SourceExpr::from(Expr::Error(msg))
    }
    pub fn function(pattern: Pattern, body: SourceExpr<M>) -> SourceExpr<M> {
        SourceExpr::from(Expr::Function {
            pattern,
            body: Box::new(body),
        })
    }
    pub fn application(function: SourceExpr<M>, argument: SourceExpr<M>) -> SourceExpr<M> {
        SourceExpr::from(Expr::Application {
            function: Box::new(function),
            argument: Box::new(argument),
        })
    }
    pub fn tuple(exprs: Vec<SourceExpr<M>>) -> SourceExpr<M> {
        SourceExpr::from(Expr::Tuple(exprs))
    }
    pub fn list(exprs: Vec<SourceExpr<M>>) -> SourceExpr<M> {
        SourceExpr::from(Expr::List(exprs))
    }
    pub fn if_then_else(
        condition: SourceExpr<M>,
        then: SourceExpr<M>,
        else_: SourceExpr<M>,
    ) -> SourceExpr<M> {
        SourceExpr::from(Expr::IfThenElse {
            condition: Box::new(condition),
            then: Box::new(then),
            else_: Box::new(else_),
        })
    }
}

/// Bindings modify the current scope of evaluation. They appear in `let`
/// expressions and in modules.
#[derive(Debug, Clone)]
pub enum Binding<M> {
    /// An import that binds all bindings from the module at `path` in the
    /// current scope. These bindings are not public in the current scope.
    // TODO make it just bind public ones
    Open(Vec<String>),
    /// Binds variables in `pattern` to the corresponding values in `expr`.
    Definition(Pattern, SourceExpr<M>),
    /// A placeholder that carries only trivia (e.g., trailing comments at end
    /// of file).
    Empty,
}

#[derive(Clone, Debug)]
/// Binding along with annotations and source information.
//
// Bindings in `let` expressions do *not* include the source of their separators
// (`,`); bindings that are part of a module do include the source of their
// terminating `;`.
// TODO should bindings in `let` expressions include the source of their
// separators?
pub struct SourceBinding<M> {
    pub binding: Binding<M>,
    /// Annotations attached to this binding.
    pub annotations: Vec<SourceAnnotation>,
    pub span: Option<Span>,
}

// TODO need to define and handle the case where the same annotation type occurs
// multiple times on a single definition

impl<M> From<Binding<M>> for SourceBinding<M> {
    fn from(binding: Binding<M>) -> Self {
        SourceBinding {
            binding,
            annotations: Vec::new(),
            span: None,
        }
    }
}

impl<M> SourceBinding<M> {
    pub fn definition(pattern: Pattern, expr: SourceExpr<M>) -> SourceBinding<M> {
        SourceBinding::from(Binding::Definition(pattern, expr))
    }

    /// Builds a binding spanning `range` of the locally-parsed text.
    pub fn with_span(binding: Binding<M>, range: Range<usize>) -> SourceBinding<M> {
        SourceBinding {
            binding,
            annotations: Vec::new(),
            span: Some(Span::local(range)),
        }
    }
}

/// An `Annotation` together with the byte range in source it was parsed from.
#[derive(Debug, Clone)]
pub struct SourceAnnotation {
    pub annotation: Annotation,
    pub span: Option<Span>,
}

impl From<Annotation> for SourceAnnotation {
    fn from(annotation: Annotation) -> Self {
        SourceAnnotation {
            annotation,
            span: None,
        }
    }
}

/// Rewrites the source identity of every span in `bindings` to `source`,
/// including annotations and nested expressions.
///
/// Freshly parsed bindings carry [`SourceId::Local`]; call this once the
/// text's true identity (a backing file, a loaded module) is known.
pub fn set_span_source<M>(bindings: &mut [SourceBinding<M>], source: SourceId) {
    for binding in bindings.iter_mut() {
        if let Some(span) = &mut binding.span {
            span.source = source;
        }
        for annotation in &mut binding.annotations {
            if let Some(span) = &mut annotation.span {
                span.source = source;
            }
        }
        if let Binding::Definition(_, expr) = &mut binding.binding {
            set_expr_span_source(expr, source);
        }
    }
}

/// Rewrites the source identity of every span in `expr`'s tree to `source`.
fn set_expr_span_source<M>(expr: &mut SourceExpr<M>, source: SourceId) {
    if let Some(span) = &mut expr.span {
        span.source = source;
    }
    match &mut expr.expr {
        Expr::Bool(_)
        | Expr::Float(_)
        | Expr::String(_)
        | Expr::Variable(_)
        | Expr::Waveform(_)
        | Expr::BuiltIn { .. }
        | Expr::Error(_) => {}
        Expr::Function { body, .. } => set_expr_span_source(body, source),
        Expr::Seq { offset, waveform } => {
            set_expr_span_source(offset, source);
            set_expr_span_source(waveform, source);
        }
        Expr::IfThenElse {
            condition,
            then,
            else_,
        } => {
            set_expr_span_source(condition, source);
            set_expr_span_source(then, source);
            set_expr_span_source(else_, source);
        }
        Expr::Application { function, argument } => {
            set_expr_span_source(function, source);
            set_expr_span_source(argument, source);
        }
        Expr::Tuple(exprs) | Expr::List(exprs) => {
            for expr in exprs {
                set_expr_span_source(expr, source);
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Slider {
    pub label: String,
    pub function: SliderFunction,
}

#[derive(Debug, Clone)]
pub enum SliderFunction {
    Linear {
        initial_value: f32, // In the output range
        min: f32,
        max: f32,
    },
    UserDefined {
        normalized_initial_value: f32, // In [0, 1]
        function_source: String,       // Tuun function expression, e.g. "fn(x) => 20 * pow(500, x)"
    },
}

#[derive(Debug, Clone)]
pub enum Annotation {
    Sliders(Vec<Slider>),
    Color(u8, u8, u8),
    Level(f32),
    /// Number of empty UI slots to leave before this `Definition`'s implicit
    /// slot. Omitting the annotation is equivalent to `skip_slots=0`.
    SkipSlots(u32),
}

/// Renders a parameter value (`level_db`, slider `initial`/`min`/`max`, slider
/// `normalized_initial_value`) as it should appear source annotations.
pub fn format_param(value: f32) -> String {
    // Default `{}` Display: shortest round-trip representation. Trims `5.0` →
    // `5` and avoids unrolling noise from float arithmetic.
    format!("{}", value)
}

impl Display for Slider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.function {
            SliderFunction::Linear {
                initial_value,
                min,
                max,
            } => write!(
                f,
                "\"{}:{}:{}:{}\"",
                self.label,
                format_param(*initial_value),
                format_param(*min),
                format_param(*max)
            ),
            SliderFunction::UserDefined {
                normalized_initial_value,
                function_source,
            } => write!(
                f,
                "\"{}:{}:{}\"",
                self.label,
                format_param(*normalized_initial_value),
                function_source
            ),
        }
    }
}

impl Display for Annotation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Annotation::Color(r, g, b) => write!(f, "color=rgb({},{},{})", r, g, b),
            Annotation::Level(v) => write!(f, "level_db={}", format_param(*v)),
            Annotation::SkipSlots(n) => write!(f, "skip_slots={}", n),
            Annotation::Sliders(sliders) => {
                write!(f, "sliders=[")?;
                for (i, s) in sliders.iter().enumerate() {
                    if i > 0 {
                        write!(f, ",")?;
                    }
                    write!(f, "{}", s)?;
                }
                write!(f, "]")
            }
        }
    }
}

// Operator precedence levels for pretty-printing. Higher = binds tighter. Must
// match the parser's precedence hierarchy so that `format!("{}", parse(s))`
// round-trips back to the same AST.
#[derive(Copy, Clone, PartialEq, PartialOrd)]
enum Precedence {
    Followed = 10,       // `\`
    ReverseApp = 20,     // `|`
    Relational = 30,     // == != < <= > >=
    Additive = 40,       // + - &
    Multiplicative = 50, // * / ~*
    Unary = 60,          // ! @ $ % - ?
    Application = 70,    // f(x)
    Atom = 80,           // literals, identifiers, ( ), [ ], { }
}

fn binary_op_precedence(op: &str) -> Option<Precedence> {
    match op {
        "*" | "/" | "~*" => Some(Precedence::Multiplicative),
        "+" | "-" | "&" => Some(Precedence::Additive),
        "==" | "!=" | "<" | "<=" | ">" | ">=" => Some(Precedence::Relational),
        "|" => Some(Precedence::ReverseApp),
        "\\" => Some(Precedence::Followed),
        _ => None,
    }
}

fn is_unary_op(op: &str) -> bool {
    matches!(op, "!" | "@" | "$" | "%" | "-" | "?")
}

/// Precedence of an expression's outermost form, used to decide whether a
/// child needs parens when nested inside an operator context.
fn expr_precedence<M>(expr: &Expr<M>) -> Precedence {
    match expr {
        Expr::Bool(_)
        | Expr::Float(_)
        | Expr::String(_)
        | Expr::Variable(_)
        | Expr::Waveform(_)
        | Expr::BuiltIn { .. }
        | Expr::Tuple(_)
        | Expr::List(_)
        | Expr::Error(_) => Precedence::Atom,
        Expr::Seq { .. } => Precedence::Application,
        Expr::Application { function, argument } => {
            if let Expr::Variable(op) = &function.expr {
                if let Expr::Tuple(args) = &argument.expr
                    && args.len() == 2
                    && let Some(p) = binary_op_precedence(op)
                {
                    return p;
                }
                if is_unary_op(op) && !matches!(argument.expr, Expr::Tuple(_)) {
                    return Precedence::Unary;
                }
            }
            // Function literal LHS → displayed as `let` (same precedence as Function).
            if matches!(function.expr, Expr::Function { .. }) {
                return Precedence::Followed;
            }
            // Application LHS → displayed as `|` pipe.
            if matches!(function.expr, Expr::Application { .. }) {
                return Precedence::ReverseApp;
            }
            Precedence::Application
        }
        Expr::Function { .. } | Expr::IfThenElse { .. } => Precedence::Followed,
    }
}

/// Emits `let p1 = e1, p2 = e2, … in body` for an
/// `Application { function: Function { pattern, body }, argument }` chain,
/// merging nested `let`s into a single comma-separated form.
fn fmt_as_let<M>(
    function: &SourceExpr<M>,
    argument: &SourceExpr<M>,
    f: &mut fmt::Formatter,
) -> fmt::Result
where
    M: fmt::Display,
{
    write!(f, "let ")?;
    let mut current_fn = function;
    let mut current_arg = argument;
    let mut first = true;
    loop {
        if let Expr::Function { pattern, body } = &current_fn.expr {
            if !first {
                write!(f, ", ")?;
            }
            first = false;
            write!(f, "{} = {}", pattern, current_arg)?;
            if let Expr::Application {
                function: next_fn,
                argument: next_arg,
            } = &body.expr
                && matches!(next_fn.expr, Expr::Function { .. })
            {
                current_fn = next_fn;
                current_arg = next_arg;
                continue;
            }
            return write!(f, " in {}", body);
        }
        unreachable!("fmt_as_let entered with non-Function function");
    }
}

/// Format `expr` in an operator context with minimum precedence `min_precedence`,
/// adding parens iff the child's outermost form binds looser than required.
fn fmt_at<M>(expr: &SourceExpr<M>, min_precedence: u8, f: &mut fmt::Formatter) -> fmt::Result
where
    M: Display,
{
    if (expr_precedence(&expr.expr) as u8) < min_precedence {
        write!(f, "({})", expr)
    } else {
        write!(f, "{}", expr)
    }
}

impl Display for Pattern {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Pattern::Identifier(name) => write!(f, "{}", name),
            Pattern::Tuple(patterns) => {
                write!(f, "(")?;
                for (i, pattern) in patterns.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", pattern)?;
                }
                write!(f, ")")
            }
        }
    }
}

impl<M> Display for SourceExpr<M>
where
    M: Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.expr.fmt(f)
    }
}

impl<M> Display for Expr<M>
where
    M: Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Expr::Bool(value) => write!(f, "{}", value),
            Expr::Float(value) => write!(f, "{}", value),
            Expr::String(value) => write!(f, "{}", value),
            Expr::Waveform(waveform) => {
                write!(f, "{}", waveform)
            }
            Expr::Function { pattern, body } => {
                write!(f, "fn")?;
                match pattern {
                    Pattern::Identifier(name) => write!(f, "({})", name)?,
                    Pattern::Tuple(_) => write!(f, "{}", pattern)?,
                }
                write!(f, " => {}", body)
            }
            Expr::BuiltIn { name, .. } => write!(f, "{}", name),
            Expr::Variable(name) => write!(f, "{}", name),
            Expr::IfThenElse {
                condition,
                then,
                else_,
            } => {
                write!(f, "if {} then {} else {}", condition, then, else_)
            }
            Expr::Application { function, argument } => {
                // Render operator applications and `{...}` / `<...>` sugar
                // using surface syntax matching the parser, so the Display
                // output round-trips.
                if let Expr::Variable(name) = &function.expr {
                    // `__chord` and `__sequence` aren't legal identifiers —
                    // they can only be entered via the `{x}` / `<x>` sugar.
                    // Emit the sugar so the output re-parses.
                    match name.as_str() {
                        "__chord" => return write!(f, "{{{}}}", argument),
                        "__sequence" => return write!(f, "<{}>", argument),
                        _ => {}
                    }
                    if let Expr::Tuple(args) = &argument.expr
                        && args.len() == 2
                        && let Some(p) = binary_op_precedence(name)
                    {
                        // Left-associative: lhs allows equal-precedence,
                        // rhs requires strictly tighter precedence.
                        fmt_at(&args[0], p as u8, f)?;
                        write!(f, " {} ", name)?;
                        fmt_at(&args[1], (p as u8) + 1, f)?;
                        return Ok(());
                    }
                    if is_unary_op(name) && !matches!(&argument.expr, Expr::Tuple(_)) {
                        write!(f, "{}", name)?;
                        return fmt_at(argument, Precedence::Unary as u8, f);
                    }
                }
                // Function literal LHS → `let p = arg in body` sugar.
                if matches!(function.expr, Expr::Function { .. }) {
                    return fmt_as_let(function, argument, f);
                }
                // Application LHS → `arg | function` pipe sugar.
                if matches!(function.expr, Expr::Application { .. }) {
                    fmt_at(argument, Precedence::ReverseApp as u8, f)?;
                    write!(f, " | ")?;
                    return fmt_at(function, (Precedence::ReverseApp as u8) + 1, f);
                }
                // Default: function-call form. The head is printed at
                // `Application` precedence — anything looser (Function,
                // IfThenElse, an operator app) gets wrapped.
                fmt_at(function, Precedence::Application as u8, f)?;
                if let Expr::Tuple(_) = &argument.expr {
                    write!(f, "{}", argument)
                } else {
                    write!(f, "({})", argument)
                }
            }
            Expr::Tuple(exprs) => {
                write!(f, "(")?;
                for (i, expr) in exprs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", expr)?;
                }
                write!(f, ")")
            }
            Expr::List(exprs) => {
                write!(f, "[")?;
                for (i, expr) in exprs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", expr)?;
                }
                write!(f, "]")
            }
            Expr::Seq { offset, waveform } => write!(f, "seq({}, {})", offset, waveform),
            Expr::Error(s) => write!(f, "{}", s),
        }
    }
}

// --- Source-preserving printer (Recast-style) ---
//
// `print_preserving(node, source)` reproduces the original source for any
// sub-tree whose every leaf still has a span (i.e. nothing inside has been
// edited since parsing), and falls back to a structural pretty-print for
// dirty regions — recursing into each child so that clean sub-sub-trees
// inside a dirty parent still come back verbatim from `source`.

/// Returns true iff `node` and every descendant either has Some span or is one
/// of the parser's "transparent packaging" forms (binary-op-argument Tuples).
fn is_clean<M>(node: &SourceExpr<M>) -> bool {
    match &node.expr {
        // Leaves: must have been stamped by the parser.
        Expr::Bool(_)
        | Expr::Float(_)
        | Expr::String(_)
        | Expr::Variable(_)
        | Expr::Waveform(_)
        | Expr::BuiltIn { .. }
        | Expr::Error(_) => node.span.is_some(),
        Expr::Function { body, .. } => node.span.is_some() && is_clean(body),
        Expr::Seq { offset, waveform } => {
            node.span.is_some() && is_clean(offset) && is_clean(waveform)
        }
        Expr::IfThenElse {
            condition,
            then,
            else_,
        } => node.span.is_some() && is_clean(condition) && is_clean(then) && is_clean(else_),
        Expr::Application { function, argument } => {
            node.span.is_some() && is_clean(function) && is_clean(argument)
        }
        // Tuples are sometimes synthesized by `fold_binary_op` as transparent
        // packaging around the two operands of a binary operator — those Tuples
        // have no span of their own but the surrounding Application does, and
        // splicing the Application's span gives the right text. So a Tuple's
        // own span isn't required; what matters is its children.
        Expr::Tuple(items) => items.iter().all(is_clean),
        // Lists in source always have brackets; we require a span.
        Expr::List(items) => node.span.is_some() && items.iter().all(is_clean),
    }
}

/// Returns a string that reproduces `node` using `source` as a verbatim
/// reference for unchanged sub-trees. For any subtree where every leaf still
/// has a span, splices `source[span]`. For dirty regions, falls back to
/// structural pretty-print, recursing into children so that clean sub-sub-trees
/// still splice their original source.
pub fn print_preserving<M>(node: &SourceExpr<M>, source: &str) -> String
where
    M: Display,
{
    let mut out = String::new();
    write_preserving(node, source, &mut out).expect("write to String");
    out
}

/// Round-trip bindings parsed from a module back to source.
///
/// Bindings that have been mutated in memory (cleared spans on the binding, its
/// expression, or an annotation) fall back to structural pretty-print.
pub fn print_preserving_module<M>(bindings: &[SourceBinding<M>], source: &str) -> String
where
    M: Display,
{
    use fmt::Write;
    let mut out = String::new();
    for binding in bindings {
        if let Some(span) = binding.clean_span() {
            out.push_str(&source[span]);
            continue;
        }
        // Structural fallback. Loses any whitespace/comments inside the
        // binding's span (we no longer know where in the source they sat), but
        // emits a syntactically valid `;`-terminated form so the surrounding
        // module still parses.
        for sa in &binding.annotations {
            writeln!(out, "#{{{}}}", sa.annotation).expect("write to String");
        }
        match &binding.binding {
            Binding::Definition(pattern, expr) => {
                write!(out, "{} = ", pattern).expect("write to String");
                write_preserving(expr, source, &mut out).expect("write to String");
                out.push_str(";\n");
            }
            Binding::Open(path) => {
                writeln!(out, "open {};", path.join(".")).expect("write to String");
            }
            // No body for an `Empty` binding — any annotations it carries
            // were emitted above.
            Binding::Empty => {}
        }
    }
    out
}

impl<M> SourceBinding<M> {
    /// Returns the source span to splice verbatim for `binding`, or `None` when
    /// something inside has been mutated since parsing and the binding needs
    /// structural re-emit.
    fn clean_span(&self) -> Option<Range<usize>> {
        if !self.annotations.iter().all(|a| a.span.is_some()) {
            return None;
        }
        let span = self.span.as_ref()?;
        match &self.binding {
            Binding::Definition(_, expr) if !is_clean(expr) => None,
            _ => Some(span.range.clone()),
        }
    }
}

fn write_preserving<M, W>(node: &SourceExpr<M>, source: &str, out: &mut W) -> fmt::Result
where
    M: Display,
    W: fmt::Write,
{
    if let Some(span) = &node.span
        && is_clean(node)
    {
        return out.write_str(&source[span.range.clone()]);
    }
    write_preserving_structural(&node.expr, source, out)
}

fn write_preserving_structural<M, W>(expr: &Expr<M>, source: &str, out: &mut W) -> fmt::Result
where
    M: Display,
    W: fmt::Write,
{
    match expr {
        Expr::Bool(v) => write!(out, "{}", v),
        Expr::Float(v) => write!(out, "{}", v),
        Expr::String(v) => write!(out, "{}", v),
        Expr::Waveform(w) => write!(out, "{}", w),
        Expr::BuiltIn { name, .. } => write!(out, "{}", name),
        Expr::Variable(name) => write!(out, "{}", name),
        Expr::Function { pattern, body } => {
            write!(out, "fn ")?;
            match pattern {
                Pattern::Identifier(name) => write!(out, "({})", name)?,
                Pattern::Tuple(_) => write!(out, "{}", pattern)?,
            }
            write!(out, " => ")?;
            write_preserving(body, source, out)
        }
        Expr::IfThenElse {
            condition,
            then,
            else_,
        } => {
            write!(out, "if ")?;
            write_preserving(condition, source, out)?;
            write!(out, " then ")?;
            write_preserving(then, source, out)?;
            write!(out, " else ")?;
            write_preserving(else_, source, out)
        }
        Expr::Application { function, argument } => {
            write_preserving_application(function, argument, source, out)
        }
        Expr::Tuple(exprs) => {
            write!(out, "(")?;
            write_preserving_elements(exprs, source, out)?;
            write!(out, ")")
        }
        Expr::List(exprs) => {
            write!(out, "[")?;
            write_preserving_elements(exprs, source, out)?;
            write!(out, "]")
        }
        Expr::Seq { offset, waveform } => {
            write!(out, "seq(")?;
            write_preserving(offset, source, out)?;
            write!(out, ", ")?;
            write_preserving(waveform, source, out)?;
            write!(out, ")")
        }
        Expr::Error(s) => write!(out, "{}", s),
    }
}

/// Emit a comma-separated element sequence. If both adjacent elements have
/// spans, splice `source[prev.end..curr.start]` to preserve any comments or
/// whitespace the user had between them; otherwise emit a plain `, `.
fn write_preserving_elements<M, W>(
    items: &[SourceExpr<M>],
    source: &str,
    out: &mut W,
) -> fmt::Result
where
    M: Display,
    W: fmt::Write,
{
    for (i, item) in items.iter().enumerate() {
        if i > 0 {
            let prev = items[i - 1].span.as_ref();
            let curr = item.span.as_ref();
            match (prev, curr) {
                (Some(p), Some(c))
                    if p.range.end <= c.range.start && c.range.start <= source.len() =>
                {
                    out.write_str(&source[p.range.end..c.range.start])?;
                }
                _ => out.write_str(", ")?,
            }
        }
        write_preserving(item, source, out)?;
    }
    Ok(())
}

fn write_preserving_application<M, W>(
    function: &SourceExpr<M>,
    argument: &SourceExpr<M>,
    source: &str,
    out: &mut W,
) -> fmt::Result
where
    M: Display,
    W: fmt::Write,
{
    if let Expr::Variable(name) = &function.expr {
        match name.as_str() {
            "_chord" => {
                out.write_str("{")?;
                write_preserving(argument, source, out)?;
                return out.write_str("}");
            }
            "_sequence" => {
                out.write_str("<")?;
                write_preserving(argument, source, out)?;
                return out.write_str(">");
            }
            _ => {}
        }
        if let Expr::Tuple(args) = &argument.expr
            && args.len() == 2
            && let Some(p) = binary_op_precedence(name)
        {
            write_preserving_at(&args[0], p as u8, source, out)?;
            write!(out, " {} ", name)?;
            return write_preserving_at(&args[1], (p as u8) + 1, source, out);
        }
        if is_unary_op(name) && !matches!(&argument.expr, Expr::Tuple(_)) {
            write!(out, "{}", name)?;
            return write_preserving_at(argument, Precedence::Unary as u8, source, out);
        }
    }
    if matches!(function.expr, Expr::Function { .. }) {
        return write_preserving_as_let(function, argument, source, out);
    }
    if matches!(function.expr, Expr::Application { .. }) {
        write_preserving_at(argument, Precedence::ReverseApp as u8, source, out)?;
        out.write_str(" | ")?;
        return write_preserving_at(function, (Precedence::ReverseApp as u8) + 1, source, out);
    }
    // Default function-call form.
    write_preserving_with_parens(function, source, out)?;
    if let Expr::Tuple(_) = &argument.expr {
        write_preserving(argument, source, out)
    } else {
        out.write_str("(")?;
        write_preserving(argument, source, out)?;
        out.write_str(")")
    }
}

fn write_preserving_at<M, W>(
    expr: &SourceExpr<M>,
    min_precedence: u8,
    source: &str,
    out: &mut W,
) -> fmt::Result
where
    M: Display,
    W: fmt::Write,
{
    if (expr_precedence(&expr.expr) as u8) < min_precedence {
        out.write_str("(")?;
        write_preserving(expr, source, out)?;
        out.write_str(")")
    } else {
        write_preserving(expr, source, out)
    }
}

fn write_preserving_with_parens<M, W>(
    expr: &SourceExpr<M>,
    source: &str,
    out: &mut W,
) -> fmt::Result
where
    M: Display,
    W: fmt::Write,
{
    match &expr.expr {
        Expr::Float(_)
        | Expr::Waveform(_)
        | Expr::Variable(_)
        | Expr::BuiltIn { .. }
        | Expr::Application { .. }
        | Expr::Tuple(_) => write_preserving(expr, source, out),
        _ => {
            out.write_str("(")?;
            write_preserving(expr, source, out)?;
            out.write_str(")")
        }
    }
}

fn write_preserving_as_let<M, W>(
    function: &SourceExpr<M>,
    argument: &SourceExpr<M>,
    source: &str,
    out: &mut W,
) -> fmt::Result
where
    M: Display,
    W: fmt::Write,
{
    out.write_str("let ")?;
    let mut current_fn = function;
    let mut current_arg = argument;
    let mut first = true;
    loop {
        if let Expr::Function { pattern, body } = &current_fn.expr {
            if !first {
                out.write_str(", ")?;
            }
            first = false;
            write!(out, "{} = ", pattern)?;
            write_preserving(current_arg, source, out)?;
            if let Expr::Application {
                function: next_fn,
                argument: next_arg,
            } = &body.expr
                && matches!(next_fn.expr, Expr::Function { .. })
            {
                current_fn = next_fn;
                current_arg = next_arg;
                continue;
            }
            out.write_str(" in ")?;
            return write_preserving(body, source, out);
        }
        unreachable!("write_preserving_as_let entered with non-Function function");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_line_col() {
        let source = "ab\ncd\n";
        assert_eq!(line_col(source, 0), (1, 1));
        assert_eq!(line_col(source, 1), (1, 2));
        assert_eq!(line_col(source, 2), (1, 3)); // the newline itself
        assert_eq!(line_col(source, 3), (2, 1));
        assert_eq!(line_col(source, 4), (2, 2));
        assert_eq!(line_col(source, 6), (3, 1)); // offset == len, past last newline
        assert_eq!(line_col(source, 100), (3, 1)); // clamps to len

        // Columns count chars, not bytes: 'é' is 2 bytes.
        let source = "é = 1\nx";
        assert_eq!(line_col(source, 2), (1, 2)); // the space after 'é'
        assert_eq!(line_col(source, 7), (2, 1)); // 'x': 'é' is bytes 0..2, '\n' is byte 6

        assert_eq!(line_col("", 0), (1, 1));
    }

    #[test]
    fn test_set_span_source() {
        let expr = SourceExpr::with_span(
            Expr::Application {
                function: Box::new(SourceExpr::<u32>::with_span(
                    Expr::Variable("f".to_string()),
                    0..1,
                )),
                argument: Box::new(SourceExpr::with_span(Expr::Float(1.0), 2..3)),
            },
            0..3,
        );
        let mut bindings = vec![SourceBinding::with_span(
            Binding::Definition(Pattern::Identifier("x".to_string()), expr),
            0..3,
        )];
        set_span_source(&mut bindings, SourceId::Module(7));

        let source_of = |span: &Option<Span>| span.as_ref().unwrap().source;
        assert_eq!(source_of(&bindings[0].span), SourceId::Module(7));
        let Binding::Definition(_, expr) = &bindings[0].binding else {
            panic!("expected a definition");
        };
        assert_eq!(source_of(&expr.span), SourceId::Module(7));
        let Expr::Application { function, argument } = &expr.expr else {
            panic!("expected an application");
        };
        assert_eq!(source_of(&function.span), SourceId::Module(7));
        assert_eq!(source_of(&argument.span), SourceId::Module(7));
    }

    #[test]
    fn test_display_with_source() {
        let source = "a = 1;\nb = nope;\n";
        let error = Error::with_span("bad".to_string(), Some(Span::local(11..15)));
        assert_eq!(error.display_with_source(source), "2:5: bad");
        let error = Error::new("bad".to_string());
        assert_eq!(error.display_with_source(source), "bad");
    }
}
