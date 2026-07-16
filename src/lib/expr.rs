//! The tuun expression language: the AST data structures, their construction
//! helpers, and printing, both the canonical `Display` form and the
//! source-preserving printer.

use std::fmt;
use std::fmt::{Debug, Display};
use std::ops::Range;
use std::rc::Rc;

use crate::waveform;

/// A byte range plus the identity of the text it indexes.
///
/// `S` is the source-identity type (which file, module, or editor buffer the
/// range indexes); the parser produces spans at the placeholder `S = ()` and
/// stamps the caller-supplied identity before returning (see
/// [`crate::parser::parse_module`]).
#[derive(Clone, Debug, PartialEq)]
pub struct Span<S = ()> {
    pub source: S,
    pub range: Range<usize>,
}

impl<S> Span<S> {
    /// Builds a span over `range` of the text identified by `source`.
    pub fn new(source: S, range: Range<usize>) -> Span<S> {
        Span { source, range }
    }
}

impl Span {
    /// Builds a span into parsed text whose source identity has not been
    /// stamped yet (see [`stamp_bindings`]).
    pub fn unstamped(range: Range<usize>) -> Span {
        Span { source: (), range }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Error<S = ()> {
    span: Option<Span<S>>,
    message: String,
}

impl<S> Error<S> {
    pub fn new(message: String) -> Self {
        Self {
            span: None,
            message,
        }
    }
    /// Constructs an `Error` located at the given span.
    pub fn with_span(message: String, span: Option<Span<S>>) -> Self {
        Self { span, message }
    }

    /// The error's byte range, without its source identity.
    pub fn range(&self) -> Option<Range<usize>> {
        self.span.as_ref().map(|span| span.range.clone())
    }

    /// The identity of the text this error's range indexes.
    pub fn source(&self) -> Option<S>
    where
        S: Copy,
    {
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

// Message-only: rendering a position requires the source text the span
// indexes (see `display_with_source` and `Evaluator::diagnose`); raw byte
// offsets would leak into user-visible messages.
impl<S> Display for Error<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

/// The shared function backing a [`BuiltInFn`]: a pure function from a
/// vector of values to a value.
pub type BuiltInImpl<M, S> = Rc<dyn Fn(Vec<Expr<M, S>>) -> Expr<M, S>>;

#[derive(Clone)]
pub struct BuiltInFn<M, S = ()>(pub BuiltInImpl<M, S>);

impl<M, S> Debug for BuiltInFn<M, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BuiltInFn(...)")
    }
}

#[derive(Clone, Debug)]
pub enum Pattern {
    Identifier(String),
    Tuple(Vec<Pattern>),
}

/// Named parameters (on a function) or named arguments (at a call site):
/// each pairs a name with its default value or supplied value.
pub type NamedExprs<M, S = ()> = Vec<(String, SourceExpr<M, S>)>;

#[derive(Clone, Debug)]
pub enum Expr<M, S = ()> {
    // Values
    Bool(bool),
    Float(f32),
    String(String),
    Waveform(waveform::Waveform<M>),
    Function {
        positional: Vec<Pattern>,
        /// Named parameters with their default values; call sites may
        /// override them by name. Defaults are evaluated once, when the
        /// function expression itself is evaluated.
        named: NamedExprs<M, S>,
        body: Box<SourceExpr<M, S>>,
    },
    BuiltIn {
        name: String,
        // Pure functions from a vector of values to a value
        function: BuiltInFn<M, S>,
    },
    // A sequence-able waveform. In value form, both offset and waveform are Expr::Waveform.
    Seq {
        offset: Box<SourceExpr<M, S>>,
        waveform: Box<SourceExpr<M, S>>,
    },
    // If-Then-Else expression
    IfThenElse {
        condition: Box<SourceExpr<M, S>>,
        then: Box<SourceExpr<M, S>>,
        else_: Box<SourceExpr<M, S>>,
    },
    // Function application
    Variable(String),
    Application {
        function: Box<SourceExpr<M, S>>,
        positional: Vec<SourceExpr<M, S>>,
        /// Named arguments, which follow the positional arguments in source and
        /// may only name the function's named parameters.
        named: NamedExprs<M, S>,
    },
    // Compound expressions
    Tuple(Vec<SourceExpr<M, S>>),
    List(Vec<SourceExpr<M, S>>),
    // Errors
    Error(String),
}

/// An expression, optionally paired with the source span it was parsed from.
///
/// A span makes two promises, one stronger than the other:
///
/// - **Provenance**: the expression originated at that range. This survives
///   semantics-preserving rewrites (substitution, evaluation), so diagnostics
///   can always use a span to point at an expression's origin.
/// - **Verbatim**: the source text at the range still reads back as exactly
///   this expression. Only the source-preserving printer needs this promise,
///   and it checks it with `is_clean` (see the printer section below): a
///   expression is only spliced verbatim from `span` if every sub-expression
///   inside it also has a span. Rewrites support this by rebuilding the nodes
///   they change *without* spans (via [`SourceExpr::from`]), so a subtree that
///   no longer matches its source text is not judged clean.
#[derive(Clone, Debug)]
pub struct SourceExpr<M, S = ()> {
    pub expr: Expr<M, S>,
    pub span: Option<Span<S>>, // Some(_) for parser output; None for synthesized
}

impl<M, S> From<Expr<M, S>> for SourceExpr<M, S> {
    fn from(expr: Expr<M, S>) -> Self {
        SourceExpr { expr, span: None }
    }
}

/// Boxes a bare `Expr<M, S>` into a `Box<SourceExpr<M, S>>` with no span.
///
/// Useful for synthesized compound expressions in builtins and the evaluator.
pub fn boxed<M, S>(expr: Expr<M, S>) -> Box<SourceExpr<M, S>> {
    Box::new(SourceExpr::from(expr))
}

impl<M, S> Expr<M, S> {
    /// Builds an application of `function` to `positional` arguments.
    pub fn application(
        function: SourceExpr<M, S>,
        positional: Vec<SourceExpr<M, S>>,
    ) -> Expr<M, S> {
        Expr::Application {
            function: Box::new(function),
            positional,
            named: Vec::new(),
        }
    }
}

impl<M> SourceExpr<M> {
    /// Builds an expression spanning `range` of parsed text, with an
    /// unstamped source identity.
    pub fn with_span(expr: Expr<M>, range: Range<usize>) -> SourceExpr<M> {
        SourceExpr {
            expr,
            span: Some(Span::unstamped(range)),
        }
    }
}

impl<M, S> SourceExpr<M, S> {
    pub fn bool(v: bool) -> SourceExpr<M, S> {
        SourceExpr::from(Expr::Bool(v))
    }
    pub fn float(v: f32) -> SourceExpr<M, S> {
        SourceExpr::from(Expr::Float(v))
    }
    pub fn string(v: String) -> SourceExpr<M, S> {
        SourceExpr::from(Expr::String(v))
    }
    pub fn variable(name: String) -> SourceExpr<M, S> {
        SourceExpr::from(Expr::Variable(name))
    }
    pub fn error(msg: String) -> SourceExpr<M, S> {
        SourceExpr::from(Expr::Error(msg))
    }
    pub fn function(positional: Vec<Pattern>, body: SourceExpr<M, S>) -> SourceExpr<M, S> {
        SourceExpr::from(Expr::Function {
            positional,
            named: Vec::new(),
            body: Box::new(body),
        })
    }
    pub fn application(
        function: SourceExpr<M, S>,
        positional: Vec<SourceExpr<M, S>>,
    ) -> SourceExpr<M, S> {
        SourceExpr::from(Expr::application(function, positional))
    }
    pub fn tuple(exprs: Vec<SourceExpr<M, S>>) -> SourceExpr<M, S> {
        SourceExpr::from(Expr::Tuple(exprs))
    }
    pub fn list(exprs: Vec<SourceExpr<M, S>>) -> SourceExpr<M, S> {
        SourceExpr::from(Expr::List(exprs))
    }
    pub fn if_then_else(
        condition: SourceExpr<M, S>,
        then: SourceExpr<M, S>,
        else_: SourceExpr<M, S>,
    ) -> SourceExpr<M, S> {
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
pub enum Binding<M, S = ()> {
    /// An import that binds all bindings from the module at `path` in the
    /// current scope. These bindings are not public in the current scope.
    // TODO make it just bind public ones
    Open(Vec<String>),
    /// Binds variables in `pattern` to the corresponding values in `expr`.
    Definition(Pattern, SourceExpr<M, S>),
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
pub struct SourceBinding<M, S = ()> {
    pub binding: Binding<M, S>,
    /// Annotations attached to this binding.
    pub annotations: Vec<SourceAnnotation<S>>,
    pub span: Option<Span<S>>,
}

// TODO need to define and handle the case where the same annotation type occurs
// multiple times on a single definition

impl<M, S> From<Binding<M, S>> for SourceBinding<M, S> {
    fn from(binding: Binding<M, S>) -> Self {
        SourceBinding {
            binding,
            annotations: Vec::new(),
            span: None,
        }
    }
}

impl<M, S> SourceBinding<M, S> {
    pub fn definition(pattern: Pattern, expr: SourceExpr<M, S>) -> SourceBinding<M, S> {
        SourceBinding::from(Binding::Definition(pattern, expr))
    }
}

impl<M> SourceBinding<M> {
    /// Builds a binding spanning `range` of parsed text, with an
    /// unstamped source identity.
    pub fn with_span(binding: Binding<M>, range: Range<usize>) -> SourceBinding<M> {
        SourceBinding {
            binding,
            annotations: Vec::new(),
            span: Some(Span::unstamped(range)),
        }
    }
}

/// An `Annotation` together with the byte range in source it was parsed from.
#[derive(Debug, Clone)]
pub struct SourceAnnotation<S = ()> {
    pub annotation: Annotation,
    pub span: Option<Span<S>>,
}

impl<S> From<Annotation> for SourceAnnotation<S> {
    fn from(annotation: Annotation) -> Self {
        SourceAnnotation {
            annotation,
            span: None,
        }
    }
}

/// Stamps `source` as the source identity of every span in `bindings`,
/// including annotations and nested expressions.
///
/// Consumes unstamped parser output; the tree's span type changes from
/// `()` to `S`, so a stamped tree cannot be stamped again.
pub(crate) fn stamp_bindings<M, S: Copy>(
    bindings: Vec<SourceBinding<M>>,
    source: S,
) -> Vec<SourceBinding<M, S>> {
    bindings
        .into_iter()
        .map(|binding| stamp_binding(binding, source))
        .collect()
}

/// Stamps `source` as the source identity of every span in `binding`.
fn stamp_binding<M, S: Copy>(binding: SourceBinding<M>, source: S) -> SourceBinding<M, S> {
    let SourceBinding {
        binding,
        annotations,
        span,
    } = binding;
    SourceBinding {
        binding: match binding {
            Binding::Open(path) => Binding::Open(path),
            Binding::Definition(pattern, expr) => {
                Binding::Definition(pattern, stamp_expr(expr, source))
            }
            Binding::Empty => Binding::Empty,
        },
        annotations: annotations
            .into_iter()
            .map(|a| SourceAnnotation {
                annotation: a.annotation,
                span: stamp_span(a.span, source),
            })
            .collect(),
        span: stamp_span(span, source),
    }
}

/// Stamps `source` as the source identity of every span in `expr`'s tree.
pub(crate) fn stamp_expr<M, S: Copy>(expr: SourceExpr<M>, source: S) -> SourceExpr<M, S> {
    let SourceExpr { expr, span } = expr;
    let expr = match expr {
        Expr::Bool(v) => Expr::Bool(v),
        Expr::Float(v) => Expr::Float(v),
        Expr::String(v) => Expr::String(v),
        Expr::Variable(name) => Expr::Variable(name),
        Expr::Waveform(w) => Expr::Waveform(w),
        Expr::Error(message) => Expr::Error(message),
        // Built-ins exist only in synthesized trees (the prelude), never in
        // parser output, and their closures cannot change span type.
        Expr::BuiltIn { .. } => unreachable!("cannot stamp a built-in"),
        Expr::Function {
            positional,
            named,
            body,
        } => Expr::Function {
            positional,
            named: stamp_named(named, source),
            body: Box::new(stamp_expr(*body, source)),
        },
        Expr::Seq { offset, waveform } => Expr::Seq {
            offset: Box::new(stamp_expr(*offset, source)),
            waveform: Box::new(stamp_expr(*waveform, source)),
        },
        Expr::IfThenElse {
            condition,
            then,
            else_,
        } => Expr::IfThenElse {
            condition: Box::new(stamp_expr(*condition, source)),
            then: Box::new(stamp_expr(*then, source)),
            else_: Box::new(stamp_expr(*else_, source)),
        },
        Expr::Application {
            function,
            positional,
            named,
        } => Expr::Application {
            function: Box::new(stamp_expr(*function, source)),
            positional: positional
                .into_iter()
                .map(|a| stamp_expr(a, source))
                .collect(),
            named: stamp_named(named, source),
        },
        Expr::Tuple(exprs) => {
            Expr::Tuple(exprs.into_iter().map(|e| stamp_expr(e, source)).collect())
        }
        Expr::List(exprs) => Expr::List(exprs.into_iter().map(|e| stamp_expr(e, source)).collect()),
    };
    SourceExpr {
        expr,
        span: stamp_span(span, source),
    }
}

/// Stamps `source` as the source identity of every span in `named`'s value
/// expressions.
fn stamp_named<M, S: Copy>(named: NamedExprs<M>, source: S) -> NamedExprs<M, S> {
    named
        .into_iter()
        .map(|(name, value)| (name, stamp_expr(value, source)))
        .collect()
}

/// Stamps `source` as the source identity of `span`, if there is one.
fn stamp_span<S>(span: Option<Span>, source: S) -> Option<Span<S>> {
    span.map(|span| Span {
        source,
        range: span.range,
    })
}

/// Stamps `source` as the source identity of each error's span.
pub(crate) fn stamp_errors<S: Copy>(errors: Vec<Error>, source: S) -> Vec<Error<S>> {
    errors
        .into_iter()
        .map(|error| Error {
            span: stamp_span(error.span, source),
            message: error.message,
        })
        .collect()
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
fn expr_precedence<M, S>(expr: &Expr<M, S>) -> Precedence {
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
        Expr::Application {
            function,
            positional,
            named,
        } => {
            // Named arguments only ever print in function-call form.
            if !named.is_empty() {
                return Precedence::Application;
            }
            if let Expr::Variable(op) = &function.expr {
                if positional.len() == 2
                    && let Some(p) = binary_op_precedence(op)
                {
                    return p;
                }
                if positional.len() == 1 && is_unary_op(op) {
                    return Precedence::Unary;
                }
            }
            // Single-binding function-literal application → displayed as
            // `let` (same precedence as Function).
            if as_let_binding(function, positional, named).is_some() {
                return Precedence::Followed;
            }
            // Single-argument application of an application → displayed as
            // `|` pipe.
            if positional.len() == 1 && matches!(function.expr, Expr::Application { .. }) {
                return Precedence::ReverseApp;
            }
            Precedence::Application
        }
        Expr::Function { .. } | Expr::IfThenElse { .. } => Precedence::Followed,
    }
}

/// A `let`-shaped application decomposed into (pattern, argument, body).
type LetBinding<'a, M, S> = (&'a Pattern, &'a SourceExpr<M, S>, &'a SourceExpr<M, S>);

/// The single binding of an application that `let` syntax can represent, if
/// any: a function literal with exactly one parameter and no named parameters,
/// applied to exactly one argument with no named arguments (`let` syntax cannot
/// express either kind of named entry).
///
/// Both printers and `expr_precedence` must share this predicate so that
/// parenthesization matches what actually gets emitted.
fn as_let_binding<'a, M, S>(
    function: &'a SourceExpr<M, S>,
    pos_args: &'a [SourceExpr<M, S>],
    named_args: &[(String, SourceExpr<M, S>)],
) -> Option<LetBinding<'a, M, S>> {
    if let Expr::Function {
        positional,
        named: defaults,
        body,
    } = &function.expr
        && named_args.is_empty()
        && defaults.is_empty()
        && positional.len() == 1
        && pos_args.len() == 1
    {
        Some((&positional[0], &pos_args[0], body))
    } else {
        None
    }
}

/// Emits `let p1 = e1, p2 = e2, … in body` for a chain of single-binding
/// function-literal applications (see [`as_let_binding`]), merging nested
/// `let`s into a single comma-separated form.
fn fmt_as_let<M, S>(
    function: &SourceExpr<M, S>,
    arguments: &[SourceExpr<M, S>],
    f: &mut fmt::Formatter,
) -> fmt::Result
where
    M: fmt::Display,
{
    write!(f, "let ")?;
    let mut current_fn = function;
    let mut current_args = arguments;
    let mut first = true;
    loop {
        let (pattern, argument, body) = as_let_binding(current_fn, current_args, &[])
            .expect("fmt_as_let entered with a non-let-shaped application");
        if !first {
            write!(f, ", ")?;
        }
        first = false;
        write!(f, "{} = {}", pattern, argument)?;
        if let Expr::Application {
            function: next_fn,
            positional: next_args,
            named: next_named,
        } = &body.expr
            && as_let_binding(next_fn, next_args, next_named).is_some()
        {
            current_fn = next_fn;
            current_args = next_args;
            continue;
        }
        return write!(f, " in {}", body);
    }
}

/// Writes `name = value` pairs, continuing a comma-separated sequence:
/// emits a separating `", "` before each pair unless `first` is still
/// true.
fn fmt_named<M, S, W>(
    named: &[(String, SourceExpr<M, S>)],
    mut first: bool,
    f: &mut W,
) -> fmt::Result
where
    M: Display,
    W: fmt::Write,
{
    for (name, value) in named {
        if !first {
            write!(f, ", ")?;
        }
        first = false;
        write!(f, "{} = {}", name, value)?;
    }
    Ok(())
}

/// Format `expr` in an operator context with minimum precedence `min_precedence`,
/// adding parens iff the child's outermost form binds looser than required.
fn fmt_at<M, S>(expr: &SourceExpr<M, S>, min_precedence: u8, f: &mut fmt::Formatter) -> fmt::Result
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

impl<M, S> Display for SourceExpr<M, S>
where
    M: Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.expr.fmt(f)
    }
}

impl<M, S> Display for Expr<M, S>
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
            Expr::Function {
                positional,
                named,
                body,
            } => {
                write!(f, "fn(")?;
                for (i, param) in positional.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", param)?;
                }
                fmt_named(named, positional.is_empty(), f)?;
                write!(f, ") => {}", body)
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
            Expr::Application {
                function,
                positional,
                named,
            } => {
                // Render operator applications and `{...}` / `<...>` sugar
                // using surface syntax matching the parser, so the Display
                // output round-trips. None of the sugar forms can carry
                // named arguments, so any named argument forces call form.
                if named.is_empty() {
                    if let Expr::Variable(name) = &function.expr {
                        // `__chord` and `__sequence` aren't legal identifiers —
                        // they can only be entered via the `{x}` / `<x>` sugar.
                        // Emit the sugar so the output re-parses.
                        match name.as_str() {
                            "__chord" if positional.len() == 1 => {
                                return write!(f, "{{{}}}", positional[0]);
                            }
                            "__sequence" if positional.len() == 1 => {
                                return write!(f, "<{}>", positional[0]);
                            }
                            _ => {}
                        }
                        if positional.len() == 2
                            && let Some(p) = binary_op_precedence(name)
                        {
                            // Left-associative: lhs allows equal-precedence,
                            // rhs requires strictly tighter precedence.
                            fmt_at(&positional[0], p as u8, f)?;
                            write!(f, " {} ", name)?;
                            fmt_at(&positional[1], (p as u8) + 1, f)?;
                            return Ok(());
                        }
                        if positional.len() == 1 && is_unary_op(name) {
                            write!(f, "{}", name)?;
                            return fmt_at(&positional[0], Precedence::Unary as u8, f);
                        }
                    }
                    // Single-binding function literal LHS → `let p = arg in body`.
                    if as_let_binding(function, positional, named).is_some() {
                        return fmt_as_let(function, positional, f);
                    }
                    // Application LHS with one argument → `arg | function` pipe.
                    if positional.len() == 1 && matches!(function.expr, Expr::Application { .. }) {
                        fmt_at(&positional[0], Precedence::ReverseApp as u8, f)?;
                        write!(f, " | ")?;
                        return fmt_at(function, (Precedence::ReverseApp as u8) + 1, f);
                    }
                }
                // Default: function-call form. The head is printed at
                // `Application` precedence — anything looser (Function,
                // IfThenElse, an operator app) gets wrapped.
                fmt_at(function, Precedence::Application as u8, f)?;
                write!(f, "(")?;
                for (i, argument) in positional.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", argument)?;
                }
                fmt_named(named, positional.is_empty(), f)?;
                write!(f, ")")
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
fn is_clean<M, S>(node: &SourceExpr<M, S>) -> bool {
    match &node.expr {
        // Leaves: must have been stamped by the parser.
        Expr::Bool(_)
        | Expr::Float(_)
        | Expr::String(_)
        | Expr::Variable(_)
        | Expr::Waveform(_)
        | Expr::BuiltIn { .. }
        | Expr::Error(_) => node.span.is_some(),
        Expr::Function { named, body, .. } => {
            node.span.is_some() && named.iter().all(|(_, value)| is_clean(value)) && is_clean(body)
        }
        Expr::Seq { offset, waveform } => {
            node.span.is_some() && is_clean(offset) && is_clean(waveform)
        }
        Expr::IfThenElse {
            condition,
            then,
            else_,
        } => node.span.is_some() && is_clean(condition) && is_clean(then) && is_clean(else_),
        Expr::Application {
            function,
            positional,
            named,
        } => {
            node.span.is_some()
                && is_clean(function)
                && positional.iter().all(is_clean)
                && named.iter().all(|(_, value)| is_clean(value))
        }
        // Tuples in source always have parens; we require a span.
        Expr::Tuple(items) => node.span.is_some() && items.iter().all(is_clean),
        // Lists in source always have brackets; we require a span.
        Expr::List(items) => node.span.is_some() && items.iter().all(is_clean),
    }
}

/// Returns a string that reproduces `node` using `source` as a verbatim
/// reference for unchanged sub-trees. For any subtree where every leaf still
/// has a span, splices `source[span]`. For dirty regions, falls back to
/// structural pretty-print, recursing into children so that clean sub-sub-trees
/// still splice their original source.
pub fn print_preserving<M, S>(node: &SourceExpr<M, S>, source: &str) -> String
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
pub fn print_preserving_module<M, S>(bindings: &[SourceBinding<M, S>], source: &str) -> String
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

impl<M, S> SourceBinding<M, S> {
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

fn write_preserving<M, S, W>(node: &SourceExpr<M, S>, source: &str, out: &mut W) -> fmt::Result
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

fn write_preserving_structural<M, S, W>(expr: &Expr<M, S>, source: &str, out: &mut W) -> fmt::Result
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
        Expr::Function {
            positional,
            named,
            body,
        } => {
            write!(out, "fn(")?;
            for (i, param) in positional.iter().enumerate() {
                if i > 0 {
                    write!(out, ", ")?;
                }
                write!(out, "{}", param)?;
            }
            write_preserving_named(named, positional.is_empty(), source, out)?;
            write!(out, ") => ")?;
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
        Expr::Application {
            function,
            positional,
            named,
        } => write_preserving_application(function, positional, named, source, out),
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
fn write_preserving_elements<M, S, W>(
    items: &[SourceExpr<M, S>],
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

/// Emit `name = value` pairs, continuing a comma-separated sequence and
/// splicing each value's original source where possible.
fn write_preserving_named<M, S, W>(
    named: &[(String, SourceExpr<M, S>)],
    mut first: bool,
    source: &str,
    out: &mut W,
) -> fmt::Result
where
    M: Display,
    W: fmt::Write,
{
    for (name, value) in named {
        if !first {
            out.write_str(", ")?;
        }
        first = false;
        write!(out, "{} = ", name)?;
        write_preserving(value, source, out)?;
    }
    Ok(())
}

fn write_preserving_application<M, S, W>(
    function: &SourceExpr<M, S>,
    arguments: &[SourceExpr<M, S>],
    named: &[(String, SourceExpr<M, S>)],
    source: &str,
    out: &mut W,
) -> fmt::Result
where
    M: Display,
    W: fmt::Write,
{
    // None of the sugar forms can carry named arguments (mirrors Display).
    if named.is_empty() {
        if let Expr::Variable(name) = &function.expr {
            match name.as_str() {
                "__chord" if arguments.len() == 1 => {
                    out.write_str("{")?;
                    write_preserving(&arguments[0], source, out)?;
                    return out.write_str("}");
                }
                "__sequence" if arguments.len() == 1 => {
                    out.write_str("<")?;
                    write_preserving(&arguments[0], source, out)?;
                    return out.write_str(">");
                }
                _ => {}
            }
            if arguments.len() == 2
                && let Some(p) = binary_op_precedence(name)
            {
                write_preserving_at(&arguments[0], p as u8, source, out)?;
                write!(out, " {} ", name)?;
                return write_preserving_at(&arguments[1], (p as u8) + 1, source, out);
            }
            if arguments.len() == 1 && is_unary_op(name) {
                write!(out, "{}", name)?;
                return write_preserving_at(&arguments[0], Precedence::Unary as u8, source, out);
            }
        }
        if as_let_binding(function, arguments, named).is_some() {
            return write_preserving_as_let(function, arguments, source, out);
        }
        if matches!(function.expr, Expr::Application { .. }) && arguments.len() == 1 {
            write_preserving_at(&arguments[0], Precedence::ReverseApp as u8, source, out)?;
            out.write_str(" | ")?;
            return write_preserving_at(function, (Precedence::ReverseApp as u8) + 1, source, out);
        }
    }
    // Default function-call form.
    write_preserving_with_parens(function, source, out)?;
    out.write_str("(")?;
    write_preserving_elements(arguments, source, out)?;
    write_preserving_named(named, arguments.is_empty(), source, out)?;
    out.write_str(")")
}

fn write_preserving_at<M, S, W>(
    expr: &SourceExpr<M, S>,
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

fn write_preserving_with_parens<M, S, W>(
    expr: &SourceExpr<M, S>,
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

fn write_preserving_as_let<M, S, W>(
    function: &SourceExpr<M, S>,
    arguments: &[SourceExpr<M, S>],
    source: &str,
    out: &mut W,
) -> fmt::Result
where
    M: Display,
    W: fmt::Write,
{
    out.write_str("let ")?;
    let mut current_fn = function;
    let mut current_args = arguments;
    let mut first = true;
    loop {
        let (pattern, argument, body) = as_let_binding(current_fn, current_args, &[])
            .expect("write_preserving_as_let entered with a non-let-shaped application");
        if !first {
            out.write_str(", ")?;
        }
        first = false;
        write!(out, "{} = ", pattern)?;
        write_preserving(argument, source, out)?;
        if let Expr::Application {
            function: next_fn,
            positional: next_args,
            named: next_named,
        } = &body.expr
            && as_let_binding(next_fn, next_args, next_named).is_some()
        {
            current_fn = next_fn;
            current_args = next_args;
            continue;
        }
        out.write_str(" in ")?;
        return write_preserving(body, source, out);
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
    fn test_stamp_bindings() {
        let expr = SourceExpr::with_span(
            Expr::application(
                SourceExpr::<u32>::with_span(Expr::Variable("f".to_string()), 0..1),
                vec![SourceExpr::with_span(Expr::Float(1.0), 2..3)],
            ),
            0..3,
        );
        let bindings = vec![SourceBinding::with_span(
            Binding::Definition(Pattern::Identifier("x".to_string()), expr),
            0..3,
        )];
        let bindings = stamp_bindings(bindings, 7u32);

        let source_of = |span: &Option<Span<u32>>| span.as_ref().unwrap().source;
        assert_eq!(source_of(&bindings[0].span), 7);
        let Binding::Definition(_, expr) = &bindings[0].binding else {
            panic!("expected a definition");
        };
        assert_eq!(source_of(&expr.span), 7);
        let Expr::Application {
            function,
            positional,
            ..
        } = &expr.expr
        else {
            panic!("expected an application");
        };
        assert_eq!(source_of(&function.span), 7);
        assert_eq!(source_of(&positional[0].span), 7);
    }

    #[test]
    fn test_display_with_source() {
        let source = "a = 1;\nb = nope;\n";
        let error = Error::with_span("bad".to_string(), Some(Span::unstamped(11..15)));
        assert_eq!(error.display_with_source(source), "2:5: bad");
        let error = Error::<()>::new("bad".to_string());
        assert_eq!(error.display_with_source(source), "bad");
    }
}
