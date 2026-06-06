use std::fmt;
use std::fmt::{Debug, Display};
use std::ops::Range;
use std::{cell::RefCell, rc::Rc};

use nom::{
    Parser,
    branch::alt,
    bytes::complete::{tag, take_while, take_while1},
    character::complete as character,
    character::complete::{alpha1, alphanumeric1, char},
    combinator::{all_consuming, not, peek, recognize, verify},
    multi::{many0, many1, separated_list0},
    number::complete as number,
    sequence::{delimited, preceded, terminated},
};

use crate::waveform;

type LocatedSpan<'a> = nom_locate::LocatedSpan<&'a str, ParseState<'a>>;
type IResult<'a, T> = nom::IResult<LocatedSpan<'a>, T>;

trait ToRange {
    fn to_range(&self) -> Range<usize>;
}

impl<'a> ToRange for LocatedSpan<'a> {
    fn to_range(&self) -> Range<usize> {
        let start = self.location_offset();
        let end = start + self.fragment().len();
        start..end
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Error {
    range: Option<Range<usize>>,
    message: String,
}

impl Error {
    pub fn new(message: String) -> Self {
        Self {
            range: None,
            message,
        }
    }
    /// Construct an `Error` with the given byte range. Use when a non-parser
    /// caller (e.g. the evaluator) wants to attach a source location to an
    /// error it didn't itself produce.
    pub fn with_range(message: String, range: Option<Range<usize>>) -> Self {
        Self { range, message }
    }
    fn new_from_span(span: &LocatedSpan, message: String) -> Self {
        Self {
            range: Some(span.to_range()),
            message,
        }
    }

    //pub fn span(&self) -> &Span { &self.span }
    pub fn range(&self) -> Option<Range<usize>> {
        self.range.clone()
    }

    // pub fn line(&self) -> u32 { self.span().location_line() }

    // pub fn offset(&self) -> usize { self.span().location_offset() }
}

impl<'a> nom::error::ParseError<LocatedSpan<'a>> for Error {
    fn from_error_kind(input: LocatedSpan<'a>, kind: nom::error::ErrorKind) -> Self {
        Self::new_from_span(&input, format!("parse error {:?}", kind))
    }

    fn append(_input: LocatedSpan<'a>, _kind: nom::error::ErrorKind, other: Self) -> Self {
        other
    }

    // fn from_char(input: Span<'a>, c: char) -> Self {
    //     Self::new(format!("unexpected character '{}'", c), input)
    // }
}

impl Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.range {
            Some(range) => write!(f, "{} at {}..{}", self.message, range.start, range.end),
            None => f.write_str(&self.message),
        }
    }
}

/// Carried around in the `LocatedSpan::extra` field in
/// between `nom` parsers.
#[derive(Clone, Copy, Debug)]
struct ParseState<'a>(&'a RefCell<Vec<Error>>);

impl<'a> ParseState<'a> {
    /// Pushes an error onto the errors stack from within a `nom`
    /// parser combinator while still allowing parsing to continue.
    fn report_error(&self, error: Error) {
        self.0.borrow_mut().push(error);
    }
}

/// https://eyalkalderon.com/blog/nom-error-recovery/
/// Evaluate `parser` and wrap the result in a `Some(_)`. Otherwise,
/// emit the  provided `error_msg` and return a `None` while allowing
/// parsing to continue.
fn expect<'a, F, E, T>(
    mut parser: F,
    error_msg: E,
) -> impl FnMut(LocatedSpan<'a>) -> IResult<Option<T>>
where
    F: FnMut(LocatedSpan<'a>) -> IResult<T>,
    E: ToString,
{
    move |input: LocatedSpan| match parser(input) {
        Ok((remaining, out)) => Ok((remaining, Some(out))),
        Err(nom::Err::Error(e)) | Err(nom::Err::Failure(e)) => {
            let input = e.input;
            let err = Error::new_from_span(&input, error_msg.to_string());
            input.extra.report_error(err); // Push error onto stack.
            Ok((input, None)) // Parsing failed, but keep going.
        }
        Err(err) => Err(err),
    }
}

/// Consumes a `//` line comment up to (but not including) the trailing newline.
///
/// The newline itself is whitespace and will be picked up by the next pass.
fn line_comment<'a, E>(input: LocatedSpan<'a>) -> nom::IResult<LocatedSpan<'a>, (), E>
where
    E: nom::error::ParseError<LocatedSpan<'a>>,
{
    let (rest, _) = tag::<_, _, E>("//").parse(input)?;
    let (rest, _) = take_while::<_, _, E>(|c: char| c != '\n').parse(rest)?;
    Ok((rest, ()))
}

/// Consumes zero or more whitespace runs and/or `//` line comments.
///
/// Use instead of `multispace0` at parser call sites so comments
/// survive in the surrounding expression's span.
fn trivia0<'a, E>(input: LocatedSpan<'a>) -> nom::IResult<LocatedSpan<'a>, (), E>
where
    E: nom::error::ParseError<LocatedSpan<'a>>,
{
    let (rest, _) = many0(alt((
        nom::character::complete::multispace1.map(|_| ()),
        line_comment,
    )))
    .parse(input)?;
    Ok((rest, ()))
}

/// Like [`trivia0`] but requires at least one whitespace run or comment.
fn trivia1<'a, E>(input: LocatedSpan<'a>) -> nom::IResult<LocatedSpan<'a>, (), E>
where
    E: nom::error::ParseError<LocatedSpan<'a>>,
{
    let (rest, _) = many1(alt((
        nom::character::complete::multispace1.map(|_| ()),
        line_comment,
    )))
    .parse(input)?;
    Ok((rest, ()))
}

// https://github.com/Geal/nom/blob/main/doc/nom_recipes.md#wrapper-combinators-that-eat-whitespace-before-and-after-a-parser
fn ws<'a, F, T, E: nom::error::ParseError<LocatedSpan<'a>>>(
    inner: F,
) -> impl Parser<LocatedSpan<'a>, Output = T, Error = E>
where
    F: Parser<LocatedSpan<'a>, Output = T, Error = E>,
{
    delimited(trivia0, inner, trivia0)
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
    pub span: Option<Range<usize>>, // Some(_) for parser output; None for synthesized
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
    pub fn with_span(expr: Expr<M>, span: Range<usize>) -> SourceExpr<M> {
        SourceExpr {
            expr,
            span: Some(span),
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

/// A `pattern = expr` binding plus its byte range in the source.
///
/// Span semantics ("absorbing" model): the span covers any leading trivia
/// (after the previous `,`, or at the start of the bindings region) through
/// the binding's content. For non-last bindings, it also absorbs trailing
/// trivia up to but not including the following `,`. For the last binding,
/// the span ends at the end of the expression — surrounding parsers (e.g.
/// `parse_let`'s ` in`) consume the trailing whitespace.
///
/// Consequence: comments between two bindings belong to whichever side of
/// the `,` they're on. `x = 1, // for y\n y = 2` puts the comment inside
/// `y`'s span (matching the convention that "comment after the separator
/// applies to the following element").
#[derive(Debug, Clone)]
pub struct Binding<M> {
    pub pattern: Pattern,
    pub expr: SourceExpr<M>,
    pub span: Range<usize>,
}

fn parse_string<M>(input: LocatedSpan) -> IResult<SourceExpr<M>> {
    let start = input.location_offset();
    #[rustfmt::skip]
    let (rest, value) = delimited(
        char('"'),
        // TODO implement escaping
        take_while(|c: char| c != '"'),
        char('"'),
    ).parse(input)?;
    let end = rest.location_offset();
    Ok((
        rest,
        SourceExpr::with_span(Expr::String(value.fragment().to_string()), start..end),
    ))
}

fn parse_float<M>(input: LocatedSpan) -> IResult<SourceExpr<M>> {
    let start = input.location_offset();
    #[rustfmt::skip]
    let (rest, value) = preceded(
        // Handle parsing negative floats ourselves
        not(peek(char('-'))),
        number::float,
    ).parse(input)?;
    let end = rest.location_offset();
    Ok((rest, SourceExpr::with_span(Expr::Float(value), start..end)))
}

fn parse_literal<M>(input: LocatedSpan) -> IResult<SourceExpr<M>> {
    alt((parse_float, parse_string)).parse(input)
}

fn parse_identifier(input: LocatedSpan) -> IResult<String> {
    #[rustfmt::skip]
    let (rest, value) =
        alt((
            verify(recognize((
                    alpha1,
                    many0(alt((alphanumeric1, tag("_"), tag("#")))),
                )),
                |s: &LocatedSpan| *s.fragment() != "fn" &&
                    *s.fragment() != "let" && *s.fragment() != "in" &&
                    *s.fragment() != "if" && *s.fragment() != "then" &&
                    *s.fragment() != "else",
            ),
            parse_unary_operator,
        )).parse(input)?;
    Ok((rest, value.to_string()))
}

fn parse_unary_operator(input: LocatedSpan) -> IResult<LocatedSpan> {
    #[rustfmt::skip]
    let (rest, value) =
        alt((
            tag("!"),
            tag("@"),
            tag("$"),
            tag("%"),
            tag("-"),
            tag("?"),
        )).parse(input)?;
    Ok((rest, value))
}

fn parse_pattern(input: LocatedSpan) -> IResult<Pattern> {
    #[rustfmt::skip]
    let (rest, pattern) =
        alt((
            parse_identifier.map(Pattern::Identifier),
            delimited(
                (char('('), trivia0),
                separated_list0(
                    ws(char(',')),
                    parse_pattern),
                (trivia0, expect(char(')'), "expected ')' at end of tuple pattern")),
            ).map(Pattern::Tuple),
        )).parse(input)?;
    Ok((rest, pattern))
}

fn parse_function<M>(input: LocatedSpan) -> IResult<SourceExpr<M>> {
    let start = input.location_offset();
    #[rustfmt::skip]
    let (rest, (pattern, body)) =
        (delimited(
            (tag("fn"), trivia0),
            alt((
                delimited((char('('), trivia0),
                        parse_identifier.map(Pattern::Identifier),
                        (trivia0, char(')'))),
                parse_pattern,
            )),
            ws(expect(tag("=>"), "expected '=>'"))),
            parse_expr,
        ).parse(input)?;
    let end = rest.location_offset();
    let expr = Expr::Function {
        pattern,
        body: Box::new(body),
    };
    Ok((rest, SourceExpr::with_span(expr, start..end)))
}

/// Parses a comma-separated list of `pattern = expr` bindings, returning
/// each with an absorbing span (see [`Binding`] for the semantics).
fn parse_bindings<M>(input: LocatedSpan) -> IResult<Vec<Binding<M>>> {
    let mut bindings = Vec::new();
    let mut rest = input;
    let mut binding_start = rest.location_offset();

    // Position tracking is done manually rather than via `separated_list0`
    // because we need to know where the next `,` lands so each binding's span
    // extends to just before it.
    loop {
        // Try to parse one binding: leading trivia + pattern + ws(=) + expr.
        let attempt = (trivia0, parse_pattern, ws(char('=')), parse_expr).parse(rest);
        let (after_expr, (_, pattern, _, expr)) = match attempt {
            Ok(x) => x,
            Err(_) => break,
        };
        let expr_end = after_expr.location_offset();

        // Consume trailing trivia in preparation for a possible `,`.
        // `trivia0` always succeeds (it's a `many0`), so the unwrap_or is
        // dead code in practice — keep the after_expr fallback for safety.
        let (after_trivia, _) = trivia0::<Error>(after_expr).unwrap_or((after_expr, ()));
        let pre_comma = after_trivia.location_offset();
        let comma: IResult<char> = char(',').parse(after_trivia);

        match comma {
            Ok((after_comma, _)) => {
                // Non-last binding: span absorbs trailing trivia, ends just
                // before the `,`. The `,` is the only gap byte.
                let next_start = after_comma.location_offset();
                bindings.push(Binding {
                    pattern,
                    expr,
                    span: binding_start..pre_comma,
                });
                rest = after_comma;
                binding_start = next_start;
            }
            Err(_) => {
                // Last binding: span ends at end of expr. Don't consume
                // trailing trivia — the surrounding parser may need it
                // (e.g. `parse_let`'s ` in` requires trivia1).
                bindings.push(Binding {
                    pattern,
                    expr,
                    span: binding_start..expr_end,
                });
                rest = after_expr;
                break;
            }
        }
    }

    Ok((rest, bindings))
}

pub fn make_let<M>(
    bindings: Vec<(Pattern, SourceExpr<M>)>,
    mut expr: SourceExpr<M>,
) -> SourceExpr<M> {
    for (pattern, binding) in bindings.into_iter().rev() {
        expr = SourceExpr::application(SourceExpr::function(pattern, expr), binding)
    }
    expr
}

fn parse_let<M>(input: LocatedSpan) -> IResult<SourceExpr<M>> {
    let start = input.location_offset();
    #[rustfmt::skip]
    let (rest, (bindings, body)) =
        (delimited(
            (tag("let"), trivia1),
            parse_bindings,
            (trivia1, expect(tag("in"), "expected 'in'"), trivia1),
        ),
        expect(parse_expr, "expected expression after 'in'")
        ).parse(input)?;
    let end = rest.location_offset();
    let body = body.unwrap_or_default();
    // Strip binding spans for `make_let`: the de-sugared lambda form doesn't
    // model them.
    let bindings: Vec<(Pattern, SourceExpr<M>)> =
        bindings.into_iter().map(|b| (b.pattern, b.expr)).collect();
    // `make_let` de-sugars to nested applications; stamp the outer span over
    // the whole `let … in …` source range.
    let folded = make_let(bindings, body);
    Ok((rest, SourceExpr::with_span(folded.expr, start..end)))
}

fn parse_if_then_else<M>(input: LocatedSpan) -> IResult<SourceExpr<M>> {
    let start = input.location_offset();
    #[rustfmt::skip]
    let (rest, (condition, then, _, else_)) =
        (delimited(
            (tag("if"), trivia1),
            parse_expr,
            delimited(trivia1, tag("then"), trivia1),
        ),
        parse_expr,
        delimited(trivia1, tag("else"), trivia1),
        parse_expr,
        ).parse(input)?;
    let end = rest.location_offset();
    let expr = Expr::IfThenElse {
        condition: Box::new(condition),
        then: Box::new(then),
        else_: Box::new(else_),
    };
    Ok((rest, SourceExpr::with_span(expr, start..end)))
}

fn parse_unary_application<M>(input: LocatedSpan) -> IResult<SourceExpr<M>> {
    let start = input.location_offset();
    let (rest, (op, expr)) = (parse_unary_operator, parse_primitive).parse(input)?;
    let end = rest.location_offset();
    let var = SourceExpr::with_span(
        Expr::Variable(op.fragment().to_string()),
        start..op.location_offset() + op.fragment().len(),
    );
    let app = Expr::Application {
        function: Box::new(var),
        argument: Box::new(expr),
    };
    Ok((rest, SourceExpr::with_span(app, start..end)))
}

fn parse_variable<M>(input: LocatedSpan) -> IResult<SourceExpr<M>> {
    let start = input.location_offset();
    let (rest, name) = parse_identifier(input)?;
    let end = rest.location_offset();
    Ok((
        rest,
        SourceExpr::with_span(Expr::Variable(name), start..end),
    ))
}

fn parse_primitive<M>(input: LocatedSpan) -> IResult<SourceExpr<M>> {
    #[rustfmt::skip]
    let (rest, value) = alt((
        parse_literal,
        parse_function,
        parse_let,
        parse_if_then_else,
        // Should come before identifiers, since operators also match the identifier rule
        parse_unary_application,
        parse_variable,
        parse_chord,
        parse_sequence,
        parse_tuple,
        parse_list
    )).parse(input)?;
    Ok((rest, value))
}

fn parse_application<M>(input: LocatedSpan) -> IResult<SourceExpr<M>> {
    let start = input.location_offset();
    let (mut rest, mut result) = parse_primitive(input)?;
    loop {
        // Peek-and-consume one (whitespace + tuple) iteration to track positions.
        let attempt = preceded(trivia0, parse_tuple).parse(rest);
        match attempt {
            Ok((new_rest, arg)) => {
                let end = new_rest.location_offset();
                let app = Expr::Application {
                    function: Box::new(result),
                    argument: Box::new(arg),
                };
                result = SourceExpr::with_span(app, start..end);
                rest = new_rest;
            }
            Err(_) => break,
        }
    }
    Ok((rest, result))
}

/// Left-folds a binary-operator parser.
///
/// After each successful (op, rhs) step, stamps the resulting Application
/// with a span that runs from the start of the first sub-expression through
/// the end of the current step, so intermediate folds carry accurate source
/// ranges.
fn fold_binary_op<'a, M, P, Q>(
    input: LocatedSpan<'a>,
    mut lhs: P,
    mut op_rhs: Q,
) -> IResult<'a, SourceExpr<M>>
where
    P: FnMut(LocatedSpan<'a>) -> IResult<'a, SourceExpr<M>>,
    Q: FnMut(LocatedSpan<'a>) -> IResult<'a, (LocatedSpan<'a>, Option<SourceExpr<M>>)>,
{
    let start = input.location_offset();
    let (mut rest, mut expr) = lhs(input)?;
    while let Ok((new_rest, (op, rhs))) = op_rhs(rest) {
        let end = new_rest.location_offset();
        let rhs = rhs.unwrap_or_default();
        let op_var = SourceExpr::with_span(
            Expr::Variable(op.fragment().to_string()),
            op.location_offset()..op.location_offset() + op.fragment().len(),
        );
        let args = SourceExpr::tuple(vec![expr, rhs]);
        let app = Expr::Application {
            function: Box::new(op_var),
            argument: Box::new(args),
        };
        expr = SourceExpr::with_span(app, start..end);
        rest = new_rest;
    }

    Ok((rest, expr))
}

fn parse_multiplicative<M>(input: LocatedSpan) -> IResult<SourceExpr<M>> {
    fold_binary_op(input, parse_application, |rest| {
        (
            ws(alt((tag("*"), tag("/"), tag("~*")))),
            expect(parse_application, "expected expression after operator"),
        )
            .parse(rest)
    })
}

fn parse_additive<M>(input: LocatedSpan) -> IResult<SourceExpr<M>> {
    fold_binary_op(input, parse_multiplicative, |rest| {
        (
            ws(alt((tag("+"), tag("-"), tag("&")))),
            expect(parse_multiplicative, "expected expression after operator"),
        )
            .parse(rest)
    })
}

fn parse_relational<M>(input: LocatedSpan) -> IResult<SourceExpr<M>> {
    fold_binary_op(input, parse_additive, |rest| {
        let (rest, op) = ws(alt((
            tag("=="),
            tag("!="),
            tag("<="),
            tag(">="),
            tag("<"),
            tag(">"),
        )))
        .parse(rest)?;
        let (rest, rhs) = parse_additive(rest)?;
        Ok((rest, (op, Some(rhs))))
    })
}

fn parse_chord<M>(input: LocatedSpan) -> IResult<SourceExpr<M>> {
    let start = input.location_offset();
    #[rustfmt::skip]
    let (rest, inner) = delimited(
        (char('{'), trivia0),
        parse_expr,
        (trivia0, expect(char('}'), "expected '}' at end of chord")),
    ).parse(input)?;
    let end = rest.location_offset();
    let app = Expr::Application {
        function: boxed(Expr::Variable("_chord".to_string())),
        argument: Box::new(inner),
    };
    Ok((rest, SourceExpr::with_span(app, start..end)))
}

fn parse_sequence<M>(input: LocatedSpan) -> IResult<SourceExpr<M>> {
    let start = input.location_offset();
    #[rustfmt::skip]
    let (rest, inner) = delimited(
        (char('<'), trivia0),
        parse_expr,
        (trivia0, expect(char('>'), "expected '>' at end of sequence")),
    ).parse(input)?;
    let end = rest.location_offset();
    let app = Expr::Application {
        function: boxed(Expr::Variable("_sequence".to_string())),
        argument: Box::new(inner),
    };
    Ok((rest, SourceExpr::with_span(app, start..end)))
}

/// Parses a parenthesized expression or expressions. If there is only one element, returns just that element;
/// otherwise returns a Tuple of elements.
fn parse_tuple<M>(input: LocatedSpan) -> IResult<SourceExpr<M>> {
    let start = input.location_offset();
    #[rustfmt::skip]
    let (rest, mut exprs) = delimited(
        (char('('), trivia0),
        separated_list0(
            (trivia0, char(','), trivia0),
            parse_expr,
        ),
        (trivia0, expect(char(')'), "expected ')' at end of tuple")),
    ).parse(input)?;
    if exprs.len() == 1 {
        // Parenthesized single expression — return it directly, preserving its inner span.
        return Ok((rest, exprs.pop().unwrap()));
    }
    let end = rest.location_offset();
    Ok((rest, SourceExpr::with_span(Expr::Tuple(exprs), start..end)))
}

fn parse_list<M>(input: LocatedSpan) -> IResult<SourceExpr<M>> {
    let start = input.location_offset();
    #[rustfmt::skip]
    let (rest, exprs) = delimited(
        (char('['), trivia0),
        separated_list0(
            (trivia0, char(','), trivia0),
            parse_expr,
        ),
        (trivia0, expect(char(']'), "expected ']' at end of list")),
    ).parse(input)?;
    let end = rest.location_offset();
    Ok((rest, SourceExpr::with_span(Expr::List(exprs), start..end)))
}

fn parse_reverse_application<M>(input: LocatedSpan) -> IResult<SourceExpr<M>> {
    let start = input.location_offset();
    let (mut rest, mut argument) = parse_relational(input)?;
    loop {
        let attempt = preceded(
            ws(char('|')),
            expect(parse_relational, "expected expression after | operator"),
        )
        .parse(rest);
        match attempt {
            Ok((new_rest, function)) => {
                let end = new_rest.location_offset();
                let function = function.unwrap_or_default();
                let app = Expr::Application {
                    function: Box::new(function),
                    argument: Box::new(argument),
                };
                argument = SourceExpr::with_span(app, start..end);
                rest = new_rest;
            }
            Err(_) => break,
        }
    }
    Ok((rest, argument))
}

fn parse_expr<M>(input: LocatedSpan) -> IResult<SourceExpr<M>> {
    let start = input.location_offset();
    let (mut rest, mut expr) = parse_reverse_application(input)?;
    loop {
        let attempt = preceded(
            ws(char('\\')),
            expect(
                parse_reverse_application,
                "expected expression after \\ operator",
            ),
        )
        .parse(rest);
        match attempt {
            Ok((new_rest, rhs)) => {
                let end = new_rest.location_offset();
                let rhs = rhs.unwrap_or_default();
                let op_var = SourceExpr::from(Expr::Variable("\\".to_string()));
                let args = SourceExpr::tuple(vec![expr, rhs]);
                let app = Expr::Application {
                    function: Box::new(op_var),
                    argument: Box::new(args),
                };
                expr = SourceExpr::with_span(app, start..end);
                rest = new_rest;
            }
            Err(_) => break,
        }
    }
    Ok((rest, expr))
}

fn translate_parse_result<T>(result: IResult<T>) -> Result<T, Vec<Error>> {
    match result {
        Ok((_, a)) => Ok(a),
        Err(nom::Err::Error(e)) => {
            //println!("Error on parsing input: {:?}", e);
            Err(vec![Error::new_from_span(
                &e.input,
                "unable to parse input".to_string(),
            )])
        }
        Err(nom::Err::Incomplete(_)) => {
            panic!("Incomplete error on input");
        }
        Err(nom::Err::Failure(e)) => {
            println!("Failed to parse input: {:?}", e);
            Err(vec![Error::new_from_span(
                &e.input,
                "unable to parse input".to_string(),
            )])
        }
    }
}

pub fn parse_program<M>(input: &str) -> Result<SourceExpr<M>, Vec<Error>>
where
    M: Display,
{
    let errors = RefCell::new(Vec::new());
    let span = LocatedSpan::new_extra(input, ParseState(&errors));
    #[rustfmt::skip]
    let result = all_consuming(
        ws(parse_expr),
    ).parse(span);
    if !errors.borrow().is_empty() {
        println!(
            "Got result {:} and errors {:?}",
            match result {
                Ok((_, node)) => node,
                _ => SourceExpr::default(),
            },
            errors.borrow()
        );
        return Err(errors.into_inner());
    }
    translate_parse_result(result)
}

pub fn parse_context<M>(input: &str) -> Result<Vec<Binding<M>>, Vec<Error>> {
    let errors = RefCell::new(Vec::new());
    let span = LocatedSpan::new_extra(input, ParseState(&errors));
    // `parse_bindings` already absorbs leading trivia inside its first
    // iteration (so the first binding's span includes any file-leading
    // comments). The trailing consumer is `multispace0` rather than
    // `trivia0` so that trailing *comments* (which aren't in any binding's
    // span) cause a parse failure rather than being silently dropped on
    // round-trip. Trailing whitespace is incidental and still consumed.
    let result = all_consuming(terminated(
        parse_bindings,
        nom::character::complete::multispace0,
    ))
    .parse(span);
    if !errors.borrow().is_empty() {
        return Err(errors.into_inner());
    }
    translate_parse_result(result)
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
    NextBank,
    Color(u8, u8, u8),
    Level(f32),
}

/// Parses a slider config entry — either linear `"label:initial:min:max"`
/// or user-defined `"label:normalized_initial_value:fn(x) => expr"`.
fn parse_slider(input: LocatedSpan) -> IResult<Slider> {
    // TODO this is a bit of a mess... and also could first parse the whole thing as a string literal.
    let label_char = |c: char| c != ':' && c != '"' && c != ',' && c != ']' && !c.is_whitespace();
    // Parse the quoted opening, label, and first float (shared prefix)
    #[rustfmt::skip]
    let (rest, (_, label, (initial_span, initial_value), _)) = (
        char('"'),
        expect(take_while1(label_char), "expected slider label"),
        preceded(
            expect(char(':'), "expected ':'"),
            (peek(recognize(number::float)), expect(number::float, "expected initial value"))),
        expect(char(':'), "expected ':'"),
    ).parse(input)?;

    let label = label.unwrap().fragment().to_string();
    let initial_value = initial_value.unwrap_or_default();

    // Peek to discriminate: digit/'-'/'.' → linear (min field), otherwise → user-defined function
    let next_char = rest.fragment().chars().next().unwrap_or('"');
    let is_numeric_start = next_char.is_ascii_digit() || next_char == '-' || next_char == '.';

    if is_numeric_start {
        // Linear: parse min and max
        #[rustfmt::skip]
        let (rest, (min, max)) = (
            expect(number::float, "expected minimum slider value"),
            preceded(
                expect(char(':'), "expected ':'"),
                expect(number::float, "expected maximum slider value")),
        ).parse(rest)?;

        let min = min.unwrap_or_default();
        let max = max.unwrap_or_default();
        if min > initial_value || max < initial_value {
            initial_span.extra.report_error(Error::new_from_span(
                &initial_span,
                format!(
                    "initial value {} is not between min {} and max {}",
                    initial_value, min, max
                ),
            ));
            return Err(nom::Err::Failure(nom::error::Error::new(
                initial_span,
                nom::error::ErrorKind::Verify,
            )));
        }

        let (rest, _) = char('"').parse(rest)?;
        Ok((
            rest,
            Slider {
                label,
                function: SliderFunction::Linear {
                    initial_value,
                    min,
                    max,
                },
            },
        ))
    } else {
        // User-defined: everything until closing quote is the function expression
        let (rest, function_source) = take_while(|c: char| c != '"').parse(rest)?;
        let (rest, _) = char('"').parse(rest)?;
        Ok((
            rest,
            Slider {
                label,
                function: SliderFunction::UserDefined {
                    normalized_initial_value: initial_value,
                    function_source: function_source.fragment().trim().to_string(),
                },
            },
        ))
    }
}

fn parse_sliders_internal(input: LocatedSpan) -> IResult<Vec<Slider>> {
    let (rest, sliders) = preceded(
        tag("sliders="),
        delimited(
            ws(char('[')),
            separated_list0(ws(char(',')), parse_slider),
            ws(char(']')),
        ),
    )
    .parse(input)?;
    Ok((rest, sliders))
}

/// Parses a bracket-delimited, comma-separated list of slider configs.
/// Example: `["volume:0.5:0:1", "cutoff:2000:200:8000"]`
pub fn parse_sliders(input: &str) -> Result<Vec<Slider>, Vec<Error>> {
    let errors = RefCell::new(Vec::new());
    let span = LocatedSpan::new_extra(input, ParseState(&errors));
    let result = all_consuming(parse_sliders_internal).parse(span);
    if !errors.borrow().is_empty() {
        return Err(errors.into_inner());
    }
    translate_parse_result(result)
}

/// Parses `color=rgb(R,G,B)` where R, G, B are integers in [0, 255].
fn parse_color(input: LocatedSpan) -> IResult<(u8, u8, u8)> {
    #[rustfmt::skip]
    let (rest, (_, r, _, g, _, b, _)) = (
        tag("color=rgb("),
        ws(character::u8),
        char(','),
        ws(character::u8),
        char(','),
        ws(character::u8),
        char(')'),
    ).parse(input)?;
    Ok((rest, (r, g, b)))
}

/// Parses `level_db=<float>` (e.g. `level_db=-6.0`).
fn parse_level(input: LocatedSpan) -> IResult<f32> {
    let (rest, (_, value)) = (tag("level_db="), number::float).parse(input)?;
    Ok((rest, value))
}

/// Parses annotations like `//#{sliders=[...],next_bank}`.
pub fn parse_annotations(line: &str) -> Result<Vec<Annotation>, Vec<Error>> {
    match line.trim().strip_prefix("//#") {
        Some(input) => {
            let errors = RefCell::new(Vec::new());
            let span = LocatedSpan::new_extra(input, ParseState(&errors));
            #[rustfmt::skip]
            let result = all_consuming(
                delimited(
                    ws(char('{')),
                    separated_list0(
                        ws(char(',')),
                        alt((
                            parse_sliders_internal.map(Annotation::Sliders),
                            tag("next_bank").map(|_| Annotation::NextBank),
                            parse_color.map(|(r, g, b)| Annotation::Color(r, g, b)),
                            parse_level.map(Annotation::Level),
                        ))),
                    ws(char('}')),
                )
            ).parse(span);
            if !errors.borrow().is_empty() {
                return Err(errors.into_inner());
            }
            translate_parse_result(result)
        }
        None => Ok(Vec::new()),
    }
}

/// Extends the context with a binding for each identifier in the pattern that is bound to
/// itself.
fn extend_with_trivial_context<M>(context: &mut Vec<(String, SourceExpr<M>)>, pattern: &Pattern) {
    match pattern {
        Pattern::Identifier(name) => {
            context.push((name.clone(), SourceExpr::variable(name.clone())));
        }
        Pattern::Tuple(patterns) => {
            for pattern in patterns {
                extend_with_trivial_context(context, pattern);
            }
        }
    }
}

fn substitute<M>(context: &[(String, SourceExpr<M>)], expr: SourceExpr<M>) -> SourceExpr<M>
where
    M: Clone,
{
    use Expr::{
        Application, Bool, BuiltIn, Float, Function, IfThenElse, List, String, Tuple, Variable,
    };
    let SourceExpr { expr, span } = expr;
    match expr {
        Bool(_) | Float(_) | String(_) => SourceExpr { expr, span },
        Expr::Waveform(w) => SourceExpr {
            expr: Expr::Waveform(w),
            span,
        },
        Expr::Seq { offset, waveform } => SourceExpr::from(Expr::Seq {
            offset: Box::new(substitute(context, *offset)),
            waveform: Box::new(substitute(context, *waveform)),
        }),
        Function { pattern, body } => {
            let mut context = Vec::from(context);
            extend_with_trivial_context(&mut context, &pattern);
            let body = substitute(&context, *body);
            SourceExpr::from(Expr::Function {
                pattern,
                body: Box::new(body),
            })
        }
        BuiltIn { name, function } => SourceExpr {
            expr: Expr::BuiltIn { name, function },
            span,
        },
        Variable(name) => {
            for (var_name, value) in context.iter().rev() {
                if var_name == &name {
                    return value.clone();
                }
            }
            SourceExpr::error(format!("Variable '{}' not found in context", name))
        }
        IfThenElse {
            condition,
            then,
            else_,
        } => {
            let condition = Box::new(substitute(context, *condition));
            let then = Box::new(substitute(context, *then));
            let else_ = Box::new(substitute(context, *else_));
            SourceExpr::from(Expr::IfThenElse {
                condition,
                then,
                else_,
            })
        }
        Application { function, argument } => {
            let function = substitute(context, *function);
            let argument = substitute(context, *argument);
            SourceExpr::from(Expr::Application {
                function: Box::new(function),
                argument: Box::new(argument),
            })
        }
        Tuple(exprs) => SourceExpr::from(Expr::Tuple(
            exprs.into_iter().map(|e| substitute(context, e)).collect(),
        )),
        List(exprs) => SourceExpr::from(Expr::List(
            exprs.into_iter().map(|e| substitute(context, e)).collect(),
        )),
        Expr::Error(s) => SourceExpr::error(s),
    }
}

fn fmt_with_parens<M>(expr: &SourceExpr<M>, f: &mut fmt::Formatter) -> fmt::Result
where
    M: fmt::Display,
{
    match &expr.expr {
        Expr::Float(_)
        | Expr::Waveform(_)
        | Expr::Variable(_)
        | Expr::BuiltIn { .. }
        | Expr::Application { .. }
        | Expr::Tuple(_) => {
            write!(f, "{}", expr)
        }
        _ => write!(f, "({})", expr),
    }
}

// Operator precedence levels for pretty-printing. Higher = binds tighter.
// Must match the parser's precedence hierarchy so that
// `format!("{}", parse(s))` round-trips back to the same AST.
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
    M: fmt::Display,
{
    if (expr_precedence(&expr.expr) as u8) < min_precedence {
        write!(f, "({})", expr)
    } else {
        write!(f, "{}", expr)
    }
}

impl fmt::Display for Pattern {
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

impl<M> fmt::Display for SourceExpr<M>
where
    M: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.expr.fmt(f)
    }
}

impl<M> fmt::Display for Expr<M>
where
    M: fmt::Display,
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
                    // `_chord` and `_sequence` aren't legal identifiers — they
                    // can only be entered via the `{x}` / `<x>` sugar. Emit
                    // the sugar so the output re-parses.
                    match name.as_str() {
                        "_chord" => return write!(f, "{{{}}}", argument),
                        "_sequence" => return write!(f, "<{}>", argument),
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
                // Default: function-call form.
                fmt_with_parens(function, f)?;
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

/// True iff `node` and every descendant either has Some span or is one of
/// the parser's "transparent packaging" forms (binary-op-argument Tuples).
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
        // packaging around the two operands of a binary operator — those
        // Tuples have no span of their own but the surrounding Application
        // does, and splicing the Application's span gives the right text.
        // So a Tuple's own span isn't required; what matters is its children.
        Expr::Tuple(items) => items.iter().all(is_clean),
        // Lists in source always have brackets; we require a span.
        Expr::List(items) => node.span.is_some() && items.iter().all(is_clean),
    }
}

/// Return a string that reproduces `node` using `source` as a verbatim
/// reference for unchanged sub-trees. For any subtree where every leaf
/// still has a span, splices `source[span]`. For dirty regions, falls back
/// to structural pretty-print, recursing into children so that clean
/// sub-sub-trees still splice their original source.
pub fn print_preserving<M>(node: &SourceExpr<M>, source: &str) -> String
where
    M: Display,
{
    let mut out = String::new();
    write_preserving(node, source, &mut out).expect("write to String");
    out
}

/// Round-trip a parsed context (a list of bindings) back to source.
///
/// Splices each binding's span verbatim. Between consecutive bindings,
/// splices `source[prev.span.end..next.span.start]` — which captures the
/// `,` separator and any whitespace the user had around it.
///
/// Comments are preserved by the [`Binding::span`] semantics:
/// - Leading comments (including any file-header before the first binding)
///   belong to the **following** binding — its span absorbs them.
/// - Trailing comments (before the next `,`) belong to the **preceding**
///   binding — its span extends through them.
/// - Trailing content *after the last binding* can only be whitespace —
///   [`parse_context`] rejects trailing comments.
pub fn print_preserving_context<M>(bindings: &[Binding<M>], source: &str) -> String
where
    M: Display,
{
    let mut out = String::new();
    for (i, binding) in bindings.iter().enumerate() {
        if i > 0 {
            let prev_end = bindings[i - 1].span.end;
            let curr_start = binding.span.start;
            if prev_end <= curr_start && curr_start <= source.len() {
                out.push_str(&source[prev_end..curr_start]);
            } else {
                out.push_str(", ");
            }
        }
        // A binding is "clean" iff its expr is clean (the Pattern is owned
        // by the binding and can't drift independently for now). Splice
        // the binding's source span when clean; structural-print pattern
        // + " = " + expr otherwise.
        if is_clean(&binding.expr) && binding.span.end <= source.len() {
            out.push_str(&source[binding.span.clone()]);
        } else {
            // Structural fallback: emit pattern + " = " + expr.
            // (Leading/trailing trivia inside binding.span is lost in this
            // path — same boundary case as Stage 4's dirty subtrees.)
            use std::fmt::Write;
            write!(out, "{} = ", binding.pattern).expect("write to String");
            write_preserving(&binding.expr, source, &mut out).expect("write to String");
        }
    }
    out
}

/// Replaces `source[target_span]` with `new_text`, re-parse the resulting
/// source, and atomically update both `source` and `root` on success. On
/// parse failure neither is modified and the parse errors are returned.
///
/// The intended use is to find a node by walking `root`, grab its `span`,
/// and pass it as `target_span` here. After the call returns Ok, every
/// node in the new `root` has fresh spans into the new `source`, so
/// `print_preserving` round-trips the edited file.
pub fn replace_at<M>(
    root: &mut SourceExpr<M>,
    target_span: Range<usize>,
    new_text: &str,
    source: &mut String,
) -> Result<(), Vec<Error>>
where
    M: fmt::Display,
{
    if target_span.end > source.len() || target_span.start > target_span.end {
        return Err(vec![Error::with_range(
            format!(
                "target span {:?} is out of bounds for source of length {}",
                target_span,
                source.len()
            ),
            Some(target_span),
        )]);
    }
    // Re-parses the whole expression rather than just the subtree —
    // simpler, keeps all spans consistent, and is plenty fast for tuun's
    // typical program-sized expressions. The subtree-only path is a future
    // optimization.

    // Speculatively splice and re-parse. If parsing fails, neither `source`
    // nor `root` is touched.
    let mut new_source = source.clone();
    new_source.replace_range(target_span, new_text);
    let new_root = parse_program::<M>(&new_source)?;

    *source = new_source;
    *root = new_root;
    Ok(())
}

fn write_preserving<M, W>(node: &SourceExpr<M>, source: &str, out: &mut W) -> fmt::Result
where
    M: Display,
    W: fmt::Write,
{
    if let Some(span) = &node.span
        && is_clean(node)
    {
        return out.write_str(&source[span.clone()]);
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
                (Some(p), Some(c)) if p.end <= c.start && c.start <= source.len() => {
                    out.write_str(&source[p.end..c.start])?;
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
    M: fmt::Display,
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

pub fn extend_context<M>(
    context: &mut Vec<(String, SourceExpr<M>)>,
    pattern: &Pattern,
    argument: &SourceExpr<M>,
) -> Result<(), Error>
where
    M: Clone + Debug + Display,
{
    match (pattern, &argument.expr) {
        (Pattern::Identifier(name), _) => {
            context.push((name.clone(), argument.clone()));
            Ok(())
        }
        (Pattern::Tuple(patterns), Expr::Tuple(arguments)) => {
            if patterns.len() != arguments.len() {
                return Err(Error::new(format!(
                    "Mismatched number of elements in pattern {:?} and arguments {:?}",
                    patterns, arguments
                )));
            }
            for (pattern, argument) in patterns.iter().zip(arguments) {
                extend_context(context, pattern, argument)?;
            }
            Ok(())
        }
        _ => Err(Error::new(format!(
            "Pattern {:?} does not match actual expression {:?}",
            pattern, argument.expr
        ))),
    }
}

fn evaluate_closed<M>(expr: SourceExpr<M>) -> Result<SourceExpr<M>, Error>
where
    M: Clone + fmt::Display + fmt::Debug,
{
    use Expr::{
        Application, Bool, BuiltIn, Float, Function, IfThenElse, List, Seq, String, Tuple,
        Variable, Waveform,
    };
    let SourceExpr { expr, span } = expr;
    match expr {
        Bool(_) | Float(_) | String(_) | Waveform(_) | Function { .. } => {
            Ok(SourceExpr { expr, span })
        }
        Variable(name) => Err(Error::new(format!(
            "Variable '{}' not found in context",
            name
        ))),
        Seq { offset, waveform } => {
            let offset = evaluate_closed(*offset)?;
            let waveform = evaluate_closed(*waveform)?;
            Ok(SourceExpr::from(Seq {
                offset: Box::new(offset),
                waveform: Box::new(waveform),
            }))
        }
        BuiltIn { name, function } => Ok(SourceExpr {
            expr: BuiltIn { name, function },
            span,
        }),
        IfThenElse {
            condition,
            then,
            else_,
        } => match evaluate_closed(*condition)?.expr {
            Bool(true) => evaluate_closed(*then),
            Bool(false) => evaluate_closed(*else_),
            _ => Err(Error::new("Expected boolean condition".to_string())),
        },
        Application { function, argument } => {
            let function = evaluate_closed(*function)?;
            let argument = evaluate_closed(*argument)?;
            match (function.expr, argument) {
                (Function { pattern, body }, argument) => {
                    let mut context = Vec::new();
                    extend_context(&mut context, &pattern, &argument)?;
                    let body = substitute(&context, *body);
                    evaluate_closed(body)
                }
                (BuiltIn { function, .. }, argument) => {
                    // Builtins operate on bare `Expr<M>` values, so unwrap
                    // the SourceExpr children before calling and wrap the
                    // result. Use the outer Application's span (`span`) so
                    // errors point at the whole `f(x, y)` call site — builtins
                    // themselves don't see spans.
                    let actuals: Vec<Expr<M>> = match argument.expr {
                        Tuple(actuals) => actuals.into_iter().map(|s| s.expr).collect(),
                        other => vec![other],
                    };
                    let result = function.0(actuals);
                    match result {
                        Expr::Error(s) => Err(Error::with_range(s, span.clone())),
                        _ => Ok(SourceExpr::from(result)),
                    }
                }
                (function, actuals) => Err(Error::with_range(
                    format!("Invalid application: {} {}", function, actuals),
                    span.clone(),
                )),
            }
        }
        Tuple(exprs) => Ok(SourceExpr::from(Tuple(
            exprs
                .into_iter()
                .map(|e| evaluate_closed(e))
                .collect::<Result<Vec<_>, _>>()?,
        ))),
        List(exprs) => Ok(SourceExpr::from(List(
            exprs
                .into_iter()
                .map(|e| evaluate_closed(e))
                .collect::<Result<Vec<_>, _>>()?,
        ))),
        Expr::Error(s) => Err(Error::new(s)),
    }
}

pub fn evaluate<M>(
    context: &[(String, SourceExpr<M>)],
    mut expr: SourceExpr<M>,
) -> Result<SourceExpr<M>, Error>
where
    M: Clone + fmt::Display + fmt::Debug,
{
    expr = substitute(context, expr);
    evaluate_closed(expr)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parse `input`, format the AST, assert the result equals `expected`,
    /// and confirm the formatted form re-parses to the same Display output
    /// (round-trip stability).
    fn assert_round_trip(input: &str, expected: &str) {
        let parsed = parse_program::<u32>(input)
            .unwrap_or_else(|errs| panic!("failed to parse {:?}: {:?}", input, errs));
        let displayed = format!("{}", parsed);
        assert_eq!(
            displayed, expected,
            "Display output didn't match expected canonical form\n input: {:?}",
            input
        );
        let parsed_again = parse_program::<u32>(&displayed).unwrap_or_else(|errs| {
            panic!("Display output {:?} didn't reparse: {:?}", displayed, errs)
        });
        let redisplayed = format!("{}", parsed_again);
        assert_eq!(
            displayed, redisplayed,
            "round-trip not stable\n  first display: {:?}\n  second display: {:?}",
            displayed, redisplayed
        );
    }

    #[test]
    fn test_parse_variable() {
        let result = parse_program::<u32>("fn");
        assert!(result.is_err());

        assert_round_trip("my_var", "my_var");
        assert_round_trip("$", "$");
    }

    #[test]
    fn test_parse_with_comments() {
        // Comments anywhere whitespace was previously allowed should be ignored.
        assert_round_trip("1 + // a comment\n 2", "1 + 2");

        // Comment before, between, and after bindings.
        let input = "
            // header comment
            x = 1, // trailing
            // standalone
            y = x + 1
        ";
        let result = parse_context::<u32>(input);
        assert!(result.is_ok(), "got {:?}", result.err());
        assert_eq!(result.unwrap().len(), 2);

        // Comment inside a function body.
        assert_round_trip("fn(x) => x // identity\n", "fn(x) => x");

        // Comment between `let` keyword and bindings.
        let result = parse_program::<u32>("let // bindings follow\n x = 1 in x");
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_arithmetic() {
        // Outer multiplication forces parens on the additive LHS; the inner
        // subtractions are left-associative so they don't.
        assert_round_trip("(10 - 8 - 1) * 6", "(10 - 8 - 1) * 6");

        // Multiplication binds tighter than addition, so no parens for the
        // middle term.
        assert_round_trip("1 + 2 * 3.5 * 8 + 10", "1 + 2 * 3.5 * 8 + 10");
    }

    #[test]
    fn test_parse_chord() {
        // `{...}` is the only surface syntax for `_chord` (the identifier
        // can't be written directly), so Display re-emits the sugar.
        assert_round_trip("{[$x, $y, $z]}", "{[$x, $y, $z]}");
    }

    #[test]
    fn test_parse_sequence() {
        // Same as `_chord` — `<...>` is the only way to write a sequence.
        assert_round_trip("<[$x, $y, $z]>", "<[$x, $y, $z]>");
    }

    #[test]
    fn test_parse_function() {
        assert_round_trip("fn(x) => x", "fn(x) => x");
        assert_round_trip("fn(x, (y, z)) => x", "fn(x, (y, z)) => x");
    }

    #[test]
    fn test_parse_let() {
        // `let` de-sugars to nested lambda applications, which Display
        // re-sugars back into the comma-separated `let` form.
        assert_round_trip(
            "let x = 1, y = x + 1 in 2 * y",
            "let x = 1, y = x + 1 in 2 * y",
        );
        assert_round_trip(
            "let (x, y) = (1, 2) in x * y",
            "let (x, y) = (1, 2) in x * y",
        );
    }

    #[test]
    fn test_parse_application() {
        assert_round_trip(
            "let f = fn(x) => x * 2 in f(3)",
            "let f = fn(x) => x * 2 in f(3)",
        );
        // Function literals on the LHS of an application are not canonical.
        assert_round_trip(
            "(fn(f) => f(3))(fn(x) => x * 2)",
            "let f = fn(x) => x * 2 in f(3)",
        );
        // Stacked unary prefix operators chain without parens.
        assert_round_trip("Q($@70)", "Q($@70)");
        assert_round_trip("f(-1) - 1 < 0", "f(-1) - 1 < 0");
    }

    #[test]
    fn test_parse_pipe() {
        assert_round_trip(
            "2 * 3 | (let x = 4 in fn(y) => x * y)",
            "2 * 3 | (let x = 4 in fn(y) => x * y)",
        );
        // Application on the LHS of an application is not canonical.
        assert_round_trip(
            "let f = fn(x) => fn(y) => x * y in f(4)(2 * 3)",
            "let f = fn(x) => fn(y) => x * y in 2 * 3 | f(4)",
        );
        // Both the `let` and pipe rules combined.
        assert_round_trip(
            "(fn(x) => fn(y) => x * y)(4)(2 * 3)",
            "2 * 3 | (let x = 4 in fn(y) => x * y)",
        );
        // Pipe chain followed by `\` — each `f(x)(y)`-style sub-chain
        // becomes `x | f` once the outer function side is an Application.
        assert_round_trip(
            r"$200 | S(0.5, .25) | R(0.5, 1) \ $400",
            r"$200 | S(0.5, 0.25) | R(0.5, 1) \ $400",
        );
    }

    #[test]
    fn test_print_preserving_clean() {
        // A parsed expression splices back to exactly the source span the
        // parser saw — including any whitespace / comments between operands
        // that are *inside* the outer node's span.
        let input = "1 + 2";
        let parsed = parse_program::<u32>(input).unwrap();
        assert_eq!(print_preserving(&parsed, input), "1 + 2");

        // A line comment between the two operands of `+` falls inside the
        // outer Application's span and round-trips verbatim. (Display alone
        // would normalize whitespace to `1 + 2`.)
        let input = "1 + // a comment\n  2";
        let parsed = parse_program::<u32>(input).unwrap();
        assert_eq!(print_preserving(&parsed, input), "1 + // a comment\n  2");

        // Multiple operators — same deal, just splice the outer span.
        let input = "(10 - 8 - 1) * 6 // total\n";
        let parsed = parse_program::<u32>(input).unwrap();
        // Outer span doesn't include the trailing `// total\n` (it's the
        // outer `ws()`'s trailing trivia), so the splice ends at `6`.
        assert_eq!(print_preserving(&parsed, input), "(10 - 8 - 1) * 6");
    }

    #[test]
    fn test_print_preserving_synthesized_falls_back() {
        // A synthesized expression with no span at all falls back to
        // structural pretty-printing (same output as Display).
        let synth: SourceExpr<u32> = SourceExpr::float(42.0);
        assert_eq!(print_preserving(&synth, ""), "42");

        // A synthesized binary op application also has no spans anywhere; pretty-
        // print emits canonical infix.
        let lhs = SourceExpr::float(1.0);
        let rhs = SourceExpr::float(2.0);
        let plus = SourceExpr::variable("+".to_string());
        let synth: SourceExpr<u32> =
            SourceExpr::application(plus, SourceExpr::tuple(vec![lhs, rhs]));
        assert_eq!(print_preserving(&synth, ""), "1 + 2");
    }

    #[test]
    fn test_print_preserving_partial_edit_preserves_clean_siblings() {
        // Parse `1 + 2`, then swap in a fresh (spanless) Float for the LHS.
        // The Application's outer span is *still* set, but is_clean walks
        // the tree, finds the edited child without a span, and forces the
        // printer to structurally recurse — emitting the new LHS while
        // splicing the RHS verbatim from source.
        let input = "1 + // sibling comment\n 2";
        let mut parsed = parse_program::<u32>(input).unwrap();
        if let Expr::Application { argument, .. } = &mut parsed.expr {
            if let Expr::Tuple(args) = &mut argument.expr {
                args[0] = SourceExpr::float(99.0); // span: None — "edit"
            } else {
                panic!("expected Tuple argument");
            }
        } else {
            panic!("expected Application root");
        }
        // The structural printer emits `99 + <splice of original "2">`.
        // Sibling whitespace/comment between operators isn't preserved at
        // this granularity (the inter-element source isn't covered by either
        // operand's span), but the RHS still splices.
        let out = print_preserving(&parsed, input);
        assert!(out.starts_with("99 + "), "got {:?}", out);
        assert!(out.ends_with('2'), "got {:?}", out);
    }

    #[test]
    fn test_print_preserving_context_round_trip() {
        // Comment after the separator → belongs to the following binding (in
        // its span). Comment before the separator → belongs to the preceding
        // binding. Splicing each binding's span reproduces every comment.
        let input = "x = 1, // for y\n y = x + 1";
        let bindings = parse_context::<u32>(input).unwrap();
        assert_eq!(bindings.len(), 2);
        assert_eq!(print_preserving_context(&bindings, input), input);

        // Comment trailing the first binding (before the `,`) is in x's span.
        let input = "x = 1 // for x\n, y = x + 1";
        let bindings = parse_context::<u32>(input).unwrap();
        assert_eq!(bindings.len(), 2);
        assert_eq!(print_preserving_context(&bindings, input), input);

        // Mixed: header + trailing + leading + inter-binding blank line.
        let input = "x = 1,\n\n// section header\ny = x + 1";
        let bindings = parse_context::<u32>(input).unwrap();
        assert_eq!(bindings.len(), 2);
        assert_eq!(print_preserving_context(&bindings, input), input);

        // A file-header comment at the very top is absorbed into the first
        // binding's span (since `parse_context` doesn't strip leading trivia
        // anymore — the first iteration's inner `trivia0` consumes it).
        let input = "// file header\n// also for the first binding\nx = 1,\ny = 2";
        let bindings = parse_context::<u32>(input).unwrap();
        assert_eq!(bindings.len(), 2);
        assert_eq!(
            bindings[0].span.start, 0,
            "first binding should absorb file-leading"
        );
        assert_eq!(print_preserving_context(&bindings, input), input);
    }

    #[test]
    fn test_parse_context_rejects_trailing_comments() {
        // Trailing whitespace is fine (consumed silently by the terminator).
        assert!(parse_context::<u32>("x = 1\n").is_ok());
        assert!(parse_context::<u32>("x = 1, y = 2 \n  ").is_ok());

        // Trailing comments aren't in any binding's span, so allowing them
        // would mean silent loss on round-trip. The terminator uses
        // `multispace0` (whitespace only) so this fails loudly instead.
        assert!(parse_context::<u32>("x = 1\n// trailing").is_err());
        assert!(parse_context::<u32>("x = 1, // trailing").is_err());
    }

    /// Walks `node` and returns the span of the first `List` it finds, or
    /// None. Used by replace_at tests as a stand-in for editor navigation.
    fn find_first_list_span<M>(node: &SourceExpr<M>) -> Option<Range<usize>> {
        if let Expr::List(_) = &node.expr {
            return node.span.clone();
        }
        match &node.expr {
            Expr::Function { body, .. } => find_first_list_span(body),
            Expr::Seq { offset, waveform } => {
                find_first_list_span(offset).or_else(|| find_first_list_span(waveform))
            }
            Expr::IfThenElse {
                condition,
                then,
                else_,
            } => find_first_list_span(condition)
                .or_else(|| find_first_list_span(then))
                .or_else(|| find_first_list_span(else_)),
            Expr::Application { function, argument } => {
                find_first_list_span(function).or_else(|| find_first_list_span(argument))
            }
            Expr::Tuple(items) | Expr::List(items) => items.iter().find_map(find_first_list_span),
            _ => None,
        }
    }

    #[test]
    fn test_replace_at_list_literal() {
        let mut source = String::from("on_beats(b, [0, 1, 2, 3])");
        let mut root = parse_program::<u32>(&source).unwrap();
        let list_span = find_first_list_span(&root).expect("should find a list");

        replace_at(&mut root, list_span, "[0, 2]", &mut source).unwrap();

        // Source updated by the splice; AST re-parsed against the new source.
        assert_eq!(source, "on_beats(b, [0, 2])");
        // The freshly-parsed root has spans into the new source, so
        // print_preserving round-trips the edited file.
        assert_eq!(print_preserving(&root, &source), "on_beats(b, [0, 2])");
    }

    #[test]
    fn test_replace_at_preserves_surrounding_comments() {
        // A comment between the function and its argument should survive
        // the edit (it's inside the outer Application's span, which is
        // re-parsed fresh against the spliced source).
        let mut source = String::from("f( // a list\n [0, 1, 2])");
        let mut root = parse_program::<u32>(&source).unwrap();
        let list_span = find_first_list_span(&root).expect("should find a list");

        replace_at(&mut root, list_span, "[7]", &mut source).unwrap();
        assert_eq!(source, "f( // a list\n [7])");
        assert_eq!(print_preserving(&root, &source), "f( // a list\n [7])");
    }

    #[test]
    fn test_replace_at_rejects_invalid_edit() {
        // If the proposed edit makes the source un-parseable, replace_at
        // returns Err and leaves both `source` and `root` untouched.
        let original = String::from("[1, 2, 3]");
        let mut source = original.clone();
        let original_root = parse_program::<u32>(&source).unwrap();
        let mut root = original_root.clone();
        let list_span = root.span.clone().unwrap();

        // Replacing the list with malformed text fails atomically.
        let result = replace_at(&mut root, list_span, "[1,", &mut source);
        assert!(result.is_err());
        // Both source and root are untouched.
        assert_eq!(source, original);
        assert_eq!(format!("{}", root), format!("{}", original_root));
    }

    #[test]
    fn test_replace_at_out_of_bounds() {
        let mut source = String::from("[1, 2]");
        let mut root = parse_program::<u32>(&source).unwrap();
        let bad_span = 0..1000;
        let result = replace_at(&mut root, bad_span, "x", &mut source);
        assert!(result.is_err());
        assert_eq!(source, "[1, 2]");
    }

    #[test]
    fn test_print_preserving_context_dirty_binding_splices_others() {
        // Edit one binding's RHS (replace Float(1) with Float(99) — no span)
        // and confirm: the dirty binding emits structurally (`x = 99`), but
        // the OTHER binding still splices its leading comment verbatim.
        let input = "x = 1,\n// leading for y\n y = 2";
        let mut bindings = parse_context::<u32>(input).unwrap();
        bindings[0].expr = SourceExpr::float(99.0); // synth, span: None
        let out = print_preserving_context(&bindings, input);
        // x is dirty → structural: "x = 99". y is clean → splices its span
        // (which includes the leading comment).
        assert!(out.starts_with("x = 99"), "got {:?}", out);
        assert!(out.contains("// leading for y"), "got {:?}", out);
        assert!(out.contains("y = 2"), "got {:?}", out);
    }

    #[test]
    fn test_function_eval() {
        let context = Vec::new();
        let input = "(fn(x) => fn(x) => x)(7)(5)";
        let result = parse_program::<u32>(input);
        assert!(result.is_ok());
        let expr = result.unwrap();
        println!("Parsed expression: {}", expr);
        let evaluated = evaluate(&context, expr).unwrap();
        assert_eq!(format!("{}", evaluated), "5");

        let input = "(fn(x) => fn(y, z) => (x, y, z))(3)(4, 5)";
        let result = parse_program::<u32>(input);
        assert!(result.is_ok());
        let expr = result.unwrap();
        let evaluated = evaluate(&context, expr).unwrap();
        assert_eq!(format!("{}", evaluated), "(3, 4, 5)");

        let input = "(fn(x, (y, z)) => (x, y, z))(3, (4, 5))";
        let result = parse_program::<u32>(input);
        assert!(result.is_ok());
        let expr = result.unwrap();
        let evaluated = evaluate(&context, expr).unwrap();
        assert_eq!(format!("{}", evaluated), "(3, 4, 5)");
    }

    #[test]
    fn test_parse_sliders() {
        // Empty list
        let result = parse_sliders("sliders=[]");
        assert!(result.is_ok());
        let configs = result.unwrap();
        assert_eq!(configs.len(), 0);

        // Single slider
        let result = parse_sliders(r#"sliders=["volume:0.75:0:1"]"#);
        assert!(result.is_ok());
        let configs = result.unwrap();
        assert_eq!(configs.len(), 1);
        assert_eq!(configs[0].label, "volume");
        match &configs[0].function {
            SliderFunction::Linear {
                initial_value,
                min,
                max,
            } => {
                assert_eq!(*initial_value, 0.75);
                assert_eq!(*min, 0.0);
                assert_eq!(*max, 1.0);
            }
            other => unreachable!("expected Linear, got {:?}", other),
        }

        // Multiple sliders
        let result = parse_sliders(r#"sliders=[ "volume:0.5:0:1" , "cutoff:2000:200:8000" ]"#);
        assert!(result.is_ok());
        let configs = result.unwrap();

        assert_eq!(configs.len(), 2);
        assert_eq!(configs[0].label, "volume");
        assert_eq!(configs[1].label, "cutoff");
        match &configs[1].function {
            SliderFunction::Linear {
                initial_value,
                min,
                max,
            } => {
                assert_eq!(*initial_value, 2000.0);
                assert_eq!(*min, 200.0);
                assert_eq!(*max, 8000.0);
            }
            other => unreachable!("expected Linear, got {:?}", other),
        }

        // Mixed linear and user-defined sliders
        let result =
            parse_sliders(r#"sliders=["volume:0.5:0:1", "freq:0.5:fn(x) => 100 * pow(100, x)"]"#);
        assert!(result.is_ok());
        let configs = result.unwrap();
        assert_eq!(configs.len(), 2);
        match &configs[0].function {
            SliderFunction::Linear { .. } => {}
            other => unreachable!("expected Linear, got {:?}", other),
        }
        match &configs[1].function {
            SliderFunction::UserDefined {
                function_source, ..
            } => {
                assert_eq!(function_source, "fn(x) => 100 * pow(100, x)");
            }
            other => unreachable!("expected UserDefined, got {:?}", other),
        }

        let result = parse_sliders("not a list");
        assert!(result.is_err());

        let result = parse_sliders(r#"sliders=["gain:-0.5:0:1"]"#);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_annotations() {
        // Valid annotation
        let result = parse_annotations(r#"  //#{sliders=["volume:0.75:0:1"]}  "#);
        assert!(result.is_ok());
        let annos = result.unwrap();
        assert_eq!(annos.len(), 1);
        let result = parse_annotations(r#"  //#{sliders=["volume:0.75:0:1"],next_bank}"#);
        assert!(result.is_ok());
        let annos = result.unwrap();
        assert_eq!(annos.len(), 2);

        // Not an annotation
        let result = parse_annotations("// regular comment");
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 0);
        let result = parse_annotations("let x = 1");
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 0);

        // Missing closing bracket
        assert!(parse_annotations("//#{next_bank").is_err());
        // Bad key
        assert!(parse_annotations("//#{bad_key=[]}").is_err());

        // Color annotation
        let result = parse_annotations("//#{color=rgb(255,0,128)}");
        assert!(result.is_ok());
        let annos = result.unwrap();
        assert_eq!(annos.len(), 1);
        assert!(matches!(annos[0], Annotation::Color(255, 0, 128)));

        // Color with whitespace
        let result = parse_annotations("//#{color=rgb( 255 , 0 , 128 )}");
        assert!(result.is_ok());
        let annos = result.unwrap();
        assert!(matches!(annos[0], Annotation::Color(255, 0, 128)));

        // Color combined with sliders
        let result = parse_annotations(r#"//#{sliders=["vol:0.5:0:1"],color=rgb(255,128,0)}"#);
        assert!(result.is_ok());
        let annos = result.unwrap();
        assert_eq!(annos.len(), 2);
        assert!(matches!(annos[1], Annotation::Color(255, 128, 0)));

        // Color out of range
        assert!(parse_annotations("//#{color=rgb(256,0,0)}").is_err());
        // Color missing component
        assert!(parse_annotations("//#{color=rgb(0,0)}").is_err());
        // Color non-integer
        assert!(parse_annotations("//#{color=rgb(1.5,0,0)}").is_err());

        // Level annotation
        let result = parse_annotations("//#{level_db=-6.0}");
        assert!(result.is_ok());
        let annos = result.unwrap();
        assert_eq!(annos.len(), 1);
        assert!(matches!(annos[0], Annotation::Level(v) if (v - -6.0).abs() < 0.01));

        // Level with other annotations
        let result = parse_annotations(r#"//#{level_db=-3.0,color=rgb(255,0,0)}"#);
        assert!(result.is_ok());
        let annos = result.unwrap();
        assert_eq!(annos.len(), 2);
        assert!(matches!(annos[0], Annotation::Level(v) if (v - -3.0).abs() < 0.01));

        // Positive level
        let result = parse_annotations("//#{level_db=6.0}");
        assert!(result.is_ok());
        let annos = result.unwrap();
        assert!(matches!(annos[0], Annotation::Level(v) if (v - 6.0).abs() < 0.01));
    }
}
