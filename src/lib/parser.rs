use std::cell::RefCell;
use std::fmt::{Debug, Display};
use std::ops::Range;

use nom::{
    Parser,
    branch::alt,
    bytes::complete::{tag, take_till, take_while, take_while1},
    character::complete as character,
    character::complete::{alphanumeric1, char},
    combinator::{all_consuming, not, opt, peek, recognize, verify},
    multi::{many0, many1, separated_list0, separated_list1},
    number::complete as number,
    sequence::{delimited, preceded, separated_pair, terminated},
};

use crate::expr::{
    self, Annotation, Binding, Error, Expr, Pattern, Slider, SliderFunction, SourceAnnotation,
    SourceBinding, SourceExpr, Span, boxed,
};

type Input<'a> = nom_locate::LocatedSpan<&'a str, ParseState<'a>>;
type IResult<'a, T> = nom::IResult<Input<'a>, T>;

trait ToRange {
    fn to_range(&self) -> Range<usize>;
}

impl<'a> ToRange for Input<'a> {
    fn to_range(&self) -> Range<usize> {
        let start = self.location_offset();
        let end = start + self.fragment().len();
        start..end
    }
}

/// Builds an [`Error`] whose range covers `input`'s fragment.
fn error_from_input(input: &Input, message: String) -> Error {
    Error::with_span(message, Some(Span::unstamped(input.to_range())))
}

impl<'a> nom::error::ParseError<Input<'a>> for Error {
    fn from_error_kind(input: Input<'a>, kind: nom::error::ErrorKind) -> Self {
        error_from_input(&input, format!("parse error {:?}", kind))
    }

    fn append(_input: Input<'a>, _kind: nom::error::ErrorKind, other: Self) -> Self {
        other
    }

    // fn from_char(input: Span<'a>, c: char) -> Self {
    //     Self::new(format!("unexpected character '{}'", c), input)
    // }
}

/// Carried around in the `Input::extra` field in
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
fn expect<'a, F, E, T>(mut parser: F, error_msg: E) -> impl FnMut(Input<'a>) -> IResult<Option<T>>
where
    F: FnMut(Input<'a>) -> IResult<T>,
    E: ToString,
{
    move |input: Input| match parser(input) {
        Ok((remaining, out)) => Ok((remaining, Some(out))),
        Err(nom::Err::Error(e)) | Err(nom::Err::Failure(e)) => {
            let input = e.input;
            let err = error_from_input(&input, error_msg.to_string());
            input.extra.report_error(err); // Push error onto stack.
            Ok((input, None)) // Parsing failed, but keep going.
        }
        Err(err) => Err(err),
    }
}

/// Builds the placeholder expression standing in where parsing failed but
/// recovery continues. Always paired with a recoverable error describing
/// the failure (reported by [`expect`] or directly at the recovery site).
fn error_placeholder<M>() -> SourceExpr<M> {
    SourceExpr::error("_".to_string())
}

/// Consumes a `//` line comment up to (but not including) the trailing newline.
fn parse_line_comment(input: Input) -> IResult<()> {
    #[rustfmt::skip]
    let (rest, _) = (
        tag("//"),
        take_while(|c: char| c != '\n'),
    ).parse(input)?;
    Ok((rest, ()))
}

/// Consumes zero or more whitespace runs and comments.
///
/// Use instead of `multispace0` at parser call sites so comments
/// survive in the surrounding expression's span.
fn trivia0(input: Input) -> IResult<()> {
    #[rustfmt::skip]
    let (rest, _) =
        many0(
            alt((
                nom::character::complete::multispace1.map(|_| ()),
                parse_line_comment,
            ))
        ).parse(input)?;
    Ok((rest, ()))
}

/// Like [`trivia0`] but requires at least one whitespace run or comment.
fn trivia1(input: Input) -> IResult<()> {
    #[rustfmt::skip]
    let (rest, _) =
        many1(
            alt((
                nom::character::complete::multispace1.map(|_| ()),
                parse_line_comment,
            ))
        ).parse(input)?;
    Ok((rest, ()))
}

// https://github.com/Geal/nom/blob/main/doc/nom_recipes.md#wrapper-combinators-that-eat-whitespace-before-and-after-a-parser
// TODO rename with_trivia
fn ws<'a, F, T>(
    inner: F,
) -> impl Parser<Input<'a>, Output = T, Error = nom::error::Error<Input<'a>>>
where
    F: Parser<Input<'a>, Output = T, Error = nom::error::Error<Input<'a>>>,
{
    delimited(trivia0, inner, trivia0)
}

fn parse_string<M>(input: Input) -> IResult<SourceExpr<M>> {
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

fn parse_float<M>(input: Input) -> IResult<SourceExpr<M>> {
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

fn parse_literal<M>(input: Input) -> IResult<SourceExpr<M>> {
    alt((parse_float, parse_string)).parse(input)
}

fn parse_identifier(input: Input) -> IResult<String> {
    #[rustfmt::skip]
    let (rest, value) =
        alt((
            verify(recognize((
                    alt((
                        alphanumeric1,
                        // One leading underscore is ok
                        recognize((tag("_"), alphanumeric1)),
                    )),
                    many0(alt((alphanumeric1, tag("_"), tag("#")))),
                )),
                |s: &Input| *s.fragment() != "fn" &&
                    *s.fragment() != "let" && *s.fragment() != "in" &&
                    *s.fragment() != "if" && *s.fragment() != "then" &&
                    *s.fragment() != "else" && *s.fragment() != "open",
            ),
            parse_unary_operator,
            // A lonely underscore is also ok.
            terminated(tag("_"), not(peek(alt((tag("_"), alphanumeric1))))),
        )).parse(input)?;
    Ok((rest, value.to_string()))
}

fn parse_unary_operator(input: Input) -> IResult<Input> {
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

fn parse_pattern(input: Input) -> IResult<Pattern> {
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

fn parse_function<M>(input: Input) -> IResult<SourceExpr<M>> {
    let start = input.location_offset();
    #[rustfmt::skip]
    let (rest, (params, body)) =
        (delimited(
            (tag("fn"), trivia0),
            delimited((char('('), trivia0),
                    separated_list0(ws(char(',')), parse_pattern),
                    (trivia0, expect(char(')'), "expected ')' at end of parameter list"))),
            ws(expect(tag("=>"), "expected '=>'"))),
            parse_expr,
        ).parse(input)?;
    let end = rest.location_offset();
    let expr = Expr::Function {
        params,
        body: Box::new(body),
    };
    Ok((rest, SourceExpr::with_span(expr, start..end)))
}

/// Parses `foo.bar.baz` and returns the module path as a list of
/// components.
fn parse_import_path(input: Input) -> IResult<Vec<String>> {
    separated_list1(char('.'), parse_identifier).parse(input)
}

/// Parses a single binding, including any annotations.
///
/// Unlike many other parsers, this parser consumes surrounding trivia (comments
/// and whitespace). It does not consume any separator or terminator. Returns
/// success if the resulting bindings span the entire input, even those bindings
/// may contain errors.
fn parse_binding<M>(input: Input) -> IResult<SourceBinding<M>> {
    let start = input.location_offset();
    #[rustfmt::skip]
    let (rest, (annos, binding)) =
        delimited(
            trivia0,
            (
                many0(terminated(parse_annotation_set, trivia0)),
                alt((
                    preceded((tag("open"), trivia1), parse_import_path).map(Binding::Open),
                    separated_pair(
                        parse_pattern,
                        ws(expect(char('='), "expected '=' in definition")),
                        alt((
                            parse_expr,
                            // If that fails, consume everything up to a ';',
                            // reporting the skipped text as a recoverable
                            // error (mirroring what `expect` does). The
                            // placeholder node carries the same message, so
                            // if it survives to evaluation (the source-file
                            // path tolerates recoverable errors) the eval
                            // error reads the same way.
                            take_till(|c| c == ';').map(|input: Input| {
                                let message = "expected expression in definition".to_string();
                                input.extra.report_error(error_from_input(&input, message.clone()));
                                SourceExpr::with_span(Expr::Error(message), input.to_range())
                            }),
                        )),
                    ).map(|(p, e)| Binding::Definition(p, e)),
                ))
            ),
            trivia0
        ).parse(input)?;
    let end = rest.location_offset();
    Ok((
        rest,
        SourceBinding {
            binding,
            annotations: annos.into_iter().flatten().collect(),
            span: Some(Span::unstamped(start..end)),
        },
    ))
}

fn parse_let<M>(input: Input) -> IResult<SourceExpr<M>> {
    let start = input.location_offset();
    #[rustfmt::skip]
    let (rest, (bindings, body)) = (
        delimited(
            tag("let"),
            // Note that parse_binding consumes surrounding trivia and annotations.
            separated_list1(char(','), parse_binding),
            (
                // Optional trailing comma
                opt((char(','), trivia0)),
                expect(tag("in"), "expected 'in'"),
                trivia1,
            ),
        ),
        ws(expect(parse_expr, "expected expression after 'in'"))
    ).parse(input)?;
    let end = rest.location_offset();
    let body = body.unwrap_or_else(error_placeholder);
    // Extract `Definition`s for `make_let` (the de-sugared lambda form doesn't
    // model spans). `Open` directives aren't valid inside `let` so report each
    // one as an error.
    let mut definitions: Vec<(Pattern, SourceExpr<M>)> = Vec::new();
    for source_binding in bindings {
        match source_binding.binding {
            Binding::Definition(pattern, expr) => definitions.push((pattern, expr)),
            Binding::Open(_) => {
                rest.extra.report_error(Error::with_span(
                    "`open` is not allowed inside `let`; use it at the top level".to_string(),
                    source_binding.span,
                ));
            }
            Binding::Empty => unreachable!("Got Binding::Empty from parse_binding"),
        }
    }
    // De-sugars to nested applications; stamp the outer span over the whole
    // `let … in …` source range.
    let mut expr = body;
    for (pattern, binding) in definitions.into_iter().rev() {
        expr = SourceExpr::application(SourceExpr::function(vec![pattern], expr), vec![binding])
    }
    Ok((rest, SourceExpr::with_span(expr.expr, start..end)))
}

fn parse_if_then_else<M>(input: Input) -> IResult<SourceExpr<M>> {
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

fn parse_unary_application<M>(input: Input) -> IResult<SourceExpr<M>> {
    let start = input.location_offset();
    let (rest, (op, expr)) = (parse_unary_operator, parse_primitive).parse(input)?;
    let end = rest.location_offset();
    let var = SourceExpr::with_span(
        Expr::Variable(op.fragment().to_string()),
        start..op.location_offset() + op.fragment().len(),
    );
    let app = Expr::Application {
        function: Box::new(var),
        arguments: vec![expr],
    };
    Ok((rest, SourceExpr::with_span(app, start..end)))
}

fn parse_variable<M>(input: Input) -> IResult<SourceExpr<M>> {
    let start = input.location_offset();
    // Variables are the same as identifiers (i.e., trivial patterns) except:
    //   "_" is only a identifier: it may be bound but *not* referenced
    //   "__chord" and others with double-underscore prefixes may be
    //     referenced but *not* bound
    #[rustfmt::skip]
    let (rest, name) = verify(
        alt((
            parse_identifier,
            recognize((
                tag("__"),
                many0(alt((alphanumeric1, tag("_"), tag("#")))),
            )).map(|v: Input| v.fragment().to_string()),
        )),
        |name: &String| name != "_",
    ).parse(input)?;
    let end = rest.location_offset();
    Ok((
        rest,
        SourceExpr::with_span(Expr::Variable(name), start..end),
    ))
}

fn parse_primitive<M>(input: Input) -> IResult<SourceExpr<M>> {
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

/// Parses a call's argument list: `'(' expr ',' ... ')'`, returning one
/// element per argument. Unlike [`parse_tuple`], a single element is NOT
/// collapsed — `f(x)` has one argument, and `f((1, 2))` has one argument
/// that is a pair, distinct from `f(1, 2)`.
fn parse_call_arguments<M>(input: Input) -> IResult<Vec<SourceExpr<M>>> {
    #[rustfmt::skip]
    let (rest, arguments) = delimited(
        (char('('), trivia0),
        separated_list0(
            (trivia0, char(','), trivia0),
            parse_expr,
        ),
        (trivia0, expect(char(')'), "expected ')' at end of arguments")),
    ).parse(input)?;
    Ok((rest, arguments))
}

fn parse_application<M>(input: Input) -> IResult<SourceExpr<M>> {
    let start = input.location_offset();
    let (mut rest, mut result) = parse_primitive(input)?;
    loop {
        // Peek-and-consume one (whitespace + argument-list) iteration to
        // track positions.
        let attempt = preceded(trivia0, parse_call_arguments).parse(rest);
        match attempt {
            Ok((new_rest, arguments)) => {
                let end = new_rest.location_offset();
                let app = Expr::Application {
                    function: Box::new(result),
                    arguments,
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
    input: Input<'a>,
    mut lhs: P,
    mut op_rhs: Q,
) -> IResult<'a, SourceExpr<M>>
where
    P: FnMut(Input<'a>) -> IResult<'a, SourceExpr<M>>,
    Q: FnMut(Input<'a>) -> IResult<'a, (Input<'a>, Option<SourceExpr<M>>)>,
{
    let start = input.location_offset();
    let (mut rest, mut expr) = lhs(input)?;
    while let Ok((new_rest, (op, rhs))) = op_rhs(rest) {
        let end = new_rest.location_offset();
        let rhs = rhs.unwrap_or_else(error_placeholder);
        let op_var = SourceExpr::with_span(
            Expr::Variable(op.fragment().to_string()),
            op.location_offset()..op.location_offset() + op.fragment().len(),
        );
        let app = Expr::Application {
            function: Box::new(op_var),
            arguments: vec![expr, rhs],
        };
        expr = SourceExpr::with_span(app, start..end);
        rest = new_rest;
    }

    Ok((rest, expr))
}

// The operator tags and nesting of the `parse_*` precedence-level parsers
// below must match `expr::Precedence` and `expr::binary_op_precedence`, so
// that `format!("{}", parse(s))` round-trips back to the same AST.
fn parse_multiplicative<M>(input: Input) -> IResult<SourceExpr<M>> {
    fold_binary_op(input, parse_application, |rest| {
        (
            ws(alt((tag("*"), tag("/"), tag("~*")))),
            expect(parse_application, "expected expression after operator"),
        )
            .parse(rest)
    })
}

fn parse_additive<M>(input: Input) -> IResult<SourceExpr<M>> {
    fold_binary_op(input, parse_multiplicative, |rest| {
        (
            ws(alt((tag("+"), tag("-"), tag("&")))),
            expect(parse_multiplicative, "expected expression after operator"),
        )
            .parse(rest)
    })
}

fn parse_relational<M>(input: Input) -> IResult<SourceExpr<M>> {
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

fn parse_chord<M>(input: Input) -> IResult<SourceExpr<M>> {
    let start = input.location_offset();
    #[rustfmt::skip]
    let (rest, inner) = delimited(
        (char('{'), trivia0),
        parse_expr,
        (trivia0, expect(char('}'), "expected '}' at end of chord")),
    ).parse(input)?;
    let end = rest.location_offset();
    let app = Expr::Application {
        function: boxed(Expr::Variable("__chord".to_string())),
        arguments: vec![inner],
    };
    Ok((rest, SourceExpr::with_span(app, start..end)))
}

fn parse_sequence<M>(input: Input) -> IResult<SourceExpr<M>> {
    let start = input.location_offset();
    #[rustfmt::skip]
    let (rest, inner) = delimited(
        (char('<'), trivia0),
        parse_expr,
        (trivia0, expect(char('>'), "expected '>' at end of sequence")),
    ).parse(input)?;
    let end = rest.location_offset();
    let app = Expr::Application {
        function: boxed(Expr::Variable("__sequence".to_string())),
        arguments: vec![inner],
    };
    Ok((rest, SourceExpr::with_span(app, start..end)))
}

/// Parses a parenthesized expression or expressions. If there is only one element, returns just that element;
/// otherwise returns a Tuple of elements.
fn parse_tuple<M>(input: Input) -> IResult<SourceExpr<M>> {
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

fn parse_list<M>(input: Input) -> IResult<SourceExpr<M>> {
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

fn parse_reverse_application<M>(input: Input) -> IResult<SourceExpr<M>> {
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
                let function = function.unwrap_or_else(error_placeholder);
                let app = Expr::Application {
                    function: Box::new(function),
                    arguments: vec![argument],
                };
                argument = SourceExpr::with_span(app, start..end);
                rest = new_rest;
            }
            Err(_) => break,
        }
    }
    Ok((rest, argument))
}

fn parse_expr<M>(input: Input) -> IResult<SourceExpr<M>> {
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
                let rhs = rhs.unwrap_or_else(error_placeholder);
                let op_var = SourceExpr::from(Expr::Variable("\\".to_string()));
                let app = Expr::Application {
                    function: Box::new(op_var),
                    arguments: vec![expr, rhs],
                };
                expr = SourceExpr::with_span(app, start..end);
                rest = new_rest;
            }
            Err(_) => break,
        }
    }
    Ok((rest, expr))
}

/// Builds the message for a hard parse failure at `input`: names the
/// offending text, truncated to its first line and a readable length.
fn unexpected_input_message(input: &Input) -> String {
    let first_line = input.fragment().lines().next().unwrap_or("");
    let mut text: String = first_line.chars().take(30).collect();
    if text.is_empty() {
        return "unexpected end of input".to_string();
    }
    if text.len() < first_line.len() {
        text.push('…');
    }
    format!("unexpected input '{}'", text)
}

fn translate_parse_result<T>(result: IResult<T>) -> Result<T, Vec<Error>> {
    // TODO consider finish()
    match result {
        Ok((_, a)) => Ok(a),
        Err(nom::Err::Error(e)) | Err(nom::Err::Failure(e)) => Err(vec![error_from_input(
            &e.input,
            unexpected_input_message(&e.input),
        )]),
        Err(nom::Err::Incomplete(_)) => {
            panic!("Incomplete error on input");
        }
    }
}

/// Parses a program expression from `input`, stamping every span (and any
/// error's span) with `source`, the identity of `input`'s text.
pub fn parse_program<M, S: Copy>(input: &str, source: S) -> Result<SourceExpr<M, S>, Vec<Error<S>>>
where
    M: Display,
{
    parse_program_unstamped(input)
        .map(|expr| expr::stamp_expr(expr, source))
        .map_err(|errors| expr::stamp_errors(errors, source))
}

fn parse_program_unstamped<M>(input: &str) -> Result<SourceExpr<M>, Vec<Error>>
where
    M: Display,
{
    let errors = RefCell::new(Vec::new());
    let input = Input::new_extra(input, ParseState(&errors));
    #[rustfmt::skip]
    let result = all_consuming(
        ws(parse_expr),
    ).parse(input);
    if !errors.borrow().is_empty() {
        return Err(errors.into_inner());
    }
    translate_parse_result(result)
}

/// Parses a module (often a file) whose contents are `input`, yielding the
/// bindings defined by that module. Every span (including any error's) is
/// stamped with `source`, the identity of `input`'s text.
///
/// On success, returns a set of bindings that span the entire input, plus
/// any recoverable errors encountered while producing them.
pub fn parse_module<M, S: Copy>(
    input: &str,
    source: S,
) -> Result<ParsedModule<M, S>, Vec<Error<S>>> {
    match parse_module_unstamped(input) {
        Ok((bindings, errors)) => Ok((
            expr::stamp_bindings(bindings, source),
            expr::stamp_errors(errors, source),
        )),
        Err(errors) => Err(expr::stamp_errors(errors, source)),
    }
}

/// A module's parsed bindings, plus any recoverable errors encountered
/// while producing them.
pub type ParsedModule<M, S = ()> = (Vec<SourceBinding<M, S>>, Vec<Error<S>>);

fn parse_module_unstamped<M>(input: &str) -> Result<ParsedModule<M>, Vec<Error>> {
    let errors = RefCell::new(Vec::new());
    let input = Input::new_extra(input, ParseState(&errors));
    #[rustfmt::skip]
    let result = all_consuming((
        many0(
            terminated(
                // Note that parse_binding consumes surrounding trivia and annotations.
                parse_binding,
                tag(";"),
            )
        ),
        recognize(trivia0),
    )).parse(input);
    // Two things: 1) we need to extend each binding to include the ";" and 2)
    // we need to add an `Empty` binding at the end if there is any trailing
    // trivia.
    let result = match result {
        Ok((rest, (mut bindings, trivia))) => {
            for binding in bindings.iter_mut() {
                if let Some(span) = &binding.span {
                    // The ";" is always the next character.
                    binding.span = Some(Span::unstamped(span.range.start..span.range.end + 1));
                }
            }
            if trivia.len() > 0 {
                bindings.push(SourceBinding {
                    binding: Binding::Empty,
                    annotations: Vec::new(),
                    span: Some(Span::unstamped(
                        trivia.location_offset()..trivia.location_offset() + trivia.len(),
                    )),
                });
            }
            Ok((rest, (bindings, errors.take())))
        }
        Err(e) => Err(e),
    };
    translate_parse_result(result)
}

/// Parses a slider config entry.
fn parse_slider(input: Input) -> IResult<Slider> {
    // TODO this is a bit of a mess... and also could first parse the whole thing as a string literal.
    let label_char = |c: char| c != ':' && c != '"' && c != ',' && c != ']' && !c.is_whitespace();
    // Parse the quoted opening, label, and first float (shared prefix)
    #[rustfmt::skip]
    let (rest, (_, label, (initial_input, initial_value), _)) = (
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
            initial_input.extra.report_error(error_from_input(
                &initial_input,
                format!(
                    "initial value {} is not between min {} and max {}",
                    initial_value, min, max
                ),
            ));
            return Err(nom::Err::Failure(nom::error::Error::new(
                initial_input,
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

fn parse_sliders_internal(input: Input) -> IResult<Vec<Slider>> {
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
    let input = Input::new_extra(input, ParseState(&errors));
    let result = all_consuming(parse_sliders_internal).parse(input);
    if !errors.borrow().is_empty() {
        return Err(errors.into_inner());
    }
    translate_parse_result(result)
}

/// Parses `color=rgb(R,G,B)` where R, G, B are integers in [0, 255].
fn parse_color(input: Input) -> IResult<(u8, u8, u8)> {
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
fn parse_level(input: Input) -> IResult<f32> {
    let (rest, (_, value)) = (tag("level_db="), number::float).parse(input)?;
    Ok((rest, value))
}

/// Parses `skip_slots=<non-negative integer>` (e.g., `skip_slots=3`).
fn parse_skip_slots(input: Input) -> IResult<u32> {
    // TODO combinators
    let (rest, _) = tag("skip_slots=").parse(input)?;
    let (rest, digits) = character::digit1(rest)?;
    let value = digits
        .fragment()
        .parse::<u32>()
        .map_err(|_| nom::Err::Error(nom::error::Error::new(rest, nom::error::ErrorKind::Digit)))?;
    Ok((rest, value))
}

/// Parses a single annotation variant (one element from inside `#{ ... }`).
fn parse_annotation(input: Input) -> IResult<SourceAnnotation> {
    // TODO can we capture this start/end+wrap pattern as a combinator here and
    // elsewhere?
    let start = input.location_offset();
    let (rest, annotation) = alt((
        parse_sliders_internal.map(Annotation::Sliders),
        parse_color.map(|(r, g, b)| Annotation::Color(r, g, b)),
        parse_level.map(Annotation::Level),
        parse_skip_slots.map(Annotation::SkipSlots),
    ))
    .parse(input)?;
    let end = rest.location_offset();
    Ok((
        rest,
        SourceAnnotation {
            annotation,
            span: Some(Span::unstamped(start..end)),
        },
    ))
}

/// Parses a set of annotations `#{anno, anno, ...}` and returns the inner
/// annotations (each with its own span).
fn parse_annotation_set(input: Input) -> IResult<Vec<SourceAnnotation>> {
    // TODO combinators
    let (rest, _) = tag("#").parse(input)?;
    let (rest, _) = ws(char('{')).parse(rest)?;
    let (rest, annos) = separated_list0(ws(char(',')), parse_annotation).parse(rest)?;
    let (rest, _) = trivia0(rest)?;
    let (rest, _) = char('}').parse(rest)?;
    Ok((rest, annos))
}
/// Replaces `source[target_range]` with `new_text`, re-parse the resulting
/// source, and atomically update both `source` and `root` on success. On parse
/// failure neither is modified and the parse errors are returned.
///
/// The intended use is to find a node by walking `root`, grab the range its
/// `span`, and pass it as `target_range` here. After the call returns Ok, every
/// node in the new `root` has fresh spans into the new `source`, so
/// `print_preserving` round-trips the edited file.
pub fn replace_at<M>(
    root: &mut SourceExpr<M>,
    target_range: Range<usize>,
    new_text: &str,
    source: &mut String,
) -> Result<(), Vec<Error>>
where
    M: Display,
{
    if target_range.end > source.len() || target_range.start > target_range.end {
        return Err(vec![Error::with_span(
            format!(
                "target span {:?} is out of bounds for source of length {}",
                target_range,
                source.len()
            ),
            Some(Span::unstamped(target_range)),
        )]);
    }
    // Re-parses the whole expression rather than just the subtree —
    // simpler, keeps all spans consistent, and is plenty fast for tuun's
    // typical program-sized expressions. The subtree-only path is a future
    // optimization.

    // Speculatively splice and re-parse. If parsing fails, neither `source`
    // nor `root` is touched.
    let mut new_source = source.clone();
    new_source.replace_range(target_range, new_text);
    let new_root = parse_program_unstamped::<M>(&new_source)?;

    *source = new_source;
    *root = new_root;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{print_preserving, print_preserving_module};

    /// Parse `input`, format the AST, assert the result equals `expected`,
    /// and confirm the formatted form re-parses to the same Display output
    /// (round-trip stability).
    fn assert_round_trip(input: &str, expected: &str) {
        let parsed = parse_program_unstamped::<u32>(input)
            .unwrap_or_else(|errs| panic!("failed to parse {:?}: {:?}", input, errs));
        let displayed = format!("{}", parsed);
        assert_eq!(
            displayed, expected,
            "Display output didn't match expected canonical form\n input: {:?}",
            input
        );
        let parsed_again = parse_program_unstamped::<u32>(&displayed).unwrap_or_else(|errs| {
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
    fn test_parse_identifier_and_variable() {
        let result = parse_program_unstamped::<u32>("fn");
        assert!(result.is_err());
        let result = parse_program_unstamped::<u32>("_");
        assert!(result.is_err());

        assert_round_trip("my_var", "my_var");
        assert_round_trip("$", "$");
        assert_round_trip("_private", "_private");
        assert_round_trip("__chord", "__chord");

        // Double underscore identifiers are internal-only
        let errors = RefCell::new(Vec::new());
        let input = Input::new_extra("__chord", ParseState(&errors));
        let result = parse_identifier(input);
        assert!(result.is_err());
    }

    fn parse_module_successfully(input: &str) -> Vec<SourceBinding<u32>> {
        let result = parse_module_unstamped::<u32>(input);
        assert!(result.is_ok(), "got {:?}", result.err());
        let (bindings, errors) = result.unwrap();
        assert!(errors.is_empty(), "got {:?}", errors);
        bindings
    }

    #[test]
    fn test_parse_with_comments() {
        // Comments anywhere whitespace was previously allowed should be ignored.
        assert_round_trip("1 + // a comment\n 2", "1 + 2");

        // Comment before, between, and after bindings.
        let input = "
            // header comment
            x = 1; // trailing
            // standalone
            y = x + 1;
        ";
        let result = parse_module_successfully(input);
        // Three bindings, since there is an empty one to hold the trailing new-line.
        assert_eq!(result.len(), 3);

        // Comment inside a function body.
        assert_round_trip("fn(x) => x // identity\n", "fn(x) => x");

        // Comment between `let` keyword and bindings.
        let result = parse_program_unstamped::<u32>("let // bindings follow\n x = 1 in x");
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
        assert_round_trip("fn() => 1", "fn() => 1");
        // The parameter list requires parens.
        assert!(parse_program_unstamped::<u32>("fn x => x").is_err());
    }

    #[test]
    fn test_parse_call_arguments() {
        // A tuple literal is ONE argument, distinct from two arguments.
        assert_round_trip("f((1, 2))", "f((1, 2))");
        assert_round_trip("f(1, 2)", "f(1, 2)");
        assert_round_trip("f()", "f()");
        // A multi-parameter function literal applied to multiple arguments
        // is not expressible as `let` — it stays in call form.
        assert_round_trip("(fn(x, y) => x)(1, 2)", "(fn(x, y) => x)(1, 2)");
        // A multi-argument call of an application can't re-sugar to a pipe
        // (a pipe delivers exactly one argument).
        assert_round_trip("f(1)(2, 3)", "f(1)(2, 3)");
    }

    #[test]
    fn test_parse_let() {
        // `let` de-sugars to nested lambda applications, which Display
        // re-sugars back into the comma-separated `let` form.
        assert_round_trip("let x = 1 in x + 1", "let x = 1 in x + 1");
        // Multiple bindings, including tuple patterns.
        assert_round_trip(
            "let x = 1, (y, z) = (x + 1, 3) in 2 * y * z",
            "let x = 1, (y, z) = (x + 1, 3) in 2 * y * z",
        );
        // Trailing comma is not canonical.
        assert_round_trip("let x = 1, in x + 1", "let x = 1 in x + 1");
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
        let parsed = parse_program_unstamped::<u32>(input).unwrap();
        assert_eq!(print_preserving(&parsed, input), "1 + 2");

        // A line comment between the two operands of `+` falls inside the
        // outer Application's span and round-trips verbatim. (Display alone
        // would normalize whitespace to `1 + 2`.)
        let input = "1 + // a comment\n  2";
        let parsed = parse_program_unstamped::<u32>(input).unwrap();
        assert_eq!(print_preserving(&parsed, input), "1 + // a comment\n  2");

        // Multiple operators — same deal, just splice the outer span.
        let input = "(10 - 8 - 1) * 6 // total\n";
        let parsed = parse_program_unstamped::<u32>(input).unwrap();
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
        let synth: SourceExpr<u32> = SourceExpr::application(plus, vec![lhs, rhs]);
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
        let mut parsed = parse_program_unstamped::<u32>(input).unwrap();
        if let Expr::Application { arguments, .. } = &mut parsed.expr {
            arguments[0] = SourceExpr::float(99.0); // span: None — "edit"
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
    fn test_print_preserving_module_round_trip() {
        // Comment after the separator → belongs to the following binding (in
        // its span). Comment before the separator → belongs to the preceding
        // binding. Splicing each binding's span reproduces every comment.
        let input = "x = 1; // for y\n y = x + 1;";
        let bindings = parse_module_successfully(input);
        assert_eq!(bindings.len(), 2);
        assert_eq!(print_preserving_module(&bindings, input), input);

        // Comment trailing the first binding (before the `;`) is in x's span.
        let input = "x = 1 // for x\n; y = x + 1;";
        let bindings = parse_module_successfully(input);
        assert_eq!(bindings.len(), 2);
        assert_eq!(print_preserving_module(&bindings, input), input);

        // Mixed: header + trailing + leading + inter-binding blank line.
        let input = "x = 1;\n\n// section header\ny = x + 1;";
        let bindings = parse_module_successfully(input);
        assert_eq!(bindings.len(), 2);
        assert_eq!(print_preserving_module(&bindings, input), input);

        // A file-header comment at the very top is absorbed into the first
        // binding's span.
        let input = "// file header\n// also for the first binding\nx = 1;\ny = 2;";
        let bindings = parse_module_successfully(input);
        assert_eq!(bindings.len(), 2);
        assert_eq!(
            bindings[0].span.as_ref().map(|s| s.range.start),
            Some(0),
            "first binding should absorb file-leading"
        );
        assert_eq!(print_preserving_module(&bindings, input), input);
    }

    #[test]
    fn test_parse_module_accepts_trailing_comments() {
        // Trailing whitespace is fine (consumed silently by the terminator).
        assert!(parse_module_unstamped::<u32>("x = 1;\n").is_ok());
        assert!(parse_module_unstamped::<u32>("x = 1; y = 2; \n  ").is_ok());

        // Trailing `//` comments at end of file become an extra `Empty`
        // binding so their span round-trips through
        // `print_preserving_module`.
        let input = "x = 1;\n// trailing";
        let bindings = parse_module_successfully(input);
        assert_eq!(bindings.len(), 2);
        assert!(matches!(bindings[1].binding, Binding::Empty));
        assert_eq!(print_preserving_module(&bindings, input), input);

        let input = "x = 1; // trailing";
        let bindings = parse_module_successfully(input);
        assert_eq!(bindings.len(), 2);
        assert!(matches!(bindings[1].binding, Binding::Empty));
        assert_eq!(print_preserving_module(&bindings, input), input);
    }

    #[test]
    fn test_unparseable_definition_body_reports_recoverable_error() {
        // A definition whose body fails to parse recovers by consuming up
        // to the `;`, but must still report a recoverable error — an
        // `Expr::Error` node with an empty error list would let the broken
        // source pass for a clean parse.
        let (bindings, errors) = parse_module_unstamped::<u32>("x = ;\n").unwrap();
        assert_eq!(errors.len(), 1, "expected one error, got {:?}", errors);
        assert_eq!(errors[0].message(), "expected expression in definition");
        assert_eq!(errors[0].range(), Some(4..4));
        assert!(matches!(
            &bindings[0].binding,
            Binding::Definition(_, expr) if matches!(expr.expr, Expr::Error(_))
        ));

        // Same for a non-empty unparseable body; the error covers the
        // skipped text.
        let (_, errors) = parse_module_unstamped::<u32>("x = );\n").unwrap();
        assert_eq!(errors.len(), 1, "expected one error, got {:?}", errors);
        assert_eq!(errors[0].message(), "expected expression in definition");
        assert_eq!(errors[0].range(), Some(4..5));
    }

    #[test]
    fn test_module_error_recovery() {
        let input = "\
#{color=rgb(255,0,128)}
// just a comment, not an annotation
_x = )(obviously not parsable, but ends with ;
y = 2;";
        // Recovery consumes the broken body up to the `;` and reports it
        // as a recoverable error; the surrounding bindings still parse.
        let (bindings, errors) = parse_module_unstamped::<u32>(input).unwrap();
        assert_eq!(errors.len(), 1, "expected one error, got {:?}", errors);
        assert_eq!(errors[0].message(), "expected expression in definition");
        assert_eq!(bindings.len(), 2);
        assert!(matches!(
            bindings[0].annotations[0].annotation,
            Annotation::Color(255, 0, 128)
        ));
        assert!(matches!(bindings[1].binding, Binding::Definition(..)));
        assert_eq!(bindings[1].annotations.len(), 0);
        assert_eq!(print_preserving_module(&bindings, input), input);
    }

    #[test]
    fn test_annotations_attach_to_following_binding() {
        let input = "\
#{color=rgb(255,0,128)}
#{level_db=-6}
// just a comment, not an annotation
x = 1;
y = 2;";
        let bindings = parse_module_successfully(input);
        assert_eq!(bindings.len(), 2);

        // [0] is `x = 1` with all preceding annotations attached.
        assert!(matches!(bindings[0].binding, Binding::Definition(..)));
        assert_eq!(bindings[0].annotations.len(), 2);
        assert!(matches!(
            bindings[0].annotations[0].annotation,
            Annotation::Color(255, 0, 128)
        ));
        assert!(matches!(
            bindings[0].annotations[1].annotation,
            Annotation::Level(_)
        ));

        // [1] is `y = 2` with no annotation.
        assert!(matches!(bindings[1].binding, Binding::Definition(..)));
        assert_eq!(bindings[1].annotations.len(), 0);

        // Annotation spans still point into source.
        for binding in &bindings {
            for sa in &binding.annotations {
                let span = sa.span.as_ref().expect("annotation should have a span");
                assert!(span.range.end <= input.len());
            }
        }

        // Full round-trip through `print_preserving_module`.
        assert_eq!(print_preserving_module(&bindings, input), input);
    }

    #[test]
    fn test_parse_skip_slots_annotation() {
        // `skip_slots=N` is a non-negative UI-gap annotation. Survives
        // round-trip through `print_preserving_module`.
        let input = "\
kick = pulse(60);
#{skip_slots=7, color=rgb(0,128,255)}
synth = saw(220);";
        let bindings = parse_module_successfully(input);
        assert_eq!(bindings.len(), 2);
        let skip0 = bindings[0]
            .annotations
            .iter()
            .find_map(|a| match &a.annotation {
                Annotation::SkipSlots(n) => Some(*n),
                _ => None,
            });
        assert_eq!(skip0, None);
        let skip1 = bindings[1]
            .annotations
            .iter()
            .find_map(|a| match &a.annotation {
                Annotation::SkipSlots(n) => Some(*n),
                _ => None,
            });
        assert_eq!(skip1, Some(7));
        assert_eq!(print_preserving_module(&bindings, input), input);
    }

    #[test]
    fn test_parse_open_bindings() {
        // `open path.to.module` is a binding with a `.`-separated module
        // path. It coexists with `Definition`s and pending annotations.
        let input = "\
open foo;
open bar.baz;
#{color=rgb(0,0,0)}
open util.synths;
x = 1;";
        let bindings = parse_module_successfully(input);
        assert_eq!(bindings.len(), 4);

        match &bindings[0].binding {
            Binding::Open(p) => assert_eq!(p, &vec!["foo".to_string()]),
            other => panic!("expected Open for [0], got {:?}", other),
        }
        match &bindings[1].binding {
            Binding::Open(p) => {
                assert_eq!(p, &vec!["bar".to_string(), "baz".to_string()]);
            }
            other => panic!("expected Open for [1], got {:?}", other),
        }
        // Pending annotation attaches to the next binding even when that
        // binding is an `open`.
        match &bindings[2].binding {
            Binding::Open(p) => {
                assert_eq!(p, &vec!["util".to_string(), "synths".to_string()]);
            }
            other => panic!("expected Open for [2], got {:?}", other),
        }
        assert_eq!(bindings[2].annotations.len(), 1);

        assert!(matches!(bindings[3].binding, Binding::Definition(..)));
        // Full round-trip.
        assert_eq!(print_preserving_module(&bindings, input), input);
    }

    #[test]
    fn test_print_preserving_module_round_trip_with_annotations() {
        let input = "\
#{color=rgb(255,0,128)}
#{level_db=-6}
kick = pulse(60);
#{sliders=[\"cutoff:800:200:4000\"]}
synth = saw(220);";
        let bindings = parse_module_successfully(input);
        assert_eq!(bindings.len(), 2);
        assert!(matches!(bindings[0].binding, Binding::Definition(..)));
        assert_eq!(bindings[0].annotations.len(), 2);
        assert!(matches!(bindings[1].binding, Binding::Definition(..)));
        assert_eq!(bindings[1].annotations.len(), 1);
        assert_eq!(print_preserving_module(&bindings, input), input);
    }

    #[test]
    fn test_dirty_annotation_falls_back_to_structural() {
        // Clear an annotation's span (simulating an in-memory mutation
        // that hasn't been written back through `replace_at`). The whole
        // binding now needs a structural re-emit — annotation lines + the
        // binding's own structural form.
        let input = "#{color=rgb(10,20,30)}\nx = 1;";
        let mut bindings = parse_module_successfully(input);
        assert_eq!(bindings.len(), 1);
        assert_eq!(bindings[0].annotations.len(), 1);

        // Mutate: bump the color and clear its span.
        bindings[0].annotations[0].annotation = Annotation::Color(99, 99, 99);
        bindings[0].annotations[0].span = None;

        let out = print_preserving_module(&bindings, input);
        // Structural fallback re-emits the annotation with its new value.
        assert!(out.contains("color=rgb(99,99,99)"), "got {:?}", out);
        // The binding's content survives via the expression printer.
        assert!(out.contains("x = 1"), "got {:?}", out);
    }

    /// Walks `node` and returns the span of the first `List` it finds, or
    /// None. Used by replace_at tests as a stand-in for editor navigation.
    fn find_first_list_span<M>(node: &SourceExpr<M>) -> Option<Range<usize>> {
        if let Expr::List(_) = &node.expr {
            return node.span.as_ref().map(|s| s.range.clone());
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
            Expr::Application {
                function,
                arguments,
            } => find_first_list_span(function)
                .or_else(|| arguments.iter().find_map(find_first_list_span)),
            Expr::Tuple(items) | Expr::List(items) => items.iter().find_map(find_first_list_span),
            _ => None,
        }
    }

    #[test]
    fn test_replace_at_list_literal() {
        let mut source = String::from("on_beats(b, [0, 1, 2, 3])");
        let mut root = parse_program_unstamped::<u32>(&source).unwrap();
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
        let mut root = parse_program_unstamped::<u32>(&source).unwrap();
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
        let original_root = parse_program_unstamped::<u32>(&source).unwrap();
        let mut root = original_root.clone();
        let list_span = root.span.clone().unwrap().range;

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
        let mut root = parse_program_unstamped::<u32>(&source).unwrap();
        let bad_span = 0..1000;
        let result = replace_at(&mut root, bad_span, "x", &mut source);
        assert!(result.is_err());
        assert_eq!(source, "[1, 2]");
    }

    #[test]
    fn test_print_preserving_module_dirty_binding_splices_others() {
        // Edit one binding's RHS (replace Float(1) with Float(99) — no span)
        // and confirm: the dirty binding emits structurally (`x = 99`), but
        // the other binding still splices its leading comment verbatim.
        let input = "x = 1;\n// leading for y\n y = 2;";
        let mut bindings = parse_module_successfully(input);
        match &mut bindings[0].binding {
            Binding::Definition(_pattern, expr) => *expr = SourceExpr::float(99.0),
            _ => panic!("expected first binding to be a Definition"),
        }
        let out = print_preserving_module(&bindings, input);
        // x is dirty → structural: "x = 99". y is clean → splices its span
        // (which includes the leading comment).
        assert!(out.starts_with("x = 99"), "got {:?}", out);
        assert!(out.contains("// leading for y"), "got {:?}", out);
        assert!(out.contains("y = 2"), "got {:?}", out);
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

    /// Test helper: run `parse_annotation_set` on `input` and either
    /// return the flat list of `Annotation`s or surface the parse error.
    fn parse_one_annotation_set(input: &str) -> Result<Vec<Annotation>, Vec<Error>> {
        let errors = RefCell::new(Vec::new());
        let input = Input::new_extra(input, ParseState(&errors));
        let result = all_consuming(parse_annotation_set).parse(input);
        if !errors.borrow().is_empty() {
            return Err(errors.into_inner());
        }
        match result {
            Ok((_, annos)) => Ok(annos.into_iter().map(|sa| sa.annotation).collect()),
            Err(_) => Err(vec![Error::new("parse failed".to_string())]),
        }
    }

    #[test]
    fn test_parse_annotation_set() {
        // Single slider annotation.
        let annos = parse_one_annotation_set(r#"#{sliders=["volume:0.75:0:1"]}"#).unwrap();
        assert_eq!(annos.len(), 1);
        assert!(matches!(annos[0], Annotation::Sliders(_)));

        // Two annotations on one line.
        let annos =
            parse_one_annotation_set(r#"#{sliders=["volume:0.75:0:1"],skip_slots=2}"#).unwrap();
        assert_eq!(annos.len(), 2);
        assert!(matches!(annos[1], Annotation::SkipSlots(2)));

        // Missing closing brace.
        assert!(parse_one_annotation_set("#{skip_slots=1").is_err());
        // Unknown key.
        assert!(parse_one_annotation_set("#{bad_key=[]}").is_err());
        // Not an annotation line at all.
        assert!(parse_one_annotation_set("// regular comment").is_err());
        assert!(parse_one_annotation_set("let x = 1").is_err());

        // Color.
        let annos = parse_one_annotation_set("#{color=rgb(255,0 , 128)}").unwrap();
        assert!(matches!(annos[0], Annotation::Color(255, 0, 128)));
        // Color out of range / missing component / non-integer.
        assert!(parse_one_annotation_set("#{color=rgb(256,0,0)}").is_err());
        assert!(parse_one_annotation_set("#{color=rgb(0,0)}").is_err());
        assert!(parse_one_annotation_set("#{color=rgb(1.5,0,0)}").is_err());

        // Level.
        let annos = parse_one_annotation_set("#{level_db=-6.0}").unwrap();
        assert!(matches!(annos[0], Annotation::Level(v) if (v - -6.0).abs() < 0.01));

        let annos = parse_one_annotation_set("#{level_db=6.0}").unwrap();
        assert!(matches!(annos[0], Annotation::Level(v) if (v - 6.0).abs() < 0.01));

        // SkipSlots alone.
        let annos = parse_one_annotation_set("#{skip_slots=5}").unwrap();
        assert!(matches!(annos[0], Annotation::SkipSlots(5)));
    }
}
