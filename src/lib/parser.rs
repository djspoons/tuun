use core::panic;
use std::fmt;
use std::ops::Range;
use std::{cell::RefCell, rc::Rc};

use nom::{
    Parser,
    branch::alt,
    bytes::complete::{tag, take_while, take_while1},
    character::complete::{alpha1, alphanumeric1, char, multispace0, multispace1},
    combinator::{all_consuming, map, not, opt, peek, recognize, verify},
    multi::{many0, separated_list0},
    number::complete::float,
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

impl<'a> Error {
    pub fn new(message: String) -> Self {
        Self {
            range: None,
            message,
        }
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

impl ToString for Error {
    fn to_string(&self) -> String {
        match &self.range {
            Some(range) => format!("{} at {}..{}", self.message, range.start, range.end),
            None => self.message.clone(),
        }
    }
}

/// Carried around in the `LocatedSpan::extra` field in
/// between `nom` parsers.
#[derive(Clone, Debug)]
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

// https://github.com/Geal/nom/blob/main/doc/nom_recipes.md#wrapper-combinators-that-eat-whitespace-before-and-after-a-parser
fn ws<'a, F, T, E: nom::error::ParseError<LocatedSpan<'a>>>(
    inner: F,
) -> impl Parser<LocatedSpan<'a>, Output = T, Error = E>
where
    F: Parser<LocatedSpan<'a>, Output = T, Error = E>,
{
    delimited(multispace0, inner, multispace0)
}

#[derive(Clone)]
pub struct BuiltInFn<M>(pub Rc<dyn Fn(Vec<Expr<M>>) -> Expr<M>>);

impl<M> std::fmt::Debug for BuiltInFn<M> {
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
        body: Box<Expr<M>>,
    },
    BuiltIn {
        name: String,
        // Pure functions from a vector of values to a value
        function: BuiltInFn<M>,
    },
    // A sequence-able waveform. In value form, both offset and waveform are Expr::Waveform.
    Seq {
        offset: Box<Expr<M>>,
        waveform: Box<Expr<M>>,
    },
    // If-Then-Else expression
    IfThenElse {
        condition: Box<Expr<M>>,
        then: Box<Expr<M>>,
        else_: Box<Expr<M>>,
    },
    // Function application
    Variable(String),
    Application {
        function: Box<Expr<M>>,
        argument: Box<Expr<M>>,
    },
    // Compound expressions
    Tuple(Vec<Expr<M>>),
    List(Vec<Expr<M>>),
    // Errors
    Error(String),
}

// TODO use this in 'expect'?
impl<M> Default for Expr<M> {
    fn default() -> Self {
        Expr::Error("_".to_string())
    }
}

fn parse_string<M>(input: LocatedSpan) -> IResult<Expr<M>> {
    #[rustfmt::skip]
    let (rest, value) = delimited(
        char('"'),
        // TODO implement escaping
        take_while(|c: char| c != '"'),
        char('"'),
    ).parse(input)?;
    return Ok((rest, Expr::String(value.fragment().to_string())));
}

fn parse_literal<M>(input: LocatedSpan) -> IResult<Expr<M>> {
    #[rustfmt::skip]
    let (rest, value) = alt((
        // Handle parsing negative floats ourselves
        preceded(
            not(peek(char('-'))),
            float,
        ).map(Expr::Float),
        parse_string,
    )).parse(input)?;
    return Ok((rest, value));
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
    return Ok((rest, value.to_string()));
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
    return Ok((rest, value));
}

fn parse_pattern(input: LocatedSpan) -> IResult<Pattern> {
    #[rustfmt::skip]
    let (rest, pattern) =
        alt((
            parse_identifier.map(Pattern::Identifier),
            delimited(
                (char('('), multispace0),
                separated_list0(
                    ws(char(',')),
                    parse_pattern),
                (multispace0, expect(char(')'), "expected ')' at end of tuple pattern")),
            ).map(Pattern::Tuple),
        )).parse(input)?;
    return Ok((rest, pattern));
}

fn parse_function<M>(input: LocatedSpan) -> IResult<Expr<M>> {
    #[rustfmt::skip]
    let (rest, expr) =
        ((delimited(
            (tag("fn"), multispace0),
            alt((
                delimited((char('('), multispace0),
                        parse_identifier.map(Pattern::Identifier),
                        (multispace0, char(')'))),
                parse_pattern,
            )),
            ws(expect(tag("=>"), "expected '=>'"))),
            parse_expr,
        )).map(|(pattern, body)| {
            Expr::Function { pattern, body: Box::new(body) }
        }).parse(input)?;
    return Ok((rest, expr));
}

fn parse_bindings<M>(input: LocatedSpan) -> IResult<Vec<(Pattern, Expr<M>)>> {
    #[rustfmt::skip]
    let (rest, bindings) =
        // TODO maybe don't allow just a comma?
        terminated(
            separated_list0(
                ws(char(',')),
                (parse_pattern,
                   ws(char('=')),
                   parse_expr,
                ).map(|(pattern, _, expr)| (pattern, expr))
            ),
            opt((multispace0, char(','))),
        ).parse(input)?;
    return Ok((rest, bindings));
}

pub fn make_let<M>(bindings: Vec<(Pattern, Expr<M>)>, mut expr: Expr<M>) -> Expr<M> {
    for (pattern, binding) in bindings.into_iter().rev() {
        expr = Expr::Application {
            function: Box::new(Expr::Function {
                pattern,
                body: Box::new(expr),
            }),
            argument: Box::new(binding),
        }
    }
    expr
}

fn parse_let<M>(input: LocatedSpan) -> IResult<Expr<M>> {
    #[rustfmt::skip]
    let (rest, expr) =
        (delimited(
            (tag("let"), multispace1),
            parse_bindings,
            (multispace1, expect(tag("in"), "expected 'in'"), multispace1),
        ),
        expect(parse_expr, "expected expression after 'in'")
        ).map(|(bindings, expr)| {
            let expr = expr.unwrap_or_default();
            make_let(bindings, expr)
        }).parse(input)?;
    return Ok((rest, expr));
}

fn parse_if_then_else<M>(input: LocatedSpan) -> IResult<Expr<M>> {
    #[rustfmt::skip]
    let (rest, expr) =
        (delimited(
            (tag("if"), multispace1),
            parse_expr,
            delimited(multispace1, tag("then"), multispace1),
        ),
        parse_expr,
        delimited(multispace1, tag("else"), multispace1),
        parse_expr,
        ).map(|(condition, then, _, else_)| {
            Expr::IfThenElse {
                condition: Box::new(condition),
                then: Box::new(then),
                else_: Box::new(else_),
            }
        }).parse(input)?;
    return Ok((rest, expr));
}

fn parse_primitive<M>(input: LocatedSpan) -> IResult<Expr<M>> {
    #[rustfmt::skip]
    let (rest, value) = alt((
        parse_literal,
        parse_function,
        parse_let,
        parse_if_then_else,
        // Should come before identifiers, since operators also match the identifier rule
        (parse_unary_operator, parse_primitive).map(
            |(op, expr)| Expr::Application {
                function: Box::new(Expr::Variable(op.fragment().to_string())),
                argument: Box::new(expr),
            },
        ),
        parse_identifier.map(Expr::Variable),
        parse_chord,
        parse_sequence,
        parse_tuple,
        parse_list
    )).parse(input)?;
    return Ok((rest, value));
}

fn parse_application<M>(input: LocatedSpan) -> IResult<Expr<M>> {
    #[rustfmt::skip]
    let (rest, expr) = map(
        (
            parse_primitive,
            many0(preceded(
                multispace0,
                parse_tuple,
            )),
        ),
        |(func, exprs)| {
            let mut result = func;
            for expr in exprs  {
                result = Expr::Application {
                    function: Box::new(result),
                    argument: Box::new(expr),
                };
            }
            return result;
        },
    ).parse(input)?;
    return Ok((rest, expr));
}

fn parse_multiplicative<M>(input: LocatedSpan) -> IResult<Expr<M>> {
    #[rustfmt::skip]
    let (rest, value) = map(
        (
            parse_application,
            many0((
                ws(alt((tag("*"), tag("/"),tag("~*")))),
                expect(parse_application, "expected expression after operator"),
            )),
        ),
        |(factor, op_factors)| {
            let mut expr = factor;
            for (op, factor) in op_factors {
                expr = Expr::Application {
                    function: Box::new(Expr::Variable(op.fragment().to_string())),
                    argument: Box::new(Expr::Tuple(vec![expr, factor.unwrap_or_default()])),
                };
            }
            return expr;
        },
    ).parse(input)?;
    return Ok((rest, value));
}

fn parse_additive<M>(input: LocatedSpan) -> IResult<Expr<M>> {
    #[rustfmt::skip]
    let (rest, value) = map(
        (
            parse_multiplicative,
            many0((
                ws(alt((tag("+"), tag("-"), tag("&")))),
                expect(parse_multiplicative, "expected expression after operator"),
            )),
        ),
        |(term, op_terms)| {
            let mut expr = term;
            for (op, term) in op_terms {
                expr = Expr::Application {
                    function: Box::new(Expr::Variable(op.fragment().to_string())),
                    argument: Box::new(Expr::Tuple(vec![expr, term.unwrap_or_default()])),
                };
            }
            return expr;
        },
    ).parse(input)?;
    return Ok((rest, value));
}

fn parse_relational<M>(input: LocatedSpan) -> IResult<Expr<M>> {
    #[rustfmt::skip]
    let (rest, value) = map(
        (
            parse_additive,
            many0((
                ws(alt((tag("=="), tag("!="), tag("<="), tag(">="), tag("<"), tag(">")))),
                parse_additive,
            )),
        ),
        |(mut expr, op_exprs)| {
            for (op, term) in op_exprs {
                expr = Expr::Application {
                    function: Box::new(Expr::Variable(op.fragment().to_string())),
                    argument: Box::new(Expr::Tuple(vec![expr, term])),
                };
            }
            return expr;
        },
    ).parse(input)?;
    return Ok((rest, value));
}

fn parse_chord<M>(input: LocatedSpan) -> IResult<Expr<M>> {
    #[rustfmt::skip]
    let (rest, expr) = delimited(
        (char('{'), multispace0),
        parse_expr,
        (multispace0, expect(char('}'), "expected '}' at end of chord")),
    ).parse(input)?;
    return Ok((
        rest,
        Expr::Application {
            function: Box::new(Expr::Variable(("_chord").to_string())),
            argument: Box::new(expr),
        },
    ));
}

fn parse_sequence<M>(input: LocatedSpan) -> IResult<Expr<M>> {
    #[rustfmt::skip]
    let (rest, expr) = delimited(
        (char('<'), multispace0),
        parse_expr,
        (multispace0, expect(char('>'), "expected '>' at end of sequence")),
    ).parse(input)?;
    return Ok((
        rest,
        Expr::Application {
            function: Box::new(Expr::Variable(("_sequence").to_string())),
            argument: Box::new(expr),
        },
    ));
}

/// Parses a parenthesized expression or expressions. If there is only one element, returns just that element;
/// otherwise returns a Tuple of elements.
fn parse_tuple<M>(input: LocatedSpan) -> IResult<Expr<M>> {
    #[rustfmt::skip]
    let (rest, mut exprs) = delimited(
        (char('('), multispace0),
        separated_list0(
            (multispace0, char(','), multispace0),
            parse_expr,
        ),
        (multispace0, expect(char(')'), "expected ')' at end of tuple")),
    ).parse(input)?;
    if exprs.len() == 1 {
        return Ok((rest, exprs.pop().unwrap()));
    }
    return Ok((rest, Expr::Tuple(exprs)));
}

fn parse_list<M>(input: LocatedSpan) -> IResult<Expr<M>> {
    #[rustfmt::skip]
    let (rest, exprs) = delimited(
        (char('['), multispace0),
        separated_list0(
            (multispace0, char(','), multispace0),
            parse_expr,
        ),
        (multispace0, expect(char(']'), "expected ']' at end of list")),
    ).parse(input)?;
    return Ok((rest, Expr::List(exprs)));
}

fn parse_reverse_application<M>(input: LocatedSpan) -> IResult<Expr<M>> {
    #[rustfmt::skip]
    let (rest, expr) = map(
        (
            parse_relational,
            many0(preceded(
                    ws(char('|')),
                    expect(parse_relational, "expected expression after | operator"),
                )),
        ),
        |(argument, fn_exprs)| {
            let mut expr = argument;
            for fn_expr in fn_exprs {
                match fn_expr {
                    Some(function) => {
                        expr = Expr::Application {
                            function: Box::new(function),
                            argument: Box::new(expr),
                        }
                    }
                    None => {
                        expr = Expr::default();
                    }
                }
            }
            return expr;
        }
    ).parse(input)?;
    return Ok((rest, expr));
}

fn parse_expr<M>(input: LocatedSpan) -> IResult<Expr<M>> {
    #[rustfmt::skip]
    let (rest, expr) = map(
        (
            parse_reverse_application,
            many0(preceded(
                    ws(char('\\')),
                    expect(parse_reverse_application, "expected expression after \\ operator"),
                )),
        ),
        |(mut expr, op_exprs)| {
            for op_expr in op_exprs {
                expr = Expr::Application {
                    function: Box::new(Expr::Variable("\\".to_string())),
                    argument: Box::new(Expr::Tuple(vec![expr, op_expr.unwrap_or_default()])),
                };
            }
            return expr;
        }
    ).parse(input)?;
    return Ok((rest, expr));
}

fn translate_parse_result<T>(result: IResult<T>) -> Result<T, Vec<Error>> {
    match result {
        Ok((_, a)) => {
            return Ok(a);
        }
        Err(nom::Err::Error(e)) => {
            //println!("Error on parsing input: {:?}", e);
            return Err(vec![Error::new_from_span(
                &e.input,
                "unable to parse input".to_string(),
            )]);
        }
        Err(nom::Err::Incomplete(_)) => {
            panic!("Incomplete error on input");
        }
        Err(nom::Err::Failure(e)) => {
            println!("Failed to parse input: {:?}", e);
            return Err(vec![Error::new_from_span(
                &e.input,
                "unable to parse input".to_string(),
            )]);
        }
    }
}

pub fn parse_program<M>(input: &str) -> Result<Expr<M>, Vec<Error>>
where
    M: fmt::Display,
{
    let errors = RefCell::new(Vec::new());
    let span = LocatedSpan::new_extra(input, ParseState(&errors));
    #[rustfmt::skip]
    let result = all_consuming(
        ws(parse_expr),
    ).parse(span);
    if errors.borrow().len() > 0 {
        println!(
            "Got result {:} and errors {:?}",
            match result {
                Ok((_, node)) => node,
                _ => Expr::default(),
            },
            errors.borrow()
        );
        return Err(errors.into_inner());
    }
    translate_parse_result(result)
}

pub fn parse_context<M>(input: &str) -> Result<Vec<(Pattern, Expr<M>)>, Vec<Error>> {
    let errors = RefCell::new(Vec::new());
    let span = LocatedSpan::new_extra(input, ParseState(&errors));
    let result = all_consuming(ws(parse_bindings)).parse(span);
    if errors.borrow().len() > 0 {
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
            (peek(recognize(float)), expect(float, "expected initial value"))),
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
            expect(float, "expected minimum slider value"),
            preceded(
                expect(char(':'), "expected ':'"),
                expect(float, "expected maximum slider value")),
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

/// Parses a bracket-delimited, comma-separated list of slider configs.
/// Example: `["volume:0.5:0:1", "cutoff:2000:200:8000"]`
pub fn parse_sliders(input: &str) -> Result<Vec<Slider>, Vec<Error>> {
    let errors = RefCell::new(Vec::new());
    let span = LocatedSpan::new_extra(input, ParseState(&errors));

    let result = delimited(
        char('['),
        separated_list0(char(','), ws(parse_slider)),
        char(']'),
    )
    .parse(span);
    if errors.borrow().len() > 0 {
        return Err(errors.into_inner());
    }
    translate_parse_result(result)
}

fn trim_annotation(line: &str) -> Option<&str> {
    let trimmed = line.trim();
    let rest = trimmed.strip_prefix("//#{")?.strip_suffix('}')?;
    Some(rest.strip_prefix("sliders=")?.trim())
}

/// Parses a slider pragma line like `//#{sliders=[...]}`.
pub fn parse_annotation(line: &str) -> Result<Vec<Slider>, Vec<Error>> {
    let rest = trim_annotation(line).unwrap_or("");
    parse_sliders(rest)
}

/// Extends the context with a binding for each identifier in the pattern that is bound to
/// itself.
fn extend_with_trivial_context<M>(context: &mut Vec<(String, Expr<M>)>, pattern: &Pattern) {
    match pattern {
        Pattern::Identifier(name) => {
            context.push((name.clone(), Expr::Variable(name.clone())));
        }
        Pattern::Tuple(patterns) => {
            for pattern in patterns {
                extend_with_trivial_context(context, pattern);
            }
        }
    }
}

fn substitute<M>(context: &Vec<(String, Expr<M>)>, expr: Expr<M>) -> Expr<M>
where
    M: Clone,
{
    use Expr::{
        Application, Bool, BuiltIn, Float, Function, IfThenElse, List, String, Tuple, Variable,
    };
    match expr {
        Bool(_) | Float(_) | String(_) => expr,
        Expr::Waveform(waveform) => Expr::Waveform(waveform),
        Expr::Seq { offset, waveform } => Expr::Seq {
            offset: Box::new(substitute(context, *offset)),
            waveform: Box::new(substitute(context, *waveform)),
        },
        Function { pattern, body } => {
            let mut context = context.clone();
            extend_with_trivial_context(&mut context, &pattern);
            let body = substitute(&context, *body);
            Expr::Function {
                pattern,
                body: Box::new(body),
            }
        }
        BuiltIn { .. } => expr,
        Variable(name) => {
            for (var_name, value) in context.iter().rev() {
                if var_name == &name {
                    return value.clone();
                }
            }
            Expr::Error(format!("Variable '{}' not found in context", name))
        }
        IfThenElse {
            condition,
            then,
            else_,
        } => {
            let condition = Box::new(substitute(context, *condition));
            let then = Box::new(substitute(context, *then));
            let else_ = Box::new(substitute(context, *else_));
            IfThenElse {
                condition,
                then,
                else_,
            }
        }
        Application { function, argument } => {
            let function = substitute(context, *function);
            let argument = substitute(context, *argument);
            Expr::Application {
                function: Box::new(function),
                argument: Box::new(argument),
            }
        }
        Tuple(exprs) => Expr::Tuple(exprs.into_iter().map(|e| substitute(context, e)).collect()),
        List(exprs) => Expr::List(exprs.into_iter().map(|e| substitute(context, e)).collect()),
        Expr::Error(_) => expr,
    }
}

fn fmt_with_parens<M>(expr: &Expr<M>, f: &mut fmt::Formatter) -> fmt::Result
where
    M: fmt::Display,
{
    match expr {
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
                write!(f, "fn ")?;
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
                fmt_with_parens(function, f)?;
                if let Expr::Tuple(_) = &**argument {
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

pub fn extend_context<M>(
    context: &mut Vec<(String, Expr<M>)>,
    pattern: &Pattern,
    argument: &Expr<M>,
) -> Result<(), Error>
where
    M: Clone + fmt::Debug,
{
    match (pattern, argument) {
        (Pattern::Identifier(name), argument) => {
            /*
            println!("  {} = {}", name, argument);
            */
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
            pattern, argument
        ))),
    }
}

fn evaluate_closed<M>(expr: Expr<M>) -> Result<Expr<M>, Error>
where
    M: Clone + fmt::Display + fmt::Debug,
{
    use Expr::{
        Application, Bool, BuiltIn, Float, Function, IfThenElse, List, Seq, String, Tuple,
        Variable, Waveform,
    };
    match expr {
        Bool(_) | Float(_) | String(_) | Waveform(_) | Function { .. } => Ok(expr),
        Variable(name) => Err(Error::new(format!(
            "Variable '{}' not found in context",
            name
        ))),
        Seq { offset, waveform } => {
            let offset = evaluate_closed(*offset)?;
            let waveform = evaluate_closed(*waveform)?;
            Ok(Seq {
                offset: Box::new(offset),
                waveform: Box::new(waveform),
            })
        }
        BuiltIn { .. } => Ok(expr),
        IfThenElse {
            condition,
            then,
            else_,
        } => match evaluate_closed(*condition)? {
            Bool(true) => return evaluate_closed(*then),
            Bool(false) => return evaluate_closed(*else_),
            _ => return Err(Error::new("Expected boolean condition".to_string())),
        },
        Application { function, argument } => {
            let function = evaluate_closed(*function)?;
            let argument = evaluate_closed(*argument)?;
            match (function, argument) {
                (Function { pattern, body }, argument) => {
                    let mut context = Vec::new();
                    extend_context(&mut context, &pattern, &argument)?;
                    let body = substitute(&context, *body);
                    evaluate_closed(body)
                }
                (BuiltIn { function, .. }, Tuple(actuals)) => {
                    let result = function.0(actuals);
                    return match result {
                        Expr::Error(s) => Err(Error::new(s)),
                        _ => Ok(result),
                    };
                }
                (BuiltIn { function, .. }, actual) => {
                    let result = function.0(vec![actual]);
                    return match result {
                        Expr::Error(s) => Err(Error::new(s)),
                        _ => Ok(result),
                    };
                }
                (function, actuals) => {
                    return Err(Error::new(format!(
                        "Invalid application: {} {}",
                        function, actuals
                    )));
                }
            }
        }
        Tuple(exprs) => {
            return Ok(Tuple(
                exprs
                    .into_iter()
                    .map(|e| evaluate_closed(e))
                    .collect::<Result<Vec<_>, _>>()?,
            ));
        }
        List(exprs) => {
            return Ok(List(
                exprs
                    .into_iter()
                    .map(|e| evaluate_closed(e))
                    .collect::<Result<Vec<_>, _>>()?,
            ));
        }
        Expr::Error(s) => Err(Error::new(s)),
    }
}

pub fn evaluate<M>(context: &Vec<(String, Expr<M>)>, mut expr: Expr<M>) -> Result<Expr<M>, Error>
where
    M: Clone + fmt::Display + fmt::Debug,
{
    expr = substitute(context, expr);
    return evaluate_closed(expr);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_variable() {
        let input = "fn";
        let result = parse_program::<u32>(input);
        assert!(result.is_err());

        let input = "my_var";
        let result = parse_program::<u32>(input);
        assert!(result.is_ok());
        assert_eq!(format!("{}", result.unwrap()), "my_var");

        let input = "$";
        let result = parse_program::<u32>(input);
        assert!(result.is_ok());
        assert_eq!(format!("{}", result.unwrap()), "$");
    }

    #[test]
    fn test_parse_arithmetic() {
        let input = "(10 - 8 - 1) * 6";
        let result = parse_program::<u32>(input);
        assert!(result.is_ok());
        assert_eq!(format!("{}", result.unwrap()), "*(-(-(10, 8), 1), 6)");

        let input = "1 + 2 * 3.5 * 8 + 10";
        let result = parse_program::<u32>(input);
        assert!(result.is_ok());
        assert_eq!(
            format!("{}", result.unwrap()),
            "+(+(1, *(*(2, 3.5), 8)), 10)"
        );
    }

    #[test]
    fn test_parse_chord() {
        let input = "{[$x, $y, $z]}";
        let result = parse_program::<u32>(input);
        assert!(result.is_ok());
        assert_eq!(format!("{}", result.unwrap()), "_chord([$(x), $(y), $(z)])");
    }

    #[test]
    fn test_parse_sequence() {
        let input = "<[$x, $y, $z]>";
        let result = parse_program::<u32>(input);
        assert!(result.is_ok());
        assert_eq!(
            format!("{}", result.unwrap()),
            "_sequence([$(x), $(y), $(z)])"
        );
    }

    #[test]
    fn test_parse_function() {
        let input = "fn (x) => x";
        let result = parse_program::<u32>(input);
        assert!(result.is_ok());
        assert_eq!(format!("{}", result.unwrap()), "fn (x) => x");

        let input = "fn(x, (y, z)) => x";
        let result = parse_program::<u32>(input);
        assert!(result.is_ok());
        assert_eq!(format!("{}", result.unwrap()), "fn (x, (y, z)) => x");
    }

    #[test]
    fn test_parse_let() {
        let input = "let x = 1, y = x + 1 in 2 * y";
        let result = parse_program::<u32>(input);
        assert!(result.is_ok());
        assert_eq!(
            format!("{}", result.unwrap()),
            "(fn (x) => (fn (y) => *(2, y))(+(x, 1)))(1)"
        );

        let input = "let (x, y) = (1, 2) in x * y";
        let result = parse_program::<u32>(input);
        assert!(result.is_ok());
        assert_eq!(
            format!("{}", result.unwrap()),
            "(fn (x, y) => *(x, y))(1, 2)"
        );
    }

    #[test]
    fn test_parse_application() {
        let input = "(fn (x) => x * 2)(3)";
        let result = parse_program::<u32>(input);
        assert!(result.is_ok());
        assert_eq!(format!("{}", result.unwrap()), "(fn (x) => *(x, 2))(3)");

        let input = "Q($@70)";
        let result = parse_program::<u32>(input);
        assert!(result.is_ok());
        assert_eq!(format!("{}", result.unwrap()), "Q($(@(70)))");

        let input = "f(-1) - 1 < 0";
        let result = parse_program::<u32>(input);
        assert!(result.is_ok());
        assert_eq!(format!("{}", result.unwrap()), "<(-(f(-(1)), 1), 0)");
    }

    #[test]
    fn test_parse_pipe() {
        let input = "2 * 3 | (fn (x) => fn(y) => x * y)(4)";
        let result = parse_program::<u32>(input);
        assert!(result.is_ok());
        assert_eq!(
            format!("{}", result.unwrap()),
            "(fn (x) => fn (y) => *(x, y))(4)(*(2, 3))"
        );

        let input = r"$200 | S(0.5, .25) | R(0.5, 1) \ $400";
        let result = parse_program::<u32>(input);
        assert!(result.is_ok());
        assert_eq!(
            format!("{}", result.unwrap()),
            r"\(R(0.5, 1)(S(0.5, 0.25)($(200))), $(400))"
        );
    }

    #[test]
    fn test_function_eval() {
        let context = Vec::new();
        let input = "(fn (x) => fn (x) => x)(7)(5)";
        let result = parse_program::<u32>(input);
        assert!(result.is_ok());
        let expr = result.unwrap();
        println!("Parsed expression: {}", expr);
        let evaluated = evaluate(&context, expr).unwrap();
        assert_eq!(format!("{}", evaluated), "5");

        let input = "(fn (x) => fn (y, z) => (x, y, z))(3)(4, 5)";
        let result = parse_program::<u32>(input);
        assert!(result.is_ok());
        let expr = result.unwrap();
        let evaluated = evaluate(&context, expr).unwrap();
        assert_eq!(format!("{}", evaluated), "(3, 4, 5)");

        let input = "(fn (x, (y, z)) => (x, y, z))(3, (4, 5))";
        let result = parse_program::<u32>(input);
        assert!(result.is_ok());
        let expr = result.unwrap();
        let evaluated = evaluate(&context, expr).unwrap();
        assert_eq!(format!("{}", evaluated), "(3, 4, 5)");
    }

    #[test]
    fn test_parse_sliders() {
        // Empty list
        let result = parse_sliders("[]");
        assert!(result.is_ok());
        let configs = result.unwrap();
        assert_eq!(configs.len(), 0);

        // Single slider
        let result = parse_sliders(r#"["volume:0.75:0:1"]"#);
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
        let result = parse_sliders(r#"["volume:0.5:0:1", "cutoff:2000:200:8000"]"#);
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
        let result = parse_sliders(r#"["volume:0.5:0:1", "freq:0.5:fn(x) => 100 * pow(100, x)"]"#);
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

        let result = parse_sliders(r#"["gain:-0.5:0:1"]"#);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_annotation() {
        // Valid pragma
        let result = parse_annotation(r#"  //#{sliders=["volume:0.75:0:1"]}  "#);
        assert!(result.is_ok());
        let configs = result.unwrap();
        assert_eq!(configs.len(), 1);
        assert_eq!(configs[0].label, "volume");

        // Not a pragma
        assert!(parse_annotation("// regular comment").is_err());
        assert!(parse_annotation("let x = 1").is_err());

        // Pragma with different key
        assert!(parse_annotation("//#{other=[]}").is_err());
    }
}
