use core::panic;
use std::fmt;
use std::ops::Range;
use std::{cell::RefCell, rc::Rc};

use nom::{
    branch::alt,
    bytes::complete::tag,
    character::complete::{alpha1, char, multispace0, multispace1},
    combinator::{all_consuming, map, not, opt, peek, recognize, verify},
    multi::{many0, separated_list0},
    number::complete::float,
    sequence::{delimited, preceded, terminated},
    Parser,
};

use crate::tracker;

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
pub struct BuiltInFn(pub Rc<dyn Fn(Vec<Expr>) -> Expr>);

impl std::fmt::Debug for BuiltInFn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BuiltInFn(...)")
    }
}

#[derive(Clone, Debug)]
pub enum Expr {
    // Values
    Float(f32),
    Waveform(tracker::Waveform),
    Function {
        arguments: Vec<String>,
        body: Box<Expr>,
    },
    BuiltIn {
        name: String,
        // Pure functions from a vector of values to a value
        function: BuiltInFn,
    },
    // Function application
    Variable(String),
    Application {
        function: Box<Expr>,
        arguments: Box<Expr>, // TODO singular? or Vec<Expr>?
    },
    // Compound expressions
    Tuple(Vec<Expr>),
    List(Vec<Expr>),
    // Errors
    Error(String),
}

fn parse_literal(input: LocatedSpan) -> IResult<Expr> {
    #[rustfmt::skip]
    let (rest, value) =
        // Handle parsing negative floats ourselves
        preceded(
            not(peek(char('-'))),
            float,
        ).parse(input)?;
    return Ok((rest, Expr::Float(value)));
}

fn parse_identifier(input: LocatedSpan) -> IResult<String> {
    #[rustfmt::skip]
    let (rest, value) =
        alt((
            verify(recognize((
                    alpha1,
                    many0(alt((alpha1, tag("_"), tag("#")))),
                )),
                |s: &LocatedSpan| *s.fragment() != "fn" &&
                    *s.fragment() != "let" && *s.fragment() != "in",
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
            tag("&"),
            tag("-"),
            tag("?"),
        )).parse(input)?;
    return Ok((rest, value));
}

fn parse_function(input: LocatedSpan) -> IResult<Expr> {
    #[rustfmt::skip]
    let (rest, expr) =
        ((delimited(
            (tag("fn"), multispace0, char('('), multispace0),
            separated_list0(
                ws(char(',')),
                parse_identifier),
            (multispace0, char(')'))),
            ws(expect(tag("=>"), "expected '=>'")),
            parse_expr,
        )).map(|(arguments, _, body)| {
            let arguments = arguments.into_iter().map(|s| s.to_string()).collect();
            Expr::Function { arguments, body: Box::new(body) }
        }).parse(input)?;
    return Ok((rest, expr));
}

fn parse_bindings(input: LocatedSpan) -> IResult<Vec<(String, Expr)>> {
    #[rustfmt::skip]
    let (rest, bindings) =
        // TODO maybe don't allow [,]?
        terminated(
            separated_list0(
            ws(char(',')),
              (delimited( // TODO maybe some extra whitespace here?
                    multispace0,
                   parse_identifier,
                    ws(char('='))),
                ws(parse_expr),
            )),
            opt(ws(char(','))),
        ).parse(input)?;
    return Ok((rest, bindings));
}

fn parse_let(input: LocatedSpan) -> IResult<Expr> {
    #[rustfmt::skip]
    let (rest, expr) =
        (delimited(
            tag("let"),
            parse_bindings,
            (expect(tag("in"), "expected 'in'"), multispace1),
        ),
        expect(parse_expr, "expected expression after 'in'")
        ).map(|(bindings, expr)| {
            let mut expr = expr.unwrap_or(Expr::Error("_".to_string()));
            for (name, binding) in bindings.into_iter().rev() {
                expr = Expr::Application {
                    function: Box::new(Expr::Function {
                        arguments: vec![name],
                        body: Box::new(expr),
                    }),
                    arguments: Box::new(binding),
                }
            }
            expr
        }).parse(input)?;
    return Ok((rest, expr));
}

fn parse_primitive(input: LocatedSpan) -> IResult<Expr> {
    #[rustfmt::skip]
    let (rest, value) = alt((
        parse_literal,
        parse_function,
        parse_let,
        // Should come before identifiers, since operators also match the identifier rule
        (parse_unary_operator, parse_primitive).map(
            |(op, expr)| Expr::Application {
                function: Box::new(Expr::Variable(op.fragment().to_string())),
                arguments: Box::new(expr),
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

fn parse_application(input: LocatedSpan) -> IResult<Expr> {
    #[rustfmt::skip]
    let (rest, expr) = map(
        (
            parse_primitive,
            many0(preceded(
                multispace0,
                preceded(
                    // Maybe sort of a hack, but don't allow unary operators
                    // in the middle of an application (require parens instead)
                    // ... or maybe just disallow '-'? Or something else?
                    // TODO fix this so we can use unary operators in the middle of applications
                    not(peek(parse_unary_operator)),
                    parse_primitive,
                )
            )),
        ),
        |(func, exprs)| {
            let mut result = func;
            for expr in exprs  {
                result = Expr::Application {
                    function: Box::new(result),
                    arguments: Box::new(expr),
                };
            }
            return result;
        },
    ).parse(input)?;
    return Ok((rest, expr));
}

fn parse_multiplicative(input: LocatedSpan) -> IResult<Expr> {
    #[rustfmt::skip]
    let (rest, value) = map(
        (
            parse_application,
            many0((
                ws(alt((tag("*"), tag("/"), tag("~."), tag("~*")))),
                expect(parse_application, "expected expression after operator"),
            )),
        ),
        |(factor, op_factors)| {
            let mut expr = factor;
            for (op, factor) in op_factors {
                expr = Expr::Application {
                    function: Box::new(Expr::Variable(op.fragment().to_string())),
                    arguments: Box::new(Expr::Tuple(vec![
                        expr, factor.unwrap_or(Expr::Error("_".to_string()))])),
                };
            }
            return expr;
        },
    ).parse(input)?;
    return Ok((rest, value));
}

fn parse_additive(input: LocatedSpan) -> IResult<Expr> {
    #[rustfmt::skip]
    let (rest, value) = map(
        (
            parse_multiplicative,
            many0((
                ws(alt((tag("+"), tag("-"), tag("~+")))),
                expect(parse_multiplicative, "expected expression after operator"),
            )),
        ),
        |(term, op_terms)| {
            let mut expr = term;
            for (op, term) in op_terms {
                expr = Expr::Application {
                    function: Box::new(Expr::Variable(op.fragment().to_string())),
                    arguments: Box::new(Expr::Tuple(vec![expr, term.unwrap_or(Expr::Error("_".to_string()))])),
                };
            }
            return expr;
        },
    ).parse(input)?;
    return Ok((rest, value));
}

fn parse_chord(input: LocatedSpan) -> IResult<Expr> {
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
            arguments: Box::new(expr),
        },
    ));
}

fn parse_sequence(input: LocatedSpan) -> IResult<Expr> {
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
            arguments: Box::new(expr),
        },
    ));
}

fn parse_tuple(input: LocatedSpan) -> IResult<Expr> {
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

fn parse_list(input: LocatedSpan) -> IResult<Expr> {
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

fn parse_expr(input: LocatedSpan) -> IResult<Expr> {
    #[rustfmt::skip]
    let (rest, expr) = map(
        (
            parse_additive,
            many0(preceded(
                    ws(char('|')),
                    expect(parse_additive, "expected expression after | operator"),
                )),
        ),
        |(argument, fn_exprs)| {
            let mut expr = argument;
            for fn_expr in fn_exprs {
                match fn_expr {
                    Some(function) => {
                        expr = Expr::Application {
                            function: Box::new(function),
                            arguments: Box::new(expr),
                        }
                    }
                    None => {
                        expr = Expr::Error("_".to_string());
                    }
                }
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
            println!("Error on parsing input: {:?}", e);
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

pub fn parse_program(input: &str) -> Result<Expr, Vec<Error>> {
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
                _ => Expr::Error("_".to_string()),
            },
            errors.borrow()
        );
        return Err(errors.into_inner());
    }
    translate_parse_result(result)
}

pub fn parse_context(input: &str) -> Result<Vec<(String, Expr)>, Vec<Error>> {
    let errors = RefCell::new(Vec::new());
    let span = LocatedSpan::new_extra(input, ParseState(&errors));
    let result = all_consuming(parse_bindings).parse(span);
    if errors.borrow().len() > 0 {
        return Err(errors.into_inner());
    }
    translate_parse_result(result)
}

fn substitute(context: &Vec<(String, Expr)>, expr: Expr) -> Expr {
    use Expr::{Application, BuiltIn, Float, Function, List, Tuple, Variable};
    match expr {
        Float(_) => expr,
        Expr::Waveform(waveform) => Expr::Waveform(waveform),
        Function { arguments, body } => {
            let mut context = context.clone();
            for arg in &arguments {
                context.push((arg.clone(), Expr::Variable(arg.clone())));
            }
            let body = substitute(&context, *body);
            Expr::Function {
                arguments,
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
        Application {
            function,
            arguments,
        } => {
            let function = substitute(context, *function);
            let arguments = substitute(context, *arguments);
            Expr::Application {
                function: Box::new(function),
                arguments: Box::new(arguments),
            }
        }
        Tuple(exprs) => Expr::Tuple(exprs.into_iter().map(|e| substitute(context, e)).collect()),
        List(exprs) => Expr::List(exprs.into_iter().map(|e| substitute(context, e)).collect()),
        Expr::Error(_) => expr,
    }
}

fn fmt_with_parens(expr: &Expr, f: &mut fmt::Formatter) -> fmt::Result {
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

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Expr::Float(value) => write!(f, "{}", value),
            Expr::Waveform(waveform) => {
                write!(f, "{:?}", waveform)
            }
            Expr::Function { arguments, body } => {
                write!(f, "fn (")?;
                for (i, arg) in arguments.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, ") => {}", body)
            }
            Expr::BuiltIn { name, .. } => write!(f, "{}", name),
            Expr::Variable(name) => write!(f, "{}", name),
            Expr::Application {
                function,
                arguments,
            } => {
                fmt_with_parens(function, f)?;
                if let Expr::Tuple(_) = &**arguments {
                    write!(f, "{}", arguments)
                } else {
                    write!(f, "({})", arguments)
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
            Expr::Error(s) => write!(f, "{}", s),
        }
    }
}

fn simplify_closed(expr: Expr) -> Result<Expr, Error> {
    use Expr::{Application, BuiltIn, Float, Function, List, Tuple, Variable, Waveform};
    match expr {
        Float(_) => Ok(expr),
        Waveform(_) => Ok(expr),
        Function { .. } => Ok(expr),
        Variable(name) => Err(Error::new(format!(
            "Variable '{}' not found in context",
            name
        ))),
        BuiltIn { .. } => Ok(expr),
        Application {
            function,
            arguments,
        } => {
            let function = simplify_closed(*function)?;
            let arguments = simplify_closed(*arguments)?;
            match (function, arguments) {
                (
                    Function {
                        arguments: formals,
                        body,
                    },
                    actuals,
                ) => match (formals, actuals) {
                    (formals, Expr::Tuple(actual_arguments))
                        if formals.len() == actual_arguments.len() =>
                    {
                        let context = formals
                            .into_iter()
                            .zip(actual_arguments)
                            .map(|(formal, actual)| (formal, actual))
                            .collect();
                        return simplify(&context, *body);
                    }
                    (formals, expr) if formals.len() == 1 => {
                        let context = vec![(formals[0].clone(), expr)];
                        return simplify(&context, *body);
                    }
                    _ => return Err(Error::new("Mismatched number of arguments".to_string())),
                },
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
                    .map(|e| simplify_closed(e))
                    .collect::<Result<Vec<_>, _>>()?,
            ));
        }
        List(exprs) => {
            return Ok(List(
                exprs
                    .into_iter()
                    .map(|e| simplify_closed(e))
                    .collect::<Result<Vec<_>, _>>()?,
            ));
        }
        Expr::Error(s) => Err(Error::new(s)),
    }
}

pub fn simplify(context: &Vec<(String, Expr)>, mut expr: Expr) -> Result<Expr, Error> {
    expr = substitute(context, expr);
    return simplify_closed(expr);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_variable() {
        let input = "fn";
        let result = parse_program(input);
        assert!(result.is_err());

        let input = "my_var";
        let result = parse_program(input);
        assert!(result.is_ok());
        assert_eq!(format!("{}", result.unwrap()), "my_var");

        let input = "$";
        let result = parse_program(input);
        assert!(result.is_ok());
        assert_eq!(format!("{}", result.unwrap()), "$");
    }

    #[test]
    fn test_parse_arithmetic() {
        let input = "(10 - 8 - 1) * 6";
        let result = parse_program(input);
        assert!(result.is_ok());
        assert_eq!(format!("{}", result.unwrap()), "*(-(-(10, 8), 1), 6)");

        let input = "1 + 2 * 3.5 * 8 + 10";
        let result = parse_program(input);
        assert!(result.is_ok());
        assert_eq!(
            format!("{}", result.unwrap()),
            "+(+(1, *(*(2, 3.5), 8)), 10)"
        );
    }

    #[test]
    fn test_parse_chord() {
        let input = "{[$x, $y, $z]}";
        let result = parse_program(input);
        assert!(result.is_ok());
        assert_eq!(format!("{}", result.unwrap()), "_chord([$(x), $(y), $(z)])");
    }

    #[test]
    fn test_parse_function() {
        let input = "fn (x) => x";
        let result = parse_program(input);
        assert!(result.is_ok());
        assert_eq!(format!("{}", result.unwrap()), "fn (x) => x");
    }

    #[test]
    fn test_parse_let() {
        let input = "let x = 1, y = x + 1 in 2 * y";
        let result = parse_program(input);
        assert!(result.is_ok());
        assert_eq!(
            format!("{}", result.unwrap()),
            "(fn (x) => (fn (y) => *(2, y))(+(x, 1)))(1)"
        );
    }

    #[test]
    fn test_parse_application() {
        let input = "(fn (x) => x * 2)(3)";
        let result = parse_program(input);
        assert!(result.is_ok());
        assert_eq!(format!("{}", result.unwrap()), "(fn (x) => *(x, 2))(3)");

        let input = "Q($@70)";
        let result = parse_program(input);
        assert!(result.is_ok());
        assert_eq!(format!("{}", result.unwrap()), "Q($(@(70)))");
    }

    #[test]
    fn test_parse_pipe() {
        let input = "2 * 3 | (fn (x) => fn(y) => x * y)(4)";
        let result = parse_program(input);
        assert!(result.is_ok());
        assert_eq!(
            format!("{}", result.unwrap()),
            "(fn (x) => fn (y) => *(x, y))(4)(*(2, 3))"
        );

        let input = "$200 | S(0.5, .25) | R(0.5, 1)";
        let result = parse_program(input);
        assert!(result.is_ok());
        assert_eq!(
            format!("{}", result.unwrap()),
            "R(0.5, 1)(S(0.5, 0.25)($(200)))"
        );
    }

    #[test]
    fn test_function_eval() {
        let context = Vec::new();
        let input = "(fn (x) => fn (x) => x)(7)(5)";
        let result = parse_program(input);
        assert!(result.is_ok());
        let expr = result.unwrap();
        println!("Parsed expression: {}", expr);
        let simplified = simplify(&context, expr).unwrap();
        assert_eq!(format!("{}", simplified), "5");

        let input = "(fn (x) => fn (y, z) => (x, y, z))(3)(4, 5)";
        let result = parse_program(input);
        assert!(result.is_ok());
        let expr = result.unwrap();
        let simplified = simplify(&context, expr).unwrap();
        assert_eq!(format!("{}", simplified), "(3, 4, 5)");
    }
}
