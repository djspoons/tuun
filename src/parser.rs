use core::panic;
use std::fmt;
use std::ops::Range;
use std::{cell::RefCell, rc::Rc};

use nom::{
    branch::alt,
    bytes::complete::tag,
    character::complete::{alpha1, char, multispace0},
    combinator::{all_consuming, map, recognize, verify},
    multi::{many0, separated_list0},
    number::complete::float,
    sequence::{delimited, preceded},
    Parser,
};

use crate::builtins;
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

pub type BuiltInFn = Rc<dyn Fn(&mut Vec<Expr>) -> Expr>;

#[derive(Clone)]
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
        arguments: Box<Expr>, // TODO simplify by always using Tuple?
    },
    // Compound expressions
    Tuple(Vec<Expr>),
    // Errors
    Error(String),
}

fn parse_literal(input: LocatedSpan) -> IResult<Expr> {
    #[rustfmt::skip]
    let (rest, value) = float.parse(input)?;
    return Ok((rest, Expr::Float(value)));
}

fn parse_identifier(input: LocatedSpan) -> IResult<String> {
    #[rustfmt::skip]
    let (rest, value) =
        verify(recognize(
            (
                alpha1,
                many0(alt((alpha1, tag("_")))),
            )),
            |s: &LocatedSpan| *s.fragment() != "fn",
        ).parse(input)?;
    return Ok((rest, value.to_string()));
}

fn parse_function(input: LocatedSpan) -> IResult<Expr> {
    #[rustfmt::skip]
    let (rest, expr) =
        ((delimited(
            (tag("fn"), multispace0, char('('), multispace0),
            separated_list0(
                (multispace0, char(','), multispace0),
                parse_identifier),
            (multispace0, char(')'))),
            (multispace0, expect(tag("=>"), "expected '=>'"), multispace0),
            parse_expr,
        )).map(|(arguments, _, body)| {
            let arguments = arguments.into_iter().map(|s| s.to_string()).collect();
            Expr::Function { arguments, body: Box::new(body) }
        }).parse(input)?;
    return Ok((rest, expr));
}

fn parse_variable(input: LocatedSpan) -> IResult<Expr> {
    #[rustfmt::skip]
    let (rest, value) =
        alt((
            parse_identifier,
            tag("$").map(|s: LocatedSpan| s.fragment().to_string()),
        )).parse(input)?;
    return Ok((rest, Expr::Variable(value.to_string())));
}

fn parse_primitive(input: LocatedSpan) -> IResult<Expr> {
    #[rustfmt::skip]
    let (rest, value) = alt((
        parse_literal,
        parse_function,
        parse_variable,
        parse_chord,
        parse_sequence,
        parse_tuple,
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
                parse_primitive,
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
                delimited(multispace0, alt((char('*'), char('/'))), multispace0),
                expect(parse_application, "expected expression after operator"),
            )),
        ),
        |(factor, op_factors)| {
            let mut expr = factor;
            for (op, factor) in op_factors {
                let builtin = Expr::BuiltIn{
                    name: op.to_string(),
                    function: match op {
                        '*' => Rc::new(builtins::multiply),
                        '/' => Rc::new(builtins::divide),
                        _ => panic!("Impossible operator: {}", op),
                    }
                };
                expr = Expr::Application {
                    function: Box::new(builtin),
                    arguments: Box::new(Expr::Tuple(vec![expr, factor.unwrap_or(Expr::Error("parse error".to_string()))])),
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
                delimited(multispace0, alt((char('+'), char('-'))), multispace0),
                expect(parse_multiplicative, "expected expression after operator"),
            )),
        ),
        |(term, op_terms)| {
            let mut expr = term;
            for (op, term) in op_terms {
                let builtin = Expr::BuiltIn{
                    name: op.to_string(),
                    function: match op {
                        '+' => Rc::new(builtins::add),
                        '-' => Rc::new(builtins::subtract),
                        _ => panic!("Impossible operator: {}", op),
                    }
                };
                expr = Expr::Application {
                    function: Box::new(builtin),
                    arguments: Box::new(Expr::Tuple(vec![expr, term.unwrap_or(Expr::Error("parse error".to_string()))])),
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
        (char('<'), multispace0),
        parse_expr,
        (multispace0, expect(char('>'), "expected '>'")),
    ).parse(input)?;
    return Ok((
        rest,
        Expr::Application {
            function: Box::new(Expr::BuiltIn {
                name: "chord".to_string(),
                function: Rc::new(builtins::chord),
            }),
            arguments: Box::new(expr),
        },
    ));
}

fn parse_sequence(input: LocatedSpan) -> IResult<Expr> {
    #[rustfmt::skip]
    let (rest, expr) = delimited(
        (char('['), multispace0),
        parse_expr,
        (multispace0, expect(char(']'), "expected ']'")),
    ).parse(input)?;
    return Ok((
        rest,
        Expr::Application {
            function: Box::new(Expr::BuiltIn {
                name: "sequence".to_string(),
                function: Rc::new(builtins::sequence),
            }),
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
        (multispace0, expect(char(')'), "expected ')'")),
    ).parse(input)?;
    if exprs.len() == 1 {
        return Ok((rest, exprs.pop().unwrap()));
    }
    return Ok((rest, Expr::Tuple(exprs)));
}

fn parse_expr(input: LocatedSpan) -> IResult<Expr> {
    #[rustfmt::skip]
    let (rest, expr) = map(
        (
            parse_additive,
            many0(preceded(
                    delimited(multispace0, char('|'), multispace0),
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
                        expr = Expr::Error("parse error".to_string());
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
        delimited(
            multispace0,
            parse_expr,
            multispace0),
    ).parse(span);
    if errors.borrow().len() > 0 {
        println!(
            "Got result {:} and errors {:?}",
            match result {
                Ok((_, node)) => node,
                _ => Expr::Error("parse error".to_string()),
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
    #[rustfmt::skip]
    let result = all_consuming(
        separated_list0(
            (multispace0, char(','), multispace0),
              (delimited(
                    multispace0,
                   parse_identifier,
                    (multispace0, char('='), multispace0)),
                delimited(
                    multispace0,
                    parse_expr,
                    multispace0),
                )
            ),
        ).parse(span);
    if errors.borrow().len() > 0 {
        return Err(errors.into_inner());
    }
    translate_parse_result(result)
}

fn substitute(context: &Vec<(String, Expr)>, expr: Expr) -> Expr {
    use Expr::{Application, BuiltIn, Float, Function, Tuple, Variable};
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
            Expr::Variable(name)
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
        Expr::Error(_) => expr,
    }
}

fn fmt_as_primitive(expr: &Expr, f: &mut fmt::Formatter) -> fmt::Result {
    match expr {
        Expr::Float(_) | Expr::Variable(_) | Expr::BuiltIn { .. } | Expr::Tuple(_) => {
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
                fmt_as_primitive(function, f)?;
                if let Expr::Tuple(_) = &**arguments {
                    fmt_as_primitive(arguments, f)
                } else {
                    write!(f, "(")?;
                    fmt_as_primitive(arguments, f)?;
                    write!(f, ")")
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
            Expr::Error(s) => write!(f, "{}", s),
        }
    }
}

fn simplify_closed(expr: Expr) -> Result<Expr, Error> {
    use Expr::{Application, BuiltIn, Float, Function, Tuple, Variable};
    match expr {
        Float(_) => Ok(expr),
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
                    let mut arguments = actuals;
                    let result = function(&mut arguments);
                    return match result {
                        Expr::Error(s) => Err(Error::new(s)),
                        _ => Ok(result),
                    };
                }
                (BuiltIn { function, .. }, actual) => {
                    let mut argument = vec![actual];
                    let result = function(&mut argument);
                    return match result {
                        Expr::Error(s) => Err(Error::new(s)),
                        _ => Ok(result),
                    };
                }
                _ => {
                    return Err(Error::new("invalid application".to_string()));
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
        Expr::Waveform(_) => Ok(expr),
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
        let input = "<($x, $y, $z)>";
        let result = parse_program(input);
        assert!(result.is_ok());
        assert_eq!(format!("{}", result.unwrap()), "chord($(x), $(y), $(z))");
    }

    #[test]
    fn test_parse_function() {
        let input = "fn (x) => x";
        let result = parse_program(input);
        assert!(result.is_ok());
        assert_eq!(format!("{}", result.unwrap()), "fn (x) => x");
    }

    #[test]
    fn test_parse_pipe() {
        let input = "2 * 3 | (fn (x) => fn(y) => x * y)(4)";
        let result = parse_program(input);
        assert!(result.is_ok());
        assert_eq!(
            format!("{}", result.unwrap()),
            "((fn (x) => fn (y) => *(x, y))(4))((*(2, 3)))"
        );

        let input = "$200 | S(0.5, .25) | R(0.5, 1)";
        let result = parse_program(input);
        assert!(result.is_ok());
        assert_eq!(
            format!("{}", result.unwrap()),
            "(R(0.5, 1))(((S(0.5, 0.25))(($(200)))))"
        );
    }

    #[test]
    fn test_function_eval() {
        let input = "(fn (x) => fn (x) => x * 2)(7)(5)";
        let result = parse_program(input);
        assert!(result.is_ok());
        let expr = result.unwrap();
        println!("Parsed expression: {}", expr);
        let context = vec![("x".to_string(), Expr::Float(3.0))];
        let simplified = simplify(&context, expr).unwrap();
        assert_eq!(format!("{}", simplified), "10");

        let input = "(fn (x) => fn (y, z) => x * 2 * y + z)(3)(4, 5)";
        let result = parse_program(input);
        assert!(result.is_ok());
        let expr = result.unwrap();
        let context = vec![("x".to_string(), Expr::Float(9.0))];
        let simplified = simplify(&context, expr).unwrap();
        assert_eq!(format!("{}", simplified), "29");
    }
}
