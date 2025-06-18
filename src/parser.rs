use std::ops::Range;
use std::cell::RefCell;
use std::time::Duration;

use nom::{
    Parser,
    branch::alt,
    combinator::{map, cut, all_consuming, peek, not},
    bytes::complete::tag,
    character::complete::{alpha1, char, multispace0, multispace1},
    number::complete::float,
    sequence::{delimited, preceded},
    multi::{many0, separated_list0},
};

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
    range: Range<usize>,
    message: String,
}

impl<'a> Error {
    fn new(span: &LocatedSpan, message: String) -> Self {
        Self { range: span.to_range(), message }
    }

    //pub fn span(&self) -> &Span { &self.span }
    pub fn range(&self) -> Range<usize> { self.range.clone() }

    // pub fn line(&self) -> u32 { self.span().location_line() }

    // pub fn offset(&self) -> usize { self.span().location_offset() }
}

impl<'a> nom::error::ParseError<LocatedSpan<'a>> for Error {
    fn from_error_kind(input: LocatedSpan<'a>, kind: nom::error::ErrorKind) -> Self {
        Self::new(&input, format!("parse error {:?}", kind))
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
fn expect<'a, F, E, T>(mut parser: F, error_msg: E) -> impl FnMut(LocatedSpan<'a>) -> IResult<Option<T>>
where
    F: FnMut(LocatedSpan<'a>) -> IResult<T>,
    E: ToString,
{
    move |input: LocatedSpan| match parser(input) {
        Ok((remaining, out)) =>
            Ok((remaining, Some(out))),
        Err(nom::Err::Error(e)) |
        Err(nom::Err::Failure(e)) => {
            let input = e.input;
            let err = Error::new(&input, error_msg.to_string());
            input.extra.report_error(err); // Push error onto stack.
            Ok((input, None)) // Parsing failed, but keep going.
        },
        Err(err) => Err(err),
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum BuiltInFn {
    Power, // float * float -> float
}

#[derive(Debug, PartialEq, Clone)]
pub enum Expr {
    // Primitives
    Float(f32),
    Function { arguments: Vec<String>, body: Box<Expr> },
    Variable(String),
    BuiltIn(BuiltInFn),
    // Waveforms
    SineWave { frequency: Box<Expr> },
    Truncated { duration: Duration, waveform: Box<Expr> },
    Amplified { gain: Box<Expr>, waveform: Box<Expr> },
    Chord(Vec<Expr>),
    Sequence(Vec<Expr>),
    // Operations
    Multiply(Box<Expr>, Box<Expr>),
    Divide(Box<Expr>, Box<Expr>),
    Application { function: Box<Expr>, argument: Box<Expr> },
    // Compounds
    Tuple(Vec<Expr>),
    // Error
    Error,
}

fn parse_literal(input: LocatedSpan) -> IResult<Expr> {
    let (rest, value) = float.parse(input)?;
    return Ok((rest, Expr::Float(value)));
}

fn parse_identifier(input: LocatedSpan) -> IResult<String> {
    let (rest, value) =
        preceded(
            peek(not(tag("fn"))),
            alpha1,
        ).parse(input)?;
    return Ok((rest, value.to_string()));
}

fn parse_function(input: LocatedSpan) -> IResult<Expr> {
    let (rest, expr) =
        ((delimited(
            (tag("fn"), multispace1, char('('), multispace0),
            separated_list0(
                (multispace0, char(','), multispace0),
                parse_identifier),
            (multispace0, char(')'))),
            (multispace0, expect(char('='), "expected '='"), multispace0),
            parse_expr,
        )).map(|(arguments, _, body)| {
            let arguments = arguments.into_iter().map(|s| s.to_string()).collect();
            Expr::Function { arguments, body: Box::new(body) }
        }).parse(input)?;
    return Ok((rest, expr));
}

fn parse_variable(input: LocatedSpan) -> IResult<Expr> {
    let (rest, value) =
        parse_identifier.parse(input)?;
    return Ok((rest, Expr::Variable(value.to_string())));
}

fn parse_primitive(input: LocatedSpan) -> IResult<Expr> {
    let (rest, value) = alt((
        parse_literal,
        parse_function,
        parse_variable,
        parse_chord,
        parse_sequence,
        parse_tuple,
        // TODO generalize symbol application
        preceded(
            char('$'),
            expect(parse_primitive, "expected expression after $")).map(
                |expr| Expr::Truncated{
                    duration: Duration::from_secs(1), 
                    waveform: Box::new(Expr::SineWave {
                         frequency: Box::new(expr.unwrap_or(Expr::Error))
                        })
                    })
    )).parse(input)?;
    return Ok((rest, value));
}

fn parse_application(input: LocatedSpan) -> IResult<Expr> {
    let (rest, expr) = map(
        (
            parse_primitive,
            many0(preceded(
                // TODO maybe require parens? and not allow space?
                multispace0,
                parse_primitive,
            )),
        ),
        |(func, arguments)| {
            let mut result = func;
            for arg in arguments {
                result = Expr::Application { function: Box::new(result), argument: Box::new(arg) };
            }
            return result;
        },
    ).parse(input)?;
    return Ok((rest, expr));
}

fn parse_multiplicative(input: LocatedSpan) -> IResult<Expr> {
    let (rest, value) =
        map((parse_application,
            many0(
                (delimited(multispace0,alt((char('*'), char('/'))), multispace0),
                expect(parse_application, "expected expression after operator"))
            )), |(factor, op_factors)| {
            let mut result = factor;
            for (op, factor) in op_factors {
                let expr_op = match op {
                    '*' => Expr::Multiply,
                    '/' => Expr::Divide,
                    _ => panic!("Impossible operator: {}", op),
                };
                result = expr_op(Box::new(result), Box::new(factor.unwrap_or(Expr::Error)))
            }
            return result;
        }).parse(input)?;
    return Ok((rest, value));
}

fn parse_chord(input: LocatedSpan) -> IResult<Expr> {
    let (rest, exprs) = delimited(
        (char('<'), multispace0),
        cut(separated_list0(
            multispace1,
            parse_primitive,
        )),
        (multispace0, expect(char('>'), "expected '>'")),
    ).parse(input)?;
    return Ok((rest, Expr::Chord(exprs)));
}

fn parse_sequence(input: LocatedSpan) -> IResult<Expr> {
    let (rest, exprs) = delimited(
        (char('['), multispace0),
        cut(separated_list0(
            multispace1,
            parse_primitive,
        )),
        (multispace0, expect(char(']'), "expected ']'")),
    ).parse(input)?;
    return Ok((rest, Expr::Sequence(exprs)));
}

fn parse_tuple(input: LocatedSpan) -> IResult<Expr> {
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
    let (rest, expr) = alt((
        parse_multiplicative,
    )).parse(input)?;
    return Ok((rest, expr));
}

pub fn parse_program(input: &str) -> Result<Expr, Vec<Error>> {
    let errors = RefCell::new(Vec::new());
    let span = LocatedSpan::new_extra(input, ParseState(&errors));
    let result = all_consuming(
        delimited(
            multispace0,
            parse_expr,
            multispace0),
    ).parse(span);
    if errors.borrow().len() > 0 {
        println!("Got result {:?} and errors {:?}", match result { Ok((_, node)) => node, _ => Expr::Error}, errors.borrow());
        return Err(errors.into_inner());
    }
    match result {
        Ok((_, node)) => {
            return Ok(node);
        },
        Err(nom::Err::Error(e)) => {
            println!("Error on parsing input: {:?}", e);
            return Err(vec![Error::new(&e.input, "unable to parse input".to_string())]);
        }
        Err(nom::Err::Incomplete(_)) => {
            panic!("Incomplete error on input");
        }
        Err(nom::Err::Failure(e)) => {
            println!("Failed to parse input: {:?}", e);
            return Err(vec![Error::new(&e.input, "unable to parse input".to_string())]);
        }
    }
}

fn substitute(context: &Vec<(String, Expr)>, expr: Expr) -> Expr {
    use Expr::{
        Float, Function, Variable, BuiltIn, SineWave, Truncated, Amplified,
        Multiply, Divide, Application, Chord, Sequence, Tuple,
    };
    match expr {
        Float(_) => expr,
        Function { arguments, body } => {
            let mut context = context.clone();
            for arg in &arguments {
                context.push((arg.clone(), Expr::Variable(arg.clone())));
            }
            let body = substitute(&context, *body);
            Expr::Function { arguments, body: Box::new(body) }
        },
        Variable(name) => {
            for (var_name, value) in context.iter().rev() {
                if var_name == &name {
                    return value.clone();
                }
            }
            Expr::Variable(name)
        },
        BuiltIn { .. } => expr,
        SineWave { frequency } => {
            let frequency = substitute(context, *frequency);
            Expr::SineWave { frequency: Box::new(frequency) }
        }
        Truncated { duration, waveform } => {
            let waveform = substitute(context, *waveform);
            Expr::Truncated { duration, waveform: Box::new(waveform) }
        },
        Amplified { gain, waveform } => {
            let gain = substitute(context, *gain);
            let waveform = substitute(context, *waveform);
            Expr::Amplified { gain: Box::new(gain), waveform: Box::new(waveform) }
        },
        Multiply(left, right) => {
            let left = substitute(context, *left);
            let right = substitute(context, *right);
            Expr::Multiply(Box::new(left), Box::new(right))
        },
        Divide(left, right) => {
            let left = substitute(context, *left);
            let right = substitute(context, *right);
            Expr::Divide(Box::new(left), Box::new(right))
        },
        Application { function, argument } => {
            let function = substitute(context, *function);
            let argument = substitute(context, *argument);
            Expr::Application { function: Box::new(function), argument: Box::new(argument) }
        },
        Chord(exprs) => Expr::Chord(
            exprs.into_iter().map(|e| substitute(context, e)).collect()),
        Sequence(exprs) => Expr::Sequence(
            exprs.into_iter().map(|e| substitute(context, e)).collect()),
        Tuple(exprs) => Expr::Tuple(
            exprs.into_iter().map(|e| substitute(context, e)).collect()),
        Expr::Error => expr,
    }

}

fn simplify_closed(expr: Expr) -> Result<Expr, Error> {
    use Expr::{
        Float, Function, Variable, BuiltIn, SineWave, Truncated, Amplified,
        Multiply, Divide, Application, Chord, Sequence, Tuple,
    };
    match expr {
        Float(_) => Ok(expr),
        Function { .. } => Ok(expr),
        Variable(name) => {
            Err(Error {
                range: 0..0, // TODO fix range?
                message: format!("Variable '{}' not found in context", name),
            })
        },
        BuiltIn { .. } => Ok(expr),
        Multiply(left, right) => {
            let left = simplify_closed(*left)?;
            let right = simplify_closed(*right)?;
            if let (Float(l), Float(r)) = (left, right) {
                return Ok(Expr::Float(l * r));
            }
            Err(Error {
                range: 0..0, // TODO fix range?
                message: "Cannot multiply non-float expressions".to_string(),
            })
        },
        Divide(left, right) => {
            let left = simplify_closed(*left)?;
            let right = simplify_closed(*right)?;
            if let (Float(l), Float(r)) = (left, right) {
                return Ok(Float(l / r));
            }
            Err(Error {
                range: 0..0, // TODO fix range?
                message: "Cannot divide non-float expressions".to_string(),
            })
        },
        Application { function, argument } => {
            let function = simplify_closed(*function)?;
            let argument = simplify_closed(*argument)?;
            match (function, argument) {
                (Function { arguments: formals, body }, actual) =>
                    match (formals, actual) {
                        (formals, Expr::Tuple(actual_arguments)) if formals.len() == actual_arguments.len() => {
                            let context = formals.into_iter().zip(actual_arguments).map(|(formal, actual)| (formal, actual)).collect();
                            return simplify(&context, *body);
                        },
                        (formals, expr) if formals.len() == 1 => {
                            let context = vec![(formals[0].clone(), expr)];
                            return simplify(&context, *body);
                        }
                        _ => {
                            return Err(Error {
                                range: 0..0, // fix range?
                                message: "Mismatched number of arguments".to_string(),
                            });
                        }
                    },
                (BuiltIn(BuiltInFn::Power), Tuple(mut actuals)) if actuals.len() == 2 => {
                    if let Float(exponent) = actuals.remove(1) {
                        if let Float(base) = actuals.remove(0) {
                            return Ok(Float(base.powf(exponent)));
                        }
                    }
                    return Err(Error {
                        range: 0..0, // fix range?
                        message: "Built-in function 'power' requires two float arguments".to_string(),
                    });
                }
                _ => {
                    return Err(Error {
                        range: 0..0, // fix range?
                        message: "Invalid application".to_string(),
                    });
                }
            }
        },
        SineWave { frequency } => {
            let frequency = simplify_closed(*frequency)?;
            return Ok(SineWave { frequency: Box::new(frequency) });
        },
        Truncated { duration, waveform } => {
            let waveform = simplify_closed(*waveform)?;
            return Ok(Truncated { duration, waveform: Box::new(waveform) });
        },
        Amplified { gain, waveform } => {
            let gain = simplify_closed(*gain)?;
            let waveform: Expr = simplify_closed(*waveform)?;
            return Ok(Amplified { gain: Box::new(gain), waveform: Box::new(waveform) });
        }

        Chord (exprs) => {
            return Ok(Chord(
                exprs.into_iter().map(|e| simplify_closed(e)).collect::<Result<Vec<_>, _>>()?
            ));
        }
        Sequence (exprs) => {
            return Ok(Sequence(
                exprs.into_iter().map(|e| { simplify_closed(e) }).collect::<Result<Vec<_>, _>>()?
            ));
        }
        Tuple (exprs) => {
            return Ok(Tuple(
                exprs.into_iter().map(|e| { simplify_closed(e) }).collect::<Result<Vec<_>, _>>()?
            ));
        }

        Expr::Error => Err(Error { range: 0..0, message: "found error".to_string() }),
    }
}


pub fn simplify(context: &Vec<(String, Expr)>, mut expr: Expr) -> Result<Expr, Error> {
    expr = substitute(context, expr);
    println!("Substitute returned {:?}", &expr);
    return simplify_closed(expr);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_literal() {
        let input = "3.14";
        let result = parse_program(input);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Expr::Float(3.14));
    }

    #[test]
    fn test_parse_variable() {
        let input = "x";
        let result = parse_program(input);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Expr::Variable("x".to_string()));
    }

    #[test]
    fn test_parse_multiplication() {
        let input = "2 * 3.5";
        let result = parse_program(input);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Expr::Multiply(Box::new(Expr::Float(2.0)), Box::new(Expr::Float(3.5))));
    }

    #[test]
    fn test_parse_chord() {
        let input = "<x y z>";
        let result = parse_program(input);
        assert!(result.is_ok());
        if let Expr::Chord(exprs) = result.unwrap() {
            assert_eq!(exprs.len(), 3);
            assert!(matches!(exprs[0], Expr::Variable(_)));
            assert!(matches!(exprs[1], Expr::Variable(_)));
            assert!(matches!(exprs[2], Expr::Variable(_)));
        } else {
            panic!("Expected a Chord expression");
        }
    }

    #[test]
    fn test_parse_function() {
        let input = "fn (x) = x";
        let result = parse_program(input);
        assert!(result.is_ok());
        if let Expr::Function { arguments, body} = result.unwrap() {
            //assert!(matches!(name, "id".to_string()));
            //assert!(matches!(arguments, vec!["x"]));
            //assert!(matches!(body, Expr::Variable("x")));
        } else {
            panic!("Expected a Function expression");
        }
    }
}