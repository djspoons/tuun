use std::ops::Range;
use std::cell::RefCell;
use std::time::Duration;

use nom::{
    Parser,
    branch::alt,
    combinator::{map, cut, all_consuming},
    character::complete::{char, multispace0, multispace1},
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
pub struct ParseError {
    range: Range<usize>,
    message: Option<String>,
}

impl<'a> ParseError {
    fn new(span: &LocatedSpan, message: String) -> Self {
        Self { range: span.to_range(), message: Some(message) }
    }

    //pub fn span(&self) -> &Span { &self.span }
    pub fn range(&self) -> Range<usize> { self.range.clone() }

    // pub fn line(&self) -> u32 { self.span().location_line() }

    // pub fn offset(&self) -> usize { self.span().location_offset() }
}

impl<'a> nom::error::ParseError<LocatedSpan<'a>> for ParseError {
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
struct ParseState<'a>(&'a RefCell<Vec<ParseError>>);

impl<'a> ParseState<'a> {
    /// Pushes an error onto the errors stack from within a `nom`
    /// parser combinator while still allowing parsing to continue.
    fn report_error(&self, error: ParseError) {
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
            let err = ParseError::new(&input, error_msg.to_string());
            input.extra.report_error(err); // Push error onto stack.
            Ok((input, None)) // Parsing failed, but keep going.
        },
        Err(err) => Err(err),
    }
}

#[derive(Debug, PartialEq)]
pub enum Expr {
    // Primitives
    Float(f32),
    Variable(String),
    // Operations
    Multiply(Box<Expr>, Box<Expr>),
    Divide(Box<Expr>, Box<Expr>),
    Application(Box<Expr>, Box<Expr>),
    // Tones
    SineWave { frequency: Box<Expr> },
    Truncated { duration: Duration, tone: Box<Expr> },
    Amplified { gain: Box<Expr>, tone: Box<Expr> },
    // Compounds
    Chord(Vec<Expr>),
    Sequence(Vec<Expr>),
    Tuple(Vec<Expr>),
    Error,
}

fn parse_literal(input: LocatedSpan) -> IResult<Expr> {
    let (rest, value) = float.parse(input)?;
    return Ok((rest, Expr::Float(value)));
}

fn parse_variable(input: LocatedSpan) -> IResult<Expr> {
    let (rest, value) = nom::character::complete::alpha1.parse(input)?;
    return Ok((rest, Expr::Variable(value.to_string())));
}

fn parse_primitive(input: LocatedSpan) -> IResult<Expr> {
    let (rest, value) = alt((
        parse_literal,
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
                    tone: Box::new(Expr::SineWave {
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
                multispace0,
                parse_primitive,
            )),
        ),
        |(func, args)| {
            let mut result = func;
            for arg in args {
                result = Expr::Application(Box::new(result), Box::new(arg));
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
    let (rest, exprs) = delimited(
        (char('('), multispace0),
        separated_list0(
            (multispace0, char(','), multispace0),
            parse_expr,
        ),
        (multispace0, expect(char(')'), "expected ')'")),
    ).parse(input)?;
    return Ok((rest, Expr::Tuple(exprs)));
}

fn parse_expr(input: LocatedSpan) -> IResult<Expr> {
    let (rest, expr) = alt((
        parse_multiplicative,
    )).parse(input)?;
    return Ok((rest, expr));
}

pub fn parse_program(input: &str) -> Result<Expr, Vec<ParseError>> {
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
            return Err(vec![ParseError::new(&e.input, "unable to parse input".to_string())]);
        }
        Err(nom::Err::Incomplete(_)) => {
            panic!("Incomplete error on input");
        }
        Err(nom::Err::Failure(e)) => {
            println!("Failed to parse input: {:?}", e);
            return Err(vec![ParseError::new(&e.input, "unable to parse input".to_string())]);
        }
    }
}
