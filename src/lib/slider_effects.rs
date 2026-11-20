//! The `slider` calls a program can reach, found without playing it.
//!
//! An abstract evaluation stands in for evaluation: a value is either known
//! (a number, string or boolean that evaluation would compute without playing
//! anything), a function, or unknown. Unknown values include waveforms and,
//! where a function is applied to parameters no caller has given, those
//! parameters and everything computed from them. A condition that isn't known
//! takes both branches, and a built-in applies the functions it's given to
//! unknown arguments, so the calls found include those any note or branch
//! could reach.

use std::collections::HashMap;
use std::mem;
use std::rc::Rc;

use crate::eval::{BuiltInCall, CallPath};
use crate::expr::{Binding, BuiltInFn, Error, Expr, Pattern, SourceBinding, SourceExpr, Span};
use crate::slider::{Curve, SliderSpec};
use crate::trace::{CarriesTrace, Trace};

/// A parameter that a value depends on, when no caller has given it.
#[derive(Clone, Debug, PartialEq)]
pub struct Parameter<S> {
    pub name: String,
    /// Where the function that takes it was written.
    pub function: Option<Span<S>>,
}

/// The part of a slider's spec that isn't known, and the parameter it depends
/// on, if any.
#[derive(Clone, Debug, PartialEq)]
pub struct Open<S> {
    /// `"default"`, `"range"` or `"curve"`.
    pub part: &'static str,
    pub parameter: Option<Parameter<S>>,
}

/// A `slider` call the abstract evaluation reaches.
#[derive(Clone, Debug)]
pub struct Introduced<S> {
    /// The label, or the parameter it depends on when it isn't known.
    pub label: Result<String, Option<Parameter<S>>>,
    /// The spec, or the part of it that isn't known.
    pub spec: Result<SliderSpec, Open<S>>,
    /// Where the call was written.
    pub site: Option<Span<S>>,
    /// The code running around the call.
    pub path: CallPath<S>,
}

impl<S> Introduced<S> {
    /// Returns where the call was written, then where each piece of code
    /// running around it was written, innermost first.
    pub fn code(&self) -> impl Iterator<Item = &Span<S>> {
        self.site.iter().chain(self.path.code())
    }
}

/// The `slider` calls one program can reach.
#[derive(Debug)]
pub struct Analysis<S> {
    /// The calls the program reaches, in the order the abstract evaluation
    /// first reaches them. A keys instrument's function is applied to a note
    /// and velocity that aren't known.
    pub program: Vec<Introduced<S>>,
    /// The calls each binding reaches as its own code: those its definition
    /// reaches, and those in the functions it defines (and the functions they
    /// return) applied to parameters that aren't known.
    pub definitions: Vec<Introduced<S>>,
}

/// Returns the `slider` calls the program `expr` can reach when evaluated
/// under `bindings`, as a keys instrument's function when `keys` is true.
///
/// `resolve` should behave identically to the resolver evaluation uses.
/// Expects a program the checker accepts: parts it can't follow, such as an
/// unbound variable or a module that doesn't resolve, are unknown.
pub fn analyze<'a, M, S, F>(
    resolve: F,
    bindings: &'a [SourceBinding<M, S>],
    expr: &'a SourceExpr<M, S>,
    keys: bool,
) -> Analysis<S>
where
    F: Fn(&[String]) -> Result<&'a [SourceBinding<M, S>], Error<S>>,
    M: CarriesTrace<S> + Clone,
    S: Clone,
{
    let mut this = Abstract {
        resolve: &resolve,
        modules: HashMap::new(),
        opening: Vec::new(),
        introduced: Vec::new(),
        reached: Vec::new(),
    };
    let mut definitions = Vec::new();
    let env = this.bindings(bindings, Env(None), Some(&mut definitions));
    this.reached.clear();
    let root = CallPath::empty().enter(expr.span.clone(), None);
    let value = this.eval(expr, &env, &root);
    if keys {
        this.apply_to_parameters(&value, &root, 0);
    }
    let program = mem::take(&mut this.reached);
    let pick = |indices: Vec<usize>| {
        let mut seen = vec![false; this.introduced.len()];
        indices
            .into_iter()
            .filter(|&i| !mem::replace(&mut seen[i], true))
            .map(|i| this.introduced[i].clone())
            .collect()
    };
    Analysis {
        program: pick(program),
        definitions: pick(definitions),
    }
}

/// How many functions deep a binding's functions are applied to unknown
/// parameters: a function, the function it returns, and so on.
const RETURNED_FUNCTIONS: usize = 4;

/// The built-ins whose results are computed from known arguments.
const COMPUTED: &[&str] = &[
    "+", "-", "*", "/", "==", "!=", "<", "<=", ">", ">=", "pow", "log", "sqrt", "exp", "sin",
    "round",
];

/// An abstract value.
#[derive(Clone)]
enum Value<'a, M, S> {
    Num(f32),
    Str(String),
    Bool(bool),
    /// A value that isn't known before playing, with the parameter it
    /// depends on, if any.
    Unknown(Option<Rc<Parameter<S>>>),
    Closure(Rc<Closure<'a, M, S>>),
    BuiltIn(&'a str, &'a BuiltInFn<M, S>),
    /// A function that returns its argument, as `debug` returns.
    Identity,
    Tuple(Vec<Value<'a, M, S>>),
    List(Vec<Value<'a, M, S>>),
    Module(Exports<'a, M, S>),
    /// One of several values, as the branches of a condition that isn't
    /// known give.
    OneOf(Vec<Value<'a, M, S>>),
}

impl<M, S> Value<'_, M, S> {
    /// Returns the parameter this value depends on, if it is unknown and
    /// depends on one.
    fn parameter(&self) -> Option<Rc<Parameter<S>>> {
        match self {
            Value::Unknown(parameter) => parameter.clone(),
            _ => None,
        }
    }

    /// Returns whether this value is a function or something a function may
    /// be found in.
    fn holds_functions(&self) -> bool {
        matches!(
            self,
            Value::Closure(_)
                | Value::BuiltIn(..)
                | Value::Identity
                | Value::Module(_)
                | Value::OneOf(_)
        )
    }
}

/// A function value: its parameters, the values of its named parameters'
/// defaults, its body, and the names in scope where it was written.
struct Closure<'a, M, S> {
    positional: &'a [Pattern],
    named: Vec<(String, Value<'a, M, S>)>,
    body: &'a SourceExpr<M, S>,
    env: Env<'a, M, S>,
    /// Where the function was written.
    span: Option<Span<S>>,
}

/// A module's exports: each name, its value, and the calls its definition
/// reaches.
type Exports<'a, M, S> = Rc<Vec<(String, Value<'a, M, S>, Rc<[usize]>)>>;

/// The names in scope, innermost first.
struct Env<'a, M, S>(Option<Rc<Link<'a, M, S>>>);

impl<M, S> Clone for Env<'_, M, S> {
    fn clone(&self) -> Self {
        Env(self.0.clone())
    }
}

/// One name in scope and the names outside it.
struct Link<'a, M, S> {
    name: String,
    value: Value<'a, M, S>,
    /// The calls the definition of a top-level name reaches.
    reached: Rc<[usize]>,
    outer: Env<'a, M, S>,
}

impl<'a, M, S> Env<'a, M, S> {
    /// Returns this scope with `name` bound to `value`.
    fn bind(&self, name: &str, value: Value<'a, M, S>, reached: Rc<[usize]>) -> Self {
        Env(Some(Rc::new(Link {
            name: name.to_string(),
            value,
            reached,
            outer: self.clone(),
        })))
    }

    /// Returns the innermost binding of `name`.
    fn lookup(&self, name: &str) -> Option<&Link<'a, M, S>> {
        let mut env = self;
        while let Some(link) = &env.0 {
            if link.name == name {
                return Some(link);
            }
            env = &link.outer;
        }
        None
    }

    /// Returns this scope with `pattern` bound to `value`.
    fn bind_pattern(&self, pattern: &Pattern, value: Value<'a, M, S>, reached: &Rc<[usize]>) -> Self
    where
        M: Clone,
        S: Clone,
    {
        match (pattern, value) {
            (Pattern::Identifier(name), value) => self.bind(name, value, reached.clone()),
            (Pattern::Tuple(patterns), Value::Tuple(values)) if patterns.len() == values.len() => {
                patterns
                    .iter()
                    .zip(values)
                    .fold(self.clone(), |env, (p, v)| env.bind_pattern(p, v, reached))
            }
            (Pattern::Tuple(patterns), value) => {
                let unknown = Value::Unknown(value.parameter());
                patterns.iter().fold(self.clone(), |env, p| {
                    env.bind_pattern(p, unknown.clone(), reached)
                })
            }
        }
    }
}

/// The state of one abstract evaluation.
struct Abstract<'a, 'r, M, S, F> {
    resolve: &'r F,
    /// Each module's exports, by path.
    modules: HashMap<Vec<String>, Exports<'a, M, S>>,
    /// The modules being opened, innermost last.
    opening: Vec<Vec<String>>,
    /// Every call reached so far.
    introduced: Vec<Introduced<S>>,
    /// The calls the evaluation in progress reaches, as indices into
    /// `introduced`.
    reached: Vec<usize>,
}

impl<'a, M, S, F> Abstract<'a, '_, M, S, F>
where
    F: Fn(&[String]) -> Result<&'a [SourceBinding<M, S>], Error<S>>,
    M: CarriesTrace<S> + Clone,
    S: Clone,
{
    /// Returns `env` extended with `bindings`. With `definitions`, also
    /// applies the functions each binding defines to unknown parameters, and
    /// adds the calls each binding reaches as its own code to it.
    fn bindings(
        &mut self,
        bindings: &'a [SourceBinding<M, S>],
        mut env: Env<'a, M, S>,
        mut definitions: Option<&mut Vec<usize>>,
    ) -> Env<'a, M, S> {
        for binding in bindings {
            match &binding.binding {
                Binding::Open(path) => {
                    for (name, value, reached) in self.module(path).iter() {
                        env = env.bind(name, value.clone(), reached.clone());
                    }
                }
                Binding::Use(path) => {
                    if let Some(name) = path.last() {
                        let module = Value::Module(self.module(path));
                        env = env.bind(name, module, Rc::from([]));
                    }
                }
                Binding::Definition(pattern, expr) => {
                    let outer = mem::take(&mut self.reached);
                    let root = CallPath::empty().enter(expr.span.clone(), None);
                    let value = self.eval(expr, &env, &root);
                    let reached: Rc<[usize]> = self.reached.as_slice().into();
                    if let Some(definitions) = definitions.as_deref_mut() {
                        definitions.extend(reached.iter());
                        self.reached.clear();
                        self.apply_to_parameters(&value, &root, RETURNED_FUNCTIONS);
                        definitions.extend(self.reached.iter());
                    }
                    self.reached = outer;
                    env = env.bind_pattern(pattern, value, &reached);
                }
                Binding::Empty => {}
            }
        }
        env
    }

    /// Returns the exports of the module at `path`: none when it doesn't
    /// resolve or is being opened.
    fn module(&mut self, path: &[String]) -> Exports<'a, M, S> {
        if let Some(exports) = self.modules.get(path) {
            return exports.clone();
        }
        let Ok(bindings) = (self.resolve)(path) else {
            return Rc::new(Vec::new());
        };
        if self.opening.iter().any(|p| p == path) {
            return Rc::new(Vec::new());
        }
        self.opening.push(path.to_vec());
        let env = self.bindings(bindings, Env(None), None);
        self.opening.pop();
        // Only the module's own definitions are exported, not those it opens.
        let own: Vec<&str> = bindings
            .iter()
            .flat_map(|binding| match &binding.binding {
                Binding::Definition(pattern, _) => names(pattern),
                _ => Vec::new(),
            })
            .collect();
        let mut exports = Vec::new();
        let mut link = &env.0;
        while let Some(l) = link {
            if own.contains(&l.name.as_str()) && !exports.iter().any(|(n, _, _)| n == &l.name) {
                exports.push((l.name.clone(), l.value.clone(), l.reached.clone()));
            }
            link = &l.outer.0;
        }
        exports.reverse();
        let exports = Rc::new(exports);
        self.modules.insert(path.to_vec(), exports.clone());
        exports
    }

    /// Returns the abstract value of `expr` in `env`, run inside `path`.
    fn eval(
        &mut self,
        expr: &'a SourceExpr<M, S>,
        env: &Env<'a, M, S>,
        path: &CallPath<S>,
    ) -> Value<'a, M, S> {
        match &expr.expr {
            Expr::Bool(b) => Value::Bool(*b),
            Expr::String(s) => Value::Str(s.clone()),
            Expr::Waveform(_) => expr
                .expr
                .as_const_float()
                .map_or(Value::Unknown(None), Value::Num),
            Expr::Function {
                positional,
                named,
                body,
            } => {
                let named = named
                    .iter()
                    .map(|(name, default)| (name.clone(), self.eval(default, env, path)))
                    .collect();
                Value::Closure(Rc::new(Closure {
                    positional,
                    named,
                    body,
                    env: env.clone(),
                    span: expr.span.clone(),
                }))
            }
            Expr::BuiltIn { name, function } => Value::BuiltIn(name, function),
            Expr::Variable(name) => match env.lookup(name) {
                Some(link) => {
                    self.reached.extend(link.reached.iter());
                    link.value.clone()
                }
                None => Value::Unknown(None),
            },
            Expr::Seq { offset, waveform } => {
                self.eval(offset, env, path);
                self.eval(waveform, env, path);
                Value::Unknown(None)
            }
            Expr::IfThenElse {
                condition,
                then,
                else_,
            } => match self.eval(condition, env, path) {
                Value::Bool(true) => self.eval(then, env, path),
                Value::Bool(false) => self.eval(else_, env, path),
                condition => {
                    let then = self.eval(then, env, path);
                    let else_ = self.eval(else_, env, path);
                    merge(then, else_, condition.parameter())
                }
            },
            Expr::Application {
                function,
                positional,
                named,
            } => {
                let function = self.eval(function, env, path);
                let positional = positional
                    .iter()
                    .map(|argument| self.eval(argument, env, path))
                    .collect();
                let named = named
                    .iter()
                    .map(|(name, argument)| (name.clone(), self.eval(argument, env, path)))
                    .collect();
                self.apply(function, positional, named, expr.span.clone(), path)
            }
            Expr::BoundModule(entries) => {
                let exports = entries
                    .iter()
                    .map(|(name, expr)| (name.clone(), self.eval(expr, env, path), Rc::from([])))
                    .collect();
                Value::Module(Rc::new(exports))
            }
            Expr::Project { module, name } => match self.eval(module, env, path) {
                Value::Module(exports) => match exports.iter().find(|(n, _, _)| n == name) {
                    Some((_, value, reached)) => {
                        self.reached.extend(reached.iter());
                        value.clone()
                    }
                    None => Value::Unknown(None),
                },
                other => Value::Unknown(other.parameter()),
            },
            Expr::Tuple(exprs) => {
                Value::Tuple(exprs.iter().map(|e| self.eval(e, env, path)).collect())
            }
            Expr::List(exprs) => {
                Value::List(exprs.iter().map(|e| self.eval(e, env, path)).collect())
            }
            Expr::Error(_) => Value::Unknown(None),
        }
    }

    /// Returns the result of applying `function` at `site`.
    fn apply(
        &mut self,
        function: Value<'a, M, S>,
        positional: Vec<Value<'a, M, S>>,
        named: Vec<(String, Value<'a, M, S>)>,
        site: Option<Span<S>>,
        path: &CallPath<S>,
    ) -> Value<'a, M, S> {
        match function {
            Value::Closure(closure) => self.call(&closure, positional, named, site, path),
            Value::BuiltIn(name, function) => self.built_in(name, function, positional, site, path),
            Value::Identity => positional
                .into_iter()
                .next()
                .unwrap_or(Value::Unknown(None)),
            Value::OneOf(functions) => {
                let mut result: Option<Value<'a, M, S>> = None;
                for function in functions {
                    let value = self.apply(
                        function,
                        positional.clone(),
                        named.clone(),
                        site.clone(),
                        path,
                    );
                    result = Some(match result {
                        Some(result) => merge(result, value, None),
                        None => value,
                    });
                }
                result.unwrap_or(Value::Unknown(None))
            }
            other => {
                for argument in positional.iter().chain(named.iter().map(|(_, v)| v)) {
                    self.release(argument, path);
                }
                Value::Unknown(other.parameter())
            }
        }
    }

    /// Returns the result of applying `closure` at `site`.
    fn call(
        &mut self,
        closure: &Closure<'a, M, S>,
        positional: Vec<Value<'a, M, S>>,
        named: Vec<(String, Value<'a, M, S>)>,
        site: Option<Span<S>>,
        path: &CallPath<S>,
    ) -> Value<'a, M, S> {
        if positional.len() != closure.positional.len() {
            return Value::Unknown(None);
        }
        let none: Rc<[usize]> = Rc::from([]);
        let mut env = closure.env.clone();
        for (pattern, argument) in closure.positional.iter().zip(positional) {
            env = env.bind_pattern(pattern, argument, &none);
        }
        for (name, default) in &closure.named {
            let value = named
                .iter()
                .find(|(n, _)| n == name)
                .map_or_else(|| default.clone(), |(_, v)| v.clone());
            env = env.bind(name, value, none.clone());
        }
        // The body runs as code written where the function was.
        let inner = path.enter(closure.span.clone(), site);
        self.eval(closure.body, &env, &inner)
    }

    /// Returns the result of applying the built-in `name` at `site`.
    fn built_in(
        &mut self,
        name: &str,
        function: &BuiltInFn<M, S>,
        arguments: Vec<Value<'a, M, S>>,
        site: Option<Span<S>>,
        path: &CallPath<S>,
    ) -> Value<'a, M, S> {
        use Value::{List, Num};
        match (name, arguments.as_slice()) {
            ("slider", _) => {
                self.introduce(&arguments, site, path);
                Value::Unknown(None)
            }
            ("map", [function, List(elements)]) => {
                let mut results = Vec::new();
                for element in elements {
                    results.push(self.apply(
                        function.clone(),
                        vec![element.clone()],
                        Vec::new(),
                        None,
                        path,
                    ));
                }
                List(results)
            }
            ("reduce", [function, initial, List(elements)]) => {
                let mut result = initial.clone();
                for element in elements {
                    result = self.apply(
                        function.clone(),
                        vec![result, element.clone()],
                        Vec::new(),
                        None,
                        path,
                    );
                }
                result
            }
            ("map", [function, list]) => {
                let element = Value::Unknown(list.parameter());
                self.apply(function.clone(), vec![element], Vec::new(), None, path);
                Value::Unknown(list.parameter())
            }
            ("reduce", [function, initial, list]) => {
                let element = Value::Unknown(list.parameter());
                let args = vec![initial.clone(), element];
                self.apply(function.clone(), args, Vec::new(), None, path);
                Value::Unknown(list.parameter())
            }
            ("nth", [Num(index), List(elements)])
                if index.fract() == 0.0 && *index >= 0.0 && (*index as usize) < elements.len() =>
            {
                elements[*index as usize].clone()
            }
            ("concat", [List(lists)]) if lists.iter().all(|l| matches!(l, List(_))) => List(
                lists
                    .iter()
                    .flat_map(|l| match l {
                        List(elements) => elements.clone(),
                        _ => Vec::new(),
                    })
                    .collect(),
            ),
            ("debug", _) => Value::Identity,
            (name, _)
                if COMPUTED.contains(&name)
                    && arguments
                        .iter()
                        .all(|a| matches!(a, Num(_) | Value::Str(_) | Value::Bool(_))) =>
            {
                compute(function, &arguments)
            }
            _ => {
                for argument in &arguments {
                    self.release(argument, path);
                }
                Value::Unknown(arguments.iter().find_map(Value::parameter))
            }
        }
    }

    /// Records the `slider` call at `site` applied to `arguments`, unless
    /// those arguments make an invalid slider.
    fn introduce(
        &mut self,
        arguments: &[Value<'a, M, S>],
        site: Option<Span<S>>,
        path: &CallPath<S>,
    ) {
        let parameter = |value: &Value<'a, M, S>| value.parameter().map(|p| (*p).clone());
        let label = match arguments.first() {
            Some(Value::Str(label)) => Ok(label.clone()),
            Some(other) => Err(parameter(other)),
            None => return,
        };
        let number = |index: usize, part: &'static str| match arguments.get(index) {
            Some(Value::Num(value)) => Ok(*value),
            other => Err(Open {
                part,
                parameter: other.and_then(parameter),
            }),
        };
        let spec = (|| {
            let default = number(1, "default")?;
            let (min, max) = if arguments.len() > 2 {
                (number(2, "range")?, number(3, "range")?)
            } else {
                (0.0, 1.0)
            };
            let curve = match arguments.get(4) {
                None => Curve::Linear,
                Some(Value::Str(name)) => match Curve::named(name) {
                    Some(curve) => curve,
                    None => return Ok(None),
                },
                Some(other) => {
                    return Err(Open {
                        part: "curve",
                        parameter: parameter(other),
                    });
                }
            };
            Ok(SliderSpec::new(default, min, max, curve).ok())
        })();
        let spec = match spec {
            Ok(Some(spec)) => Ok(spec),
            // Evaluation reports an invalid spec.
            Ok(None) => return,
            Err(open) => Err(open),
        };
        self.reached.push(self.introduced.len());
        self.introduced.push(Introduced {
            label,
            spec,
            site,
            path: path.clone(),
        });
    }

    /// Applies each function `value` holds to unknown arguments, for the
    /// calls they reach.
    fn release(&mut self, value: &Value<'a, M, S>, path: &CallPath<S>) {
        match value {
            Value::Closure(closure) => {
                let unknowns = vec![Value::Unknown(None); closure.positional.len()];
                self.call(closure, unknowns, Vec::new(), None, path);
            }
            Value::Tuple(values) | Value::List(values) | Value::OneOf(values) => {
                for value in values {
                    self.release(value, path);
                }
            }
            _ => {}
        }
    }

    /// Applies each function `value` holds to its own parameters, unknown,
    /// and so for the functions those return, `depth` more times.
    fn apply_to_parameters(&mut self, value: &Value<'a, M, S>, path: &CallPath<S>, depth: usize) {
        match value {
            Value::Closure(closure) => {
                let parameter = |name: &str| {
                    Value::Unknown(Some(Rc::new(Parameter {
                        name: name.to_string(),
                        function: closure.span.clone(),
                    })))
                };
                let positional = closure
                    .positional
                    .iter()
                    .map(|pattern| unknown_pattern(pattern, &parameter))
                    .collect();
                let named = closure
                    .named
                    .iter()
                    .map(|(name, _)| (name.clone(), parameter(name)))
                    .collect();
                let result = self.call(closure, positional, named, None, path);
                if depth > 0 {
                    self.apply_to_parameters(&result, path, depth - 1);
                }
            }
            Value::Tuple(values) | Value::List(values) | Value::OneOf(values) => {
                for value in values {
                    self.apply_to_parameters(value, path, depth);
                }
            }
            _ => {}
        }
    }
}

/// Returns the value `pattern` matches when each name it binds is the
/// parameter `parameter` makes of it.
fn unknown_pattern<'a, M, S>(
    pattern: &Pattern,
    parameter: &impl Fn(&str) -> Value<'a, M, S>,
) -> Value<'a, M, S> {
    match pattern {
        Pattern::Identifier(name) => parameter(name),
        Pattern::Tuple(patterns) => Value::Tuple(
            patterns
                .iter()
                .map(|p| unknown_pattern(p, parameter))
                .collect(),
        ),
    }
}

/// Returns the names `pattern` binds.
fn names(pattern: &Pattern) -> Vec<&str> {
    match pattern {
        Pattern::Identifier(name) => vec![name.as_str()],
        Pattern::Tuple(patterns) => patterns.iter().flat_map(names).collect(),
    }
}

/// Returns the value that is `a` or `b`, as chosen by a condition that
/// depends on `parameter`.
fn merge<'a, M, S>(
    a: Value<'a, M, S>,
    b: Value<'a, M, S>,
    parameter: Option<Rc<Parameter<S>>>,
) -> Value<'a, M, S> {
    use Value::*;
    match (a, b) {
        (Num(x), Num(y)) if x == y => Num(x),
        (Str(x), Str(y)) if x == y => Str(x),
        (Bool(x), Bool(y)) if x == y => Bool(x),
        (Tuple(xs), Tuple(ys)) if xs.len() == ys.len() => Tuple(
            xs.into_iter()
                .zip(ys)
                .map(|(x, y)| merge(x, y, parameter.clone()))
                .collect(),
        ),
        (List(xs), List(ys)) if xs.len() == ys.len() => List(
            xs.into_iter()
                .zip(ys)
                .map(|(x, y)| merge(x, y, parameter.clone()))
                .collect(),
        ),
        (a, b) if a.holds_functions() || b.holds_functions() => {
            let mut values = Vec::new();
            for value in [a, b] {
                match value {
                    OneOf(more) => values.extend(more),
                    value => values.push(value),
                }
            }
            OneOf(values)
        }
        (a, b) => Unknown(parameter.or(a.parameter()).or(b.parameter())),
    }
}

/// Returns the result of the built-in `function` applied to the known
/// `arguments`.
fn compute<'a, M, S>(function: &BuiltInFn<M, S>, arguments: &[Value<'a, M, S>]) -> Value<'a, M, S>
where
    M: CarriesTrace<S>,
    S: Clone,
{
    let arguments: Vec<SourceExpr<M, S>> = arguments
        .iter()
        .map(|argument| {
            SourceExpr::from(match argument {
                Value::Num(value) => Expr::float(*value),
                Value::Str(value) => Expr::String(value.clone()),
                Value::Bool(value) => Expr::Bool(*value),
                _ => Expr::Error(String::new()),
            })
        })
        .collect();
    let call = BuiltInCall::new(None, Trace::new::<M>(None, None), &arguments);
    match (function.0)(arguments, &call) {
        Ok(result) => match result.expr {
            Expr::Bool(value) => Value::Bool(value),
            Expr::String(value) => Value::Str(value),
            other => other
                .as_const_float()
                .map_or(Value::Unknown(None), Value::Num),
        },
        Err(_) => Value::Unknown(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::{parse_module, parse_program};

    /// Returns a binding of `name` to a built-in that combines two numbers
    /// with `op`, as `*` and `>` do.
    fn binary(name: &str, op: fn(f32, f32) -> Expr<u32, ()>) -> SourceBinding<u32, ()> {
        let function = BuiltInFn::new(
            move |arguments: Vec<SourceExpr<u32, ()>>, _: &BuiltInCall<()>| match arguments
                .as_slice()
            {
                [a, b] => match (a.expr.as_const_float(), b.expr.as_const_float()) {
                    (Some(a), Some(b)) => Ok(SourceExpr::from(op(a, b))),
                    _ => Err(Error::eval_here("expected numbers")),
                },
                _ => Err(Error::eval_here("expected two arguments")),
            },
        );
        built_in(name, function)
    }

    /// Returns a binding of `name` to a built-in the analysis never applies.
    fn unapplied(name: &str) -> SourceBinding<u32, ()> {
        built_in(
            name,
            BuiltInFn::new(|_, _| Err(Error::eval_here("not applied"))),
        )
    }

    /// Returns a binding of `name` to `function`.
    fn built_in(name: &str, function: BuiltInFn<u32, ()>) -> SourceBinding<u32, ()> {
        let expr = SourceExpr::from(Expr::BuiltIn {
            name: name.to_string(),
            function,
        });
        Binding::Definition(Pattern::Identifier(name.to_string()), expr).into()
    }

    /// Returns the analysis of `program` under the stand-in built-ins and then
    /// the bindings `module` defines.
    fn analysis(module: &str, program: &str, keys: bool) -> Analysis<()> {
        let mut bindings = vec![
            unapplied("slider"),
            unapplied("map"),
            binary("*", |a, b| Expr::float(a * b)),
            binary(">", |a, b| Expr::Bool(a > b)),
        ];
        bindings.extend(parse_module::<u32, _>(module, ()).unwrap().0);
        let expr = parse_program::<u32, _>(program, ()).unwrap();
        analyze(
            |_: &[String]| Err(Error::eval_here("no modules")),
            &bindings,
            &expr,
            keys,
        )
    }

    /// Returns each call's label and default, or the part of its spec that
    /// isn't known and the parameter that part depends on.
    fn calls(calls: &[Introduced<()>]) -> Vec<String> {
        calls
            .iter()
            .map(|call| {
                let label = call.label.clone().unwrap_or_default();
                match &call.spec {
                    Ok(spec) => format!("{label} = {}", spec.default),
                    Err(open) => {
                        let parameter = open.parameter.as_ref().map_or("?", |p| p.name.as_str());
                        format!("{label}: {} from {parameter}", open.part)
                    }
                }
            })
            .collect()
    }

    #[test]
    fn a_condition_that_isnt_known_takes_both_branches() {
        let program = "fn(note) => if note > 60 then slider(\"a\", 0.5) else slider(\"b\", 0.25)";
        assert_eq!(
            calls(&analysis("", program, true).program),
            ["a = 0.5", "b = 0.25"]
        );
    }

    #[test]
    fn a_known_condition_takes_one_branch() {
        let program = "if 2 > 1 then slider(\"a\", 0.5) else slider(\"b\", 0.25)";
        assert_eq!(calls(&analysis("", program, false).program), ["a = 0.5"]);
    }

    #[test]
    fn a_callers_argument_is_known_where_it_is_given() {
        let analysis = analysis("f = fn(d) => slider(\"q\", d * 2);", "f(0.25)", false);
        assert_eq!(calls(&analysis.program), ["q = 0.5"]);
        // Applied to a parameter no caller gives, the default depends on it.
        assert_eq!(calls(&analysis.definitions), ["q: default from d"]);
    }

    #[test]
    fn a_function_a_built_in_is_given_is_applied() {
        let program = "map(fn(x) => slider(\"q\", x), [0.25, 0.5])";
        assert_eq!(
            calls(&analysis("", program, false).program),
            ["q = 0.25", "q = 0.5"]
        );
    }

    #[test]
    fn a_bindings_calls_are_reached_through_its_name() {
        let module = "w = slider(\"q\", 0.5);";
        assert_eq!(calls(&analysis(module, "w", false).program), ["q = 0.5"]);
        assert_eq!(
            calls(&analysis(module, "1", false).program),
            Vec::<String>::new()
        );
    }
}
