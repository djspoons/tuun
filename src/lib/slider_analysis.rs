//! A static check of how a program uses sliders: which sliders it can reach,
//! on any branch and any note of a keys instrument; which binding provides
//! each; and whether every call of one slider gives one spec, fixed where its
//! provider is.
//!
//! The check is an abstract evaluation. A value the program has when it loads
//! is computed exactly, applying the real built-ins. A value that depends on
//! the note a keys instrument plays, or on what a listing binding's callers
//! give it, is unknown, and records where it comes from. A condition that
//! isn't known takes both branches.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::rc::{Rc, Weak};

use crate::builtins;
use crate::eval::BuiltInCall;
use crate::expr::{Binding, BuiltInFn, Error, Expr, Pattern, SourceBinding, SourceExpr, Span};
use crate::slider::{Curve, Listing, Listings, Provider, SliderSpec};
use crate::trace::{CarriesTrace, Trace};
use crate::waveform::{ShareId, Waveform};

/// A slider a program can reach: the id of its provider, its label, and its
/// spec.
#[derive(Clone, Debug, PartialEq)]
pub struct Reached {
    /// The id of the listing binding that provides it, or of the played
    /// program when none does.
    pub provider: usize,
    pub label: String,
    pub spec: SliderSpec,
}

/// What [`analyze`] finds.
#[derive(Debug)]
pub struct Analysis<S> {
    /// Each slider the program can reach, or provides when it is a
    /// [`Role::Provides`] function, once, in the order the text reaches
    /// its first call: by where the outermost call in progress around it was
    /// written, then the next one in, and so on, and last where the `slider`
    /// call itself was written.
    pub reached: Vec<Reached>,
    /// The errors in how the program uses sliders.
    pub errors: Vec<Error<S>>,
}

/// A program as [`analyze`] checks it.
pub struct Program<'a, M, S> {
    /// The bindings the program is evaluated under.
    pub bindings: &'a [SourceBinding<M, S>],
    /// The program's text, parsed.
    pub expr: &'a SourceExpr<M, S>,
    /// What the program's value is for.
    pub role: Role,
    /// The listing bindings code is written in.
    pub listings: Listings<S>,
    /// The id of the played program as a provider. When the program's own
    /// text lists sliders, its listing has this id too.
    pub played: usize,
    /// The name of the provider with each id in messages.
    pub name: &'a dyn Fn(usize) -> String,
    /// Where the annotation of the listing with each id is.
    pub annotation: &'a dyn Fn(usize) -> Option<Span<S>>,
}

/// What a program's value is for, as [`analyze`] checks it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Role {
    /// It plays.
    Plays,
    /// It is a keys instrument, applied to each note.
    Keys,
    /// It is a function that provides the sliders its own listing lists to
    /// its callers, checked as a listing binding is from its value alone.
    Provides,
}

/// The most evaluation steps a check takes before it gives up.
const FUEL: u64 = 20_000_000;

/// Returns the sliders `program` can reach and the errors in how it uses
/// them.
///
/// A slider's provider is the listing binding whose code, or code it runs, is
/// running at the call and which lists the call's label, or the played program
/// when none does, as in evaluation (see
/// [`crate::eval::DynamicScope::provider`]). The errors are:
///
/// - a call whose label isn't a string written in it;
/// - a call reached through two listings of its label, reported at each
///   listing;
/// - a call whose spec differs from that of an earlier call of the same
///   slider;
/// - a call the played program provides whose spec depends on the note a keys
///   instrument plays;
/// - a listing binding a call of whose provided sliders has a spec that
///   depends on what its callers give it;
/// - a listing binding that applies a function its callers give it while it
///   provides sliders;
/// - a call whose spec is invalid.
///
/// Expects a program the type checker accepts.
///
/// # Example
///
/// ```
/// use tuun::builtins;
/// use tuun::expr::Error;
/// use tuun::parser;
/// use tuun::slider::Listings;
/// use tuun::slider_analysis::{self, Program, Role};
///
/// let mut prelude = Vec::new();
/// builtins::add_bindings::<u32, u32>(&mut prelude);
/// let expr = parser::parse_program("time * slider(\"mix\", 0.5)", 0u32).unwrap();
/// let program = Program {
///     bindings: &prelude,
///     expr: &expr,
///     role: Role::Plays,
///     listings: Listings::new(|_| None),
///     played: 7,
///     name: &|_| "the program".to_string(),
///     annotation: &|_| None,
/// };
/// let analysis = slider_analysis::analyze(|_| Err(Error::eval_here("no modules")), &program);
/// assert!(analysis.errors.is_empty());
/// assert_eq!(analysis.reached[0].label, "mix");
/// assert_eq!(analysis.reached[0].provider, 7);
/// ```
pub fn analyze<'a, M, S, F>(resolve: F, program: &Program<'a, M, S>) -> Analysis<S>
where
    F: Fn(&[String]) -> Result<&'a [SourceBinding<M, S>], Error<S>>,
    M: Clone + Display + Debug + CarriesTrace<S>,
    S: Clone + Debug + PartialEq,
{
    let mut analyzer = Analyzer {
        resolve,
        listings: program.listings.clone(),
        tops: Vec::new(),
        modules: HashMap::new(),
        specs: HashMap::new(),
        reached: Vec::new(),
        errors: Vec::new(),
        escaped: Vec::new(),
        fuel: FUEL,
        name: program.name,
        annotation: program.annotation,
        played: program.played,
    };
    analyzer.run(program);
    if analyzer.fuel == 0 {
        analyzer.report(Error::eval(
            "this program takes too long to check how it uses sliders",
            program.expr.span.clone(),
        ));
    }
    analyzer.reached.sort_by(|(a, _), (b, _)| a.cmp(b));
    Analysis {
        reached: analyzer
            .reached
            .into_iter()
            .map(|(_, reached)| reached)
            .collect(),
        errors: analyzer.errors,
    }
}

/// Where an unknown value comes from.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Origin {
    /// The key a keys instrument's note plays.
    Key,
    /// The velocity a keys instrument's note plays with.
    Velocity,
    /// The parameter `name` of a function written in the listing binding with
    /// id `listing`, which that binding's callers give.
    Parameter { listing: usize, name: String },
    /// Something the check doesn't compute, though it is known when the
    /// program loads.
    Opaque,
}

/// A path of an evaluation that fails, so it plays nothing.
#[derive(Clone, Copy, Debug)]
struct Dead;

type Outcome<'a, M, S> = Result<Value<'a, M, S>, Dead>;

/// What the check knows about a value.
#[derive(Clone)]
enum Value<'a, M, S> {
    /// A value known exactly, holding no user function.
    Known(Rc<SourceExpr<M, S>>),
    /// A user function.
    Closure(Rc<Closure<'a, M, S>>),
    Tuple(Rc<Vec<Value<'a, M, S>>>),
    List(Rc<Vec<Value<'a, M, S>>>),
    /// A module, by its exported entries.
    Module(Rc<Entries<'a, M, S>>),
    Unknown(Rc<Unknown<'a, M, S>>),
}

/// A value the check doesn't know: it comes from `origins` (never empty), and
/// if it is a function that runs user code, or holds one, that function is
/// one of `callees`.
struct Unknown<'a, M, S> {
    origins: Vec<Origin>,
    callees: Vec<Callee<'a, M, S>>,
}

/// A function that may run user code.
#[derive(Clone)]
enum Callee<'a, M, S> {
    Closure(Rc<Closure<'a, M, S>>),
    /// A built-in the check applies itself (see [`SPECIAL`]).
    BuiltIn(Rc<SourceExpr<M, S>>),
    /// A function the callers of the listing binding with id `listing` give
    /// it for its parameter `name`.
    Caller {
        listing: usize,
        name: String,
    },
}

/// A user function: its parameters, with named defaults evaluated, its body,
/// the environment it was written in, and where it was written.
struct Closure<'a, M, S> {
    positional: &'a [Pattern],
    named: Vec<(String, Value<'a, M, S>)>,
    body: &'a SourceExpr<M, S>,
    env: Env<'a, M, S>,
    code: Option<Span<S>>,
}

/// The names visible to a piece of code.
#[derive(Clone)]
enum Env<'a, M, S> {
    /// The first `len` entries of a top level: a file's or a module's bindings.
    Top(Weak<Top<'a, M, S>>, usize),
    Local(Rc<Local<'a, M, S>>),
}

struct Local<'a, M, S> {
    name: String,
    value: Value<'a, M, S>,
    next: Env<'a, M, S>,
}

/// The entries of a top level, each with its position, by name.
struct Top<'a, M, S> {
    names: RefCell<HashMap<String, Positioned<'a, M, S>>>,
    len: Cell<usize>,
}

/// Names and what each is bound to, in order.
type Entries<'a, M, S> = Vec<(String, Entry<'a, M, S>)>;

/// What one name is bound to at each position it is bound at, in order.
type Positioned<'a, M, S> = Vec<(usize, Entry<'a, M, S>)>;

/// What a name is bound to: a value, or the name `.1` of a definition whose
/// value is computed when first read.
#[derive(Clone)]
enum Entry<'a, M, S> {
    Value(Value<'a, M, S>),
    Lazy(Rc<Thunk<'a, M, S>>, String),
}

/// A top-level definition, evaluated when one of its names is first read.
struct Thunk<'a, M, S> {
    pattern: &'a Pattern,
    state: RefCell<ThunkState<'a, M, S>>,
}

enum ThunkState<'a, M, S> {
    Pending {
        expr: &'a SourceExpr<M, S>,
        env: Env<'a, M, S>,
    },
    Forcing,
    Done(Outcome<'a, M, S>),
}

/// What an evaluation reads from the point it has reached: which listings
/// provide each label, where the calls in progress were written, and whether
/// it checks the whole program or one listing binding alone.
#[derive(Clone)]
struct Scope {
    /// The ids of the listings that provide each label here, outermost first.
    providers: Rc<Vec<(String, Vec<usize>)>>,
    sites: Rc<Vec<usize>>,
    mode: Mode,
}

#[derive(Clone, PartialEq)]
enum Mode {
    /// Checking the program: every slider reached counts.
    Program,
    /// Checking a listing binding from its definition alone: only the sliders
    /// it provides count.
    Alone(Listing),
}

/// An application as the check sees it: where it was written, the label
/// argument as written, and where the call was written for text order.
struct Call<'a, M, S> {
    span: Option<Span<S>>,
    label: Option<&'a SourceExpr<M, S>>,
    site: Option<usize>,
}

impl<'a, M, S: Clone> Call<'a, M, S> {
    /// Returns a call written nowhere.
    fn nowhere() -> Call<'a, M, S> {
        Call {
            span: None,
            label: None,
            site: None,
        }
    }

    /// Returns a call a built-in applied at this call makes.
    fn applied(&self) -> Call<'a, M, S> {
        Call {
            span: self.span.clone(),
            label: None,
            site: None,
        }
    }
}

/// The built-ins the check doesn't apply directly: those that apply functions,
/// and `slider`.
pub const SPECIAL: &[&str] = &["slider", "map", "reduce", "unfold", "share"];

/// The other built-ins whose result can hold a function an argument holds:
/// those whose signatures are polymorphic. Any other built-in's result holds
/// no function that runs user code, except the identity `debug(label)`
/// returns (see [`crate::builtins::is_debug_result`]), which the check applies itself so that nothing prints.
pub const PASSING: &[&str] = &["nth", "concat"];

struct Analyzer<'a, 'n, M, S, F> {
    resolve: F,
    /// The listing bindings code is written in.
    listings: Listings<S>,
    /// Every top level built, kept alive for environments that refer to it.
    tops: Vec<Rc<Top<'a, M, S>>>,
    /// Each module's exports, by path, or `None` while it is being built.
    modules: HashMap<Vec<String>, Option<Rc<Entries<'a, M, S>>>>,
    /// The spec of the first call of each slider, by its provider's id.
    specs: HashMap<(usize, String), SliderSpec>,
    /// Each slider the program reaches, with the text order of its first call.
    reached: Vec<(Vec<usize>, Reached)>,
    errors: Vec<Error<S>>,
    /// The user functions given to functions a listing binding's callers
    /// give it, which those may apply to anything.
    escaped: Vec<Rc<Closure<'a, M, S>>>,
    fuel: u64,
    name: &'n dyn Fn(usize) -> String,
    annotation: &'n dyn Fn(usize) -> Option<Span<S>>,
    /// The id of the played program as a provider.
    played: usize,
}

impl<'a, M, S, F> Analyzer<'a, '_, M, S, F>
where
    F: Fn(&[String]) -> Result<&'a [SourceBinding<M, S>], Error<S>>,
    M: Clone + Display + Debug + CarriesTrace<S>,
    S: Clone + Debug + PartialEq,
{
    /// Checks `program`, gathering what it finds.
    fn run(&mut self, program: &Program<'a, M, S>) {
        let top = self.top();
        if self.build(program.bindings, &top, None).is_err() {
            return;
        }
        let env = Env::Top(Rc::downgrade(&top), top.len.get());
        let code = program.expr.span.as_ref();
        let scope = Scope::new(Mode::Program).enter(code, None, &self.listings);
        let Ok(value) = self.eval(program.expr, &env, &scope) else {
            return;
        };
        match program.role {
            Role::Plays => {}
            Role::Keys => {
                // Each note applies the instrument afresh, outside the
                // program's evaluation.
                let note = vec![unknown(vec![Origin::Key]), unknown(vec![Origin::Velocity])];
                let _ = self.apply(
                    value,
                    note,
                    Vec::new(),
                    &Call::nowhere(),
                    &Scope::new(Mode::Program),
                );
            }
            Role::Provides => {
                if let Some(listing) = code.and_then(|code| self.listings.find(code)) {
                    self.check_alone(listing, &value);
                }
            }
        }
    }

    /// Returns a new, empty top level.
    fn top(&mut self) -> Rc<Top<'a, M, S>> {
        let top = Rc::new(Top {
            names: RefCell::new(HashMap::new()),
            len: Cell::new(0),
        });
        self.tops.push(top.clone());
        top
    }

    /// Adds the entries `bindings` define and open to `top`. Adds the entries
    /// of the definitions themselves to `own` too.
    fn build(
        &mut self,
        bindings: &'a [SourceBinding<M, S>],
        top: &Rc<Top<'a, M, S>>,
        mut own: Option<&mut Entries<'a, M, S>>,
    ) -> Result<(), Dead> {
        for binding in bindings {
            match &binding.binding {
                Binding::Open(path) => {
                    for (name, entry) in self.module(path)?.iter() {
                        top.push(name, entry.clone());
                    }
                }
                Binding::Use(path) => {
                    let name = path.last().ok_or(Dead)?;
                    let exports = self.module(path)?;
                    let mut entries: Entries<'a, M, S> = Vec::new();
                    for (name, entry) in exports.iter() {
                        match entries.iter_mut().find(|(n, _)| n == name) {
                            Some(existing) => existing.1 = entry.clone(),
                            None => entries.push((name.clone(), entry.clone())),
                        }
                    }
                    top.push(name, Entry::Value(Value::Module(Rc::new(entries))));
                }
                Binding::Definition(pattern, expr) => {
                    let thunk = Rc::new(Thunk {
                        pattern,
                        state: RefCell::new(ThunkState::Pending {
                            expr,
                            env: Env::Top(Rc::downgrade(top), top.len.get()),
                        }),
                    });
                    let mut names = Vec::new();
                    pattern_names(pattern, &mut names);
                    for name in names {
                        let entry = Entry::Lazy(thunk.clone(), name.to_string());
                        if let Some(own) = own.as_deref_mut() {
                            own.push((name.to_string(), entry.clone()));
                        }
                        top.push(name, entry);
                    }
                }
                Binding::Empty => {}
            }
        }
        Ok(())
    }

    /// Returns the entries the module at `path` defines.
    fn module(&mut self, path: &[String]) -> Result<Rc<Entries<'a, M, S>>, Dead> {
        match self.modules.get(path) {
            Some(Some(exports)) => return Ok(exports.clone()),
            // Opened from itself.
            Some(None) => return Err(Dead),
            None => {}
        }
        let bindings = (self.resolve)(path).map_err(|_| Dead)?;
        self.modules.insert(path.to_vec(), None);
        let top = self.top();
        let mut own = Vec::new();
        self.build(bindings, &top, Some(&mut own))?;
        let exports = Rc::new(own);
        self.modules.insert(path.to_vec(), Some(exports.clone()));
        Ok(exports)
    }

    /// Uses up one evaluation step, or fails once there are none left.
    fn tick(&mut self) -> Result<(), Dead> {
        if self.fuel == 0 {
            return Err(Dead);
        }
        self.fuel -= 1;
        Ok(())
    }

    /// Evaluates `expr` in `env` and `scope`.
    fn eval(
        &mut self,
        expr: &'a SourceExpr<M, S>,
        env: &Env<'a, M, S>,
        scope: &Scope,
    ) -> Outcome<'a, M, S> {
        self.tick()?;
        match &expr.expr {
            Expr::Bool(_)
            | Expr::String(_)
            | Expr::Waveform(_)
            | Expr::BuiltIn { .. }
            | Expr::BoundModule(_) => Ok(known(expr.expr.clone())),
            Expr::Function {
                positional,
                named,
                body,
            } => {
                let mut defaults = Vec::with_capacity(named.len());
                for (name, default) in named {
                    defaults.push((name.clone(), self.eval(default, env, scope)?));
                }
                Ok(Value::Closure(Rc::new(Closure {
                    positional,
                    named: defaults,
                    body,
                    env: env.clone(),
                    code: expr.span.clone(),
                })))
            }
            Expr::Variable(name) => {
                let entry = env.lookup(name).ok_or(Dead)?;
                self.entry(&entry)
            }
            Expr::Seq { offset, waveform } => {
                let offset = self.eval(offset, env, scope)?;
                let waveform = self.eval(waveform, env, scope)?;
                Ok(match (exact(&offset), exact(&waveform)) {
                    (Some(offset), Some(waveform)) => known(Expr::Seq {
                        offset: Box::new(offset),
                        waveform: Box::new(waveform),
                    }),
                    _ => widen(&[offset, waveform], &[]),
                })
            }
            Expr::Project { module, name } => match self.eval(module, env, scope)? {
                Value::Module(entries) => {
                    let (_, entry) = entries.iter().find(|(n, _)| n == name).ok_or(Dead)?;
                    self.entry(entry)
                }
                Value::Known(module) => match &module.expr {
                    Expr::BoundModule(entries) => entries
                        .iter()
                        .find(|(n, _)| n == name)
                        .map(|(_, value)| known(value.expr.clone()))
                        .ok_or(Dead),
                    _ => Err(Dead),
                },
                value @ Value::Unknown(_) => Ok(value),
                _ => Err(Dead),
            },
            Expr::IfThenElse {
                condition,
                then,
                else_,
            } => match self.eval(condition, env, scope)? {
                Value::Known(condition) => match condition.expr {
                    Expr::Bool(true) => self.eval(then, env, scope),
                    Expr::Bool(false) => self.eval(else_, env, scope),
                    _ => Err(Dead),
                },
                Value::Unknown(condition) => {
                    let then = self.eval(then, env, scope);
                    let else_ = self.eval(else_, env, scope);
                    Ok(join(then, else_, &condition.origins)?)
                }
                _ => Err(Dead),
            },
            Expr::Application {
                function,
                positional,
                named,
            } => {
                let f = self.eval(function, env, scope)?;
                let mut arguments = Vec::with_capacity(positional.len());
                for argument in positional {
                    arguments.push(self.eval(argument, env, scope)?);
                }
                let mut named_arguments = Vec::with_capacity(named.len());
                for (name, argument) in named {
                    named_arguments.push((name.clone(), self.eval(argument, env, scope)?));
                }
                let call = Call {
                    span: expr.span.clone(),
                    label: positional.first(),
                    site: call_site(expr, function),
                };
                self.apply(f, arguments, named_arguments, &call, scope)
            }
            Expr::Tuple(exprs) | Expr::List(exprs) => {
                let mut values = Vec::with_capacity(exprs.len());
                for expr in exprs {
                    values.push(self.eval(expr, env, scope)?);
                }
                Ok(match &expr.expr {
                    Expr::Tuple(_) => Value::Tuple(Rc::new(values)),
                    _ => Value::List(Rc::new(values)),
                })
            }
            Expr::Error(_) => Err(Dead),
        }
    }

    /// Returns the value `entry` binds.
    fn entry(&mut self, entry: &Entry<'a, M, S>) -> Outcome<'a, M, S> {
        match entry {
            Entry::Value(value) => Ok(value.clone()),
            Entry::Lazy(thunk, name) => {
                let value = self.force(thunk)?;
                select(thunk.pattern, value, name)
            }
        }
    }

    /// Returns the value of `thunk`'s definition, evaluating it the first time.
    ///
    /// A listing binding is checked from its definition alone once its value
    /// is known.
    fn force(&mut self, thunk: &Rc<Thunk<'a, M, S>>) -> Outcome<'a, M, S> {
        let pending = {
            let mut state = thunk.state.borrow_mut();
            match &*state {
                ThunkState::Done(outcome) => return outcome.clone(),
                // Defined in terms of itself.
                ThunkState::Forcing => return Err(Dead),
                ThunkState::Pending { .. } => std::mem::replace(&mut *state, ThunkState::Forcing),
            }
        };
        let ThunkState::Pending { expr, env } = pending else {
            unreachable!("the state was pending");
        };
        // A top-level definition is evaluated once, whoever reads it, so it
        // runs in no one's scope.
        let code = expr.span.as_ref();
        let scope = Scope::new(Mode::Program).enter(code, None, &self.listings);
        let outcome = self.eval(expr, &env, &scope);
        *thunk.state.borrow_mut() = ThunkState::Done(outcome.clone());
        if let (Ok(value), Some(listing)) = (&outcome, code.and_then(|c| self.listings.find(c))) {
            self.check_alone(listing, value);
        }
        outcome
    }

    /// Checks the listing binding `listing` from its value `value` alone:
    /// applies each function it holds, and each function those return, to
    /// parameters its callers give.
    fn check_alone(&mut self, listing: Listing, value: &Value<'a, M, S>) {
        // Another listing may be checked alone while this one is.
        let outer = std::mem::take(&mut self.escaped);
        let mut queue = Vec::new();
        exposed(value, &mut queue);
        let mut seen: Vec<Rc<Closure<'a, M, S>>> = Vec::new();
        while let Some(closure) = queue.pop() {
            if seen.iter().any(|c| Rc::ptr_eq(c, &closure)) {
                continue;
            }
            seen.push(closure.clone());
            let arguments = closure
                .positional
                .iter()
                .map(|pattern| parameter(pattern, listing.id))
                .collect();
            let named = closure
                .named
                .iter()
                .map(|(name, _)| (name.clone(), given(listing.id, name)))
                .collect();
            let scope = Scope::new(Mode::Alone(listing.clone()));
            if let Ok(result) = self.enter(&closure, arguments, named, &Call::nowhere(), &scope) {
                exposed(&result, &mut queue);
            }
            queue.append(&mut self.escaped);
        }
        self.escaped = outer;
    }

    /// Applies `f` to `positional` and `named` in a call `call`, in `scope`.
    fn apply(
        &mut self,
        f: Value<'a, M, S>,
        positional: Vec<Value<'a, M, S>>,
        named: Vec<(String, Value<'a, M, S>)>,
        call: &Call<'a, M, S>,
        scope: &Scope,
    ) -> Outcome<'a, M, S> {
        match f {
            Value::Closure(closure) => self.enter(&closure, positional, named, call, scope),
            Value::Known(f) => {
                let Expr::BuiltIn { name, function } = &f.expr else {
                    return Err(Dead);
                };
                if !named.is_empty() {
                    return Err(Dead);
                }
                match name.as_str() {
                    "slider" => self.slider(positional, call, scope),
                    "map" => self.map(positional, call, scope),
                    "reduce" => self.reduce(positional, call, scope),
                    "unfold" => self.unfold(positional, call, scope),
                    "share" => self.share(positional, call, scope),
                    _ if builtins::is_debug_result(name) => {
                        let [value] = <[_; 1]>::try_from(positional).map_err(|_| Dead)?;
                        Ok(value)
                    }
                    _ => builtin(name, function, positional, call),
                }
            }
            Value::Unknown(f) => {
                self.check_applied_parameter(&f, call, scope);
                // A function none of the callees may be runs no user code,
                // though it may hand back an argument.
                let mut outcome = Ok(widen(&positional, &f.origins));
                for callee in &f.callees {
                    let result = match callee {
                        Callee::Closure(closure) => {
                            self.enter(closure, positional.clone(), named.clone(), call, scope)
                        }
                        Callee::BuiltIn(builtin) => self.apply(
                            Value::Known(builtin.clone()),
                            positional.clone(),
                            named.clone(),
                            call,
                            scope,
                        ),
                        Callee::Caller { listing, name } => {
                            // The caller's function may apply any function it
                            // is given, and return any function.
                            for argument in positional.iter().chain(named.iter().map(|(_, v)| v)) {
                                exposed(argument, &mut self.escaped);
                            }
                            let origin = Origin::Parameter {
                                listing: *listing,
                                name: name.clone(),
                            };
                            let Value::Unknown(result) = widen(&positional, &[origin]) else {
                                unreachable!("widening makes an unknown value");
                            };
                            let mut callees = result.callees.clone();
                            callees.push(callee.clone());
                            Ok(Value::Unknown(Rc::new(Unknown {
                                origins: result.origins.clone(),
                                callees,
                            })))
                        }
                    };
                    outcome = join(outcome, result, &f.origins);
                }
                outcome
            }
            _ => Err(Dead),
        }
    }

    /// Applies the user function `closure` as [`Analyzer::apply`] does.
    fn enter(
        &mut self,
        closure: &Rc<Closure<'a, M, S>>,
        positional: Vec<Value<'a, M, S>>,
        named: Vec<(String, Value<'a, M, S>)>,
        call: &Call<'a, M, S>,
        scope: &Scope,
    ) -> Outcome<'a, M, S> {
        if positional.len() != closure.positional.len()
            || named
                .iter()
                .any(|(name, _)| !closure.named.iter().any(|(n, _)| n == name))
        {
            return Err(Dead);
        }
        let mut env = closure.env.clone();
        for (pattern, argument) in closure.positional.iter().zip(positional) {
            env = bind(env, pattern, argument)?;
        }
        for (name, default) in &closure.named {
            let value = named
                .iter()
                .find(|(n, _)| n == name)
                .map_or_else(|| default.clone(), |(_, value)| value.clone());
            env = Env::local(name.clone(), value, env);
        }
        // The body runs as code written where the function was, whoever
        // applies it.
        let scope = scope.enter(closure.code.as_ref(), call.site, &self.listings);
        self.eval(closure.body, &env, &scope)
    }

    /// Reports applying `f` when it may be a function a listing binding's
    /// callers give it, while that binding provides sliders.
    fn check_applied_parameter(
        &mut self,
        f: &Unknown<'a, M, S>,
        call: &Call<'a, M, S>,
        scope: &Scope,
    ) {
        let Mode::Alone(listing) = &scope.mode else {
            return;
        };
        if !scope
            .providers
            .iter()
            .any(|(_, providers)| providers.contains(&listing.id))
        {
            return;
        }
        for callee in &f.callees {
            if let Callee::Caller { listing: l, name } = callee
                && *l == listing.id
            {
                let labels: Vec<String> =
                    listing.labels.iter().map(|l| format!("{:?}", l)).collect();
                let message = format!(
                    "{} provides slider {} here, so it can't apply `{}`, a function its callers \
                     give",
                    (self.name)(listing.id),
                    labels.join(" and "),
                    name
                );
                let span = call.span.clone().or_else(|| (self.annotation)(listing.id));
                self.report(Error::eval(message, span));
                return;
            }
        }
    }

    /// Applies `slider` to `arguments` in `call`.
    fn slider(
        &mut self,
        arguments: Vec<Value<'a, M, S>>,
        call: &Call<'a, M, S>,
        scope: &Scope,
    ) -> Outcome<'a, M, S> {
        let count = arguments.len();
        if ![2, 4, 5].contains(&count) {
            return Err(Dead);
        }
        let label = match call.label.map(|label| &label.expr) {
            Some(Expr::String(label)) => Some(label.clone()),
            _ => {
                let span = call
                    .label
                    .and_then(|label| label.span.clone())
                    .or_else(|| call.span.clone());
                self.report(Error::eval(
                    "a slider's label must be a string written in its `slider` call",
                    span,
                ));
                None
            }
        };
        // The parts of the spec that aren't known, and where they come from.
        let mut open: Vec<(&str, Vec<Origin>)> = Vec::new();
        let mut numbers = Vec::new();
        for (index, argument) in arguments.iter().enumerate().take(count.min(4)).skip(1) {
            let part = if index == 1 { "default" } else { "range" };
            match argument {
                Value::Known(argument) => {
                    numbers.push(Some(argument.expr.as_const_float().ok_or(Dead)?))
                }
                Value::Unknown(argument) => {
                    open.push((part, argument.origins.clone()));
                    numbers.push(None);
                }
                _ => return Err(Dead),
            }
        }
        let curve = match arguments.get(4) {
            None => Some(Ok(Curve::Linear)),
            Some(Value::Known(curve)) => match &curve.expr {
                Expr::String(name) => Some(Curve::named(name).ok_or_else(|| name.clone())),
                _ => return Err(Dead),
            },
            Some(Value::Unknown(curve)) => {
                open.push(("curve", curve.origins.clone()));
                None
            }
            Some(_) => return Err(Dead),
        };
        let value = match numbers[0] {
            Some(default) => known(Expr::Waveform(Waveform::Const(default))),
            None => arguments[1].clone(),
        };
        let provider = match &label {
            Some(label) => self.provider(label, scope),
            None => None,
        };
        // Checked alone, a listing binding answers only for what it provides.
        if let Mode::Alone(listing) = &scope.mode
            && provider != Some(Provider::Listing(listing.id))
        {
            return Ok(value);
        }
        if !open.is_empty() {
            if let (Some(label), Some(provider)) = (&label, provider) {
                self.report_open(label, provider, &open, call);
            }
            return Ok(value);
        }
        let (default, min, max) = match numbers[..] {
            [Some(default)] => (default, 0.0, 1.0),
            [Some(default), Some(min), Some(max)] => (default, min, max),
            _ => unreachable!("every number is known"),
        };
        let named = match &label {
            Some(label) => format!("slider {:?}", label),
            None => "slider".to_string(),
        };
        let curve = match curve {
            Some(Ok(curve)) => curve,
            Some(Err(name)) => {
                self.report(Error::eval(
                    format!(
                        "{}: the curve must be \"linear\" or \"exp\", not {:?}",
                        named, name
                    ),
                    call.span.clone(),
                ));
                return Err(Dead);
            }
            None => unreachable!("the curve is known"),
        };
        let spec = match SliderSpec::new(default, min, max, curve) {
            Ok(spec) => spec,
            Err(message) => {
                self.report(Error::eval(
                    format!("{}: {}", named, message),
                    call.span.clone(),
                ));
                return Err(Dead);
            }
        };
        if let (Some(label), Some(provider)) = (label, provider) {
            self.record(provider, label, spec, call, scope);
        }
        Ok(value)
    }

    /// Returns the id of `provider`.
    fn id(&self, provider: Provider) -> usize {
        match provider {
            Provider::Played => self.played,
            Provider::Listing(id) => id,
        }
    }

    /// Returns the provider of a call of the slider `label` in `scope`, or
    /// `None` after reporting two or more listings of it there.
    fn provider(&mut self, label: &str, scope: &Scope) -> Option<Provider> {
        let listing = scope
            .providers
            .iter()
            .find(|(l, _)| l == label)
            .map(|(_, providers)| providers.clone())
            .unwrap_or_default();
        match listing[..] {
            [] => Some(Provider::Played),
            [one] => Some(Provider::Listing(one)),
            _ => {
                let mut many = listing;
                many.sort();
                let names: Vec<String> = many.iter().map(|&id| (self.name)(id)).collect();
                let message = format!(
                    "slider {:?} is listed by {}, and this waveform is built through each of \
                     them; keep only one listing",
                    label,
                    names.join(" and ")
                );
                for id in many {
                    let span = (self.annotation)(id);
                    self.report(Error::eval(message.clone(), span));
                }
                None
            }
        }
    }

    /// Reports why a call of the slider `label` of `provider` whose spec has
    /// the unknown parts `open` is an error, if it is one this check answers
    /// for here.
    fn report_open(
        &mut self,
        label: &str,
        provider: Provider,
        open: &[(&str, Vec<Origin>)],
        call: &Call<'a, M, S>,
    ) {
        for (part, origins) in open {
            for origin in origins {
                let error = match origin {
                    // A slider the played program lists is its own, and the
                    // played program is never checked alone.
                    Origin::Key | Origin::Velocity if self.id(provider) == self.played => {
                        let what = match origin {
                            Origin::Key => "the key a note plays",
                            _ => "the velocity a note plays with",
                        };
                        Error::eval(
                            format!(
                                "slider {:?}'s {} depends on {}, which isn't known until it plays",
                                label, part, what
                            ),
                            call.span.clone(),
                        )
                    }
                    Origin::Parameter { listing, name }
                        if Provider::Listing(*listing) == provider =>
                    {
                        Error::eval(
                            format!(
                                "{} lists slider {:?}, but its {} depends on `{}`, which its callers \
                             give; list it where `{}` is given instead",
                                (self.name)(*listing),
                                label,
                                part,
                                name,
                                name
                            ),
                            (self.annotation)(*listing),
                        )
                    }
                    Origin::Opaque => Error::eval(
                        format!(
                            "slider {:?}'s {} isn't known when the program loads",
                            label, part
                        ),
                        call.span.clone(),
                    ),
                    // A note reaches another listing's spec only through what
                    // its callers give it, and a parameter of another
                    // listing only through its own parameters: each is
                    // reported when that listing is checked alone.
                    _ => continue,
                };
                self.report(error);
                return;
            }
        }
    }

    /// Records a call in `call` of the slider `label` of `provider` with
    /// `spec`, reporting it when an earlier call's spec differs.
    fn record(
        &mut self,
        provider: Provider,
        label: String,
        spec: SliderSpec,
        call: &Call<'a, M, S>,
        scope: &Scope,
    ) {
        let key = (self.id(provider), label);
        match self.specs.get(&key) {
            Some(first) if *first != spec => {
                let message = format!(
                    "slider {:?} of {} is {} here but {} at another call; calls of one slider \
                     must agree",
                    key.1,
                    (self.name)(key.0),
                    spec,
                    first
                );
                self.report(Error::eval(message, call.span.clone()));
            }
            Some(_) => {}
            None => {
                self.specs.insert(key.clone(), spec.clone());
            }
        }
        // What the played program provides when checked alone is what it
        // gives its callers, so it counts too.
        match &scope.mode {
            Mode::Program => {}
            Mode::Alone(listing) if listing.id == self.played => {}
            Mode::Alone(_) => return,
        }
        let (provider, label) = key;
        let mut order = (*scope.sites).clone();
        order.extend(call.span.as_ref().map(|span| span.range.start));
        match self
            .reached
            .iter_mut()
            .find(|(_, reached)| reached.provider == provider && reached.label == label)
        {
            Some((first, _)) => {
                if order < *first {
                    *first = order;
                }
            }
            None => self.reached.push((
                order,
                Reached {
                    provider,
                    label,
                    spec,
                },
            )),
        }
    }

    /// Applies `map` to `arguments` in `call`.
    fn map(
        &mut self,
        arguments: Vec<Value<'a, M, S>>,
        call: &Call<'a, M, S>,
        scope: &Scope,
    ) -> Outcome<'a, M, S> {
        let [f, list] = <[_; 2]>::try_from(arguments).map_err(|_| Dead)?;
        let applied = call.applied();
        match elements(&list) {
            Some(elements) => {
                let mut results = Vec::with_capacity(elements.len());
                for element in elements {
                    results.push(self.apply(
                        f.clone(),
                        vec![element],
                        Vec::new(),
                        &applied,
                        scope,
                    )?);
                }
                Ok(Value::List(Rc::new(results)))
            }
            None => {
                let Value::Unknown(unknown) = &list else {
                    return Err(Dead);
                };
                // The list may be empty, so the function's failing doesn't
                // fail the call.
                let result = self.apply(f, vec![list.clone()], Vec::new(), &applied, scope);
                Ok(widen(
                    &result.into_iter().collect::<Vec<_>>(),
                    &unknown.origins,
                ))
            }
        }
    }

    /// Applies `reduce` to `arguments` in `call`.
    fn reduce(
        &mut self,
        arguments: Vec<Value<'a, M, S>>,
        call: &Call<'a, M, S>,
        scope: &Scope,
    ) -> Outcome<'a, M, S> {
        let [f, seed, list] = <[_; 3]>::try_from(arguments).map_err(|_| Dead)?;
        let applied = call.applied();
        match elements(&list) {
            Some(elements) => {
                let mut acc = seed;
                for element in elements {
                    acc = self.apply(f.clone(), vec![acc, element], Vec::new(), &applied, scope)?;
                }
                Ok(acc)
            }
            None => {
                let Value::Unknown(unknown) = &list else {
                    return Err(Dead);
                };
                let origins = unknown.origins.clone();
                self.iterate(seed, &origins, |analyzer, acc| {
                    analyzer.apply(
                        f.clone(),
                        vec![acc, list.clone()],
                        Vec::new(),
                        &applied,
                        scope,
                    )
                })
            }
        }
    }

    /// Applies `unfold` to `arguments` in `call`.
    fn unfold(
        &mut self,
        arguments: Vec<Value<'a, M, S>>,
        call: &Call<'a, M, S>,
        scope: &Scope,
    ) -> Outcome<'a, M, S> {
        let [f, seed, count] = <[_; 3]>::try_from(arguments).map_err(|_| Dead)?;
        let applied = call.applied();
        match &count {
            Value::Known(count) => {
                let n = count.expr.as_const_float().ok_or(Dead)?;
                if n < 0.0 || n.fract() != 0.0 {
                    return Err(Dead);
                }
                let mut results = Vec::new();
                let mut current = seed;
                for _ in 0..n as u32 {
                    results.push(current.clone());
                    current = self.apply(f.clone(), vec![current], Vec::new(), &applied, scope)?;
                }
                Ok(Value::List(Rc::new(results)))
            }
            Value::Unknown(count) => {
                let origins = count.origins.clone();
                self.iterate(seed, &origins, |analyzer, current| {
                    analyzer.apply(f.clone(), vec![current], Vec::new(), &applied, scope)
                })
            }
            _ => Err(Dead),
        }
    }

    /// Returns an unknown value, from `origins`, that covers `seed` and every
    /// value `step` returns after applying it any number of times to what it
    /// last returned.
    fn iterate(
        &mut self,
        seed: Value<'a, M, S>,
        origins: &[Origin],
        mut step: impl FnMut(&mut Self, Value<'a, M, S>) -> Outcome<'a, M, S>,
    ) -> Outcome<'a, M, S> {
        let mut acc = widen(&[seed], origins);
        loop {
            let next = match step(self, acc.clone()) {
                Ok(result) => widen(&[acc.clone(), result], origins),
                Err(Dead) => return Ok(acc),
            };
            if covers(&acc, &next) || self.fuel == 0 {
                return Ok(next);
            }
            acc = next;
        }
    }

    /// Applies `share` to `arguments` in `call`.
    fn share(
        &mut self,
        arguments: Vec<Value<'a, M, S>>,
        call: &Call<'a, M, S>,
        scope: &Scope,
    ) -> Outcome<'a, M, S> {
        let [waveform, f] = <[_; 2]>::try_from(arguments).map_err(|_| Dead)?;
        let applied = call.applied();
        match &waveform {
            Value::Known(w) if w.expr.as_const_float().is_some() => {
                self.apply(f, vec![waveform], Vec::new(), &applied, scope)
            }
            Value::Known(w) => {
                let Expr::Waveform(w) = &w.expr else {
                    return Err(Dead);
                };
                // Another check's `Shared` can't be in scope, so any id will do.
                let id = ShareId(u32::MAX);
                let reference = known(Expr::Waveform(Waveform::Ref(id)));
                let body = self.apply(f, vec![reference], Vec::new(), &applied, scope)?;
                let shared = |body: &Waveform<M>| Waveform::Shared {
                    id,
                    waveform: Box::new(w.clone()),
                    body: Box::new(body.clone()),
                };
                Ok(match exact(&body).map(|body| body.expr) {
                    Some(Expr::Waveform(body)) => known(Expr::Waveform(shared(&body))),
                    Some(Expr::Seq { offset, waveform }) => match waveform.expr {
                        Expr::Waveform(body) => known(Expr::Seq {
                            offset,
                            waveform: Box::new(SourceExpr::from(Expr::Waveform(shared(&body)))),
                        }),
                        _ => return Err(Dead),
                    },
                    _ => widen(&[body], &[]),
                })
            }
            Value::Unknown(_) => {
                // A constant is passed itself and any other waveform by
                // reference, so the function sees no more than the waveform.
                let body = self.apply(f, vec![waveform.clone()], Vec::new(), &applied, scope)?;
                Ok(widen(&[waveform, body], &[]))
            }
            _ => Err(Dead),
        }
    }

    /// Adds `error`, unless it is already there.
    fn report(&mut self, error: Error<S>) {
        if !self.errors.contains(&error) {
            self.errors.push(error);
        }
    }
}

impl Scope {
    /// Returns the scope with no listing providing anything and no call in
    /// progress.
    fn new(mode: Mode) -> Scope {
        Scope {
            providers: Rc::new(Vec::new()),
            sites: Rc::new(Vec::new()),
            mode,
        }
    }

    /// Returns this scope with the code written at `code` running inside it,
    /// run by a call written at the offset `site`: the listing binding
    /// `listings` finds for the code, if any, now provides each label it
    /// lists.
    fn enter<S>(
        &self,
        code: Option<&Span<S>>,
        site: Option<usize>,
        listings: &Listings<S>,
    ) -> Scope {
        let mut scope = self.clone();
        if let Some(site) = site {
            let mut sites = (*self.sites).clone();
            sites.push(site);
            scope.sites = Rc::new(sites);
        }
        let Some(listing) = code.and_then(|code| listings.find(code)) else {
            return scope;
        };
        let mut providers = (*self.providers).clone();
        for label in listing.labels.iter() {
            match providers.iter_mut().find(|(l, _)| l == label) {
                Some((_, ids)) => {
                    if !ids.contains(&listing.id) {
                        ids.push(listing.id);
                    }
                }
                None => providers.push((label.clone(), vec![listing.id])),
            }
        }
        scope.providers = Rc::new(providers);
        scope
    }
}

impl<'a, M, S> Env<'a, M, S> {
    /// Returns `next` with `name` bound to `value`.
    fn local(name: String, value: Value<'a, M, S>, next: Env<'a, M, S>) -> Env<'a, M, S> {
        Env::Local(Rc::new(Local { name, value, next }))
    }

    /// Returns what `name` is bound to, if anything.
    fn lookup(&self, name: &str) -> Option<Entry<'a, M, S>>
    where
        M: Clone,
        S: Clone,
    {
        let mut env = self;
        loop {
            match env {
                Env::Local(local) if local.name == name => {
                    return Some(Entry::Value(local.value.clone()));
                }
                Env::Local(local) => env = &local.next,
                Env::Top(top, len) => {
                    let top = top.upgrade()?;
                    let names = top.names.borrow();
                    return names
                        .get(name)?
                        .iter()
                        .rev()
                        .find(|(position, _)| position < len)
                        .map(|(_, entry)| entry.clone());
                }
            }
        }
    }
}

impl<'a, M, S> Top<'a, M, S> {
    /// Adds an entry binding `name` to `entry` after the others.
    fn push(&self, name: &str, entry: Entry<'a, M, S>) {
        let position = self.len.get();
        self.names
            .borrow_mut()
            .entry(name.to_string())
            .or_default()
            .push((position, entry));
        self.len.set(position + 1);
    }
}

/// Returns `expr`, with no span, as a known value.
fn known<'a, M, S>(expr: Expr<M, S>) -> Value<'a, M, S> {
    Value::Known(Rc::new(SourceExpr::from(expr)))
}

/// Returns an unknown value from `origins`, which is no function.
fn unknown<'a, M, S>(origins: Vec<Origin>) -> Value<'a, M, S> {
    Value::Unknown(Rc::new(Unknown {
        origins,
        callees: Vec::new(),
    }))
}

/// Returns `value` as an expression, if it is known exactly.
fn exact<M: Clone, S: Clone>(value: &Value<'_, M, S>) -> Option<SourceExpr<M, S>> {
    match value {
        Value::Known(expr) => Some((**expr).clone()),
        Value::Tuple(values) => Some(SourceExpr::tuple(
            values.iter().map(exact).collect::<Option<_>>()?,
        )),
        Value::List(values) => Some(SourceExpr::list(
            values.iter().map(exact).collect::<Option<_>>()?,
        )),
        _ => None,
    }
}

/// Returns the elements of `list`, if it is a list of known length.
fn elements<'a, M: Clone, S: Clone>(list: &Value<'a, M, S>) -> Option<Vec<Value<'a, M, S>>> {
    match list {
        Value::List(values) => Some(values.to_vec()),
        Value::Known(expr) => match &expr.expr {
            Expr::List(exprs) => Some(exprs.iter().map(|e| known(e.expr.clone())).collect()),
            _ => None,
        },
        _ => None,
    }
}

/// Returns `value` split into `n` parts, as a tuple pattern of `n` parts
/// splits it.
fn parts<'a, M: Clone, S: Clone>(
    value: Value<'a, M, S>,
    n: usize,
) -> Result<Vec<Value<'a, M, S>>, Dead> {
    match value {
        Value::Tuple(values) if values.len() == n => Ok(values.to_vec()),
        Value::Known(expr) => match &expr.expr {
            Expr::Tuple(exprs) if exprs.len() == n => {
                Ok(exprs.iter().map(|e| known(e.expr.clone())).collect())
            }
            _ => Err(Dead),
        },
        unknown @ Value::Unknown(_) => Ok(vec![unknown; n]),
        _ => Err(Dead),
    }
}

/// Returns `env` with the names in `pattern` bound to the parts of `value`.
fn bind<'a, M: Clone, S: Clone>(
    env: Env<'a, M, S>,
    pattern: &Pattern,
    value: Value<'a, M, S>,
) -> Result<Env<'a, M, S>, Dead> {
    match pattern {
        Pattern::Identifier(name) => Ok(Env::local(name.clone(), value, env)),
        Pattern::Tuple(patterns) => {
            let mut env = env;
            for (pattern, part) in patterns.iter().zip(parts(value, patterns.len())?) {
                env = bind(env, pattern, part)?;
            }
            Ok(env)
        }
    }
}

/// Returns the part of `value` that `pattern` binds to `name`.
fn select<'a, M: Clone, S: Clone>(
    pattern: &Pattern,
    value: Value<'a, M, S>,
    name: &str,
) -> Outcome<'a, M, S> {
    match pattern {
        Pattern::Identifier(_) => Ok(value),
        Pattern::Tuple(patterns) => {
            for (pattern, part) in patterns.iter().zip(parts(value, patterns.len())?) {
                if pattern.binds(name) {
                    return select(pattern, part, name);
                }
            }
            Err(Dead)
        }
    }
}

/// Adds the names `pattern` binds to `names`, in order.
fn pattern_names<'p>(pattern: &'p Pattern, names: &mut Vec<&'p str>) {
    match pattern {
        Pattern::Identifier(name) => names.push(name),
        Pattern::Tuple(patterns) => {
            for pattern in patterns {
                pattern_names(pattern, names);
            }
        }
    }
}

/// Returns a value standing for what a listing's caller gives for `pattern`.
fn parameter<'a, M, S>(pattern: &Pattern, listing: usize) -> Value<'a, M, S> {
    match pattern {
        Pattern::Identifier(name) => given(listing, name),
        Pattern::Tuple(patterns) => Value::Tuple(Rc::new(
            patterns.iter().map(|p| parameter(p, listing)).collect(),
        )),
    }
}

/// Returns a value standing for what the callers of `listing` give for its
/// parameter `name`: anything, maybe a function.
fn given<'a, M, S>(listing: usize, name: &str) -> Value<'a, M, S> {
    let name = name.to_string();
    Value::Unknown(Rc::new(Unknown {
        origins: vec![Origin::Parameter {
            listing,
            name: name.clone(),
        }],
        callees: vec![Callee::Caller { listing, name }],
    }))
}

/// Adds the user functions `value` is or holds to `closures`.
fn exposed<'a, M, S>(value: &Value<'a, M, S>, closures: &mut Vec<Rc<Closure<'a, M, S>>>) {
    match value {
        Value::Closure(closure) => closures.push(closure.clone()),
        Value::Tuple(values) | Value::List(values) => {
            for value in values.iter() {
                exposed(value, closures);
            }
        }
        Value::Unknown(unknown) => {
            for callee in &unknown.callees {
                if let Callee::Closure(closure) = callee {
                    closures.push(closure.clone());
                }
            }
        }
        Value::Known(_) | Value::Module(_) => {}
    }
}

/// Returns where an application `expr` of the function expression `function`
/// was written as a call, for text order: where the function expression was
/// written for a pipe `a | f(…)`, and the application otherwise.
fn call_site<M, S>(expr: &SourceExpr<M, S>, function: &SourceExpr<M, S>) -> Option<usize> {
    let start = expr.span.as_ref()?.range.start;
    let written = function.span.as_ref()?.range.start;
    Some(written.max(start))
}

/// Applies the built-in `function`, named `name`, to `arguments` in `call`:
/// exactly when every argument is known, and otherwise to an unknown value.
fn builtin<'a, M, S>(
    name: &str,
    function: &BuiltInFn<M, S>,
    arguments: Vec<Value<'a, M, S>>,
    call: &Call<'a, M, S>,
) -> Outcome<'a, M, S>
where
    M: Clone + CarriesTrace<S>,
    S: Clone,
{
    match arguments.iter().map(exact).collect::<Option<Vec<_>>>() {
        Some(exprs) => {
            let call = BuiltInCall::new(call.span.clone(), Trace::new::<M>(None, None), &exprs);
            match function.0(exprs, &call) {
                Ok(result) => Ok(Value::Known(Rc::new(result))),
                Err(_) => Err(Dead),
            }
        }
        None => Ok(match PASSING.contains(&name) {
            true => widen(&arguments, &[]),
            false => {
                let Value::Unknown(result) = widen(&arguments, &[]) else {
                    unreachable!("widening makes an unknown value");
                };
                unknown(result.origins.clone())
            }
        }),
    }
}

/// Returns an unknown value that may be any of `values`, or anything computed
/// from them, from `origins` as well as from where they come from.
fn widen<'a, M: Clone, S: Clone>(
    values: &[Value<'a, M, S>],
    origins: &[Origin],
) -> Value<'a, M, S> {
    let mut all = origins.to_vec();
    let mut callees = Vec::new();
    for value in values {
        gather(value, &mut all, &mut callees);
    }
    all.sort();
    all.dedup();
    if all.is_empty() {
        all.push(Origin::Opaque);
    }
    Value::Unknown(Rc::new(Unknown {
        origins: all,
        callees,
    }))
}

/// Adds where `value` comes from to `origins` and the functions that run user
/// code it is or holds to `callees`.
fn gather<'a, M: Clone, S: Clone>(
    value: &Value<'a, M, S>,
    origins: &mut Vec<Origin>,
    callees: &mut Vec<Callee<'a, M, S>>,
) {
    match value {
        Value::Known(expr) => gather_known(expr, callees),
        Value::Closure(closure) => add_callee(callees, Callee::Closure(closure.clone())),
        Value::Tuple(values) | Value::List(values) => {
            for value in values.iter() {
                gather(value, origins, callees);
            }
        }
        Value::Module(_) => {}
        Value::Unknown(unknown) => {
            origins.extend(unknown.origins.iter().cloned());
            for callee in &unknown.callees {
                add_callee(callees, callee.clone());
            }
        }
    }
}

/// Adds the built-ins in `expr` that the check applies itself to `callees`.
fn gather_known<'a, M: Clone, S: Clone>(
    expr: &SourceExpr<M, S>,
    callees: &mut Vec<Callee<'a, M, S>>,
) {
    match &expr.expr {
        Expr::BuiltIn { name, .. } if SPECIAL.contains(&name.as_str()) => {
            add_callee(callees, Callee::BuiltIn(Rc::new(expr.clone())));
        }
        Expr::Tuple(exprs) | Expr::List(exprs) => {
            for expr in exprs {
                gather_known(expr, callees);
            }
        }
        Expr::Seq { offset, waveform } => {
            gather_known(offset, callees);
            gather_known(waveform, callees);
        }
        Expr::BoundModule(entries) => {
            for (_, expr) in entries {
                gather_known(expr, callees);
            }
        }
        _ => {}
    }
}

/// Adds `callee` to `callees` unless it is already there.
fn add_callee<'a, M, S>(callees: &mut Vec<Callee<'a, M, S>>, callee: Callee<'a, M, S>) {
    let same = |c: &Callee<'a, M, S>| match (c, &callee) {
        (Callee::Closure(a), Callee::Closure(b)) => Rc::ptr_eq(a, b),
        (Callee::BuiltIn(a), Callee::BuiltIn(b)) => builtin_name(a) == builtin_name(b),
        (
            Callee::Caller { listing, name },
            Callee::Caller {
                listing: l,
                name: n,
            },
        ) => listing == l && name == n,
        _ => false,
    };
    if !callees.iter().any(same) {
        callees.push(callee);
    }
}

/// Returns the name of the built-in `expr` is, if it is one.
fn builtin_name<M, S>(expr: &SourceExpr<M, S>) -> Option<&str> {
    match &expr.expr {
        Expr::BuiltIn { name, .. } => Some(name),
        _ => None,
    }
}

/// Returns whether the unknown value `acc` covers `next`: `next` comes from
/// nowhere `acc` doesn't, and each of its callees is written where one of
/// `acc`'s is.
fn covers<M, S>(acc: &Value<'_, M, S>, next: &Value<'_, M, S>) -> bool {
    let (Value::Unknown(acc), Value::Unknown(next)) = (acc, next) else {
        return false;
    };
    next.origins.iter().all(|o| acc.origins.contains(o))
        && next.callees.iter().all(|callee| {
            acc.callees.iter().any(|c| match (c, callee) {
                (Callee::Closure(c), Callee::Closure(callee)) => std::ptr::eq(c.body, callee.body),
                (Callee::BuiltIn(c), Callee::BuiltIn(callee)) => {
                    builtin_name(c) == builtin_name(callee)
                }
                (
                    Callee::Caller { listing, name },
                    Callee::Caller {
                        listing: l,
                        name: n,
                    },
                ) => listing == l && name == n,
                _ => false,
            })
        })
}

/// Returns a value covering both `a` and `b`, one of which a condition from
/// `origins` chooses. A failed path chooses the other.
fn join<'a, M, S>(
    a: Outcome<'a, M, S>,
    b: Outcome<'a, M, S>,
    origins: &[Origin],
) -> Outcome<'a, M, S>
where
    M: Clone + CarriesTrace<S>,
    S: Clone,
{
    match (a, b) {
        (Err(Dead), b) => b,
        (a, Err(Dead)) => a,
        (Ok(a), Ok(b)) => Ok(merge(a, b, origins)),
    }
}

/// Returns a value covering both `a` and `b`, as [`join`] does.
fn merge<'a, M, S>(a: Value<'a, M, S>, b: Value<'a, M, S>, origins: &[Origin]) -> Value<'a, M, S>
where
    M: Clone + CarriesTrace<S>,
    S: Clone,
{
    match (&a, &b) {
        (Value::Known(x), Value::Known(y)) if same_constant(x, y) => a,
        (Value::Closure(x), Value::Closure(y)) if Rc::ptr_eq(x, y) => a,
        (Value::Module(x), Value::Module(y)) if Rc::ptr_eq(x, y) => a,
        (Value::Tuple(xs), Value::Tuple(ys)) if xs.len() == ys.len() => Value::Tuple(Rc::new(
            xs.iter()
                .zip(ys.iter())
                .map(|(x, y)| merge(x.clone(), y.clone(), origins))
                .collect(),
        )),
        (Value::List(xs), Value::List(ys)) if xs.len() == ys.len() => Value::List(Rc::new(
            xs.iter()
                .zip(ys.iter())
                .map(|(x, y)| merge(x.clone(), y.clone(), origins))
                .collect(),
        )),
        _ => widen(&[a, b], origins),
    }
}

/// Returns whether `a` and `b` are the same boolean, string, or number.
fn same_constant<M: CarriesTrace<S>, S>(a: &SourceExpr<M, S>, b: &SourceExpr<M, S>) -> bool {
    match (&a.expr, &b.expr) {
        (Expr::Bool(a), Expr::Bool(b)) => a == b,
        (Expr::String(a), Expr::String(b)) => a == b,
        _ => matches!(
            (a.expr.as_const_float(), b.expr.as_const_float()),
            (Some(a), Some(b)) if a == b
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_program;

    /// Returns a built-in named `name` that returns its first argument.
    fn stand_in(name: &str) -> SourceBinding<u32, ()> {
        SourceBinding::definition(
            Pattern::Identifier(name.to_string()),
            SourceExpr::from(Expr::BuiltIn {
                name: name.to_string(),
                function: BuiltInFn::new(|arguments: Vec<SourceExpr<u32, ()>>, _call| {
                    arguments
                        .into_iter()
                        .next()
                        .ok_or_else(|| Error::internal_here("expected an argument"))
                }),
            }),
        )
    }

    /// Returns stand-ins for the built-ins the check applies itself, which it
    /// knows only by name, and for the operators `>` and `*`.
    fn prelude() -> Vec<SourceBinding<u32, ()>> {
        ["slider", "map", "reduce", "unfold", ">", "*"]
            .into_iter()
            .map(stand_in)
            .collect()
    }

    /// Checks the keys instrument `source` under [`prelude`], with no modules
    /// or listings.
    fn check_keys(source: &str) -> Analysis<()> {
        let bindings = prelude();
        let expr = parse_program::<u32, _>(source, ()).unwrap();
        let program = Program {
            bindings: &bindings,
            expr: &expr,
            role: Role::Keys,
            listings: Listings::new(|_| None),
            played: 0,
            name: &|_| "the instrument".to_string(),
            annotation: &|_| None,
        };
        analyze(|_| Err(Error::eval_here("no modules")), &program)
    }

    /// Returns the messages of `analysis`'s errors.
    fn messages(analysis: &Analysis<()>) -> Vec<&str> {
        analysis.errors.iter().map(Error::message).collect()
    }

    /// Returns the labels of the sliders `analysis` reached.
    fn labels(analysis: &Analysis<()>) -> Vec<&str> {
        analysis.reached.iter().map(|r| r.label.as_str()).collect()
    }

    /// A list whose length the key a note plays, `k`, chooses.
    const KEYS_LIST: &str = "unfold(fn(x) => x, 0, k)";

    const DEPENDS_ON_KEY: &str =
        "slider \"q\"'s default depends on the key a note plays, which isn't known until it plays";

    #[test]
    fn reducing_a_list_the_note_chooses_depends_on_the_note() {
        // The step returns its accumulator unchanged, but the result still
        // depends on how many steps the note's list takes.
        let analysis = check_keys(&format!(
            "fn(k, v) => slider(\"q\", reduce(fn(acc, x) => acc, 0.5, {KEYS_LIST}))"
        ));
        assert_eq!(messages(&analysis), [DEPENDS_ON_KEY]);
    }

    #[test]
    fn a_function_the_steps_build_is_applied_with_the_result() {
        // The seed calls no slider; each step's result does, so the slider is
        // reached only if the result includes what the steps build.
        let analysis = check_keys(&format!(
            "fn(k, v) => reduce(fn(f, x) => fn(y) => slider(\"q\", 0.5) * f(y), fn(y) => y, \
             {KEYS_LIST})(1)"
        ));
        assert_eq!(messages(&analysis), Vec::<&str>::new());
        assert_eq!(labels(&analysis), ["q"]);
    }

    #[test]
    fn steps_that_build_the_same_function_afresh_stop() {
        // Each step builds a new closure with the same body, which covers the
        // last; otherwise the steps would go on until the check ran out of
        // fuel.
        let analysis = check_keys(
            "fn(k, v) => unfold(fn(f) => fn(y) => f(y) * slider(\"q\", 0.5), fn(y) => y, k)",
        );
        assert_eq!(messages(&analysis), Vec::<&str>::new());
    }

    /// Returns a new closure of no parameters with `body`.
    fn closure<'a>(body: &'a SourceExpr<u32, ()>) -> Rc<Closure<'a, u32, ()>> {
        Rc::new(Closure {
            positional: &[],
            named: Vec::new(),
            body,
            env: Env::Top(Weak::new(), 0),
            code: None,
        })
    }

    /// Returns an unknown value from `origins` that may be any of `closures`.
    fn calling<'a>(
        origins: &[Origin],
        closures: &[&Rc<Closure<'a, u32, ()>>],
    ) -> Value<'a, u32, ()> {
        Value::Unknown(Rc::new(Unknown {
            origins: origins.to_vec(),
            callees: closures
                .iter()
                .map(|&closure| Callee::Closure(closure.clone()))
                .collect(),
        }))
    }

    #[test]
    fn covering_compares_closures_by_where_they_were_written() {
        let (one, two) = (SourceExpr::float(1.0), SourceExpr::float(2.0));
        let (first, again, other) = (closure(&one), closure(&one), closure(&two));
        let acc = calling(&[Origin::Key], &[&first]);
        assert!(covers(&acc, &calling(&[Origin::Key], &[&again])));
        assert!(!covers(&acc, &calling(&[Origin::Key], &[&other])));
        assert!(!covers(&acc, &calling(&[Origin::Velocity], &[&again])));
        assert!(covers(
            &calling(&[Origin::Key, Origin::Velocity], &[&first, &other]),
            &acc
        ));
    }

    #[test]
    fn branches_the_note_chooses_between_that_agree_are_known() {
        let analysis = check_keys("fn(k, v) => slider(\"q\", if k > 60 then 0.5 else 0.5)");
        assert_eq!(messages(&analysis), Vec::<&str>::new());
        assert_eq!(labels(&analysis), ["q"]);
    }

    #[test]
    fn branches_the_note_chooses_between_that_differ_depend_on_the_note() {
        let analysis = check_keys("fn(k, v) => slider(\"q\", if k > 60 then 0.4 else 0.5)");
        assert_eq!(messages(&analysis), [DEPENDS_ON_KEY]);
    }

    #[test]
    fn a_failing_branch_leaves_the_other_branch() {
        let analysis = check_keys("fn(k, v) => slider(\"q\", if k > 60 then 0.4 else missing)");
        assert_eq!(messages(&analysis), Vec::<&str>::new());
        assert_eq!(labels(&analysis), ["q"]);
    }

    #[test]
    fn branches_merge_part_by_part() {
        let pair = "if k > 60 then (0.5, 0.1) else (0.5, 0.2)";
        let agreeing = check_keys(&format!(
            "fn(k, v) => let (a, b) = {pair} in slider(\"q\", a)"
        ));
        assert_eq!(messages(&agreeing), Vec::<&str>::new());
        let differing = check_keys(&format!(
            "fn(k, v) => let (a, b) = {pair} in slider(\"q\", b)"
        ));
        assert_eq!(messages(&differing), [DEPENDS_ON_KEY]);
    }

    #[test]
    fn branches_the_note_chooses_between_apply_each_function() {
        let analysis = check_keys(
            "fn(k, v) => (if k > 60 then fn(x) => slider(\"a\", x) else fn(x) => slider(\"b\", x))(0.5)",
        );
        assert_eq!(messages(&analysis), Vec::<&str>::new());
        assert_eq!(labels(&analysis), ["a", "b"]);
    }
}
