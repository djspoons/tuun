# Tuun Expression Language Specification

Tuun expressions are a way of specifying Tuun waveforms. The [overview](overview.md) gives examples of both these high-level expressions and the low-level waveforms they specify. This document provides a more detailed (but still informal) specification of those high-level expressions and their semantics.

Tuun expressions are a call-by-value functional programming language with a simple file-based module system.

## Types and Patterns

**Types** are not present in the concrete syntax, but if they were, we could imagine something like this:

```
type ::= "float"
       | "string"
       | "bool"
       | type "->" type
       | "(" type "," ... ")"
       | "[" type "]"
       | "waveform"
       | "seq"
```

That is, Tuun includes a few of the usual base types along with function, tuple, and list types. It also includes two types specific to audio generation.

A Tuun **identifier** (or `id`) is an alphanumeric sequence that may also contain `_` and `#`. Identifiers start with an alphabetic character or `_`. An identifier consisting of only a single `_` may be bound but not referenced. Identifiers starting with two `_` are reserved for internal use.

Tuun **patterns** are the only way to destructure tuples and are used in function definitions and bindings:
```
pattern ::= id
          | "(" pattern "," ... ")"
```

## Bindings and Modules

A **binding** modifies the evaluation context by introducing one or more new identifiers into that context. A module is a set of bindings, usually contained in a single file.

```
binding ::= pattern "=" expr
          | "open" id "." ... id

module  ::= binding ";" ...
```

Bindings are evaluated in order, and subsequent bindings may refer to and shadow previous bindings.

An `open` binding refers to another module. The bindings of that module are introduced in the current context but _not_ into modules that `open` the current module.


<!--
binding ::= ... "use" var "." ... var

expr ::= ...
     | var "." ... var
-->

<!--

private bindings

-->

### Prelude

Every Tuun module includes an implicit `open` that injects a special set of bindings called the _prelude_. The prelude includes definitions of waveform constructors (`sine`), list helpers (`map`), and mathematical functions (`sqrt`). The prelude may also contain environment-specific bindings such as `tempo` or a `debug` function that prints information to a console or log.

## Expressions and Values

```
expr ::= float
       | string
       | bool
       | "fn" "(" pattern "," ... "," id "=" expr "," ... ")" "=>" expr
       | id
       | expr "(" expr ")"
       | "(" expr "," ... ")"
       | "[" expr "," ... "]"
       | expr binary_op expr
       | unary_op expr
       | "if" expr "then" expr "else" expr
       | "let" binding "," ... "in" expr
       | "{" expr "}"
       | "<" expr ">"
       | ...

unary_op ::= "-" | "$" | "@" | ...
binary_op ::= "+" | "-" | "*" | "/" | "&" | "|" | "==" | "!=" | "<" | ...
```

Binary and unary operators are de-sugared to applications, as are `let` bindings.

```
value ::= float
        | string
        | bool
        | "fn" "(" pattern "," ... "," id "=" value "," ... ")" "=>" expr
        | "(" value "," ... ")"
        | "[" value "," ... "]"
        | waveform
        | seq "(" waveform "," waveform ")"
```

### Functions and Application

Tuun functions support _named_ parameters, which have default values and may be omitted when the function is applied.

* The default expression of a named parameter must always be present: that's what distinguishes it as named!
* Named parameters in a function definition must always appear _after_ positional parameters. 
* The default expression may reference other bindings defined in the environment, but may not reference other parameters.
* Default expressions are evaluated once and at the same time as the function itself.
* Each named parameter must be a single identifier (unlike positional parameters, which take the form of arbitrary patterns).
* The set of identifiers used in a functions parameters must not contain any duplicates.

When a function is applied, providing arguments for named parameters is optional. (As with a standard call-by-value functional language, all positional parameters must be present.)

* Named arguments must appear after positional arguments.
* When named parameter is not provided in an application, its default value is used.
* As with positional arguments, the value of a named argument is evaluated in the same context as the application itself (and may not reference other named parameters).
* Each named argument should appear at most once at a given application site.

Here is an example of a function definition and application using named parameters and arguments.
```
open std;
filtered = fn(freq_hz, cutoff_freq_hz = 2000) => sawtooth(freq_hz) | lpf(0.6, cutoff_freq_hz);

uses_default = filtered(440);
low_cutoff = filtered(440, cutoff_freq_hz = 600);
high_cutoff = filtered(440, cutoff_freq_hz = 6000);
```

## Waveforms

### Fin and Append

### Seq and Followed-By



### Overloading

Many Tuun operators are overloaded to operate on floats, waveforms, and (where it makes sense) sequenced waveforms. For example, the `+` operator can be applied to both floating point numbers and waveforms, and will promote a floating point value to a constant waveform.

| Operator    | Left-hand type   | Right-hand type   | Result type | 
| ---------   | ---------------  | ----------------  | ------      |
| + - * / &   | float            | float             | float       |
| + - * / &   | waveform         | waveform          | waveform    |
| + - * / &   | float            | waveform          | waveform    |
| + - * / &   | waveform         | float             | waveform    |
| + - * / &   | seq              | waveform          | seq         |
| + - * / &   | waveform         | seq               | seq         |
| + - * / &   | seq              | float             | seq         |
| + - * / &   | float            | seq               | seq         |
| \\          | seq              | float             | seq         |
| \\          | seq              | waveform          | seq         |
| \\          | seq              | seq               | seq         |






## Comments

Tuun supports `//` comments like those in BCPL. Comments are ignored during evaluation.

## Annotations

Annotations affect how Tuun bindings are displayed, played, and how they are controlled by the user interface. Annotations are applied to bindings.

Any binding with at least one annotation is a UI program; use `skip_slots=N` to leave a gap of empty slots before it. UI programs are laid out in source order, starting at slot 0.

* `skip_slots=N`
* `color=rgb(N,N,N)`
* `level_db=D`
* `sliders=[...]`

<!--

Notebooks may be 

```
use drums;

// Define a kick drum
kick = drums.tuned_kick_drum(100);

// Four on the floor
#{level_db=-2.0}
_ = on_beats(kick, [1, 2, 3, 4]);
```
<!-- 
```
// lib/std/math.tuun
trig = let 
  use std.trig
in
  trig,

// program.tuun
use std.math.trig // same as use std.trig


```

### Web page notebooks
 -->




