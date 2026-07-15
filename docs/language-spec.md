# Tuun Expression Language Specification

Tuun expressions are a way of specifying Tuun waveforms. The [language overview](tuun-langs.md) gives examples of both these high-level expressions and the low-level waveforms they specify. This document provides a more detailed (but still informal) specification of those high-level expressions and their semantics.

Tuun expressions are a call-by-value functional programming language. 

## Types

Types are not present in the concrete syntax, but if they were we could imagine something like this
```
type ::= "float"
      | "string"
      | "bool"
      | type "->" type
      | "waveform"
      | "seq"
```

## Expressions and Values

```
binding ::= pattern "=" expr
        | "open" var "." ... var

module :== binding ";" ...
```

<!--
binding ::= ... "use" var "." ... var

expr ::= ...
     | var "." ... var
-->

```
value ::= float
      | string
      | bool
      | "fn" "(" var "," ... ")" "=>" expr
      | waveform
      | seq "(" waveform "," waveform ")"


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

Tuun supports `//` comments like those in BCPL. Comments are ignored during evaluation/

## Modules

Tuun 

In its simplest form, a module is a list of Tuun bindings. Recall that Tuun `let` bindings are of the form `pattern = exp`. Unlike `let` bindings, which are separated by `,`, each module binding must be followed by a `;`. 

Modules may also change the evaluation context through the use of `use` bindings.

<!-- (Modules can also support a special "empty" binding; see below.) -->

Tuun takes the approach that anything that is not used in the specification of a waveform is 

<!-- Let's do ids that start with _ are private -->

### Annotations

Annotations affect how Tuun bindings are displayed, played, and how they are controlled by the user interface. Any binding with at least one annotation is a UI program; use `skip_slots=N` to leave a gap of empty slots before it. UI programs are laid out in source order, starting at slot 0.

* `skip_slots=N`
* `color=rgb(N,N,N)`
* `level_db=D`
* `sliders=[...]`

### 

Notebooks may be 

```
use std.drums;

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
 -->
### Web page notebooks





