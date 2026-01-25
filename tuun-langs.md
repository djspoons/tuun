#  Tuun Languages

Tuun is an interactive, language-based sound and music generation system. It has two main components:

 * A tracker, which interfaces with the underlying audio subsystem, managing things like sample rates and buffers. It takes commands like "play this!" and gives updates on the state of playback.
 * An visual UI that enables users to compose sounds, enter those commands, and see the results.

Tuun has two languages:

 * A language of low-level waveforms and waveform combinators that are used by the tracker to generate samples to feed to the audio system.
 * A language that's used in the UI to specify those waveforms.

We'll start with a brief introduction to the second one before we dive into the details.

Tuun's interactive language is a simple, functional language like OCaml or Standard ML. It supports floating point numbers, arithmetic, functions, and tuples. In addition, it has some built-in syntax and semantics that make it easy to express complicated waveforms. For example, the `$` operator takes a frequency and returns a waveform that will generate a tone at that frequency.

```
$440
```

Tuun lets you define abstractions so that you can easily convert from MIDI note numbers to frequencies (`@`), create notes of different lengths (`Q` and `H`) and combine notes in a sequence:

```
<[$(@60)(Q), $(@64)(Q), $(@67)(H)]>
```

The way that you use the Tuun language is up to you! There's nothing baked in about western music or MIDI. Instead it's all built from the following waveforms and waveform combinators.

## Tuun Waveform Language

Tuun has several primitive waveforms and waveform combinators. The first, `Const`, isn't exactly a "wave" but is used to create waves: `Const` generates a stream where every sample is the same value. This can be used with the `Sin` combinator to produce a sine wave. `Sin` takes two arguments: one for the angular frequency (in radians per second) and one for the phase offset. For example, the following will generate a tone at 440 Hz.

```
Sin(Const(2 * PI * 440), Const(0))
```

This should be reminiscent of the definition from trigonometry class:
$$
\sin(t) = 2πft + \phi
$$
where $f$ is the desired frequency (in Hertz) and $\phi$ is the desired phase offset.

Every waveform has an intrinsic property called its _length_, which may be finite or infinite. The lengths of `Const` is infinite, and the length of `Sin` is determined by its inputs: `Sin` generates one sample of output for each sample of inputs. This means that the expression above will generate a tone that goes on forever.

Since it's often useful to have waveforms that *don't* go on forever, Tuun includes the `Fin` combinator, which modifies a waveform to be finite. This leverages the `Time` combinator, which generates a stream where each sample is the time elapsed since the beginning of the waveform (in seconds). For example, the following will generate a tone at 440Hz for 2 seconds.

```
Fin(Time ~- Const(-2), Sin(Const(2 * PI * 440), Const(0)))
```

The length of `Fin` is given by its first parameter: `Fin` generates samples from this waveform until it gets a sample >= 0, at which point it stops. In the example above, it generates samples from the waveform `Time ~- Const(2)`. The `~-` combinator subtracts each pair of corresponding samples, yielding a new stream. You can think of this waveform a bit like a countdown clock: it starts at -2 and then "counts down" (well, _up_) until it reaches 0.

The `~*` we can use `~*` to modify outputs as well. This combinator multiplies each sample in the first waveform by the corresponding sample in the second waveform. Below it's used to control the amplitude of a waveform. For example, we can half the amplitude of the sine wave by multiplying by a constant waveform.

```
Fin(Time ~- Const(2), Sin(Const(2 * PI * 440), Const(0)) ~. Const(0.5))
```

> The length of a waveform like `a ~+ b` or `a ~- b` is the _maximum_ of the length of `a` and the length of `b` (since the sum can continue generating samples as long as one of the components is), and the length of a waveform `a ~. b` or `a ~/ b` is the _minimum_ of the length of `a` and the length of `b` (since once the length of one waveform has been exceeded, it's as if it will generate only zeros forever).

Putting these combinators together enables us to create more interesting sounds. For example, we can generate harmonics by applying progressively smaller constants to higher frequencies like the following.

```
Fin(Time ~- Const(4), Sin(Const(2 * PI * 440), Const(0)) ~+ Sin(Const(2 * PI * 1320), Const(0)) ~. Const(0.33) ~+ Sin(Const(2 * PI * 2200), Const(0)) ~. Const(0.2))
```

Writing this out is starting to get a little tedious, though, and we'll see how to use the waveform specification language below to build a library of functions for easily generating harmonics and other complex waveforms.

While we've seen how to create finite waveforms, we haven't yet see how to describe a _sequence_ of waveforms. While the tracker is responsible for high-level sequencing waveforms (for example, at the level of musical phrases), the components within a given waveform don't all need start at the same time (for example, notes within a musical phrase).

You might imagine that sequencing two waveforms is just a matter of appending the samples from the second to those of the first or maybe adding some silence to the beginning of second one and combining them using `~+`. While those are two ways of looking at the problem (and ones that we shall revisit shortly!), the other way to add a property to the _first_ waveform to indicate when the _next_ waveform should start. This is often convenient when we are describing waveforms as we might want to specify, for example, a quarter note without yet knowing what notes will come before or after it.

To support sequencing, every waveform has another property that determines the _offset_ of the subsequent waveform. A waveform's offset doesn't effect how that waveform will generate samples, but it does affect how it's combined with other waveforms. Waveforms like `Const` have an offset of 0.

To give a waveform a non-zero offset, we use the `Seq` combinator, which modifies another waveform to have a specified offset. `Seq(duration, a)` always has an offset of `duration` regardless of the offset of `a`. (`Seq` is the analogue to the `Fin`!)

You could imagine a combinator called `Then` that takes two waveforms `a` and `b` and that first generates samples `a`, followed by samples from `b` starting at the offset of `a` (and combining them where they overlap). The following would generate a 440Hz tone for two seconds that overlaps (by one second) with two-second 880Hz tone.

```
Then(
    Seq(1, Fin(Time ~- Const(2), Sin(Const(2 * PI * 440), Const(0)))),
    Fin(Time ~- Const(2), Sin(Const(2 * PI * 880), Const(0)))
)
```

Why do we need separate notions of length and offset? One example where we want both is when emulating notes played on a piano with the sustain pedal held down: we might want the second note to start on the second beat, but we don't want the first note to stop until the pedal is released. Again, a waveform's length is essential to how it generates its own samples, while its offset controls how it is combined with other waveforms.

We need to revisit the behavior of operators like `~+` and `~.` for waveforms with offsets. In the examples above, they were only applied to waveforms with offsets equal to 0. However, the offset of the left argument is used by both combinators: the second waveform only takes effect _after_ the offset indicated by the first. Thus in the case of `a ~+ b`, the samples of `b` are added to `a` only starting at the offset of `a` -- any samples in the left-hand operand that occur before its offset are just passed through. This means that for waveforms with non-zero offsets, `~+` and `~.` are not communicative combinators! This is not very convenient for optimizing waveforms, and we'll see how to eliminate offsets below.

Given this, we can now see there no need for a separate `Then` combinator: `~+` already provides the required functionality! That is, `a ~+ b` generates samples from `a` until the offset of `a` is reached, at which point it adds corresponding samples from the two waveforms together. When the length of `a` is less than or equal to its offset (as in the example above), `~+` works like a pure sequence, and it also allows waveforms to be combined in other ways.

In the harmonics example above, we knew that the offsets of all of the component waveforms were 0 so `~+` behaves more like pure addition, but we can also use `Seq` to combine two arbitrary waveforms `a` and `b` into a chord (as shown below) using `Seq` to set the offset of the first waveform to 0.

```
Seq(0, a) ~+ b
```

As another example of how to use offsets, let's create a simple envelope using `~+`, `~-`, `~.`, and the `Time` waveform.

```
Sin(Const(2 * PI * 440), Const(0)) ~. (Seq(2, Fin(Time ~- Const(2), Const(0.5), Const(0))) ~+ Fin(Time ~- Const(1), Const(1.0) ~+ Const(-1.0), Const(0)))
```
This plays a 440Hz tone for three seconds, increasing the amplitude for the first two seconds (the "attack") and decreasing it to silence during the third (the "release"). Notice how the `~+` and `Seq` combinators are used to sequence the attack and release, and how the `~.` is used to combine the envelope with the tone. 

We noted above that non-zero offsets make binary combinators on waveforms non-communicative. Once we have specified an entire waveform -- that is, one that _won't_ be used to build other waveforms -- we can eliminate offsets by replacing `Seq` with a delay and using the `Append` combinator, which takes two waveforms and simply appends the samples from the second after those of the first. Every waveform containing `Seq` combinators can be translated to an overall offset and a waveform that uses `Append` but without any `Seq`s. For example, if `a` and `b` can be translated to `a'` and `b'` (with associated offsets), then the sum of those waveforms can be translated using `Append`.

```
Assuming...
a         ==>   a_offset,               a'
b         ==>   b_offset,               b'
Then...
a ~+ b    ==>   a_offset + b_offset,    a' ~+ Append(Fin(a_offset, Const(0)), b')
```

Happily, we can now reorder the two operands to `~+` or distribute `~*` over the arguments to `~+`. This will make it easier to optimize waveforms so that they can be used to generate samples more efficiently.

Our last example show how to combine waveforms at a much smaller scale. Up until now, we've considered combining waveforms that last for one or two seconds. What about waveforms that last for 0.002 seconds? The tones we've created so far have all been sine waves at their root. Sine waves are one type of periodic waveform, but they can be used to create other periodic waveforms as well. The `Alt` combinator picks between two waveforms based on the value of a third, called a trigger. For example, the following will generate a square wave.

```
Alt(Sin(Const(2.0 * PI * 440.0), Const(0)), Const(-1.0), Const(1.0))
```

There are a few other waveforms and waveform combinators available in Tuun and that are described briefly below. 

In summary, there are these basic waveforms:

 * `Const(value)` - generates an infinite number of samples with the given value
 * `Time` - generates samples with the time elapsed since the beginning of the waveform in seconds
 * `Noise` - generates an infinite number of random samples between -1 and 1
 * `Fixed([..])` - generates a fixed sequence of samples

These combinators that change how a waveform behaves in time and in relation to other waveforms:

 * `Fin(length, a)` - generates samples of `a` until `length` >= 0.0 (then truncates)
 * `Seq(offset, a)` - sets the offset of `a` (and the start of subsequent waveforms) when `offset` >= 0.0
 * `Append(a, b)` - generates samples from `a` and then from `b`

There are the arithmetic combinators that combine the samples themselves.

 * `a ~+ b` - adds sample points together
 * `a ~- b` - subtracts sample points
 * `a ~. b` - multiplies sample points together
 * `a ~/ b` - divides sample points
 * `Filter(a, b, c)` - modifies `a` using a finite or infinite impulse response filter

There are three combinators for describing periodic waveforms:

 * `Sin(a, b)` - generates a sine wave with frequency `a` and phase offset `b`
 * `Alt(trigger, a, b)` - generates samples from `a` when `trigger` is positive and from `b` otherwise
 * `Res(trigger, a)` - restarts the second waveform each time the `trigger` switches from negative to positive

And finally, there are these two waveforms that provide ways of dynamically interacting with waveforms through a user interface.

 * `Slider(_)` - generates samples dynamically based on user input (for example, a track-pad)
 * `Marked(a)` - provides updates as to when `a` starts and stops


For comparison, here are the lengths and offsets of each waveform:

| Waveform             | length                             | offset                      |
| ----------------     | ------                             | ------                      |
| `Const(_)`           | ∞                                  | 0                           |
| `Time`               | ∞                                  | 0                           |
| `Noise`              | ∞                                  | 0                           |
| `Slider(_)`          | ∞                                  | 0                           |
| `Fixed(v)`           | length of v                        | 0                           |
| `Fin(length, a)`     | position w/ `length` >= 0.0        | a.offset                    |
| `Seq(offset, a)`     | a.length                           | position w/ `offset` >= 0.0 |
| `Filter(a, b, c)`    | a.length                           | a.offset                    |
| `Marked(a)`          | a.length                           | a.offset                    |
| `Append(a, b)`.      | a.length + b.length                | a.offset + b.offset         |
| `a ~+ b`             | max(a.length, a.offset + b.length) | a.offset + b.offset         |
| `a ~. b`             | min(a.length, a.offset + b.length) | a.offset + b.offset         |
| `Sin(a, b)`          | min(a.length, b.length)            | a.offset + b.offset         |
| `Res(trigger, a)`    | trigger.length                     | trigger.offset              |
| `Alt(trigger, a, b)` | trigger.length                     | trigger.offset              |

This small set of combinators is enough to create synthesizers, filters, and even musical compositions with the help of the Tuun specification language.


## Tuun Specification Language

Tuun waveforms are designed to be simple, and they are more like an assembly language then a programming language. On the other hand, the Tuun specification language is a higher-order functional language that can be used to build abstractions and easily create complex sounds and even music!

```
expr ::= float
     | "fn" "(" var "," ... ")" "=>" expr
     | var 
     | expr "(" expr ")"
     | "(" expr "," ... ")"
     | "[" expr "," ... "]"
     | expr binary_op expr
     | unary_op expr
     | "let" binding, ... "in" expr
     | ...

pattern ::= var
        | "(" pattern, ... ")"

binding ::= pattern "=" expr

unary_op ::= "-" | "$" | "@" | ...
binary_op ::= "+" | "-" | "*" | "/" | "==" | "!=" | "<" | ...
```

Floating point literals, functions, variables, application, tuples, lists, operators, and "let" bindings are all pretty standard!

```
expr ::= ...
     | "{" expr "}"
     | "<" expr ">"
     | expr "|" expr
```

Tuun supports special syntax for chords and sequences – by "chords", we really just mean combining waveforms so that they are played simultaneously. Curly brackets (`{` and `}`) take a tuple of waveforms and turn that tuple into a single waveform representing a chord. Angle brackets (`<` and `>`) take a tuple of waveforms and sequence them. (As you might guess from above, those waveforms must include proper offsets to create a true sequence!)

All waveforms are values, and Tuun provides built-in functions to create them. When a floating point value appears in the context of a waveform, it's implicitly coerced into a constant waveform. Functions like `sin` are overloaded so that they can take either floating point values or waveforms. Note that, by convention, the specification language built-ins for waveforms (like `fin`, `seq`, and `time`) are written in lowercase.

We can now give a slightly more extensive version of the harmonics example, in part by defining a helper function that creates overtones. The dollar sign is used a shorthand for a sine wave with the given frequency in hertz and no phase offset.

```
$ = fn(freq) => sin(2 * pi * freq, 0),
overtone = fn(freq) => fn (x) => $(freq*x) * (1/x),
harmonics = fn(freq) => {map(overtone(freq), [1, 3, 5, 7, 9, 11])},
```
The `overtone` function creates a waveform by multiplying `freq` by `x` and then multiplying that waveform by a constant waveform to scale it down. Notice that there are _two_ types of multiplication here: multiplication in the specification language and multiplication in the waveform language (written as `~.` for clarity).

| Expression           | Evaluates to...                                        |
| ----------           | ---------------                                        |
| `3 * 440`            | `1320`                                                 |
| `overtone(440)(3)`   | `Sin(Const(2 * PI * 1320), Const(0)) ~. Const(.3333)`  |

Notice that, since `*` is overloaded in the specification language, the function `$` can take the frequency a single floating point value or as a waveform. (In the second case, the frequency will vary with time.)

We can also revisit our envelope example from above. This example makes use of the `|` operator, which denotes reverse application (enabling you to write the argument before the function you are passing it to). It's conventional in Tuun to write filters like ADSR in a curried-form, as shown below, so that they can be chained together.

This example first provides functions to create waveforms for the four steps of the filter. The filter function itself takes the five parameters, builds those waveforms, sequences them, and then combines them with the given waveform using `*`. (Remember that `<...>` is shorthand for a sequence of waveforms, that is, for reducing the given list of waveforms using `+`.)

```
// Helper function that takes a pair of floats and returns a linear waveform
linear = fn(initial, slope) => initial + (time * slope),
// Create waveforms for the four parts of the envelope:
Aw = fn(dur) => linear(0.0, 1.0 / dur) | fin(time - dur) | seq(time - dur),
Dw = fn(dur, level) => linear(1.0, (level - 1.0) / dur) | fin(time - dur) | seq(time - dur),
Sw = fn(dur, level) => level | fin(time - dur) | seq(time - dur),
Rw = fn(dur, level) => linear(level, -level / dur) | fin(time - dur) | seq(time - dur),
// Combine them to create a new filter:
ADSR = fn(attack_dur, decay_dur, sustain_level, sustain_dur, release_dur) =>
  fn(w) => (w | seq(0)) * <[Aw(attack_dur),
                            Dw(decay_dur, sustain_level),
                            Sw(sustain_dur, sustain_level),
                            Rw(release_dur, sustain_level)]>,
```
(If you want to use quartic instead of linear ramps, you'll just need to replace `linear` with a different waveform!)
