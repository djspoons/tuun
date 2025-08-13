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

Tuun has several primitive waveforms and waveform combinators. The first two, `Const` and `Time`, aren't exactly "waves" but are used to create them: `Const` generates a stream where every sample is the same value, and `Time` generates a stream where each sample is the time elapsed since the beginning of the waveform. These two can be used with the `Sin` combinator to produce a sine wave. For example, the following will generate a tone at 440Hz.

```
Sin(Const(440) ~. Time)
```
`Const` and `Time` are both infinite waveforms, and so the expression above will generate a tone that goes on forever. However, since it's often useful to have waveforms that *don't* go forever, the `Fin` combinator modifies a waveform to be finite. For example, the following will generate a tone at 440Hz for 1 second.

```
Fin(1, Sin(Const(440) ~. Time))
```

Every waveform has an intrinsic property called its _length_, which may be finite or infinite. The lengths of `Const` and `Time` are infinite, the length of `Sin` is determined by its input, and the length of `Fin` is given by its parameter.

We used the `~.` combinator above to combine the inputs to `Sin`, and we can also use it to modify its output. This combinator multiplies each sample in the first waveform by the corresponding sample in the second waveform. Below it's used to control the amplitude of a waveform. For example, we can half the amplitude of the sine wave by multiplying by a constant waveform.

```
Fin(1, Sin(Const(440) ~. Time) ~. Const(0.5))
```

As you might expect, and analogous to `~.`, the `~+` combinator _adds_ each pair of corresponding samples.

> The length of a waveform `a ~+ b` is the _maximum_ of the length of `a` and the length of `b` (since the sum can continue generating samples as long as one of the components is), and the length of a waveform `a ~. b` is the _minimum_ of the length of `a` and the length of `b` (since once the length of one waveform has been exceeded, it's as if it will generate only zeros forever).

Putting these combinators together enables us to create more interesting sounds. For example, we can generate harmonics by applying progressively smaller constants to higher frequencies like the following.

```
Fin(4, Sin(Const(440) ~. Time) ~+ (Sin(Const(1320) ~. Time) ~. Const(0.33)) ~+ (Sin(Const(2200 ~. Time)) ~. Const(0.2)))
```

Writing this out is starting to get a little tedious, though, and we'll see how to use the waveform specification language below to build a library of functions for easily generating harmonics and other complex waveforms.

While we've seen how to create finite waveforms, we haven't yet see how to describe a _sequence_ of waveforms. While the tracker is responsible for high-level sequencing waveforms (for example, at the level of musical phrases), the components within a given waveform don't all need start at the same time (for example, notes within a musical phrase).

To support sequencing, every waveform has another property that determines the _offset_ of the subsequent waveform. A waveform's offset doesn't effect how that waveform will generate samples, but it does affect how it's combined with other waveforms. Waveforms like `Const` have an offset of 0.

To give a waveform a non-zero offset, we use the `Seq` combinator, which modifies another waveform to have a specified offset. `Seq(duration, a)` always has an offset of `duration` regardless of the offset of `a`. (`Seq` is the analogue to the `Fin`!)

You could imagine a combinator called `Then` that takes two waveforms `a` and `b` and that first generates samples `a`, followed by samples from `b` starting at the offset of `a`. The following would generate a 440Hz tone for one second, followed by a 880Hz tone for one second.


```
Then(
    Seq(1, Fin(1, Sin(Const(440) ~. Time))),
    Fin(1, Sin(Const(880) ~. Time))
)
```

Why do we need separate notions of length and offset? One example where we want both is when emulating notes played on a piano with the sustain pedal held down: we might want the second note to start on the second beat, but we don't want the first note to stop until the pedal is released. Again, a waveform's length is essential to how it generates its own samples, while its offset controls how it is combined with other waveforms.

We need to revisit the behavior of `~+` and `~.` for waveforms with offsets. In the examples above, they were only applied to waveforms with offsets equal to 0. However, the offset of the left argument is used by both combinators: the second waveform only takes effect _after_ the offset indicated by the first. Thus in the case of `a ~+ b`, the samples of `b` are added to `a` only starting at the offset of `a`. This means that these are not communicative combinators! (Note that any samples in the left-hand operand that occur before its offset are just passed through.)

Given this, we can now see there no need for a separate `Then` combinator: `~+` already provides the required functionality! That is, `a ~+ b` generates samples from `a` until the offset of `a` is reached, at which point it adds corresponding samples from the two waveforms together. When the length of `a` is less than or equal to its offset (as in the example above), `~+` works like a pure sequence, and it also allows waveforms to be combined in other ways.

In the harmonics example above, we knew that the offsets of all of the component waveforms were 0 so `~+` behaves more like pure addition, but we can also use `Seq` to combine two arbitrary waveforms `a` and `b` into a chord (as shown below) using `Seq` to set the offset of the first waveform to 0.

```
Seq(0, a) ~+ b
```

As another example of how to use offsets, let's create a simple envelope using `~+`, `~.`, and the `Time` waveform.

```
Sin(Const(440) ~. Time) ~. (Seq(2, Fin(2, Const(0.5) ~. Time)) ~+ Fin(1, Const(1.0) ~+ Const(-1.0) ~. Time))
```
This plays a 440Hz tone for three seconds, increasing the amplitude for the first two seconds (the "attack") and decreasing it to silence during the third (the "release"). Notice how the `~+` and `Seq` combinators are used to sequence the attack and release, and how the `~.` is used to combine the envelope with the tone. 

Sine waves are one type of periodic waveform, and they can be used to create other periodic waveforms as well. The `Alt` combinator picks between two waveforms based on the value of a third, called a trigger. For example, the following will generate a square wave.
```
Alt(Sin(Const(220.0) ~. Time), Const(-1.0), Const(1.0))
```

There are a few other waveforms and waveform combinators available in Tuun and that are described briefly below. 

In summary, there are these basic waveforms:

 * `Const(value)` - generates an infinite number of samples with the given value
 * `Time` - generates samples with the time elapsed since the beginning of the waveform in seconds
 * `Noise` - generates an infinite number of random samples between -1 and 1
 * `Fixed([..])` - generates a fixed sequence of samples

These two combinators that change how a waveform behaves in time in relation to other waveforms:

 * `Fin(duration)` - stops generating samples after the given duration
 * `Seq(duration)` - sets the offset at which subsequent waveforms should take effect

There are the arithmetic combinators that combine the samples themselves.

 * `a ~+ b` - adds sample points together
 * `a ~. b` - multiplies sample points together
 * `a ~* b` - convolves the points of `a` with `b`

There are three combinators for describing periodic waveforms:

 * `Sin(a)` - takes the sine of each sample in `a`
 * `Alt(trigger, a, b)` - generates samples from `a` when `trigger` is positive and from `b` otherwise
 * `Res(trigger, a)` - restarts the second waveform each time the `trigger` switches from negative to positive

And finally, there are these two waveforms that provide ways of dynamically interacting with waveforms through a user interface.

 * `Dial(_)` - generates values dynamically based on user input
 * `Marked(a)` - generates the values of `a` and also provides updates as to when `a` starts and stops


For comparison, here are the lengths and offsets of each waveform:

| Waveform             | length                             | offset              |
| ----------------     | ------                             | ------              |
| `Const(_)`           | ∞                                  | 0                   |
| `Time`               | ∞                                  | 0                   |
| `Noise`              | ∞                                  | 0                   |
| `Dial(_)`            | ∞                                  | 0                   |
| `Fixed(v)`           | length of v                        | 0                   |
| `Fin(duration, a)`   | duration                           | a.offset            |
| `Seq(duration, a)`   | a.length                           | duration            |
| `Sin(a)`             | a.length                           | a.offset            |
| `Marked(a)`          | a.length                           | a.offset            |
| `a ~* b`             | a.length + (b.length / 2)          | a.offset            |
| `a ~+ b`             | max(a.length, a.offset + b.length) | a.offset + b.offset |
| `a ~. b`             | min(a.length, a.offset + b.length) | a.offset + b.offset |
| `Res(trigger, a)`    | trigger.length                     | trigger.offset      |
| `Alt(trigger, a, b)` | trigger.length                     | trigger.offset      |

This might seem like a small set of combinators, but it's enough to create synthesizers, filters, and even musical compositions with the help of the Tuun specification language.


## Tuun Specification Language

Tuun waveforms are designed to be simple, and they are more like an assembly language then a programming language. On the other hand, the Tuun specification language is a higher-order functional language that can be used to build abstractions and easily create complex sounds and even music!

```
expr ::= float
     | fn (var, ...) => expr
     | var 
     | expr expr
     | (expr, ...)
     | [expr, ...]
     | expr binary_op expr
     | unary_op expr
     | let var = expr, ... in expr
     | ...

unary_op ::= - | $ | @ | ...
binary_op ::= + | - | * | / | ~+ | ~. | ...
```

Floating point literals, functions, variables, application, tuples, lists, operators, and "let" bindings are all pretty standard!

```
expr ::= ...
     | {expr}
     | <expr>
     | pow | sqrt | map | reduce | time | noise | fixed | fin | seq | mark | sin | res | alt | ...
```

Tuun supports special syntax for chords and sequences – by "chords", we really just mean combining waveforms so that they are played simultaneously. Curly brackets (`{` and `}`) take a tuple of waveforms and turn that tuple into a single waveform representing a chord. Angle brackets (`<` and `>`) take a tuple of waveforms and sequence them. (As you might guess from above, those waveforms must include proper offsets to create a true sequence!)

All waveforms are values, and Tuun provides built-in functions to create them. When a floating point value appears in the context of a waveform, it's implicitly coerced into a constant waveform. Note that, by convention, the specification language built-ins for waveforms (like `fin`, `seq`, and `time`) are written in lowercase.

We can now give a slightly more extensive version of the harmonics example, in part by defining a helper function that creates overtones.

```
$ = fn(freq) => sin(freq ~. time),
overtone = fn(freq) => fn (x) => $(freq*x) ~. (1/x),
harmonics = fn(freq) => {map(overtone(freq), [1, 3, 5, 7, 9, 11])},
```
The `overtone` function creates a waveform by multiplying `freq` by `x` and then multiplying that waveform by a constant waveform to scale it down. Notice that there are _two_ types of multiplication here: multiplication in the specification language using `*` and multiplication in the waveform language using `~.`.

| Expression           | Evaluates to...                             |
| ----------           | ---------------                             |
| `3 * 440`            | `1320`                                      |
| `overtone(440)(3)`   | `Sin(Const(1320) ~. Time) ~. Const(.3333)`  |

You can write `~.` in the specification language since it is bound to a built-in operator, but all that operator does is to create the combinator that will be evaluated by the tracker.

We can also revisit our envelope example from above. This example makes use of the `|` operator, which denotes reverse application (enabling you to write the argument before the function you are passing it to). It's conventional in Tuun to write filters like ADSR in a curried-form, as shown below, so that they can be chained together.

This example first provides functions to create waveforms for the four steps of the filter. The filter function itself takes the five parameters, builds those waveforms, sequences them, and then combines them with the given waveform using `~.`. (Remember that `<...>` is shorthand for a sequence of waveforms, that is, for reducing the given list of waveforms using `~+`.)

```
// Helper function that takes a pair of floats and returns a linear waveform
linear = fn(initial, slope) => initial ~+ (time ~. slope),
// Create waveforms for the four parts of the envelope:
Aw = fn(dur) => linear(0.0, 1.0 / dur) | fin(dur) | seq(dur),
Dw = fn(dur, level) => linear(1.0, (level - 1.0) / dur) | fin(dur) | seq(dur),
Sw = fn(dur, level) => level | fin(dur) | seq(dur),
Rw = fn(dur, level) => linear(level, -level / dur) | fin(dur) | seq(dur),
// Combine them to create a new filter:
ADSR = fn(attack_dur, decay_dur, sustain_level, sustain_dur, release_dur) =>
  fn(w) => (w | seq(0)) ~. <[Aw(attack_dur),
                             Dw(decay_dur, sustain_level),
                             Sw(sustain_dur, sustain_level),
                             Rw(release_dur, sustain_level)]>,
```
(If you want to use quartic instead of linear ramps, you'll just need to change how the `Time` waveform is used!)
