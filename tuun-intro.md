#  Tuun

Tuun is an interactive, language-based sound and music generation system. It has two main components:

 * A tracker, which interfaces with the underlying audio subsystem, managing things like sample rates and buffers. It takes commands like "play this!" and gives updates on the state of playback.
 * An visual UI that enables users to compose sounds, enter those commands, and see the results.

Tuun has two languages:

 * A language of low-level waveform combinators that's used by the tracker to generate samples to feed to the audio system.
 * A language that's used in the UI to specify those waveform combinators.

We'll start with a brief introduction to the second one before we dive into the details.

Tuun's interactive language is a simple, functional language like OCaml or Standard ML. It supports floating point numbers, arithmetic, functions, and tuples. In addition, it has some built-in syntax and semantics that make it easy to express complicated waveforms. For example, the `$` operator takes a frequency and returns a waveform combinator that will generate a tone at that frequency.

```
$440
```

Tuun lets you define abstractions so that you can easily convert from MIDI note numbers to frequencies (`#`), create notes of different lengths (`Q` and `H`) and combine notes in a sequence:

```
[(Q$#60, Q$#64, H$#67)]
```

The way that you use the Tuun language is up to you! There's nothing baked in about western music or notation. Instead it's all built from the following waveform combinators.

## Tuun Waveform Combinators

Tuun has seven primitive waveform combinators.

The first is a combinator that generates a sine wave. For example, the following will generate a tone at 440Hz.

```
SineWave(440)
```

When executed by the tracker, `SineWave` generates samples in the form of a sine wave with the given frequency and a unit amplitude. `SineWave` is an unbounded generator: it will generate samples forever. 

Since it's often useful to have waveforms that *don't* go forever, the `Fin` combinator modifies a waveform to be finite. Like other parts of the tracker, duration is specified in terms of the number of beats. For example, the following will generate a tone at 440Hz for 1 beat.

```
Fin(1, SineWave(440))
```

Every waveform has an intrinsic property called its _length_, which may be finite or infinite. The length of `SineWave` is always infinite. The length of `Fin` is given by its parameter.

Another pair of combinators lets us control the amplitude of a wave form. For example, we can half the amplitude of the sine wave by multiplying by a constant waveform.

```
Fin(1, SineWave(440) ~. Const(0.5))
```

Here, the `Const` combinator generates a constant stream of samples. The `~.` combinator multiplies each sample in the first waveform by the corresponding sample in the second waveform. Similarly, the `~+` combinator adds each pair of corresponding samples.

> Note that the length of a waveform `a ~+ b` is the _maximum_ of the length of `a` and the length of `b` (since the sum can continue generating samples as long as one of the components is), and the length of a waveform `a ~. b` is the _minimum_ of the length of `a` and the length of `b` (since once the length of one waveform has been exceeded, it will effectively generate only zeros forever).

Putting these combinators together enables us to create a number of interesting sounds. For example, we can generate harmonics by applying progressively smaller constants to higher frequencies like the following.

```
Fin(4, SineWave(440) ~+ (SineWave(1320) ~. Const(0.33)) ~+ (SineWave(2200) ~. Const(0.2)))
```

Writing this out is starting to get a little tedious, though, and we'll see how to use the waveform specification language below to build a library of functions for easily generating harmonics and other complex waveforms.

While we've seen how to create finite waveforms, we haven't yet see how to describe a _sequence_ of waveforms. While the tracker is responsible for high-level sequencing waveforms (usually at the level of musical phrases), the combinators within a given waveform don't all need start at the same time.

To support sequencing, every waveform has another property that determines the _offset_ of the subsequent waveform. A waveform's offset doesn't effect how that waveform will generate samples, but it does affect how it's combined with other waveforms. Waveforms like `SineWave` and `Const` have an offset of 0.

To give a waveform an offset, we use the `Seq` combinator, which modifies another waveform to have a specified offset. `Seq(duration, a)` always has an offset of `duration` regardless of the offset of `a`. (`Seq` is the analogue to the `Fin`!)

You could imagine a combinator called `Then` that takes two waveforms `a` and `b` and that first generates samples `a`, followed by samples from `b` starting at the offset of `a`. The following would generate a 440Hz tone for one beat, followed by a 880Hz tone for one beat.


```
Then(
    Seq(1, Fin(1, SineWave(440))),
    Fin(1, SineWave(880))
)
```

Why do we need separate notions of length and offset? One example would be the notes played on a piano with the sustain pedal held down: we want the second note to start on the second beat, but we don't want the first note to stop. In general, a waveform's length is crucial to how it generates samples, while its offset controls how it can be combined with other waveforms.

We need to revisit the behavior of `~+` and `~.` in the context of offsets. In the examples above, they were only applied to waveforms with offsets equal to 0. However, the offset of the left argument is used by both combinators: the second waveform only takes effect _after_ the offset indicated by the first. Thus in the case of `a ~+ b`, the samples of `b` are added to `a` only starting at the offset of `a`. (This means that these are not communicative combinators!)

Given this, we can now see why we don't need a separate `Then` combinator: `~+` already provides the functionality we need! That is, `a ~+ b` generates samples from `a` until the offset of `a` is reached, at which point it adds corresponding samples from the two waveforms together. When the length of `a` is less than or equal to its offset (as in the example above), `~+` works like a pure sequence, but it also allows waveforms to be combined in other ways.

In the harmonics example above, we knew that the offsets of all of the component waveforms were 0 so `~+` behaves more like pure addition, but we can also use `Seq` to combine two arbitrary waveforms `a` and `b` into a chord (as shown below) using `Seq` to set the offset of the first waveform to 0.

```
Seq(0, a) ~+ b
```

As another example of how to use offsets, let's create a simple envelope using `~+`, `~.`, and another combinator called `Linear`, which generates samples along a line according to an intercept and slope. 

```
SineWave(440) ~. (Seq(1, Fin(1, Linear(0.5, 2.0))) ~+ Fin(1, Linear(1.0, -1.0)))
```
This plays a 440Hz tone for two beats, increasing the amplitude for the first beat (the "attack") and decreasing it to silence during the second (the "release"). Notice how the `~+` and `Seq` combinators are used to sequence the attack and release, and how the `~.` is used to combine the envelope with the tone. 

To review, there are three base combinators in Tuun:

 * `SineWave(frequency)` - generates a sine wave
 * `Const(value)` - generates all samples with the given value
 * `Linear()` - generates samples along a line

There are two combinators that change how a waveform behaves in time:

 * `Fin(duration)` - stops generating samples after the given duration
 * `Seq(duration)` - marks the offset at which subsequent waveforms should take effect

And finally there are two binary combinators that modify the samples themselves.

 * `a ~+ b` - adds sample points together
 * `a ~. b` - multiplies sample points together

And for comparison, here are the lengths and offsets of each combinator:

|                    | length                             | offset              |
| ----------------   | ------                             | ------              |
| `SineWave(_)`      | ∞                                  | 0                   |
| `Const(_)`         | ∞                                  | 0                   |
| `Linear(_)`        | ∞                                  | 0                   |
| `Fin(duration, a)` | duration                           | a.offset            |
| `Seq(duration, a)` | a.length                           | duration            |
| `a ~+ b`           | max(a.length, a.offset + b.length) | a.offset + b.offset | 
| `a ~. b`           | min(a.length, a.offset + b.length) | a.offset + b.offset |

This might seem like a small set of combinators, but 


## Tuun Specification Language

Tuun waveforms are designed to be simple, and they are more like an assembly language then a programming language. On the other hand, the Tuun specification language is a higher-order functional language that can be used to build abstractions and easily create complex sounds and even music!

```
expr ::= float
     | fn (var, ...) => expr
     | var 
     | expr expr
     | (expr, ...)
     | expr binary_op expr
     | unary_op expr
     | let var = expr, ... in expr
     | ...

unary_op ::= $ | ...
binary_op ::= + | - | * | / | ~+ | ~* | ...
```

Floating point literals, functions, variables, application, tuples, operators, and "let" bindings are all pretty standard!

```
expr ::= ...
     | <expr>
     | [expr]
     | fin | seq | const | ...
```

Tuun supports special syntax for chords and sequences. Angle brackets (`<` and `>`) take a tuple of waveforms and turn that tuple into a single waveform representing a chord. Similarly, square brackets (`[` and `]`) take a tuple of waveforms and sequence them. (As you might guess from above, those waveforms must include proper offsets to create a true sequence!)

All combinator waveforms are values, and Tuun provides built-in functions (including the unary operator `$` for `SineWave`) to create them. Note that, by convention, the specification language built-ins for waveform combinators (like `fin`, `seq`, `const`, and `linear`) are spelled with lowercase.

We can now give a slightly more extensive version of the harmonics example, in part by defining a helper function that creates overtones.

```
overtone = fn (x, freq) => $(freq*x) ~. const(1/x),
harmonics = fn(freq) =>
   <($freq, 
     overtone(3, freq),
     overtone(5, freq),
     overtone(7, freq),
     overtone(9, freq))>,
```
The definition of `overtone` can be read as "create a waveform by multiplying `freq` by `x` and then multiply that waveform by a constant waveform." Notice that there are _two_ types of multiplication here: multiplication in the specification language using `*` and multiplication in the waveform language using `~.`.

| Expression           | Evaluates to...                   |
| ----------           | ---------------                   |
| `3 * 440`            | `1320`                            |
| `overtone(3, 440)`   | `SineWave(1320) ~. Const(.3333)`  |

You can write `~.` in the specification language since it is bound to a built-in operator, but all that operator does is create the combinator that will be evaluated by the tracker.

We can also revisit our envelope example from above. This example makes use of the `|` operator, which denotes reverse application (enabling you to write the argument before the function you are passing it to). It's conventional in Tuun to write filters like ADSR in a curried-form, as shown below, so that they can be chained together.

This example first provides functions to create waveforms for the four steps of the filter. The filter function itself takes the five parameters, builds those waveforms, sequences them, and then combines them with the given waveform.

```
Aw = fn (dur) => linear(0.0, 1.0 * dur) | fin(dur) | seq(dur),
Dw = fn (dur, level) => linear(1.0, (level - 1.0) * dur) | fin(dur) | seq(dur),
Sw = fn (dur, level) => const(level) | fin(dur) | seq(dur),
Rw = fn (level, dur) => linear(level, -level * dur) | fin(dur) | seq(dur),
ADSR = fn (attack_dur, decay_dur, sustain_level, sustain_dur, release_dur) => 
  fn(w) => (w | seq(0)) ~. [(Aw(attack_dur), 
                           Dw(decay_dur, sustain_level),
                           Sw(sustain_dur, sustain_level),
                           Rw(sustain_level, release_dur))],
```
(If you want to use quartic instead of linear ramps, you'll need to wait until someone builds a primitive waveform for that!)
