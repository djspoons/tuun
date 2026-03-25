#  Tuun Language Overview

Tuun is an interactive, language-based sound and music generation system. It has two main components:

 * A tracker, which interfaces with the underlying audio subsystem, managing things like sample rates and buffers. It takes commands like "play this!" and gives updates on the state of playback.
 * A visual UI that enables users to compose sounds, enter those commands, and see the results.

Tuun has two languages:

 * A language of low-level waveforms and waveform combinators that are used by the tracker to generate samples to feed to the audio system.
 * An expression language that's used in the UI to specify those waveforms.

We'll start with a brief introduction to the second one before we dive into the details.

Tuun's interactive expression language is a simple, functional language like OCaml or Standard ML. It supports floating point numbers, arithmetic, functions, and tuples. In addition, it has some built-in syntax and semantics that make it easy to express complicated waveforms. For example, the `$` operator takes a frequency and returns a waveform that will generate a tone at that frequency.

<div class="container">
  <tuun-synth expression="$220" />
</div>

Tuun lets you define abstractions so that you can easily convert from MIDI note numbers to frequencies (`@`), create notes of different lengths (`Qw` and `Hw`) and combine notes in a sequence:

<div class="container">
  <tuun-synth expression="<[$(@60) * Qw, $(@64) * Qw, $(@67) * Hw]>" />
</div>

The way that you use the Tuun language is up to you! There's nothing baked in about western music or MIDI. Instead, it's all built from the following waveforms and waveform combinators.

## Tuun Waveforms

Tuun has several primitive waveforms and waveform combinators. These are the "assembly language" of Tuun, and akin to the "unit generators" of the [MUSIC languages](https://en.wikipedia.org/wiki/MUSIC-N). Waveforms are designed to be orthogonal (each serves a different purpose) and minimal (there are no extra waveforms).

The first primitive, `Const`, isn't exactly a "wave" but is used to create waves: `Const` generates a stream where every sample is the same value. This can be used with the [`Sine` combinator](sine.md) to produce a sine wave. `Sine` takes two arguments: one for the angular frequency (in radians per second) and one for the phase offset. For example, the following will generate a tone at 220 Hz.

```
Sine(Const(2 * PI * 220), Const(0))
```

This should be reminiscent of the definition from trigonometry class:
$$
\sin(t) = 2πft + \phi
$$
where $f$ is the desired frequency (in Hertz) and $\phi$ is the desired phase offset.

Every waveform has an intrinsic property called its _length_, which may be finite or infinite. The length of `Const` is infinite, and the length of `Sine` is determined by its inputs: `Sine` generates one sample of output for each sample of inputs. This means that the expression above will generate a tone that goes on forever.

Since it's often useful to have waveforms that *don't* go on forever, Tuun includes the `Fin` combinator, which modifies a waveform to be finite. This leverages the `Time` combinator, which generates a stream where each sample is the time elapsed since the beginning of the waveform (in seconds). For example, the following will generate a tone at 220Hz for 2 seconds.

```
Fin(Subtract(Time, Const(-2)),
  Sine(Const(2 * PI * 220), Const(0)))
```

The length of `Fin` is given by its first parameter: `Fin` generates samples from this waveform until it gets a sample >= 0, at which point it stops. In the example above, it generates samples from the waveform `Subtract(Time, Const(2))`. The `Subtract` combinator subtracts each pair of corresponding samples, yielding a new stream. You can think of this waveform a bit like a countdown clock: it starts at -2 and then "counts down" (well, _up_) until it reaches 0. 

We can use binary operators to modify outputs of waveforms as well. The `Multiply` combinator multiplies each sample in the first waveform by the corresponding sample in the second waveform. For example, we can change the amplitude of the Sine wave by multiplying by a constant waveform.

```
Fin(Subtract(Time, Const(2)), 
  Multiply(Const(0.5), 
    Sine(Const(2 * PI * 220), Const(0))))
```

The length of a waveform like `Subtract(a, b)` (or `Add`, `Multiply`, or `Divide`) is the _minimum_ of the length of `a` and the length of `b`.

Putting these combinators together enables us to create more interesting sounds. For example, we can generate harmonics by applying progressively smaller constants to higher frequencies like the following.

```
Fin(Subtract(Time, Const(3)), 
  Add(Sine(Const(2 * PI * 220), Const(0)),
    Add(Multiply(Const(0.33), Sine(Const(2 * PI * 1320), Const(0))),
    Multiply(Const(0.2), Sine(Const(2 * PI * 2200), Const(0))))))
```
<div class="container">
  <tuun-synth description="Harmonics" expression="sine(2 * pi * 220, 0) + 0.33 * sine(2 * pi * 1320, 0) + 0.2 * sine(2 * pi * 2200, 0) | fin(time - 3)" />
</div>


Writing this out is starting to get a little tedious, though, and we'll see how to use Tuun expressions below to build a library of functions for easily generating harmonics and other complex waveforms.

As another example of how to use offsets, let's create a simple amplitude envelope using the `Multiply`, `Subtract`, `Fin`, `Time` and `Append` waveforms. `Append` takes two waveforms and outputs all of the samples of the first waveform, followed by samples from the second waveform.

```
Multiply(
  Sine(Const(2 * PI * 220), Const(0)),
  Append(
    Fin(Subtract(Time, Const(2)), Multiply(Time, Const(0.5))), 
    Fin(Subtract(Time, Const(1)), Add(Multiply(Time, Const(-1), 1)))))
```
<div class="container">
  <tuun-synth description="Simple amplitude envelope" expression="sine(2 * pi * 220, 0) * append(time * 0.5 | fin(time - 2), time * -1 + 1 | fin(time - 1))" />
</div>

This plays a 220Hz tone for three seconds, increasing the amplitude for the first two seconds (the "attack") and decreasing it to silence during the third (the "release"). 

Our last example shows how to combine waveforms at a much smaller scale. Up until now, we've considered combining waveforms that last for one or two seconds. What about waveforms that last for 0.002 seconds? The tones we've created so far have all been sine waves at their root. Sine waves are one type of periodic waveform, but they can be used to create other periodic waveforms as well. The `Alt` combinator picks between two waveforms based on the sign of a third, called a trigger. For example, the following will generate a square wave.

```
Alt(Sine(Const(2 * PI * 220), Const(0)), Const(1), Const(-1))
```
<div class="container">
  <tuun-synth description="Square wave">
    alt(sine(2 * pi * 220, 0), 1, -1) * 0.4 // cut the amplitude
  </tuun-synth>
</div>
There are a few other waveforms and waveform combinators available in Tuun and that are described briefly below. 

In summary, there are these basic waveforms:

 * `Const(value)` - generates samples with the given value
 * `Time` - generates samples with the time elapsed since the beginning of the waveform in seconds
 * `Noise` - generates random samples between -1 and 1
 * `Fixed([..])` - generates a fixed sequence of samples

These combinators that change how a waveform behaves in time and in relation to other waveforms:

 * `Fin(length, a)` - generates samples of `a` until `length` >= 0.0 (then truncates)
 * `Append(a, b)` - generates samples from `a` and then from `b`

There are the combinators that combine the samples themselves.

 * `Add(a, b)` - adds samples together
 * `Subtract(a, b)` - subtracts samples
 * `Multiply(a, b)` - multiplies samples together
 * `Divide(a, b)` - divides samples
 * `Merge(a, b)` - merges two waveforms by extending the shorter one, and then adding samples together
 * [`Filter(c, [b_0, ...], [a_1, ...])`](filter.md) - processes `c` using a finite or infinite impulse response filter with the given coefficients

There are three combinators for describing periodic waveforms:

 * [`Sine(a, b)`](sine.md) - generates a sine wave with angular frequency `a` and phase offset `b`
 * `Alt(trigger, a, b)` - generates samples from `a` when `trigger` is positive and from `b` otherwise
 * `Reset(trigger, a)` - restarts the second waveform each time the `trigger` switches from negative to positive

And finally, a combinator that provides ways of dynamically interacting with waveforms through a user interface. Marked waveforms are used to indicate when parts of a waveform start and stop as well as part of dynamically updating waveforms during playback.

 * `Marked(a)` - generates samples from `a`


For comparison, here are the lengths and offsets of each waveform:

| Waveform                | Length
| ----------------        | ------
| `Const(_)`              | ∞
| `Time`                  | ∞
| `Noise`                 | ∞
| `Fixed(v)`              | length of v
| `Fin(a, b)`             | position w/ `a` >= 0.0
| `Sine(a, b)`            | min(a.length, b.length)
| `Add(a, b)`, `Subtract(a, b)`, `Multiply(a, b)`, `Divide(a, b)` | min(a.length, b.length)
| `Merge(a, b)`.          |  max(a.length, b.length)
| `Append(a, b)`.         | a.length + b.length
| `Filter(c, [b_0, ...], [a_1, ...])` | c.length
| `Reset(trigger, a)`     | trigger.length
| `Alt(trigger, a, b)`    | trigger.length

This small set of combinators is enough to create synthesizers, filters, and even musical compositions with the help of the Tuun expressions.

## Tuun Expressions

While Tuun waveforms are designed to be simple, Tuun expressions form a higher-order functional language that can be used to build abstractions and easily create complex sounds and even music!

```
expr ::= float
     | string
     | bool
     | "fn" "(" var "," ... ")" "=>" expr
     | var 
     | expr "(" expr ")"
     | "(" expr "," ... ")"
     | "[" expr "," ... "]"
     | expr binary_op expr
     | unary_op expr
     | "if" expr "then" expr "else" expr
     | "let" binding, ... "in" expr
     | "{" expr "}"
     | ...

pattern ::= var
        | "(" pattern, ... ")"

binding ::= pattern "=" expr

unary_op ::= "-" | "$" | "@" | ...
binary_op ::= "+" | "-" | "*" | "/" | "&" | "|" | "==" | "!=" | "<" | ...
```

Tuun includes standard features like floating point literals, strings, booleans, functions, variables, application, tuples, lists, operators, and `let` bindings. Values include floating point literals, booleans, and strings. 

Tuun *waveforms* are also values, and Tuun provides built-in functions like `sine` to create them. Functions like `sine` are overloaded so that they can take either floating point values or waveforms. When a floating point value appears as argument to a built-in function like `sine`, it's implicitly coerced into a constant waveform. Note that the expression language built-ins for creating waveforms (like `fin`, `alt`, and `time`) are written in lowercase. Binary operators are written infix, with `&` used as the `Merge` waveform operator.

Tuun also includes a `|` ("pipe") operator, which denotes reverse application, enabling you to write the argument before the function you are passing it to. It's conventional in Tuun to write filters (like the ADSR example below) in a curried-form, so that they can be chained together. Built-in functions for `fin` and `filter` are also written this way. For example, a two second sine wave would be written as follows:
<div class="container">
  <tuun-synth expression="$220 | fin(time - 2)" />
</div>

Tuun supports special syntax for chords, combining waveforms so that they are played simultaneously using `Merge`. Curly brackets (`{` and `}`) take a list of waveforms and return a single waveform that plays them simultaneously. This is used both for chords as well as for creating complex tones with multiple overtones. 

We can now give a more extensive — and more concise — version of the harmonics example, in part by defining a helper function that creates overtones. The dollar sign is shorthand for a sine wave with the given frequency in hertz and no phase offset. The `over` function creates an overtone whose amplitude in inversely proportional to the distance between that overtone and the fundamental. (`$` and `over` are both in the standard context.)

<div class="container">
  <tuun-synth>
    <script type="text/tuun">
      let
        // $ computes a sine wave at the given frequency in hertz (which must be a float or waveform).
        $ = fn(freq_hz) => sine(2*pi * freq_hz, 0),
        over = fn(freq) => fn (x) => (1/x) * $(freq*x),
        odd_harmonics =
          fn(freq) => {map(over(freq), [1, 3, 5, 7, 9, 11])},
      in
        odd_harmonics(220)
    </script>
  </tuun-synth>
</div>

Notice that there are _two_ types of multiplication here: multiplication in the expression language and the waveform multiplication operator.

| Expression       | Evaluates to...                                                |
| ----------       | ---------------                                                |
| `3 * 220`        | `1320`                                                         |
| `over(220)(3)`   | `Multiply(Const(.3333), Sine(Const(8293.80), Const(0)))`       |

Since `*` is overloaded in the expression language, the function `$` can take the frequency a single floating point value or as a waveform. (In the second case, the frequency may vary with time.) The description of the [`Sine` waveform](sine.md#dynamic-frequency-and-phase) gives some examples of how this might be used.


### Sequencing

In most of the examples above, there's only a single note (whether played as a simple tone or one with harmonics). Of course, we also want to be able to play a sequence of notes! 

While the `Append` waveform combinator offers a way of combining two waveforms sequentially, there are two challenges in using it. The first challenge is that `Append` always starts the second waveform immediately after the end of the first, but the length of the first waveform might be shorter or longer than the point at which the second should start. For example, the staccato sound of a drum may end before the next note should start, or the sound of a piano with the sustain pedal held down may extend past the start of the next.

The second challenge is that using `Append` together with a silent waveform (as in the envelope example above) puts the onus of timing on the *second* waveform. However, it's usually the first note that "knows" when the second should start. For example, when we define the first waveform as a quarter note (in 4/4 time), we're saying that the second note should start one beat later.

To solve this problem, Tuun defines a new type of waveform that extends the underlying notion of a waveform with a new property called its *offset*. The offset of a waveform indicates when the next waveform in the sequence (if any) should start. Not all waveforms have offsets, but if they do, they can be put into sequences, and we call them "sequence-able" waveforms or (for reasons that will be clear in a moment) "seq" waveforms. A seq waveform has both a length and an offset. Its length determines how many samples it will generate, while its offset determines how it will be combined with other waveforms.

Concretely, seq waveforms are created and consumed using the following expressions:

```
expr ::= ... | seq | unseq | "<" expr ">"
binary_op ::= ... | "\"
```

The first is `seq` (pronounced like "seek") and it take takes two waveforms: the first determines the offset, while the second is the waveform to be played. Effectively, it turns that second waveform into a seq waveform. Analogous to the first parameter to `fin`, the offset is determined by the first position at which the offset waveform is positive. Also like `fin`, `seq` is written in curried form, and it's not uncommon to see the two used together. The following plays three notes, each with a length of two seconds but with only one second from the start of one to the start of the next.

<div class="container">
  <tuun-synth>
    <script type="text/tuun">
      let
        R = fn(dur) => fn(w) => w * (1 - time / dur) | fin(time - dur)
      in
        $220 * 0.6 | R(2) | seq(time - 1) 
          \ $440 * 0.6 | R(2) | seq(time - 1) 
          \ $660 * 0.6 | R(2)
    </script>
  </tuun-synth>
</div>

This example also makes use of the `\` or "followed by" operator. This operator takes a seq waveform and combines it with a second waveform using the offset of the first one.  It does so by reifying the offset as a finite, constant waveform and combining that with the other waveforms using `Merge` and `Append`. That is, the first parameter of `seq` becomes the first parameter of `fin`.

```
seq(offset, a) \ b     ==> a & append(0 | fin(offset), b)
```

Other operators are overloaded to pass offsets through. For example, when adding a seq waveform, the `seq` is pulled to the outside.
```
seq(offset, a) + b     ==> seq(offset, a + b)
a + seq(offset, b)     ==> seq(offset, a + b)
```
Note that most binary operators (other than `\`) can only take a single seq waveform, since it would be ambiguous how to combine multiple offsets.

Sometimes you want to use a seq waveform in a context where you want to ignore its offset. Use `unseq` to return the underlying waveform.
```
unseq(seq(offset, a))     ==> a
```

Angle brackets (`<` and `>`) take a list of waveforms and sequence them using the `\` operator. As you might guess, those waveforms must be seq waveforms (that is, they must include offsets) to create a true sequence!

Note that a `seq` applied to two values (that is, two waveforms) is also a value. If a seq waveform appears at the outermost level of evaluation (just before a waveform is played), an implicit `unseq` is applied before passing the waveform to the sample generator.

Finally, we now can revisit our envelope example from above. It uses `seq` and `<...>` to sequence the components of an attack-decay-sustain-release (ADSR) envelope. The example first defines functions to help create waveforms for those four components. In addition to an input waveform, the `ADSR` function takes five parameters denoting the duration and levels of the four parts of the envelope. (`ADSR` is written in a curried form so that it may be used with `|`.) It builds those component waveforms, sequences them, and then combines them with the input waveform using `*`.

<div class="container">
  <tuun-synth>
    let
      // Helper function that takes a pair of floats and returns a linear waveform
      linear = fn(initial, slope) => initial + (time * slope),
      // Create waveforms for the four parts of the envelope:
      Aw = fn(dur) => linear(0.0, 1.0 / dur) | fin(time - dur) | seq(time - dur),
      Dw = fn(dur, level) => linear(1.0, (level - 1.0) / dur) | fin(time - dur) | seq(time - dur),
      Sw = fn(dur, level) => level | fin(time - dur) | seq(time - dur),
      // N.B. that R is not seq, since it is assumed to be the last part of the envelope.
      Rw = fn(dur, level) => linear(level, -level / dur) | fin(time - dur),
      // Combine them to create a new filter:
      ADSR = fn(attack_dur, decay_dur, sustain_level, sustain_dur, release_dur) =>
        fn(w) => w * <[Aw(attack_dur),
                       Dw(decay_dur, sustain_level),
                       Sw(sustain_dur, sustain_level),
                       Rw(release_dur, sustain_level)]>,
    in
      $220 | ADSR(0.1, 0.5, 0.6, 2, 1)
  </tuun-synth>
</div>
(If you want to use quartic instead of linear ramps, you'll just need to replace `linear` with a different waveform!)

<script type="module" src="tuun/tuun-synth.js"></script>
