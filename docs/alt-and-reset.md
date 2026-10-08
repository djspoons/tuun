# Alt and Reset

Tuun's `Alt` and `Reset` waveform combinators both rely on a "trigger" to determine their outputs. In both cases, the only **sign** of each sample in the trigger waveform is used. In the case of `Alt`, the sign of each sample of the trigger is examined to choose which of two other waveforms to use for the output. In the case of `Reset`, the trigger is used to determine when to restart sampling from an inner waveform: when the trigger passes from a negative to a positive value, the inner waveform's state is reset so that it begins re-generating samples from the beginning.

## Periodic Triggers

One important use of these combinators is the creation of new periodic waveforms, as demonstrated in the examples below. To do so, they use periodic triggers whose frequency is in the audible range.

Because of the behavior of `Reset`, it's conventional in Tuun for periodic waveforms that will be used with `Reset` to start the _positive_ portion of their output at the _beginning_ of the cycle. This is true of `sine` and of the other simple periodic waveforms defined in Tuun's standard library. In contrast, it's _not_ true of a cosine waveform, which splits the positive portion of its output between the beginning and end of its cycle (when measured from the origin).

### Square Waves

A square wave is a periodic waveform that alternates between a maximum value (in Tuun, 1.0) and a minimum value (-1.0). A square wave at a frequency $f$ is equivalent to the (weighted) sum of sine waves whose frequencies are the odd multiples of $f$. As such, it's a computationally cheap way of creating a rich tone that can be used in [subtractive synthesis](synthesis.md#Subtractive).

Square waves are a textbook application of the `Alt` waveform combinator. Using a translated `Phase` waveform as a trigger (which spends half of each cycle in the positive range and half in the negative range) `Alt` selects a sample from either a constant waveform with the value 1.0 or a constant waveform with the value -1.0.

<div class="container">
  <tuun-synth description="Square wave"
      open='["std"]'
      sliders='["frequency:0.3424:fn(x) => 100 * pow(10, x)"]'
      expanded>
    let
      square = fn(freq_hz) => alt(phase(freq_hz, 0) - 0.5, -1, 1)
    in
      0.2 * square(frequency)
  </tuun-synth>
</div>

<!-- TODO add anti-aliasing -->
<!-- TODO add sawtooth waves? -->

### Synchronized Oscillators

Many dual-oscillator analog synthesizers allow one oscillator to be synchronized with the other. This means that whenever the first oscillator restarts its cycle, the second oscillator _also_ restarts its cycle. The result is that the pitch of second oscillator will match that of the first, but it will add a richer timbre.

The example starts with two sine waves where the second is a minor sixth above the first. However, since the second waveform is synchronized using `Reset`, the resulting fundamental frequency matches that of the first.

You can use the slider to adjust the mix between the two oscillators. (Though even when you can't hear the first oscillator, the second is still sync'd with it.) You can also uncomment the second definition of `osc2` to hear an unsynchronized version of that waveform.

<div class="container">
  <tuun-synth description="Synchronized oscillator" 
      open='["std"]' 
      sliders='["mix:0.5:0:1"]'
      expanded>
    <script type="text/tuun">
      let
        dual = fn(freq_hz) => let 
          osc1 = $(freq_hz),
          osc2 = reset(osc1, $(add_semitones(freq_hz, 8))),
          // Uncomment to hear the second oscillator without reset
          //osc2 = $(add_semitones(freq_hz, 8)),
        in
          (osc1 * (1 - mix)) + (osc2 * mix)
      in
        dual(220)
    </script>
  </tuun-synth>
</div>

### Repeated Waveforms

Sometimes is useful to repeat a waveform. `Reset` can also be used with a trigger with much longer periods (and therefore lower frequencies).

<div class="container">
  <tuun-synth description="Repeated waveform" 
      open='["std"]' 
      sliders='["period_secs:1:0.25:2"]'
      expanded>
    <script type="text/tuun">
      let
        inst = fn(dur, freq_hz) => $freq_hz | fin(time - dur)
      in
        reset($(1/period_secs), inst(0.2, 220))
    </script>
  </tuun-synth>
</div>

## Non-periodic Triggers

Tuun does not require that trigger waveforms are periodic, and there are several useful examples of `Alt` that do use non-periodic triggers. For example, `Alt` can be used to implement mathematical functions that express some sort of condition: `min` and `max` can be implemented with `Alt`.

```
min = fn(x, y) => alt(x - y, y, x);
max = fn(x, y) => alt(x - y, x, y);
```

When defined using `Alt`, these `min` and `max` define the point-wise minimum and maximum of two waveforms. This has the expected behavior when used on constant waveforms, but can also be used to define functions like `clamp` on waveforms and even envelopes.

> Though Tuun also includes a conditional expression `if ... then ... else ...`, it's preferable to use `alt` unless the result will be passed to a function like `nth` or `unfold` that require a constant, integral waveform.


<script type="module" src="tuun/tuun-synth.js"></script>
