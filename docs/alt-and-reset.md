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

You can use the slider to adjust the mix between the two oscillators. (Though even when you can't hear the first oscillator, the second is still synchronized with it.) You can also uncomment the second definition of `osc2` to hear an unsynchronized version of that waveform.

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

This example does not work, however, if the frequency of synchronized oscillator varies with time. A trivial example would be a frequency sweep as shown here:

```
osc2 = reset(osc1, $(freq_hz + 100 * time))
```

Since the `time` waveform appears within the `reset`, it restarts from 0.0 at the beginning of each period of `osc1`. If `freq_hz` is 220, the `time` waveform never gets larger than about 0.0045, meaning that the second oscillator frequency only varies by about 0.5 Hz (instead of sweeping across 100s of Hz).

The correct way to implement this is to use a `Shared` waveform. We want to compute the frequency of the second oscillator independently of the `reset`, as follows.

```
share(freq_hz + 100 * time, fn(osc2_freq_hz) =>
  ...
  osc2 = reset(osc1, $osc2_freq_hz),
  ...
```

Here, `osc2_freq_hz` will be generated in parallel with the `reset`. The `phase` waveform's accumulator (used inside of `$`) is reset to 0.0 every time `osc1` triggers, but when `phase` reads from the frequency waveform, it will receive samples based on the time that has passed since the beginning of `osc1` (rather than since the beginning of the most recent period of `osc1`).

You can think of each waveform as having its own "clock," which it receives from its parent and passes on to its children. This clock determines that value of a `time` sample as well as the position within a `Fixed` waveform. Whenever it is triggered, a `Reset` waveform restarts its child with a new clock. `Shared` and `Ref`, on the other hand, enable a nested waveform (the `Ref`) to reach up and use the clock of an enclosing waveform (other than its parent).

The example below synchronizes a 10-second frequency sweep to a fixed oscillator at 220 Hz. Again, you can adjust the mix between the two oscillators using the slider.

<div class="container">
  <tuun-synth description="Synchronized oscillator with variable frequency" 
      open='["std"]' 
      sliders='["mix:1:0:1"]'
      expanded>
    <script type="text/tuun">
      let
        dual = fn(freq_hz) =>
          share(freq_hz + 100 * time, fn(osc2_freq_hz) =>
          // Replace the above line to hear a version without `share`
          //let osc2_freq_hz = freq_hz + 100 * time in (
          let 
            osc1 = $(freq_hz),   
            osc2 = reset(osc1, $osc2_freq_hz),
          in
            (osc1 * (1 - mix)) + (osc2 * mix))
      in
        dual(220) | fin(time - 10)
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

Tuun does not require that trigger waveforms are periodic, and there are several useful examples of `Alt` that do use non-periodic triggers. For example, `Alt` can be used to implement mathematical functions that express some sort of condition: `min`, `max`, and `abs` can be implemented with `Alt`.

```
min = fn(x, y) => alt(x - y, y, x);
max = fn(x, y) => alt(x - y, x, y);
abs = fn(x) => alt(x, x, -x));
```

When defined using `Alt`, these `min` and `max` define the point-wise minimum and maximum of two waveforms, and `abs` computes the point-wise absolute value. These have the expected behavior when used on constant waveforms; they can also be used to define functions like `clamp` on waveforms and even envelopes.

> Though Tuun also includes a conditional expression `if ... then ... else ...`, it's preferable to use `alt` unless the result will be passed to a function like `nth` or `unfold` that require a constant, integral waveform.

The standard library uses `share` when defining these functions.

```
min = fn(x, y) => share(x, fn(x) => share(y, fn(y) => alt(x - y, y, x)));
max = fn(x, y) => share(x, fn(x) => share(y, fn(y) => alt(x - y, x, y)));
abs = fn(x) => share(x, fn(x) => alt(x, x, -x));
```

There are two important reasons for this approach. First, this implementation is more efficient since samples for each of the parameters will only be generated once. Second, if a parameter contains a `Noise` waveform (which is not deterministic), copying the parameter can yield unexpected results. When a `Noise` waveform is copied for each use of a parameter, each copy will produce different random samples. As the example below shows, this means that `abs_copy` generates random samples in the full range between -1 and 1.

```
abs_copy(noise)
 --> (fn(x) => alt(x, x, -x))(noise)
 --> alt(noise, noise, -noise)
 --> alt([0.1, -0.3, ...], [0.2, -0.4, ...], -[-0.5, 0.7, ...])
 --> alt([0.1, -0.3, ...], [0.2, -0.4, ...], [0.5, -0.7, ...])
 --> [0.2, -0.7, ...]

```

Using a `share`-based implementation avoids this by ensuring that the same samples are used both for the comparison and the result.

<!-- TODO Maybe talk about min/max-based envelopes here? -->

<script type="module" src="tuun/tuun-synth.js"></script>
