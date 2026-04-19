# Dynamic Waveforms

Tuun's `Generator` is inherently static: with the exception of random waveforms (`Noise`), each waveform describes a fixed set of samples, and given that waveform, the `Generator` will always produce that same set of samples. However, in many cases, we'd like to dynamically change the behavior of a waveform based on changes to the environment: keys are pressed, dials are turned, or a song has ended. All of these represent cases where we'd like to diverge from the original waveform definition. Helpfully, Tuun's `Tracker` supports not only the ability to `Play` a waveform, but also to `Modify` one. Several uses of `Modify` are described below. 

Waveform modification is accomplished with the help of a special waveform: `Marked` waveforms provide a way of substituting one waveform for part of another. For example, to change the frequency of a sine wave, we can use a marked waveform for the first parameter:

```
Sine(Marked(Slider("A"), Const(1643.9)), Const(0.0))
   — substitute Const(1845.1) for "Slider(A)" -->
Sine(Marked(Slider("A"), Const(1845.1)), Const(0.0))
```

Modification is different than wholesale replacement because the parts of the waveform that are not changed maintain their state. In the example using `Sine` above, the [accumulator](sine.md#accumulation) of the `Sine` waveform will persist, limiting audible artifacts that might otherwise occur.

Since modification occurs only as a command to the Tracker (or a call to the web component), changes only occur at the beginning of a generation quantum. Setting the buffer size will therefore affect the latency with which modifications go into effect.

## Sliders: faders and encoders

When creating new sounds or programming synthesizer components, it's often helpful to have real-time feedback on the effect of a parameter. For example, the best way to find the cutoff for a low-pass filter is often just to adjust the cutoff and listen to the result. Tuun provides a mechanism called "sliders" that supports this kind of real-time parameter tuning.

### Basic configuration and use

Sliders can be configured using four values:
```
label:initial-value:minimum:maximum
```
For example, a slider with the label "Q" that starts at 0.707 and can range between 0.1 and 1.1 would be written as the following.
```
Q:0.707:0.1:1.1
```

In the native app, sliders are written as annotations: special comments that appear on the line just before the expression that they annotate.
```
//#{sliders=["Q:0.707:0.1:1.1"]}
square(220) | lpf(Q, 2000)
```
If a MIDI controller is detected, sliders are mapped to encoders or faders. If no MIDI controller is detected, the first two sliders are mapped to the x- and y-axis of the mouse position.

In the web component, sliders are configured using an HTML attribute.
```html
<tuun-synth
  sliders='["Q:0.707:0.1:1.1"]'
  expression="square(220) | lpf(Q, 2000)"
  />
```

You can play with this example here:
<div class="container">
  <tuun-synth
    sliders='["Q:0.707:0.1:1.1"]'
    expression="square(220) | lpf(Q, 2000)" />
</div>

### Custom range mappings

In the examples above, movement of the physical or virtual control is always mapped linearly to a point between the minimum and maximum values. For many audio parameters, however, our perceptual response to a physical phenomenon is not linear. For example, our perception of pitch varies with exponential changes to frequency.

To accommodate this, sliders may also be specified with a custom mapping between the normalized value (which always falls between 0.0 and 1.0, inclusive) and the value that will be substituted into the waveform. This mapping is specified as a Tuun expression, which must evaluate to a function that takes a single parameter. Sliders with custom range mappings are specified by three values:
```
label:initial-normalized-value:denormalizing-function
```
Note that the initial value is specified in the *normalized* range; it will be passed to the denormalizing function to determine the initial value that is provided to the waveform.

For example, a frequency cutoff slider can be specified as follows.
```
cutoff:0.5:fn(x) => 100 * pow(100, x)
```

This can then be used in a different version of our low-pass filter example.
<div class="container">
  <tuun-synth
    sliders='["cutoff:0.5:fn(x) => 100 * pow(100, x)"]'
    expression="square(220) | lpf(0.707, cutoff)" />
</div>

### Implementation

Sliders are implemented by prepending a waveform expression with a set of bindings. For example, our first low-pass filter example would look something like this:
```
let
  Q = 0.707 | mark(slider("Q"))
in
  square(220) | lpf(Q, 2000)
```
Note that this `mark` pseudo-syntax is not supported by the parser, but is added automatically before waveform expressions are evaluated. In the example above, this corresponds to the following waveform:
```
Marked(slider("Q"), Const(0.707))
```

Changes to the slider (either in the linear case or the custom one) are then implemented as `Modify` commands to the `Tracker` or directly as substitutions in the web component. Since commands are only processed once per generation quantum, Tuun attempts to send at most one `Modify` command per slider per quantum. In addition, the updated waveform is constructed as a linear ramp between the old value and the new value. This helps to eliminate artifacts that would occur when parameters make sudden, large changes. For example, if the slider's (denormalized) value is changed to 0.8, the `Modify` command would look something like the following.

```
Modify {
    mark_id: slider("Q")
    waveform:
        Append(
            Fin {
                length: Subtract(Time, Const(quantum_duration_secs)),
                waveform: Add(
                    Multiply(
                        Time,
                        Const((0.8 - 0.707) / quantum_duration_secs),
                    ),
                    Const(0.707),
                ),
            },
            Const(0.8),
        )
    ..
}
```

## Stop

Tuun's `Tracker` doesn't support a `Stop` command, so infinite waveforms play forever unless they are modified. A simple implementation of "stop" would be to simply replace a waveform with an empty one:
```
Modify {
    mark_id: ...
    waveform: Fixed([])
    ..
}
```

However, this simple implementation can lead to a "pop" sound if the amplitude of the waveform is large. Instead, "stop" is implemented with a special `Marked` waveform and a short ramp. When `waveform` is played, it is first multiplied by a constant 1.0 that is marked using a mark called `Level`.

```
Multiply(
    waveform,
    Marked {
        id: Level,
        waveform: Const(1.0),
    },
)
```

When it's time to stop a waveform, this constant is replaced by the following ramp.

```
Modify {
    mark_id: Level
    waveform:
        Fin(
            length: Subtract(
                Time,
                Const(STOP_DURATION_SECS),
            ),
            waveform: Subtract(
                Const(1.0),
                Multiply(
                    Time,
                    Const(1.0 / STOP_DURATION_SECS),
                ),
            ),
        ),
    ..
}
```


## MIDI note-on and note-off

Handling MIDI note-on and note-off events is similar. Each MIDI instrument in Tuun is defined by a function that returns a pair of waveforms:
```
(key, velocity) -> (waveform: note_on, waveform: note_off)
```
When a note-on event is received from a controller, the note (or key) number and the velocity (as a value between 0.0 and 1.0) are passed to this function. The first waveform is played immediately. This waveform may be infinite or finite. In either case, when a note-off event is received, the second waveform (`note_off`) is substituted for the `Level` mark, just as in the case of stopping a waveform above. As in the case of stopping, a simple `Fixed([])` would work, but other waveforms can be used to produce instrument-specific results.

The following example of a MIDI instrument (based on parameters from [Jim Woodhouse's Euphonics site](https://euphonics.org/3-3-marimbas-and-xylophones/)) has a long (but finite) sustain when keys are held.

```
(fn(key, vel) =>
  ({map(over(@key), [1.0, 3.92, 9.24, 16.27, 24.22, 33.54, 42.97])}
      * vel
      | ADSR(0, 0.1, 0.3, 3.0, 2.0),
   Rw(0.5, 1.0)))
```
(Unfortunately, the Tuun web component doesn't yet support MIDI!)

<script type="module" src="tuun/tuun-synth.js"></script>
