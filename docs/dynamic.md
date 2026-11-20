# Dynamic Waveforms

Tuun's `Generator` is inherently static: with the exception of random waveforms (`Noise`), each waveform describes a fixed set of samples, and given that waveform, the `Generator` will always produce that same set of samples. However, in many cases, we'd like to dynamically change the behavior of a waveform based on changes to the environment: keys are pressed, dials are turned, or a song has ended. All of these represent cases where we'd like to diverge from the original waveform definition. Helpfully, Tuun's `Tracker` supports not only the ability to `Play` a waveform, but also to `Modify` one. Several uses of `Modify` are described below. 

Waveform modification is accomplished with the help of a special waveform: `Marked` waveforms provide a way of substituting one waveform for part of another. For example, to change the frequency of a sine wave, we can use a marked waveform for the first parameter:

```
Sin(Const(2 * PI) * Phase(Marked(slider(A:1, "freq"), Const(1643.9)), Const(0.0)))
   — substitute Marked(slider(A:1, "freq"), Const(1845.1)) for slider(A:1, "freq") -->
Sin(Const(2 * PI) * Phase(Marked(slider(A:1, "freq"), Const(1845.1)), Const(0.0)))
```

Modification takes a "mark" identifier and new waveform, and replaces each `Marked` waveform with a matching identifier with the new waveform. To allow later modifications, the new waveform should include the mark again (as shown in this example).

Modification is different than wholesale replacement because the parts of the waveform that are _not_ changed maintain their state. In the example using `Phase` above, the [accumulator](periodic.md#accumulation) of the `Phase` waveform will persist, limiting audible artifacts that might otherwise occur.

Since modification occurs only as a command to the Tracker (or a call to the web component), changes only occur at the beginning of a generation quantum. Setting the buffer size will therefore affect the latency with which modifications go into effect.

## Sliders: faders and encoders

When creating new sounds or programming synthesizer components, it's often helpful to have real-time feedback on the effect of a parameter. For example, the best way to find the cutoff for a low-pass filter is often just to adjust the cutoff and listen to the result. Tuun provides a mechanism called "sliders" that supports this kind of real-time parameter tuning.

### Basic use

A slider is created by calling the `slider` built-in where the value should go. It takes a label, a default value, and optionally a range and a curve:
```
slider(label, default)                    // ranges from 0 to 1
slider(label, default, min, max)          // ranges from min to max
slider(label, default, min, max, curve)   // curve is "linear" or "exp"
```
For example, a low-pass filter whose Q starts at 0.707 and can range between 0.1 and 1.1 would be written as the following.
```
#{level_db=0}
_ = square(220) | rbj.lpf(slider("Q", 0.707, 0.1, 1.1), 2000);
```
The value of a `slider` call is a waveform. The label must be a string written directly in the call (not a variable), and the default must lie between the minimum and maximum. If a program calls a slider with the same label more than once, every call must give the same default, range, and curve; those calls then share a single slider.

In the native app, each program shows the sliders it reaches. If a MIDI controller is detected, sliders are mapped to encoders or faders. If no MIDI controller is detected, the first two sliders are mapped to the x- and y-axis of the mouse position.

In the web component, a control appears for each slider the expression reaches. You can play with this example here:
<div class="container">
  <tuun-synth
    open='["std"]'
    use='["filters.rbj"]'
    expression='square(220) | rbj.lpf(slider("Q", 0.707, 0.1, 1.1), 2000)' />
</div>

### Curves

In the example above, movement of the physical or virtual control is mapped linearly to a point between the minimum and maximum values. For many audio parameters, however, our perceptual response to a physical phenomenon is not linear. For example, our perception of pitch varies with exponential changes to frequency.

To accommodate this, a slider may be given the curve `"exp"`. Equal movements of an `"exp"` slider multiply its value by equal ratios rather than adding equal amounts. (Since its range is divided into ratios, an `"exp"` slider's minimum must be greater than zero.) For example, the following cutoff slider ranges from 100 Hz to 10,000 Hz, and its default of 1,000 Hz sits exactly halfway along it.
```
slider("cutoff", 1000, 100, 10000, "exp")
```

This can then be used in a different version of our low-pass filter example.
<div class="container">
  <tuun-synth
    open='["std"]'
    use='["filters.rbj"]'
    expression='square(220) | rbj.lpf(0.707, slider("cutoff", 1000, 100, 10000, "exp"))' />
</div>

### Sharing sliders between programs

By default, a slider belongs to the program being played even if it's defined in a library function. If a slider is called in the definition of a binding that is used by several programs, each program gets its own slider, and moving one program's slider doesn't change what the others hear.

To share one slider among several programs, list its label in the `sliders` annotation of the binding that should provide it. Every program built through that binding then sees the same slider value, controlled from that binding's program slot.
```
#{sliders=["vibrato"]}
wave = triangle(220 + slider("vibrato", 0, 0, 10) * $5);
#{level_db=0}
_ = wave | rbj.lpf(0.707, 2000); // vibrato is controlled from wave's slot
```
At most one of the bindings a waveform is built through may list a given label; listing it on two of them is an error.

A program shows the sliders in its annotation first (in the order they appear in that annotation), followed by any other sliders it provides. When the file is saved, each slider that has moved from its default is written back to the annotation along with its current value, so it starts there the next time the file is loaded.
```
#{sliders=["Q:0.9"]}
_ = square(220) | rbj.lpf(slider("Q", 0.707, 0.1, 1.1), 2000);
```

Because a program's sliders are shown when it loads, a slider's default, range, and curve must be known before anything plays. For a keys instrument, they can't depend on the key or velocity of a note. To scale a slider by something that varies from note to note, multiply its value instead, as in `freq * slider("ratio", 4, 1, 16, "exp")`.

### Implementation

Each `slider` call evaluates to its default value, wrapped in a mark that names the slider and the program that provides it. For example, if our first low-pass filter example is the first program in the file, its call `slider("Q", 0.707, 0.1, 1.1)` evaluates to the following waveform.
```
Marked(slider(A:1, "Q"), Const(0.707))
```
Before a program (or a note of a keys instrument) starts to play, Tuun substitutes the current value of each slider into its `Marked` waveform.

Changes to the slider are then implemented as `Modify` commands to the `Tracker` or directly as substitutions in the web component. Each command modifies every voice that is playing, so all of the programs and notes that share a slider follow it together. Since commands are only processed once per generation quantum, Tuun attempts to send at most one `Modify` command per slider per quantum. In addition, the updated waveform is constructed as a linear ramp between the old value and the new value. This helps to eliminate artifacts that would occur when parameters make sudden, large changes. For example, if the slider's value is changed to 0.8, the `Modify` command would look something like the following.

```
Modify {
    selector: AllVoices,
    mark_id: slider(A:1, "Q"),
    waveform: Marked(
        slider(A:1, "Q"),
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
    )
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

However, this simple implementation can lead to a "pop" sound if the amplitude of the waveform is large. Instead, "stop" is implemented with a special `Marked` waveform and a short ramp. When `waveform` is played, it is first multiplied by a constant 1.0 that is marked using a mark called `Terminator`.

```
Multiply(
    waveform,
    Marked(
        Terminator,
        Const(1.0),
    ),
)
```

When it's time to stop a waveform, this marked waveform is replaced as follows.

```
Modify {
    mark_id: Terminator
    waveform: Multiply(
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
        Marked(
            Terminator,
            Const(1.0),
        )
    )
}
```

Notice that we keep marked waveform in the result. This ensures that multiple "stop" modifications will never make the waveform louder: subsequent substitutions will be nested inside each other.

## MIDI note-on and note-off

Handling MIDI note-on and note-off events is similar. Each MIDI instrument in Tuun is defined by a function that returns a pair of waveforms:
```
(key, velocity) -> (waveform, waveform) // (note_on, note_off)
```
When a note-on event is received from a controller, the note (or key) number and the velocity (as a value between 0.0 and 1.0) are passed to this function. The first waveform is played immediately. This waveform may be infinite or finite. In either case, when a note-off event is received, the second waveform (`note_off`) is used as part of the substitution for the `Terminator` mark, just as in the case of stopping a waveform above. As in the case of stopping, a simple `Fixed([])` would be sufficient to stop playback, but other waveforms can be used to produce instrument-specific results.

The following example of a MIDI instrument (based on parameters from [Jim Woodhouse's Euphonics site](https://euphonics.org/3-3-marimbas-and-xylophones/)) has a long (but finite) sustain when keys are held.

```
fn(key, vel) =>
  ({map(over(@key), [1.0, 3.92, 9.24, 16.27, 24.22, 33.54, 42.97])}
      * vel
      | ADSR(0.3, 0, 0.1, 0.3, 0.2),
   Rw(0.5, 1.0))
```
(Unfortunately, the Tuun web component doesn't yet support MIDI!)

<script type="module" src="tuun/tuun-synth.js"></script>
