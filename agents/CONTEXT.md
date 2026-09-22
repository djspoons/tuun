# Tuun

Tuun is a live instrument built around a small functional language: you write expressions that describe sound, then play them from a computer keyboard or a Launchkey controller. This glossary fixes the vocabulary of that domain.

## Language

### Sound

**Waveform**:
The abstract description of a sound, independent of sample rate — a tree, not audio data.
_Avoid_: signal, buffer, audio

**Samples**:
The concrete audio data a waveform is rendered into.
_Avoid_: waveform, sound

**Binding**:
A named definition at the top level of a file. Later bindings in the same file may refer to it; it may carry annotations, and one that does also occupies a program slot.
_Avoid_: variable, assignment, declaration

**Program**:
An expression that evaluates to a waveform, occupying one numbered slot. The unit a performer selects, edits, and plays.
_Avoid_: patch, clip, track, preset, sound

**Voice**:
One sounding instance of a program's waveform. A single program may have several voices sounding at once. A sequenceable program's voice is one hit of the waveform its pattern is built from, not the pattern — launching the pattern plays a voice per listed step.
_Avoid_: instance, playback, note

**Step**:
One element of a sequenceable program's `on_beats` list. Its value is the beat of the measure it plays on, so `on_beats(w, [1, 2.5])` has two steps, not two beats.
_Avoid_: beat, element, hit, note

**Mark**:
A labelled point inside a waveform that can later be substituted — how a sound is stopped, and how a live slider value reaches it.
_Avoid_: tag, label, handle

**Slider**:
A single value, declared on a binding, that can be changed while sound is playing. Every voice whose waveform uses that binding shares one synchronized view of it. The name the slider binds is visible only inside its declaring binding; other programs reach the slider by referencing that binding, not by naming the slider.
_Avoid_: parameter, knob, control

**Capture**:
The recording of a sounding waveform to a WAV file as it plays.
_Avoid_: record, export, bounce

### The program set

**Bank**:
A group of eight program slots. Eight banks, A through H, hold 64 programs.
_Avoid_: page, group, set

**Bank-relative address**:
How a program is named to the performer when it has no binding name: the bank letter and the 1-based slot within that bank, such as `B:3`. It matches the digit the performer types to select it.
_Avoid_: index, program number

### The control surface

**Pad**:
One of the Launchkey's sixteen velocity-sensitive squares, arranged in a top row and a bottom row of eight.
_Avoid_: button, key

**Button**:
A physical Launchkey control that is neither a pad nor a piano key — transport, navigation, `>`.
_Avoid_: pad

**Key**:
One of the Launchkey's piano keys.
_Avoid_: note, pad

**Pending playback**:
Playback scheduled to begin at a future moment in time (often a measure boundary), not yet sounding.
_Avoid_: queued, armed, scheduled, cued

**Keys instrument**:
The single program bound to the piano keys, sounding one voice per held key.
_Avoid_: synth, patch, instrument

### Modes

Five unrelated things here are called a "mode". Always qualify which.

**Device DAW mode**:
The protocol state the Launchkey is placed in so that it speaks the DAW protocol on its DAW ports.
_Avoid_: DAW mode

**Pad mode**:
The pad layout the device reports itself to be in — DAW, or some other layout we do not own.
_Avoid_: DAW mode

**DAW pad sub-mode**:
What the sixteen pads currently do: Clip Launcher, Keys Installer, or Sequencer. Ours, not the device's.
_Avoid_: DAW mode, pad mode

**Editor mode**:
What the computer keyboard currently does: Select, Edit, Move Sliders, or Keys.
_Avoid_: mode

**Launch mode**:
How a top-pad press relates to playback — Toggle or Trigger. A setting of the Clip Launcher that is independent of which program is selected.
_Avoid_: sub-mode, drum mode

**Toggle**:
The launch mode in which a top pad starts its program when silent and stops it when sounding. The default.
_Avoid_: start/stop

**Trigger**:
The launch mode in which a top pad always starts a new voice and never stops one, so that repeated presses layer.
_Avoid_: one-shot, drum mode, always-play, retrigger
