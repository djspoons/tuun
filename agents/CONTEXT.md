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

**Keys instrument**:
A function from a key and a velocity to a pair of waveforms: the note-on that sounds while the key is held and the note-off that replaces it on release.
_Avoid_: synth, patch, instrument

**Voice**:
One sounding instance of a program's waveform. A single program may have several voices sounding at once. A sequenceable program's voice is one hit of the waveform its pattern is built from, not the pattern — launching the pattern plays a voice per listed step.
_Avoid_: instance, playback, note

**Step**:
One element of a sequenceable program's `on_beats` list. Its value is the beat of the measure it plays on, so `[1, 2.5] | on_beats(w)` has two steps, not two beats.
_Avoid_: beat, element, hit, note

**Note**:
One element of a phrase: a beat, a key, a velocity and a duration, what one key press and its release produce. Not the voice that sounds it, and not a step.
_Avoid_: event, hit, tuple, voice

**Phrase**:
A complete musical unit of notes, expressed as MIDI parameters and timing: a beat, a key, a velocity and a duration for each. What recording produces; a program plays one by passing it through a keys instrument.
_Avoid_: loop, clip, pattern, sequence, list

**Take**:
The notes gathered by the recorder from arming until the boundary margin before the end boundary. A take is written as a phrase or discarded.
_Avoid_: recording, buffer, phrase

**Record**:
Writing what was played on the keys, as a phrase, into the selected slot.
_Avoid_: capture, sample, loop

**Boundary margin**:
The eighth of a beat on either side of a measure boundary within which the recorder treats what happens as happening on the boundary. A key struck in the margin before a take's start counts as struck on it; a Record or Stop press in the margin after a boundary counts as pressed on it; a take closes one margin before its end boundary, and a key still held then counts as released on the boundary. A recorded note is never shorter than one margin.
_Avoid_: early-hit window, pre-roll, anticipation window, grace period, tolerance

**Mark**:
A labelled point inside a waveform that can later be substituted — how a sound is stopped, and how a live slider value reaches it.
_Avoid_: tag, label, handle

**Slider**:
A single value, declared on a binding, that can be changed while sound is playing. Every voice whose waveform uses that binding shares one synchronized view of it. The name the slider binds is visible only inside its declaring binding; other programs reach the slider by referencing that binding, not by naming the slider.
_Avoid_: parameter, knob, control

**Capture**:
Saving a sounding waveform to a WAV file as it plays. Not Record, which writes notes as text.
_Avoid_: record, export, bounce

### The program set

**Keys program**:
A program annotated `#{keys}` that evaluates to a keys instrument. A file may hold several.
_Avoid_: instrument program, keys patch

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
Playback scheduled to begin at a future moment in time (often a measure boundary), not yet sounding. Not armed, which is the recorder waiting for its start boundary.
_Avoid_: queued, armed, scheduled, cued

**Repeat**:
The app-wide setting for whether a launched program plays again after one measure, after two, or not at all.
_Avoid_: loop, looping, cycle

**Installed keys instrument**:
The one keys instrument that sounds when MIDI note-on and note-off events arrive, one voice per held key. Taken from a keys program; at most one is installed at a time.
_Avoid_: the keys, current keys, active instrument

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
What the computer keyboard currently does: Select, Edit, Move Sliders, Keys, or Record.
_Avoid_: mode

**Owned program**:
The active program while Edit mode or Record mode holds it. Changes that would affect or depend on its text are refused, including changing which program is active and starting or queueing its playback; stopping it, removing its pending playback, uninstalling it as the keys instrument, sliders, level, and other programs' playback stay free.
_Avoid_: locked, busy, frozen

**Launch mode**:
How a top-pad press relates to playback — Toggle or Trigger. A setting of the Clip Launcher that is independent of which program is selected.
_Avoid_: sub-mode, drum mode

**Toggle**:
The launch mode in which a top pad starts its program when silent and stops it when sounding. The default.
_Avoid_: start/stop

**Trigger**:
The launch mode in which a top pad always starts a new voice and never stops one, so that repeated presses layer.
_Avoid_: one-shot, drum mode, always-play, retrigger
