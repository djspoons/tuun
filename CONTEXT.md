# Tuun

A live-coding synthesizer: a source file of bound programs, each in a slot, played and edited from a window.

## Language

### Programs and source

**Slot**:
One of the 64 numbered positions (8 banks of 8) that a program occupies.

**Annotation**:
The `#{…}` header above a binding in source, holding settings such as `level_db`, `skip_slots`, sliders and `keys`.

**`skip_slots`**:
An annotation giving the number of empty slots before its program, which moves that program and every later one.

**Keys program**:
A program whose annotation includes `keys`, making it a candidate keys instrument.

**Keys instrument**:
The one keys program currently installed to play notes from the keyboard or MIDI.
_Avoid_: using "keys" alone when either the annotation or the installed instrument is meant

### Keyboard input

**Prefix key**:
A modifier key combination (such as Cmd-N) that does nothing by itself and waits for the next key to choose a command.
_Avoid_: chord (reads as a musical chord)
