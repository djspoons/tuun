# Tuun

Tuun is an interactive sound and music generation system. It's based on the idea of using *programming languages* to specify sounds and music.

Tuun serves several purposes, roughly in this order:

 1. As a vehicle to help me (@djspoons) learn about sound and music and Rust.
 2. As a system for interactively exploring how sounds and music are created.
 3. As a system for performing and recording music.

Tuun can be built as both a native app and for the web using WebAssembly.

## Documentation

Visit the [Tuun Documentation](https://djspoons.github.io/tuun/) for more information about Tuun waveforms and expressions. It includes many interactive examples and an online playground.

If you're looking for more examples, some of [my writing](https://djspoons.github.io/essays/) also makes use of Tuun.

## Native App

### Installation

 * Install Rust
 * Install sdl2 and sdl2_ttf

On my Mac this means:

```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
brew install sdl2 sdl2_ttf
export LIBRARY_PATH="$LIBRARY_PATH:$(brew --prefix)/lib"
```

### Getting started

For help running the native Tuun application:
```
cargo run --bin tuun -- --help
```

Tuun reads and writes programs from `.tuun` files, which contain lists of bindings. Bindings with annotations will be visible in the user interface.
```
echo 'open std;\n#{level_db=0}\n_ = $440 * Qw;' > test.tuun
cargo run --bin tuun -- test.tuun
```

Or for a slightly more complex example:

```
echo 'open std;\n#{level_db=0,sliders=["X:0.5:0:1","Y:0.5:0:1"]}\n_ = pulse(X, Y * 440) * 0.5;' > test.tuun
cargo run --bin tuun -- test.tuun
```
In this case, try holding the option key, and moving your mouse around.


### Keyboard Navigation

Once Tuun has started, use the following keys to navigate and edit.

In "select" mode (when a solid triangle appears at the left-hand side):
* Enter - switch to "edit" mode for the current program
* Cmd + Enter - play the current program's waveform at the beginning of the next measure and every measure afterward
* Shift + Cmd + Enter - play the current program's waveform at the beginning of the next measure and every _other_ measure afterward
* Escape - stop playback of future iterations of the current waveform
* Cmd + Escape - immediately stop playback of the current waveform
* (hold) Option - switch to "slider" mode
* K - enter **keys** mode
* Shift + D - evaluate the current program and **dump** the result to stdout
* Shift + K - install the current program as the keys instrument
* 1 to 8 - select the program with the given number
* Down - select the next program
* Up - select the previous program
* Right - select the next program bank
* Left - select the previous program bank
* Cmd + R - stop playback and **reload** all source files

In "edit" mode (when the current program is rendered in white):
* Enter - play the current program's waveform at the beginning of the next measure
* Cmd + Enter - play the current program's waveform at the beginning of the next measure and every measure afterward
* Shift + Cmd + Enter - play the current program's waveform at the beginning of the next measure and every _other_ measure afterward
* Cntl + H - display the inferred type of the identifier at the cursor
* Escape - switch to "select" mode

In "slider" mode (slider marks at top and left turn green):
* Move mouse (or track-pad) left and right - adjust "X" slider
* Move mouse (or track-pad) up and down - adjust "Y" slider
* (release) Option - return to "select" mode

In "keys" mode:
* Bottom two rows of keys (Z to /, S to ;) behave as the keys on MIDI keyboard
* Escape - return to "select" mode

### MIDI Integration

Tuun provides a limited MIDI integration, specifically for the Novation Launchkey keyboard controller.

* Keys play the currently installed keys instrument

* Track navigation buttons select the previous or next program; with "Shift" they select the previous or next program bank

* Encoders support the following two modes. (Use "Shift" to change modes.)
  * Plugin mode: encoders control the first eight sliders of the current program
  * Mixer mode: encoders control the levels of the eight programs in the current program bank

* Pads only support DAW mode; there are three DAW "sub-modes." (Use "Shift" + "DAW" to rotate through them.) In the first two sub-modes, the eight columns of pads map to the eight programs in the current program bank. 
  * "Clip launcher" mode
    * Top row pads play the given waveform immediately
      * "Scene" (`>`) button to the right toggles between "Toggle" (start or stop playback) and "Trigger" (always start playback)
        * For sequenced waveforms, "Trigger" mode only plays one step instead of the whole sequence
    * Bottom row pads play the given waveform at the start of the next measure and optionally repeat every measure or every _other_ measure afterward
      * "Function" button to the right switches through different repeating behaviors.
  * "Keys installer" mode: bottom row pads install the given program as a keys instrument (top row pads do nothing)
  * Sequencer mode: for sequenced programs (for example, those that use `on_beats`), each pad maps to a sixteenth note and controls playback of the waveform at that point in time
    * Arrow buttons to the left of the pads scroll from beats 1-4 to beats 5-8 and so on

* "Play" button starts the current waveform at the beginning of the next measure with the same repeating behavior as the bottom pads in "clip launcher" mode

## WebAssembly

Tuun can run in web browsers via WebAssembly! This allows you to experiment with the synthesizer without installing native dependencies.

### Building for Web

```bash
# Install wasm-pack (only needed once)
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Build the WASM package
./build-wasm.sh

# Serve locally
cd web && python3 -m http.server 8080
```

Then open http://localhost:8080 in your browser.

### Web Features

The web version provides:
- Real-time audio synthesis using the Web Audio API
- Interactive expression editor
- Adjustable sample rate and duration
- Expression parser and optimizer

**Note:** The web version focuses on the core synthesis engine. The native version provides additional features like the interactive UI with sliders, file I/O, and real-time waveform visualization.

### Browser Compatibility

The web version requires:
- Chrome 66+ or Edge 79+
- Firefox 60+
- Safari 11.1+

All browsers must support:
- WebAssembly
- Web Audio API
- ES6 modules
