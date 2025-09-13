# Tuun

Tuun is an interactive sound and music generation system based inspired by programming language design.

Tuun serves two primary purposes:

 1. As a vehicle to help me (@djspoons) learn about sound and music and learn Rust.
 2. As a system for interactively exploring how sounds and music are created.
 3. As a system for performing and recording music.

## Installation

 * Install Rust
 * Install sdl2 and sdl2_ttf

On my Mac this means:

```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
brew install sdl2 sdl2_ttf
export LIBRARY_PATH="$LIBRARY_PATH:$(brew --prefix)/lib"
```

## Getting started

For help running Tuun:
```
cargo run -- --help
```

Tuun reads specifications both from files and in the user interface. Files supplied with the `--context_file` (or `-C`) flag should be bindings of the form `var = expr` separated by commas. Files supplied with the `--program_file` (or `-P`) flag should be expressions (one per line) that evaluate to waveforms. Specifications supplied the `--program` (or `-p`) should also be expressions that evaluate to waveforms. All of these flags can be supplied multiple times. 

```
cargo run -- -C context.tuun -p '$440 * Qw'
```

Or for a slightly more complex example:

```
cargo run -- -C context.tuun -p 'let h = harmonica(Q, 440) in <[h, h, h, h]>'
```

Once Tuun has started, use the following keys to navigate and edit.

In "select" mode (when a solid triangle appears at the left-hand side):
* enter - switch to "edit" mode for the current program
* cmd + enter - evaluate current program and play the resulting waveform at the beginning of the next measure and every measure afterward
* shift + cmd + enter - evaluate current program and play the resulting waveform at the beginning of the next measure and every _other_ measure afterward
* escape - stop playback of future iterations of the current waveform
* cmd + escape - immediate stop playback of the current waveform
* R - **reload** all context files
* L - **load** all program files (and overwrite the current programs)
* S - **save** all programs to a file
* D - evaluate the current program and **dump** the waveform to stdout
* number - select the program with the given number
* down - select the next program
* up - select the previous program

In "edit" mode (when the current program is rendered in white):
* enter - evaluate the current program and play the resulting waveform at the beginning of the next measure
* cmd + enter - evaluate current program and play the resulting waveform at the beginning of the next measure and every measure afterward
* shift + cmd + enter - evaluate current program and play the resulting waveform at the beginning of the next measure and every _other_ measure afterward
* escape - switch to "select" mode