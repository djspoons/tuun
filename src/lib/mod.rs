pub mod builtins;
pub mod eval;
pub mod expr;
pub mod generator;
pub mod infer;
pub mod metric;
pub mod modules;
pub mod optimizer;
pub mod parser;
pub mod signatures;
pub mod slider;
pub mod types;
pub mod waveform;

// Native-only modules (SDL2-dependent)
#[cfg(feature = "native")]
pub mod actions;
#[cfg(feature = "native")]
pub mod diagnostics;
#[cfg(feature = "native")]
pub mod effects;
#[cfg(feature = "native")]
pub mod evaluator;
#[cfg(feature = "native")]
pub mod ids;
#[cfg(feature = "native")]
pub mod keys;
#[cfg(feature = "native")]
pub mod launchkey;
#[cfg(feature = "native")]
pub mod midi_input;
#[cfg(feature = "native")]
pub mod player;
#[cfg(feature = "native")]
pub mod programs;
#[cfg(feature = "native")]
pub mod renderer;
#[cfg(feature = "native")]
pub mod sdl2_input;
#[cfg(feature = "native")]
pub mod sequencer;
#[cfg(feature = "native")]
pub mod tracker;

// WASM module (only compiled when targeting wasm32 or when wasm feature is enabled for testing)
#[cfg(any(target_arch = "wasm32", feature = "wasm"))]
pub mod wasm;
