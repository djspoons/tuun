pub mod builtins;
pub mod generator;
pub mod metric;
pub mod optimizer;
pub mod parser;
pub mod slider;
pub mod waveform;

// Native-only modules (SDL2-dependent)
#[cfg(feature = "native")]
pub mod launchkey;
#[cfg(feature = "native")]
pub mod midi_input;
#[cfg(feature = "native")]
pub mod play_helper;
#[cfg(feature = "native")]
pub mod renderer;
#[cfg(feature = "native")]
pub mod sdl2_input;
#[cfg(feature = "native")]
pub mod tracker;

// WASM module (only compiled when targeting wasm32 or when wasm feature is enabled for testing)
#[cfg(any(target_arch = "wasm32", feature = "wasm"))]
pub mod wasm;
