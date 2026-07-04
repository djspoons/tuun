//! Identifier types for waveforms and marks.
//!
//! The tracker, generator, and effect pipeline are generic over these; this
//! module holds the app's concrete instantiation.

use std::fmt;

// TODO: rename Program as Clip? Or make Clip a type of program?
// And the other type is Key(channel, key_number)?

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum WaveformId {
    // Beats are silent waveforms that are used to keep time. The bool tracks whether
    // it is an odd or even measure (false == odd).
    Beats(bool),
    /// A program identified by its 0-based index in the program set.
    Program(usize),
    /// Identifies a waveform playing in response to striking a key on a MIDI keyboard
    /// or equivalent controller.
    Key(u8),
}

impl WaveformId {
    pub fn is_beats(&self) -> bool {
        matches!(self, WaveformId::Beats(_))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum MarkId {
    TopLevel, // a mark for the whole Program
    Slider(String),
    Amplitude,  // use to set top-level amplitude
    Terminator, // used to stop programs
    UserDefined(u32),
    // TODO consider replacing "UserDefined" with cases that better describe the cases
    //VisualizeTiming, // How mark(1) is currently used
    //ShowSample(String), // For debugging filter params, etc.
}

impl fmt::Display for MarkId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MarkId::TopLevel => write!(f, "top-level"),
            MarkId::Slider(name) => write!(f, "slider({:?})", name),
            MarkId::Amplitude => write!(f, "amplitude"),
            MarkId::Terminator => write!(f, "terminator"),
            MarkId::UserDefined(id) => write!(f, "{:?}", id),
        }
    }
}
