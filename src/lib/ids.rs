//! Identifier types for waveforms and marks.
//!
//! The tracker, generator, and effect pipeline are generic over these; this
//! module holds the app's concrete instantiation.

use std::fmt;

use crate::tracker;

// TODO: rename Program as Clip? Or make Clip a type of program?
// And the other type is Key(channel, key_number)?

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum WaveformId {
    // Beats are silent waveforms that are used to keep time. The bool tracks whether
    // it is an odd or even measure (false == odd).
    Beats(bool),
    /// A program identified by its 0-based index in the program set.
    Program(usize),
    /// One sequencer step of a program: `slot` addresses a sixteenth within the
    /// program's measure grid.
    Step {
        program: usize,
        slot: u8,
    },
    /// Identifies a waveform playing in response to striking a key on a MIDI keyboard
    /// or equivalent controller.
    Key(u8),
}

impl WaveformId {
    pub fn is_beats(&self) -> bool {
        matches!(self, WaveformId::Beats(_))
    }
}

/// Addresses one waveform or a family of related waveforms in tracker commands
/// and status queries.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum WaveformSelector {
    /// Exactly the waveform with the given id.
    Only(WaveformId),
    /// Every key waveform.
    AllKeys,
    /// A program's whole voice family: its top-level waveform and any of its
    /// step waveforms.
    ProgramVoices(usize),
}

impl tracker::Select<WaveformId> for WaveformSelector {
    fn matches(&self, id: &WaveformId) -> bool {
        match self {
            WaveformSelector::Only(only) => only == id,
            WaveformSelector::AllKeys => matches!(id, WaveformId::Key(_)),
            WaveformSelector::ProgramVoices(p) => matches!(
                id,
                WaveformId::Program(i) | WaveformId::Step { program: i, .. } if i == p
            ),
        }
    }
}

impl tracker::Id for WaveformId {
    type Selector = WaveformSelector;
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

#[cfg(test)]
mod tests {
    use crate::tracker::Select;

    use super::*;

    #[test]
    fn selector_only_matches_exact_id() {
        let selector = WaveformSelector::Only(WaveformId::Key(60));
        assert!(selector.matches(&WaveformId::Key(60)));
        assert!(!selector.matches(&WaveformId::Key(62)));
        assert!(!selector.matches(&WaveformId::Program(0)));
    }

    #[test]
    fn selector_all_keys_matches_every_key_only() {
        let selector = WaveformSelector::AllKeys;
        assert!(selector.matches(&WaveformId::Key(0)));
        assert!(selector.matches(&WaveformId::Key(127)));
        assert!(!selector.matches(&WaveformId::Program(0)));
        assert!(!selector.matches(&WaveformId::Step {
            program: 0,
            slot: 0
        }));
        assert!(!selector.matches(&WaveformId::Beats(false)));
    }

    #[test]
    fn selector_program_voices_matches_program_and_its_steps() {
        let selector = WaveformSelector::ProgramVoices(2);
        assert!(selector.matches(&WaveformId::Program(2)));
        assert!(selector.matches(&WaveformId::Step {
            program: 2,
            slot: 0
        }));
        assert!(selector.matches(&WaveformId::Step {
            program: 2,
            slot: u8::MAX
        }));
        assert!(!selector.matches(&WaveformId::Program(3)));
        assert!(!selector.matches(&WaveformId::Step {
            program: 3,
            slot: 0
        }));
        assert!(!selector.matches(&WaveformId::Beats(true)));
        assert!(!selector.matches(&WaveformId::Key(60)));
    }
}
