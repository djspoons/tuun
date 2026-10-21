//! Identifier types for waveforms and marks.
//!
//! The tracker, generator, and effect pipeline are generic over these; this
//! module holds the app's concrete instantiation.

use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::diagnostics::Source;
use crate::expr::Span;
use crate::programs;
use crate::trace::{CarriesTrace, Trace};
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
    /// One sequencer step of a program: `sixteenth` addresses a quarter-beat
    /// within the program's measure grid.
    Step {
        program: usize,
        sixteenth: u8,
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
    /// Every program, step, and key waveform; not the Beats timekeepers.
    AllVoices,
    /// Both Beats timekeepers.
    AllBeats,
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
            WaveformSelector::AllVoices => !id.is_beats(),
            WaveformSelector::AllBeats => id.is_beats(),
        }
    }
}

impl tracker::Id for WaveformId {
    type Selector = WaveformSelector;
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum MarkId {
    TopLevel, // a mark for the whole Program
    /// One slider, qualified by the program slot whose binding declares it.
    Slider {
        program: usize,
        label: String,
    },
    Amplitude,          // use to set top-level amplitude
    Terminator,         // used to stop programs
    MetronomeAmplitude, // used to set the level of the metronome click
    UserDefined(u32),
    /// Records which program text built the waveform it marks.
    Trace(TraceRef),
}

impl fmt::Display for MarkId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MarkId::TopLevel => write!(f, "top-level"),
            MarkId::Slider { program, label } => {
                write!(
                    f,
                    "slider({}, {:?})",
                    programs::bank_address(*program),
                    label
                )
            }
            MarkId::Amplitude => write!(f, "amplitude"),
            MarkId::Terminator => write!(f, "terminator"),
            MarkId::MetronomeAmplitude => write!(f, "metronome"),
            MarkId::UserDefined(id) => write!(f, "{:?}", id),
            MarkId::Trace(_) => write!(f, "trace"),
        }
    }
}

/// A trace as a mark id carries it: equal and hashed by pointer, so two mark
/// ids are the same only if they share one trace.
#[derive(Clone)]
pub struct TraceRef(pub Arc<Trace<Source>>);

impl PartialEq for TraceRef {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for TraceRef {}

impl Hash for TraceRef {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.0).hash(state)
    }
}

impl fmt::Debug for TraceRef {
    // The whole trace is a graph of the value's history; its written place
    // is enough to tell mark ids apart when debugging.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0.span {
            Some(span) => write!(f, "Trace({:?} {:?})", span.source, span.range),
            None => write!(f, "Trace"),
        }
    }
}

impl CarriesTrace<Source> for MarkId {
    fn from_trace(trace: &Arc<Trace<Source>>) -> Option<Self> {
        Some(MarkId::Trace(TraceRef(trace.clone())))
    }

    fn as_trace(&self) -> Option<&Arc<Trace<Source>>> {
        match self {
            MarkId::Trace(trace) => Some(&trace.0),
            _ => None,
        }
    }

    fn in_program(span: &Span<Source>) -> bool {
        span.source == Source::Program
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
            sixteenth: 0
        }));
        assert!(!selector.matches(&WaveformId::Beats(false)));
    }

    #[test]
    fn selector_all_voices_matches_everything_but_beats() {
        let selector = WaveformSelector::AllVoices;
        assert!(selector.matches(&WaveformId::Program(0)));
        assert!(selector.matches(&WaveformId::Step {
            program: 3,
            sixteenth: 7
        }));
        assert!(selector.matches(&WaveformId::Key(60)));
        assert!(!selector.matches(&WaveformId::Beats(false)));
        assert!(!selector.matches(&WaveformId::Beats(true)));
    }

    #[test]
    fn selector_all_beats_matches_only_beats() {
        let selector = WaveformSelector::AllBeats;
        assert!(selector.matches(&WaveformId::Beats(false)));
        assert!(selector.matches(&WaveformId::Beats(true)));
        assert!(!selector.matches(&WaveformId::Program(0)));
        assert!(!selector.matches(&WaveformId::Key(60)));
    }

    #[test]
    fn selector_program_voices_matches_program_and_its_steps() {
        let selector = WaveformSelector::ProgramVoices(2);
        assert!(selector.matches(&WaveformId::Program(2)));
        assert!(selector.matches(&WaveformId::Step {
            program: 2,
            sixteenth: 0
        }));
        assert!(selector.matches(&WaveformId::Step {
            program: 2,
            sixteenth: u8::MAX
        }));
        assert!(!selector.matches(&WaveformId::Program(3)));
        assert!(!selector.matches(&WaveformId::Step {
            program: 3,
            sixteenth: 0
        }));
        assert!(!selector.matches(&WaveformId::Beats(true)));
        assert!(!selector.matches(&WaveformId::Key(60)));
    }
}
