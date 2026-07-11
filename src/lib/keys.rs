//! The installed keys instrument: a program set up to respond to MIDI
//! note-on/-off events.

use std::collections::HashMap;

use crate::diagnostics::SourceId;
use crate::expr;
use crate::ids::MarkId;
use crate::waveform;

/// A program installed to respond to MIDI note-on/-off events.
pub struct Keys {
    /// 0-based index into the program set of the program installed as the
    /// keys instrument. The instrument's sliders and level are read live
    /// from that program.
    pub id: usize,
    /// A function which takes a pair (MIDI note, velocity) and returns a pair
    /// of Waveforms to be used for note-on and note-off events.
    ///
    /// A snapshot from install time — editing the program's text afterwards
    /// does not change the installed instrument. Should be a closed value
    /// except for references to sliders.
    pub function: expr::SourceExpr<MarkId, SourceId>,
    /// The note-off waveform captured at note-on time for each held key
    /// (keys are MIDI note numbers).
    pub note_off_waveforms: HashMap<u8, waveform::Waveform<MarkId>>,
}
