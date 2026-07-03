//! The installed keys instrument: a program set up to respond to MIDI
//! note-on/-off events.

use std::collections::HashMap;

use crate::parser;
use crate::programs::ProgramSliders;
use crate::renderer::MarkId;
use crate::waveform;

/// A program installed to respond to MIDI note-on/-off events.
pub struct Keys {
    /// 0-based index into the program set of the program installed as the
    /// keys instrument.
    pub id: usize,
    /// A function which takes a pair (MIDI note, velocity) and returns a pair
    /// of Waveforms to be used for note-on and note-off events.
    ///
    /// A snapshot from install time — editing the program's text afterwards
    /// does not change the installed instrument. Should be a closed value
    /// except for references to sliders.
    pub function: parser::SourceExpr<MarkId>,
    pub sliders: ProgramSliders,
    pub level_db: f32,
    /// The note-off waveform captured at note-on time for each held key
    /// (keys are MIDI note numbers).
    pub note_off_waveforms: HashMap<u8, waveform::Waveform<MarkId>>,
}
