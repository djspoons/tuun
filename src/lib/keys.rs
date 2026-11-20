//! The installed keys instrument: a program set up to respond to MIDI
//! note-on/-off events.

use std::collections::HashMap;
use std::sync::Arc;

use crate::ids::MarkId;
use crate::programs::KeysInstrument;
use crate::waveform;

/// A program installed to respond to MIDI note-on/-off events.
pub struct Keys {
    /// 0-based index into the program set of the program installed as the
    /// keys instrument. The instrument's sliders and level are read live
    /// from that program.
    pub id: usize,
    /// The instrument, as the program last evaluated before install —
    /// editing the program's text afterwards does not change it.
    pub instrument: KeysInstrument,
    /// The program's text that `instrument` was evaluated from, which the
    /// byte ranges of its notes' marks index into.
    pub text: Arc<str>,
    /// The note-off waveform captured at note-on time for each held key
    /// (keys are MIDI note numbers).
    pub note_off_waveforms: HashMap<u8, waveform::Waveform<MarkId>>,
}
