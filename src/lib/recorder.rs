//! Recording a MIDI phrase: the take's phases and the recording window.
//!
//! A take runs from the Record press until its notes are written or discarded.
//! Times are `Instant`s and positions are beats, where beat 1 is the take's
//! start boundary.

use std::time::{Duration, Instant};

/// The boundary margin in beats: half a sixteenth.
///
/// Near a boundary, an early key press or a late transport press within this
/// margin counts as on the boundary. The take also closes one margin before its
/// end boundary, and no written note is shorter than one margin.
pub const MARGIN_BEATS: f64 = 0.125;

/// The name of the standard-library helper that plays a phrase.
pub const PHRASE_HELPER: &str = "as_midi_phrase";

/// How a take ends once its notes are written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Finish {
    /// Write the phrase without playing it.
    Silent,
    /// Write the phrase and play it from the end boundary.
    Play,
}

/// Where a take is between the Record press and the close.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// Waiting for the start boundary.
    Armed,
    /// Recording, with no end boundary chosen yet.
    Recording,
    /// The end boundary is chosen; waiting for the close.
    Finishing(Finish),
}

/// The written result of a take that has closed.
#[derive(Debug, Clone, PartialEq)]
pub struct Closed {
    /// How the take ends.
    pub finish: Finish,
    /// The end boundary, where a played phrase starts.
    pub end: Instant,
    /// The number of notes in the phrase.
    pub notes: usize,
    /// The phrase as program text, e.g. `[(1.000, 60, 0.630, 0.412)] | as_midi_phrase(kb)`.
    pub text: String,
}

/// The result of a transport press handed to a take.
#[derive(Debug, Clone, PartialEq)]
pub enum Response {
    /// The take was still armed and is dropped. With `play`, the press is
    /// Play and acts on the active program as it would outside a take.
    Disarmed { play: bool },
    /// The take waits for the close. `previous` is the finish before this
    /// press, or `None` if the take was recording.
    Finishing { previous: Option<Finish> },
    /// The take closed at once, because the close is already due.
    Closed(Closed),
    /// The take was recording and is dropped with this many notes.
    Discarded { notes: usize },
}

/// The result of a boundary tick handed to a take.
///
/// After `Closed`, the take is spent.
#[derive(Debug, Clone, PartialEq)]
pub enum Tick {
    /// Nothing is due yet.
    Waiting,
    /// The start boundary arrived; the take is now recording.
    Started,
    /// The close arrived; the take is done.
    Closed(Closed),
}

/// One key press as received, before the window rules apply.
#[derive(Debug, Clone)]
struct Press {
    key: u8,
    velocity: u8,
    onset: Instant,
    /// When the key was released or retriggered, if it has been.
    release: Option<Instant>,
}

/// One note as written into a phrase.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Note {
    /// The 1-based beat the note starts on.
    pub beat: f64,
    /// The MIDI note number.
    pub key: u8,
    /// The MIDI velocity, 1 to 127.
    pub velocity: u8,
    /// The note's length in beats.
    pub duration: f64,
}

/// The notes in flight from the Record press until they are written or
/// discarded.
///
/// # Example
///
/// ```
/// use std::time::{Duration, Instant};
/// use tuun::recorder::{Finish, Response, Take, Tick};
///
/// let now = Instant::now();
/// let beat = |b: f64| now + Duration::from_secs_f64(b); // 60 bpm
/// let mut take = Take::arm("kb".to_string(), 60, now, beat(1.0));
/// assert_eq!(take.tick(beat(1.0)), Tick::Started);
/// take.note_on(60, 127, beat(1.5));
/// take.note_off(60, beat(2.0));
/// let response = take.finish(Finish::Silent, beat(2.5), beat(5.0));
/// assert_eq!(response, Response::Finishing { previous: None });
/// let Tick::Closed(closed) = take.tick(beat(4.875)) else { panic!() };
/// assert_eq!(closed.text, "[(1.500, 60, 1.000, 0.500)] | as_midi_phrase(kb)");
/// ```
#[derive(Debug, Clone)]
pub struct Take {
    keys_name: String,
    tempo: u32,
    armed_at: Instant,
    start: Instant,
    /// The end boundary, once a finishing press has chosen it.
    end: Option<Instant>,
    phase: Phase,
    /// Every press since arming, in onset order.
    presses: Vec<Press>,
}

impl Take {
    /// Returns a take armed at `now` that starts recording at `start`, a
    /// measure boundary, and plays through the instrument bound to `keys_name`.
    ///
    /// The take is recording at once if `start` is not after `now`. `tempo` is
    /// in beats per minute.
    pub fn arm(keys_name: String, tempo: u32, now: Instant, start: Instant) -> Take {
        Take {
            keys_name,
            tempo,
            armed_at: now,
            start,
            end: None,
            phase: if start <= now {
                Phase::Recording
            } else {
                Phase::Armed
            },
            presses: Vec::new(),
        }
    }

    /// Returns the take's phase.
    pub fn phase(&self) -> Phase {
        self.phase
    }

    /// Returns the name of the keys binding the phrase plays through.
    pub fn keys_name(&self) -> &str {
        &self.keys_name
    }

    /// Returns when the next tick is due: the start boundary while armed,
    /// the close while finishing, and `None` while recording.
    pub fn next_tick(&self) -> Option<Instant> {
        match (self.phase, self.end) {
            (Phase::Armed, _) => Some(self.start),
            (Phase::Finishing(_), Some(end)) => Some(self.close_time(end)),
            _ => None,
        }
    }

    /// Records a key press received at `stamp`.
    ///
    /// A press of a key that is already held releases the held note at
    /// `stamp`. Presses stamped before the take was armed are ignored.
    pub fn note_on(&mut self, key: u8, velocity: u8, stamp: Instant) {
        if stamp < self.armed_at {
            return;
        }
        self.note_off(key, stamp);
        self.presses.push(Press {
            key,
            velocity,
            onset: stamp,
            release: None,
        });
    }

    /// Records a key release received at `stamp`.
    ///
    /// A release of a key with no held note is ignored.
    pub fn note_off(&mut self, key: u8, stamp: Instant) {
        if let Some(press) = self
            .presses
            .iter_mut()
            .find(|p| p.key == key && p.release.is_none())
        {
            press.release = Some(stamp);
        }
    }

    /// Handles a Record (`Finish::Silent`) or Play (`Finish::Play`) press at
    /// `now`, where `boundary` is the press's boundary (see
    /// [`boundary_for_press`]).
    ///
    /// Armed, the take is disarmed. Recording, `boundary` becomes the end
    /// boundary and the take closes at once if the close is due, else waits for
    /// it. Finishing, the press only sets how the take ends; `boundary` is
    /// ignored.
    pub fn finish(&mut self, finish: Finish, now: Instant, boundary: Instant) -> Response {
        match self.phase {
            Phase::Armed => Response::Disarmed {
                play: finish == Finish::Play,
            },
            Phase::Recording => {
                self.end = Some(boundary);
                self.phase = Phase::Finishing(finish);
                if now >= self.close_time(boundary) {
                    Response::Closed(self.close(finish, boundary))
                } else {
                    Response::Finishing { previous: None }
                }
            }
            Phase::Finishing(previous) => {
                self.phase = Phase::Finishing(finish);
                Response::Finishing {
                    previous: Some(previous),
                }
            }
        }
    }

    /// Handles a Stop press: disarms an armed take, and otherwise discards it.
    pub fn stop(&self) -> Response {
        match self.phase {
            Phase::Armed => Response::Disarmed { play: false },
            _ => Response::Discarded {
                notes: self.note_count(),
            },
        }
    }

    /// Advances the take to `now`: starts recording at the start boundary and
    /// closes one margin before the end boundary.
    pub fn tick(&mut self, now: Instant) -> Tick {
        match (self.phase, self.end) {
            (Phase::Armed, _) if now >= self.start => {
                self.phase = Phase::Recording;
                Tick::Started
            }
            (Phase::Finishing(finish), Some(end)) if now >= self.close_time(end) => {
                Tick::Closed(self.close(finish, end))
            }
            _ => Tick::Waiting,
        }
    }

    /// Returns the number of notes in the take so far.
    ///
    /// Before the end boundary is chosen, every press from one margin before
    /// the start boundary counts.
    pub fn note_count(&self) -> usize {
        match self.end {
            Some(end) => self.notes_until(end).len(),
            None => self
                .presses
                .iter()
                .filter(|p| self.beats_between(self.start, p.onset) >= -MARGIN_BEATS)
                .count(),
        }
    }

    /// Returns the take's notes as program text for an end boundary of
    /// `end`.
    ///
    /// Beat, velocity (over 127) and duration are written with three
    /// decimals, the key as an integer.
    pub fn phrase_text(&self, end: Instant) -> String {
        let notes: Vec<String> = self
            .notes_until(end)
            .iter()
            .map(|n| {
                format!(
                    "({:.3}, {}, {:.3}, {:.3})",
                    n.beat,
                    n.key,
                    n.velocity as f64 / 127.0,
                    n.duration
                )
            })
            .collect();
        format!(
            "[{}] | {}({})",
            notes.join(", "),
            PHRASE_HELPER,
            self.keys_name
        )
    }

    /// Returns the notes written for an end boundary of `end`.
    fn notes_until(&self, end: Instant) -> Vec<Note> {
        let take_length = self.beats_between(self.start, end);
        self.presses
            .iter()
            .filter(|p| {
                // In the take from one margin before the start boundary
                // (rule 6) to one margin before the end boundary (rule 8).
                self.beats_between(self.start, p.onset) >= -MARGIN_BEATS
                    && self.beats_between(end, p.onset) < -MARGIN_BEATS
            })
            .map(|p| {
                // Releases from the close onward are not observed; the key
                // is released at the end boundary (rule 9).
                let release = match p.release {
                    Some(r) if self.beats_between(end, r) < -MARGIN_BEATS => r,
                    _ => end,
                };
                let hold = self.beats_between(p.onset, release);
                let beat = (1.0 + self.beats_between(self.start, p.onset)).max(1.0);
                Note {
                    beat,
                    key: p.key,
                    velocity: p.velocity,
                    duration: hold.min(take_length + 1.0 - beat).max(MARGIN_BEATS),
                }
            })
            .collect()
    }

    /// Returns the take's result at the end boundary `end`.
    fn close(&self, finish: Finish, end: Instant) -> Closed {
        Closed {
            finish,
            end,
            notes: self.notes_until(end).len(),
            text: self.phrase_text(end),
        }
    }

    /// Returns when the take closes for an end boundary of `end`.
    fn close_time(&self, end: Instant) -> Instant {
        end.checked_sub(margin(self.tempo)).unwrap_or(end)
    }

    /// Returns the signed number of beats from `from` to `to`.
    fn beats_between(&self, from: Instant, to: Instant) -> f64 {
        let seconds = if to >= from {
            (to - from).as_secs_f64()
        } else {
            -(from - to).as_secs_f64()
        };
        seconds * self.tempo as f64 / 60.0
    }
}

/// Returns the measure boundary a transport press at `now` refers to:
/// `previous` if it passed less than one margin ago, else `next`.
///
/// `previous` is the latest boundary at or before `now`, `next` the first
/// after it. `tempo` is in beats per minute.
///
/// # Example
///
/// ```
/// use std::time::{Duration, Instant};
/// use tuun::recorder::boundary_for_press;
///
/// let boundary = Instant::now();
/// let next = boundary + Duration::from_secs(4);
/// let late = boundary + Duration::from_millis(100); // under 0.125 beat at 60 bpm
/// assert_eq!(boundary_for_press(late, Some(boundary), next, 60), boundary);
/// let later = boundary + Duration::from_millis(200);
/// assert_eq!(boundary_for_press(later, Some(boundary), next, 60), next);
/// ```
pub fn boundary_for_press(
    now: Instant,
    previous: Option<Instant>,
    next: Instant,
    tempo: u32,
) -> Instant {
    match previous {
        Some(p) if p <= now && now - p < margin(tempo) => p,
        _ => next,
    }
}

/// Returns the boundary margin as a duration at `tempo` beats per minute.
fn margin(tempo: u32) -> Duration {
    Duration::from_secs_f64(MARGIN_BEATS * 60.0 / tempo as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// At 60 bpm one beat is one second, so the margin is 125 ms.
    const TEMPO: u32 = 60;

    /// Returns a clock where `at(b)` is beat `b`, with beat 1 at 10 s so
    /// that times before it stay representable.
    fn clock() -> impl Fn(f64) -> Instant {
        let base = Instant::now();
        move |b: f64| base + Duration::from_secs_f64(9.0 + b)
    }

    /// Returns a take armed at beat -3, starting at beat 1, recording.
    fn recording(at: &impl Fn(f64) -> Instant) -> Take {
        let mut take = Take::arm("kb".to_string(), TEMPO, at(-3.0), at(1.0));
        assert_eq!(take.tick(at(1.0)), Tick::Started);
        take
    }

    /// Returns the notes of `take` closed with an end boundary at beat 5.
    fn close_at_5(take: &mut Take, at: &impl Fn(f64) -> Instant) -> Vec<Note> {
        take.finish(Finish::Silent, at(4.0), at(5.0));
        match take.tick(at(4.875)) {
            Tick::Closed(_) => take.notes_until(at(5.0)),
            other => panic!("expected close, got {:?}", other),
        }
    }

    fn assert_note(note: &Note, beat: f64, key: u8, duration: f64) {
        assert!(
            (note.beat - beat).abs() < 1e-6,
            "beat {} != {}",
            note.beat,
            beat
        );
        assert_eq!(note.key, key);
        assert!(
            (note.duration - duration).abs() < 1e-6,
            "duration {} != {}",
            note.duration,
            duration
        );
    }

    // --- phases and ticks ---

    #[test]
    fn arm_waits_for_the_start_boundary() {
        let at = clock();
        let mut take = Take::arm("kb".to_string(), TEMPO, at(0.0), at(1.0));
        assert_eq!(take.phase(), Phase::Armed);
        assert_eq!(take.next_tick(), Some(at(1.0)));
        assert_eq!(take.tick(at(0.9)), Tick::Waiting);
        assert_eq!(take.tick(at(1.0)), Tick::Started);
        assert_eq!(take.phase(), Phase::Recording);
        assert_eq!(take.next_tick(), None);
    }

    #[test]
    fn arm_at_a_passed_boundary_records_at_once() {
        let at = clock();
        let take = Take::arm("kb".to_string(), TEMPO, at(1.05), at(1.0));
        assert_eq!(take.phase(), Phase::Recording);
    }

    #[test]
    fn armed_presses_disarm() {
        let at = clock();
        let mut take = Take::arm("kb".to_string(), TEMPO, at(0.0), at(1.0));
        assert_eq!(take.stop(), Response::Disarmed { play: false });
        assert_eq!(
            take.finish(Finish::Silent, at(0.5), at(1.0)),
            Response::Disarmed { play: false }
        );
        assert_eq!(
            take.finish(Finish::Play, at(0.5), at(1.0)),
            Response::Disarmed { play: true }
        );
    }

    #[test]
    fn take_closes_one_margin_before_the_end_boundary() {
        let at = clock();
        let mut take = recording(&at);
        assert_eq!(
            take.finish(Finish::Play, at(4.0), at(5.0)),
            Response::Finishing { previous: None }
        );
        assert_eq!(take.phase(), Phase::Finishing(Finish::Play));
        assert_eq!(take.next_tick(), Some(at(4.875)));
        assert_eq!(take.tick(at(4.87)), Tick::Waiting);
        let Tick::Closed(closed) = take.tick(at(4.875)) else {
            panic!("expected close")
        };
        assert_eq!(closed.finish, Finish::Play);
        assert_eq!(closed.end, at(5.0));
        assert_eq!(closed.notes, 0);
        assert_eq!(closed.text, "[] | as_midi_phrase(kb)");
    }

    #[test]
    fn last_press_wins_while_finishing() {
        let at = clock();
        let mut take = recording(&at);
        take.finish(Finish::Silent, at(4.0), at(5.0));
        assert_eq!(
            take.finish(Finish::Play, at(4.2), at(9.0)),
            Response::Finishing {
                previous: Some(Finish::Silent)
            }
        );
        // The end boundary stays the one chosen first.
        assert_eq!(take.next_tick(), Some(at(4.875)));
        let Tick::Closed(closed) = take.tick(at(4.9)) else {
            panic!("expected close")
        };
        assert_eq!(closed.finish, Finish::Play);
        assert_eq!(closed.end, at(5.0));
    }

    #[test]
    fn press_within_a_margin_before_the_boundary_closes_at_once() {
        let at = clock();
        let mut take = recording(&at);
        take.note_on(60, 100, at(2.0));
        let Response::Closed(closed) = take.finish(Finish::Silent, at(4.9), at(5.0)) else {
            panic!("expected close")
        };
        assert_eq!(closed.end, at(5.0));
        assert_eq!(closed.notes, 1);
    }

    #[test]
    fn snapped_press_after_the_boundary_closes_at_once() {
        let at = clock();
        let mut take = recording(&at);
        take.note_on(60, 100, at(2.0));
        // A note struck on the downbeat before the late press is not in the take.
        take.note_on(62, 100, at(5.0));
        let end = boundary_for_press(at(5.1), Some(at(5.0)), at(9.0), TEMPO);
        assert_eq!(end, at(5.0));
        let Response::Closed(closed) = take.finish(Finish::Silent, at(5.1), end) else {
            panic!("expected close")
        };
        assert_eq!(
            closed.text,
            "[(2.000, 60, 0.787, 3.000)] | as_midi_phrase(kb)"
        );
    }

    #[test]
    fn stop_discards_with_the_note_count() {
        let at = clock();
        let mut take = recording(&at);
        take.note_on(60, 100, at(1.5));
        take.note_on(62, 100, at(2.5));
        assert_eq!(take.stop(), Response::Discarded { notes: 2 });
        take.finish(Finish::Silent, at(3.0), at(5.0));
        assert_eq!(take.stop(), Response::Discarded { notes: 2 });
    }

    // --- press snapping (rule 5 as amended) ---

    #[test]
    fn press_snaps_back_within_a_margin() {
        let at = clock();
        assert_eq!(
            boundary_for_press(at(1.1), Some(at(1.0)), at(5.0), TEMPO),
            at(1.0)
        );
        assert_eq!(
            boundary_for_press(at(1.0), Some(at(1.0)), at(5.0), TEMPO),
            at(1.0)
        );
        assert_eq!(
            boundary_for_press(at(1.125), Some(at(1.0)), at(5.0), TEMPO),
            at(5.0)
        );
        assert_eq!(boundary_for_press(at(1.1), None, at(5.0), TEMPO), at(5.0));
    }

    // --- window rules ---

    #[test]
    fn early_hit_is_pulled_to_beat_one_keeping_its_hold() {
        // Rule 6.
        let at = clock();
        let mut take = recording(&at);
        take.note_on(60, 100, at(0.9));
        take.note_off(60, at(1.4));
        let notes = close_at_5(&mut take, &at);
        assert_eq!(notes.len(), 1);
        assert_note(&notes[0], 1.0, 60, 0.5);
    }

    #[test]
    fn window_never_reaches_back_before_arming() {
        // Rule 6.
        let at = clock();
        let mut take = Take::arm("kb".to_string(), TEMPO, at(0.95), at(1.0));
        take.note_on(60, 100, at(0.9));
        take.note_off(60, at(1.4));
        take.tick(at(1.0));
        assert!(close_at_5(&mut take, &at).is_empty());
    }

    #[test]
    fn key_struck_before_the_margin_is_not_a_note() {
        // Rule 7: held into the take, and its note-off is ignored.
        let at = clock();
        let mut take = recording(&at);
        take.note_on(60, 100, at(0.8));
        take.note_off(60, at(1.5));
        take.note_off(61, at(1.6));
        assert!(close_at_5(&mut take, &at).is_empty());
    }

    #[test]
    fn note_on_in_the_closing_margin_is_not_a_note() {
        // Rule 8.
        let at = clock();
        let mut take = recording(&at);
        take.note_on(60, 100, at(4.9));
        assert!(close_at_5(&mut take, &at).is_empty());
    }

    #[test]
    fn open_key_at_the_close_is_released_at_the_end_boundary() {
        // Rule 9 as amended: the note-off in [end - margin, end) is not observed.
        let at = clock();
        let mut take = recording(&at);
        take.note_on(60, 100, at(4.0));
        take.note_on(62, 100, at(4.5));
        take.note_off(62, at(4.9));
        let notes = close_at_5(&mut take, &at);
        assert_note(&notes[0], 4.0, 60, 1.0);
        assert_note(&notes[1], 4.5, 62, 0.5);
    }

    #[test]
    fn duration_has_a_floor_of_one_margin() {
        // Rule 10.
        let at = clock();
        let mut take = recording(&at);
        take.note_on(60, 100, at(2.0));
        take.note_off(60, at(2.01));
        assert_note(&close_at_5(&mut take, &at)[0], 2.0, 60, MARGIN_BEATS);
    }

    #[test]
    fn pulled_note_is_clipped_to_the_end_of_the_take() {
        // Rules 9 and 10: an early hit held to the end keeps its hold, which
        // would run past the take's last beat, so it is clipped there.
        let at = clock();
        let mut take = recording(&at);
        take.note_on(60, 100, at(0.9));
        assert_note(&close_at_5(&mut take, &at)[0], 1.0, 60, 4.0);
    }

    #[test]
    fn retrigger_closes_the_open_note() {
        // Rule 11.
        let at = clock();
        let mut take = recording(&at);
        take.note_on(60, 100, at(2.0));
        take.note_on(60, 90, at(2.5));
        take.note_off(60, at(3.0));
        let notes = close_at_5(&mut take, &at);
        assert_eq!(notes.len(), 2);
        assert_note(&notes[0], 2.0, 60, 0.5);
        assert_note(&notes[1], 2.5, 60, 0.5);
        assert_eq!(notes[1].velocity, 90);
    }

    #[test]
    fn phrase_text_has_three_decimals() {
        // Rule 12.
        let at = clock();
        let mut take = recording(&at);
        take.note_on(60, 80, at(1.0));
        take.note_off(60, at(1.412));
        take.note_on(64, 65, at(2.5));
        take.note_off(64, at(2.701));
        close_at_5(&mut take, &at);
        assert_eq!(
            take.phrase_text(at(5.0)),
            "[(1.000, 60, 0.630, 0.412), (2.500, 64, 0.512, 0.201)] | as_midi_phrase(kb)"
        );
    }

    #[test]
    fn beats_follow_the_tempo() {
        // Rule 5: beat = 1 + (t - start) * tempo / 60.
        let at = clock();
        let mut take = Take::arm("kb".to_string(), 120, at(0.0), at(1.0));
        take.tick(at(1.0));
        take.note_on(60, 100, at(1.5));
        take.note_off(60, at(1.75));
        take.finish(Finish::Silent, at(2.0), at(3.0));
        let Tick::Closed(closed) = take.tick(at(3.0)) else {
            panic!("expected close")
        };
        assert_eq!(
            closed.text,
            "[(2.000, 60, 0.787, 0.500)] | as_midi_phrase(kb)"
        );
    }
}
