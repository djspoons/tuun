//! Executes `Effect`s emitted by the reducer in `actions.rs`.
//!
//! This module owns the concrete I/O handles (MPSC senders, the Launchkey
//! controller). Keeping it separate from `actions.rs` means the reducer
//! stays pure and easy to test.

use std::mem;
use std::sync::mpsc;
use std::time::Instant;

use std::collections::HashMap;

use crate::actions::{self, AppState, Effect, PendingJob};
use crate::diagnostics;
use crate::environment;
use crate::expr;
use crate::ids::{MarkId, WaveformId, WaveformSelector};
use crate::keys::Keys;
use crate::launchkey;
use crate::player;
use crate::processing;
use crate::programs::{self, PROGRAMS_PER_BANK};
use crate::slider;
use crate::tracker;

/// Events for the slider worker thread, which coalesces them per audio
/// quantum into tracker `Modify` ramps.
pub enum SliderEvent {
    UpdateSlider {
        mark: MarkId,
        value: f32,
    },
    /// Resets the "last" values for all sliders to the given set. If used when
    /// waveforms are playing will have unpredictable audible effects.
    SetBaselines(HashMap<MarkId, f32>),
}

/// Returns the current value of every slider in `set`, keyed by its mark.
pub fn slider_values(set: &programs::ProgramSet) -> HashMap<MarkId, f32> {
    let mut values = HashMap::new();
    for (program_index, program) in set.programs().iter().enumerate() {
        let sliders = program.sliders();
        for (config, &normalized) in sliders.configs().iter().zip(sliders.normalized_values()) {
            values.insert(
                MarkId::Slider {
                    program: program_index,
                    label: config.label.clone(),
                },
                slider::denormalize(&config.function, normalized).unwrap_or(0.0),
            );
        }
    }
    values
}

/// Converts one coalesced batch of slider targets into tracker ramp commands,
/// ramping each from its last flushed value.
pub fn flush_slider_updates(
    pending: &mut HashMap<MarkId, f32>,
    last_slider_values: &mut HashMap<MarkId, f32>,
    ramp_duration_secs: f32,
) -> Vec<tracker::Command<WaveformId, MarkId>> {
    let mut commands = Vec::with_capacity(pending.len());
    for (mark_id, value) in pending.drain() {
        let last_value = last_slider_values.entry(mark_id.clone()).or_insert(value);
        let waveform = slider::make_ramp(mark_id.clone(), *last_value, value, ramp_duration_secs);
        *last_value = value;
        commands.push(tracker::Command::Modify {
            selector: WaveformSelector::AllVoices,
            mark_id,
            waveform,
        });
    }
    commands
}

/// External handles the runner needs but doesn't own.
pub struct World<'a> {
    pub launchkey: Option<&'a mut launchkey::Launchkey>,
    pub status: &'a tracker::Status<WaveformId>,
}

pub struct EffectRunner {
    player: player::Player,
    environment: environment::Environment,
    slider_sender: mpsc::Sender<SliderEvent>,
    /// Where `Effect::EvaluateProgram` sends its jobs; their replies come back
    /// through [`EffectRunner::finish`].
    jobs: mpsc::Sender<processing::Job>,
}

impl EffectRunner {
    pub fn new(
        player: player::Player,
        environment: environment::Environment,
        slider_sender: mpsc::Sender<SliderEvent>,
        jobs: mpsc::Sender<processing::Job>,
    ) -> Self {
        Self {
            player,
            environment,
            slider_sender,
            jobs,
        }
    }

    /// Runs every effect against `state` and `world` in order.
    pub fn run_all(&mut self, state: &mut AppState, world: &mut World, effects: Vec<Effect>) {
        for effect in effects {
            self.run_one(state, world, effect);
        }
    }

    /// Runs the full "actions → effects" cycle for one event: applies each
    /// action through the reducer (collecting effects) and then runs all
    /// resulting effects in order.
    pub fn dispatch(
        &mut self,
        state: &mut AppState,
        world: &mut World,
        actions: Vec<actions::Action>,
    ) {
        //println!("dispatch: actions = {:?}", actions);
        let ctx = actions::Context {
            status: world.status,
            now: Instant::now(),
            environment: &self.environment,
            source_stale: state.programs.disk_changed(),
            tempo: self.player.tempo(),
            beats_per_measure: self.player.beats_per_measure(),
        };
        let mut all_effects = Vec::new();
        for action in actions {
            all_effects.extend(actions::apply(state, &ctx, action));
        }
        //println!("  -> effects = {:?}", all_effects);
        self.run_all(state, world, all_effects);
    }

    /// Applies `reply`, the processing thread's answer to the oldest job in
    /// `state.processing`, and runs the effects that waited on it.
    ///
    /// A program's result is recorded, its message shown, and its `on_success`
    /// or `on_failure` effects run, unless the program's text has changed since
    /// the newest job for it was sent; then the result answers text that is
    /// gone and the whole reply is dropped.
    ///
    /// A loaded program set is installed (see [`EffectRunner::load`]).
    pub fn finish(&mut self, state: &mut AppState, world: &mut World, reply: processing::Reply) {
        let outcome = match reply {
            processing::Reply::Program(outcome) => outcome,
            processing::Reply::Loaded(loaded) => {
                match state.processing.pop_front() {
                    Some(PendingJob::Load { startup, select }) => {
                        self.install(state, loaded, startup, select)
                    }
                    _ => println!("Got a load reply with no load waiting for it"),
                }
                return;
            }
        };
        let Some(PendingJob::Program {
            index,
            generation,
            text,
            on_success,
            on_failure,
            message,
        }) = state.processing.pop_front()
        else {
            println!("Got a program reply with no program waiting for it");
            return;
        };
        // A pending load still owns the status line.
        if state.message == actions::PROCESSING_MESSAGE && !state.is_loading() {
            state.message = message;
        }
        // An older job's result still applies while a newer job for the same
        // text is on its way, so that each sequencer toggle schedules its
        // step.
        let newest = state
            .processing
            .iter()
            .rev()
            .find_map(|job| match job {
                PendingJob::Program {
                    index: i,
                    generation,
                    ..
                } if *i == index => Some(*generation),
                _ => None,
            })
            .unwrap_or(generation);
        if state
            .programs
            .program(index)
            .map(programs::Program::generation)
            != Some(newest)
        {
            return;
        }
        match state.programs.record(index, text, outcome) {
            Ok(warnings) => {
                // Warnings are informational: the program plays regardless, and
                // only the status line reports them.
                if !warnings.is_empty() {
                    state.message = diagnostics::error_message(&warnings);
                }
                self.run_all(state, world, on_success);
            }
            Err(diagnostics) => {
                state.message = diagnostics::error_message(&diagnostics);
                self.run_all(state, world, on_failure);
                // Hand the diagnostics to the editor, in whichever mode
                // `on_failure` left, so evaluation errors highlight.
                if let actions::Mode::Edit { errors, .. } = &mut state.mode {
                    *errors = diagnostics;
                }
            }
        }
    }

    /// Sends `source`, the contents of the program set's backing file, to the
    /// processing thread to be built into a new program set, which
    /// [`EffectRunner::finish`] installs.
    ///
    /// Playback and the current set are untouched until the reply. Installing
    /// stops every voice; a `source` that doesn't parse leaves everything as it
    /// is and shows the error. `startup` marks the load at startup, whose
    /// status message names only the failed programs. Installing selects slot
    /// `select`, if given.
    pub fn load(
        &mut self,
        state: &mut AppState,
        source: String,
        startup: bool,
        select: Option<usize>,
    ) {
        let job = processing::Job::Load {
            source,
            input_path: state.programs.input_path().to_path_buf(),
        };
        if self.jobs.send(job).is_err() {
            state.message = "The processing thread has stopped".to_string();
            return;
        }
        state.message = actions::PROCESSING_MESSAGE.to_string();
        state
            .processing
            .push_back(PendingJob::Load { startup, select });
    }

    /// Installs `loaded`, the set a load built, in place of the current set,
    /// selecting slot `select` if given, or shows its error.
    fn install(
        &mut self,
        state: &mut AppState,
        loaded: Result<processing::Loaded, String>,
        startup: bool,
        select: Option<usize>,
    ) {
        let processing::Loaded {
            set,
            warning,
            failed,
        } = match loaded {
            Ok(loaded) => loaded,
            Err(message) => {
                state.message = message;
                return;
            }
        };
        // TODO this can cause audible artifacts if a waveform is already in
        // the middle of being stopped. For example, if a note-off has a very
        // long ramp in progress, this stop can momentarily increase the
        // amplitude.
        self.player.stop_waveform(WaveformSelector::AllVoices);
        self.player
            .remove_pending(WaveformSelector::AllVoices, None);
        // The installed instrument was a snapshot of the old set.
        state.keys = None;
        if let Some(index) = select {
            state.active_program_index = index;
        }
        // Keys mode plays the instrument just uninstalled.
        if matches!(state.mode, actions::Mode::Keys) {
            state.mode = actions::Mode::Select;
        }
        state.programs = set;
        // Baselines are re-seeded from the new set rather than kept since a
        // slider mark names a program slot, which the new set may fill with an
        // entirely different binding.
        let _ = self
            .slider_sender
            .send(SliderEvent::SetBaselines(slider_values(&state.programs)));
        let failed: Vec<String> = failed
            .into_iter()
            .map(|i| state.programs.display_name(i))
            .collect();
        state.message = if startup {
            // The parse warning would sit in the status line with nothing
            // saying which program it came from; each error resurfaces when
            // its program is evaluated interactively.
            if failed.is_empty() {
                String::new()
            } else {
                format!("Errors in {}", failed.join(", "))
            }
        } else {
            let mut message = format!("Reloaded {}", state.programs.display_path());
            if !warning.is_empty() {
                message = format!("{} — {}", message, warning);
            }
            if !failed.is_empty() {
                message = format!("{} — failed: {}", message, failed.join(", "));
            }
            message
        };
    }

    fn run_one(&mut self, state: &mut AppState, world: &mut World, effect: Effect) {
        match effect {
            Effect::PlayProgram {
                program_index,
                start,
                repeat_after_measures,
            } => {
                let message = self.player.play_program(
                    &state.programs,
                    program_index,
                    world.status,
                    start,
                    repeat_after_measures,
                    &mut state.waveform_marks,
                );
                if let Some(message) = message
                    && !matches!(start, player::Start::At(_))
                {
                    state.message = message;
                }
            }
            Effect::PlayProgramVoice(program_index) => {
                if let Some(message) = self.player.play_program_voice(
                    &state.programs,
                    program_index,
                    &mut state.waveform_marks,
                ) {
                    state.message = message;
                }
            }
            Effect::PlaySequencerStep {
                program_index,
                sixteenth,
            } => {
                // A no-op when the program lost its sequence (e.g. the
                // evaluation preceding this effect failed).
                if let Some(message) = self.player.play_step(
                    &state.programs,
                    program_index,
                    sixteenth,
                    world.status,
                    state.repeat_after_measures,
                    &mut state.waveform_marks,
                ) {
                    state.message = message;
                }
            }
            Effect::RemovePendingSequencerStep {
                program_index,
                sixteenth,
            } => {
                self.player.remove_pending(
                    WaveformSelector::Only(WaveformId::Step {
                        program: program_index,
                        sixteenth,
                    }),
                    None,
                );
            }
            Effect::RecordSequencerStepMarks(program_index) => {
                player::record_step_marks(
                    &state.programs,
                    program_index,
                    &mut state.waveform_marks,
                );
            }
            Effect::StopProgram(program_index) => {
                if state.programs.program(program_index).is_none() {
                    return;
                }
                self.player
                    .stop_waveform(WaveformSelector::ProgramVoices(program_index));
                // A looping program keeps a pending copy queued for its next
                // cycle; remove it so "stop" doesn't leave the fade ramp
                // re-playing at every repeat.
                self.player
                    .remove_pending(WaveformSelector::ProgramVoices(program_index), None);
            }
            Effect::RemovePendingProgram(program_index) => {
                if state.programs.program(program_index).is_none() {
                    return;
                }
                // Cancel playback from the next cycle boundary on — the
                // next pending playback (a monolithic program's queued copy,
                // or a sequence's anchor). Steps of the current cycle still
                // play out, matching how a monolithic program's active
                // waveform finishes its measure.
                if let Some(after) = world.status.next_pending_start(
                    Instant::now(),
                    &WaveformSelector::ProgramPlayback(program_index),
                ) {
                    self.player.remove_pending(
                        WaveformSelector::ProgramVoices(program_index),
                        Some(after),
                    );
                }
            }
            Effect::SetMetronome(on) => {
                self.player.set_metronome(on);
            }
            Effect::ModifyWaveform {
                selector,
                mark_id,
                waveform,
            } => {
                self.player.modify(selector, mark_id, waveform);
            }

            Effect::EvaluateProgram {
                program_index,
                on_success,
                on_failure,
            } => {
                let Some(program) = state.programs.program(program_index) else {
                    return;
                };
                let generation = program.generation();
                let input = state.programs.evaluation_input(program_index);
                let text = input.text.as_str().into();
                if self.jobs.send(processing::Job::Program(input)).is_err() {
                    state.message = "The processing thread has stopped".to_string();
                    return;
                }
                let message =
                    mem::replace(&mut state.message, actions::PROCESSING_MESSAGE.to_string());
                state.processing.push_back(PendingJob::Program {
                    index: program_index,
                    generation,
                    text,
                    on_success,
                    on_failure,
                    message,
                });
            }

            Effect::UpdateSource(program_index) => {
                // The loaded set replaces the source this would write.
                if state.is_loading() {
                    return;
                }
                if let Err(message) = state.programs.splice(program_index) {
                    state.message = message;
                }
            }

            Effect::MoveProgram {
                program_index,
                skip_slots,
                to,
            } => match state.programs.write_skip_slots(program_index, skip_slots) {
                Ok(source) => self.load(state, source, false, Some(to)),
                Err(message) => state.message = message,
            },

            Effect::ReloadSource => match state.programs.read_backing_file() {
                Ok(source) => self.load(state, source, false, None),
                Err(message) => state.message = message,
            },

            Effect::InstallKeys(program_index) => {
                let program = &state.programs.programs()[program_index];
                if let Some(expr) = program.keys_instrument() {
                    let new_keys = Keys {
                        id: program_index,
                        function: expr.clone(),
                        text: program.evaluated_text().clone(),
                        note_off_waveforms: HashMap::new(),
                    };
                    state.keys = Some(new_keys);
                    state.message = format!(
                        "Installed keys from program {}",
                        state.programs.display_name(program_index)
                    );
                } else {
                    state.message = match program.kind() {
                        programs::ProgramKind::Waveform => {
                            "Not a keys program (annotate it with #{keys})".to_string()
                        }
                        programs::ProgramKind::Keys => {
                            "Keys program hasn't evaluated; fix its errors first".to_string()
                        }
                    };
                }
            }

            Effect::PlayNoteOn { key, velocity } => {
                let Some(keys) = state.keys.as_mut() else {
                    return;
                };
                // The instrument's sliders and level are read live from the
                // source program (only the function is an install-time
                // snapshot).
                let Some(program) = state.programs.program(keys.id) else {
                    return;
                };
                let args = vec![
                    expr::SourceExpr::float(key as f32),
                    // TODO use a marked waveform for velocity so we can implement after-touch
                    expr::SourceExpr::float(velocity as f32 / 127.0),
                ];
                match self.environment.apply_note_function(&keys.function, args) {
                    Ok((mut note_on, note_off, marks)) => {
                        keys.note_off_waveforms.insert(key, note_off);
                        // `note_on`'s `Marked(Slider { .. })` nodes still hold
                        // the values from when the instrument was originally
                        // evaluated, so swap in the current runtime ones.
                        player::substitute_current_slider_values(&mut note_on, &state.programs);
                        let marks = player::PlayedMarks {
                            program: keys.id,
                            text: keys.text.clone(),
                            marks,
                            extra: vec![],
                        };
                        self.player.play_note(
                            key,
                            note_on,
                            program.level_db(),
                            marks,
                            &mut state.waveform_marks,
                        );
                    }
                    Err(error) => {
                        let diagnostic = self.environment.diagnose(
                            &error,
                            program.text(),
                            state.programs.source(),
                        );
                        state.message = diagnostics::error_message(&[diagnostic]);
                    }
                }
            }
            Effect::PlayNoteOff { key } => {
                let id = WaveformId::Key(key);
                if let Some(keys) = state.keys.as_mut()
                    && let Some(mut note_off) = keys.note_off_waveforms.remove(&key)
                {
                    // The stored note-off's `Marked(Slider { .. })` nodes still
                    // hold the values baked in when the instrument was
                    // evaluated: `PlayNoteOn` substitutes fresh values into the
                    // note-on only, and slider moves made while the note was
                    // held reach the tracker's active waveform, not this map.
                    // Swap in the current values so the release doesn't snap
                    // back to stale ones.
                    player::substitute_current_slider_values(&mut note_off, &state.programs);
                    self.player.modify(
                        WaveformSelector::Only(id),
                        MarkId::Terminator,
                        player::terminate(note_off),
                    );
                    return;
                }
                // No stored note-off (key wasn't NoteOn'd, or keys were
                // uninstalled mid-note). Send a generic stop ramp; it's a
                // no-op if there's no matching waveform on the tracker.
                self.player.stop_waveform(WaveformSelector::Only(id));
            }

            Effect::UpdateSlider { mark, value } => {
                let _ = self
                    .slider_sender
                    .send(SliderEvent::UpdateSlider { mark, value });
            }

            Effect::SetMode(mode) => state.mode = mode,
            Effect::ShowMessage(msg) => {
                state.message = msg;
            }

            Effect::SetEncoderDisplay { index, name, value } => {
                if let Some(lk) = world.launchkey.as_deref_mut() {
                    lk.set_encoder_display(index, &name, &value);
                }
            }
            Effect::SyncEncoders => {
                if let Some(lk) = world.launchkey.as_deref_mut() {
                    sync_encoders(state, lk);
                }
            }

            Effect::SetLaunchkeyEncoderMode(new_mode) => {
                if let Some(lk) = world.launchkey.as_deref_mut()
                    && lk.encoder_mode != new_mode
                {
                    lk.encoder_mode = new_mode;
                    // The device resets the relative-output feature
                    // when the user switches encoder modes, so we
                    // have to re-assert it every time.
                    lk.set_encoder_relative_output();
                    sync_encoders(state, lk);
                }
            }
            Effect::SetLaunchkeyPadMode(new_mode) => {
                if let Some(lk) = world.launchkey.as_deref_mut() {
                    lk.pad_mode = new_mode;
                }
            }
            Effect::SetDawModeDisplay(label) => {
                if let Some(lk) = world.launchkey.as_deref_mut() {
                    lk.set_daw_mode_display(&label);
                }
            }

            Effect::PrintEvaluatedProgram(program_index) => {
                let display_name = state.programs.display_name(program_index);
                let evaluated = state.programs.program(program_index).and(
                    self.environment
                        .evaluate_program(&state.programs.evaluation_input(program_index))
                        .ok(),
                );
                match evaluated {
                    Some(programs::Evaluated::KeysInstrument(function)) => {
                        // The lambda itself is noisy to print, so show what it
                        // computes for one sample note instead.

                        // TODO this is sort of a lot of work just for some
                        // mediocre debugging, but printing `function` isn't
                        // great either (either Debug or Display).
                        const SAMPLE_KEY: f32 = 60.0; // middle C
                        const SAMPLE_VELOCITY: f32 = 0.7;
                        let args = vec![
                            expr::SourceExpr::float(SAMPLE_KEY),
                            expr::SourceExpr::float(SAMPLE_VELOCITY),
                        ];
                        match self.environment.apply_note_function(&function, args) {
                            Ok((note_on, note_off, _marks)) => {
                                println!(
                                    "Keys instrument of program {} applied at key {}, velocity {}:",
                                    display_name, SAMPLE_KEY, SAMPLE_VELOCITY
                                );
                                println!("note on: {:#?}", note_on);
                                println!("note off: {:#?}", note_off);
                                state.message =
                                    format!("Printed program {} to console", display_name);
                            }
                            Err(error) => {
                                println!("Keys instrument of program {}:", display_name);
                                println!("{}", function);
                                let diagnostic = self.environment.diagnose(
                                    &error,
                                    state.programs.programs()[program_index].text(),
                                    state.programs.source(),
                                );
                                let message = diagnostics::error_message(&[diagnostic]);
                                println!("Applying it at a sample note failed: {}", message);
                                state.message = message;
                            }
                        }
                    }
                    Some(evaluated) => {
                        println!("Evaluated form of program {}:", display_name);
                        println!("{:#?}", evaluated);
                        state.message = format!("Printed program {} to console", display_name);
                    }
                    None => {
                        println!("Program {} has no evaluated form", display_name);
                        state.message = format!("Program {} has no evaluated form", display_name);
                    }
                }
            }
            Effect::Exit => {
                state.should_exit = true;
            }
        }
    }
}

/// Pushes the current bank/program's encoder state to the controller.
///
/// Called only via `Effect::SyncEncoders`, never on a tick.
fn sync_encoders(state: &AppState, launchkey: &mut launchkey::Launchkey) {
    match launchkey.encoder_mode {
        launchkey::EncoderMode::Plugin => {
            let program = match state.programs.program(state.active_program_index) {
                Some(p) => p,
                None => return,
            };
            // Set the display for all encoders.
            for i in 0..launchkey::NUM_ENCODERS {
                if let Some(value) = program.sliders().normalized_values().get(i as usize) {
                    let config = &program.sliders().configs()[i as usize];
                    let actual_value = slider::denormalize(&config.function, *value).unwrap_or(0.0);
                    launchkey.set_encoder_display(
                        i,
                        &config.label,
                        &programs::format_sig_digits(actual_value, 3).to_string(),
                    );
                } else {
                    launchkey.set_encoder_display(i, "", "");
                }
            }
        }
        launchkey::EncoderMode::Mixer => {
            let bank_start = state.bank_start();
            for i in 0..PROGRAMS_PER_BANK {
                let program = match state.programs.program(bank_start + i) {
                    Some(p) => p,
                    None => continue,
                };
                launchkey.set_encoder_display(
                    i as u8,
                    "level",
                    &programs::format_level_db(program.level_db()),
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::ids::ANCHOR_SIXTEENTH;
    use crate::tracker::Select;
    use crate::waveform;

    use super::*;

    /// Stands in for the processing thread: evaluates each job sent to
    /// `jobs`, including those the replies' effects send, and hands its reply
    /// to `runner`.
    fn process_jobs(
        runner: &mut EffectRunner,
        state: &mut AppState,
        world: &mut World,
        jobs: &mpsc::Receiver<processing::Job>,
    ) {
        let environment = environment::Environment::new(44100, 90, std::path::PathBuf::new());
        while let Ok(job) = jobs.try_recv() {
            runner.finish(state, world, reply_to(&environment, job));
        }
    }

    /// Returns the processing thread's reply to `job`, worked out in
    /// `environment`.
    fn reply_to(environment: &environment::Environment, job: processing::Job) -> processing::Reply {
        match job {
            processing::Job::Program(input) => {
                processing::Reply::Program(processing::process(environment, &input, false))
            }
            processing::Job::Load { source, input_path } => {
                processing::Reply::Loaded(processing::load(environment, source, input_path, false))
            }
        }
    }

    fn empty_status() -> tracker::Status<WaveformId> {
        tracker::Status {
            buffer_start: Instant::now(),
            start_times: Vec::new(),
            tracker_load: None,
            allocations: None,
        }
    }

    /// The mark of the slider labelled `label` declared on the program at
    /// `program`.
    fn slider_mark(program: usize, label: &str) -> MarkId {
        MarkId::Slider {
            program,
            label: label.to_string(),
        }
    }

    /// Walks `waveform` and collects the `Const` value under every
    /// `Marked(Slider { .. }, …)` node, as (mark, value) pairs.
    fn slider_mark_values(waveform: &waveform::Waveform<MarkId>, found: &mut Vec<(MarkId, f32)>) {
        use waveform::Waveform;
        match waveform {
            Waveform::Marked { id, waveform } => {
                if matches!(id, MarkId::Slider { .. })
                    && let Waveform::Const(v) = **waveform
                {
                    found.push((id.clone(), v));
                }
                slider_mark_values(waveform, found);
            }
            Waveform::BinaryOp(_, a, b) => {
                slider_mark_values(a, found);
                slider_mark_values(b, found);
            }
            Waveform::Fin {
                length, waveform, ..
            } => {
                slider_mark_values(length, found);
                slider_mark_values(waveform, found);
            }
            _ => {}
        }
    }

    #[test]
    fn note_off_reflects_slider_value_at_release_time() {
        // Slider moves made after install (and while a note is held) must reach
        // the note-off waveform spliced in at release — not the values baked in
        // when the instrument was evaluated.
        // TODO all of the sender set-up is repeated in many tests
        let (command_sender, command_receiver) = mpsc::channel();
        let (slider_sender, _slider_receiver) = mpsc::channel();
        let player = player::Player::new(90, 4, command_sender);
        let environment = environment::Environment::new(44100, 90, std::path::PathBuf::new());
        let (job_sender, job_receiver) = mpsc::channel();
        let mut runner = EffectRunner::new(player, environment, slider_sender, job_sender);

        let mut state = AppState::from_source(
            "#{sliders=[\"vol:0.5:0:1\"], keys}\nk = fn(note, vel) => (vol, vol);".to_string(),
            std::path::PathBuf::new(),
        )
        .expect("test source should parse");
        let status = empty_status();
        let mut world = World {
            launchkey: None,
            status: &status,
        };
        runner.run_one(
            &mut state,
            &mut world,
            Effect::EvaluateProgram {
                program_index: 0,
                on_success: vec![],
                on_failure: vec![],
            },
        );
        process_jobs(&mut runner, &mut state, &mut world, &job_receiver);
        runner.run_one(&mut state, &mut world, Effect::InstallKeys(0));
        assert!(
            state.keys.is_some(),
            "keys should install: {}",
            state.message
        );

        runner.run_one(
            &mut state,
            &mut world,
            Effect::PlayNoteOn {
                key: 60,
                velocity: 127,
            },
        );

        // Move the slider while the note is held.
        state
            .programs
            .program_mut(0)
            .unwrap()
            .set_slider_normalized(0, 1.0)
            .expect("program has a vol slider");

        runner.run_one(&mut state, &mut world, Effect::PlayNoteOff { key: 60 });

        // The release splice is the Modify command on the note's terminator
        // mark.
        let mut note_off = None;
        while let Ok(command) = command_receiver.try_recv() {
            if let tracker::Command::Modify {
                selector: WaveformSelector::Only(WaveformId::Key(60)),
                mark_id: MarkId::Terminator,
                waveform,
            } = command
            {
                note_off = Some(waveform);
            }
        }
        let note_off = note_off.expect("expected a Terminator Modify command");
        let mut marks = Vec::new();
        slider_mark_values(&note_off, &mut marks);
        assert_eq!(marks, vec![(slider_mark(0, "vol"), 1.0)]);
    }

    /// Returns a runner whose channels the test holds: the tracker's commands
    /// and the processing thread's jobs.
    fn runner_with_jobs() -> (
        EffectRunner,
        mpsc::Receiver<tracker::Command<WaveformId, MarkId>>,
        mpsc::Receiver<processing::Job>,
    ) {
        let (command_sender, command_receiver) = mpsc::channel();
        let (slider_sender, _slider_receiver) = mpsc::channel();
        let (job_sender, job_receiver) = mpsc::channel();
        let player = player::Player::new(90, 4, command_sender);
        let environment = environment::Environment::new(44100, 90, std::path::PathBuf::new());
        let runner = EffectRunner::new(player, environment, slider_sender, job_sender);
        (runner, command_receiver, job_receiver)
    }

    /// Returns a state with one waveform program, `tone`, in slot A:1.
    fn tone_state() -> AppState {
        AppState::from_source(
            "#{level_db=0}\ntone = 1 | fin(time - 1);\n".to_string(),
            std::path::PathBuf::new(),
        )
        .expect("test source should parse")
    }

    /// Leaving Edit mode waits in Edit mode for the result, then applies it
    /// as an immediate evaluation would have.
    #[test]
    fn leaving_edit_mode_applies_when_the_reply_arrives() {
        let (mut runner, _command_receiver, job_receiver) = runner_with_jobs();
        let mut state = tone_state();
        state.mode = actions::Mode::Edit {
            cursor_position: 0,
            errors: vec![],
            completion: None,
        };
        state.message = "before".to_string();
        let status = empty_status();
        let mut world = World {
            launchkey: None,
            status: &status,
        };
        runner.dispatch(
            &mut state,
            &mut world,
            vec![actions::Action::EvaluateAndLeaveEditMode {
                mode_on_failure: actions::Mode::Select,
                play: false,
                repeat_after_measures: None,
            }],
        );
        assert!(matches!(state.mode, actions::Mode::Edit { .. }));
        assert_eq!(state.message, actions::PROCESSING_MESSAGE);
        assert!(state.is_processing(0));

        process_jobs(&mut runner, &mut state, &mut world, &job_receiver);
        assert!(matches!(state.mode, actions::Mode::Select));
        assert_eq!(state.message, "before");
        assert!(state.processing.is_empty());
        assert!(state.programs.programs()[0].waveform().is_some());
    }

    /// A failed evaluation hands its diagnostics to the Edit mode it
    /// returns to.
    #[test]
    fn a_failed_evaluation_reports_into_edit_mode() {
        let (mut runner, _command_receiver, job_receiver) = runner_with_jobs();
        let mut state = tone_state();
        state
            .programs
            .program_mut(0)
            .unwrap()
            .set_text("time | fin(\"a\")".to_string());
        let editing = actions::Mode::Edit {
            cursor_position: 0,
            errors: vec![],
            completion: None,
        };
        state.mode = editing.clone();
        let status = empty_status();
        let mut world = World {
            launchkey: None,
            status: &status,
        };
        runner.run_one(
            &mut state,
            &mut world,
            Effect::EvaluateProgram {
                program_index: 0,
                on_success: vec![Effect::SetMode(actions::Mode::Select)],
                on_failure: vec![Effect::SetMode(editing)],
            },
        );
        process_jobs(&mut runner, &mut state, &mut world, &job_receiver);
        assert!(
            matches!(&state.mode, actions::Mode::Edit { errors, .. } if !errors.is_empty()),
            "got {:?}",
            state.mode
        );
        assert!(
            state.message.contains("expected waveform"),
            "got: {}",
            state.message
        );
    }

    /// The effects that read a result run only once it is recorded.
    #[test]
    fn success_effects_wait_for_the_result() {
        let (mut runner, command_receiver, job_receiver) = runner_with_jobs();
        let mut state = tone_state();
        let status = empty_status();
        let mut world = World {
            launchkey: None,
            status: &status,
        };
        runner.run_one(
            &mut state,
            &mut world,
            Effect::EvaluateProgram {
                program_index: 0,
                on_success: vec![Effect::PlayProgram {
                    program_index: 0,
                    start: player::Start::Now,
                    repeat_after_measures: None,
                }],
                on_failure: vec![],
            },
        );
        assert!(command_receiver.try_recv().is_err());

        process_jobs(&mut runner, &mut state, &mut world, &job_receiver);
        assert!(matches!(
            command_receiver.try_recv(),
            Ok(tracker::Command::Play {
                id: WaveformId::Program(0),
                ..
            })
        ));
    }

    /// A failed evaluation runs `on_failure` and not `on_success`, and its
    /// error stays on the status line.
    #[test]
    fn a_failure_runs_only_its_failure_effects() {
        let (mut runner, _command_receiver, job_receiver) = runner_with_jobs();
        let mut state = AppState::from_source(
            "#{level_db=0}\nw = time | fin(\"a\");\n".to_string(),
            std::path::PathBuf::new(),
        )
        .expect("test source should parse");
        let status = empty_status();
        let mut world = World {
            launchkey: None,
            status: &status,
        };
        runner.run_one(
            &mut state,
            &mut world,
            Effect::EvaluateProgram {
                program_index: 0,
                on_success: vec![Effect::ShowMessage("on_success ran".to_string())],
                on_failure: vec![Effect::SetMode(actions::Mode::MoveSliders)],
            },
        );
        process_jobs(&mut runner, &mut state, &mut world, &job_receiver);
        assert!(matches!(state.mode, actions::Mode::MoveSliders));
        assert!(
            state.message.contains("expected waveform"),
            "got: {}",
            state.message
        );
    }

    /// A sequencer toggle whose evaluation fails leaves the error on the
    /// status line rather than the toggle's message.
    #[test]
    fn a_failed_sequencer_toggle_shows_its_error() {
        let (mut runner, _command_receiver, job_receiver) = runner_with_jobs();
        let mut state = AppState::from_source(
            "on_beats = fn(w) => fn(bs) => w;\n\
             #{level_db=0}\n\
             _ = [1, 2.5] | on_beats(time | fin(\"a\"));\n"
                .to_string(),
            std::path::PathBuf::new(),
        )
        .expect("test source should parse");
        let status = empty_status();
        let mut world = World {
            launchkey: None,
            status: &status,
        };
        runner.dispatch(
            &mut state,
            &mut world,
            vec![actions::Action::ToggleSequencerStep { sixteenth: 8 }],
        );
        assert!(state.is_processing(0), "got: {}", state.message);
        process_jobs(&mut runner, &mut state, &mut world, &job_receiver);
        assert!(
            state.message.contains("expected waveform"),
            "got: {}",
            state.message
        );
    }

    /// Returns the program text each mark of program 0's step at `sixteenth`
    /// in `state.waveform_marks` lights, its literals last, or `None` when the
    /// step has no entry or its entry was lit against other text.
    fn step_lit_texts(state: &AppState, sixteenth: u8) -> Option<Vec<String>> {
        let played = state.waveform_marks.get(&WaveformId::Step {
            program: 0,
            sixteenth,
        })?;
        let text = state.programs.program(0)?.text();
        if *played.text != *text {
            return None;
        }
        let marks = played.marks.marks.iter().chain(&played.extra);
        Some(
            marks
                .map(|m| text[m.text_range.clone()].to_string())
                .collect(),
        )
    }

    /// Toggling a step of a playing sequence re-records the marks of the steps
    /// already playing against the new text, and drops a removed step's.
    #[test]
    fn a_sequencer_toggle_keeps_the_playing_steps_marks_current() {
        let (mut runner, _command_receiver, job_receiver) = runner_with_jobs();
        let mut state = AppState::from_source(
            "on_beats = fn(w) => fn(bs) => w;\n\
             #{level_db=0}\n\
             _ = [1, 2] | on_beats(phase(440, 0) | fin(time - 0.1));\n"
                .to_string(),
            std::path::PathBuf::new(),
        )
        .expect("test source should parse");
        let now = Instant::now();
        let mut status = empty_status();
        let anchor = WaveformId::Step {
            program: 0,
            sixteenth: ANCHOR_SIXTEENTH,
        };
        status.start_times = vec![
            (anchor.clone(), now),
            (anchor, now + std::time::Duration::from_secs(4)),
        ];
        let mut world = World {
            launchkey: None,
            status: &status,
        };
        runner.run_one(
            &mut state,
            &mut world,
            Effect::EvaluateProgram {
                program_index: 0,
                on_success: vec![Effect::PlayProgram {
                    program_index: 0,
                    start: player::Start::Now,
                    repeat_after_measures: Some(1),
                }],
                on_failure: vec![],
            },
        );
        process_jobs(&mut runner, &mut state, &mut world, &job_receiver);
        let before = step_lit_texts(&state, 0).expect("step 0 should have marks");
        assert_eq!(before.last().map(String::as_str), Some("1"));

        runner.dispatch(
            &mut state,
            &mut world,
            vec![actions::Action::ToggleSequencerStep { sixteenth: 8 }],
        );
        process_jobs(&mut runner, &mut state, &mut world, &job_receiver);
        assert_eq!(
            state.programs.program(0).unwrap().text(),
            "[1, 2, 3] | on_beats(phase(440, 0) | fin(time - 0.1))"
        );
        assert_eq!(step_lit_texts(&state, 0), Some(before));
        for (sixteenth, literal) in [(4, "2"), (8, "3")] {
            let lit = step_lit_texts(&state, sixteenth).expect("each step should have marks");
            assert_eq!(lit.last().map(String::as_str), Some(literal));
        }

        runner.dispatch(
            &mut state,
            &mut world,
            vec![actions::Action::ToggleSequencerStep { sixteenth: 4 }],
        );
        process_jobs(&mut runner, &mut state, &mut world, &job_receiver);
        assert!(!state.waveform_marks.contains_key(&WaveformId::Step {
            program: 0,
            sixteenth: 4,
        }));
        let lit = step_lit_texts(&state, 8).expect("step 8 should have marks");
        assert_eq!(lit.last().map(String::as_str), Some("3"));
    }

    /// A result whose text was replaced after its job was sent is dropped,
    /// effects and all.
    #[test]
    fn a_result_for_replaced_text_is_dropped() {
        let (mut runner, _command_receiver, job_receiver) = runner_with_jobs();
        let mut state = tone_state();
        state.message = "before".to_string();
        let status = empty_status();
        let mut world = World {
            launchkey: None,
            status: &status,
        };
        runner.run_one(
            &mut state,
            &mut world,
            Effect::EvaluateProgram {
                program_index: 0,
                on_success: vec![Effect::ShowMessage("on_success ran".to_string())],
                on_failure: vec![],
            },
        );
        state
            .programs
            .program_mut(0)
            .unwrap()
            .set_text("2 | fin(time - 1)".to_string());

        process_jobs(&mut runner, &mut state, &mut world, &job_receiver);
        assert!(state.processing.is_empty());
        assert!(state.programs.programs()[0].evaluated().is_none());
        assert_eq!(state.message, "before");
    }

    /// An edit keeps the program playing its last result until the edit's
    /// own result is recorded.
    #[test]
    fn the_last_result_plays_until_the_next_is_recorded() {
        let (mut runner, command_receiver, job_receiver) = runner_with_jobs();
        let mut state = tone_state();
        let status = empty_status();
        let mut world = World {
            launchkey: None,
            status: &status,
        };
        let evaluate = || Effect::EvaluateProgram {
            program_index: 0,
            on_success: vec![],
            on_failure: vec![],
        };
        let play = || Effect::PlayProgram {
            program_index: 0,
            start: player::Start::Now,
            repeat_after_measures: None,
        };
        let played = |receiver: &mpsc::Receiver<tracker::Command<WaveformId, MarkId>>| {
            receiver
                .try_iter()
                .find_map(|command| match command {
                    tracker::Command::Play { waveform, .. } => Some(format!("{}", waveform)),
                    _ => None,
                })
                .expect("expected a Play")
        };
        runner.run_one(&mut state, &mut world, evaluate());
        process_jobs(&mut runner, &mut state, &mut world, &job_receiver);
        runner.run_one(&mut state, &mut world, play());
        let first = played(&command_receiver);

        state
            .programs
            .program_mut(0)
            .unwrap()
            .set_text("2 | fin(time - 1)".to_string());
        runner.run_one(&mut state, &mut world, evaluate());
        runner.run_one(&mut state, &mut world, play());
        assert_eq!(played(&command_receiver), first);

        process_jobs(&mut runner, &mut state, &mut world, &job_receiver);
        runner.run_one(&mut state, &mut world, play());
        assert_ne!(played(&command_receiver), first);
    }

    /// While a newer job for the current text is on its way, an older job's
    /// result still applies, so each of two quick edits runs its effects.
    #[test]
    fn an_older_result_applies_while_a_newer_job_is_pending() {
        let (mut runner, _command_receiver, job_receiver) = runner_with_jobs();
        let mut state = tone_state();
        let status = empty_status();
        let mut world = World {
            launchkey: None,
            status: &status,
        };
        for (text, message) in [
            ("2 | fin(time - 1)", "first"),
            ("3 | fin(time - 1)", "second"),
        ] {
            state
                .programs
                .program_mut(0)
                .unwrap()
                .set_text(text.to_string());
            runner.run_one(
                &mut state,
                &mut world,
                Effect::EvaluateProgram {
                    program_index: 0,
                    on_success: vec![Effect::ShowMessage(message.to_string())],
                    on_failure: vec![],
                },
            );
        }

        let environment = environment::Environment::new(44100, 90, std::path::PathBuf::new());
        for message in ["first", "second"] {
            let Ok(processing::Job::Program(input)) = job_receiver.try_recv() else {
                panic!("expected a job");
            };
            let outcome = processing::process(&environment, &input, false);
            runner.finish(&mut state, &mut world, processing::Reply::Program(outcome));
            assert_eq!(state.message, message);
            assert!(state.programs.programs()[0].waveform().is_some());
        }
        assert!(state.processing.is_empty());
    }

    #[test]
    fn reload_source_stops_all_and_swaps_programs() {
        let (command_sender, command_receiver) = mpsc::channel();
        let (slider_sender, slider_receiver) = mpsc::channel();
        let player = player::Player::new(90, 4, command_sender);
        let environment = environment::Environment::new(44100, 90, std::path::PathBuf::new());
        let (job_sender, job_receiver) = mpsc::channel();
        let mut runner = EffectRunner::new(player, environment, slider_sender, job_sender);

        let path =
            std::env::temp_dir().join(format!("tuun_reload_runner_{}.tuun", std::process::id()));
        std::fs::write(&path, "#{level_db=0}\ntone = 1 | fin(time - 1);\n").unwrap();
        let mut state =
            AppState::from_source(std::fs::read_to_string(&path).unwrap(), path.clone())
                .expect("test source should parse");
        state.keys = Some(Keys {
            id: 0,
            function: expr::SourceExpr::float(0.0),
            text: Arc::default(),
            note_off_waveforms: HashMap::new(),
        });

        // The reload picks up an external rename, and a slider the old file did
        // not have.
        std::fs::write(
            &path,
            "#{level_db=0, sliders=[\"gain:0.25:0:1\"]}\nrenamed = gain | fin(time - 1);\n",
        )
        .unwrap();
        let status = empty_status();
        let mut world = World {
            launchkey: None,
            status: &status,
        };
        runner.run_one(&mut state, &mut world, Effect::ReloadSource);
        std::fs::remove_file(&path).ok();

        // Nothing changes until the set is built.
        assert!(command_receiver.try_recv().is_err());
        assert!(state.keys.is_some());
        assert_eq!(state.programs.display_name(0), "A:1 (tone)");
        assert_eq!(state.message, actions::PROCESSING_MESSAGE);
        assert!(state.is_loading());

        process_jobs(&mut runner, &mut state, &mut world, &job_receiver);
        assert!(!state.is_loading());
        let commands: Vec<_> = command_receiver.try_iter().collect();
        assert!(
            matches!(
                commands[0],
                tracker::Command::Modify {
                    selector: WaveformSelector::AllVoices,
                    mark_id: MarkId::Terminator,
                    ..
                }
            ),
            "expected a stop of all voices first"
        );
        assert!(matches!(
            commands[1],
            tracker::Command::RemovePending {
                selector: WaveformSelector::AllVoices,
                after: None,
            }
        ));
        // The slot a baseline names may now hold a different binding, so
        // the worker is re-seeded from the set that just loaded.
        let Ok(SliderEvent::SetBaselines(baselines)) = slider_receiver.try_recv() else {
            panic!("expected the reload to re-seed the slider baselines");
        };
        assert_eq!(
            baselines,
            HashMap::from([(slider_mark(0, "gain"), 0.25)]),
            "baselines must come from the set that just loaded"
        );
        assert!(state.keys.is_none());
        assert_eq!(state.programs.display_name(0), "A:1 (renamed)");
        // The sweep evaluated the reloaded program.
        assert!(state.programs.programs()[0].waveform().is_some());
        assert!(
            state.message.starts_with("Reloaded"),
            "got: {}",
            state.message
        );
    }

    #[test]
    fn move_program_writes_the_file_and_reloads_with_the_selection() {
        let (command_sender, _command_receiver) = mpsc::channel();
        let (slider_sender, _slider_receiver) = mpsc::channel();
        let player = player::Player::new(90, 4, command_sender);
        let environment = environment::Environment::new(44100, 90, std::path::PathBuf::new());
        let (job_sender, job_receiver) = mpsc::channel();
        let mut runner = EffectRunner::new(player, environment, slider_sender, job_sender);

        let path =
            std::env::temp_dir().join(format!("tuun_move_runner_{}.tuun", std::process::id()));
        std::fs::write(
            &path,
            "#{level_db=0}\na = 1;\n#{skip_slots=1, level_db=0}\nb = 2;\n#{level_db=0}\nc = 3;\n",
        )
        .unwrap();
        let mut state =
            AppState::from_source(std::fs::read_to_string(&path).unwrap(), path.clone())
                .expect("test source should parse");
        state.active_program_index = 2;
        state.mode = actions::Mode::Keys;
        let status = empty_status();
        let mut world = World {
            launchkey: None,
            status: &status,
        };
        runner.run_one(
            &mut state,
            &mut world,
            Effect::MoveProgram {
                program_index: 2,
                skip_slots: 0,
                to: 1,
            },
        );
        let written = std::fs::read_to_string(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(
            written,
            "#{level_db=0}\na = 1;\n#{level_db=0}\nb = 2;\n#{level_db=0}\nc = 3;\n"
        );
        assert!(state.is_loading());
        // The selection waits for the moved set.
        assert_eq!(state.active_program_index, 2);
        process_jobs(&mut runner, &mut state, &mut world, &job_receiver);
        // `b` and the program after it both moved up, and the selection
        // followed `b`.
        assert_eq!(state.active_program_index, 1);
        assert_eq!(state.programs.display_name(1), "A:2 (b)");
        assert_eq!(state.programs.display_name(2), "A:3 (c)");
        // The reload uninstalled the keys instrument Keys mode would play.
        assert!(matches!(state.mode, actions::Mode::Select));
    }

    #[test]
    fn print_evaluated_program_applies_a_keys_instrument() {
        let (command_sender, _command_receiver) = mpsc::channel();
        let (slider_sender, _slider_receiver) = mpsc::channel();
        let player = player::Player::new(90, 4, command_sender);
        let environment = environment::Environment::new(44100, 90, std::path::PathBuf::new());
        let (job_sender, job_receiver) = mpsc::channel();
        let mut runner = EffectRunner::new(player, environment, slider_sender, job_sender);

        let mut state = AppState::from_source(
            "#{keys}\nk = fn(note, vel) => (note, vel);".to_string(),
            std::path::PathBuf::new(),
        )
        .expect("test source should parse");
        let status = empty_status();
        let mut world = World {
            launchkey: None,
            status: &status,
        };
        runner.run_one(
            &mut state,
            &mut world,
            Effect::EvaluateProgram {
                program_index: 0,
                on_success: vec![],
                on_failure: vec![],
            },
        );
        process_jobs(&mut runner, &mut state, &mut world, &job_receiver);
        assert!(
            state.programs.programs()[0].keys_instrument().is_some(),
            "program should evaluate to a keys instrument: {}",
            state.message
        );

        runner.run_one(&mut state, &mut world, Effect::PrintEvaluatedProgram(0));
        assert!(
            state.message.starts_with("Printed program"),
            "the sample note application should succeed, got: {}",
            state.message
        );
    }

    #[test]
    fn reload_source_read_failure_leaves_everything_untouched() {
        let (command_sender, command_receiver) = mpsc::channel();
        let (slider_sender, _slider_receiver) = mpsc::channel();
        let player = player::Player::new(90, 4, command_sender);
        let environment = environment::Environment::new(44100, 90, std::path::PathBuf::new());
        let (job_sender, job_receiver) = mpsc::channel();
        let mut runner = EffectRunner::new(player, environment, slider_sender, job_sender);

        // An empty input path can't be re-read: the reload must not touch
        // playback or programs.
        let mut state = AppState::from_source(
            "#{level_db=0}\ntone = 1 | fin(time - 1);\n".to_string(),
            std::path::PathBuf::new(),
        )
        .expect("test source should parse");
        let status = empty_status();
        let mut world = World {
            launchkey: None,
            status: &status,
        };
        runner.run_one(&mut state, &mut world, Effect::ReloadSource);

        assert!(command_receiver.try_iter().next().is_none());
        assert!(job_receiver.try_recv().is_err());
        assert!(!state.is_loading());
        assert_eq!(state.programs.display_name(0), "A:1 (tone)");
        assert!(!state.message.is_empty());
    }

    /// Returns a state over `source`, backed by a fresh temporary file named
    /// for `test` that holds it, and that file's path.
    fn file_backed_state(test: &str, source: &str) -> (AppState, std::path::PathBuf) {
        let path = std::env::temp_dir().join(format!("tuun_{}_{}.tuun", test, std::process::id()));
        std::fs::write(&path, source).unwrap();
        let state = AppState::from_source(source.to_string(), path.clone())
            .expect("test source should parse");
        (state, path)
    }

    /// A source that doesn't parse leaves playback, programs and keys as they
    /// were, and reports the error.
    #[test]
    fn a_failed_load_leaves_state_untouched() {
        let (mut runner, command_receiver, job_receiver) = runner_with_jobs();
        let (mut state, path) =
            file_backed_state("failed_load", "#{level_db=0}\ntone = 1 | fin(time - 1);\n");
        state.keys = Some(Keys {
            id: 0,
            function: expr::SourceExpr::float(0.0),
            text: Arc::default(),
            note_off_waveforms: HashMap::new(),
        });
        std::fs::write(&path, "#{").unwrap();
        let status = empty_status();
        let mut world = World {
            launchkey: None,
            status: &status,
        };
        runner.run_one(&mut state, &mut world, Effect::ReloadSource);
        process_jobs(&mut runner, &mut state, &mut world, &job_receiver);
        std::fs::remove_file(&path).ok();

        assert!(command_receiver.try_recv().is_err());
        assert!(!state.is_loading());
        assert!(state.keys.is_some());
        assert_eq!(state.programs.display_name(0), "A:1 (tone)");
        assert!(
            state.message.starts_with("Parse error"),
            "got: {}",
            state.message
        );
    }

    /// A program job sent before a reload is applied to the set it was sent
    /// from, and the load installs after it.
    #[test]
    fn a_program_job_before_a_reload_applies_to_the_old_set() {
        let (mut runner, command_receiver, job_receiver) = runner_with_jobs();
        let (mut state, path) = file_backed_state(
            "job_before_reload",
            "#{level_db=0}\ntone = 1 | fin(time - 1);\n",
        );
        std::fs::write(&path, "#{level_db=0}\nrenamed = 1 | fin(time - 1);\n").unwrap();
        let status = empty_status();
        let mut world = World {
            launchkey: None,
            status: &status,
        };
        runner.run_all(
            &mut state,
            &mut world,
            vec![
                Effect::EvaluateProgram {
                    program_index: 0,
                    on_success: vec![],
                    on_failure: vec![],
                },
                Effect::ReloadSource,
            ],
        );
        std::fs::remove_file(&path).ok();

        let environment = environment::Environment::new(44100, 90, std::path::PathBuf::new());
        let job = job_receiver.try_recv().expect("the program's job");
        assert!(matches!(job, processing::Job::Program(_)));
        runner.finish(&mut state, &mut world, reply_to(&environment, job));
        assert_eq!(state.programs.display_name(0), "A:1 (tone)");
        assert!(state.programs.programs()[0].waveform().is_some());
        assert!(command_receiver.try_recv().is_err(), "nothing stops yet");
        // The load still owns the status line.
        assert_eq!(state.message, actions::PROCESSING_MESSAGE);
        assert!(state.is_loading());

        let job = job_receiver.try_recv().expect("the load's job");
        assert!(matches!(job, processing::Job::Load { .. }));
        runner.finish(&mut state, &mut world, reply_to(&environment, job));
        assert!(state.processing.is_empty());
        assert_eq!(state.programs.display_name(0), "A:1 (renamed)");
        assert!(
            state.message.starts_with("Reloaded"),
            "got: {}",
            state.message
        );
    }

    /// While a load is pending, writing a program back to the file is skipped:
    /// the loaded set replaces the source it would write.
    #[test]
    fn update_source_is_skipped_while_a_set_loads() {
        let (mut runner, _command_receiver, _job_receiver) = runner_with_jobs();
        let source = "#{level_db=0}\ntone = 1 | fin(time - 1);\n";
        let (mut state, path) = file_backed_state("skip_update", source);
        let status = empty_status();
        let mut world = World {
            launchkey: None,
            status: &status,
        };
        runner.run_one(&mut state, &mut world, Effect::ReloadSource);
        state
            .programs
            .program_mut(0)
            .unwrap()
            .set_text("2 | fin(time - 1)".to_string());
        runner.run_one(&mut state, &mut world, Effect::UpdateSource(0));
        let on_disk = std::fs::read_to_string(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(on_disk, source);
    }

    /// The load at startup reports only the programs that failed.
    #[test]
    fn the_startup_load_names_the_failed_programs() {
        let (mut runner, _command_receiver, job_receiver) = runner_with_jobs();
        let source = "#{level_db=0}\ngood = 1 | fin(time - 1);\n\
                      #{level_db=0}\nbad = time | fin(\"a\");\n";
        let mut state = AppState::from_source(source.to_string(), std::path::PathBuf::new())
            .expect("test source should parse");
        let status = empty_status();
        let mut world = World {
            launchkey: None,
            status: &status,
        };
        runner.load(&mut state, source.to_string(), true, None);
        assert_eq!(state.message, actions::PROCESSING_MESSAGE);
        process_jobs(&mut runner, &mut state, &mut world, &job_receiver);
        assert_eq!(state.message, "Errors in A:2 (bad)");
        assert!(state.programs.programs()[0].waveform().is_some());

        // A clean set clears the status line.
        let clean = "#{level_db=0}\ngood = 1 | fin(time - 1);\n";
        runner.load(&mut state, clean.to_string(), true, None);
        process_jobs(&mut runner, &mut state, &mut world, &job_receiver);
        assert_eq!(state.message, "");
    }

    #[test]
    fn remove_pending_program_cancels_from_the_next_cycle() {
        let (command_sender, command_receiver) = mpsc::channel();
        let (slider_sender, _slider_receiver) = mpsc::channel();
        let player = player::Player::new(90, 4, command_sender);
        let environment = environment::Environment::new(44100, 90, std::path::PathBuf::new());
        let (job_sender, _job_receiver) = mpsc::channel();
        let mut runner = EffectRunner::new(player, environment, slider_sender, job_sender);

        let mut state = AppState::from_source(
            "#{level_db=0}\n_ = 1 | fin(time - 1);\n".to_string(),
            std::path::PathBuf::new(),
        )
        .expect("test source should parse");

        // With a pending playback (here a sequence anchor's next cycle),
        // remove-pending is bounded at that cycle boundary so the current
        // cycle's steps play out, even one still pending.
        let next_cycle = Instant::now() + std::time::Duration::from_secs(1);
        let mut status = empty_status();
        status.start_times.push((
            WaveformId::Step {
                program: 0,
                sixteenth: 4,
            },
            next_cycle - std::time::Duration::from_millis(500),
        ));
        status.start_times.push((
            WaveformId::Step {
                program: 0,
                sixteenth: ANCHOR_SIXTEENTH,
            },
            next_cycle,
        ));
        let mut world = World {
            launchkey: None,
            status: &status,
        };
        runner.run_one(&mut state, &mut world, Effect::RemovePendingProgram(0));
        let command = command_receiver.try_recv().expect("expected a command");
        assert!(
            matches!(
                command,
                tracker::Command::RemovePending {
                    selector: WaveformSelector::ProgramVoices(0),
                    after: Some(after),
                } if after == next_cycle
            ),
            "expected a bounded RemovePending"
        );

        // Without any pending playback there's nothing to cancel.
        let status = empty_status();
        let mut world = World {
            launchkey: None,
            status: &status,
        };
        runner.run_one(&mut state, &mut world, Effect::RemovePendingProgram(0));
        assert!(command_receiver.try_recv().is_err());
    }

    #[test]
    fn flush_seeds_unknown_mark_from_target_without_panicking() {
        let key = slider_mark(0, "cutoff");
        let mut pending = HashMap::from([(key.clone(), 0.7)]);
        let mut last_slider_values = HashMap::new();

        let commands = flush_slider_updates(&mut pending, &mut last_slider_values, 0.02);

        assert!(pending.is_empty());
        assert_eq!(last_slider_values.get(&key), Some(&0.7));
        assert_eq!(commands.len(), 1);
        let tracker::Command::Modify {
            selector,
            mark_id,
            waveform,
        } = &commands[0]
        else {
            panic!("expected a Modify command");
        };
        assert_eq!(*selector, WaveformSelector::AllVoices);
        assert_eq!(*mark_id, key);
        // Seeded from the target, so the ramp is flat.
        assert_eq!(
            format!("{}", waveform),
            format!("{}", slider::make_ramp(key.clone(), 0.7, 0.7, 0.02))
        );
    }

    /// Same-labelled sliders on different programs are different marks, so
    /// they coalesce and ramp independently.
    #[test]
    fn flush_keeps_same_labelled_sliders_of_different_programs_apart() {
        let mut last_slider_values =
            HashMap::from([(slider_mark(0, "freq"), 0.1), (slider_mark(1, "freq"), 0.9)]);
        let mut pending =
            HashMap::from([(slider_mark(0, "freq"), 0.2), (slider_mark(1, "freq"), 0.8)]);

        let commands = flush_slider_updates(&mut pending, &mut last_slider_values, 0.02);

        assert_eq!(commands.len(), 2);
        for command in &commands {
            let tracker::Command::Modify {
                mark_id, waveform, ..
            } = command
            else {
                panic!("expected a Modify command");
            };
            let expected = match mark_id {
                MarkId::Slider { program: 0, .. } => {
                    slider::make_ramp(mark_id.clone(), 0.1, 0.2, 0.02)
                }
                MarkId::Slider { program: 1, .. } => {
                    slider::make_ramp(mark_id.clone(), 0.9, 0.8, 0.02)
                }
                other => panic!("unexpected mark {}", other),
            };
            assert_eq!(format!("{}", waveform), format!("{}", expected));
        }
        assert_eq!(last_slider_values.get(&slider_mark(0, "freq")), Some(&0.2));
        assert_eq!(last_slider_values.get(&slider_mark(1, "freq")), Some(&0.8));
    }

    #[test]
    fn flush_ramps_from_last_flushed_value() {
        let key = slider_mark(0, "vol");
        let mut last_slider_values = HashMap::from([(key.clone(), 0.2)]);

        let mut pending = HashMap::from([(key.clone(), 0.8)]);
        let commands = flush_slider_updates(&mut pending, &mut last_slider_values, 0.02);
        let tracker::Command::Modify { waveform, .. } = &commands[0] else {
            panic!("expected a Modify command");
        };
        assert_eq!(
            format!("{}", waveform),
            format!("{}", slider::make_ramp(key.clone(), 0.2, 0.8, 0.02))
        );
        assert_eq!(last_slider_values.get(&key), Some(&0.8));

        // The next flush chains from the value just flushed.
        let mut pending = HashMap::from([(key.clone(), 0.5)]);
        let commands = flush_slider_updates(&mut pending, &mut last_slider_values, 0.02);
        let tracker::Command::Modify { waveform, .. } = &commands[0] else {
            panic!("expected a Modify command");
        };
        assert_eq!(
            format!("{}", waveform),
            format!("{}", slider::make_ramp(key.clone(), 0.8, 0.5, 0.02))
        );
        assert_eq!(last_slider_values.get(&key), Some(&0.5));
    }

    /// A play from an instant goes straight to the tracker with that start,
    /// repeating per the given measures, and leaves the message alone.
    #[test]
    fn a_play_at_an_instant_goes_to_the_tracker_silently() {
        let (command_sender, command_receiver) = mpsc::channel();
        let (slider_sender, _slider_receiver) = mpsc::channel();
        let player = player::Player::new(60, 4, command_sender);
        let environment = environment::Environment::new(44100, 60, std::path::PathBuf::new());
        let (job_sender, job_receiver) = mpsc::channel();
        let mut runner = EffectRunner::new(player, environment, slider_sender, job_sender);
        let mut state = AppState::from_source(
            "#{level_db=0}\n_ = 1 | fin(time - 1);\n".to_string(),
            std::path::PathBuf::new(),
        )
        .expect("test source should parse");
        state.message = "Recorded 1 note into A:1".to_string();
        let status = empty_status();
        let mut world = World {
            launchkey: None,
            status: &status,
        };
        let at = Instant::now() - std::time::Duration::from_millis(50);
        runner.run_one(
            &mut state,
            &mut world,
            Effect::EvaluateProgram {
                program_index: 0,
                on_success: vec![Effect::PlayProgram {
                    program_index: 0,
                    start: player::Start::At(at),
                    repeat_after_measures: Some(1),
                }],
                on_failure: vec![],
            },
        );
        process_jobs(&mut runner, &mut state, &mut world, &job_receiver);

        let Ok(tracker::Command::Play {
            id: WaveformId::Program(0),
            start,
            repeat_every,
            ..
        }) = command_receiver.try_recv()
        else {
            panic!("expected a Play");
        };
        assert_eq!(start, Some(at));
        assert_eq!(repeat_every, Some(std::time::Duration::from_secs(4)));
        assert_eq!(state.message, "Recorded 1 note into A:1");
    }

    /// A slider declared on one binding reaches a sounding voice of another
    /// program: the consumer embeds the declaring program's mark, and the
    /// move addresses exactly that mark on every voice.
    #[test]
    fn a_slider_move_reaches_a_voice_of_another_program() {
        let (command_sender, command_receiver) = mpsc::channel();
        let (slider_sender, slider_receiver) = mpsc::channel();
        let player = player::Player::new(90, 4, command_sender);
        let environment = environment::Environment::new(44100, 90, std::path::PathBuf::new());
        let (job_sender, job_receiver) = mpsc::channel();
        let mut runner = EffectRunner::new(player, environment, slider_sender, job_sender);

        // A:1 owns the slider; A:2 opts in by referencing A:1's binding.
        let mut state = AppState::from_source(
            "#{sliders=[\"mix:0.5:0:1\"]}\nm = mix;\n#{level_db=0}\ntone = m | fin(time - 1);\n"
                .to_string(),
            std::path::PathBuf::new(),
        )
        .expect("test source should parse");
        let status = empty_status();
        let mut world = World {
            launchkey: None,
            status: &status,
        };
        runner.run_one(
            &mut state,
            &mut world,
            Effect::EvaluateProgram {
                program_index: 1,
                on_success: vec![],
                on_failure: vec![],
            },
        );
        process_jobs(&mut runner, &mut state, &mut world, &job_receiver);
        runner.run_one(
            &mut state,
            &mut world,
            Effect::PlayProgram {
                program_index: 1,
                start: player::Start::Now,
                repeat_after_measures: None,
            },
        );

        // A:2's sounding voice carries A:1's mark.
        let played = command_receiver
            .try_iter()
            .find_map(|command| match command {
                tracker::Command::Play {
                    id: WaveformId::Program(1),
                    waveform,
                    ..
                } => Some(waveform),
                _ => None,
            })
            .expect("expected A:2 to play");
        let mut marks = Vec::new();
        slider_mark_values(&played, &mut marks);
        assert_eq!(marks, vec![(slider_mark(0, "mix"), 0.5)]);

        // Moving A:1's slider addresses that same mark...
        runner.dispatch(
            &mut state,
            &mut world,
            vec![actions::Action::SetSliderNormalized {
                program: 0,
                slider_index: 0,
                normalized: 1.0,
            }],
        );
        let Ok(SliderEvent::UpdateSlider { mark, value }) = slider_receiver.try_recv() else {
            panic!("expected an UpdateSlider event");
        };
        assert_eq!(mark, slider_mark(0, "mix"));
        assert_eq!(value, 1.0);

        // ...and the ramp it flushes to reaches A:2's voice.
        let mut pending = HashMap::from([(mark.clone(), value)]);
        let commands = flush_slider_updates(&mut pending, &mut HashMap::new(), 0.02);
        let [
            tracker::Command::Modify {
                selector, mark_id, ..
            },
        ] = &commands[..]
        else {
            panic!("expected one Modify command, got {:?}", commands.len());
        };
        assert_eq!(*mark_id, mark);
        assert!(selector.matches(&WaveformId::Program(1)));
    }
}
