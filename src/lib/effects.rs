//! Executes `Effect`s emitted by the reducer in `actions.rs`.
//!
//! This module owns the concrete I/O handles (MPSC senders, the Launchkey
//! controller). Keeping it separate from `actions.rs` means the reducer
//! stays pure and easy to test.

use std::sync::mpsc;
use std::time::Instant;

use std::collections::HashMap;

use crate::actions::{self, AppState, Effect};
use crate::diagnostics;
use crate::evaluator;
use crate::expr;
use crate::ids::{MarkId, WaveformId, WaveformSelector};
use crate::keys::Keys;
use crate::launchkey;
use crate::player;
use crate::programs::{self, PROGRAMS_PER_BANK};
use crate::slider;
use crate::tracker;

/// Events for the slider worker thread, which coalesces them per audio
/// quantum into tracker `Modify` ramps.
pub enum SliderEvent {
    UpdateSlider {
        selector: WaveformSelector,
        slider: String,
        value: f32,
    },
    SetInitialValues(HashMap<(WaveformSelector, String), f32>),
    UpdateInitialValues(HashMap<(WaveformSelector, String), f32>),
}

/// Converts one coalesced batch of slider targets into tracker ramp commands,
/// ramping each from its last flushed value.
///
/// A selector/slider pair seen for the first time is seeded at the incoming
/// target, producing a flat ramp.
pub fn flush_slider_updates(
    pending: &mut HashMap<(WaveformSelector, String), f32>,
    last_slider_values: &mut HashMap<(WaveformSelector, String), f32>,
    ramp_duration_secs: f32,
) -> Vec<tracker::Command<WaveformId, MarkId>> {
    let mut commands = Vec::with_capacity(pending.len());
    for ((selector, slider), value) in pending.drain() {
        let last_value = last_slider_values
            .entry((selector.clone(), slider.clone()))
            .or_insert(value);
        let waveform = slider::make_ramp(*last_value, value, ramp_duration_secs);
        *last_value = value;
        commands.push(tracker::Command::Modify {
            selector,
            mark_id: MarkId::Slider(slider),
            waveform,
        });
    }
    commands
}

/// External handles the runner needs but doesn't own.
pub struct World<'a> {
    pub launchkey: Option<&'a mut launchkey::Launchkey>,
    pub status: &'a tracker::Status<WaveformId, MarkId>,
}

pub struct EffectRunner {
    player: player::Player,
    evaluator: evaluator::Evaluator,
    slider_sender: mpsc::Sender<SliderEvent>,
}

impl EffectRunner {
    pub fn new(
        player: player::Player,
        evaluator: evaluator::Evaluator,
        slider_sender: mpsc::Sender<SliderEvent>,
    ) -> Self {
        Self {
            player,
            evaluator,
            slider_sender,
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
            evaluator: &self.evaluator,
            source_stale: state.programs.disk_changed(),
        };
        let mut all_effects = Vec::new();
        for action in actions {
            all_effects.extend(actions::apply(state, &ctx, action));
        }
        //println!("  -> effects = {:?}", all_effects);
        self.run_all(state, world, all_effects);
    }

    fn run_one(&mut self, state: &mut AppState, world: &mut World, effect: Effect) {
        match effect {
            Effect::PlayProgram {
                program_index,
                start_at_next_measure,
                repeat_after_measures,
            } => {
                // A sequenceable program always plays decomposed — one
                // tracker entry per beat — so the sequencer can edit its
                // steps live.
                let sequenced = state
                    .programs
                    .program(program_index)
                    .is_some_and(|p| p.sequence().is_some());
                let message = if sequenced {
                    self.player.play_program_steps(
                        &state.programs,
                        program_index,
                        world.status,
                        start_at_next_measure,
                        repeat_after_measures,
                    )
                } else {
                    self.player.play_program(
                        &state.programs,
                        program_index,
                        world.status,
                        start_at_next_measure,
                        repeat_after_measures,
                    )
                };
                if let Some(message) = message {
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
                // pending TopLevel mark (a monolithic program's queued copy,
                // or a sequence's anchor). Steps of the current cycle still
                // play out, matching how a monolithic program's active
                // waveform finishes its measure.
                let selector = WaveformSelector::ProgramVoices(program_index);
                if let Some(after) = world.status.next_pending_mark_start(
                    Instant::now(),
                    &selector,
                    &MarkId::TopLevel,
                ) {
                    self.player.remove_pending(selector, Some(after));
                }
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
                mode_on_success,
                mode_on_failure,
            } => {
                match state
                    .programs
                    .evaluate_and_record(&self.evaluator, program_index)
                {
                    Ok(()) => {
                        if let Some(mode) = mode_on_success {
                            state.mode = mode;
                        }
                    }
                    Err(diagnostics) => {
                        state.message = diagnostics::error_message(&diagnostics);
                        // Hand the diagnostics to the editor so evaluation
                        // errors highlight like parse errors do — whether the
                        // Edit mode comes from `mode_on_failure` or is the
                        // mode being kept.
                        match mode_on_failure {
                            Some(mut mode) => {
                                if let actions::Mode::Edit { errors, .. } = &mut mode {
                                    *errors = diagnostics;
                                }
                                state.mode = mode;
                            }
                            None => {
                                if let actions::Mode::Edit { errors, .. } = &mut state.mode {
                                    *errors = diagnostics;
                                }
                            }
                        }
                    }
                }
            }

            Effect::UpdateSource(program_index) => {
                if let Err(message) = state.programs.splice(program_index) {
                    state.message = message;
                }
            }

            Effect::ReloadSource => {
                // Validate before silencing: a file that doesn't read or
                // parse leaves playback and state untouched.
                let (new_set, warning) = match state.programs.reload_from_disk() {
                    Err(message) => {
                        state.message = message;
                        return;
                    }
                    Ok(reloaded) => reloaded,
                };
                self.player.stop_waveform(WaveformSelector::AllVoices);
                self.player
                    .remove_pending(WaveformSelector::AllVoices, None);
                // The installed instrument was a snapshot of the old set.
                state.keys = None;
                state.programs = new_set;
                let mut failed: Vec<String> = Vec::new();
                for i in 0..state.programs.programs().len() {
                    if state.programs.programs()[i].is_empty() {
                        continue;
                    }
                    if state
                        .programs
                        .evaluate_and_record(&self.evaluator, i)
                        .is_err()
                    {
                        failed.push(state.programs.display_name(i));
                    }
                }
                let mut message = format!("Reloaded {}", state.programs.display_path());
                if !warning.is_empty() {
                    message = format!("{} — {}", message, warning);
                }
                if !failed.is_empty() {
                    message = format!("{} — failed: {}", message, failed.join(", "));
                }
                state.message = message;
            }

            Effect::InstallKeys(program_index) => {
                let program = &state.programs.programs()[program_index];
                if let Some(expr) = program.keys_instrument() {
                    let new_keys = Keys {
                        id: program_index,
                        function: expr.clone(),
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
                match self
                    .evaluator
                    .apply_note_function(&keys.function, args, program.sliders())
                {
                    Ok((mut note_on, note_off)) => {
                        keys.note_off_waveforms.insert(key, note_off);
                        // `note_on`'s `Marked(Slider(_))` nodes still hold the
                        // values from when the instrument was originally
                        // evaluated, so swap in the program's current runtime
                        // values and at the same time collect the (label,
                        // value) pairs we need to seed the slider worker's key
                        // family (so its next ramp has a sensible "previous
                        // value" to start from). All sounding keys hold the
                        // same slider values — every note-on bakes the current
                        // ones in, and later moves ramp the whole family — so
                        // one family entry per slider is enough.
                        let last_slider_values: HashMap<(WaveformSelector, String), f32> =
                            player::substitute_current_slider_values(
                                &mut note_on,
                                program.sliders(),
                            )
                            .into_iter()
                            .map(|(label, value)| ((WaveformSelector::AllKeys, label), value))
                            .collect();
                        let _ = self
                            .slider_sender
                            .send(SliderEvent::UpdateInitialValues(last_slider_values));
                        self.player.play_note(key, note_on, program.level_db());
                    }
                    Err(error) => {
                        let diagnostic = self.evaluator.diagnose(&error, &state.programs, keys.id);
                        state.message = diagnostics::error_message(&[diagnostic]);
                    }
                }
            }
            Effect::PlayNoteOff { key } => {
                let id = WaveformId::Key(key);
                if let Some(keys) = state.keys.as_mut()
                    && let Some(mut note_off) = keys.note_off_waveforms.remove(&key)
                {
                    // The stored note-off's `Marked(Slider(_))` nodes still
                    // hold the values baked in when the instrument was
                    // evaluated: `PlayNoteOn` substitutes fresh values into the
                    // note-on only, and slider moves made while the note was
                    // held reach the tracker's active waveform, not this map.
                    // Swap in the program's current values so the release
                    // doesn't snap back to stale ones.
                    if let Some(program) = state.programs.program(keys.id) {
                        player::substitute_current_slider_values(&mut note_off, program.sliders());
                    }
                    self.player
                        .modify(WaveformSelector::Only(id), MarkId::Terminator, note_off);
                    return;
                }
                // No stored note-off (key wasn't NoteOn'd, or keys were
                // uninstalled mid-note). Send a generic stop ramp; it's a
                // no-op if there's no matching waveform on the tracker.
                self.player.stop_waveform(WaveformSelector::Only(id));
            }

            Effect::UpdateSlider {
                selector,
                slider,
                value,
            } => {
                let _ = self.slider_sender.send(SliderEvent::UpdateSlider {
                    selector,
                    slider,
                    value,
                });
            }

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

            Effect::DumpActiveWaveform => {
                let program_index = state.active_program_index;
                let display_name = state.programs.display_name(program_index);
                let program = state.active_program();
                if let Some(waveform) = program.waveform() {
                    println!("Waveform definition for program {}:", display_name);
                    println!("{:#?}", waveform);
                    state.message = "Dumped waveform to console".to_string();
                } else {
                    println!("No waveform associated with program {}:", display_name);
                    state.message = "No waveform associated with current program".to_string();
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
    use crate::waveform;

    use super::*;

    fn empty_status() -> tracker::Status<WaveformId, MarkId> {
        tracker::Status {
            buffer_start: Instant::now(),
            marks: vec![],
            buffer: None,
            tracker_load: None,
            allocations_per_sample: None,
        }
    }

    /// Walks `waveform` and collects the `Const` value under every
    /// `Marked(Slider(label), …)` node.
    fn slider_mark_values(waveform: &waveform::Waveform<MarkId>, found: &mut Vec<(String, f32)>) {
        use waveform::Waveform;
        match waveform {
            Waveform::Marked { id, waveform } => {
                if let MarkId::Slider(label) = id
                    && let Waveform::Const(v) = **waveform
                {
                    found.push((label.clone(), v));
                }
                slider_mark_values(waveform, found);
            }
            Waveform::BinaryPointOp(_, a, b) => {
                slider_mark_values(a, found);
                slider_mark_values(b, found);
            }
            Waveform::Fin { length, waveform } => {
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
        let (precompute_sender, _precompute_receiver) = mpsc::channel();
        let (fast_sender, fast_receiver) = mpsc::channel();
        let (slider_sender, _slider_receiver) = mpsc::channel();
        let player = player::Player::new(90, 4, precompute_sender, fast_sender);
        let evaluator = evaluator::Evaluator::new(44100, 90, std::path::PathBuf::new());
        let mut runner = EffectRunner::new(player, evaluator, slider_sender);

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
                mode_on_success: None,
                mode_on_failure: None,
            },
        );
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
        while let Ok(command) = fast_receiver.try_recv() {
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
        assert_eq!(marks, vec![("vol".to_string(), 1.0)]);
    }

    #[test]
    fn reload_source_stops_all_and_swaps_programs() {
        let (precompute_sender, _precompute_receiver) = mpsc::channel();
        let (fast_sender, fast_receiver) = mpsc::channel();
        let (slider_sender, _slider_receiver) = mpsc::channel();
        let player = player::Player::new(90, 4, precompute_sender, fast_sender);
        let evaluator = evaluator::Evaluator::new(44100, 90, std::path::PathBuf::new());
        let mut runner = EffectRunner::new(player, evaluator, slider_sender);

        let path =
            std::env::temp_dir().join(format!("tuun_reload_runner_{}.tuun", std::process::id()));
        std::fs::write(&path, "#{level_db=0}\ntone = 1 | fin(time - 1);\n").unwrap();
        let mut state =
            AppState::from_source(std::fs::read_to_string(&path).unwrap(), path.clone())
                .expect("test source should parse");
        state.keys = Some(Keys {
            id: 0,
            function: expr::SourceExpr::float(0.0),
            note_off_waveforms: HashMap::new(),
        });

        // The reload picks up an external rename.
        std::fs::write(&path, "#{level_db=0}\nrenamed = 1 | fin(time - 1);\n").unwrap();
        let status = empty_status();
        let mut world = World {
            launchkey: None,
            status: &status,
        };
        runner.run_one(&mut state, &mut world, Effect::ReloadSource);
        std::fs::remove_file(&path).ok();

        let commands: Vec<_> = fast_receiver.try_iter().collect();
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
    fn reload_source_read_failure_leaves_everything_untouched() {
        let (precompute_sender, _precompute_receiver) = mpsc::channel();
        let (fast_sender, fast_receiver) = mpsc::channel();
        let (slider_sender, _slider_receiver) = mpsc::channel();
        let player = player::Player::new(90, 4, precompute_sender, fast_sender);
        let evaluator = evaluator::Evaluator::new(44100, 90, std::path::PathBuf::new());
        let mut runner = EffectRunner::new(player, evaluator, slider_sender);

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

        assert!(fast_receiver.try_iter().next().is_none());
        assert_eq!(state.programs.display_name(0), "A:1 (tone)");
        assert!(!state.message.is_empty());
    }

    #[test]
    fn remove_pending_program_cancels_from_the_next_cycle() {
        let (precompute_sender, _precompute_receiver) = mpsc::channel();
        let (fast_sender, fast_receiver) = mpsc::channel();
        let (slider_sender, _slider_receiver) = mpsc::channel();
        let player = player::Player::new(90, 4, precompute_sender, fast_sender);
        let evaluator = evaluator::Evaluator::new(44100, 90, std::path::PathBuf::new());
        let mut runner = EffectRunner::new(player, evaluator, slider_sender);

        let mut state = AppState::from_source(
            "#{level_db=0}\n_ = 1 | fin(time - 1);\n".to_string(),
            std::path::PathBuf::new(),
        )
        .expect("test source should parse");

        // With a pending TopLevel mark (here a sequence anchor's next
        // cycle), remove-pending is bounded at that cycle boundary so the
        // current cycle's steps play out.
        let next_cycle = Instant::now() + std::time::Duration::from_secs(1);
        let mut status = empty_status();
        status.marks.push(tracker::Mark {
            waveform_id: WaveformId::Step {
                program: 0,
                sixteenth: player::ANCHOR_SIXTEENTH,
            },
            mark_id: MarkId::TopLevel,
            start: next_cycle,
            duration: std::time::Duration::ZERO,
        });
        let mut world = World {
            launchkey: None,
            status: &status,
        };
        runner.run_one(&mut state, &mut world, Effect::RemovePendingProgram(0));
        let command = fast_receiver.try_recv().expect("expected a command");
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

        // Without any pending TopLevel mark there's nothing to cancel.
        let status = empty_status();
        let mut world = World {
            launchkey: None,
            status: &status,
        };
        runner.run_one(&mut state, &mut world, Effect::RemovePendingProgram(0));
        assert!(fast_receiver.try_recv().is_err());
    }

    #[test]
    fn flush_seeds_unknown_family_from_target_without_panicking() {
        let key = (WaveformSelector::ProgramVoices(0), "cutoff".to_string());
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
        assert_eq!(*selector, WaveformSelector::ProgramVoices(0));
        assert_eq!(*mark_id, MarkId::Slider("cutoff".to_string()));
        // Seeded from the target, so the ramp is flat.
        assert_eq!(
            format!("{}", waveform),
            format!("{}", slider::make_ramp::<MarkId>(0.7, 0.7, 0.02))
        );
    }

    #[test]
    fn flush_ramps_from_last_flushed_value() {
        let key = (WaveformSelector::AllKeys, "vol".to_string());
        let mut last_slider_values = HashMap::from([(key.clone(), 0.2)]);

        let mut pending = HashMap::from([(key.clone(), 0.8)]);
        let commands = flush_slider_updates(&mut pending, &mut last_slider_values, 0.02);
        let tracker::Command::Modify { waveform, .. } = &commands[0] else {
            panic!("expected a Modify command");
        };
        assert_eq!(
            format!("{}", waveform),
            format!("{}", slider::make_ramp::<MarkId>(0.2, 0.8, 0.02))
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
            format!("{}", slider::make_ramp::<MarkId>(0.8, 0.5, 0.02))
        );
        assert_eq!(last_slider_values.get(&key), Some(&0.5));
    }

    #[test]
    fn note_on_seeds_all_keys_family_baseline() {
        let (precompute_sender, _precompute_receiver) = mpsc::channel();
        let (fast_sender, _fast_receiver) = mpsc::channel();
        let (slider_sender, slider_receiver) = mpsc::channel();
        let player = player::Player::new(90, 4, precompute_sender, fast_sender);
        let evaluator = evaluator::Evaluator::new(44100, 90, std::path::PathBuf::new());
        let mut runner = EffectRunner::new(player, evaluator, slider_sender);

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
                mode_on_success: None,
                mode_on_failure: None,
            },
        );
        runner.run_one(&mut state, &mut world, Effect::InstallKeys(0));
        runner.run_one(
            &mut state,
            &mut world,
            Effect::PlayNoteOn {
                key: 60,
                velocity: 127,
            },
        );

        let mut seeded = None;
        while let Ok(event) = slider_receiver.try_recv() {
            if let SliderEvent::UpdateInitialValues(values) = event {
                seeded = Some(values);
            }
        }
        let seeded = seeded.expect("expected an UpdateInitialValues event");
        assert_eq!(
            seeded.get(&(WaveformSelector::AllKeys, "vol".to_string())),
            Some(&0.5)
        );
    }
}
