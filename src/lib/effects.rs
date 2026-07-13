//! Executes `Effect`s emitted by the reducer in `actions.rs`.
//!
//! This module owns the concrete I/O handles (MPSC senders, the Launchkey
//! controller). Keeping it separate from `actions.rs` means the reducer
//! stays pure and easy to test.

use std::sync::mpsc;
use std::time::Instant;

use std::collections::HashMap;

use crate::actions::{self, AppState, Effect};
use crate::evaluator;
use crate::expr;
use crate::ids::{MarkId, WaveformId};
use crate::keys::Keys;
use crate::player;
use crate::programs::{self, PROGRAMS_PER_BANK};
use crate::slider;
use crate::tracker;
use crate::waveform;
use crate::{launchkey, optimizer};

/// Events for the slider worker thread, which coalesces them per audio
/// quantum into tracker `Modify` ramps.
pub enum SliderEvent {
    UpdateSlider {
        id: WaveformId,
        slider: String,
        value: f32,
    },
    SetInitialValues(HashMap<(WaveformId, String), f32>),
    UpdateInitialValues(HashMap<(WaveformId, String), f32>),
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
                if let Some(message) = self.player.play_program(
                    &state.programs,
                    program_index,
                    world.status,
                    start_at_next_measure,
                    repeat_after_measures,
                ) {
                    state.message = message;
                }
            }
            Effect::StopProgram(program_index) => {
                if state.programs.program(program_index).is_none() {
                    return;
                }
                self.player
                    .stop_waveform(WaveformId::Program(program_index));
            }
            Effect::RemovePendingProgram(program_index) => {
                if state.programs.program(program_index).is_none() {
                    return;
                }
                self.player
                    .remove_pending(WaveformId::Program(program_index));
            }
            Effect::ModifyWaveform {
                id,
                mark_id,
                waveform,
            } => {
                self.player.modify(id, mark_id, waveform);
            }

            Effect::EvaluateProgram {
                program_index,
                mut mode_on_failure,
            } => {
                match state
                    .programs
                    .evaluate_and_record(&self.evaluator, program_index)
                {
                    Ok(()) => state.mode = actions::Mode::Select,
                    Err(diagnostics) => {
                        state.message = match diagnostics.len() {
                            1 => format!("Error: {}", diagnostics[0]),
                            n => format!("Error: {} (+{} more)", diagnostics[0], n - 1),
                        };
                        // Hand the diagnostics to the editor so evaluation
                        // errors highlight like parse errors do.
                        if let actions::Mode::Edit { errors, .. } = &mut mode_on_failure {
                            *errors = diagnostics;
                        }
                        state.mode = mode_on_failure;
                    }
                }
            }

            Effect::UpdateSource(program_index) => {
                if let Err(message) = state.programs.splice(program_index) {
                    state.message = message;
                }
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
                    state.message = "Not a valid keys instrument".to_string();
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
                    Ok((note_on, note_off)) => {
                        let mut note_on = optimizer::optimize(note_on);
                        let note_off = optimizer::optimize(note_off);
                        keys.note_off_waveforms.insert(key, note_off);
                        // `note_on`'s `Marked(Slider(_))` nodes still hold the
                        // values from when the instrument was originally
                        // evaluated, so swap in the program's current runtime
                        // values and at the same time collect the (label,
                        // value) pairs we need to seed the slider worker for
                        // this fresh `Key` id (so its next ramp has a sensible
                        // "previous value" to start from).
                        let id = WaveformId::Key(key);
                        let last_slider_values: HashMap<(WaveformId, String), f32> =
                            player::substitute_current_slider_values(
                                &mut note_on,
                                program.sliders(),
                            )
                            .into_iter()
                            .map(|(label, value)| ((id.clone(), label), value))
                            .collect();
                        let _ = self
                            .slider_sender
                            .send(SliderEvent::UpdateInitialValues(last_slider_values));
                        self.player.play_note(key, note_on, program.level_db());
                    }
                    Err(error) => {
                        state.message = format!("Error: {}", error);
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
                    self.player.modify(id, MarkId::Terminator, note_off);
                    return;
                }
                // No stored note-off (key wasn't NoteOn'd, or keys were
                // uninstalled mid-note). Send a generic stop ramp; it's a
                // no-op if there's no matching waveform on the tracker.
                self.player.stop_waveform(id);
            }

            Effect::UpdateSlider { id, slider, value } => {
                let _ = self
                    .slider_sender
                    .send(SliderEvent::UpdateSlider { id, slider, value });
            }
            Effect::UpdateActiveKeySliders { slider, value } => {
                for mark in &world.status.marks {
                    if let WaveformId::Key(_) = mark.waveform_id {
                        let _ = self.slider_sender.send(SliderEvent::UpdateSlider {
                            id: mark.waveform_id.clone(),
                            slider: slider.clone(),
                            value,
                        });
                    }
                }
            }
            Effect::ModifyActiveKeysAmplitude { amplitude } => {
                use waveform::Waveform;
                for mark in &world.status.marks {
                    if let WaveformId::Key(_) = mark.waveform_id {
                        self.player.modify(
                            mark.waveform_id.clone(),
                            MarkId::Amplitude,
                            Waveform::Const(amplitude),
                        );
                    }
                }
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
            "#{sliders=[\"vol:0.5:0:1\"]}\nk = fn(note, vel) => (vol, vol);".to_string(),
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
                mode_on_failure: actions::Mode::Select,
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
                id: WaveformId::Key(60),
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
}
