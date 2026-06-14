//! Executes `Effect`s emitted by the reducer in `actions.rs`.
//!
//! This module owns the concrete I/O handles (MPSC senders, the Launchkey
//! controller). Keeping it separate from `actions.rs` means the reducer
//! stays pure and easy to test.

use std::sync::mpsc;

use std::collections::HashMap;

use crate::actions::{self, AppState, Effect};
use crate::launchkey;
use crate::midi_input::Keys;
use crate::parser;
use crate::play_helper;
use crate::renderer::{self, MarkId, PROGRAMS_PER_BANK, WaveformId};
use crate::slider;
use crate::tracker;
use crate::waveform;

/// Applies a note function `expr` to the given `args`, expecting a pair
/// of Waveforms as a result.
///
/// The expressions `expr` and `args` may reference slider variables but
/// should otherwise be closed.
//
// Used by `PlayNoteOn` and `EvaluateUpdateSource` to invoke the keys
// function with `(midi_note, velocity)` arguments.
fn apply_note_function(
    expr: &parser::SourceExpr<MarkId>,
    args: Vec<parser::SourceExpr<MarkId>>,
    sliders: &renderer::ProgramSliders,
) -> Result<(waveform::Waveform<MarkId>, waveform::Waveform<MarkId>), String> {
    use parser::Expr::{Tuple, Waveform};
    let expr = parser::SourceExpr::from(parser::Expr::Application {
        function: Box::new(expr.clone()),
        argument: Box::new(parser::SourceExpr::from(Tuple(args))),
    });
    let mut bindings = vec![];
    slider::append_slider_bindings(
        &sliders.configs,
        &sliders.normalized_values,
        MarkId::Slider,
        &mut bindings,
    );
    let resolve = |_: &[String]| {
        Err(parser::Error::new(
            "Didn't expect to resolve in apply_note_function".to_string(),
        ))
    };
    let expr = parser::evaluate(resolve, &bindings, expr).map_err(|e| e.to_string());
    match expr.map(|s| s.expr) {
        Ok(Tuple(mut exprs)) => {
            if exprs.len() != 2 {
                return Err(format!(
                    "Expected 2 waveforms for note, got {} elements",
                    exprs.len()
                ));
            }
            match (exprs.remove(0).expr, exprs.remove(0).expr) {
                (Waveform(note_on), Waveform(note_off)) => Ok((note_on, note_off)),
                (expr, Waveform(_)) => Err(format!("Expected waveform for note-on, got: {}", expr)),
                (_, expr) => Err(format!("Expected waveform for note-off, got: {}", expr)),
            }
        }
        Ok(expr) => Err(format!("Expected 2 waveforms for note, got: {}", expr)),
        Err(e) => Err(format!("Error evaluating note: {}", e)),
    }
}

/// Splices the active program's edited `text` back into `state.source`,
/// re-parses the file to refresh `state.bindings`, realigns each program's
/// `span`/`binding_index` (matching by `slot=N` annotation), and writes
/// the new source to `state.input_path`.
///
/// Programs whose `span` is empty (`0..0`) are treated as brand-new: a
/// fresh `Definition` is appended at the end of source with a generated
/// name and a `//#{slot=N}` annotation so the file picks it up on
/// re-parse. Existing programs are spliced in place.
///
/// Returns a user-visible warning message if any step failed; in that
/// case `state.bindings`, `state.source`, and the file on disk are all
/// left unchanged.
fn splice_active_program(state: &mut AppState) -> Result<(), String> {
    let active = state.active_program_index;
    let slot = active + 1;
    let edited_span = state.programs[active].span.clone();
    let edited_text = state.programs[active].text.clone();
    let is_new = edited_span.start == edited_span.end;

    let mut new_source = state.source.clone();
    if is_new {
        if edited_text.trim().is_empty() {
            // Padding slot still empty after edit — nothing to do.
            return Ok(());
        }
        // Append a fresh Definition with a `slot=N` annotation. `_` is
        // the auto-generated name.
        // TODO double check this comma logic
        let separator = if new_source.is_empty() {
            ""
        } else if new_source.ends_with(',') || new_source.ends_with('\n') {
            "\n"
        } else {
            ",\n"
        };
        new_source.push_str(&format!(
            "{}//#{{slot={}}}\n_ = {}",
            separator, slot, edited_text
        ));
    } else {
        if edited_span.end > state.source.len() {
            return Err(format!(
                "active program span {:?} is out of bounds for source of length {}",
                edited_span,
                state.source.len()
            ));
        }
        new_source.replace_range(edited_span, &edited_text);
    }

    let new_bindings = parser::parse_file::<MarkId>(&new_source)
        .map_err(|errs| format!("Warning: source re-parse failed: {:?}", errs))?;

    // Build a slot -> (binding_index, expr_span) map from the new bindings,
    // so we can realign each `state.programs[slot-1]` independently of any
    // shifts in source order.
    let slot_lookup: Vec<Option<(usize, std::ops::Range<usize>)>> = {
        let mut by_slot: std::collections::HashMap<u32, (usize, std::ops::Range<usize>)> =
            std::collections::HashMap::new();
        for (i, sb) in new_bindings.iter().enumerate() {
            let slot = sb.annotations.iter().find_map(|sa| match &sa.annotation {
                parser::Annotation::Slot(n) => Some(*n),
                _ => None,
            });
            if let Some(slot) = slot
                && let parser::Binding::Definition(_, expr) = &sb.binding
                && let Some(s) = &expr.span
                && s.end <= new_source.len()
            {
                by_slot.insert(slot, (i, s.clone()));
            }
        }
        (1..=state.programs.len() as u32)
            .map(|slot| by_slot.remove(&slot))
            .collect()
    };

    for (i, program) in state.programs.iter_mut().enumerate() {
        match &slot_lookup[i] {
            Some((binding_index, span)) => {
                program.binding_index = *binding_index;
                program.span = span.clone();
                program.text = new_source[span.clone()].to_string();
            }
            None => {
                // No binding for this slot — keep it as a padding slot.
                program.binding_index = new_bindings.len();
                program.span = 0..0;
            }
        }
    }

    // Skip the file write when no path is set — tests use an empty
    // `PathBuf` so they can exercise the splice without touching disk.
    if !state.input_path.as_os_str().is_empty()
        && let Err(e) = std::fs::write(&state.input_path, &new_source)
    {
        return Err(format!(
            "Warning: failed to write {}: {}",
            state.input_path.display(),
            e
        ));
    }

    state.source = new_source;
    state.bindings = new_bindings;
    Ok(())
}

/// External handles the runner needs but doesn't own.
pub struct World<'a> {
    pub launchkey: Option<&'a mut launchkey::Launchkey>,
    pub status: &'a tracker::Status<WaveformId, MarkId>,
    pub play_helper: &'a mut play_helper::PlayHelper,
}

pub struct EffectRunner {
    pub command_sender: mpsc::Sender<tracker::Command<WaveformId, MarkId>>,
    pub slider_sender: mpsc::Sender<renderer::SliderEvent>,
}

impl EffectRunner {
    pub fn new(
        command_sender: mpsc::Sender<tracker::Command<WaveformId, MarkId>>,
        slider_sender: mpsc::Sender<renderer::SliderEvent>,
    ) -> Self {
        Self {
            command_sender,
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
        let mut all_effects = Vec::new();
        for action in actions {
            all_effects.extend(actions::apply(state, action));
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
                // TODO do we need to get()/match here?
                let program = match state.programs.get(program_index) {
                    Some(p) => p,
                    None => return,
                };
                if let Some(message) = world.play_helper.play_program_as_waveform(
                    program,
                    world.status,
                    start_at_next_measure,
                    repeat_after_measures,
                ) {
                    state.message = message;
                }
            }
            Effect::StopProgram(program_index) => {
                let program = match state.programs.get(program_index) {
                    Some(p) => p,
                    None => return,
                };
                world
                    .play_helper
                    .stop_waveform(WaveformId::Program(program.id));
            }
            Effect::RemovePendingProgram(program_index) => {
                let program = match state.programs.get(program_index) {
                    Some(p) => p,
                    None => return,
                };
                let _ = self.command_sender.send(tracker::Command::RemovePending {
                    id: WaveformId::Program(program.id),
                });
            }
            Effect::ModifyWaveform {
                id,
                mark_id,
                waveform,
            } => {
                let _ = self.command_sender.send(tracker::Command::Modify {
                    id,
                    mark_id,
                    waveform,
                });
            }

            Effect::EvaluateAndUpdateSource { mode_on_failure } => {
                // Try to parse and evaluate the active program's text.
                let result = world
                    .play_helper
                    .evaluate_program(&state.bindings, state.active_program());

                let expr = match result {
                    Ok(expr) => expr,
                    Err(message) => {
                        // If not successful, we still update the program's
                        // cached values. We still do this even though editing
                        // clears these as it may have been a dependency that
                        // caused the problem.
                        state.active_program_mut().cached_waveform = None;
                        state.active_program_mut().cached_keys_instrument = None;

                        // Return to the indicated mode.
                        state.message = message;
                        state.mode = mode_on_failure;
                        return;
                    }
                };

                // Splice the edited program text back into the file-level
                // source, re-parse to refresh `state.bindings`, realign each
                // program's span/text with its (possibly shifted) binding, and
                // persist the new source to disk.
                if let Err(message) = splice_active_program(state) {
                    state.message = message;
                }

                // If successful, see if it's a waveform or a keys instrument and update
                // the program's cached values.
                if let parser::Expr::Waveform(w) = expr.expr {
                    state.active_program_mut().cached_waveform = Some(w);
                    state.active_program_mut().cached_keys_instrument = None;
                } else if matches!(
                    expr.expr,
                    parser::Expr::Function { .. } | parser::Expr::BuiltIn { .. }
                ) {
                    state.active_program_mut().cached_waveform = None;
                    // Sanity check: actually invoke with dummy args.
                    // TODO use a waveform for velocity
                    if let Err(message) = apply_note_function(
                        &expr,
                        vec![
                            parser::SourceExpr::float(60.0),
                            parser::SourceExpr::float(0.7),
                        ],
                        &state.active_program().sliders,
                    ) {
                        state.message = message;
                        state.active_program_mut().cached_keys_instrument = None;
                        state.mode = mode_on_failure;
                        return;
                    }
                    state.active_program_mut().cached_keys_instrument = Some(expr);
                }
                state.mode = renderer::Mode::Select;
            }

            Effect::InstallKeys(program_index) => {
                let program = &state.programs[program_index];
                if let Some(expr) = &program.cached_keys_instrument {
                    let new_keys = Keys {
                        id: program.id,
                        function: expr.clone(),
                        sliders: program.sliders.clone(),
                        level_db: program.level_db,
                        note_off_waveforms: HashMap::new(),
                    };
                    let id = new_keys.id;
                    state.keys = Some(new_keys);
                    state.message = format!("Installed keys from program {}", id);
                }
            }

            Effect::PlayNoteOn { key, velocity } => {
                let Some(keys) = state.keys.as_mut() else {
                    return;
                };
                let args = vec![
                    parser::SourceExpr::float(key as f32),
                    // TODO use a marked waveform for velocity so we can implement after-touch
                    parser::SourceExpr::float(velocity as f32 / 127.0),
                ];
                match apply_note_function(&keys.function, args, &keys.sliders) {
                    Ok((note_on, note_off)) => {
                        let level_db = keys.level_db;
                        keys.note_off_waveforms.insert(key, note_off);
                        // Seed the slider-update worker's `last_slider_values`
                        // map for this new Key id. Without this, the next
                        // PluginEncoderChange for the active keys program
                        // would fail to look up a previous value when building
                        // the slider ramp (and the renderer wouldn't have
                        // initial values to show either).
                        let id = WaveformId::Key(key);
                        let program = &state.programs[renderer::index_from_id(keys.id)];
                        let mut last_slider_values = HashMap::new();
                        for (j, config) in program.sliders.configs.iter().enumerate() {
                            let value = slider::denormalize(
                                &config.function,
                                program.sliders.normalized_values[j],
                            )
                            .unwrap_or(0.0);
                            last_slider_values.insert((id.clone(), config.label.clone()), value);
                        }
                        let _ =
                            self.slider_sender
                                .send(renderer::SliderEvent::UpdateInitialValues(
                                    last_slider_values,
                                ));
                        let _ = self.command_sender.send(tracker::Command::Play {
                            id,
                            waveform: play_helper::build_top_level_waveform(note_on, level_db),
                            start: None,
                            repeat_every: None,
                        });
                    }
                    Err(message) => {
                        state.message = message;
                    }
                }
            }
            Effect::PlayNoteOff { key } => {
                let id = WaveformId::Key(key);
                if let Some(keys) = state.keys.as_mut()
                    && let Some(note_off) = keys.note_off_waveforms.remove(&key)
                {
                    let _ = self.command_sender.send(tracker::Command::Modify {
                        id,
                        mark_id: MarkId::Terminator,
                        waveform: note_off,
                    });
                    return;
                }
                // No stored note-off (key wasn't NoteOn'd, or keys were
                // uninstalled mid-note). Send a generic stop ramp; it's a
                // no-op if there's no matching waveform on the tracker.
                world.play_helper.stop_waveform(id);
            }

            Effect::UpdateSlider { id, slider, value } => {
                let _ = self
                    .slider_sender
                    .send(renderer::SliderEvent::UpdateSlider { id, slider, value });
            }
            Effect::UpdateActiveKeySliders { slider, value } => {
                for mark in &world.status.marks {
                    if let WaveformId::Key(_) = mark.waveform_id {
                        let _ = self
                            .slider_sender
                            .send(renderer::SliderEvent::UpdateSlider {
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
                        let _ = self.command_sender.send(tracker::Command::Modify {
                            id: mark.waveform_id.clone(),
                            mark_id: MarkId::Amplitude,
                            waveform: Waveform::Const(amplitude),
                        });
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
                let program = state.active_program();
                if let Some(waveform) = &program.cached_waveform {
                    println!("Waveform definition for program {}:", program.id);
                    println!("{:#?}", waveform);
                    state.message = "Dumped waveform to console".to_string();
                } else {
                    println!("No waveform associated with program {}:", program.id);
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
            let program = match state.programs.get(state.active_program_index) {
                Some(p) => p,
                None => return,
            };
            // Set the display for all encoders.
            for i in 0..launchkey::NUM_ENCODERS {
                if let Some(value) = program.sliders.normalized_values.get(i as usize) {
                    let config = &program.sliders.configs[i as usize];
                    let actual_value = slider::denormalize(&config.function, *value).unwrap_or(0.0);
                    launchkey.set_encoder_display(
                        i,
                        &config.label,
                        &renderer::format_sig_digits(actual_value, 3).to_string(),
                    );
                } else {
                    launchkey.set_encoder_display(i, "", "");
                }
            }
        }
        launchkey::EncoderMode::Mixer => {
            let bank_start = state.bank_start();
            for i in 0..PROGRAMS_PER_BANK {
                let program = match state.programs.get(bank_start + i) {
                    Some(p) => p,
                    None => continue,
                };
                launchkey.set_encoder_display(
                    i as u8,
                    "level",
                    &format!("{:.1} dB", program.level_db),
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    //! Exercises the open-file → edit-program → splice → reparse loop
    //! without touching `PlayHelper` or disk: `AppState::from_source`
    //! gives a state matching what `main` would build, and
    //! `splice_active_program` skips the file write when `input_path` is
    //! empty.
    use super::*;
    use std::path::PathBuf;

    /// Builds an `AppState` from inline source, with `input_path` empty
    /// so `splice_active_program` is a no-op against disk.
    fn state_from(source: &str) -> AppState {
        AppState::from_source(source.to_string(), PathBuf::new()).expect("test source should parse")
    }

    #[test]
    fn editing_existing_program_splices_into_its_span() {
        let source = "\
//#{slot=1}
kick = pulse(60),
//#{slot=2}
synth = saw(220)";
        let mut state = state_from(source);

        // Slot 2 → index 1. Sanity-check the initial extraction.
        assert_eq!(state.programs[0].text, "pulse(60)");
        assert_eq!(state.programs[1].text, "saw(220)");

        // Simulate the Edit-mode flow: select slot 2, change the text.
        state.active_program_index = 1;
        state.programs[1].set_text("saw(440)".to_string());

        splice_active_program(&mut state).expect("splice should succeed");

        assert_eq!(
            state.source,
            "\
//#{slot=1}
kick = pulse(60),
//#{slot=2}
synth = saw(440)"
        );
        // Slot 1 untouched, slot 2's program now reflects the new text and
        // span.
        assert_eq!(state.programs[0].text, "pulse(60)");
        assert_eq!(state.programs[1].text, "saw(440)");
    }

    #[test]
    fn editing_program_does_not_disturb_library_bindings() {
        // A `Definition` without a `slot=N` annotation is a library
        // binding, not a UI program. Editing a UI program must leave
        // library bindings (and `Open` directives) in place.
        let source = "\
open util.synths,
pi = 3.14159,
//#{slot=1}
tone = saw(440)";
        let mut state = state_from(source);
        assert_eq!(state.programs[0].text, "saw(440)");

        state.active_program_index = 0;
        state.programs[0].set_text("saw(220)".to_string());
        splice_active_program(&mut state).unwrap();

        assert_eq!(
            state.source,
            "\
open util.synths,
pi = 3.14159,
//#{slot=1}
tone = saw(220)"
        );
    }

    #[test]
    fn new_program_is_appended_with_slot_annotation() {
        // Editing a previously-empty padding slot (span 0..0) should
        // append a fresh `//#{slot=N}\n_ = <text>` Definition at the
        // end of source rather than splicing at offset 0.
        let source = "\
//#{slot=1}
kick = pulse(60)";
        let mut state = state_from(source);
        assert_eq!(state.programs[0].text, "pulse(60)");
        // Slot 5 is padding initially.
        assert_eq!(state.programs[4].text, "");
        assert_eq!(state.programs[4].span, 0..0);

        // Edit slot 5.
        state.active_program_index = 4;
        state.programs[4].set_text("saw(440)".to_string());

        splice_active_program(&mut state).unwrap();

        assert_eq!(
            state.source,
            "\
//#{slot=1}
kick = pulse(60),
//#{slot=5}
_ = saw(440)"
        );
        // The new binding is now part of the bindings vec, and the slot 5
        // program has a real span/text.
        assert_eq!(state.programs[4].text, "saw(440)");
        assert!(state.programs[4].span.start < state.programs[4].span.end);
    }

    // TODO add a test where a new program is adding in the middle

    #[test]
    fn editing_padding_slot_with_empty_text_is_a_no_op() {
        // The user enters Edit on an empty slot, types nothing, hits
        // Return. We must not append a phantom `_ = ` definition.
        let source = "\
//#{slot=1}
kick = pulse(60)";
        let mut state = state_from(source);
        let before = state.source.clone();
        state.active_program_index = 4;
        // text is already empty; just splice.
        splice_active_program(&mut state).unwrap();
        assert_eq!(state.source, before);
    }
}
