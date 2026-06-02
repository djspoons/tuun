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
use crate::renderer::{self, MarkId, Mode, PROGRAMS_PER_BANK, WaveformId};
use crate::slider;
use crate::tracker;
use crate::waveform;

/// Applies a note function expecting a pair of Waveforms as a result.
//
// Used by `PlayNoteOn` and `InstallKeysFromActive` to invoke the installed keys
// function with `(midi_note, velocity)` arguments. Also used by
// `midi_input::update_launchkey_state` to validate whether a program is a
// valid keys instrument before coloring its pad.
pub fn apply_note_function_as_waveforms(
    context: &[(String, parser::Expr<MarkId>)],
    expr: &parser::Expr<MarkId>,
    args: Vec<parser::Expr<MarkId>>,
    sliders: &renderer::ProgramSliders,
) -> Result<(waveform::Waveform<MarkId>, waveform::Waveform<MarkId>), String> {
    use parser::Expr::{Tuple, Waveform};
    let expr = parser::Expr::Application {
        function: Box::new(expr.clone()),
        argument: Box::new(Tuple(args)),
    };
    let expr = slider::prepend_slider_bindings(
        &sliders.configs,
        &sliders.normalized_values,
        MarkId::Slider,
        expr,
    );
    let expr = parser::evaluate(context, expr).map_err(|e| e.to_string());
    match expr {
        Ok(Tuple(mut exprs)) => {
            if exprs.len() != 2 {
                return Err(format!(
                    "Expected 2 waveforms for note, got {} elements",
                    exprs.len()
                ));
            }
            match (exprs.remove(0), exprs.remove(0)) {
                (Waveform(note_on), Waveform(note_off)) => Ok((note_on, note_off)),
                (expr, Waveform(_)) => Err(format!("Expected waveform for note-on, got: {}", expr)),
                (_, expr) => Err(format!("Expected waveform for note-off, got: {}", expr)),
            }
        }
        Ok(expr) => Err(format!("Expected 2 waveforms for note, got: {}", expr)),
        Err(e) => Err(format!("Error evaluating note: {}", e)),
    }
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
                cursor_position,
                start_at_next_measure,
                repeat_after_measures,
                return_to_select_on_success,
            } => {
                let program = match state.programs.get(program_index) {
                    Some(p) => p,
                    None => return,
                };
                match world.play_helper.play_waveform(
                    &state.context,
                    program,
                    world.status,
                    start_at_next_measure,
                    repeat_after_measures,
                ) {
                    Ok(message) => {
                        if return_to_select_on_success {
                            state.mode = Mode::Select;
                        }
                        state.message = message;
                    }
                    Err(parse_err) => {
                        if return_to_select_on_success {
                            state.mode = Mode::Edit {
                                cursor_position,
                                errors: parse_err.errors,
                            };
                        }
                        state.message = parse_err.message;
                    }
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

            Effect::InstallKeysFromActive(program_index) => {
                let program = match state.programs.get(program_index) {
                    Some(p) => p,
                    None => return,
                };
                let function: parser::Expr<MarkId> = match parser::parse_program(&program.text) {
                    Ok(expr @ (parser::Expr::Function { .. } | parser::Expr::BuiltIn { .. })) => {
                        expr
                    }
                    Ok(other) => {
                        state.message = format!("Expected note function, got: {}", other);
                        return;
                    }
                    Err(errors) => {
                        state.message = format!("Error: {}", errors[0].to_string());
                        return;
                    }
                };
                // Sanity check: actually invoke with dummy args.
                // TODO use a waveform for velocity
                if let Err(message) = apply_note_function_as_waveforms(
                    &state.context,
                    &function,
                    vec![parser::Expr::Float(60.0), parser::Expr::Float(0.7)],
                    &program.sliders,
                ) {
                    state.message = message;
                    return;
                }
                let new_keys = Keys {
                    id: program.id,
                    context: state.context.clone(),
                    function,
                    sliders: program.sliders.clone(),
                    level_db: program.level_db,
                    note_off_waveforms: HashMap::new(),
                };
                let id = new_keys.id;
                state.keys = Some(new_keys);
                state.message = format!("Installed keys from program {}", id);
            }

            Effect::PlayNoteOn { key, velocity } => {
                let Some(keys) = state.keys.as_mut() else {
                    return;
                };
                let args = vec![
                    parser::Expr::Float(key as f32),
                    // TODO use a marked waveform for velocity so we can implement after-touch
                    parser::Expr::Float(velocity as f32 / 127.0),
                ];
                match apply_note_function_as_waveforms(
                    &keys.context,
                    &keys.function,
                    args,
                    &keys.sliders,
                ) {
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
                if let Some(keys) = state.keys.as_mut() {
                    if let Some(note_off) = keys.note_off_waveforms.remove(&key) {
                        let _ = self.command_sender.send(tracker::Command::Modify {
                            id,
                            mark_id: MarkId::Terminator,
                            waveform: note_off,
                        });
                        return;
                    }
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
                if let Some(lk) = world.launchkey.as_deref_mut() {
                    if lk.encoder_mode != new_mode {
                        lk.encoder_mode = new_mode;
                        // The device resets the relative-output feature
                        // when the user switches encoder modes, so we
                        // have to re-assert it every time.
                        lk.set_encoder_relative_output();
                        sync_encoders(state, lk);
                    }
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

            Effect::SaveProgramsToFile => {
                use std::io::Write;
                let datetime = chrono::Local::now().format("%Y-%m-%d_%H-%M-%S");
                let filename = format!("programs_{}.tuunp", datetime);
                let mut file = match std::fs::File::create(&filename) {
                    Ok(f) => f,
                    Err(e) => {
                        state.message = format!("Failed to create {}: {}", filename, e);
                        return;
                    }
                };
                for (i, program) in state.programs.iter().enumerate() {
                    if program.text.is_empty() {
                        continue;
                    }
                    let ps = &program.sliders;
                    let mut anno_parts: Vec<String> = Vec::new();
                    if i % PROGRAMS_PER_BANK == 0 && i > 0 && state.programs[i - 1].text.is_empty()
                    {
                        anno_parts.push("next_bank".to_string());
                    }
                    if !ps.configs.is_empty() {
                        let slider_strings: Vec<String> = ps
                            .configs
                            .iter()
                            .enumerate()
                            .map(|(j, c)| match &c.function {
                                parser::SliderFunction::Linear { min, max, .. } => {
                                    // Use the current value as the new initial value.
                                    let actual = min + ps.normalized_values[j] * (max - min);
                                    format!("\"{}:{:.3}:{:.3}:{:.3}\"", c.label, actual, min, max)
                                }
                                parser::SliderFunction::UserDefined {
                                    function_source, ..
                                } => format!(
                                    "\"{}:{:.3}:{}\"",
                                    c.label, ps.normalized_values[j], function_source
                                ),
                            })
                            .collect();
                        anno_parts.push(format!("sliders=[{}]", slider_strings.join(",")));
                    }
                    if let Some((r, g, b)) = program.color {
                        anno_parts.push(format!("color=rgb({},{},{})", r, g, b));
                    }
                    if program.level_db != 0.0 {
                        anno_parts.push(format!("level_db={}", program.level_db));
                    }
                    if !anno_parts.is_empty() {
                        let _ = writeln!(file, "//#{{{}}}", anno_parts.join(","));
                    }
                    let _ = writeln!(file, "{}", program.text);
                }
                state.message = format!("Saved to {}", filename);
            }
            Effect::DumpActiveWaveform => {
                let program = state.active_program();
                match play_helper::prepare_waveform(&state.context, program) {
                    renderer::WaveformOrError::Waveform(waveform) => {
                        println!("Waveform definition for program {}:", program.id);
                        println!("{:#?}", waveform);
                        state.message = "Dumped waveform to console".to_string();
                    }
                    renderer::WaveformOrError::Error(err) => {
                        state.message = err.message;
                    }
                }
            }
            Effect::LoadContext => {
                state.message = crate::loader::load_context(&state.config, &mut state.context);
                // The keys-validity cache memoizes an evaluation that
                // depends on `state.context`, so a context reload
                // invalidates every program's cached result.
                for program in &state.programs {
                    program.valid_keys_program.set(None);
                }
            }
            Effect::LoadPrograms => {
                let (slider_values, errors) =
                    crate::loader::load_programs(&state.config, &mut state.programs);
                let _ = self
                    .slider_sender
                    .send(renderer::SliderEvent::SetInitialValues(slider_values));
                state.message = if errors.is_empty() {
                    "Loaded programs".to_string()
                } else {
                    format!("Error loading programs: {}", errors[0].to_string())
                };
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
                        i as u8,
                        &config.label,
                        &format!("{}", renderer::format_sig_digits(actual_value, 3)),
                    );
                } else {
                    launchkey.set_encoder_display(i as u8, &"", &"");
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
