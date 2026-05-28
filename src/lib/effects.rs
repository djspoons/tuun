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

/// Applies a note function expecting a pair of Waveforms as a result. Used
/// by `PlayNoteOn` and `InstallKeysFromActive` to invoke the installed keys
/// function with `(midi_note, velocity)` arguments.
fn apply_note_function_as_waveforms(
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
    pub context: &'a [(String, parser::Expr<MarkId>)],
    pub play_helper: &'a mut play_helper::PlayHelper,
}

pub struct EffectRunner {
    pub command_sender: mpsc::Sender<tracker::Command<WaveformId, MarkId>>,
    pub slider_sender: mpsc::Sender<renderer::SliderEvent>,
    /// Set by `Effect::ShowMessage` and the play-helper paths; drained at
    /// the end of every `run_all` and folded into `state.mode`. Private
    /// so callers can't accidentally read stale values across batches.
    last_message: Option<String>,
}

impl EffectRunner {
    pub fn new(
        command_sender: mpsc::Sender<tracker::Command<WaveformId, MarkId>>,
        slider_sender: mpsc::Sender<renderer::SliderEvent>,
    ) -> Self {
        Self {
            command_sender,
            slider_sender,
            last_message: None,
        }
    }

    /// Runs every effect against `state` and `world`, then writes any
    /// resulting status message into `state.mode`.
    //
    // Callers don't see `last_message` — the fold is the runner's job, so there
    // is one place for the policy to live.
    pub fn run_all(&mut self, state: &mut AppState, world: &mut World, effects: Vec<Effect>) {
        for effect in effects {
            self.run_one(state, world, effect);
        }
        if let Some(msg) = self.last_message.take() {
            match &mut state.mode {
                Mode::Select { message } | Mode::Edit { message, .. } | Mode::Keys { message } => {
                    *message = msg
                }
                _ => {}
            }
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
        let mut all_effects = Vec::new();
        for action in actions {
            all_effects.extend(actions::apply(state, action));
        }
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
                let resulting_mode = world.play_helper.play_waveform(
                    world.context,
                    cursor_position,
                    program,
                    world.status,
                    start_at_next_measure,
                    repeat_after_measures,
                );
                if return_to_select_on_success {
                    // Caller (e.g. SDL2 Edit+Return) wants the mode to
                    // follow play_waveform's verdict: Select on success,
                    // Edit (with the error) on parse failure. The mode
                    // already carries the message, so we don't fold it
                    // into last_message a second time.
                    state.mode = resulting_mode;
                } else if let Mode::Select { message } = resulting_mode {
                    if !message.is_empty() {
                        self.last_message = Some(message);
                    }
                } else if let Mode::Edit { message, .. } = resulting_mode {
                    if !message.is_empty() {
                        self.last_message = Some(message);
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
                        self.last_message = Some(format!("Expected note function, got: {}", other));
                        return;
                    }
                    Err(errors) => {
                        self.last_message = Some(format!("Error: {}", errors[0].to_string()));
                        return;
                    }
                };
                // Sanity check: actually invoke with dummy args.
                // TODO use a waveform for velocity
                if let Err(message) = apply_note_function_as_waveforms(
                    world.context,
                    &function,
                    vec![parser::Expr::Float(60.0), parser::Expr::Float(0.7)],
                    &program.sliders,
                ) {
                    self.last_message = Some(message);
                    return;
                }
                let new_keys = Keys {
                    id: program.id,
                    context: Vec::from(world.context),
                    function,
                    sliders: program.sliders.clone(),
                    level_db: program.level_db,
                    note_off_waveforms: HashMap::new(),
                };
                let id = new_keys.id;
                state.keys = Some(new_keys);
                if let Some(lk) = world.launchkey.as_deref_mut() {
                    lk.set_capture_midi_brightness(127);
                }
                self.last_message = Some(format!("Installed keys from program {}", id));
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
                        self.last_message = Some(message);
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
                self.last_message = Some(msg);
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

            Effect::SetCaptureMidiBrightness(b) => {
                if let Some(lk) = world.launchkey.as_deref_mut() {
                    lk.set_capture_midi_brightness(b);
                }
            }
            Effect::SetLaunchkeyEncoderMode(new_mode) => {
                if let Some(lk) = world.launchkey.as_deref_mut() {
                    if lk.encoder_mode != new_mode {
                        lk.encoder_mode = new_mode;
                        sync_encoders(state, lk);
                    }
                }
            }

            Effect::SaveProgramsToFile => {
                use std::io::Write;
                let datetime = chrono::Local::now().format("%Y-%m-%d_%H-%M-%S");
                let filename = format!("programs_{}.tuunp", datetime);
                let mut file = match std::fs::File::create(&filename) {
                    Ok(f) => f,
                    Err(e) => {
                        self.last_message = Some(format!("Failed to create {}: {}", filename, e));
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
                self.last_message = Some(format!("Saved to {}", filename));
            }
            Effect::DumpActiveWaveform => {
                let program = state.active_program();
                match play_helper::prepare_waveform(world.context, program.text.len(), program) {
                    renderer::WaveformOrMode::Waveform(waveform) => {
                        println!("Waveform definition for program {}:", program.id);
                        println!("{:#?}", waveform);
                        self.last_message = Some("Dumped waveform to console".to_string());
                    }
                    renderer::WaveformOrMode::Mode(mode) => {
                        if let Mode::Edit { message, .. } = mode {
                            self.last_message = Some(message);
                        }
                    }
                }
            }
        }
    }
}

/// Pushes the current bank/program's encoder values to the controller.
/// Called only via `Effect::SyncEncoders`, never on a tick.
fn sync_encoders(state: &AppState, launchkey: &mut launchkey::Launchkey) {
    let bank_start = state.bank_start();
    match launchkey.encoder_mode {
        launchkey::EncoderMode::Plugin => {
            let program = match state.programs.get(state.active_program_index) {
                Some(p) => p,
                None => return,
            };
            for (i, value) in program.sliders.normalized_values.iter().enumerate() {
                let config = &program.sliders.configs[i];
                let actual_value = slider::denormalize(&config.function, *value).unwrap_or(0.0);
                launchkey.set_encoder_display(
                    i as u8,
                    &config.label,
                    &format!("{:.3}", actual_value),
                );
                launchkey
                    .update_encoder_state(i as u8, (launchkey::ENCODER_MAX as f32 * value) as u16);
            }
        }
        launchkey::EncoderMode::Mixer => {
            for i in 0..PROGRAMS_PER_BANK {
                let program = match state.programs.get(bank_start + i) {
                    Some(p) => p,
                    None => continue,
                };
                let encoder_value = actions::level_db_to_encoder(program.level_db);
                launchkey.update_encoder_state(i as u8, encoder_value);
            }
        }
    }
}
