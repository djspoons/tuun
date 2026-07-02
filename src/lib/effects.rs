//! Executes `Effect`s emitted by the reducer in `actions.rs`.
//!
//! This module owns the concrete I/O handles (MPSC senders, the Launchkey
//! controller). Keeping it separate from `actions.rs` means the reducer
//! stays pure and easy to test.

use std::sync::mpsc;

use std::collections::HashMap;

use crate::actions::{self, AppState, Effect};
use crate::midi_input::Keys;
use crate::parser;
use crate::play_helper;
use crate::renderer::{self, MarkId, PROGRAMS_PER_BANK, WaveformId};
use crate::slider;
use crate::tracker;
use crate::waveform;
use crate::{launchkey, optimizer};

/// Applies a note function `expr` to the given `args`, expecting a pair
/// of Waveforms as a result.
///
/// The expressions `expr` and `args` should be closed.
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

/// Float tolerance for "this hasn't moved" comparisons against parsed
/// annotation values. Tighter than the source format's precision would
/// matter (which would be confusing) and looser than encoder/MIDI noise.
const ANNOTATION_EPSILON: f32 = 1e-4;

/// Returns annotation-persistence edits for `program` against its parsed
/// `binding`.
///
/// This includes one for `level_db` if the runtime level has diverged from what
/// the binding's last `Level` annotation encodes, and one for `sliders` if any
/// slider's current normalized value has diverged from its parsed initial
/// value. Returns an empty list when nothing has changed.
fn annotation_edits(
    program: &renderer::Program,
    binding: &parser::SourceBinding<MarkId>,
    source: &str,
) -> Vec<(std::ops::Range<usize>, String)> {
    let mut edits = Vec::new();
    if let Some(edit) = level_edit(program, binding, source) {
        edits.push(edit);
    }
    if let Some(edit) = sliders_edit(program, binding) {
        edits.push(edit);
    }
    edits
}

/// Returns the edit for the binding's `level_db` annotation, or `None` if
/// the runtime level matches what the binding currently encodes.
fn level_edit(
    program: &renderer::Program,
    binding: &parser::SourceBinding<MarkId>,
    source: &str,
) -> Option<(std::ops::Range<usize>, String)> {
    let (parsed_value, parsed_span) = match last_annotation_of(binding, |a| match a {
        parser::Annotation::Level(v) => Some(*v),
        _ => None,
    }) {
        Some((v, span)) => (v, span),
        None => (0.0, None),
    };
    if (program.level_db - parsed_value).abs() < ANNOTATION_EPSILON {
        return None;
    }
    let annotation = parser::Annotation::Level(program.level_db);
    let body = format!("{}", annotation);
    match parsed_span {
        Some(span) => Some((span, body)),
        None => {
            // Fall back to inserting a fresh `#{level_db=…}` line at the
            // binding's start when no `Level` annotation exists in source yet.
            let pos = binding
                .span
                .as_ref()
                .expect("parsed binding has span")
                .start;
            Some(insert_annotation_line(pos, &body, source))
        }
    }
}

/// Builds an `(range, replacement)` edit that inserts a fresh `#{…}`
/// annotation line at `pos`, prepending or appending `\n` only when
/// needed so the inserted line doesn't collapse onto a neighbor.
fn insert_annotation_line(
    pos: usize,
    body: &str,
    source: &str,
) -> (std::ops::Range<usize>, String) {
    let bytes = source.as_bytes();
    let prefix = if pos == 0 || bytes.get(pos - 1) == Some(&b'\n') {
        ""
    } else {
        "\n"
    };
    let suffix = if bytes.get(pos) == Some(&b'\n') {
        ""
    } else {
        "\n"
    };
    (pos..pos, format!("{}#{{{}}}{}", prefix, body, suffix))
}

/// Returns the edit for the binding's `sliders` annotation, or `None` if
/// every slider's current normalized value matches its parsed initial.
fn sliders_edit(
    program: &renderer::Program,
    binding: &parser::SourceBinding<MarkId>,
) -> Option<(std::ops::Range<usize>, String)> {
    if program.sliders.configs.is_empty() {
        return None;
    }
    let diverged = program
        .sliders
        .configs
        .iter()
        .zip(&program.sliders.normalized_values)
        .any(|(config, &current)| {
            (current - parsed_normalized_value(&config.function)).abs() > ANNOTATION_EPSILON
        });
    if !diverged {
        return None;
    }
    let (_, span) = last_annotation_of(binding, |a| match a {
        parser::Annotation::Sliders(_) => Some(()),
        _ => None,
    })?;
    let span = span?;
    let updated: Vec<parser::Slider> = program
        .sliders
        .configs
        .iter()
        .zip(&program.sliders.normalized_values)
        .map(|(config, &normalized)| {
            let function = match &config.function {
                parser::SliderFunction::Linear { min, max, .. } => parser::SliderFunction::Linear {
                    initial_value: min + normalized * (max - min),
                    min: *min,
                    max: *max,
                },
                parser::SliderFunction::UserDefined {
                    function_source, ..
                } => parser::SliderFunction::UserDefined {
                    normalized_initial_value: normalized,
                    function_source: function_source.clone(),
                },
            };
            parser::Slider {
                label: config.label.clone(),
                function,
            }
        })
        .collect();
    Some((span, format!("{}", parser::Annotation::Sliders(updated))))
}

/// Returns the (value, span) of the last annotation on `binding` that `pick`
/// matches.
///
/// The reverse walk matches the parser's "last wins" semantics for repeated
/// annotations of the same kind, so persisting onto the same span keeps the
/// source authoritative.
fn last_annotation_of<T, F>(
    binding: &parser::SourceBinding<MarkId>,
    mut pick: F,
) -> Option<(T, Option<std::ops::Range<usize>>)>
where
    F: FnMut(&parser::Annotation) -> Option<T>,
{
    binding
        .annotations
        .iter()
        .rev()
        .find_map(|a| pick(&a.annotation).map(|v| (v, a.span.clone())))
}

/// Returns the normalized 0..1 position implied by a parsed slider's
/// initial value, so it can be compared against the runtime's current
/// `normalized_value` directly.
fn parsed_normalized_value(function: &parser::SliderFunction) -> f32 {
    match function {
        parser::SliderFunction::Linear {
            initial_value,
            min,
            max,
        } => (initial_value - min) / (max - min),
        parser::SliderFunction::UserDefined {
            normalized_initial_value,
            ..
        } => *normalized_initial_value,
    }
}

/// Splices the given program's edited `text` back into `state.source`, persists
/// every program's changed state as annotation edits, re-parses the file to
/// refresh `state.bindings`, realigns each program's `span`/`binding_index`
/// (matching by `slot=N` annotation), and writes the new source to
/// `state.input_path`.
///
/// Programs whose `span` is empty (`0..0`) are treated as brand-new: a fresh
/// `Definition` is appended at the end of source with a generated name and a
/// `#{slot=N}` annotation so the file picks it up on re-parse. Existing
/// programs are spliced in place.
///
/// Returns a user-visible warning message if any step failed; in that case
/// `state.bindings`, `state.source`, and the file on disk are all left
/// unchanged.
fn splice_program(state: &mut AppState, program_index: usize) -> Result<(), String> {
    let slot = program_index + 1;
    let edited_span = state.programs[program_index].span.clone();
    let mut edited_text = state.programs[program_index].text.clone();
    // Remove any semicolons since these aren't valid within an expression and
    // can defeat parsing error recovery.
    edited_text.retain(|c| c != ';');
    let is_new = edited_span.start == edited_span.end;

    // Collect every edit we want to apply to `state.source` in one pass:
    // the active program's expression splice (or a fresh-binding append)
    // plus, for every program, annotation edits that bring level/slider
    // state on disk in line with the runtime. Applying in reverse-position
    // order keeps each remaining edit's span valid.
    let mut edits: Vec<(std::ops::Range<usize>, String)> = Vec::new();
    let mut append: Option<String> = None;
    if is_new {
        if edited_text.trim().is_empty() {
            // Padding slot still empty after edit — nothing to do.
            // TODO is this true? what about sliders/levels?
            return Ok(());
        }
        // Build the appended binding inline: slot is mandatory; level_db
        // gets added when the runtime has a non-default level (new
        // programs have no sliders since their configs come from a
        // parsed `sliders=…` annotation).
        let mut annos: Vec<parser::Annotation> = vec![parser::Annotation::Slot(slot as u32)];
        let level = state.programs[program_index].level_db;
        if level.abs() > ANNOTATION_EPSILON {
            annos.push(parser::Annotation::Level(level));
        }
        let anno_body = annos
            .iter()
            .map(|a| format!("{}", a))
            .collect::<Vec<_>>()
            .join(", ");
        append = Some(format!("\n#{{{}}}\n_ = {};", anno_body, edited_text));
    } else {
        edits.push((edited_span, edited_text));
    }

    // Walk every program (active and not) and append annotation edits
    // for level_db / sliders that have diverged from the parsed source.
    // We use the *pre-splice* `state.bindings` so all spans line up with
    // `state.source` before any of these edits land.
    for program in &state.programs {
        if let Some(binding) = state.bindings.get(program.binding_index) {
            edits.extend(annotation_edits(program, binding, &state.source));
        }
    }

    let mut new_source = state.source.clone();
    edits.sort_by_key(|(span, _)| std::cmp::Reverse(span.start));
    for (span, replacement) in edits {
        new_source.replace_range(span, &replacement);
    }
    if let Some(text) = append {
        new_source.push_str(&text);
    }

    // It's ok to drop the errors here: if parse_module succeeded then we know
    // that the resulting bindings span the entire input (even if there are
    // errors).
    let (new_bindings, _errors) = parser::parse_module::<MarkId>(&new_source)
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
                if state.programs.get(program_index).is_none() {
                    return;
                }
                let display_name = actions::program_display_name(state, program_index);
                let program = &state.programs[program_index];
                if let Some(message) = world.play_helper.play_program_as_waveform(
                    program_index,
                    program,
                    &display_name,
                    world.status,
                    start_at_next_measure,
                    repeat_after_measures,
                ) {
                    state.message = message;
                }
            }
            Effect::StopProgram(program_index) => {
                if state.programs.get(program_index).is_none() {
                    return;
                }
                world
                    .play_helper
                    .stop_waveform(WaveformId::Program(program_index));
            }
            Effect::RemovePendingProgram(program_index) => {
                if state.programs.get(program_index).is_none() {
                    return;
                }
                let _ = self.command_sender.send(tracker::Command::RemovePending {
                    id: WaveformId::Program(program_index),
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

            Effect::EvaluateProgram {
                program_index,
                mode_on_failure,
            } => {
                // Try to parse and evaluate the given program's text.
                let result = world
                    .play_helper
                    .evaluate_program(&state.bindings, &state.programs[program_index]);

                let expr = match result {
                    Ok(expr) => expr,
                    Err(message) => {
                        // If not successful, we still update the program's
                        // cached values. We still do this even though editing
                        // clears these as it may have been a dependency that
                        // caused the problem.
                        state.programs[program_index].cached_waveform = None;
                        state.programs[program_index].cached_keys_instrument = None;

                        // Return to the indicated mode.
                        state.message = message;
                        state.mode = mode_on_failure;
                        return;
                    }
                };

                match expr.expr {
                    parser::Expr::Waveform(w) => {
                        state.programs[program_index].cached_waveform = Some(w);
                        state.programs[program_index].cached_keys_instrument = None;
                    }
                    parser::Expr::Seq { waveform, .. } => {
                        if let parser::Expr::Waveform(w) = waveform.expr {
                            state.programs[program_index].cached_waveform = Some(w);
                            state.programs[program_index].cached_keys_instrument = None;
                        }
                    }
                    parser::Expr::Function { .. } | parser::Expr::BuiltIn { .. } => {
                        state.programs[program_index].cached_waveform = None;
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
                            state.programs[program_index].cached_keys_instrument = None;
                            state.mode = mode_on_failure;
                            return;
                        }
                        state.programs[program_index].cached_keys_instrument = Some(expr);
                    }
                    _ => {
                        state.programs[program_index].cached_waveform = None;
                        state.programs[program_index].cached_keys_instrument = None;

                        state.message = "Program is not a waveform or keys instrument".to_string();
                        state.mode = mode_on_failure;
                        return;
                    }
                }
                state.mode = renderer::Mode::Select;
            }

            Effect::UpdateSource(program_index) => {
                // Splice the indicated program text back into the file-level
                // source, re-parse to refresh `state.bindings`, realign each
                // program's span/text with its (possibly shifted) binding, and
                // persist the new source to disk.
                if let Err(message) = splice_program(state, program_index) {
                    state.message = message;
                }
            }

            Effect::InstallKeys(program_index) => {
                let program = &state.programs[program_index];
                if let Some(expr) = &program.cached_keys_instrument {
                    let new_keys = Keys {
                        id: program_index,
                        function: expr.clone(),
                        sliders: program.sliders.clone(),
                        level_db: program.level_db,
                        note_off_waveforms: HashMap::new(),
                    };
                    state.keys = Some(new_keys);
                    state.message = format!(
                        "Installed keys from program {}",
                        actions::program_display_name(state, program_index)
                    );
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
                        let mut note_on = optimizer::optimize(note_on);
                        let note_off = optimizer::optimize(note_off);
                        let level_db = keys.level_db;
                        keys.note_off_waveforms.insert(key, note_off);
                        // `note_on`'s `Marked(Slider(_))` nodes still hold the
                        // values from when the instrument was originally
                        // evaluated, so swap in the program's current runtime
                        // values and at the same time collect the (label,
                        // value) pairs we need to seed the slider worker for
                        // this fresh `Key` id (so its next ramp has a sensible
                        // "previous value" to start from).
                        let id = WaveformId::Key(key);
                        let program = &state.programs[keys.id];
                        let last_slider_values: HashMap<(WaveformId, String), f32> =
                            play_helper::substitute_current_slider_values(
                                &mut note_on,
                                &program.sliders,
                            )
                            .into_iter()
                            .map(|(label, value)| ((id.clone(), label), value))
                            .collect();
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
                let program_index = state.active_program_index;
                let display_name = actions::program_display_name(state, program_index);
                let program = state.active_program();
                if let Some(waveform) = &program.cached_waveform {
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
                    &renderer::format_level_db(program.level_db),
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
#{slot=1}
kick = pulse(60);
#{slot=2}
synth = saw(220);";
        let mut state = state_from(source);

        // Slot 2 → index 1. Sanity-check the initial extraction.
        assert_eq!(state.programs[0].text, "pulse(60)");
        assert_eq!(state.programs[1].text, "saw(220)");

        // Simulate the Edit-mode flow: select slot 2, change the text.
        state.programs[1].set_text("saw(440)".to_string());

        splice_program(&mut state, 1).expect("splice should succeed");

        assert_eq!(
            state.source,
            "\
#{slot=1}
kick = pulse(60);
#{slot=2}
synth = saw(440);"
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
open util.synths;
pi = 3.14159;
#{slot=1}
tone = saw(440);";
        let mut state = state_from(source);
        assert_eq!(state.programs[0].text, "saw(440)");

        state.programs[0].set_text("saw(220)".to_string());
        splice_program(&mut state, 0).unwrap();

        assert_eq!(
            state.source,
            "\
open util.synths;
pi = 3.14159;
#{slot=1}
tone = saw(220);"
        );
    }

    #[test]
    fn new_program_is_appended_with_slot_annotation() {
        // Editing a previously-empty padding slot (span 0..0) should
        // append a fresh `#{slot=N}\n_ = <text>` Definition at the
        // end of source rather than splicing at offset 0.
        let source = "\
#{slot=1}
kick = pulse(60);";
        let mut state = state_from(source);
        assert_eq!(state.programs[0].text, "pulse(60)");
        // Slot 5 is padding initially.
        assert_eq!(state.programs[4].text, "");
        assert_eq!(state.programs[4].span, 0..0);

        // Edit slot 5.
        state.programs[4].set_text("saw(440)".to_string());

        splice_program(&mut state, 4).unwrap();

        assert_eq!(
            state.source,
            "\
#{slot=1}
kick = pulse(60);
#{slot=5}
_ = saw(440);"
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
#{slot=1}
kick = pulse(60);";
        let mut state = state_from(source);
        let before = state.source.clone();
        // text is already empty; just splice.
        splice_program(&mut state, 4).unwrap();
        assert_eq!(state.source, before);
    }

    #[test]
    fn changed_level_db_replaces_existing_annotation_in_place() {
        // The existing `level_db=-6` annotation has its own span; the
        // splice should land exactly on those bytes, leaving the rest
        // of the binding (including its leading comment) untouched.
        let source = "\
#{slot=1, level_db=-6}
// keep this comment
kick = pulse(60);";
        let mut state = state_from(source);
        assert!((state.programs[0].level_db - -6.0).abs() < 1e-6);

        state.programs[0].level_db = -3.5;
        splice_program(&mut state, 0).unwrap();

        assert_eq!(
            state.source,
            "\
#{slot=1, level_db=-3.5}
// keep this comment
kick = pulse(60);"
        );
    }

    #[test]
    fn changed_level_db_inserts_fresh_annotation_when_none_exists() {
        // No `level_db=…` in source. Add a new `#{level_db=…}` line just before
        // the binding, preserving the existing annotation block and the
        // pre-binding comment.
        let source = "\
// header comment
#{slot=1}
kick = pulse(60);";
        let mut state = state_from(source);
        assert!((state.programs[0].level_db - 0.0).abs() < 1e-6);

        state.programs[0].level_db = -3.0;
        splice_program(&mut state, 0).unwrap();

        assert_eq!(
            state.source,
            "\
#{level_db=-3}
// header comment
#{slot=1}
kick = pulse(60);"
        );
    }

    #[test]
    fn changed_slider_value_rewrites_sliders_annotation() {
        // Move the slider from its parsed initial 0.5 to 0.75. Only
        // the `sliders=…` annotation's bytes change; the surrounding
        // `slot=…` annotation and binding body stay verbatim.
        let source = "\
#{slot=1, sliders=[\"vol:0.5:0:1\"]}
tone = saw(220);";
        let mut state = state_from(source);
        assert_eq!(state.programs[0].sliders.configs.len(), 1);
        assert!((state.programs[0].sliders.normalized_values[0] - 0.5).abs() < 1e-6);

        state.programs[0].sliders.normalized_values[0] = 0.75;
        splice_program(&mut state, 0).unwrap();

        // Linear slider: actual value = min + normalized * (max - min)
        // = 0 + 0.75 * 1 = 0.75.
        assert_eq!(
            state.source,
            "\
#{slot=1, sliders=[\"vol:0.75:0:1\"]}
tone = saw(220);"
        );
    }

    #[test]
    fn no_divergence_means_no_annotation_edits() {
        // Splicing without any runtime divergence should round-trip
        // the source exactly — even though splice_program walks every
        // program looking for level/slider drift.
        let source = "\
#{slot=1, level_db=-6, sliders=[\"vol:0.25:0:1\"]}
kick = pulse(60);
#{slot=2}
synth = saw(220);";
        let mut state = state_from(source);
        // No runtime mutations — same edit-mode-exit pattern but with
        // the program's text unchanged from what parse_module produced.
        splice_program(&mut state, 0).unwrap();
        assert_eq!(state.source, source);
    }

    #[test]
    fn non_active_program_divergence_persists_on_any_save() {
        // The user adjusts level on program 2 via encoder, then edits
        // program 1's text and hits Return. Both writes need to land.
        let source = "\
#{slot=1}
kick = pulse(60);
#{slot=2}
synth = saw(220);";
        let mut state = state_from(source);

        // Program 2: level changed (encoder).
        state.programs[1].level_db = -9.0;
        // Program 1: text edited (active save target).
        state.programs[0].set_text("pulse(80)".to_string());

        splice_program(&mut state, 0).unwrap();

        assert_eq!(
            state.source,
            "\
#{slot=1}
kick = pulse(80);
#{level_db=-9}
#{slot=2}
synth = saw(220);"
        );
    }

    #[test]
    fn new_program_appends_level_db_when_set() {
        // A brand-new program with a non-default runtime level should emit
        // `level_db` alongside `slot` in its appended `#{…}` header so the file
        // picks it up on the next parse.
        let source = "\
#{slot=1}
kick = pulse(60);";
        let mut state = state_from(source);
        state.programs[4].set_text("saw(440)".to_string());
        state.programs[4].level_db = -2.5;

        splice_program(&mut state, 4).unwrap();

        assert_eq!(
            state.source,
            "\
#{slot=1}
kick = pulse(60);
#{slot=5, level_db=-2.5}
_ = saw(440);"
        );
    }
}
