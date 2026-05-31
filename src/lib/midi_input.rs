use std::collections::HashMap;
use std::time::Instant;

use crate::actions;
use crate::launchkey;
use crate::parser;
use crate::renderer::{self, MarkId, PROGRAMS_PER_BANK, Program, WaveformId};
use crate::tracker;
use crate::waveform;

/// Represents a program installed to respond to MIDI note-on/-off events.
//
// TODO maybe move out of here since this isn't tied to MIDI input anymore
pub struct Keys {
    pub id: renderer::ProgramId,
    pub context: Vec<(String, parser::Expr<MarkId>)>,
    pub function: parser::Expr<MarkId>,
    pub sliders: renderer::ProgramSliders,
    pub level_db: f32,
    pub note_off_waveforms: HashMap<u8, waveform::Waveform<MarkId>>, // keys are MIDI note numbers
}

/// The number of rotations of the encoder that represents the full range.
const ENCODER_ROTATIONS: f32 = 4.0;

/// Classifies a launchkey event into a list of `Action`s.
pub fn classify(
    event: &launchkey::Event,
    state: &actions::AppState,
    status: &tracker::Status<WaveformId, MarkId>,
) -> Option<Vec<actions::Action>> {
    use actions::Action;
    use launchkey::Event;
    use std::time::Instant;
    let active_program_index = state.active_program_index;
    let programs: &[Program] = &state.programs;
    let keys: &Option<Keys> = &state.keys;
    let repeat_after_measures: Option<u32> = state.repeat_after_measures;
    let bank_start = state.bank_start();
    match event {
        Event::PluginEncoderChange { index, delta } => {
            // Encoders are in Relative output mode: one detent = one
            // unit. Map that to a fraction of the slider's full range.
            let slider_index = *index as usize;
            let program = programs.get(active_program_index)?;
            let current = *program.sliders.normalized_values.get(slider_index)?;
            let normalized_delta = *delta as f32 / (ENCODER_ROTATIONS * 128.0);
            let normalized = (current + normalized_delta).clamp(0.0, 1.0);
            Some(vec![Action::SetSliderNormalized {
                program: active_program_index,
                slider_index,
                normalized,
            }])
        }
        Event::MixerEncoderChange { index, delta } => {
            let program_index = bank_start + *index as usize;
            let program = programs.get(program_index)?;
            // ~0.5 dB per detent; spans the -60..+6 range in roughly four full turns.
            let db_delta = *delta as f32 * 0.25;
            let level_db = (program.level_db + db_delta).clamp(-60.0, 6.0);
            Some(vec![Action::SetLevelDb {
                program: program_index,
                level_db,
            }])
        }
        Event::EncoderModeChanged(new_mode) => Some(vec![Action::SetEncoderMode(*new_mode)]),

        Event::NextTrackDown => Some(vec![Action::AdvanceProgram(1)]),
        Event::PreviousTrackDown => Some(vec![Action::AdvanceProgram(-1)]),
        Event::NextTrackBankDown => Some(vec![Action::AdvanceProgram(PROGRAMS_PER_BANK as i32)]),
        Event::PreviousTrackBankDown => {
            Some(vec![Action::AdvanceProgram(-(PROGRAMS_PER_BANK as i32))])
        }

        Event::DAWTopPadDown { index } => {
            let program_index = bank_start + *index as usize;
            let program = programs.get(program_index)?;
            let id = WaveformId::Program(program.id);
            let now = Instant::now();
            let is_installed_keys = keys.as_ref().is_some_and(|k| k.id == program.id);
            if status.has_active_mark(now, id.clone(), MarkId::TopLevel) {
                Some(vec![Action::StopProgram(program_index)])
            } else if is_installed_keys {
                Some(vec![]) // no-op
            } else {
                Some(vec![Action::PlayProgram {
                    program_index,
                    cursor_position: program.text.len(),
                    start_at_next_measure: false,
                    repeat_after_measures: None,
                    return_to_select_on_success: false,
                }])
            }
        }
        Event::DAWBottomPadDown { index } => {
            let program_index = bank_start + *index as usize;
            let program = programs.get(program_index)?;
            let id = WaveformId::Program(program.id);
            let now = Instant::now();
            let is_installed_keys = keys.as_ref().is_some_and(|k| k.id == program.id);
            if status.has_pending_mark(now, id.clone(), MarkId::TopLevel) {
                Some(vec![Action::RemovePendingProgram(program_index)])
            } else if is_installed_keys {
                Some(vec![]) // no-op
            } else {
                Some(vec![Action::PlayProgram {
                    program_index,
                    cursor_position: program.text.len(),
                    start_at_next_measure: true,
                    repeat_after_measures,
                    return_to_select_on_success: false,
                }])
            }
        }
        Event::PadFunctionDown => Some(vec![Action::CycleRepeatAfterMeasures]),

        Event::NoteOn { key, velocity } => Some(vec![Action::NoteOn {
            key: *key,
            velocity: *velocity,
        }]),
        Event::NoteOff { key } => Some(vec![Action::NoteOff { key: *key }]),

        // InstallKeys toggles based on current state: if keys are already
        // installed (for any program), the reducer uninstalls them;
        // otherwise it installs the active program as the keys instrument.
        Event::CaptureMIDIDown => Some(vec![Action::InstallKeys(active_program_index)]),
    }
}

/// Pushes the current app state out to the Launchkey hardware: pad colors
/// for active/pending program waveforms and the installed-keys program,
/// plus the pad-function button color reflecting `repeat_after_measures`.
///
/// `status` and the controller handle stay as separate args — they don't
/// live on `AppState`.
pub fn update_launchkey_state(
    state: &actions::AppState,
    status: &tracker::Status<WaveformId, MarkId>,
    launchkey: &mut launchkey::Launchkey,
) {
    // TODO update slider state

    match state.repeat_after_measures {
        None => {
            launchkey.set_pad_function_color(launchkey::Color::BrightGreen);
        }
        Some(1) => {
            launchkey.set_pad_function_color(launchkey::Color::YellowGreen);
        }
        Some(2) => {
            launchkey.set_pad_function_color(launchkey::Color::GoldenOrange);
        }
        i => {
            println!("unexpected repeat_after_measures: {:?}", i);
        }
    }

    let now = Instant::now();
    let (_current_beat, current_beat_start, current_beat_duration) =
        renderer::current_beat_info(now, status);
    let bank_start = state.bank_start();
    for (i, program) in state.programs[bank_start..bank_start + renderer::PROGRAMS_PER_BANK]
        .iter()
        .enumerate()
    {
        const U7_MAX: u8 = u8::MAX / 2;
        // 7-bit color values for the current program.
        let (red, blue, green) = match program.color {
            Some(color) => (color.0 / 2, color.1 / 2, color.2 / 2),
            None => (0, 127, 127),
        };
        // Top row is based on active waveforms
        if status.has_active_mark(now, WaveformId::Program(program.id), MarkId::TopLevel)
            || (if let Some(Keys { id, .. }) = &state.keys
                && *id == program.id
            {
                status
                    .marks
                    .iter()
                    .find(|m| {
                        if let WaveformId::Key(_) = m.waveform_id {
                            true
                        } else {
                            false
                        }
                    })
                    .is_some()
            } else {
                false
            })
        {
            // Based on time since beginning of the current beat.
            let intensity = U7_MAX.saturating_sub(
                (now.duration_since(current_beat_start)
                    .div_duration_f32(current_beat_duration)
                    * U7_MAX as f32) as u8,
            );
            launchkey.set_daw_top_pad_color(i as u8, 0, intensity, 0);
        } else if let Some(Keys { id, .. }) = &state.keys
            && *id == program.id
        {
            // If it's the installed keys program, don't color the top pad (unless it's playing).
            launchkey.set_daw_top_pad_color(i as u8, 0, 0, 0);
        } else if !program.text.is_empty() {
            launchkey.set_daw_top_pad_color(i as u8, red, blue, green);
        } else {
            // empty
            launchkey.set_daw_top_pad_color(i as u8, 0, 0, 0);
        }
        // Bottom row is based on pending waveforms
        if status.has_pending_mark(now, WaveformId::Program(program.id), MarkId::TopLevel) {
            launchkey.set_daw_bottom_pad_color(i as u8, 0, 127, 0);
        } else if let Some(Keys { id, .. }) = &state.keys
            && *id == program.id
        {
            // If it's the installed keys program, pulse the configured color.
            let intensity = now
                .duration_since(current_beat_start)
                .div_duration_f32(current_beat_duration);
            launchkey.set_daw_bottom_pad_color(
                i as u8,
                red.saturating_sub((intensity * red as f32) as u8),
                green.saturating_sub((intensity * green as f32) as u8),
                blue.saturating_sub((intensity * blue as f32) as u8),
            );
        } else if !program.text.is_empty() {
            launchkey.set_daw_bottom_pad_color(i as u8, red, blue, green);
        } else {
            // empty
            launchkey.set_daw_bottom_pad_color(i as u8, 0, 0, 0);
        }
    }
}
