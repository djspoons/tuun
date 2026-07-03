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
    /// 0-based index into `AppState.programs` of the program installed as
    /// the keys instrument.
    pub id: usize,
    /// A function which takes a pair (MIDI note, velocity) and returns a pair
    /// of Waveforms to be used for note-on and note-off events.
    ///
    /// Should be a closed value except for references to sliders.
    pub function: parser::SourceExpr<MarkId>,
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
) -> Option<Vec<actions::Action>> {
    use actions::Action;
    use launchkey::Event;
    let active_program_index = state.active_program_index;
    let programs: &[Program] = &state.programs;
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

        Event::DAWTopPadDown { index } => match state.daw_pad_mode {
            actions::DawPadMode::ClipLauncher => {
                let program_index = bank_start + *index as usize;
                programs.get(program_index)?;
                Some(vec![Action::ToggleProgramPlayback(program_index)])
            }
            // Top row does nothing in the keys-installer mode.
            actions::DawPadMode::KeysInstaller => Some(vec![]),
        },
        Event::DAWBottomPadDown { index } => {
            let program_index = bank_start + *index as usize;
            programs.get(program_index)?;
            match state.daw_pad_mode {
                actions::DawPadMode::ClipLauncher => {
                    Some(vec![Action::ToggleProgramPendingPlayback(program_index)])
                }
                actions::DawPadMode::KeysInstaller => {
                    Some(vec![Action::ToggleInstalledKeys(program_index)])
                }
            }
        }
        Event::PadFunctionDown => Some(vec![Action::CycleRepeatAfterMeasures]),

        Event::NoteOn { key, velocity } => Some(vec![Action::NoteOn {
            key: *key,
            velocity: *velocity,
        }]),
        Event::NoteOff { key } => Some(vec![Action::NoteOff { key: *key }]),

        Event::PadModeChanged { previous, current } => Some(vec![Action::PadModeChanged {
            previous: *previous,
            current: *current,
        }]),
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
    if launchkey.pad_mode != launchkey::PadMode::DAW {
        // Some other pad layout (Drum, Custom, etc.) owns the pads —
        // leave the LEDs alone so we don't fight it.
        return;
    }
    match state.daw_pad_mode {
        actions::DawPadMode::ClipLauncher => {
            update_pads_clip_launcher(
                state,
                status,
                launchkey,
                now,
                current_beat_start,
                current_beat_duration,
                bank_start,
            );
        }
        actions::DawPadMode::KeysInstaller => {
            update_pads_keys_installer(
                state,
                launchkey,
                now,
                current_beat_start,
                current_beat_duration,
                bank_start,
            );
        }
    }
}

/// The maximum 7-bit color channel value the pads accept.
const U7_MAX: u8 = u8::MAX / 2;

/// Returns the 7-bit (red, green, blue) pad color for `program`: its configured
/// color at half intensity or a cyan default when none is set.
fn program_pad_color(program: &Program) -> (u8, u8, u8) {
    match program.color {
        Some((r, g, b)) => (r / 2, g / 2, b / 2),
        None => (0, 127, 127),
    }
}

/// Fades `color` toward black over the current beat: full intensity at the beat
/// start, darkening as the beat progresses.
fn pulsed(
    color: (u8, u8, u8),
    now: Instant,
    beat_start: Instant,
    beat_duration: std::time::Duration,
) -> (u8, u8, u8) {
    let fraction = now
        .duration_since(beat_start)
        .div_duration_f32(beat_duration);
    let dim = |channel: u8| channel.saturating_sub((fraction * channel as f32) as u8);
    (dim(color.0), dim(color.1), dim(color.2))
}

fn update_pads_clip_launcher(
    state: &actions::AppState,
    status: &tracker::Status<WaveformId, MarkId>,
    launchkey: &mut launchkey::Launchkey,
    now: Instant,
    current_beat_start: Instant,
    current_beat_duration: std::time::Duration,
    bank_start: usize,
) {
    for (i, program) in state.programs[bank_start..bank_start + renderer::PROGRAMS_PER_BANK]
        .iter()
        .enumerate()
    {
        let program_index = bank_start + i;
        let (red, green, blue) = program_pad_color(program);
        let is_installed_keys = state.keys.as_ref().is_some_and(|k| k.id == program_index);
        // Top row is based on active waveforms
        if status.has_active_mark(now, WaveformId::Program(program_index), MarkId::TopLevel)
            || (is_installed_keys
                && status
                    .marks
                    .iter()
                    .any(|m| matches!(m.waveform_id, WaveformId::Key(_))))
        {
            let (r, g, b) = pulsed(
                (0, U7_MAX, 0),
                now,
                current_beat_start,
                current_beat_duration,
            );
            launchkey.set_daw_top_pad_color(i as u8, r, g, b);
        } else if is_installed_keys {
            // If it's the installed keys program, don't color the top pad (unless it's playing).
            launchkey.set_daw_top_pad_color(i as u8, 0, 0, 0);
        } else if program.cached_waveform.is_some() {
            launchkey.set_daw_top_pad_color(i as u8, red, green, blue);
        } else {
            // empty
            launchkey.set_daw_top_pad_color(i as u8, 0, 0, 0);
        }
        // Bottom row is based on pending waveforms
        if status.has_pending_mark(now, WaveformId::Program(program_index), MarkId::TopLevel) {
            launchkey.set_daw_bottom_pad_color(i as u8, 0, 127, 0);
        } else if is_installed_keys {
            // If it's the installed keys program, pulse the configured color.
            let (r, g, b) = pulsed(
                (red, green, blue),
                now,
                current_beat_start,
                current_beat_duration,
            );
            launchkey.set_daw_bottom_pad_color(i as u8, r, g, b);
        } else if program.cached_waveform.is_some() {
            launchkey.set_daw_bottom_pad_color(i as u8, red, green, blue);
        } else {
            // empty
            launchkey.set_daw_bottom_pad_color(i as u8, 0, 0, 0);
        }
    }
}

/// Updates the controller state for keys-installer mode.
fn update_pads_keys_installer(
    state: &actions::AppState,
    launchkey: &mut launchkey::Launchkey,
    now: Instant,
    current_beat_start: Instant,
    current_beat_duration: std::time::Duration,
    bank_start: usize,
) {
    for i in 0..renderer::PROGRAMS_PER_BANK {
        launchkey.set_daw_top_pad_color(i as u8, 0, 0, 0);

        let program_index = bank_start + i;
        let program = match state.programs.get(program_index) {
            Some(p) => p,
            None => {
                launchkey.set_daw_bottom_pad_color(i as u8, 0, 0, 0);
                continue;
            }
        };
        let installed = state.keys.as_ref().is_some_and(|k| k.id == program_index);
        // If this program is the installed keys instrument, light it up
        // regardless of whether the current text is still a valid keys
        // program — the installed function is what's actually playing.
        // Otherwise only color pads that can be installed right now.
        if !installed && program.cached_keys_instrument.is_none() {
            launchkey.set_daw_bottom_pad_color(i as u8, 0, 0, 0);
            continue;
        }
        let (red, green, blue) = program_pad_color(program);
        if installed {
            let (r, g, b) = pulsed(
                (red, green, blue),
                now,
                current_beat_start,
                current_beat_duration,
            );
            launchkey.set_daw_bottom_pad_color(i as u8, r, g, b);
        } else {
            launchkey.set_daw_bottom_pad_color(i as u8, red, green, blue);
        }
    }
}
