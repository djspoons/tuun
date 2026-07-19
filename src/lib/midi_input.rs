use std::time::Instant;

use crate::actions;
use crate::ids::{MarkId, WaveformId, WaveformSelector};
use crate::launchkey;
use crate::player;
use crate::programs::{PROGRAMS_PER_BANK, Program, ProgramKind};
use crate::renderer;
use crate::sequencer;
use crate::tracker;

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
    let programs: &[Program] = state.programs.programs();
    let bank_start = state.bank_start();
    match event {
        Event::PluginEncoderChange { index, delta } => {
            // Encoders are in Relative output mode: one detent = one
            // unit. Map that to a fraction of the slider's full range.
            let slider_index = *index as usize;
            let program = programs.get(active_program_index)?;
            let current = *program.sliders().normalized_values().get(slider_index)?;
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
            let level_db = (program.level_db() + db_delta).clamp(-60.0, 6.0);
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

        // Transport buttons act on the active program like the clip launcher's
        // bottom pads: play queues at the next measure, stop removes the queued
        // playback (which for a looping program ends the loop after the current
        // cycle).
        Event::PlayDown => Some(vec![Action::EnqueuePendingPlayback(active_program_index)]),
        Event::StopDown => Some(vec![Action::RemovePendingProgram(active_program_index)]),

        // The pad-navigation arrows page the sequencer grid through the
        // pattern's measures: down moves toward later beats.
        Event::PadPageUpDown => match state.daw_pad_mode {
            actions::DawPadMode::Sequencer => Some(vec![Action::ChangeSequencerPage(-1)]),
            _ => Some(vec![]),
        },
        Event::PadPageDownDown => match state.daw_pad_mode {
            actions::DawPadMode::Sequencer => Some(vec![Action::ChangeSequencerPage(1)]),
            _ => Some(vec![]),
        },
        Event::DAWTopPadDown { index } => match state.daw_pad_mode {
            actions::DawPadMode::ClipLauncher => {
                let program_index = bank_start + *index as usize;
                programs.get(program_index)?;
                Some(vec![Action::ToggleProgramPlayback(program_index)])
            }
            // Top row does nothing in the keys-installer mode.
            actions::DawPadMode::KeysInstaller => Some(vec![]),
            // The top row is the first half-measure of the visible page.
            actions::DawPadMode::Sequencer => Some(vec![Action::ToggleSequencerStep {
                sixteenth: state.sequencer_page * sequencer::SIXTEENTHS_PER_PAGE + *index,
            }]),
        },
        Event::DAWBottomPadDown { index } => match state.daw_pad_mode {
            actions::DawPadMode::ClipLauncher => {
                let program_index = bank_start + *index as usize;
                programs.get(program_index)?;
                Some(vec![Action::ToggleProgramPendingPlayback(program_index)])
            }
            actions::DawPadMode::KeysInstaller => {
                let program_index = bank_start + *index as usize;
                programs.get(program_index)?;
                Some(vec![Action::ToggleInstalledKeys(program_index)])
            }
            // The bottom row is the second half-measure of the visible page.
            actions::DawPadMode::Sequencer => Some(vec![Action::ToggleSequencerStep {
                sixteenth: state.sequencer_page * sequencer::SIXTEENTHS_PER_PAGE
                    + PROGRAMS_PER_BANK as u8
                    + *index,
            }]),
        },
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
        actions::DawPadMode::Sequencer => {
            update_pads_sequencer(state, status, launchkey, now, current_beat_duration);
        }
    }
}

/// The maximum 7-bit color channel value the pads accept.
const U7_MAX: u8 = u8::MAX / 2;

/// Returns the 7-bit (red, green, blue) pad color for `program`: its configured
/// color at half intensity or a cyan default when none is set.
fn program_pad_color(program: &Program) -> (u8, u8, u8) {
    match program.color() {
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
    for (i, program) in state.programs.programs()[bank_start..bank_start + PROGRAMS_PER_BANK]
        .iter()
        .enumerate()
    {
        let program_index = bank_start + i;
        let (red, green, blue) = program_pad_color(program);
        let is_installed_keys = state.keys.as_ref().is_some_and(|k| k.id == program_index);
        // Top row is based on active waveforms
        if status.has_active_mark(
            now,
            &WaveformSelector::ProgramVoices(program_index),
            &MarkId::TopLevel,
        ) || (is_installed_keys
            && status.has_active_mark(now, &WaveformSelector::AllKeys, &MarkId::TopLevel))
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
        } else if program.waveform().is_some() {
            launchkey.set_daw_top_pad_color(i as u8, red, green, blue);
        } else {
            // empty
            launchkey.set_daw_top_pad_color(i as u8, 0, 0, 0);
        }
        // Bottom row is based on pending waveforms
        if status.has_pending_mark(
            now,
            &WaveformSelector::ProgramVoices(program_index),
            &MarkId::TopLevel,
        ) {
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
        } else if program.waveform().is_some() {
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
    for i in 0..PROGRAMS_PER_BANK {
        launchkey.set_daw_top_pad_color(i as u8, 0, 0, 0);

        let program_index = bank_start + i;
        let program = match state.programs.program(program_index) {
            Some(p) => p,
            None => {
                launchkey.set_daw_bottom_pad_color(i as u8, 0, 0, 0);
                continue;
            }
        };
        let installed = state.keys.as_ref().is_some_and(|k| k.id == program_index);
        let (red, green, blue) = program_pad_color(program);
        if installed {
            // The installed keys instrument pulses regardless of whether
            // the current text is still valid — the installed function is
            // what's actually playing.
            let (r, g, b) = pulsed(
                (red, green, blue),
                now,
                current_beat_start,
                current_beat_duration,
            );
            launchkey.set_daw_bottom_pad_color(i as u8, r, g, b);
            continue;
        }
        match program.kind() {
            // Installable right now: full color.
            ProgramKind::Keys if program.keys_instrument().is_some() => {
                launchkey.set_daw_bottom_pad_color(i as u8, red, green, blue);
            }
            // Marked as a keys slot but currently invalid: dim, so the
            // slot still reads as a keys slot while its text is broken.
            ProgramKind::Keys => {
                launchkey.set_daw_bottom_pad_color(i as u8, red / 4, green / 4, blue / 4);
            }
            ProgramKind::Waveform => {
                launchkey.set_daw_bottom_pad_color(i as u8, 0, 0, 0);
            }
        }
    }
}

/// Updates the controller state for sequencer mode: the 16 pads show the active
/// program's step grid (top row is the first eight sixteenths, bottom row is
/// the second eight sixteenths; as determined by state.sequencer_page), with a
/// white playhead pulse on the current sixteenth while the program is playing.
/// All pads are dark when the active program isn't sequenceable.
fn update_pads_sequencer(
    state: &actions::AppState,
    status: &tracker::Status<WaveformId, MarkId>,
    launchkey: &mut launchkey::Launchkey,
    now: Instant,
    current_beat_duration: std::time::Duration,
) {
    let program = state.active_program();
    let beats: Option<Vec<f32>> = program
        .sequence()
        .map(|sequence| sequence.beats.clone())
        .or_else(|| {
            // A program that was edited (but not yet re-evaluated) or that
            // isn't playable can still show its pattern from the text.
            sequencer::analyze(program.text())
                .map(|shape| shape.beats.iter().map(|(beat, _)| *beat).collect())
        });

    // The playhead only shows while the step family is sounding: its
    // absolute sixteenth is measured from the current cycle's anchor start,
    // so it lands on the right page of a multi-measure pattern.
    let anchor = WaveformId::Step {
        program: state.active_program_index,
        sixteenth: player::ANCHOR_SIXTEENTH,
    };
    let mut cycle_start: Option<Instant> = None;
    for mark in &status.marks {
        if mark.waveform_id == anchor && mark.mark_id == MarkId::TopLevel && mark.start <= now {
            cycle_start = Some(cycle_start.map_or(mark.start, |s| s.max(mark.start)));
        }
    }
    let playhead = cycle_start.map(|cycle_start| {
        let sixteenth_duration = current_beat_duration / 4;
        let position = now
            .duration_since(cycle_start)
            .div_duration_f32(sixteenth_duration) as u32;
        (
            position,
            cycle_start + sixteenth_duration * position,
            sixteenth_duration,
        )
    });

    let page_base = state.sequencer_page * sequencer::SIXTEENTHS_PER_PAGE;
    for pad in 0..(2 * PROGRAMS_PER_BANK as u8) {
        let sixteenth = page_base + pad;
        let color = match &beats {
            None => (0, 0, 0),
            Some(beats) => match sequencer::sixteenth_state(beats, sixteenth) {
                sequencer::SixteenthState::Empty => (0, 0, 0),
                sequencer::SixteenthState::OnGrid => program_pad_color(program),
                sequencer::SixteenthState::OffGridOnly => {
                    let (r, g, b) = program_pad_color(program);
                    (r / 4, g / 4, b / 4)
                }
            },
        };
        let color = match playhead {
            Some((position, sixteenth_start, sixteenth_duration))
                if position == sixteenth as u32 =>
            {
                pulsed(
                    (U7_MAX, U7_MAX, U7_MAX),
                    now,
                    sixteenth_start,
                    sixteenth_duration,
                )
            }
            _ => color,
        };
        if pad < PROGRAMS_PER_BANK as u8 {
            launchkey.set_daw_top_pad_color(pad, color.0, color.1, color.2);
        } else {
            launchkey.set_daw_bottom_pad_color(
                pad - PROGRAMS_PER_BANK as u8,
                color.0,
                color.1,
                color.2,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::actions::{Action, AppState, DawPadMode};

    use super::*;

    fn test_state(daw_pad_mode: DawPadMode) -> AppState {
        let mut state = AppState::from_source(
            "#{level_db=0}\n_ = 1 | fin(time - 1);\n".to_string(),
            std::path::PathBuf::new(),
        )
        .expect("test source should parse");
        state.daw_pad_mode = daw_pad_mode;
        state
    }

    #[test]
    fn classify_sequencer_top_pad_toggles_sixteenths_0_to_7() {
        let state = test_state(DawPadMode::Sequencer);
        for index in 0..8u8 {
            let actions = classify(&launchkey::Event::DAWTopPadDown { index }, &state)
                .expect("top pads classify in sequencer mode");
            assert_eq!(actions.len(), 1);
            assert!(matches!(
                actions[0],
                Action::ToggleSequencerStep { sixteenth } if sixteenth == index
            ));
        }
    }

    #[test]
    fn classify_sequencer_pads_respect_the_page() {
        let mut state = test_state(DawPadMode::Sequencer);
        state.sequencer_page = 1;
        let actions = classify(&launchkey::Event::DAWTopPadDown { index: 2 }, &state)
            .expect("top pads classify in sequencer mode");
        assert!(matches!(
            actions[0],
            Action::ToggleSequencerStep { sixteenth: 18 }
        ));
        let actions = classify(&launchkey::Event::DAWBottomPadDown { index: 2 }, &state)
            .expect("bottom pads classify in sequencer mode");
        assert!(matches!(
            actions[0],
            Action::ToggleSequencerStep { sixteenth: 26 }
        ));
    }

    #[test]
    fn classify_page_buttons_only_page_in_sequencer_mode() {
        let state = test_state(DawPadMode::Sequencer);
        let actions = classify(&launchkey::Event::PadPageDownDown, &state)
            .expect("page buttons classify in sequencer mode");
        assert!(matches!(actions[0], Action::ChangeSequencerPage(1)));
        let actions = classify(&launchkey::Event::PadPageUpDown, &state)
            .expect("page buttons classify in sequencer mode");
        assert!(matches!(actions[0], Action::ChangeSequencerPage(-1)));

        let state = test_state(DawPadMode::ClipLauncher);
        let actions = classify(&launchkey::Event::PadPageDownDown, &state)
            .expect("page buttons are swallowed outside sequencer mode");
        assert!(actions.is_empty());
    }

    #[test]
    fn classify_transport_buttons_target_active_program() {
        let state = test_state(DawPadMode::ClipLauncher);
        let actions =
            classify(&launchkey::Event::PlayDown, &state).expect("play button classifies");
        assert!(matches!(actions[0], Action::EnqueuePendingPlayback(0)));
        let actions =
            classify(&launchkey::Event::StopDown, &state).expect("stop button classifies");
        assert!(matches!(actions[0], Action::RemovePendingProgram(0)));
    }

    #[test]
    fn classify_sequencer_bottom_pad_toggles_sixteenths_8_to_15() {
        let state = test_state(DawPadMode::Sequencer);
        for index in 0..8u8 {
            let actions = classify(&launchkey::Event::DAWBottomPadDown { index }, &state)
                .expect("bottom pads classify in sequencer mode");
            assert_eq!(actions.len(), 1);
            assert!(matches!(
                actions[0],
                Action::ToggleSequencerStep { sixteenth } if sixteenth == index + 8
            ));
        }
    }
}
