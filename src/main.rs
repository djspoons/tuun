use std::collections::HashMap;
use std::fs;
use std::path;
use std::process;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use clap::Parser as ClapParser;
use sdl2::audio::AudioSpecDesired;
use sdl2::ttf::Sdl2TtfContext;

use tuun::actions;
use tuun::effects;
use tuun::generator;
use tuun::launchkey;
use tuun::metric;
use tuun::midi_input;
use tuun::parser;
use tuun::play_helper;
use tuun::renderer;
use tuun::sdl2_input;
use tuun::slider;
use tuun::tracker;
use tuun::waveform;

use metric::Metric;
use renderer::{MarkId, Renderer, WaveformId};
use tracker::Command;

#[derive(ClapParser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(long = "tempo", default_value_t = 90)]
    tempo: u32,
    #[arg(long = "beats_per_measure", default_value_t = 4)]
    beats_per_measure: u32,
    #[arg(long, default_value_t = 44100)]
    sample_rate: u32,
    #[arg(long, default_value_t = 1024)]
    buffer_size: u16,
    // Date format to use when saving captured waveforms
    #[arg(long, default_value = "_%Y-%m-%d_%H-%M-%S")]
    date_format: String,
    #[arg(long,
        num_args(0..=1), // Allows --flag or --flag=value
        action = clap::ArgAction::Set,
        default_value = "true", // Default if the flag is not present
        default_missing_value = "true")]
    precompute: bool,
    #[arg(long,
        num_args(0..=1), // Allows --flag or --flag=value
        action = clap::ArgAction::Set,
        default_value = "true", // Default if the flag is not present
        default_missing_value = "true")]
    ui: bool, // When set to value, just runs each of the programs once then exits
    /// Root directory for module resolution. A module path `["foo", "bar"]`
    /// resolves to `<library_root>/foo/bar.tuun`.
    #[arg(long, default_value = "./lib/v0")]
    library_root: path::PathBuf,
    input_file: String,
    #[arg(short = 'O', long, default_value = ".")]
    output_dir: String, // Captures waveforms to the specified directory
}

pub fn main() {
    let args = Args::parse();

    let (status_sender, status_receiver) = mpsc::channel();
    let (command_sender, command_receiver) = mpsc::channel();

    let input_path = path::PathBuf::from(&args.input_file);
    let input = fs::read_to_string(&input_path)
        .unwrap_or_else(|_| panic!("Failed to read input_file: {}", args.input_file));
    // TODO we don't want to ever fail here:
    let mut state = actions::AppState::from_source(input, input_path)
        .unwrap_or_else(|errors| panic!("Failed to parse {}: {:?}", args.input_file, errors));

    if !args.ui {
        println!("Starting in non-UI mode");
        // Filter out any empty programs
        state.programs.retain(|p| !p.text.is_empty());

        let mut tracker = tracker::Tracker::<WaveformId, MarkId>::new(
            args.sample_rate,
            args.output_dir.clone().into(),
            args.date_format.clone(),
            command_receiver,
            status_sender,
        );
        use sdl2::audio::AudioCallback;
        let mut out = vec![0.0f32; args.buffer_size as usize];

        // Call the callback to get at least one status update
        tracker.callback(&mut out);
        let status = status_receiver.recv().unwrap();
        // Parse and send commands to play all of the waveforms.
        let mut play_helper = play_helper::PlayHelper::new(
            args.sample_rate,
            args.tempo,
            args.beats_per_measure,
            args.library_root.clone(),
            command_sender.clone(),
            command_sender,
        );
        // Snapshot display names ahead of the mutable iteration below —
        // `program_display_name` borrows `state` immutably, which would
        // conflict with `state.programs.iter_mut()` if computed inline.
        let display_names: Vec<String> = (0..state.programs.len())
            .map(|i| actions::program_display_name(&state, i))
            .collect();
        for (program_index, program) in state.programs.iter_mut().enumerate() {
            let display_name = &display_names[program_index];
            println!("Playing program {}: {}", display_name, program.text);
            match play_helper.evaluate_program(&state.bindings, program) {
                Ok(expr) => match expr.expr {
                    // TODO also need to handle Seq here
                    parser::Expr::Waveform(w) => {
                        program.cached_waveform = Some(w);
                        play_helper.play_program_as_waveform(
                            program_index,
                            program,
                            display_name,
                            &status,
                            false,
                            None,
                        );
                    }
                    other => {
                        println!(
                            "Program {} did not evaluate to a waveform: {}",
                            display_name, other
                        );
                    }
                },
                Err(message) => {
                    println!("{}", message);
                    process::exit(1);
                }
            }
        }
        // Call the tracker's callback until all of the waveforms have finished playing
        loop {
            tracker.callback(&mut out);
            let status = status_receiver.recv().unwrap();
            let mark_count = status
                .marks
                .iter()
                .filter(|mark| mark.mark_id == MarkId::TopLevel)
                .count();
            if mark_count == 0 {
                println!("All waveforms finished");
                break;
            }
            println!("Still running, {} waveforms remaining", mark_count);
        }
        process::exit(0);
    }

    let sdl_context = sdl2::init().unwrap();
    let audio_subsystem = sdl_context.audio().unwrap();
    let desired_spec = AudioSpecDesired {
        freq: Some(args.sample_rate as i32),
        channels: Some(1), // mono
        samples: Some(args.buffer_size),
    };
    let device = audio_subsystem
        .open_playback(None, &desired_spec, |spec| {
            println!("Spec: {:?}", spec);
            tracker::Tracker::<WaveformId, MarkId>::new(
                args.sample_rate,
                args.output_dir.clone().into(),
                args.date_format.clone(),
                command_receiver,
                status_sender,
            )
        })
        .unwrap();
    device.resume();

    // Spin up the precompute thread so PlayHelpers can route their
    // commands through it. The thread pre-computes `Command::Play` and
    // passes everything else through unchanged.
    //
    // What goes through (the two PlayHelpers wired to the returned
    // sender below: `start_beats` and `effect_play_helper`):
    //   - play_program_as_waveform with start_at_next_measure = true
    //
    // What bypasses entirely (uses `command_sender` directly):
    //   - The slider thread's Modify ramps
    //   - The effect runner's ModifyWaveform / ModifyActiveKeysAmplitude /
    //     RemovePending
    //   - Effect::PlayNoteOn — yes, this is a Command::Play but it
    //     deliberately skips precompute to keep keystroke latency low
    //   - play_helper's use of `fast_command_sender`
    let precomputing_command_sender = {
        let play_command_sender = command_sender.clone();
        let (tx, play_receiver) = mpsc::channel();
        let precompute = args.precompute;
        let sample_rate = args.sample_rate;
        thread::spawn(move || {
            loop {
                match play_receiver.recv() {
                    Ok(Command::Play {
                        id,
                        mut waveform,
                        start,
                        repeat_every,
                    }) => {
                        if precompute {
                            // TODO probably precompute should happen on another thread
                            let mut generator = generator::Generator::new(sample_rate);
                            waveform = waveform::remove_state(generator.precompute(waveform));
                            println!("precompute returned: {}", &waveform);
                        }

                        play_command_sender
                            .send(Command::Play {
                                id,
                                waveform,
                                start,
                                repeat_every,
                            })
                            .unwrap();
                    }
                    Ok(cmd) => {
                        play_command_sender.send(cmd).unwrap();
                    }
                    Err(e) => {
                        println!("Play thread got error receiving: {}", e);
                        return;
                    }
                }
            }
        });
        tx
    };

    // Start the beats!
    play_helper::PlayHelper::new(
        args.sample_rate,
        args.tempo,
        args.beats_per_measure,
        args.library_root.clone(),
        precomputing_command_sender.clone(),
        command_sender.clone(),
    )
    .start_beats(&status_receiver);

    // Copy initial values for each slider for all programs
    let mut last_slider_values: HashMap<(WaveformId, String), f32> = HashMap::new();
    for (program_index, program) in state.programs.iter().enumerate() {
        for (j, config) in program.sliders.configs.iter().enumerate() {
            let value = slider::denormalize(&config.function, program.sliders.normalized_values[j])
                .unwrap_or(0.0);
            last_slider_values.insert(
                (WaveformId::Program(program_index), config.label.clone()),
                value,
            );
        }
    }
    // Spawn a thread that batches slider updates, sending them approximately once per audio buffer
    let buffer_duration =
        Duration::from_secs_f32(args.buffer_size as f32 / args.sample_rate as f32);
    let (slider_sender, slider_receiver) = mpsc::channel::<renderer::SliderEvent>();
    let slider_command_sender = command_sender.clone();
    thread::spawn(move || {
        let mut last_slider_values = last_slider_values;
        loop {
            let mut pending: HashMap<(WaveformId, String), f32> = HashMap::new();
            let deadline;

            // Idle: block until the first slider update event arrives.
            loop {
                use renderer::SliderEvent;
                match slider_receiver.recv() {
                    Ok(SliderEvent::UpdateSlider { id, slider, value }) => {
                        pending.insert((id, slider), value);
                        deadline = Instant::now() + buffer_duration;
                        break;
                    }
                    Ok(SliderEvent::SetInitialValues(values)) => {
                        // This occurs when the set of programs is reloaded.
                        last_slider_values = values;
                    }
                    Ok(SliderEvent::UpdateInitialValues(values)) => {
                        values.into_iter().for_each(|(k, v)| {
                            last_slider_values.insert(k, v);
                        });
                    }
                    Err(_) => return, // channel closed, exit thread
                };
            }

            // Accumulate: keep receiving events until the deadline.
            loop {
                let remaining = deadline.saturating_duration_since(Instant::now());
                if remaining.is_zero() {
                    break;
                }
                use renderer::SliderEvent;
                match slider_receiver.recv_timeout(remaining) {
                    Ok(SliderEvent::UpdateSlider { id, slider, value }) => {
                        pending.insert((id, slider), value);
                    }
                    Ok(SliderEvent::SetInitialValues(values)) => {
                        // What to do with these new values for waveforms that are playing? Especially ones
                        // where the sliders are moving? Maybe think of something better in the future once
                        // we understand better happens to currently playing waveforms.
                        last_slider_values = values;
                    }
                    Ok(SliderEvent::UpdateInitialValues(values)) => {
                        values.into_iter().for_each(|(k, v)| {
                            last_slider_values.insert(k, v);
                        });
                    }
                    Err(mpsc::RecvTimeoutError::Timeout) => break,
                    Err(mpsc::RecvTimeoutError::Disconnected) => return,
                }
            }

            // Flush: send one Modify command per changed slider
            for ((id, slider), value) in pending.drain() {
                // TODO: this unwrap can fail if programs are reloaded while a MIDI note is active.
                // Should we just default to some value? Or maybe better to not to clear?
                let last_value = last_slider_values
                    .get_mut(&(id.clone(), slider.clone()))
                    .unwrap();

                let waveform = slider::make_ramp(*last_value, value, buffer_duration.as_secs_f32());

                // Update the last value
                *last_value = value;

                // Technically we don't need to send a modify if the waveform is not playing...
                let _ = slider_command_sender.send(Command::Modify {
                    id,
                    mark_id: MarkId::Slider(slider),
                    waveform,
                });
            }
        }
    });

    let launchkey = launchkey::Launchkey::new();
    if let Err(e) = &launchkey {
        println!(
            "Couldn't find Launchkey... falling back on mouse slider controls ({})",
            e
        );
    }

    let ttf_context: Sdl2TtfContext = sdl2::ttf::init().unwrap();
    let mut renderer = Renderer::new(
        &sdl_context,
        &ttf_context,
        args.tempo,
        args.beats_per_measure,
        args.sample_rate,
    );
    renderer.video_subsystem.text_input().start();
    let mut event_pump = sdl_context.event_pump().unwrap();

    let sdl2_handler =
        sdl2_input::InputHandler::new(launchkey.is_err(), renderer.width, renderer.height);

    let mut launchkey = launchkey.ok();
    let mut effect_runner =
        effects::EffectRunner::new(command_sender.clone(), slider_sender.clone());
    // A PlayHelper owned by the effect runner's "World" — used to dispatch
    // Effect::PlayProgram and friends without going through the per-handler
    // PlayHelpers. Routes through the precompute thread so `--precompute`
    // actually takes effect on interactive playback.
    let mut effect_play_helper = play_helper::PlayHelper::new(
        args.sample_rate,
        args.tempo,
        args.beats_per_measure,
        args.library_root.clone(),
        precomputing_command_sender,
        command_sender.clone(),
    );

    let mut status = tracker::Status {
        buffer_start: Instant::now(),
        marks: Vec::new(),
        buffer: None,
        tracker_load: None,
        allocations_per_sample: None,
    };
    let mut metrics = renderer::Metrics {
        tracker_load: Metric::new(Duration::from_secs(10), 100),
        allocations_per_sample: Metric::new(Duration::from_secs(10), 100),
    };

    // Populate each program's cached_waveform / cached_keys_instrument from
    // its initial source so that playback and InstallKeys work without
    // needing to enter and exit Edit mode first.
    for i in 0..state.programs.len() {
        if state.programs[i].text.is_empty() {
            continue;
        }
        run_effects(
            &mut effect_runner,
            &mut state,
            launchkey.as_mut(),
            &status,
            &mut effect_play_helper,
            vec![actions::Effect::EvaluateProgram {
                program_index: i,
                mode_on_failure: renderer::Mode::Select,
            }],
        );
    }

    const BUFFER_REFRESH_INTERVAL: Duration = Duration::from_millis(200);
    let mut next_buffer_refresh = Instant::now();
    command_sender.send(Command::SendCurrentBuffer).unwrap();
    // Push the initial encoder + DAW-mode display state to the controller
    // if present.
    {
        let daw_mode_label = state.daw_pad_mode.display_name().to_string();
        run_effects(
            &mut effect_runner,
            &mut state,
            launchkey.as_mut(),
            &status,
            &mut effect_play_helper,
            vec![
                actions::Effect::SyncEncoders,
                actions::Effect::SetDawModeDisplay(daw_mode_label),
            ],
        );
    }
    loop {
        if state.should_exit {
            return;
        }
        for event in event_pump.poll_iter() {
            let sdl_actions = sdl2_handler.classify(&event, &state);
            if let Some(actions) = sdl_actions {
                dispatch_actions(
                    &mut effect_runner,
                    &mut state,
                    launchkey.as_mut(),
                    &status,
                    &mut effect_play_helper,
                    actions,
                );
            } else {
                // classify() should cover every SDL event we care about.
                println!("Unhandled SDL event: {:?}", event);
            }
        }

        if let Some(launchkey) = launchkey.as_mut() {
            loop {
                match launchkey.events.try_recv() {
                    Ok(event) => {
                        let actions = midi_input::classify(&event, &state);
                        if let Some(actions) = actions {
                            dispatch_actions(
                                &mut effect_runner,
                                &mut state,
                                Some(&mut *launchkey),
                                &status,
                                &mut effect_play_helper,
                                actions,
                            );
                        } else {
                            // classify() should be exhaustive for launchkey
                            // events.
                            println!("Unhandled launchkey event: {:?}", event);
                        }
                    }
                    Err(mpsc::TryRecvError::Empty) => {
                        break;
                    }
                    Err(e) => {
                        println!("Got error receiving from Launchkey: {}", e);
                        break;
                    }
                }
            }
        }

        if next_buffer_refresh <= Instant::now() {
            command_sender.send(Command::SendCurrentBuffer).unwrap();
            next_buffer_refresh = Instant::now() + BUFFER_REFRESH_INTERVAL;
        }

        // Drain the status buffer, using the last status received. But don't wait more too long
        // if there are no statuses available. (And don't wait at all if we've already received one).
        let mut statuses_received = 0;
        loop {
            match status_receiver.try_recv() {
                Ok(tracker_status) => {
                    apply_status(tracker_status, &mut metrics, &mut status);
                    statuses_received += 1;
                }
                Err(mpsc::TryRecvError::Empty) => {
                    if statuses_received > 0 {
                        break;
                    }
                    match status_receiver.recv_timeout(Duration::from_millis(10)) {
                        Ok(tracker_status) => {
                            apply_status(tracker_status, &mut metrics, &mut status);
                            statuses_received += 1;
                        }
                        Err(mpsc::RecvTimeoutError::Timeout) => break,
                        Err(e) => println!("Error receiving status with timeout: {:?}", e),
                    }
                }
                Err(e) => println!("Error receiving status: {:?}", e),
            }
        }
        renderer.render(
            &ttf_context,
            &state,
            &status,
            &mut metrics,
            launchkey.as_ref().map(|l| l.encoder_mode),
        );

        if let Some(launchkey) = launchkey.as_mut() {
            midi_input::update_launchkey_state(&state, &status, launchkey);
        }
    }
}

/// Runs the `effects` against a `World` built from the given arguments.
fn run_effects(
    runner: &mut effects::EffectRunner,
    state: &mut actions::AppState,
    launchkey: Option<&mut launchkey::Launchkey>,
    status: &tracker::Status<WaveformId, MarkId>,
    play_helper: &mut play_helper::PlayHelper,
    effects: Vec<actions::Effect>,
) {
    let mut world = effects::World {
        launchkey,
        status,
        play_helper,
    };
    runner.run_all(state, &mut world, effects);
}

/// Dispatches `actions` through the reducer against a `World` built from the
/// given arguments.
fn dispatch_actions(
    runner: &mut effects::EffectRunner,
    state: &mut actions::AppState,
    launchkey: Option<&mut launchkey::Launchkey>,
    status: &tracker::Status<WaveformId, MarkId>,
    play_helper: &mut play_helper::PlayHelper,
    actions: Vec<actions::Action>,
) {
    let mut world = effects::World {
        launchkey,
        status,
        play_helper,
    };
    runner.dispatch(state, &mut world, actions);
}

/// Applies a freshly received tracker status: folds its metrics into
/// `metrics` and replaces `status` with it.
fn apply_status(
    new_status: tracker::Status<WaveformId, MarkId>,
    metrics: &mut renderer::Metrics,
    status: &mut tracker::Status<WaveformId, MarkId>,
) {
    if let Some(ratio) = new_status.tracker_load {
        metrics.tracker_load.set(ratio);
    }
    if let Some(allocations) = new_status.allocations_per_sample {
        metrics.allocations_per_sample.set(allocations);
    }
    *status = new_status;
}
