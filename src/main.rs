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
use tuun::diagnostics;
use tuun::effects;
use tuun::environment;
use tuun::launchkey;
use tuun::metric;
use tuun::midi_input;
use tuun::player;
use tuun::processing;
use tuun::renderer;
use tuun::scope;
use tuun::sdl2_input;
use tuun::tracker;

use metric::Metric;
use renderer::Renderer;
use tuun::ids::{MarkId, WaveformId};

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
    // Formats the prefix shared by the name of every waveform captured during a run. Applied
    // once, at startup. Two runs starting within the same second share a prefix and overwrite
    // each other's captures, so include something like a process id when launching in a loop.
    #[arg(long, default_value = "tuun_%Y-%m-%d_%H-%M-%S")]
    date_format: String,
    /// Whether the finite parts of each evaluated waveform (and of the Beats
    /// waveforms) are replaced by their samples before they are played.
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
    /// What type-checker findings become: errors or warnings.
    #[arg(long, default_value_t = environment::CheckMode::Errors)]
    check: environment::CheckMode,
    input_file: String,
    #[arg(short = 'O', long, default_value = ".")]
    output_dir: String, // Captures waveforms to the specified directory
    /// Which scope view the screen starts in: snapshot or scan. Ctrl+S
    /// switches between them.
    #[arg(long, default_value_t = scope::ScopeView::Snapshot)]
    scope_view: scope::ScopeView,
    /// How often, in milliseconds, the Snapshot scope view and the spectrum
    /// redraw.
    #[arg(long, default_value_t = 100)]
    snapshot_refresh_ms: u64,
}

/// The number of sample headers the tracker's header ring holds: far more
/// callbacks than the sample ring, which holds one second of samples, has room for.
const SAMPLE_HEADER_CAPACITY: usize = 1024;

pub fn main() {
    let args = Args::parse();
    let captured_file_prefix = chrono::Local::now().format(&args.date_format).to_string();

    let (status_sender, status_receiver) = mpsc::channel();
    let (command_sender, command_receiver) = mpsc::channel();

    let input_path = path::PathBuf::from(&args.input_file);
    let input = fs::read_to_string(&input_path)
        .unwrap_or_else(|_| panic!("Failed to read input_file: {}", args.input_file));
    // TODO we don't want to ever fail here:
    let mut state = match actions::AppState::from_source(input.clone(), input_path) {
        Ok(state) => state,
        Err(errors) => {
            for error in &errors {
                eprintln!("{}:{}", args.input_file, error.display_with_source(&input));
                if let Some(range) = error.range() {
                    eprintln!("{}", diagnostics::render_snippet(&input, &range));
                }
            }
            process::exit(1);
        }
    };
    state.scope_view = args.scope_view;

    if !args.ui {
        println!("Starting in non-UI mode");
        // Recoverable parse problems don't stop startup, but in batch mode
        // there is no status line to show the warning on — print it.
        if !state.message.is_empty() {
            println!("{}", state.message);
        }

        // Nothing drains the samples in batch mode; once the rings fill, the
        // tracker just counts its pushes as dropped.
        let (sample_producer, _sample_consumer) =
            tracker::sample_rings(args.sample_rate as usize, SAMPLE_HEADER_CAPACITY);
        // TODO the tracker clock resync doesn't really make sense for batch
        // mode... maybe we shouldn't try to resync in this case?
        let mut tracker = tracker::Tracker::<WaveformId, MarkId>::new(
            args.sample_rate,
            args.output_dir.clone().into(),
            captured_file_prefix.clone(),
            command_receiver,
            status_sender,
            sample_producer,
            // No audio device plays these samples, so just use one buffer.
            Duration::from_secs_f32(args.buffer_size as f32 / args.sample_rate as f32),
        );
        use sdl2::audio::AudioCallback;
        let mut out = vec![0.0f32; args.buffer_size as usize];

        // Call the callback to get at least one status update
        tracker.callback(&mut out);
        let status = status_receiver.recv().unwrap();
        // Parse and send commands to play all of the waveforms, without
        // precomputing them.
        let mut environment =
            environment::Environment::new(args.sample_rate, args.tempo, args.library_root.clone());
        environment.set_check_mode(args.check);
        let player = player::Player::new(args.tempo, args.beats_per_measure, command_sender);
        for program_index in 0..state.programs.programs().len() {
            if state.programs.programs()[program_index].is_empty() {
                continue;
            }
            let display_name = state.programs.display_name(program_index);
            println!(
                "Playing program {}: {}",
                display_name,
                state.programs.programs()[program_index].text()
            );
            let input = state.programs.evaluation_input(program_index);
            let outcome = processing::process(&environment, &input, false);
            match state
                .programs
                .record(program_index, input.text.into(), outcome)
            {
                Ok(warnings) => {
                    for warning in &warnings {
                        println!("Warning: {}", warning);
                    }
                    if state.programs.programs()[program_index].is_playable() {
                        player.play_program(
                            &state.programs,
                            program_index,
                            &status,
                            player::Start::Now,
                            None,
                            &mut state.waveform_marks,
                        );
                    } else {
                        println!("Program {} did not evaluate to a waveform", display_name);
                    }
                }
                Err(diagnostics) => {
                    for diagnostic in &diagnostics {
                        println!("{}: {}", diagnostic.severity.label(), diagnostic);
                        if let Some(snippet) = &diagnostic.snippet {
                            println!("{}", snippet);
                        }
                    }
                    process::exit(1);
                }
            }
        }
        // Call the tracker's callback until all of the waveforms have finished
        // playing.
        loop {
            tracker.callback(&mut out);
            let status = status_receiver.recv().unwrap();
            let remaining = status.start_times.len();
            if remaining == 0 {
                println!("All waveforms finished");
                break;
            }
            println!("Still running, {} waveforms remaining", remaining);
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
    let (sample_producer, sample_consumer) =
        tracker::sample_rings(args.sample_rate as usize, SAMPLE_HEADER_CAPACITY);
    let device = audio_subsystem
        .open_playback(None, &desired_spec, |spec| {
            let output_latency = tracker::sdl_output_latency(spec.freq as u32, spec.samples as u32);
            println!("Spec: {:?}, output latency: {:?}", spec, output_latency);
            tracker::Tracker::<WaveformId, MarkId>::new(
                args.sample_rate,
                args.output_dir.clone().into(),
                captured_file_prefix.clone(),
                command_receiver,
                status_sender,
                sample_producer,
                output_latency,
            )
        })
        .unwrap();
    device.resume();

    // One environment and one tracker-facing player for the whole UI
    // session; both end up owned by the effect runner.
    let mut environment =
        environment::Environment::new(args.sample_rate, args.tempo, args.library_root.clone());
    environment.set_check_mode(args.check);
    let player = player::Player::new(args.tempo, args.beats_per_measure, command_sender.clone());

    // Start the beats!
    player.start_beats(&environment, &status_receiver, args.precompute);

    // Seed each slider's ramp baseline where the slider currently sits, so its
    // first move ramps instead of stepping to the target.
    let last_slider_values = effects::slider_values(&state.programs);
    // Spawn a thread that batches slider updates, sending them approximately once per audio buffer
    let buffer_duration =
        Duration::from_secs_f32(args.buffer_size as f32 / args.sample_rate as f32);
    let (slider_sender, slider_receiver) = mpsc::channel::<effects::SliderEvent>();
    let slider_command_sender = command_sender.clone();
    thread::spawn(move || {
        let mut last_slider_values = last_slider_values;
        loop {
            let mut pending: HashMap<MarkId, f32> = HashMap::new();
            let deadline;

            // Idle: block until the first slider update event arrives.
            loop {
                use effects::SliderEvent;
                match slider_receiver.recv() {
                    Ok(SliderEvent::UpdateSlider { mark, value }) => {
                        pending.insert(mark, value);
                        deadline = Instant::now() + buffer_duration;
                        break;
                    }
                    Ok(SliderEvent::SetBaselines(values)) => {
                        // The program set was reloaded: the slots the old
                        // baselines name may now hold different bindings.
                        last_slider_values = values;
                    }
                    Err(_) => return, // channel closed, exit thread
                };
            }

            // Accumulate: keep receiving events until the deadline.
            loop {
                let remaining = deadline.saturating_duration_since(Instant::now());
                if remaining.is_zero() {
                    // XXX is this check necessary?
                    break;
                }
                use effects::SliderEvent;
                match slider_receiver.recv_timeout(remaining) {
                    Ok(SliderEvent::UpdateSlider { mark, value }) => {
                        pending.insert(mark, value);
                    }
                    Ok(SliderEvent::SetBaselines(values)) => {
                        last_slider_values = values;
                    }
                    Err(mpsc::RecvTimeoutError::Timeout) => break,
                    Err(mpsc::RecvTimeoutError::Disconnected) => return,
                }
            }

            // Flush: send one Modify command per changed slider.
            // Technically we don't need to send a modify if the waveform is
            // not playing...
            for command in effects::flush_slider_updates(
                &mut pending,
                &mut last_slider_values,
                buffer_duration.as_secs_f32(),
            ) {
                let _ = slider_command_sender.send(command);
            }
        }
    });

    // Other threads wake the main loop by posting this payload-free event.
    let event_subsystem = sdl_context.event().unwrap();
    // SAFETY: the type is only matched by number, never read as a custom
    // event with `as_user_event_type`.
    let wake_event_type = unsafe { event_subsystem.register_event() }.unwrap();

    let launchkey = launchkey::Launchkey::new(main_loop_waker(&event_subsystem, wake_event_type));
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
        Duration::from_millis(args.snapshot_refresh_ms),
    );
    renderer.video_subsystem.text_input().start();
    // Scan shows two measures across the window, one column per pixel.
    let scan_window =
        Duration::from_secs_f64(2.0 * args.beats_per_measure as f64 * 60.0 / args.tempo as f64);
    let mut scope = scope::Scope::new(
        sample_consumer,
        args.sample_rate,
        scan_window,
        renderer.width as usize,
    );
    let mut event_pump = sdl_context.event_pump().unwrap();

    let sdl2_handler =
        sdl2_input::InputHandler::new(launchkey.is_err(), renderer.width, renderer.height);

    let mut launchkey = launchkey.ok();

    let (job_sender, replies) = processing::spawn(
        args.sample_rate,
        args.tempo,
        args.library_root.clone(),
        args.check,
        args.precompute,
        environment.module_table(),
        main_loop_waker(&event_subsystem, wake_event_type),
    );
    let mut effect_runner =
        effects::EffectRunner::new(player, environment, slider_sender.clone(), job_sender);
    // Evaluate each program from its initial source so that playback and
    // InstallKeys work without needing to enter and exit Edit mode first. The
    // window shows the unevaluated programs until the set is installed.
    effect_runner.load(&mut state, input, true, None);

    let mut status = tracker::Status {
        buffer_start: Instant::now(),
        start_times: Vec::new(),
        tracker_load: None,
        allocations: None,
    };
    let mut metrics = renderer::Metrics {
        tracker_load: Metric::new(Duration::from_secs(10), 100),
        allocations: Metric::new(Duration::from_secs(10), 100),
        renderer_load: Metric::new(Duration::from_secs(10), 100),
    };

    // Push the initial encoder + DAW-mode display state to the controller
    // if present.
    {
        let daw_mode_label = state.daw_pad_mode.display_name().to_string();
        run_effects(
            &mut effect_runner,
            &mut state,
            launchkey.as_mut(),
            &status,
            vec![
                actions::Effect::SyncEncoders,
                actions::Effect::SetDawModeDisplay(daw_mode_label),
            ],
        );
    }
    let mut next_render = Instant::now();
    loop {
        if state.should_quit {
            return;
        }

        // Sleep until an input or a wake event arrives, or until the next
        // deadline.
        let mut deadline = next_render;
        if let Some(tick) = state.next_take_tick() {
            deadline = deadline.min(tick);
        }
        let first_event = match deadline.checked_duration_since(Instant::now()) {
            // Round up: waking before the deadline would only wait again.
            Some(wait) if !wait.is_zero() => {
                event_pump.wait_event_timeout(wait.as_micros().div_ceil(1000) as u32)
            }
            _ => event_pump.poll_event(),
        };

        // Statuses don't wake the loop; use the latest that has arrived.
        while let Ok(tracker_status) = status_receiver.try_recv() {
            apply_status(tracker_status, &mut metrics, &mut status);
        }
        // Like statuses, samples don't wake the loop.
        scope.drain(scope::measure_origin(&status));

        // Before input, so that an input in this pass sees the results.
        loop {
            match replies.try_recv() {
                Ok(reply) => finish_job(
                    &mut effect_runner,
                    &mut state,
                    launchkey.as_mut(),
                    &status,
                    reply,
                ),
                Err(mpsc::TryRecvError::Empty) => break,
                // Only a panic stops the thread while the loop holds its
                // job sender.
                Err(mpsc::TryRecvError::Disconnected) => panic!("The processing thread stopped"),
            }
        }

        for event in first_event.into_iter().chain(event_pump.poll_iter()) {
            if matches!(event, sdl2::event::Event::User { type_, .. } if type_ == wake_event_type) {
                // The events behind a wake are drained from their own channels.
                continue;
            }
            let sdl_actions = sdl2_handler.classify(&event, &state);
            if let Some(actions) = sdl_actions {
                dispatch_actions(
                    &mut effect_runner,
                    &mut state,
                    launchkey.as_mut(),
                    &status,
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

        if state.take_tick_due(Instant::now()) {
            dispatch_actions(
                &mut effect_runner,
                &mut state,
                launchkey.as_mut(),
                &status,
                vec![actions::Action::TakeTick],
            );
        }

        // Render at most once a frame; inputs only wait behind a render
        // that has already started.
        let now = Instant::now();
        if next_render <= now {
            let due = next_render;
            next_render = now + renderer::FRAME_BUDGET;
            renderer.render(
                &state,
                &status,
                &scope,
                &mut metrics,
                launchkey.as_ref().map(|l| l.encoder_mode),
                due,
            );

            if let Some(launchkey) = launchkey.as_mut() {
                midi_input::update_launchkey_state(
                    &state,
                    &status,
                    args.tempo,
                    args.beats_per_measure,
                    launchkey,
                );
            }
        }
    }
}

/// Runs the `effects` against a `World` built from the given arguments.
fn run_effects(
    runner: &mut effects::EffectRunner,
    state: &mut actions::AppState,
    launchkey: Option<&mut launchkey::Launchkey>,
    status: &tracker::Status<WaveformId>,
    effects: Vec<actions::Effect>,
) {
    let mut world = effects::World { launchkey, status };
    runner.run_all(state, &mut world, effects);
}

/// Applies the processing thread's `reply` against a `World` built from the
/// given arguments.
fn finish_job(
    runner: &mut effects::EffectRunner,
    state: &mut actions::AppState,
    launchkey: Option<&mut launchkey::Launchkey>,
    status: &tracker::Status<WaveformId>,
    reply: processing::Reply,
) {
    let mut world = effects::World { launchkey, status };
    runner.finish(state, &mut world, reply);
}

/// Dispatches `actions` through the reducer against a `World` built from the
/// given arguments.
fn dispatch_actions(
    runner: &mut effects::EffectRunner,
    state: &mut actions::AppState,
    launchkey: Option<&mut launchkey::Launchkey>,
    status: &tracker::Status<WaveformId>,
    actions: Vec<actions::Action>,
) {
    let mut world = effects::World { launchkey, status };
    runner.dispatch(state, &mut world, actions);
}

/// Returns a function that wakes the main loop from any thread.
fn main_loop_waker(
    event_subsystem: &sdl2::EventSubsystem,
    wake_event_type: u32,
) -> impl Fn() + Send + Sync + 'static {
    let sender = event_subsystem.event_sender();
    // Posts a payload-free SDL event of type `wake_event_type`.
    move || {
        let event = sdl2::event::Event::User {
            timestamp: 0,
            window_id: 0,
            type_: wake_event_type,
            code: 0,
            data1: std::ptr::null_mut(),
            data2: std::ptr::null_mut(),
        };
        // A full queue already wakes the loop.
        let _ = sender.push_event(event);
    }
}

/// Applies a freshly received tracker status: folds its metrics into
/// `metrics` and replaces `status` with it.
fn apply_status(
    new_status: tracker::Status<WaveformId>,
    metrics: &mut renderer::Metrics,
    status: &mut tracker::Status<WaveformId>,
) {
    if let Some(ratio) = new_status.tracker_load {
        metrics.tracker_load.set_max(ratio);
    }
    if let Some(allocations) = new_status.allocations {
        metrics.allocations.set_max(allocations as f32);
    }
    *status = new_status;
}
