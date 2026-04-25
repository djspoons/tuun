use std::collections::HashMap;
use std::fs;
use std::process;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use clap::Parser as ClapParser;
use sdl2::audio::AudioSpecDesired;
use sdl2::ttf::Sdl2TtfContext;

use tuun::builtins;
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
use parser::Expr;
use renderer::{MarkId, Mode, Program, ProgramSliders, Renderer, WaveformId};
use tracker::Command;

#[derive(ClapParser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(long = "tempo", default_value_t = 90)]
    tempo: u32,
    #[arg(long = "beats_per_measure", default_value_t = 4)]
    beats_per_measure: u32,
    #[arg(long, default_value_t = 44100)]
    sample_rate: i32,
    #[arg(long, default_value_t = 1024)]
    buffer_size: u16,
    #[arg(short = 'C', long = "context_file", number_of_values = 1)]
    context_files: Vec<String>,
    #[arg(short = 'P', long = "programs_file", default_value = "")]
    programs_file: String,
    // Additional programs to load
    #[arg(short, long = "program", default_value = "", number_of_values = 1)]
    programs: Vec<String>,
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
    #[arg(short = 'O', long, default_value = ".")]
    output_dir: String, // Captures waveforms to the specified directory
}

fn load_context(args: &Args, mut context: &mut Vec<(String, Expr<MarkId>)>) -> Mode {
    context.clear();
    context.push(("tempo".to_string(), Expr::Float(args.tempo as f32)));
    context.push((
        "sample_rate".to_string(),
        Expr::Float(args.sample_rate as f32),
    ));
    builtins::add_prelude(&mut context);
    context.push((
        "mark".to_string(),
        Expr::BuiltIn {
            name: "mark".to_string(),
            function: parser::BuiltInFn(std::rc::Rc::new(renderer::mark)),
        },
    ));

    let mut bindings = 0;
    let mut errors = Vec::new();
    for file in args.context_files.iter() {
        let raw_context = std::fs::read_to_string(file)
            .expect(format!("Failed to read context file: {}", file).as_str());
        // Strip out comments (that is any after // on a line)
        let raw_context: String = raw_context
            .lines()
            .map(|line| {
                if let Some(comment_index) = line.find("//") {
                    &line[..comment_index]
                } else {
                    line
                }
            })
            .collect::<Vec<&str>>()
            .join("\n");
        match parser::parse_context(&raw_context) {
            Ok(parsed_exprs) => {
                println!("Parsed context from {}:", file);
                for (pattern, parsed_expr) in parsed_exprs {
                    match parser::evaluate(&context, parsed_expr) {
                        Ok(expr) => {
                            match parser::extend_context(&mut context, &pattern, &expr) {
                                Ok(_) => println!("   {}", &pattern),
                                Err(error) => errors.push(error),
                            }
                            // Not exactly one binding... :shrug:
                            bindings += 1;
                        }
                        Err(error) => {
                            println!(
                                "Error evaluating context expression for {}: {:?}",
                                pattern, error
                            );
                            errors.push(error);
                        }
                    }
                }
            }
            Err(es) => {
                println!("Errors parsing context: {:?}", es);
                errors.extend_from_slice(&es);
            }
        }
    }

    Mode::Select {
        message: if errors.len() == 0 {
            format!("Loaded {} bindings from context", bindings)
        } else {
            format!("Error loading context: {}", errors[0].to_string())
        },
    }
}

// Returns the initial values for all sliders
fn load_programs(
    args: &Args,
    programs: &mut Vec<Program>,
) -> (HashMap<(WaveformId, String), f32>, Vec<parser::Error>) {
    let mut errors = Vec::new();
    programs.clear();
    if !args.programs_file.is_empty() {
        let mut count = 0;
        let contents = fs::read_to_string(&args.programs_file).unwrap_or_default();
        let mut pending_sliders: Option<Vec<parser::Slider>> = None;
        let mut pending_color: Option<(u8, u8, u8)> = None;
        let mut pending_level_db: f32 = 0.0;
        for line in contents.lines() {
            // Check for annotations before stripping comments
            let annos = match parser::parse_annotations(line) {
                Ok(annos) => annos,
                Err(mut e) => {
                    println!("Got errors parsing annotations: {:?}", e);
                    errors.append(&mut e);
                    continue;
                }
            };
            for anno in annos {
                match anno {
                    parser::Annotation::Sliders(sliders) => {
                        pending_sliders = Some(sliders);
                    }
                    parser::Annotation::Color(r, g, b) => {
                        pending_color = Some((r, g, b));
                    }
                    parser::Annotation::Level(v) => {
                        pending_level_db = v;
                    }
                    parser::Annotation::NextBank => {
                        while programs.len() % renderer::PROGRAMS_PER_BANK != 0 {
                            programs.push(Program {
                                text: String::new(),
                                id: renderer::id_from_index(programs.len()),
                                sliders: ProgramSliders::default(),
                                color: None,
                                level_db: 0.0,
                            })
                        }
                    }
                }
            }

            let line = if let Some(comment_index) = line.find("//") {
                &line[..comment_index]
            } else {
                line
            }
            .trim();
            if !line.is_empty() {
                let sliders = if let Some(configs) = pending_sliders.take() {
                    use parser::SliderFunction;
                    let normalized_values = configs
                        .iter()
                        .map(|c| match &c.function {
                            SliderFunction::Linear {
                                initial_value,
                                min,
                                max,
                            } => ((initial_value - min) / (max - min)).clamp(0.0, 1.0),
                            SliderFunction::UserDefined {
                                normalized_initial_value,
                                ..
                            } => normalized_initial_value.clamp(0.0, 1.0),
                        })
                        .collect();
                    ProgramSliders {
                        configs,
                        normalized_values,
                    }
                } else {
                    ProgramSliders::default()
                };
                let level_db = std::mem::replace(&mut pending_level_db, 0.0);
                programs.push(Program {
                    text: line.to_string(),
                    id: renderer::id_from_index(programs.len()),
                    sliders,
                    color: pending_color.take(),
                    level_db,
                });
                count += 1;
            }
        }
        println!("Loaded {} programs from {}", count, args.programs_file);
    }
    // Add in any additional programs specified on the command line
    for program_text in &args.programs {
        if !program_text.is_empty() {
            programs.push(Program {
                text: program_text.to_string(),
                id: renderer::id_from_index(programs.len()),
                sliders: ProgramSliders::default(),
                color: None,
                level_db: 0.0,
            });
        }
    }
    // Fill up with empty entries if necessary
    while programs.len() < renderer::NUM_PROGRAM_BANKS * renderer::PROGRAMS_PER_BANK {
        programs.push(Program {
            text: String::new(),
            id: renderer::id_from_index(programs.len()),
            sliders: ProgramSliders::default(),
            color: None,
            level_db: 0.0,
        });
    }
    // Copy initial values for each slider for all programs
    let mut last_slider_values: HashMap<(WaveformId, String), f32> = HashMap::new();
    for Program { id, sliders, .. } in programs.iter() {
        for (j, config) in sliders.configs.iter().enumerate() {
            let value =
                slider::denormalize(&config.function, sliders.normalized_values[j]).unwrap_or(0.0);
            last_slider_values.insert(
                (WaveformId::Program(id.clone()), config.label.clone()),
                value,
            );
        }
    }
    (last_slider_values, errors)
}

pub fn main() {
    let args = Args::parse();
    let mut context = Vec::new();
    let mut mode = load_context(&args, &mut context);
    let mut programs: Vec<Program> = Vec::new();
    let (last_slider_values, errors) = load_programs(&args, &mut programs);
    if errors.len() > 0 {
        mode = Mode::Select {
            message: format!("Error loading programs: {}", errors[0].to_string()),
        };
    }

    let (status_sender, status_receiver) = mpsc::channel();
    let (command_sender, command_receiver) = mpsc::channel();

    if !args.ui {
        println!("Starting in non-UI mode");
        // Filter out any empty programs
        programs.retain(|p| !p.text.is_empty());

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
        let mut status = status_receiver.recv().unwrap();
        // We need to add one Beats mark so that we can reuse the helper below to play waveforms
        status.marks = vec![tracker::Mark {
            waveform_id: WaveformId::Beats(false),
            mark_id: MarkId::TopLevel,
            start: Instant::now() + Duration::from_millis(100), // Some time in the near future
            duration: Duration::from_secs(4),                   // Doesn't matter
        }];

        // Parse and send commands to play all of the waveforms.
        let mut play_helper =
            play_helper::PlayHelper::new(args.tempo, args.beats_per_measure, command_sender);
        for program in programs.iter() {
            println!("Playing program {}: {}", program.id, program.text);
            let program = Program {
                text: program.text.to_string(),
                id: program.id,
                sliders: program.sliders.clone(),
                color: program.color,
                level_db: program.level_db,
            };
            let mode = play_helper.play_waveform(
                &context,
                program.text.len(),
                &program,
                &status,
                false,
                None,
            );
            match mode {
                // If mode is Edit, there was an error; print it and exit.
                Mode::Edit { message, .. } => {
                    println!("{}", message);
                    process::exit(1);
                }
                _ => (),
            }
        }
        // Call the tracker's callback until all of the waveforms have finished playing
        loop {
            tracker.callback(&mut out);
            let status = status_receiver.recv().unwrap();
            let mark_count = status
                .marks
                .iter()
                .filter(|mark| !mark.waveform_id.is_beats() && mark.mark_id == MarkId::TopLevel)
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
        freq: Some(args.sample_rate),
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

    play_helper::PlayHelper::new(args.tempo, args.beats_per_measure, command_sender.clone())
        .start_beats(&status_receiver, &context);

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
                        // we understand better what "load_programs" does to currently playing waveforms.
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

    // N.B. We shadow the command sender here so that any uses below will go through here.
    // This simplifies things a bit, but it means that all commands will wait on a long
    // pre-computation.
    let play_command_sender = command_sender.clone();
    let (precomputing_command_sender, play_receiver) = mpsc::channel();
    thread::spawn(move || {
        loop {
            match play_receiver.recv() {
                Ok(Command::Play {
                    id,
                    mut waveform,
                    start,
                    repeat_every,
                }) => {
                    if args.precompute {
                        // TODO probably precompute should happen on another thread
                        let mut generator = generator::Generator::new(args.sample_rate);
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

    let mut sdl2_handler = sdl2_input::InputHandler::new(
        launchkey.is_err(),
        renderer.width,
        renderer.height,
        play_helper::PlayHelper::new(
            args.tempo,
            args.beats_per_measure,
            precomputing_command_sender.clone(),
        ),
        precomputing_command_sender.clone(),
        slider_sender.clone(),
    );

    let mut midi_handler = if let Ok(launchkey) = launchkey {
        Some(midi_input::InputHandler::new(
            launchkey,
            play_helper::PlayHelper::new(
                args.tempo,
                args.beats_per_measure,
                command_sender.clone(),
            ),
            command_sender.clone(),
            slider_sender.clone(),
        ))
    } else {
        None
    };

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

    const BUFFER_REFRESH_INTERVAL: Duration = Duration::from_millis(200);
    let mut next_buffer_refresh = Instant::now();
    command_sender.send(Command::SendCurrentBuffer).unwrap();
    let mut active_program_index = 0;
    // Track which program we last pushed to the MIDI controller so we don't
    // overwrite the user's encoder turns on every loop iteration.
    let mut last_synced_program_index: Option<usize> = None;
    loop {
        for event in event_pump.poll_iter() {
            //println!("Event: {:?} with mode {:?}", event, mode);
            mode = sdl2_handler.handle_event(
                event,
                &context,
                mode,
                &mut active_program_index,
                &status,
                &mut programs,
            );
            mode = match mode {
                Mode::Exit => {
                    return;
                }
                Mode::LoadContext {} => load_context(&args, &mut context),
                Mode::LoadPrograms {} => {
                    let (slider_values, errors) = load_programs(&args, &mut programs);
                    slider_sender
                        .send(renderer::SliderEvent::SetInitialValues(slider_values))
                        .unwrap();
                    last_synced_program_index = None;
                    if errors.len() > 0 {
                        Mode::Select {
                            message: format!("Error loading programs: {}", errors[0].to_string()),
                        }
                    } else {
                        Mode::Select {
                            message: format!("Loaded programs"),
                        }
                    }
                }
                _ => mode,
            }
        }

        if let Some(midi_handler) = &mut midi_handler {
            // Only push encoder state when the active program changes (or after
            // LoadPrograms). Otherwise we'd overwrite the user's encoder turns
            // on every loop iteration.
            if let Mode::Select { .. } = mode {
                if last_synced_program_index != Some(active_program_index) {
                    midi_handler.update_encoder_state_for_programs(&programs, active_program_index);
                    last_synced_program_index = Some(active_program_index);
                }
            }

            loop {
                match midi_handler.events().try_recv() {
                    Ok(event) => {
                        // TODO handle modes like LoadContext?
                        mode = midi_handler.handle_event(
                            event,
                            &context,
                            mode,
                            &mut active_program_index,
                            &status,
                            &mut programs,
                        )
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
                    if let Some(ratio) = tracker_status.tracker_load {
                        metrics.tracker_load.set(ratio);
                    }
                    if let Some(allocations) = tracker_status.allocations_per_sample {
                        metrics.allocations_per_sample.set(allocations);
                    }
                    status = tracker_status;
                    statuses_received += 1;
                }
                Err(mpsc::TryRecvError::Empty) => {
                    if statuses_received > 0 {
                        break;
                    }
                    match status_receiver.recv_timeout(Duration::from_millis(10)) {
                        Ok(tracker_status) => {
                            if let Some(ratio) = tracker_status.tracker_load {
                                metrics.tracker_load.set(ratio);
                            }
                            if let Some(allocations) = tracker_status.allocations_per_sample {
                                metrics.allocations_per_sample.set(allocations);
                            }
                            status = tracker_status;
                            statuses_received += 1;
                        }
                        Err(mpsc::RecvTimeoutError::Timeout) => break,
                        Err(e) => println!("Error receiving status with timeout: {:?}", e),
                    }
                }
                Err(e) => println!("Error receiving status: {:?}", e),
            }
        }
        /*
        if statuses_received > 1 {
            println!("Received {} statuses", statuses_received);
        }
        */
        renderer.render(
            &ttf_context,
            &programs,
            &status,
            &mode,
            active_program_index,
            &mut metrics,
        );

        if let Some(midi_handler) = &mut midi_handler {
            midi_handler.update_state(&programs, &status, &mode, active_program_index);
        }
    }
}
