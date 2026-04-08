use core::panic;
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
use tuun::renderer;
use tuun::sdl2_input;
use tuun::slider;
use tuun::tracker;
use tuun::waveform;

use metric::Metric;
use parser::Expr;
use renderer::{MarkId, Mode, Program, ProgramSliders, Renderer, WaveformId, WaveformOrMode};
use tracker::Command;

// Additional built-ins

fn mark(arguments: Vec<Expr<MarkId>>) -> Expr<MarkId> {
    match &arguments[..] {
        [Expr::Float(id)] if *id >= 1.0 && id.fract() == 0.0 => {
            let id = id.round() as u32;
            Expr::BuiltIn {
                name: format!("mark({})", id),
                function: builtins::curry(move |waveform: Box<waveform::Waveform<MarkId>>| {
                    waveform::Waveform::Marked {
                        id: MarkId::UserDefined(id),
                        waveform,
                    }
                }),
            }
        }
        _ => Expr::Error("Invalid argument for mark".to_string()),
    }
}

#[derive(ClapParser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(long = "tempo", default_value_t = 90)]
    tempo: u32,
    #[arg(long = "beats_per_measure", default_value_t = 4)]
    beats_per_measure: u32,
    #[arg(long, default_value_t = 44100)]
    sample_rate: i32,
    #[arg(long, default_value_t = 2048)]
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

fn load_context(
    active_program_index: usize,
    args: &Args,
    mut context: &mut Vec<(String, Expr<MarkId>)>,
) -> Mode {
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
            function: parser::BuiltInFn(std::rc::Rc::new(mark)),
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
        active_program_index,
        message: if errors.len() == 0 {
            format!("Loaded {} bindings from context", bindings)
        } else {
            format!("Error loading context: {}", errors[0].to_string())
        },
    }
}

const NUM_PROGRAMS: usize = 8;

// Returns the initial values for all sliders
fn load_programs(args: &Args, programs: &mut Vec<Program>) -> HashMap<(WaveformId, String), f32> {
    programs.clear();
    if !args.programs_file.is_empty() {
        let mut count = 0;
        let contents = fs::read_to_string(&args.programs_file).unwrap_or_default();
        let mut pending_sliders: Option<Vec<parser::Slider>> = None;
        for line in contents.lines() {
            // Check for slider pragma before stripping comments
            // TODO this doesn't report errors when there is something that looks a bit like
            // a slider config but isn't.
            if let Ok(sliders) = parser::parse_annotation(line) {
                pending_sliders = Some(sliders);
                continue;
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
                programs.push(Program {
                    text: line.to_string(),
                    id: renderer::id_from_index(programs.len()),
                    sliders,
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
            });
        }
    }
    // Fill up to NUM_PROGRAMS with empty entries if necessary
    while programs.len() < NUM_PROGRAMS {
        programs.push(Program {
            text: String::new(),
            id: renderer::id_from_index(programs.len()),
            sliders: ProgramSliders::default(),
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
    last_slider_values
}

pub fn main() {
    let args = Args::parse();
    let mut context = Vec::new();
    let mut mode = load_context(0, &args, &mut context);
    let mut programs: Vec<Program> = Vec::new();
    let last_slider_values = load_programs(&args, &mut programs);

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
        for (index, program) in programs.iter().enumerate() {
            println!("Playing program {}: {}", program.id, program.text);
            let program = Program {
                text: program.text.to_string(),
                id: program.id,
                sliders: program.sliders.clone(),
            };
            let mode = play_waveform(
                &context,
                &status,
                &args,
                index,
                program.text.len(),
                &program,
                &command_sender,
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

    start_beats(&command_sender, &status_receiver, &args, &context);

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
                    Err(mpsc::RecvTimeoutError::Timeout) => break,
                    Err(mpsc::RecvTimeoutError::Disconnected) => return,
                }
            }

            // Flush: send one Modify command per changed slider
            for ((id, slider), value) in pending.drain() {
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
    );
    renderer.video_subsystem.text_input().start();
    let mut event_pump = sdl_context.event_pump().unwrap();

    let sdl2_handler = sdl2_input::InputHandler::new(
        launchkey.is_err(),
        renderer.width,
        renderer.height,
        &command_sender,
        &slider_sender,
    );

    let mut midi_handler = if let Ok(launchkey) = launchkey {
        Some(midi_input::InputHandler::new(
            launchkey,
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
    loop {
        for event in event_pump.poll_iter() {
            //println!("Event: {:?} with mode {:?}", event, mode);
            mode = sdl2_handler.handle_event(event, &context, mode, &status, &mut programs);
            mode = match mode {
                Mode::Exit => {
                    return;
                }
                Mode::Play {
                    active_program_index,
                    cursor_position,
                    program,
                    repeat_after_measures,
                } => play_waveform(
                    &context,
                    &status,
                    &args,
                    active_program_index,
                    cursor_position,
                    &program,
                    &command_sender,
                    repeat_after_measures,
                ),
                Mode::LoadContext {
                    active_program_index,
                } => load_context(active_program_index, &args, &mut context),
                Mode::LoadPrograms {
                    active_program_index,
                } => {
                    let slider_values = load_programs(&args, &mut programs);
                    slider_sender
                        .send(renderer::SliderEvent::SetInitialValues(slider_values))
                        .unwrap();
                    Mode::Select {
                        active_program_index,
                        message: format!("Loaded programs"),
                    }
                }
                _ => mode,
            }
        }

        if let Some(midi_handler) = &mut midi_handler {
            // TODO this is a little like calling "render" but on the MIDI device. Generalize?
            if let Mode::Select {
                active_program_index,
                ..
            } = mode
            {
                // This might be a new program, in which case we need to update any device state.

                midi_handler.update_slider_state(&programs[active_program_index]);
            }

            loop {
                match midi_handler.events().try_recv() {
                    Ok(event) => {
                        // TODO handle modes like Play
                        mode =
                            midi_handler.handle_event(event, &context, mode, &status, &mut programs)
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
        if statuses_received > 1 {
            println!("Received {} statuses", statuses_received);
        }
        renderer.render(&ttf_context, &programs, &status, &mode, &mut metrics);
    }
}

fn start_beats(
    command_sender: &mpsc::Sender<Command<WaveformId, MarkId>>,
    status_receiver: &mpsc::Receiver<tracker::Status<WaveformId, MarkId>>,
    args: &Args,
    context: &Vec<(String, Expr<MarkId>)>,
) {
    // Play the odd Beats waveform starting immediately and repeating every two measures
    command_sender
        .send(Command::Play {
            id: WaveformId::Beats(false),
            waveform: renderer::beats_waveform(args.tempo, args.beats_per_measure, context),
            start: Instant::now(),
            repeat_every: Some(
                renderer::duration_from_beats(args.tempo, args.beats_per_measure as u64) * 2,
            ),
        })
        .unwrap();
    // We need to wait to start the even Beats until we know when the odd Beats started
    'start_even_beats: loop {
        match status_receiver.recv() {
            Ok(status) => {
                for mark in status.marks {
                    if mark.waveform_id == WaveformId::Beats(false)
                        && mark.mark_id == MarkId::TopLevel
                    {
                        command_sender
                            .send(Command::Play {
                                id: WaveformId::Beats(true),
                                waveform: renderer::beats_waveform(
                                    args.tempo,
                                    args.beats_per_measure,
                                    context,
                                ),
                                start: mark.start + mark.duration,
                                repeat_every: Some(
                                    renderer::duration_from_beats(
                                        args.tempo,
                                        args.beats_per_measure as u64,
                                    ) * 2,
                                ),
                            })
                            .unwrap();
                        break 'start_even_beats;
                    }
                }
            }
            Err(_) => {}
        }
    }
}

// Returns the start time of the next measure
fn next_measure_start(status: &tracker::Status<WaveformId, MarkId>) -> Instant {
    for mark in &status.marks {
        match mark.waveform_id {
            WaveformId::Beats(_)
                if mark.mark_id == MarkId::TopLevel && mark.start > Instant::now() =>
            {
                return mark.start;
            }
            _ => (),
        }
    }
    panic!("No next measure found in marks");
}

fn play_waveform(
    context: &Vec<(String, Expr<MarkId>)>,
    status: &tracker::Status<WaveformId, MarkId>,
    args: &Args,
    active_program_index: usize,
    cursor_position: usize,
    program: &Program,
    command_sender: &std::sync::mpsc::Sender<Command<WaveformId, MarkId>>,
    repeat_after_measures: Option<u32>,
) -> Mode {
    match renderer::play_waveform_helper(&context, active_program_index, cursor_position, program) {
        WaveformOrMode::Waveform(mut waveform) => {
            if args.precompute {
                // TODO probably precompute should happen on another thread
                let mut generator = generator::Generator::new(args.sample_rate);
                waveform = waveform::remove_state(generator.precompute(waveform));
                println!("precompute returned: {}", &waveform);
            }
            let message;
            let repeat_every;
            if let Some(measures) = repeat_after_measures {
                let beats = (measures * args.beats_per_measure) as u64;
                message = format!("Looping waveform {} every {:?} beats", program.id, beats);
                repeat_every = Some(renderer::duration_from_beats(args.tempo, beats));
            } else {
                // Otherwise, play it once
                message = format!("Playing waveform {}", program.id);
                repeat_every = None;
            }
            command_sender
                .send(Command::Play {
                    // TODO maybe extend the mark to the full measure?
                    id: WaveformId::Program(program.id),
                    waveform: waveform::Waveform::Marked {
                        id: MarkId::TopLevel,
                        waveform: Box::new(waveform),
                    },
                    start: next_measure_start(&status),
                    repeat_every,
                })
                .unwrap();

            return Mode::Select {
                active_program_index,
                message,
            };
        }
        WaveformOrMode::Mode(new_mode) => {
            return new_mode;
        }
    }
}
