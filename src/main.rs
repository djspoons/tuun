use core::panic;
use std::fs;
use std::process;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use chrono;
use clap::Parser as ClapParser;
use sdl2::audio::AudioSpecDesired;
use sdl2::event::Event;
use sdl2::ttf::Sdl2TtfContext;

use metric::Metric;
use renderer::{Mode, Program, ProgramSliders, Renderer, WaveformId};
use tracker::Command;
use tuun::builtins;
use tuun::generator;
use tuun::metric;
use tuun::optimizer;
use tuun::parser;
use tuun::renderer;
use tuun::tracker;
use tuun::waveform;

fn slider_tracker_key(program_id: renderer::ProgramId, label: &str) -> String {
    format!("{}:{}", program_id, label)
}

fn prepend_native_slider_bindings(program: &Program) -> String {
    if program.sliders.configs.is_empty() {
        return program.text.clone();
    }
    let bindings = program
        .sliders
        .configs
        .iter()
        .map(|c| {
            format!(
                "{} = slider(\"{}\")",
                c.label,
                slider_tracker_key(program.id, &c.label)
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("let {} in {}", bindings, program.text)
}

fn send_initial_slider_values(
    programs: &[Program],
    command_sender: &std::sync::mpsc::Sender<Command<WaveformId>>,
) {
    for program in programs {
        for (j, config) in program.sliders.configs.iter().enumerate() {
            let key = slider_tracker_key(program.id, &config.label);
            let actual_value =
                config.min + program.sliders.normalized_values[j] * (config.max - config.min);
            command_sender
                .send(Command::MoveSlider {
                    slider: key,
                    value: actual_value,
                })
                .unwrap();
        }
    }
}

#[derive(ClapParser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(default_value_t = 90)]
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
    optimize: bool,
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

fn load_context(active_program_index: usize, args: &Args) -> (Vec<(String, parser::Expr)>, Mode) {
    let mut context: Vec<(String, parser::Expr)> = Vec::new();
    context.push(("tempo".to_string(), parser::Expr::Float(args.tempo as f32)));
    context.push((
        "sampling_frequency".to_string(),
        parser::Expr::Float(args.sample_rate as f32),
    ));
    builtins::add_prelude(&mut context);

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
                    match parser::simplify(&context, parsed_expr) {
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
                                "Error simplifying context expression for {}: {:?}",
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
    return (
        context,
        Mode::Select {
            active_program_index,
            message: if errors.len() == 0 {
                format!("Loaded {} bindings from context", bindings)
            } else {
                format!("Error loading context: {}", errors[0].to_string())
            },
        },
    );
}

const NUM_PROGRAMS: usize = 8;

fn load_programs(args: &Args, programs: &mut Vec<Program>) {
    *programs = Vec::new();
    if !args.programs_file.is_empty() {
        let mut count = 0;
        let contents = fs::read_to_string(&args.programs_file).unwrap_or_default();
        let mut pending_slider_configs: Option<Vec<parser::SliderConfig>> = None;
        for line in contents.lines() {
            // Check for slider pragma before stripping comments
            if let Some(mut configs) = parser::parse_slider_pragma(line) {
                if configs.len() > 2 {
                    eprintln!("Warning: more than 2 sliders specified, using first 2");
                    configs.truncate(2);
                }
                pending_slider_configs = Some(configs);
                continue;
            }

            let line = if let Some(comment_index) = line.find("//") {
                &line[..comment_index]
            } else {
                line
            }
            .trim();
            if !line.is_empty() {
                let sliders = if let Some(configs) = pending_slider_configs.take() {
                    let normalized_values = configs
                        .iter()
                        .map(|c| ((c.value - c.min) / (c.max - c.min)).clamp(0.0, 1.0))
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
                    id: id_from_index(programs.len()),
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
                id: id_from_index(programs.len()),
                sliders: ProgramSliders::default(),
            });
        }
    }
    // Fill up to NUM_PROGRAMS with empty entries if necessary
    while programs.len() < NUM_PROGRAMS {
        programs.push(Program {
            text: String::new(),
            id: id_from_index(programs.len()),
            sliders: ProgramSliders::default(),
        });
    }
}

pub fn main() {
    let args = Args::parse();
    let (mut context, mut mode) = load_context(0, &args);
    let mut programs: Vec<Program> = Vec::new();
    load_programs(&args, &mut programs);

    let (status_sender, status_receiver) = mpsc::channel();
    let (command_sender, command_receiver) = mpsc::channel();

    if !args.ui {
        println!("Starting in non-UI mode");
        // Filter out any empty programs
        programs.retain(|p| !p.text.is_empty());

        let mut tracker = tracker::Tracker::<WaveformId>::new(
            args.sample_rate,
            args.output_dir.clone().into(),
            args.date_format.clone(),
            command_receiver,
            status_sender,
        );
        use sdl2::audio::AudioCallback;
        let mut out = vec![0.0f32; args.buffer_size as usize];

        // Send initial slider values to the tracker
        send_initial_slider_values(&programs, &command_sender);

        // Call the callback to get at least one status update
        tracker.callback(&mut out);
        let mut status = status_receiver.recv().unwrap();
        // We need to add one Beats mark so that we can reuse the helper below to play waveforms
        status.marks = vec![tracker::Mark {
            waveform_id: WaveformId::Beats(false),
            mark_id: 0,
            start: Instant::now() + Duration::from_millis(100), // Some time in the near future

            duration: Duration::from_secs(4), // Doesn't matter
        }];
        const MARK_ID: u32 = 100;
        // Parse and send commands to play all of the waveforms.
        for (index, program) in programs.iter().enumerate() {
            println!("Playing program {}: {}", program.id, program.text);
            // Wrap each program in a mark so that we can wait for it to finish
            let marked = Program {
                text: format!("({}) | mark({})", program.text, MARK_ID),
                id: program.id,
                sliders: program.sliders.clone(),
            };
            let (_, mode) = play_waveform(
                context.clone(),
                &status,
                &args,
                index,
                program.text.len(),
                &marked,
                &command_sender,
                sdl2::keyboard::Mod::empty(),
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
                .filter(|mark| mark.mark_id == MARK_ID)
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
            tracker::Tracker::<WaveformId>::new(
                args.sample_rate,
                args.output_dir.clone().into(),
                args.date_format.clone(),
                command_receiver,
                status_sender,
            )
        })
        .unwrap();
    device.resume();

    let ttf_context: Sdl2TtfContext = sdl2::ttf::init().unwrap();
    let mut renderer = Renderer::new(
        &sdl_context,
        &ttf_context,
        args.tempo,
        args.beats_per_measure,
    );
    renderer.video_subsystem.text_input().start();
    let mut event_pump = sdl_context.event_pump().unwrap();

    start_beats(&command_sender, &status_receiver, &args);

    // Send initial slider values to the tracker for all programs
    send_initial_slider_values(&programs, &command_sender);

    let mut status = tracker::Status {
        buffer_start: Instant::now(),
        marks: Vec::new(),
        slider_values: std::collections::HashMap::new(),
        buffer: None,
        tracker_load: None,
    };
    let mut metrics = renderer::Metrics {
        tracker_load: Metric::new(std::time::Duration::from_secs(10), 100),
    };

    const BUFFER_REFRESH_INTERVAL: Duration = Duration::from_millis(200);
    let mut next_buffer_refresh = Instant::now();
    command_sender.send(Command::SendCurrentBuffer).unwrap();
    loop {
        for event in event_pump.poll_iter() {
            //println!("Event: {:?} with mode {:?}", event, mode);
            (context, mode) = process_event::<WaveformId>(
                &args,
                context,
                &renderer,
                event,
                mode,
                &status,
                &mut programs,
                &command_sender,
            );
            if let Mode::Exit = mode {
                return;
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
    command_sender: &mpsc::Sender<Command<WaveformId>>,
    status_receiver: &mpsc::Receiver<tracker::Status<WaveformId>>,
    args: &Args,
) {
    // Play the odd Beats waveform starting immediately and repeating every two measures
    command_sender
        .send(Command::Play {
            id: WaveformId::Beats(false),
            waveform: optimizer::replace_seq(renderer::beats_waveform(
                args.tempo,
                args.beats_per_measure,
            ))
            .1,
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
                    if mark.waveform_id == WaveformId::Beats(false) && mark.mark_id == 0 {
                        command_sender
                            .send(Command::Play {
                                id: WaveformId::Beats(true),
                                waveform: optimizer::replace_seq(renderer::beats_waveform(
                                    args.tempo,
                                    args.beats_per_measure,
                                ))
                                .1,
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

fn edit_mode_from_program(
    active_program_index: usize,
    cursor_position: usize,
    program: &str,
) -> Mode {
    Mode::Edit {
        active_program_index,
        cursor_position,
        errors: if program.is_empty() {
            Vec::new()
        } else {
            match parser::parse_program(program) {
                Ok(_) => Vec::new(),
                Err(errors) => errors,
            }
        },
        // TODO could show the current sliders here
        message: String::new(),
    }
}

// These two functions allow for explicit conversion from index to id.

fn index_from_id(id: renderer::ProgramId) -> usize {
    return (id - 1) as usize;
}

fn id_from_index(index: usize) -> renderer::ProgramId {
    return (index + 1) as renderer::ProgramId;
}

fn process_event<I>(
    args: &Args,
    context: Vec<(String, parser::Expr)>,
    renderer: &Renderer,
    event: Event,
    mode: Mode,
    status: &tracker::Status<WaveformId>,
    programs: &mut Vec<Program>,
    command_sender: &std::sync::mpsc::Sender<Command<WaveformId>>,
) -> (Vec<(String, parser::Expr)>, Mode) {
    use sdl2::keyboard::Mod;
    use sdl2::keyboard::Scancode;
    match event {
        // XXX should we use Instant::now() or buffer_start?
        Event::Quit { .. } => return (context, Mode::Exit),
        Event::KeyDown {
            scancode, keymod, ..
        } => {
            match (mode, scancode) {
                // Exit on control-C
                (mode, Some(Scancode::C)) => {
                    if keymod.contains(Mod::LCTRLMOD) || keymod.contains(Mod::RCTRLMOD) {
                        return (context, Mode::Exit);
                    } else {
                        return (context, mode);
                    }
                }
                (
                    Mode::Select {
                        active_program_index,
                        ..
                    },
                    Some(Scancode::Return),
                ) => {
                    if keymod.contains(Mod::LGUIMOD) || keymod.contains(Mod::RGUIMOD) {
                        return play_waveform(
                            context,
                            status,
                            args,
                            active_program_index,
                            programs[active_program_index].text.len(),
                            &programs[active_program_index],
                            command_sender,
                            keymod,
                        );
                    }
                    // Check to see whether or not the current index is in the tracker's
                    // pending waveforms
                    if renderer::is_pending_program(
                        &status,
                        Instant::now(),
                        programs[active_program_index].id,
                    ) {
                        // If it is, send a command to remove it.
                        command_sender
                            .send(Command::RemovePending {
                                id: WaveformId::Program(programs[active_program_index].id),
                            })
                            .unwrap();
                    }
                    let mut mode = edit_mode_from_program(
                        active_program_index,
                        programs[active_program_index].text.len(),
                        &programs[active_program_index].text,
                    );
                    mode = match mode {
                        Mode::Edit {
                            active_program_index,
                            cursor_position,
                            message,
                            errors,
                        } => {
                            if !errors.is_empty() {
                                Mode::Edit {
                                    active_program_index,
                                    cursor_position,
                                    message: format!("Error: {}", errors[0].to_string()),
                                    errors,
                                }
                            } else if !programs[active_program_index].sliders.configs.is_empty() {
                                let ps = &programs[active_program_index].sliders;
                                Mode::Edit {
                                    active_program_index,
                                    cursor_position,
                                    message: format!(
                                        "{}",
                                        ps.slider_display()
                                            .iter()
                                            .map(|s| format!("{}", s))
                                            .collect::<Vec<_>>()
                                            .join(", ")
                                    ),

                                    errors,
                                }
                            } else {
                                Mode::Edit {
                                    active_program_index,
                                    cursor_position,
                                    errors,
                                    message,
                                }
                            }
                        }
                        _ => mode,
                    };
                    (context, mode)
                }
                (
                    Mode::Select {
                        active_program_index,
                        ..
                    },
                    Some(Scancode::Escape),
                ) => {
                    let mut message = String::new();

                    if keymod.contains(Mod::LGUIMOD)
                        || keymod.contains(Mod::RGUIMOD)
                            && renderer::is_active_program(
                                &status,
                                Instant::now(),
                                programs[active_program_index].id,
                            )
                    {
                        // If the program is active, stop it
                        command_sender
                            .send(Command::Stop {
                                id: WaveformId::Program(programs[active_program_index].id),
                            })
                            .unwrap();
                        message = format!("Stopped program {}", programs[active_program_index].id);
                    } else if !keymod.contains(Mod::LGUIMOD)
                        && !keymod.contains(Mod::RGUIMOD)
                        && renderer::is_pending_program(
                            &status,
                            Instant::now(),
                            programs[active_program_index].id,
                        )
                    {
                        // If it is, send a command to remove it.
                        command_sender
                            .send(Command::RemovePending {
                                id: WaveformId::Program(programs[active_program_index].id),
                            })
                            .unwrap();
                        message = format!(
                            "Removed pending waveform for program {}",
                            programs[active_program_index].id
                        );
                    }
                    (
                        context,
                        Mode::Select {
                            active_program_index,
                            message,
                        },
                    )
                }
                (
                    Mode::Select {
                        active_program_index,
                        ..
                    },
                    Some(Scancode::Up),
                ) => (
                    context,
                    Mode::Select {
                        active_program_index: (active_program_index + programs.len() - 1)
                            % programs.len(),
                        message: String::new(),
                    },
                ),
                (
                    Mode::Select {
                        active_program_index,
                        ..
                    },
                    Some(Scancode::Down),
                ) => (
                    context,
                    Mode::Select {
                        active_program_index: (active_program_index + 1) % programs.len(),
                        message: String::new(),
                    },
                ),
                (
                    Mode::Select {
                        active_program_index,
                        ..
                    },
                    Some(Scancode::LAlt) | Some(Scancode::RAlt),
                ) => (
                    context,
                    Mode::MoveSliders {
                        active_program_index,
                    },
                ),
                (
                    Mode::Edit {
                        active_program_index,
                        cursor_position,
                        ..
                    },
                    Some(Scancode::Return),
                ) => play_waveform(
                    context,
                    status,
                    args,
                    active_program_index,
                    cursor_position,
                    &programs[active_program_index],
                    command_sender,
                    keymod,
                ),
                (
                    Mode::Edit {
                        active_program_index,
                        cursor_position,
                        ..
                    },
                    Some(Scancode::Backspace),
                ) => {
                    // If the option key is down, clear the last word
                    let mut new_cursor_position = cursor_position;
                    let mut text = programs[active_program_index].text.clone();
                    if keymod.contains(Mod::LALTMOD) {
                        if let Some(char_index) =
                            text[..cursor_position].rfind(|e: char| !e.is_whitespace())
                        {
                            if let Some(space_index) = text[..char_index].rfind(char::is_whitespace)
                            {
                                // Remove everything between that whitespace and the cursor
                                let mut new_text = text[..=space_index].to_string();
                                new_text.push_str(&text[cursor_position..]);
                                text = new_text;
                                new_cursor_position = space_index + 1;
                            } else {
                                text = text[cursor_position..].to_string();
                                new_cursor_position = 0;
                            }
                        } else {
                            // No non-whitespace characters, so clear everything before the cursor
                            text = text[cursor_position..].to_string();
                            new_cursor_position = 0;
                        }
                    } else {
                        if !text.is_empty() && cursor_position > 0 {
                            text.remove(cursor_position - 1);
                            new_cursor_position = cursor_position - 1;
                        }
                    }
                    programs[active_program_index].text = text;
                    let mode = edit_mode_from_program(
                        active_program_index,
                        new_cursor_position,
                        &programs[active_program_index].text,
                    );
                    (context, mode)
                }
                (
                    Mode::Edit {
                        active_program_index,
                        cursor_position,
                        errors,
                        ..
                    },
                    Some(Scancode::Escape),
                ) => {
                    if keymod.contains(Mod::LGUIMOD)
                        || keymod.contains(Mod::RGUIMOD)
                            && renderer::is_active_program(
                                &status,
                                Instant::now(),
                                programs[active_program_index].id,
                            )
                    {
                        // If the program is active, stop it
                        command_sender
                            .send(Command::Stop {
                                id: WaveformId::Program(programs[active_program_index].id),
                            })
                            .unwrap();
                        let message =
                            format!("Stopped program {}", programs[active_program_index].id);
                        return (
                            context,
                            Mode::Edit {
                                active_program_index,
                                cursor_position,
                                errors,
                                message,
                            },
                        );
                    }

                    // Otherwise, return to select mode
                    return (
                        context,
                        Mode::Select {
                            active_program_index,
                            message: String::new(),
                        },
                    );
                }
                (
                    Mode::Edit {
                        active_program_index,
                        cursor_position,
                        errors,
                        message,
                    },
                    Some(Scancode::Left),
                ) => {
                    let new_cursor_position;
                    if keymod.contains(Mod::LALTMOD) {
                        let text = &programs[active_program_index].text;
                        if let Some(char_index) =
                            text[..cursor_position].rfind(|e: char| !e.is_whitespace())
                        {
                            if let Some(space_index) = text[..char_index].rfind(char::is_whitespace)
                            {
                                new_cursor_position = space_index + 1;
                            } else {
                                new_cursor_position = 0;
                            }
                        } else {
                            new_cursor_position = 0;
                        }
                    } else {
                        new_cursor_position = cursor_position.saturating_sub(1);
                    }
                    (
                        context,
                        Mode::Edit {
                            active_program_index,
                            cursor_position: new_cursor_position,
                            errors,
                            message,
                        },
                    )
                }
                (
                    Mode::Edit {
                        active_program_index,
                        cursor_position,
                        errors,
                        message,
                    },
                    Some(Scancode::Right),
                ) => {
                    // TODO check for LALTMOD and move to next word
                    let cursor_position = programs[active_program_index]
                        .text
                        .len()
                        .min(cursor_position + 1);
                    (
                        context,
                        Mode::Edit {
                            active_program_index,
                            cursor_position,
                            errors,
                            message,
                        },
                    )
                }
                (
                    Mode::Edit {
                        active_program_index,
                        cursor_position: _,
                        errors,
                        message,
                    },
                    Some(Scancode::A),
                ) if keymod.contains(Mod::LCTRLMOD) || keymod.contains(Mod::RCTRLMOD) => (
                    context,
                    Mode::Edit {
                        active_program_index,
                        cursor_position: 0,
                        errors,
                        message,
                    },
                ),
                (
                    Mode::Edit {
                        active_program_index,
                        cursor_position: _,
                        errors,
                        message,
                    },
                    Some(Scancode::E),
                ) if keymod.contains(Mod::LCTRLMOD) || keymod.contains(Mod::RCTRLMOD) => (
                    context,
                    Mode::Edit {
                        active_program_index,
                        cursor_position: programs[active_program_index].text.len(),
                        errors,
                        message,
                    },
                ),

                (mode, _) => return (context, mode),
            }
        }
        Event::KeyUp {
            scancode: Some(Scancode::LAlt) | Some(Scancode::RAlt),
            ..
        } => {
            // Exit move sliders mode when the left alt key is released
            match mode {
                Mode::MoveSliders {
                    active_program_index,
                    ..
                } => {
                    // If we were in move sliders mode, return to select mode
                    (
                        context,
                        Mode::Select {
                            active_program_index,
                            message: String::new(),
                        },
                    )
                }
                _ => (context, mode),
            }
        }
        Event::TextInput { text, .. } => {
            match mode {
                Mode::Select {
                    active_program_index,
                    ..
                } => {
                    // If the text is a number less than programs.len(), update the index
                    if let Ok(new_active_program_id) = text.parse::<renderer::ProgramId>() {
                        if new_active_program_id > 0
                            && new_active_program_id as usize <= programs.len()
                        {
                            return (
                                context,
                                Mode::Select {
                                    active_program_index: index_from_id(new_active_program_id),
                                    message: String::new(),
                                },
                            );
                        } else {
                            return (
                                context,
                                Mode::Select {
                                    active_program_index,
                                    message: format!(
                                        "Invalid program id: {}",
                                        new_active_program_id
                                    ),
                                },
                            );
                        }
                    } else if text == "R" {
                        // Reload context
                        return load_context(active_program_index, &args);
                    } else if text == "L" {
                        // Load programs
                        load_programs(&args, programs);
                        send_initial_slider_values(&programs, command_sender);
                        return (
                            context,
                            Mode::Select {
                                active_program_index,
                                message: format!("Loaded programs"),
                            },
                        );
                    } else if text == "S" {
                        // Save programs
                        use std::io::Write;
                        let datetime = chrono::Local::now().format("%Y-%m-%d_%H-%M-%S");
                        let filename = format!("programs_{}.tuunp", datetime);
                        let mut file = fs::File::create(&filename).unwrap();
                        for program in programs.iter() {
                            if !program.text.is_empty() {
                                let ps = &program.sliders;
                                if !ps.configs.is_empty() {
                                    let slider_strs: Vec<String> = ps
                                        .configs
                                        .iter()
                                        .enumerate()
                                        .map(|(j, c)| {
                                            let actual =
                                                c.min + ps.normalized_values[j] * (c.max - c.min);
                                            format!(
                                                "\"{}:{}:{}:{}\"",
                                                c.label, c.min, c.max, actual
                                            )
                                        })
                                        .collect();
                                    writeln!(
                                        file,
                                        "//#{}{}{}",
                                        "{sliders=[",
                                        slider_strs.join(","),
                                        "]}"
                                    )
                                    .unwrap();
                                }
                                writeln!(file, "{}", program.text).unwrap();
                            }
                        }
                        return (
                            context,
                            Mode::Select {
                                active_program_index,
                                message: format!("Saved to {}", &filename),
                            },
                        );
                    } else if text == "D" {
                        // Dump the current waveform definition to the console
                        match play_waveform_helper(
                            &context,
                            active_program_index,
                            programs[active_program_index].text.len(),
                            &programs[active_program_index],
                            args,
                            false,
                        ) {
                            WaveformOrMode::Waveform(waveform) => {
                                println!(
                                    "Waveform definition for program {}:",
                                    programs[active_program_index].id
                                );
                                println!("{:#?}", waveform);
                            }
                            _ => (),
                        }
                        return (
                            context,
                            Mode::Select {
                                active_program_index,
                                message: format!("Dumped waveform to console"),
                            },
                        );
                    } else {
                        return (
                            context,
                            Mode::Select {
                                active_program_index,
                                message: format!("Invalid command: {}", text),
                            },
                        );
                    }
                }
                Mode::Edit {
                    active_program_index,
                    cursor_position,
                    ..
                } => {
                    let mut new_text =
                        programs[active_program_index].text[..cursor_position].to_string();
                    new_text.push_str(&text);
                    new_text.push_str(&programs[active_program_index].text[cursor_position..]);
                    programs[active_program_index].text = new_text;
                    return (
                        context,
                        edit_mode_from_program(
                            active_program_index,
                            cursor_position + text.len(),
                            &programs[active_program_index].text,
                        ),
                    );
                }
                Mode::MoveSliders { .. } | Mode::Exit => {
                    return (context, mode);
                }
            }
        }
        Event::MouseMotion { xrel, yrel, .. } => match mode {
            Mode::MoveSliders {
                active_program_index,
                ..
            } => {
                let program = &mut programs[active_program_index];
                let ps = &mut program.sliders;
                // First slider maps to mouse X axis
                if xrel != 0 && !ps.configs.is_empty() {
                    let norm = &mut ps.normalized_values[0];
                    *norm = (*norm + xrel as f32 / renderer.width as f32).clamp(0.0, 1.0);
                    let config = &ps.configs[0];
                    let actual_value = config.min + *norm * (config.max - config.min);
                    command_sender
                        .send(Command::MoveSlider {
                            slider: slider_tracker_key(program.id, &config.label),
                            value: actual_value,
                        })
                        .unwrap();
                }
                // Second slider maps to mouse Y axis
                if yrel != 0 && ps.configs.len() >= 2 {
                    let norm = &mut ps.normalized_values[1];
                    *norm = (*norm - yrel as f32 / renderer.height as f32).clamp(0.0, 1.0);
                    let config = &ps.configs[1];
                    let actual_value = config.min + *norm * (config.max - config.min);
                    command_sender
                        .send(Command::MoveSlider {
                            slider: slider_tracker_key(program.id, &config.label),
                            value: actual_value,
                        })
                        .unwrap();
                }
                return (
                    context,
                    Mode::MoveSliders {
                        active_program_index,
                    },
                );
            }
            _ => {
                return (context, mode);
            }
        },
        _ => {
            return (context, mode);
        }
    }
}

// Returns the start time of the next measure
fn next_measure_start(status: &tracker::Status<WaveformId>) -> Instant {
    for mark in &status.marks {
        match mark.waveform_id {
            WaveformId::Beats(_) if mark.mark_id == 0 && mark.start > Instant::now() => {
                return mark.start;
            }
            _ => (),
        }
    }
    panic!("No next measure found in marks");
}

enum WaveformOrMode {
    Waveform(waveform::Waveform),
    Mode(Mode),
}

fn play_waveform(
    context: Vec<(String, parser::Expr)>,
    status: &tracker::Status<WaveformId>,
    args: &Args,
    active_program_index: usize,
    cursor_position: usize,
    program: &Program,
    command_sender: &std::sync::mpsc::Sender<Command<WaveformId>>,
    keymod: sdl2::keyboard::Mod,
) -> (Vec<(String, parser::Expr)>, Mode) {
    use sdl2::keyboard::Mod;
    match play_waveform_helper(
        &context,
        active_program_index,
        cursor_position,
        program,
        args,
        true,
    ) {
        WaveformOrMode::Waveform(waveform) => {
            let message;
            let repeat_every;
            if keymod.contains(Mod::LGUIMOD) || keymod.contains(Mod::RGUIMOD) {
                // If the alt key is down, play the waveform in a loop
                let mut repeat_every_beats = args.beats_per_measure as u64;
                if keymod.contains(Mod::LSHIFTMOD) || keymod.contains(Mod::RSHIFTMOD) {
                    repeat_every_beats *= 2;
                }
                message = format!(
                    "Looping waveform {} every {:?} beats",
                    program.id, repeat_every_beats
                );
                repeat_every = Some(renderer::duration_from_beats(
                    args.tempo,
                    repeat_every_beats,
                ));
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
                        id: 0,
                        waveform: Box::new(waveform),
                    },
                    start: next_measure_start(&status),
                    repeat_every,
                })
                .unwrap();

            return (
                context,
                Mode::Select {
                    active_program_index,
                    message,
                },
            );
        }
        WaveformOrMode::Mode(new_mode) => {
            return (context, new_mode);
        }
    }
}
fn play_waveform_helper(
    context: &Vec<(String, parser::Expr)>,
    active_program_index: usize,
    cursor_position: usize,
    program: &Program,
    args: &Args,
    should_precompute: bool,
) -> WaveformOrMode {
    let program_text = prepend_native_slider_bindings(program);
    match parser::parse_program(&program_text) {
        Ok(expr) => {
            println!("Parser returned: {}", &expr);
            match parser::simplify(context, expr) {
                Ok(expr) => {
                    println!("parser::simplify returned: {}", &expr);
                    if let parser::Expr::Waveform(waveform) = expr {
                        let (_, mut waveform) = optimizer::replace_seq(waveform);
                        println!("replace_seq returned: {}", &waveform);
                        if args.optimize {
                            waveform = optimizer::simplify(waveform);
                            println!("optimizer::simplify returned: {}", &waveform);
                        }
                        if should_precompute && args.precompute {
                            let generator = generator::Generator::new(args.sample_rate);
                            waveform = waveform::remove_state(
                                generator.precompute(generator::initialize_state(waveform)),
                            );
                            println!("precompute returned: {}", &waveform);
                        }
                        return WaveformOrMode::Waveform(waveform);
                    } else {
                        println!("Expression is not a waveform, cannot play: {:#?}", expr);
                        return WaveformOrMode::Mode(Mode::Edit {
                            active_program_index,
                            cursor_position,
                            errors: vec![parser::Error::new(
                                "Expression is not a waveform".to_string(),
                            )],
                            message: format!("Not a waveform: {}", expr),
                        });
                    }
                }
                Err(error) => {
                    // If there are errors, we stay in edit mode
                    println!("Errors while simplifying input: {:?}", error);
                    let message = format!("Error: {}", error.to_string());
                    return WaveformOrMode::Mode(Mode::Edit {
                        active_program_index,
                        cursor_position,
                        errors: vec![error],
                        message: message,
                    });
                }
            }
        }
        Err(errors) => {
            // If there are errors, we stay in edit mode
            println!("Errors while parsing input: {:?}", errors);
            let message = format!("Error: {}", errors[0].to_string());
            return WaveformOrMode::Mode(Mode::Edit {
                active_program_index,
                cursor_position,
                errors,
                message,
            });
        }
    }
}
