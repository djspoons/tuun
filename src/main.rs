use core::panic;
use std::fs;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use chrono;
use clap::Parser as ClapParser;
use sdl2::audio::AudioSpecDesired;
use sdl2::event::Event;
use sdl2::ttf::Sdl2TtfContext;

use metric::Metric;
use renderer::{Mode, Renderer, WaveformId};
use tracker::Command;
use tuun::builtins;
use tuun::metric;
use tuun::optimizer;
use tuun::parser;
use tuun::renderer;
use tuun::tracker;

#[derive(ClapParser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(long = "tempo", default_value_t = 90)]
    beats_per_minute: u32,
    #[arg(long = "beats_per_measure", default_value_t = 4)]
    beats_per_measure: u32,
    #[arg(long, default_value_t = 44100)]
    sample_frequency: i32,
    #[arg(long, default_value_t = 2048)]
    buffer_size: u16,
    #[arg(short = 'C', long = "context_file", number_of_values = 1)]
    context_files: Vec<String>,
    #[arg(short = 'P', long = "programs_file", default_value = "")]
    programs_file: String,
    // Additional programs to load
    #[arg(short, long = "program", default_value = "", number_of_values = 1)]
    programs: Vec<String>,
    #[arg(long,
        num_args(0..=1), // Allows --flag or --flag=value
        action = clap::ArgAction::Set,
        default_value = "true", // Default if the flag is not present
        default_missing_value = "true")]
    optimize: bool,
}

fn load_context(program_index: usize, args: &Args) -> (Vec<(String, parser::Expr)>, Mode) {
    let mut context: Vec<(String, parser::Expr)> = Vec::new();
    context.push((
        "tempo".to_string(),
        parser::Expr::Float(args.beats_per_minute as f32),
    ));
    context.push((
        "sampling_frequency".to_string(),
        parser::Expr::Float(args.sample_frequency as f32),
    ));
    builtins::add_prelude(&mut context);
    let mut bindings = 0;
    let mut errors = Vec::new();
    for file in args.context_files.iter() {
        let raw_context = std::fs::read_to_string(file).unwrap();
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
            program_index,
            message: if errors.len() == 0 {
                format!("Loaded {} bindings from context", bindings)
            } else {
                format!("Error loading context: {}", errors[0].to_string())
            },
        },
    );
}

const NUM_PROGRAMS: usize = 8;

fn load_programs(args: &Args, programs: &mut Vec<String>) {
    *programs = Vec::new();
    if !args.programs_file.is_empty() {
        let mut count = 0;
        let contents = fs::read_to_string(&args.programs_file).unwrap_or_default();
        for line in contents.lines() {
            let line = if let Some(comment_index) = line.find("//") {
                &line[..comment_index]
            } else {
                line
            }
            .trim();
            if !line.is_empty() {
                programs.push(line.to_string());
                count += 1;
            }
        }
        println!("Loaded {} programs from {}", count, args.programs_file);
    }
    // Add in any additional programs specified on the command line
    for program in &args.programs {
        if !program.is_empty() {
            programs.push(program.to_string());
        }
    }
    // Fill up to NUM_PROGRAMS with empty strings if necessary
    while programs.len() < NUM_PROGRAMS {
        programs.push(String::new());
    }
}

pub fn main() {
    let args = Args::parse();
    let sdl_context = sdl2::init().unwrap();
    let audio_subsystem = sdl_context.audio().unwrap();
    let desired_spec = AudioSpecDesired {
        freq: Some(args.sample_frequency),
        channels: Some(1), // mono
        samples: Some(args.buffer_size),
    };

    let (status_sender, status_receiver) = mpsc::channel();
    let (command_sender, command_receiver) = mpsc::channel();

    let device = audio_subsystem
        .open_playback(None, &desired_spec, |spec| {
            println!("Spec: {:?}", spec);
            tracker::Tracker::<WaveformId>::new(
                args.sample_frequency,
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
        args.beats_per_minute,
        args.beats_per_measure,
    );
    renderer.video_subsystem.text_input().start();
    let mut event_pump = sdl_context.event_pump().unwrap();

    let (mut context, mut mode) = load_context(1, &args);
    let mut programs = Vec::new();
    load_programs(&args, &mut programs);

    start_beats(&command_sender, &status_receiver, &args);

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
            waveform: renderer::beats_waveform(args.beats_per_minute, args.beats_per_measure),
            start: Instant::now(),
            repeat_every: Some(
                renderer::duration_from_beats(args.beats_per_minute, args.beats_per_measure as u64)
                    * 2,
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
                                waveform: renderer::beats_waveform(
                                    args.beats_per_minute,
                                    args.beats_per_measure,
                                ),
                                start: mark.start + mark.duration,
                                repeat_every: Some(
                                    renderer::duration_from_beats(
                                        args.beats_per_minute,
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

fn edit_mode_from_program(program_index: usize, cursor_position: usize, program: &str) -> Mode {
    Mode::Edit {
        program_index,
        cursor_position,
        errors: if program.is_empty() {
            Vec::new()
        } else {
            match parser::parse_program(program) {
                Ok(_) => Vec::new(),
                Err(errors) => errors,
            }
        },
        message: String::new(),
    }
}

fn process_event<I>(
    args: &Args,
    context: Vec<(String, parser::Expr)>,
    renderer: &Renderer,
    event: Event,
    mode: Mode,
    status: &tracker::Status<WaveformId>,
    programs: &mut Vec<String>,
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
                (Mode::Select { program_index, .. }, Some(Scancode::Return)) => {
                    if keymod.contains(Mod::LGUIMOD) || keymod.contains(Mod::RGUIMOD) {
                        return play_waveform(
                            context,
                            status,
                            args,
                            program_index,
                            programs[program_index - 1].len(),
                            &programs[program_index - 1],
                            command_sender,
                            keymod,
                        );
                    }
                    // Check to see whether or not the current index is in the tracker's
                    // pending waveforms
                    if renderer::is_pending_program(&status, Instant::now(), program_index) {
                        // If it is, send a command to remove it.
                        command_sender
                            .send(Command::RemovePending {
                                id: WaveformId::Program(program_index),
                            })
                            .unwrap();
                    }
                    let mut mode = edit_mode_from_program(
                        program_index,
                        programs[program_index - 1].len(),
                        &programs[program_index - 1],
                    );
                    mode = match mode {
                        Mode::Edit {
                            program_index,
                            cursor_position,
                            errors,
                            ..
                        } if !errors.is_empty() => Mode::Edit {
                            program_index,
                            cursor_position,
                            message: format!("Error: {}", errors[0].to_string()),
                            errors,
                        },
                        _ => mode,
                    };
                    (context, mode)
                }
                (Mode::Select { program_index, .. }, Some(Scancode::Escape)) => {
                    let mut message = String::new();

                    if keymod.contains(Mod::LGUIMOD)
                        || keymod.contains(Mod::RGUIMOD)
                            && renderer::is_active_program(&status, Instant::now(), program_index)
                    {
                        // If the program is active, stop it
                        command_sender
                            .send(Command::Stop {
                                id: WaveformId::Program(program_index),
                            })
                            .unwrap();
                        message = format!("Stopped program {}", program_index);
                    } else if !keymod.contains(Mod::LGUIMOD)
                        && !keymod.contains(Mod::RGUIMOD)
                        && renderer::is_pending_program(&status, Instant::now(), program_index)
                    {
                        // If it is, send a command to remove it.
                        command_sender
                            .send(Command::RemovePending {
                                id: WaveformId::Program(program_index),
                            })
                            .unwrap();
                        message = format!("Removed pending waveform for program {}", program_index);
                    }
                    (
                        context,
                        Mode::Select {
                            program_index,
                            message,
                        },
                    )
                }
                (Mode::Select { program_index, .. }, Some(Scancode::Up)) => (
                    context,
                    Mode::Select {
                        program_index: (program_index + programs.len() - 2) % programs.len() + 1,
                        message: String::new(),
                    },
                ),
                (Mode::Select { program_index, .. }, Some(Scancode::Down)) => (
                    context,
                    Mode::Select {
                        program_index: (program_index) % programs.len() + 1,
                        message: String::new(),
                    },
                ),
                (
                    Mode::Select { program_index, .. },
                    Some(Scancode::LAlt) | Some(Scancode::RAlt),
                ) => (
                    context,
                    Mode::MoveSliders {
                        program_index,
                        message: String::new(),
                    },
                ),
                (
                    Mode::Edit {
                        program_index,
                        cursor_position,
                        ..
                    },
                    Some(Scancode::Return),
                ) => play_waveform(
                    context,
                    status,
                    args,
                    program_index,
                    cursor_position,
                    &programs[program_index - 1],
                    command_sender,
                    keymod,
                ),
                (
                    Mode::Edit {
                        program_index,
                        cursor_position,
                        ..
                    },
                    Some(Scancode::Backspace),
                ) => {
                    // If the option key is down, clear the last word
                    let mut new_cursor_position = cursor_position;
                    let mut program = programs[program_index - 1].clone();
                    if keymod.contains(Mod::LALTMOD) {
                        if let Some(char_index) =
                            program[..cursor_position].rfind(|e: char| !e.is_whitespace())
                        {
                            if let Some(space_index) =
                                program[..char_index].rfind(char::is_whitespace)
                            {
                                // Remove everything between that whitespace and the cursor
                                let mut new_program = program[..=space_index].to_string();
                                new_program.push_str(&program[cursor_position..]);
                                program = new_program;
                                new_cursor_position = space_index + 1;
                            } else {
                                program = program[cursor_position..].to_string();
                                new_cursor_position = 0;
                            }
                        } else {
                            // No non-whitespace characters, so clear everything before the cursor
                            program = program[cursor_position..].to_string();
                            new_cursor_position = 0;
                        }
                    } else {
                        if !program.is_empty() && cursor_position > 0 {
                            program.remove(cursor_position - 1);
                            new_cursor_position = cursor_position - 1;
                        }
                    }
                    programs[program_index - 1] = program;
                    let mode = edit_mode_from_program(
                        program_index,
                        new_cursor_position,
                        &programs[program_index - 1],
                    );
                    (context, mode)
                }
                (Mode::Edit { program_index, .. }, Some(Scancode::Escape)) => (
                    context,
                    Mode::Select {
                        program_index,
                        message: String::new(),
                    },
                ),
                (
                    Mode::Edit {
                        program_index,
                        cursor_position,
                        errors,
                        message,
                    },
                    Some(Scancode::Left),
                ) => {
                    let cursor_position = cursor_position.saturating_sub(1);
                    (
                        context,
                        Mode::Edit {
                            program_index,
                            cursor_position,
                            errors,
                            message,
                        },
                    )
                }
                (
                    Mode::Edit {
                        program_index,
                        cursor_position,
                        errors,
                        message,
                    },
                    Some(Scancode::Right),
                ) => {
                    let cursor_position =
                        programs[program_index - 1].len().min(cursor_position + 1);
                    (
                        context,
                        Mode::Edit {
                            program_index,
                            cursor_position,
                            errors,
                            message,
                        },
                    )
                }
                (
                    Mode::Edit {
                        program_index,
                        cursor_position: _,
                        errors,
                        message,
                    },
                    Some(Scancode::A),
                ) if keymod.contains(Mod::LCTRLMOD) || keymod.contains(Mod::RCTRLMOD) => (
                    context,
                    Mode::Edit {
                        program_index,
                        cursor_position: 0,
                        errors,
                        message,
                    },
                ),
                (
                    Mode::Edit {
                        program_index,
                        cursor_position: _,
                        errors,
                        message,
                    },
                    Some(Scancode::E),
                ) if keymod.contains(Mod::LCTRLMOD) || keymod.contains(Mod::RCTRLMOD) => (
                    context,
                    Mode::Edit {
                        program_index,
                        cursor_position: programs[program_index - 1].len(),
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
                Mode::MoveSliders { program_index, .. } => {
                    // If we were in move sliders mode, return to select mode
                    (
                        context,
                        Mode::Select {
                            program_index,
                            message: String::new(),
                        },
                    )
                }
                _ => (context, mode),
            }
        }
        Event::TextInput { text, .. } => {
            match mode {
                Mode::Select { program_index, .. } => {
                    // If the text is a number less than programs.len(), update the index
                    if let Ok(new_program_index) = text.parse::<usize>() {
                        if new_program_index > 0 && new_program_index <= programs.len() {
                            return (
                                context,
                                Mode::Select {
                                    program_index: new_program_index,
                                    message: String::new(),
                                },
                            );
                        } else {
                            return (
                                context,
                                Mode::Select {
                                    program_index,
                                    message: format!(
                                        "Invalid program index: {}",
                                        new_program_index
                                    ),
                                },
                            );
                        }
                    } else if text == "R" {
                        // Reload context
                        return load_context(program_index, &args);
                    } else if text == "L" {
                        // Load programs
                        load_programs(&args, programs);
                        return (
                            context,
                            Mode::Select {
                                program_index,
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
                            if !program.is_empty() {
                                writeln!(file, "{}", program).unwrap();
                            }
                        }
                        return (
                            context,
                            Mode::Select {
                                program_index,
                                message: format!("Saved to {}", &filename),
                            },
                        );
                    } else if text == "D" {
                        // Dump the current waveform definition to the console
                        match play_waveform_helper(
                            &context,
                            program_index,
                            programs[program_index - 1].len(),
                            programs[program_index - 1].as_str(),
                            args.optimize,
                        ) {
                            WaveformOrMode::Waveform(waveform) => {
                                println!("Waveform definition for program {}:", program_index);
                                println!("{:#?}", waveform);
                            }
                            _ => (),
                        }
                        return (
                            context,
                            Mode::Select {
                                program_index,
                                message: format!("Dumped waveform to console"),
                            },
                        );
                    } else {
                        return (
                            context,
                            Mode::Select {
                                program_index,
                                message: format!("Invalid command: {}", text),
                            },
                        );
                    }
                }
                Mode::Edit {
                    program_index,
                    cursor_position,
                    ..
                } => {
                    let mut new_program =
                        programs[program_index - 1][..cursor_position].to_string();
                    new_program.push_str(&text);
                    new_program.push_str(&programs[program_index - 1][cursor_position..]);
                    programs[program_index - 1] = new_program;
                    return (
                        context,
                        edit_mode_from_program(
                            program_index,
                            cursor_position + text.len(),
                            &programs[program_index - 1],
                        ),
                    );
                }
                Mode::MoveSliders { .. } | Mode::Exit => {
                    return (context, mode);
                }
            }
        }
        Event::MouseMotion { xrel, yrel, .. } => {
            use tracker::Slider;
            match mode {
                Mode::MoveSliders { program_index, .. } => {
                    if xrel != 0 {
                        command_sender
                            .send(Command::MoveSlider {
                                slider: Slider::X,
                                delta: xrel as f32 / (renderer.width as f32 / 2.0),
                            })
                            .unwrap();
                    }
                    if yrel != 0 {
                        command_sender
                            .send(Command::MoveSlider {
                                slider: Slider::Y,
                                delta: -yrel as f32 / (renderer.height as f32 / 2.0),
                            })
                            .unwrap();
                    }
                    return (
                        context,
                        Mode::MoveSliders {
                            program_index,
                            message: String::new(),
                        },
                    );
                }
                _ => {
                    return (context, mode);
                }
            }
        }
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
    Waveform(tracker::Waveform),
    Mode(Mode),
}

fn play_waveform(
    context: Vec<(String, parser::Expr)>,
    status: &tracker::Status<WaveformId>,
    args: &Args,
    program_index: usize,
    cursor_position: usize,
    program: &str,
    command_sender: &std::sync::mpsc::Sender<Command<WaveformId>>,
    keymod: sdl2::keyboard::Mod,
) -> (Vec<(String, parser::Expr)>, Mode) {
    use sdl2::keyboard::Mod;
    match play_waveform_helper(
        &context,
        program_index,
        cursor_position,
        program,
        args.optimize,
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
                    program_index, repeat_every_beats
                );
                repeat_every = Some(renderer::duration_from_beats(
                    args.beats_per_minute,
                    repeat_every_beats,
                ));
            } else {
                // Otherwise, play it once
                message = format!("Playing waveform {}", program_index);
                repeat_every = None;
            }
            command_sender
                .send(Command::Play {
                    // TODO maybe extend the mark to the full measure?
                    id: WaveformId::Program(program_index),
                    waveform: tracker::Waveform::Marked {
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
                    program_index,
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
    program_index: usize,
    cursor_position: usize,
    program: &str,
    optimize: bool,
) -> WaveformOrMode {
    match parser::parse_program(program) {
        Ok(expr) => {
            println!("Parser returned: {:}", &expr);
            match parser::simplify(context, expr) {
                Ok(expr) => {
                    println!("parser::simplify returned: {:}", &expr);
                    if let parser::Expr::Waveform(waveform) = expr {
                        let (_, mut waveform) = optimizer::replace_seq(waveform);
                        println!("replace_seq returned: {:?}", &waveform);
                        if optimize {
                            waveform = optimizer::simplify(waveform);
                            println!("optimizer::simplify returned: {:?}", &waveform);
                        }
                        return WaveformOrMode::Waveform(waveform);
                    } else {
                        println!("Expression is not a waveform, cannot play: {:#?}", expr);
                        return WaveformOrMode::Mode(Mode::Edit {
                            program_index,
                            cursor_position,
                            errors: vec![parser::Error::new(
                                "Expression is not a waveform".to_string(),
                            )],
                            message: format!("Not a waveform: {}", expr.to_string()),
                        });
                    }
                }
                Err(error) => {
                    // If there are errors, we stay in edit mode
                    println!("Errors while simplifying input: {:?}", error);
                    let message = format!("Error: {}", error.to_string());
                    return WaveformOrMode::Mode(Mode::Edit {
                        program_index,
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
                program_index,
                cursor_position,
                errors,
                message,
            });
        }
    }
}
