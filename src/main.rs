use std::time::{Duration, Instant};

extern crate sdl2;
use sdl2::audio::AudioSpecDesired;
use sdl2::event::Event;
use sdl2::ttf::Sdl2TtfContext;

use clap::Parser as ClapParser;

mod builtins;
mod metric;
use metric::Metric;
mod parser;
mod renderer;
mod tracker;
use tracker::Command;

#[derive(ClapParser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(long = "tempo", default_value_t = 90)]
    beats_per_minute: u32,
    #[arg(long = "beats_per_measure", default_value_t = 4)]
    beats_per_measure: u32,
    #[arg(long, default_value_t = 44100)]
    sample_frequency: i32,
    #[arg(short, long = "program", default_value = "", number_of_values = 1)]
    programs: Vec<String>,
    #[arg(short = 'C', long = "context_file", number_of_values = 1)]
    context: Vec<String>,
}

#[derive(Debug)]
enum Mode {
    Select {
        index: usize,
        message: String,
    },
    Edit {
        index: usize,
        // TODO unicode!!
        cursor_position: usize, // Cursor is located before the character this position
        errors: Vec<parser::Error>,
        message: String,
    },
    TurnDials {
        index: usize, // Don't forget this
        message: String,
    },
    Exit,
}

fn load_context(index: usize, args: &Args) -> (Vec<(String, parser::Expr)>, Mode) {
    let mut context: Vec<(String, parser::Expr)> = Vec::new();
    context.push((
        "tempo".to_string(),
        parser::Expr::Float(args.beats_per_minute as f32),
    ));
    builtins::add_prelude(&mut context);
    let mut bindings = 0;
    let mut errors = Vec::new();
    for file in args.context.iter() {
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
            index,
            message: if errors.len() == 0 {
                format!("Loaded {} bindings from context", bindings)
            } else {
                format!("Error loading context: {}", errors[0].to_string())
            },
        },
    );
}

const NUM_PROGRAMS: usize = 8;

pub fn main() {
    let args = Args::parse();
    let sdl_context = sdl2::init().unwrap();
    let audio_subsystem = sdl_context.audio().unwrap();
    let desired_spec = AudioSpecDesired {
        freq: Some(args.sample_frequency),
        channels: Some(1), // mono
        samples: None,     // default buffer size
    };

    let (status_sender, status_receiver) = std::sync::mpsc::channel();
    let (command_sender, command_receiver) = std::sync::mpsc::channel();

    let device = audio_subsystem
        .open_playback(None, &desired_spec, |spec| {
            println!("Spec: {:?}", spec);
            tracker::Tracker::new(
                args.sample_frequency,
                args.beats_per_minute,
                command_receiver,
                status_sender,
            )
        })
        .unwrap();
    device.resume();

    let ttf_context: Sdl2TtfContext = sdl2::ttf::init().unwrap();
    let mut renderer = renderer::Renderer::new(
        &sdl_context,
        &ttf_context,
        args.beats_per_minute,
        args.beats_per_measure,
    );

    let (mut context, mut mode) = load_context(0, &args);
    let mut programs = args.programs.clone();
    while programs.len() < NUM_PROGRAMS {
        programs.push(String::new());
    }
    let mut status = tracker::Status {
        active_waveforms: Vec::new(),
        pending_waveforms: Vec::new(),
        current_beat: 0,
        next_beat_start: Instant::now()
            + Duration::from_secs_f32(1.0 / (args.beats_per_minute as f32 * 60.0)),
        buffer: None,
        tracker_load: None,
    };
    let mut metrics = renderer::Metrics {
        tracker_load: Metric::new(std::time::Duration::from_secs(10), 100),
    };

    renderer.video_subsystem.text_input().start();
    let mut event_pump = sdl_context.event_pump().unwrap();
    const BUFFER_REFRESH_INTERVAL: Duration = Duration::from_millis(200);
    let mut next_buffer_refresh = Instant::now();
    command_sender.send(Command::SendCurrentBuffer).unwrap();
    loop {
        for event in event_pump.poll_iter() {
            //println!("Event: {:?} with mode {:?}", event, mode);
            (context, mode) = process_event(
                &args,
                context,
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

        match status_receiver.recv_timeout(Duration::from_millis(10)) {
            Ok(tracker_status) => {
                // TODO Meh...
                status.active_waveforms = tracker_status.active_waveforms;
                status.pending_waveforms = tracker_status.pending_waveforms;
                status.current_beat = tracker_status.current_beat;
                status.next_beat_start = tracker_status.next_beat_start;
                if let Some(ratio) = tracker_status.tracker_load {
                    metrics.tracker_load.set(ratio);
                }
                match tracker_status.buffer {
                    Some(_) => status.buffer = tracker_status.buffer,
                    _ => (),
                }
                renderer.render(&ttf_context, &programs, &status, &mode, &mut metrics);
            }
            Err(_) => {}
        }
    }
}

fn edit_mode_from_program(index: usize, cursor_position: usize, program: &str) -> Mode {
    Mode::Edit {
        index,
        cursor_position,
        errors: match parser::parse_program(program) {
            Ok(_) => Vec::new(),
            Err(errors) => errors,
        },
        message: String::new(),
    }
}

fn process_event(
    args: &Args,
    context: Vec<(String, parser::Expr)>,
    event: Event,
    mode: Mode,
    status: &tracker::Status,
    programs: &mut Vec<String>,
    command_sender: &std::sync::mpsc::Sender<Command>,
) -> (Vec<(String, parser::Expr)>, Mode) {
    use sdl2::keyboard::Mod;
    use sdl2::keyboard::Scancode;
    match event {
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
                (Mode::Select { index, .. }, Some(Scancode::Return)) => {
                    if keymod.contains(Mod::LGUIMOD) || keymod.contains(Mod::RGUIMOD) {
                        return play_waveform(
                            context,
                            status,
                            args,
                            index,
                            programs[index].len(),
                            &programs[index],
                            command_sender,
                            keymod,
                        );
                    }
                    // Check to see whether or not the current index is in the tracker's
                    // pending waveforms
                    if status
                        .pending_waveforms
                        .iter()
                        .any(|w| w.id == index as u32)
                    {
                        // If it is, send a command to remove it.
                        command_sender
                            .send(Command::RemovePending { id: index as u32 })
                            .unwrap();
                    }
                    let mut mode =
                        edit_mode_from_program(index, programs[index].len(), &programs[index]);
                    mode = match mode {
                        Mode::Edit {
                            index,
                            cursor_position,
                            errors,
                            ..
                        } if !errors.is_empty() => Mode::Edit {
                            index,
                            cursor_position,
                            message: format!("Error: {}", errors[0].to_string()),
                            errors,
                        },
                        _ => mode,
                    };
                    (context, mode)
                }
                (Mode::Select { index, .. }, Some(Scancode::Escape)) => {
                    let mut message = String::new();
                    // Remove the current waveform from the pending waveforms
                    if status
                        .pending_waveforms
                        .iter()
                        .any(|w| w.id == index as u32)
                    {
                        // If it is, send a command to remove it.
                        command_sender
                            .send(Command::RemovePending { id: index as u32 })
                            .unwrap();
                        message = format!("Removed pending waveform for program {}", index);
                    }
                    (context, Mode::Select { index, message })
                }
                (Mode::Select { index, .. }, Some(Scancode::Up)) => (
                    context,
                    Mode::Select {
                        index: (index + programs.len() - 1) % programs.len(),
                        message: String::new(),
                    },
                ),
                (Mode::Select { index, .. }, Some(Scancode::Down)) => (
                    context,
                    Mode::Select {
                        index: (index + 1) % programs.len(),
                        message: String::new(),
                    },
                ),
                (Mode::Select { index, .. }, Some(Scancode::LAlt) | Some(Scancode::RAlt)) => (
                    context,
                    Mode::TurnDials {
                        index,
                        message: String::new(),
                    },
                ),
                (
                    Mode::Edit {
                        index,
                        cursor_position,
                        ..
                    },
                    Some(Scancode::Return),
                ) => play_waveform(
                    context,
                    status,
                    args,
                    index,
                    cursor_position,
                    &programs[index],
                    command_sender,
                    keymod,
                ),
                (
                    Mode::Edit {
                        index,
                        cursor_position,
                        ..
                    },
                    Some(Scancode::Backspace),
                ) => {
                    // If the option key is down, clear the last word
                    let mut new_cursor_position = cursor_position;
                    let mut program = programs[index].clone();
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
                    programs[index] = program;
                    let mode = edit_mode_from_program(index, new_cursor_position, &programs[index]);
                    (context, mode)
                }
                (Mode::Edit { index, .. }, Some(Scancode::Escape)) => (
                    context,
                    Mode::Select {
                        index: index,
                        message: String::new(),
                    },
                ),
                (
                    Mode::Edit {
                        index,
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
                            index,
                            cursor_position,
                            errors,
                            message,
                        },
                    )
                }
                (
                    Mode::Edit {
                        index,
                        cursor_position,
                        errors,
                        message,
                    },
                    Some(Scancode::Right),
                ) => {
                    let cursor_position = programs[index].len().min(cursor_position + 1);
                    (
                        context,
                        Mode::Edit {
                            index,
                            cursor_position,
                            errors,
                            message,
                        },
                    )
                }

                (mode, _) => return (context, mode),
            }
        }
        Event::KeyUp {
            scancode: Some(Scancode::LAlt) | Some(Scancode::RAlt),
            ..
        } => {
            // Exit turn dials mode when the left alt key is released
            match mode {
                Mode::TurnDials { index, .. } => {
                    // If we were in turn dials mode, return to select mode
                    (
                        context,
                        Mode::Select {
                            index,
                            message: String::new(),
                        },
                    )
                }
                _ => (context, mode),
            }
        }
        Event::TextInput { text, .. } => {
            match mode {
                Mode::Select { index, .. } => {
                    // If the text is a number less than programs.len(), update the index
                    if let Ok(index) = text.parse::<usize>() {
                        if index > 0 && index <= programs.len() {
                            return (
                                context,
                                Mode::Select {
                                    index: (index + programs.len() - 1) % programs.len(),
                                    message: String::new(),
                                },
                            );
                        } else {
                            return (
                                context,
                                Mode::Select {
                                    index,
                                    message: format!("Invalid program index: {}", index),
                                },
                            );
                        }
                    } else if text == "r" {
                        // Reload context
                        return load_context(index, &args);
                    } else if text == "w" {
                        // Write waveform
                        let (status_sender, _status_receiver) = std::sync::mpsc::channel();
                        let (command_sender, command_receiver) = std::sync::mpsc::channel();
                        let mut tmp = tracker::Tracker::new(
                            args.sample_frequency,
                            args.beats_per_minute,
                            command_receiver,
                            status_sender,
                        );
                        match play_waveform_helper(
                            &context,
                            index,
                            programs[index].len(),
                            &programs[index],
                        ) {
                            WaveformOrMode::Waveform(waveform) => {
                                command_sender
                                    .send(Command::PlayOnce {
                                        id: index as u32,
                                        waveform,
                                        at_beat: None,
                                    })
                                    .unwrap();
                            }
                            WaveformOrMode::Mode(new_mode) => {
                                return (context, new_mode);
                            }
                        }
                        let filename = format!("program_{}.wav", index + 1);
                        match tmp.write_to_file(&filename) {
                            Ok(_) => (
                                context,
                                Mode::Select {
                                    index,
                                    message: format!(
                                        "Waveform {} written to file {}",
                                        index + 1,
                                        filename
                                    ),
                                },
                            ),
                            Err(e) => (
                                context,
                                Mode::Select {
                                    index,
                                    message: format!("Error writing waveform {}: {}", index, e),
                                },
                            ),
                        }
                    } else if text == "p" {
                        // Print programs
                        for program in programs.iter() {
                            println!(" -p {} \\", program);
                        }
                        return (
                            context,
                            Mode::Select {
                                index,
                                message: "Printed programs to console".to_string(),
                            },
                        );
                    } else {
                        return (
                            context,
                            Mode::Select {
                                index,
                                message: format!("Invalid command: {}", text),
                            },
                        );
                    }
                }
                Mode::Edit {
                    index,
                    cursor_position,
                    ..
                } => {
                    let mut new_program = programs[index][..cursor_position].to_string();
                    new_program.push_str(&text);
                    new_program.push_str(&programs[index][cursor_position..]);
                    programs[index] = new_program;
                    return (
                        context,
                        edit_mode_from_program(
                            index,
                            cursor_position + text.len(),
                            &programs[index],
                        ),
                    );
                }
                Mode::TurnDials { .. } | Mode::Exit => {
                    return (context, mode);
                }
            }
        }
        Event::MouseMotion { xrel, yrel, .. } => {
            use tracker::Dial;
            match mode {
                Mode::TurnDials { index, .. } => {
                    //println!("Mouse motion: xrel: {}, yrel: {}", xrel, yrel);
                    if xrel != 0 {
                        command_sender
                            .send(Command::TurnDial {
                                dial: Dial::X,
                                delta: xrel as f32 / 100.0,
                            })
                            .unwrap();
                    }
                    if yrel != 0 {
                        command_sender
                            .send(Command::TurnDial {
                                dial: Dial::Y,
                                delta: yrel as f32 / 100.0,
                            })
                            .unwrap();
                    }
                    return (
                        context,
                        Mode::TurnDials {
                            index,
                            message: format!("Turned dials by xrel: {}, yrel: {}", xrel, yrel),
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

// Returns the first beat of the next measure
fn next_measure_beat(args: &Args, status: &tracker::Status) -> u64 {
    return ((status.current_beat + 1) / args.beats_per_measure as u64 + 1)
        * args.beats_per_measure as u64;
}

enum WaveformOrMode {
    Waveform(tracker::Waveform),
    Mode(Mode),
}

fn play_waveform(
    context: Vec<(String, parser::Expr)>,
    status: &tracker::Status,
    args: &Args,
    index: usize,
    cursor_position: usize,
    program: &str,
    command_sender: &std::sync::mpsc::Sender<Command>,
    keymod: sdl2::keyboard::Mod,
) -> (Vec<(String, parser::Expr)>, Mode) {
    use sdl2::keyboard::Mod;
    match play_waveform_helper(&context, index, cursor_position, program) {
        WaveformOrMode::Waveform(waveform) => {
            let message: String;
            if keymod.contains(Mod::LGUIMOD) || keymod.contains(Mod::RGUIMOD) {
                // If the alt key is down, play the waveform in a loop
                let repeat_after_beats =
                    if keymod.contains(Mod::LSHIFTMOD) || keymod.contains(Mod::RSHIFTMOD) {
                        args.beats_per_measure as u64 * 2
                    } else {
                        args.beats_per_measure as u64
                    };
                command_sender
                    .send(Command::PlayInLoop {
                        id: index as u32,
                        waveform,
                        at_beat: next_measure_beat(&args, &status),
                        repeat_after_beats,
                    })
                    .unwrap();
                message = format!(
                    "Looping waveform {} every {} beats",
                    index + 1,
                    repeat_after_beats
                );
            } else {
                // Otherwise, play it once
                command_sender
                    .send(Command::PlayOnce {
                        id: index as u32,
                        waveform,
                        at_beat: Some(next_measure_beat(&args, &status)),
                    })
                    .unwrap();
                message = format!("Playing waveform {}", index + 1);
            }
            return (context, Mode::Select { index, message });
        }
        WaveformOrMode::Mode(new_mode) => {
            return (context, new_mode);
        }
    }
}
fn play_waveform_helper(
    context: &Vec<(String, parser::Expr)>,
    index: usize,
    cursor_position: usize,
    program: &str,
) -> WaveformOrMode {
    match parser::parse_program(program) {
        Ok(expr) => {
            println!("Parser returned: {:}", &expr);
            match parser::simplify(context, expr) {
                Ok(expr) => {
                    println!("Simplify returned: {:}", &expr);
                    if let parser::Expr::Waveform(waveform) = expr {
                        return WaveformOrMode::Waveform(waveform);
                    } else {
                        println!("Expression is not a waveform, cannot play: {:#?}", expr);
                        return WaveformOrMode::Mode(Mode::Edit {
                            index,
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
                        index,
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
                index,
                cursor_position,
                errors,
                message,
            });
        }
    }
}
