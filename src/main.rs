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
    #[arg(short = 'C', long = "context_file")]
    context: String,
}

#[derive(Debug)]
enum Mode {
    Select {
        index: usize,
    },
    Edit {
        index: usize,
        // TODO unicode!!
        cursor_position: usize, // Cursor is located before the character this position
        errors: Vec<parser::Error>,
    },
    TurnDials {
        index: usize, // Don't forget this
    },
    Exit,
}

fn load_context(args: &Args) -> Vec<(String, parser::Expr)> {
    let mut context: Vec<(String, parser::Expr)> = Vec::new();
    context.push((
        "tempo".to_string(),
        parser::Expr::Float(args.beats_per_minute as f32),
    ));
    builtins::add_prelude(&mut context);
    if args.context != "" {
        let raw_context = std::fs::read_to_string(&args.context).unwrap();
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
                println!("Parsed context:");
                for (name, parsed_expr) in parsed_exprs {
                    match parser::simplify(&context, parsed_expr) {
                        Ok(expr) => {
                            println!("   {}", &name);
                            context.push((name.trim().to_string(), expr));
                        }
                        Err(error) => println!(
                            "Error simplifying context expression for {}: {:?}",
                            name, error
                        ),
                    }
                }
            }
            Err(errors) => {
                println!("Errors parsing context: {:?}", errors);
            }
        }
    }
    return context;
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

    let mut context = load_context(&args);
    let mut programs = args.programs.clone();
    while programs.len() < NUM_PROGRAMS {
        programs.push(String::new());
    }
    let mut mode = Mode::Select { index: 0 };
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
    }
}

fn process_event(
    args: &Args,
    mut context: Vec<(String, parser::Expr)>,
    event: Event,
    mode: Mode,
    status: &tracker::Status,
    programs: &mut Vec<String>,
    command_sender: &std::sync::mpsc::Sender<Command>,
) -> (Vec<(String, parser::Expr)>, Mode) {
    match event {
        Event::Quit { .. } => return (context, Mode::Exit),
        Event::KeyDown {
            scancode, keymod, ..
        } => {
            match (mode, scancode) {
                // Exit on control-C
                (mode, Some(sdl2::keyboard::Scancode::C)) => {
                    if keymod.contains(sdl2::keyboard::Mod::LCTRLMOD)
                        || keymod.contains(sdl2::keyboard::Mod::RCTRLMOD)
                    {
                        return (context, Mode::Exit);
                    } else {
                        return (context, mode);
                    }
                }
                (Mode::Select { index }, Some(sdl2::keyboard::Scancode::Return)) => {
                    // Check to see whether or not the current index is in the tracker's
                    // pending waveforms
                    if status
                        .pending_waveforms
                        .iter()
                        .any(|w| w.id == index as u32)
                    {
                        // If it is, we just stay in select mode
                        return (context, Mode::Select { index });
                    }
                    return (
                        context,
                        edit_mode_from_program(index, programs[index].len(), &programs[index]),
                    );
                }
                (Mode::Select { index }, Some(sdl2::keyboard::Scancode::Up)) => {
                    return (
                        context,
                        Mode::Select {
                            index: (index + programs.len() - 1) % programs.len(),
                        },
                    );
                }
                (Mode::Select { index }, Some(sdl2::keyboard::Scancode::Down)) => {
                    return (
                        context,
                        Mode::Select {
                            index: (index + 1) % programs.len(),
                        },
                    );
                }
                (Mode::Select { index }, Some(sdl2::keyboard::Scancode::LGui)) => {
                    return (context, Mode::TurnDials { index });
                }
                (
                    Mode::Edit {
                        index,
                        cursor_position,
                        ..
                    },
                    Some(sdl2::keyboard::Scancode::Return),
                ) => {
                    match play_waveform_helper(&context, index, cursor_position, &programs[index]) {
                        WaveformOrMode::Waveform(waveform) => {
                            command_sender
                                .send(Command::PlayOnce {
                                    id: index as u32,
                                    waveform,
                                    at_beat: Some(
                                        ((status.current_beat + 1) / args.beats_per_measure as u64
                                            + 1)
                                            * args.beats_per_measure as u64,
                                    ),
                                })
                                .unwrap();
                            return (context, Mode::Select { index });
                        }
                        WaveformOrMode::Mode(new_mode) => {
                            return (context, new_mode);
                        }
                    }
                }
                (
                    Mode::Edit {
                        index,
                        cursor_position,
                        ..
                    },
                    Some(sdl2::keyboard::Scancode::Backspace),
                ) => {
                    // If the option key is down, clear the last word
                    let mut new_cursor_position = cursor_position;
                    let mut program = programs[index].clone();
                    if keymod.contains(sdl2::keyboard::Mod::LALTMOD) {
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
                    return (
                        context,
                        edit_mode_from_program(index, new_cursor_position, &programs[index]),
                    );
                }
                (Mode::Edit { index, .. }, Some(sdl2::keyboard::Scancode::Escape)) => {
                    return (context, Mode::Select { index: index });
                }
                (
                    Mode::Edit {
                        index,
                        cursor_position,
                        errors,
                    },
                    Some(sdl2::keyboard::Scancode::Left),
                ) => {
                    return (
                        context,
                        Mode::Edit {
                            index,
                            cursor_position: cursor_position.saturating_sub(1),
                            errors,
                        },
                    );
                }
                (
                    Mode::Edit {
                        index,
                        cursor_position,
                        errors,
                    },
                    Some(sdl2::keyboard::Scancode::Right),
                ) => {
                    return (
                        context,
                        Mode::Edit {
                            index,
                            cursor_position: programs[index].len().min(cursor_position + 1),
                            errors,
                        },
                    );
                }

                (mode, _) => return (context, mode),
            }
        }
        Event::KeyUp {
            scancode: Some(sdl2::keyboard::Scancode::LGui),
            ..
        } => {
            // Exit turn dials mode when the left alt key is released
            match mode {
                Mode::TurnDials { index } => {
                    // If we were in turn dials mode, return to select mode
                    return (context, Mode::Select { index });
                }
                _ => {
                    return (context, mode);
                }
            }
        }
        Event::TextInput { text, .. } => {
            match mode {
                Mode::Select { index } => {
                    // If the text is a number less than programs.len(), update the index
                    if let Ok(index) = text.parse::<usize>() {
                        if index <= programs.len() {
                            return (
                                context,
                                Mode::Select {
                                    index: (index + programs.len() - 1) % programs.len(),
                                },
                            );
                        } else {
                            println!("Invalid program index: {}", index);
                        }
                    } else if text == "r" {
                        context = load_context(&args);
                    } else if text == "w" {
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
                        let filename = format!("program_{}.wav", index);
                        match tmp.write_to_file(&filename) {
                            Ok(_) => println!("Wrote program {} to {}", index, filename),
                            Err(e) => println!("Error writing program {}: {}", index, e),
                        }
                        return (context, mode);
                    } else {
                        println!("Invalid command in select mode: {}", text);
                    }
                    return (context, mode);
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
                Mode::TurnDials { .. } => {
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
                    return (context, mode);
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

enum WaveformOrMode {
    Waveform(tracker::Waveform<()>),
    Mode(Mode),
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
                        });
                    }
                }
                Err(error) => {
                    // If there are errors, we stay in edit mode
                    println!("Errors while simplifying input: {:?}", error);
                    return WaveformOrMode::Mode(Mode::Edit {
                        index,
                        cursor_position,
                        errors: vec![error],
                    });
                }
            }
        }
        Err(errors) => {
            // If there are errors, we stay in edit mode
            println!("Errors while parsing input: {:?}", errors);
            return WaveformOrMode::Mode(Mode::Edit {
                index,
                cursor_position,
                errors,
            });
        }
    }
}
