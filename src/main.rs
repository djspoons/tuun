use std::time::Duration;

extern crate sdl2;
use sdl2::event::Event;
use sdl2::audio::AudioSpecDesired;
use sdl2::render::{TextureCreator, TextureQuery};
use sdl2::pixels::Color;
use sdl2::ttf::Font;
use sdl2::video::WindowContext;

use clap::Parser as ClapParser;

mod parser;
mod sequence;

enum Command {
    PlayOnce {
        node: parser::Expr,
        beat: i32, // Offset in beats from the beginning
    },
}

fn make_texture<'a>(font: &Font<'a, 'static>, color: Color, texture_creator: &'a TextureCreator<WindowContext>, s: &str) -> sdl2::render::Texture<'a> {
    let surface = font
        .render(s)
        .blended(color)
        .map_err(|e| e.to_string()).unwrap();
    let texture = texture_creator
        .create_texture_from_surface(&surface)
        .map_err(|e| e.to_string()).unwrap();
    return texture;
}

#[derive(ClapParser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(long = "bpm", default_value_t = 90)]
    beats_per_minute: i32,
    #[arg(long, default_value_t = 44100)]
    sample_frequency: i32,
    #[arg(short, long = "program", default_value = "", number_of_values = 1)]
    programs: Vec<String>,
}

#[derive(Debug)]
enum Mode {
    Select { index: usize },
    Edit { index: usize, errors: Vec<parser::ParseError> },
    Exit,
}

const NUM_PROGRAMS: usize = 5;

pub fn main() {
    let args = Args::parse();
    let sdl_context = sdl2::init().unwrap();
    let audio_subsystem = sdl_context.audio().unwrap();
    let desired_spec = AudioSpecDesired {
        freq: Some(args.sample_frequency),
        channels: Some(1),  // mono
        samples: None       // default sample size
    };

    let (sample_sender, sample_receiver) = std::sync::mpsc::channel();
    let (command_sender, command_receiver) = std::sync::mpsc::channel();

    let device = 
        audio_subsystem.open_playback(None, &desired_spec, 
            |spec| {
                println!("Spec: {:?}", spec);
                sequence::new_sequencer(args.sample_frequency,
                    args.beats_per_minute,
                    command_receiver,
                    sample_sender)
            }).unwrap();
    device.resume();

    let video_subsystem = sdl_context.video().unwrap();
    let ttf_context = sdl2::ttf::init().map_err(|e| e.to_string()).unwrap();
    let width = 1200;
    let height = 600;
    let font_path = "/Library/Fonts/Arial Unicode.ttf";
    let window = video_subsystem
        .window("tuunel", width, height)
        .position_centered()
        .build()
        .map_err(|e| e.to_string()).unwrap();
    let mut canvas = window.into_canvas().build().map_err(
        |e| e.to_string()).unwrap();
    let texture_creator = canvas.texture_creator();
    let font = ttf_context.load_font(font_path, 64).unwrap();
    const INACTIVE_COLOR: Color = Color::RGBA(0, 255, 0, 255);
    const EDIT_COLOR: Color = Color::RGBA(0, 255, 255, 255);
    const ERROR_COLOR: Color = Color::RGBA(255, 0, 0, 255);

    let prompt_texture = make_texture(&font, INACTIVE_COLOR, &texture_creator, " ▸ ");
    let TextureQuery { width: prompt_width, height: line_height, .. } = prompt_texture.query();
    let number_texture = make_texture(&font, INACTIVE_COLOR, &texture_creator, "① " );
    let TextureQuery { width: number_width, .. } = number_texture.query();
    let nav_width = prompt_width + number_width;

    let mut programs = args.programs;
    while programs.len() < NUM_PROGRAMS {
        programs.push(String::new());
    }
    let mut mode = Mode::Select { index: 0 };

    video_subsystem.text_input().start();
    let mut event_pump = sdl_context.event_pump().unwrap();
    loop {
        for event in event_pump.poll_iter() {
            println!("Event: {:?} with mode {:?}", event, mode);
            mode = process_event(event, mode, &mut programs, &command_sender);
            if let Mode::Exit = mode {
                return;
            }
        }

        match sample_receiver.recv_timeout(Duration::new(0, 1_000_000)) {
            Ok(out) => {
                canvas.set_draw_color(Color::RGB(0, 0, 0));
                canvas.clear();

                let mut y = 10;
                for (i, program) in programs.iter().enumerate() {
                    let color = match mode {
                        Mode::Edit { index, .. } if i == index => EDIT_COLOR,
                        _ => INACTIVE_COLOR,
                    };
                    let number = char::from_u32(0x2460 + i as u32).unwrap().to_string();
                    let number_texture = make_texture(&font, color, &texture_creator, &number);
                    let TextureQuery { width: number_width, .. } = number_texture.query();
                    match mode {
                        Mode::Edit { index, ref errors } => {
                            canvas.copy(&number_texture, None, Some(sdl2::rect::Rect::new(prompt_width as i32, y, number_width, line_height))).unwrap();
                            if i != index && !program.is_empty() {
                                let text_texture = make_texture(&font, INACTIVE_COLOR, &texture_creator, program);
                                let TextureQuery { width: text_width, height: text_height, .. } = text_texture.query();
                                canvas.copy(&text_texture, None, Some(sdl2::rect::Rect::new(nav_width as i32, y, text_width, text_height))).unwrap();
                            } else if i == index {
                                // Loop over each character in program and check to see if it's in any of the error
                                // ranges
                                let mut x = nav_width as i32;
                                for (j, c) in program.chars().enumerate() {
                                    let color = if errors.iter().any(|e| e.range().contains(&j)) {
                                        ERROR_COLOR
                                    } else {
                                        EDIT_COLOR
                                    };
                                    let char_texture = make_texture(&font, color, &texture_creator, &c.to_string());
                                    let TextureQuery { width: char_width, height: char_height, .. } = char_texture.query();
                                    canvas.copy(&char_texture, None, Some(sdl2::rect::Rect::new(x, y, char_width, char_height))).unwrap();
                                    x += char_width as i32;
                                }
                                let color = if !errors.is_empty() {
                                    ERROR_COLOR
                                } else {
                                    EDIT_COLOR
                                };
                                let cursor_texture = make_texture(&font, color, &texture_creator, "‸");
                                let TextureQuery { width: cursor_width, height: cursor_height, .. } = cursor_texture.query();
                                canvas.copy(&cursor_texture, None, Some(sdl2::rect::Rect::new(x, y, cursor_width, cursor_height))).unwrap();
                            }
                        },
                        Mode::Select { index } => {
                            if index == i {
                                canvas.copy(&prompt_texture, None, Some(sdl2::rect::Rect::new(0, y, prompt_width, line_height))).unwrap();
                            }
                            canvas.copy(&number_texture, None, Some(sdl2::rect::Rect::new(prompt_width as i32, y, number_width, line_height))).unwrap();
                            if !program.is_empty() {
                                let text_texture = make_texture(&font, INACTIVE_COLOR, &texture_creator, program);
                                let TextureQuery { width: text_width, height: text_height, .. } = text_texture.query();
                                canvas.copy(&text_texture, None, Some(sdl2::rect::Rect::new(nav_width as i32, y, text_width, text_height))).unwrap();
                            }
                        },
                        Mode::Exit => (),
                    }
                    y += line_height as i32;
                }

                // Draw the waveform
                let x_scale = width as f32 / out.len() as f32;
                canvas.set_draw_color(Color::RGB(0, 255, 0));
                for (i, f) in out.iter().enumerate() {
                    let x = (i as f32 * x_scale) as i32;
                    let y = (f * (height as f32 / 2.4) + (height as f32 / 2.0)) as i32;
                    canvas.draw_point((x, y)).unwrap();
                }
                canvas.present();
            }
            Err(_) => {}
        }
        ::std::thread::sleep(Duration::new(0, 1_000_000_000u32 / 30));
    }
}

fn edit_mode_from_program(index: usize, program: &str) -> Mode {
    match parser::parse_program(program) {
        Ok(_) => Mode::Edit { index, errors: Vec::new() },
        Err(errors) => Mode::Edit { index, errors },
    }
}

fn process_event(event: Event, mode: Mode, programs: &mut Vec<String>, command_sender: &std::sync::mpsc::Sender<Command>) -> Mode {
    match event {
        Event::Quit { .. } => return Mode::Exit,
        Event::KeyDown { scancode, keymod, ..} => {
            match (mode, scancode) {
                // Exit on control-C
                (mode, Some(sdl2::keyboard::Scancode::C)) => {
                    if keymod.contains(sdl2::keyboard::Mod::LCTRLMOD)
                    || keymod.contains(sdl2::keyboard::Mod::RCTRLMOD) {
                        return Mode::Exit;
                    } else {
                        return mode;
                    }
                },
                (Mode::Select { index }, Some(sdl2::keyboard::Scancode::Return)) => {
                    return edit_mode_from_program(index, &programs[index]);
                },
                (Mode::Select { index }, Some(sdl2::keyboard::Scancode::Up)) => {
                    return Mode::Select { index: (index + programs.len() - 1) % programs.len() };
                },
                (Mode::Select { index }, Some(sdl2::keyboard::Scancode::Down)) => {
                    return  Mode::Select { index: (index + 1) % programs.len() };
                },
                (Mode::Edit { index, .. }, Some(sdl2::keyboard::Scancode::Return)) => {
                    let program = &programs[index];
                    match parser::parse_program(program) {
                        Ok(node) => {
                            command_sender.send(Command::PlayOnce{node, beat: 0}).unwrap();
                            return Mode::Select { index };
                        },
                        Err(errors) => {
                            // If there are errors, we stay in edit mode
                            println!("Errors while parsing input: {:?}", errors);
                            return Mode::Edit { index, errors };
                        }
                    }
                },
                (Mode::Edit { index, .. }, Some(sdl2::keyboard::Scancode::Backspace)) => {
                    // If the option key is down, clear the last word
                    let mut program = programs[index].clone();
                    if keymod.contains(sdl2::keyboard::Mod::LALTMOD) {
                        if let Some(char_index) = program.rfind(|e| !char::is_whitespace(e)) {
                            if let Some(space_index) = program[..char_index].rfind(char::is_whitespace) {
                                // Remove everything after that whitespace
                                program.truncate(space_index);
                            } else {
                                program.clear();
                            }
                        } else {
                            // No non-whitespace characters, so clear the whole string
                            program.clear();
                        }
                    } else {
                        program.pop();
                    }
                    programs[index] = program;
                    return edit_mode_from_program(index, &programs[index]);
                },
                (Mode::Edit { index, .. }, Some(sdl2::keyboard::Scancode::Escape)) => {
                    return Mode::Select { index: index };
                },
                (mode, _) => return mode,
            }

        },
        Event::TextInput { text, ..} => {
            match mode {
                Mode::Select { .. } => {
                    // If the text is a number less than NUM_PROGRAMS, update the index
                    if let Ok(index) = text.parse::<usize>() {
                        if index <= NUM_PROGRAMS {
                            return Mode::Select { index: index - 1};
                        } else {
                            println!("Invalid program index: {}", index);
                        }
                    } else {
                        println!("Invalid input for program selection: {}", text);
                    }
                    // TODO change mode in some cases
                    return mode;
                }
                Mode::Edit { index, .. } => {
                    programs[index].push_str(&text);
                    return edit_mode_from_program(index, &programs[index]);
                },
                _ => {
                    println!("Unexpected text input in mode: {:?}", mode);
                    return mode;
                }
            }
        },
        _ => { return mode; }
    }
}
