mod sequence;

use core::panic;
use std::time::Duration;

extern crate sdl2;
use sdl2::event::Event;
use sdl2::audio::AudioSpecDesired;
use sdl2::render::{TextureCreator, TextureQuery};
use sdl2::pixels::Color;
use sdl2::ttf::Font;
use sdl2::video::WindowContext;

use clap::Parser as ClapParser;

use nom::{
    IResult,
    Parser,
    branch::alt,
    combinator::{eof, map},
    character::complete::{char, multispace0, multispace1},
    number::complete::float,
    sequence::{delimited, preceded, terminated},
    multi::{many0, separated_list0},
};

enum FloatExpr {
    Value(f32),
    Multiply(Box<FloatExpr>, Box<FloatExpr>),
    Divide(Box<FloatExpr>, Box<FloatExpr>),
}
enum Node {
    SineWave { frequency: FloatExpr},
    Truncated { duration: Duration, node: Box<Node> },
    Chord(Vec<Node>),
    Sequence(Vec<Node>),
}

fn parse_node(input: &str) -> IResult<&str, Node> {
    let (rest, node) = alt((
        parse_chord,
        parse_sequence,
        parse_tone,
    )).parse(input)?;
    return Ok((rest, node));
}

fn parse_float_literal(input: &str) -> IResult<&str, FloatExpr> {
    let (rest, value) = float.parse(input)?;
    println!("Parsed float literal: {} with rest {}", value, rest);
    return Ok((rest, FloatExpr::Value(value)));
}

fn parse_float_term(input: &str) -> IResult<&str, FloatExpr> {
    let (rest, value) =
        map((parse_float_factor,
            many0(
                (delimited(multispace0,alt((char('*'), char('/'))), multispace0),
                parse_float_factor),
            )), |(factor, op_factors)| {
            let mut result = factor;
            for (op, factor) in op_factors {
                result = match op {
                    '*' => FloatExpr::Multiply(Box::new(result), Box::new(factor)),
                    '/' => FloatExpr::Divide(Box::new(result), Box::new(factor)),
                    _ => panic!("Unexpected operator: {}", op),
                };
            }
            return result;
        }).parse(input)?;
    return Ok((rest, value));
}

fn parse_float_factor(input: &str) -> IResult<&str, FloatExpr> {
    let (rest, value) = alt((
        parse_float_literal,
        delimited(
            (char('('), multispace0),
            parse_float_term,
            (multispace0, char(')')),
        ),
    )).parse(input)?;
    return Ok((rest, value));
}

fn parse_tone(input: &str) -> IResult<&str, Node> {
    let (rest, freq) = preceded(
        char('$'),
        parse_float_factor
    ).parse(input)?;
    return Ok((rest, Node::Truncated{duration: Duration::from_secs(2), 
        node: Box::new(Node::SineWave { frequency: freq })}));
}

fn parse_chord(input: &str) -> IResult<&str, Node> {
    let (rest, nodes) = delimited(
        char('<'),
        separated_list0(
            multispace1,
            parse_node,
        ),
        char('>'),
    ).parse(input)?;
    return Ok((rest, Node::Chord(nodes)));
}

fn parse_sequence(input: &str) -> IResult<&str, Node> {
    let (rest, nodes) = delimited(
        terminated(char('['), multispace0),
        separated_list0(
            multispace1,
            parse_node,
        ),
        preceded(multispace0, char(']')),
    ).parse(input)?;
    return Ok((rest, Node::Sequence(nodes)));
}

fn parse_program(input: &str) -> Result<Node, nom::error::Error<&str>> {
    match terminated(
        delimited(
            multispace0,
            parse_node,
            multispace0),
            eof,
    ).parse(input) {
        Ok((_, node)) => {
            return Ok(node);
        },
        Err(nom::Err::Error(e)) => {
            println!("Error on parsing input: {:?}", e);
            return Err(e);
        }
        Err(nom::Err::Incomplete(_)) => {
            panic!("Incomplete error on input");
        }
        Err(nom::Err::Failure(e)) => {
            println!("Failed to parse input: {:?}", e);
            return Err(e);
        }
    }
}

enum Command {
    PlayOnce {
        node: Node,
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
    #[arg(short, long, default_value = "")]
    program: String,
}

#[derive(Debug)]
enum Mode {
    Select { index: usize },
    Edit { index: usize },
    Exit,
}

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

    let prompt_texture = make_texture(&font, Color::RGBA(0, 255, 0, 255), &texture_creator, " → ");
    let TextureQuery { width: prompt_width, height: prompt_height, .. } = prompt_texture.query();
    let cursor_texture = make_texture(&font, Color::RGBA(0, 255, 255, 255), &texture_creator, "‸");
    let TextureQuery { width: cursor_width, height: cursor_height, .. } = cursor_texture.query();

    let mut programs = vec![String::new(); 5];
    programs[0] = args.program.clone();
    let mut mode = Mode::Edit { index: 0 };

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
                    let mut color = Color::RGBA(0, 255, 0, 255);
                    if let Mode::Edit { index } = mode {
                        if index == i {
                            color = Color::RGBA(0, 255, 255, 255);
                        }
                    }
                    canvas.set_draw_color(color);
                    if let Mode::Select { index } = mode {
                        if index == i {
                            canvas.copy(&prompt_texture, None, Some(sdl2::rect::Rect::new(0, y, prompt_width, prompt_height))).unwrap();
                        }
                    }
                    let mut cursor_x = prompt_width as i32;
                    if !program.is_empty() {
                        let text_texture = make_texture(&font, color, &texture_creator, program);
                        let TextureQuery { width: text_width, height: text_height, .. } = text_texture.query();
                        canvas.copy(&text_texture, None, Some(sdl2::rect::Rect::new(prompt_width as i32, y, text_width, text_height))).unwrap();
                        cursor_x += text_width as i32;
                    }
                    if let Mode::Edit { index } = mode {
                        if index == i {
                            canvas.copy(&cursor_texture, None, Some(sdl2::rect::Rect::new(cursor_x, y, cursor_width, cursor_height))).unwrap();
                        }
                    }
                    y += prompt_height as i32;
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
                    return Mode::Edit { index };
                },
                (Mode::Select { index }, Some(sdl2::keyboard::Scancode::Up)) => {
                    return Mode::Select { index: (index + programs.len() - 1) % programs.len() };
                },
                (Mode::Select { index }, Some(sdl2::keyboard::Scancode::Down)) => {
                    return  Mode::Select { index: (index + 1) % programs.len() };
                },
                (Mode::Edit { index }, Some(sdl2::keyboard::Scancode::Return)) => {
                    let program = &programs[index];
                    if let Ok(node) = parse_program(program) {
                        command_sender.send(Command::PlayOnce{node, beat: 0}).unwrap();
                        return Mode::Select { index };
                    } else {
                        println!("Failed to parse input: {:?}", program);
                        return Mode::Select { index };
                    }
                },
                (Mode::Edit { index }, Some(sdl2::keyboard::Scancode::Backspace)) => {
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
                    return Mode::Edit { index: index };
                },
                (Mode::Edit { index }, Some(sdl2::keyboard::Scancode::Escape)) => {
                    return Mode::Select { index: index };
                },
                (mode, _) => return mode,
            }

        },
        Event::TextInput { text, ..} => {
            match mode {
                Mode::Select { .. } => {
                    // TODO change mode in some cases
                    return mode;
                }
                Mode::Edit { index } => {
                    programs[index].push_str(&text);
                    return mode;
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
