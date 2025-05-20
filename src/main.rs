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

use nom::{
    IResult,
    Parser,
    branch::alt,
    combinator::eof,
    character::complete::{char, multispace0, multispace1},
    number::complete::float,
    sequence::{delimited, preceded, terminated},
    multi::separated_list0,
};

// TODO make this a
const sample_frequency : i32 = 44100;

enum Node {
    SineWave { frequency: f32 },
    Truncated { duration: Duration, node: Box<Node> },
    Chord(Vec<Node>),
    Sequence(Vec<Node>),
}

fn parse_node(input: &str) -> IResult<&str, Node> {
    let (i, node) = alt((
        parse_chord,
        parse_sequence,
        parse_tone,
    )).parse(input)?;
    return Ok((i, node));
}

fn parse_tone(input: &str) -> IResult<&str, Node> {
    let (rest, freq) = float.parse(input)?;
    println!("Parsed freq: {:?}", freq);
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

pub fn main() {
    let sdl_context = sdl2::init().unwrap();
    let audio_subsystem = sdl_context.audio().unwrap();
    let desired_spec = AudioSpecDesired {
        freq: Some(sample_frequency),
        channels: Some(1),  // mono
        samples: None       // default sample size
    };


    let (sample_sender, sample_receiver) = std::sync::mpsc::channel();
    let (node_sender, node_receiver) = std::sync::mpsc::channel();

    let device = 
        audio_subsystem.open_playback(None, &desired_spec, 
            |spec| {
                println!("Spec: {:?}", spec);
                sequence::new_sequencer(sample_frequency, node_receiver, sample_sender)
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

    let prompt_texture = make_texture(&font, Color::RGBA(0, 255, 255, 255), &texture_creator, "... ");
    let TextureQuery { width: prompt_width, height: prompt_height, .. } = prompt_texture.query();

    let mut should_exit = false;
    let mut current_program = String::new();
    let mut next_program = String::new();

    video_subsystem.text_input().start();
    let mut event_pump = sdl_context.event_pump().unwrap();
    'running: while !should_exit {
        for event in event_pump.poll_iter() {
            println!("Event: {:?} with next_program = {}", event, next_program);
            match event {
                Event::Quit { .. } => break 'running,
                Event::KeyDown { scancode, keymod, ..} => {
                    match scancode {
                        // Exit on crtl-C
                        Some(sdl2::keyboard::Scancode::C) => {
                            if keymod.contains(sdl2::keyboard::Mod::LCTRLMOD)
                                || keymod.contains(sdl2::keyboard::Mod::RCTRLMOD) {
                                should_exit = true;
                            }
                        },
                        Some(sdl2::keyboard::Scancode::Return) => {
                            // Parse the next program, create a node and send it offset
                            // to the audio device 
                            if let Ok(node) = parse_program(&next_program) {
                                node_sender.send(node).unwrap();
                                current_program = next_program.clone();
                                next_program.clear();
                            } else {
                                println!("Failed to parse input: {:?}", next_program);
                            }
                        },
                        Some(sdl2::keyboard::Scancode::Up) => {
                            next_program = current_program.clone();
                        },
                        Some(sdl2::keyboard::Scancode::Backspace) => {
                            // If the option key is down, clear the last word
                            if keymod.contains(sdl2::keyboard::Mod::LALTMOD) {
                                if let Some(char_index) = next_program.rfind(|e| !char::is_whitespace(e)) {
                                    if let Some(space_index) = next_program[..char_index].rfind(char::is_whitespace) {
                                        // Remove everything after that whitespace
                                        next_program.truncate(space_index);
                                    } else {
                                        next_program.clear();
                                    }
                                } else {
                                    // No non-whitespace characters, so clear the whole string
                                    next_program.clear();
                                }
                            } else {
                                next_program.pop();
                            }
                        },
                        _ => {}
                    }

                },
                Event::TextInput { text, ..} => {
                    next_program.push_str(&text);
                },
                _ => {}
            }
        }

        match sample_receiver.recv_timeout(Duration::new(0, 1_000_000)) {
            Ok(out) => {
                canvas.set_draw_color(Color::RGB(0, 0, 0));
                canvas.clear();

                let mut y = 10;
                if current_program.len() > 0 {
                    let text_texture = make_texture(&font, Color::RGBA(0, 255, 0, 255), &texture_creator, &current_program);
                    let TextureQuery { width: text_width, height: text_height, .. } = text_texture.query();
                    canvas.copy(&text_texture, None, Some(sdl2::rect::Rect::new(prompt_width as i32, y, text_width, text_height))).unwrap();
                    y += text_height as i32;
                }

                canvas.copy(&prompt_texture, None, Some(sdl2::rect::Rect::new(10, y, prompt_width, prompt_height))).unwrap();

                if next_program.len() > 0 {
                    let text_texture = make_texture(&font, Color::RGBA(0, 255, 255, 255), &texture_creator, &next_program);
                    let TextureQuery { width: text_width, height: text_height, .. } = text_texture.query();
                    canvas.copy(&text_texture, None, Some(sdl2::rect::Rect::new(prompt_width as i32, y, text_width, text_height))).unwrap();
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
