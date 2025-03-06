mod sequence;

use std::time::Duration;
use std::io::{self, Write};

extern crate sdl2;
use sdl2::event::Event;
use sdl2::audio::AudioSpecDesired;
use sdl2::render::{TextureCreator, TextureQuery};

use nom::{
    IResult,
    Parser,
    character::complete::{char, multispace1},
    number::complete::float,
    sequence::delimited,
    multi::separated_list0,
};

fn parse_sequence(input: &str) -> IResult<&str, Vec<f32>> {
    let mut parser = delimited(
        char('['),
        separated_list0(
            multispace1,
            float
        ),
        char(']')
    );
    return parser.parse(input);
}

//use std::time::Duration;
//use sequence::{new_sequence, wave_from_frequency};

use sdl2::pixels::Color;
//use sdl2::sys::Font;
use sdl2::ttf::Font;
use sdl2::video::WindowContext;

// 440^1 494^1 

fn make_texture<'a>(font: &Font<'a, 'static>, texture_creator: &'a TextureCreator<WindowContext>, s: &str) -> sdl2::render::Texture<'a> {
    let surface = font
        .render(s)
        .blended(Color::RGBA(0, 255, 0, 255))
        .map_err(|e| e.to_string()).unwrap();
    let texture = texture_creator
        .create_texture_from_surface(&surface)
        .map_err(|e| e.to_string()).unwrap();
    return texture;
}

pub fn main() {
    let sample_frequency = 44100;

    let sdl_context = sdl2::init().unwrap();
    let audio_subsystem = sdl_context.audio().unwrap();
    let desired_spec = AudioSpecDesired {
        freq: Some(sample_frequency),
        channels: Some(1),  // mono
        samples: None       // default sample size
    };

    let mut input = String::new();
    print!("> ");
    io::stdout().flush().unwrap();
    io::stdin().read_line(&mut input).unwrap();

    let mut freqs = Vec::new();
    match parse_sequence(&input.trim()) {
        Ok((_, fs)) => {
            println!("Parsed freqs: {:?}", fs);
            freqs.extend(fs);
        },
        Err(e) => println!("Failed to parse input: {:?}", e),
    }

    let mut generators = Vec::new();
    // Chord
    let mut chord_components = Vec::new();
            for freq in freqs {
                chord_components.push(sequence::wave_from_frequency(
                    sample_frequency, freq as f32));
            }
            generators.push(sequence::truncate(sample_frequency,
                Duration::from_secs(2),
                sequence::chord(chord_components)));
    /*
    // Sequence
    for freq in freqs {
        generators.push(sequence::truncate(sample_frequency,
            Duration::from_secs(2),
            sequence::wave_from_frequency(
                sample_frequency, freq)));
    }
    */

    let (sender, receiver) = std::sync::mpsc::channel();

    let device = 
        audio_subsystem.open_playback(None, &desired_spec, 
            |spec| {
                //            offsets: [Duration::from_millis(0), Duration::from_millis(1000)],
                //            current_offset: Duration::from_millis(0)
                println!("Spec: {:?}", spec);
                sequence::new_sequence(generators, sender)
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

    let prompt_texture = make_texture(&font, &texture_creator, "> ");
    let TextureQuery { width: prompt_width, height: prompt_height, .. } = prompt_texture.query();

    let mut next_program = String::new();

    video_subsystem.text_input().start();
    let mut event_pump = sdl_context.event_pump().unwrap();
    'running: loop {
        for event in event_pump.poll_iter() {
            println!("Event: {:?}", event);
            match event {
                Event::Quit { .. } => break 'running,
                Event::KeyDown { scancode, keymod, ..} => {
                    match scancode {
                        Some(sdl2::keyboard::Scancode::Return) => {
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
                                    // No non-whitespace
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

        match receiver.recv_timeout(Duration::new(0, 1_000_000)) {
            Ok(out) => {
                canvas.set_draw_color(Color::RGB(0, 0, 0));
                canvas.clear();
                canvas.copy(&prompt_texture, None, Some(sdl2::rect::Rect::new(10, 10, prompt_width, prompt_height))).unwrap();
                if next_program.len() > 0 {
                    let text_texture = make_texture(&font, &texture_creator, &next_program);
                    let TextureQuery { width: text_width, height: text_height, .. } = text_texture.query();
                    canvas.copy(&text_texture, None, Some(sdl2::rect::Rect::new(prompt_width as i32, 10, text_width, text_height))).unwrap();
                }

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
