mod sequence;

extern crate sdl2;


use sdl2::event::Event;
use std::time::Duration;

use sdl2::audio::AudioSpecDesired;

//use std::time::Duration;
//use sequence::{new_sequence, wave_from_frequency};

use sdl2::pixels::Color;

// 440^1 494^1 

pub fn main() {
    let mut freqs = Vec::new();
    let mode = std::env::args().nth(1).unwrap();
    for arg in std::env::args().skip(2) {
        freqs.push(arg.parse::<u32>().unwrap());
    }
    println!("Freqs: {:?}", freqs);

    let sample_frequency = 44100;

    let sdl_context = sdl2::init().unwrap();
    let audio_subsystem = sdl_context.audio().unwrap();
    let desired_spec = AudioSpecDesired {
        freq: Some(sample_frequency),
        channels: Some(1),  // mono
        samples: None       // default sample size
    };

    let mut generators = Vec::new();
    match mode.as_str() {
        "S" => {
            for freq in freqs {
                generators.push(sequence::truncate(sample_frequency,
                    Duration::from_secs(2),
                    sequence::wave_from_frequency(
                        sample_frequency, freq as f32)));
            }
        }
        "C" => {
            let mut chord_components = Vec::new();
            for freq in freqs {
                chord_components.push(sequence::wave_from_frequency(
                    sample_frequency, freq as f32));
            }
            generators.push(sequence::truncate(sample_frequency,
                Duration::from_secs(2),
                sequence::chord(chord_components)));
        }
        _ => {
            println!("Unknown mode: {}", mode);
        }
    }

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

    let width = 1200;
    let height = 600;
    let window = video_subsystem
        .window("tuunel waveform", width, height)
        .position_centered()
        .build()
        .map_err(|e| e.to_string()).unwrap();
    let mut canvas = window.into_canvas().build().map_err(
        |e| e.to_string()).unwrap();

    let mut event_pump = sdl_context.event_pump().unwrap();

    'running: loop {
        for event in event_pump.poll_iter() {
            println!("Event: {:?}", event);
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                        ..
                } => break 'running,
                _ => {}
            }
        }

        match receiver.recv_timeout(Duration::new(0, 1_000_000)) {
            Ok(out) => {
                canvas.set_draw_color(Color::RGB(0, 0, 0));
                canvas.clear();
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
