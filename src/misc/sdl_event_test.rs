//! Interactive SDL event logger for checking how quitting arrives. Opens a
//! window and prints each keyboard, text, window-close and quit event with the
//! milliseconds since the previous one, so you can see whether Cmd-Q, the app
//! menu's Quit item, the window's close button and Ctrl-C in the terminal each
//! send a key press, a window close, a quit event, or some combination.
//!
//! Quit events don't exit. Press Escape in the window to exit.

use std::time::Instant;

use sdl2::event::{Event, WindowEvent};
use sdl2::keyboard::Keycode;

fn main() -> Result<(), String> {
    let sdl = sdl2::init()?;
    let video = sdl.video()?;
    let _window = video
        .window("SDL event test (Escape exits)", 480, 240)
        .position_centered()
        .build()
        .map_err(|e| e.to_string())?;
    let mut event_pump = sdl.event_pump()?;
    println!("Try Cmd-Q, the app menu's Quit, the close button, and Ctrl-C in this terminal.");
    println!("Press Escape in the window to exit.\n");

    let mut last = Instant::now();
    loop {
        // One batch per pump, so events that arrive together print together.
        let event = event_pump.wait_event();
        let batch: Vec<Event> = std::iter::once(event)
            .chain(event_pump.poll_iter())
            .collect();
        println!("--- batch ---");
        for event in batch {
            let line = match &event {
                Event::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                } => return Ok(()),
                Event::KeyDown {
                    scancode,
                    keymod,
                    repeat,
                    ..
                } => format!("KeyDown {:?} mod={:?} repeat={}", scancode, keymod, repeat),
                Event::KeyUp {
                    scancode, keymod, ..
                } => format!("KeyUp {:?} mod={:?}", scancode, keymod),
                Event::TextInput { text, .. } => format!("TextInput {:?}", text),
                Event::Window {
                    win_event: WindowEvent::Close,
                    ..
                } => "Window Close".to_string(),
                Event::Quit { .. } => "Quit".to_string(),
                // Mouse motion, focus and the like aren't of interest.
                _ => continue,
            };
            let now = Instant::now();
            println!("{:>6} ms  {}", now.duration_since(last).as_millis(), line);
            last = now;
        }
    }
}
