//! Checks the two hardware facts the per-program Loop spec leaves unverified
//! (see `agents/new-loop-after/spec.md`): whether the renderer's font has the
//! gutter glyphs, and how the Launchkey's Loop button and its LED behave.
//!
//! Run with `cargo run --bin loop_hw_check`, optionally passing a font path.
//! Each LED step asks what you saw; the answers are printed together at the
//! end so they can be pasted back into the spec.

use std::env;
use std::io::{self, Write, stdin, stdout};
use std::sync::{Arc, Mutex};

use midir::{Ignore, MidiInput, MidiOutput, MidiOutputConnection};

/// The renderer's font (`FONT_PATH` in `renderer.rs`).
const DEFAULT_FONT_PATH: &str = "/Library/Fonts/Arial Unicode.ttf";

/// The Launchkey's Loop button, on the DAW port.
const LOOP_CC: u8 = 118;

fn main() {
    let font_path = env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_FONT_PATH.to_string());
    let mut report = vec![format!("Font: {}", font_path)];
    report.extend(check_font(&font_path));
    match check_launchkey() {
        Ok(lines) => report.extend(lines),
        Err(e) => report.push(format!("Launchkey: not checked ({})", e)),
    }
    println!("\n===== Report (paste this back) =====");
    for line in &report {
        println!("{}", line);
    }
}

/// Returns one line per glyph saying whether the font at `path` provides it.
fn check_font(path: &str) -> Vec<String> {
    let ttf = match sdl2::ttf::init() {
        Ok(ttf) => ttf,
        Err(e) => return vec![format!("  could not start SDL_ttf: {}", e)],
    };
    let font = match ttf.load_font(path, 24) {
        Ok(font) => font,
        Err(e) => return vec![format!("  could not load font: {}", e)],
    };
    // `▸` is already drawn by the renderer, so it is the control.
    [('▸', "control"), ('↻', "Loop"), ('⌨', "keys")]
        .iter()
        .map(|&(ch, role)| {
            let found = font.find_glyph(ch).is_some();
            format!(
                "  {} U+{:04X} ({}): {}",
                ch,
                ch as u32,
                role,
                if found { "present" } else { "MISSING" }
            )
        })
        .collect()
}

/// Walks through the Loop button checks and returns one line per answer.
fn check_launchkey() -> Result<Vec<String>, Box<dyn std::error::Error>> {
    println!("\n== Launchkey Loop button ==");
    let output = MidiOutput::new("loop_hw_check output")?;
    let out_port = output
        .ports()
        .into_iter()
        .find(|p| {
            output
                .port_name(p)
                .is_ok_and(|n| n == "Launchkey MK4 37 DAW In")
        })
        .ok_or("couldn't find Launchkey DAW In port")?;
    let mut out = output.connect(&out_port, "loop_hw_check-out")?;

    // Log every CC 118 message so press and release values can be read off.
    let seen = Arc::new(Mutex::new(Vec::<String>::new()));
    let mut input = MidiInput::new("loop_hw_check input")?;
    input.ignore(Ignore::Time);
    let in_port = input
        .ports()
        .into_iter()
        .find(|p| {
            input
                .port_name(p)
                .is_ok_and(|n| n == "Launchkey MK4 37 DAW Out")
        })
        .ok_or("couldn't find Launchkey DAW Out port")?;
    let log = Arc::clone(&seen);
    let _in_conn = input.connect(
        &in_port,
        "loop_hw_check-in",
        move |_, data, _| {
            if data.len() == 3 && data[0] & 0xF0 == 0xB0 && data[1] == LOOP_CC {
                let line = format!("channel {} value {}", (data[0] & 0x0F) + 1, data[2]);
                println!("  [Loop button] {}", line);
                log.lock().unwrap().push(line);
            }
        },
        (),
    )?;

    // Enter device DAW mode.
    send(&mut out, &[0x9F, 0x0C, 0x7F]);

    let mut report = Vec::new();
    ask("Press and release the Loop button a couple of times, then press Enter.")?;
    let messages = seen.lock().unwrap().clone();
    report.push(format!(
        "Launchkey: Loop button sent: {}",
        if messages.is_empty() {
            "nothing".to_string()
        } else {
            messages.join("; ")
        }
    ));

    // Brightness, the way tuun drives Record's LED (CC on channel 4).
    for value in [127, 64, 16, 0] {
        send(&mut out, &[0xB3, LOOP_CC, value]);
        let answer = ask(&format!(
            "Sent channel 4 CC {} = {}. Describe the Loop LED (off / dim / bright / colour):",
            LOOP_CC, value
        ))?;
        report.push(format!("  ch4 value {:>3}: {}", value, answer));
    }

    // Palette colours on channel 1, as tuun drives the RGB Function and `>`
    // LEDs. A coloured light here would mean the LED is not single-colour.
    for (value, name) in [(5, "red"), (21, "green"), (45, "blue"), (0, "off")] {
        send(&mut out, &[0xB0, LOOP_CC, value]);
        let answer = ask(&format!(
            "Sent channel 1 CC {} = {} (palette {}). Describe the Loop LED:",
            LOOP_CC, value, name
        ))?;
        report.push(format!("  ch1 value {:>3} ({}): {}", value, name, answer));
    }

    // Leave the LED off and the device out of DAW mode.
    send(&mut out, &[0xB3, LOOP_CC, 0]);
    send(&mut out, &[0x9F, 0x0C, 0x00]);
    Ok(report)
}

fn send(conn: &mut MidiOutputConnection, data: &[u8]) {
    println!("  sending {:02X?}", data);
    if let Err(e) = conn.send(data) {
        println!("  -> error: {}", e);
    }
}

/// Prints `question` and returns the trimmed line typed in reply.
fn ask(question: &str) -> io::Result<String> {
    print!("{} ", question);
    stdout().flush()?;
    let mut answer = String::new();
    stdin().read_line(&mut answer)?;
    Ok(answer.trim().to_string())
}
