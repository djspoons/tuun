//! Interactive Launchkey MK4 protocol scratchpad. Pick MIDI ports, toggle any
//! combination of Launchkey "feature controls" (14-bit analogue output,
//! relative encoder output, touch events), then watch the parsed event stream
//! until you press Enter. Only the features you enable here are reverted on
//! exit, so each can be exercised in isolation.

use std::collections::HashMap;
use std::io::{self, Write, stdin, stdout};
use std::sync::OnceLock;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use midir::{Ignore, MidiInput, MidiOutput, MidiOutputConnection};
use midly::MidiMessage;
use midly::live::LiveEvent;

/// A feature-control toggle (page 22 of the MK4 programmer's reference).
/// All listed CCs are sent on channel 7 (status byte 0xB6).
#[derive(Clone, Copy)]
struct Feature {
    /// CC number for the toggle.
    cc: u8,
    /// Used both for the y/N prompt and the send-log labels.
    name: &'static str,
    /// Whether y is the default when the user just hits Enter.
    default_on: bool,
}

const F_14BIT: Feature = Feature {
    cc: 0x44,
    name: "14-bit analogue encoder output",
    default_on: false,
};
const F_RELATIVE: Feature = Feature {
    cc: 0x45,
    name: "relative encoder output",
    default_on: true,
};
const F_TOUCH: Feature = Feature {
    cc: 0x47,
    name: "touch events",
    default_on: false,
};
const ALL_FEATURES: &[Feature] = &[F_14BIT, F_RELATIVE, F_TOUCH];

fn feature_control_msg(cc: u8, on: bool) -> [u8; 3] {
    [0xB6, cc, if on { 0x7F } else { 0x00 }]
}

/// Tracks the last MSB seen for each (channel, CC) pair for 14-bit pairing
/// detection. Standard MIDI 14-bit: MSB on CC n (0-31), LSB on CC n+32, sent
/// in close succession.
struct CCTracker {
    last_msb: HashMap<(u8, u8), (u8, Instant)>,
}

fn cc_tracker() -> &'static Mutex<CCTracker> {
    static TRACKER: OnceLock<Mutex<CCTracker>> = OnceLock::new();
    TRACKER.get_or_init(|| {
        Mutex::new(CCTracker {
            last_msb: HashMap::new(),
        })
    })
}

fn send_and_log(conn: &mut MidiOutputConnection, label: &str, data: &[u8]) {
    println!("  Sending [{}]: {:02X?}", label, data);
    if let Err(e) = conn.send(data) {
        println!("  -> ERROR: {}", e);
    }
}

fn prompt_yn(question: &str, default_on: bool) -> io::Result<bool> {
    let hint = if default_on { "[Y/n]" } else { "[y/N]" };
    print!("{} {}: ", question, hint);
    stdout().flush()?;
    let mut input = String::new();
    stdin().read_line(&mut input)?;
    Ok(match input.trim().to_ascii_lowercase().as_str() {
        "y" | "yes" => true,
        "n" | "no" => false,
        _ => default_on,
    })
}

fn prompt_index(prompt: &str, n: usize) -> io::Result<usize> {
    print!("Select {} [0-{}]: ", prompt, n.saturating_sub(1));
    stdout().flush()?;
    let mut input = String::new();
    stdin().read_line(&mut input)?;
    input
        .trim()
        .parse::<usize>()
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --- output port ---
    let midi_out = MidiOutput::new("midir test output")?;
    let out_ports = midi_out.ports();
    println!("Available output ports:");
    for (i, p) in out_ports.iter().enumerate() {
        println!("  {}: {}", i, midi_out.port_name(p).unwrap());
    }
    let out_port = out_ports
        .get(prompt_index("output port", out_ports.len())?)
        .ok_or("invalid output port selected")?;
    let out_port_name = midi_out.port_name(out_port)?;
    // Shared so the input listener can re-assert features in response to
    // device events (encoder-mode change in particular — see below).
    let out_conn = Arc::new(Mutex::new(midi_out.connect(out_port, "midir-output-port")?));
    println!("Using output port: {}", out_port_name);

    // --- enter DAW mode + enable feature-control CC routing ---
    {
        let mut conn = out_conn.lock().unwrap();
        println!("\nEntering DAW mode...");
        send_and_log(&mut conn, "DAW mode on", &[0x9F, 0x0C, 0x7F]);
        thread::sleep(Duration::from_millis(500));
        println!("Enabling feature controls...");
        send_and_log(&mut conn, "feature controls on", &[0x9F, 0x0B, 0x7F]);
        thread::sleep(Duration::from_millis(200));
    }

    // --- per-feature opt-in ---
    let mut enabled: Vec<Feature> = Vec::new();
    for feature in ALL_FEATURES {
        if prompt_yn(&format!("Enable {}", feature.name), feature.default_on)? {
            enabled.push(*feature);
        }
    }
    {
        let mut conn = out_conn.lock().unwrap();
        for feature in &enabled {
            println!("\nEnabling {}...", feature.name);
            send_and_log(
                &mut conn,
                feature.name,
                &feature_control_msg(feature.cc, true),
            );
            thread::sleep(Duration::from_millis(200));
        }
    }

    // Pair MSB/LSB CCs only when the user actually turned on 14-bit
    // analogue output — otherwise everything is plain 7-bit and high
    // CCs (>= 32) carry independent values.
    let bit14 = enabled.iter().any(|f| f.cc == F_14BIT.cc);
    let relative = enabled.iter().any(|f| f.cc == F_RELATIVE.cc);

    // --- input port + listener ---
    let mut midi_in = MidiInput::new("midir test input")?;
    midi_in.ignore(Ignore::None);
    let in_ports = midi_in.ports();
    println!("\nAvailable input ports:");
    for (i, p) in in_ports.iter().enumerate() {
        println!("  {}: {}", i, midi_in.port_name(p).unwrap());
    }
    let in_port = in_ports
        .get(prompt_index("input port", in_ports.len())?)
        .ok_or("invalid input port selected")?;
    let in_port_name = midi_in.port_name(in_port)?;

    let listener_conn = Arc::clone(&out_conn);
    let _conn_in = midi_in.connect(
        in_port,
        "midir-read-input",
        move |_stamp, message, _| handle_incoming(message, bit14, relative, &listener_conn),
        (),
    )?;

    println!(
        "\nListening on '{}', sending on '{}'. Press Enter to exit.",
        in_port_name, out_port_name
    );
    let mut input = String::new();
    stdin().read_line(&mut input)?;

    // --- cleanup: revert just the features we toggled on, then exit DAW ---
    {
        let mut conn = out_conn.lock().unwrap();
        for feature in enabled.iter().rev() {
            println!("\nDisabling {}...", feature.name);
            send_and_log(
                &mut conn,
                feature.name,
                &feature_control_msg(feature.cc, false),
            );
            thread::sleep(Duration::from_millis(100));
        }
        println!("Exiting DAW mode...");
        send_and_log(&mut conn, "DAW mode off", &[0x9F, 0x0C, 0x00]);
    }

    Ok(())
}

fn handle_incoming(
    message: &[u8],
    bit14: bool,
    relative: bool,
    out_conn: &Arc<Mutex<MidiOutputConnection>>,
) {
    if message == [0xF8] {
        return; // skip the MIDI clock tick
    }
    let event = match LiveEvent::parse(message) {
        Ok(e) => e,
        Err(_) => {
            let hex: Vec<String> = message.iter().map(|b| format!("{:02X}", b)).collect();
            println!("(parse failed) [{}]", hex.join(" "));
            return;
        }
    };
    let LiveEvent::Midi { channel, message } = event else {
        println!("{:?}", event);
        return;
    };
    let ch = u8::from(channel) + 1;
    match message {
        MidiMessage::NoteOn { key, vel } => {
            println!("NoteOn ch{} key={} vel={}", ch, key, vel);
        }
        MidiMessage::NoteOff { key, vel } => {
            println!("NoteOff ch{} key={} vel={}", ch, key, vel);
        }
        MidiMessage::Controller { controller, value } => {
            handle_cc(
                ch,
                u8::from(controller),
                u8::from(value),
                bit14,
                relative,
                out_conn,
            );
        }
        MidiMessage::Aftertouch { key, vel } => {
            println!("Aftertouch ch{} key={} vel={}", ch, key, vel);
        }
        other => {
            println!("ch{}: {:?}", ch, other);
        }
    }
}

/// Encoder-mode change report (page 10 of the MK4 reference): channel 7,
/// CC 1Eh. Switching encoder modes silently resets the "DAW Encoder Relative
/// output" feature on the device, so we re-assert it from here whenever the
/// session was started with relative output on.
const ENCODER_MODE_CHANGED_CHANNEL: u8 = 7;
const ENCODER_MODE_CHANGED_CC: u8 = 0x1E;

fn handle_cc(
    ch: u8,
    cc: u8,
    v: u8,
    bit14: bool,
    relative: bool,
    out_conn: &Arc<Mutex<MidiOutputConnection>>,
) {
    if ch == ENCODER_MODE_CHANGED_CHANNEL && cc == ENCODER_MODE_CHANGED_CC {
        println!("Encoder mode changed (val={})", v);
        if relative {
            let mut conn = out_conn.lock().unwrap();
            send_and_log(
                &mut conn,
                "relative encoder output (re-assert)",
                &feature_control_msg(F_RELATIVE.cc, true),
            );
        }
        return;
    }
    if !bit14 {
        println!("CC ch{} cc={} (0x{:02X}) val={}", ch, cc, cc, v);
        return;
    }
    let now = Instant::now();
    let mut tracker = cc_tracker().lock().unwrap();
    if cc < 32 {
        // MSB — stash and wait for the matching LSB
        tracker.last_msb.insert((ch, cc), (v, now));
    } else if cc < 64 {
        let msb_cc = cc - 32;
        match tracker.last_msb.remove(&(ch, msb_cc)) {
            Some((msb, when)) if now.duration_since(when) < Duration::from_millis(50) => {
                let combined = ((msb as u16) << 7) | (v as u16);
                println!(
                    "CC14 ch{} cc={} (0x{:02X}) val={}",
                    ch, msb_cc, msb_cc, combined
                );
            }
            Some(_) => {
                println!(
                    "CC ch{} cc={} (0x{:02X}) val={} (orphan LSB: paired MSB too old)",
                    ch, cc, cc, v
                );
            }
            None => {
                println!(
                    "CC ch{} cc={} (0x{:02X}) val={} (orphan LSB: no MSB)",
                    ch, cc, cc, v
                );
            }
        }
    } else {
        // High CC range — outside the 14-bit pairing window.
        println!("CC ch{} cc={} (0x{:02X}) val={}", ch, cc, cc, v);
    }
}
