use std::io::{Write, stdin, stdout};
use std::sync::Mutex;
use std::sync::OnceLock;
use std::thread;
use std::time::{Duration, Instant};

use midir::{Ignore, MidiInput, MidiOutput};
use midly::MidiMessage;
use midly::live::LiveEvent;

/// Tracks the last MSB seen for each (channel, CC) pair for 14-bit pairing detection.
/// Standard MIDI 14-bit: MSB on CC n (0-31), LSB on CC n+32, sent in close succession.
struct CCTracker {
    last_msb: std::collections::HashMap<(u8, u8), (u8, Instant)>,
}

fn cc_tracker() -> &'static Mutex<CCTracker> {
    static TRACKER: OnceLock<Mutex<CCTracker>> = OnceLock::new();
    TRACKER.get_or_init(|| {
        Mutex::new(CCTracker {
            last_msb: std::collections::HashMap::new(),
        })
    })
}

fn send_and_log(conn: &mut midir::MidiOutputConnection, label: &str, data: &[u8]) {
    println!("  Sending [{}]: {:02?}", label, data);
    match conn.send(data) {
        Ok(()) => println!("  -> OK"),
        Err(e) => println!("  -> ERROR: {}", e),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let midi_out = MidiOutput::new("midir test output")?;

    let out_ports = midi_out.ports();
    println!("Available output ports:");
    for (i, p) in out_ports.iter().enumerate() {
        let name = midi_out.port_name(p).unwrap();
        println!("  {}: {}", i, name);
    }
    print!("Select output port: ");
    stdout().flush()?;
    let mut input = String::new();
    stdin().read_line(&mut input)?;
    let out_port = out_ports
        .get(input.trim().parse::<usize>()?)
        .ok_or("invalid output port selected")?;
    let out_port_name = midi_out.port_name(out_port)?;
    println!("Using output port: {}", out_port_name);

    let mut out_conn = midi_out.connect(out_port, "midir-output-port")?;

    // Enter DAW mode
    println!("\nEntering DAW mode...");
    send_and_log(&mut out_conn, "DAW mode on", &[0x9F, 0x0C, 0x7F]);
    thread::sleep(Duration::from_millis(500));

    // Enable feature controls (required to set the 14-bit option below)
    println!("\nEnabling feature controls...");
    send_and_log(&mut out_conn, "feature controls on", &[0x9F, 0x0B, 0x7F]);
    thread::sleep(Duration::from_millis(200));

    print!("Enable 14-bit analogue output [Y/n]: ");
    stdout().flush()?;
    let mut input = String::new();
    stdin().read_line(&mut input)?;
    if input.is_empty() || input == "Y" || input == "y" {
        // Enable DAW 14-bit Analogue output (CC 0x44 on channel 7 = status B6h)
        println!("\nEnabling 14-bit analogue output...");
        send_and_log(&mut out_conn, "14-bit analogue on", &[0xB6, 0x44, 0x7F]);
        thread::sleep(Duration::from_millis(200));
    } else {
        println!("\nDisabling 14-bit analogue output...");
        send_and_log(&mut out_conn, "14-bit analogue off", &[0xB6, 0x44, 0x00]);
        thread::sleep(Duration::from_millis(200));
    }

    // Set up input listener
    let mut midi_in = MidiInput::new("midir test input")?;
    midi_in.ignore(Ignore::None);

    let in_ports = midi_in.ports();
    println!("\nAvailable input ports:");
    for (i, p) in in_ports.iter().enumerate() {
        let name = midi_in.port_name(p).unwrap();
        println!("  {}: {}", i, name);
    }
    print!("Select input port: ");
    stdout().flush()?;
    input.clear();
    stdin().read_line(&mut input)?;
    let in_port = in_ports
        .get(input.trim().parse::<usize>()?)
        .ok_or("invalid input port selected")?;
    let in_port_name = midi_in.port_name(in_port)?;

    let _conn_in = midi_in.connect(
        in_port,
        "midir-read-input",
        move |_stamp, message, _| {
            if message == &[248] {
                return; // Skip timing clock
            }
            if let Ok(event) = LiveEvent::parse(message) {
                match event {
                    LiveEvent::Midi { channel, message } => {
                        let ch = u8::from(channel) + 1;
                        match message {
                            MidiMessage::NoteOn { key, vel } => {
                                println!("NoteOn ch{} key={} vel={}", ch, key, vel);
                            }
                            MidiMessage::NoteOff { key, vel } => {
                                println!("NoteOff ch{} key={} vel={}", ch, key, vel);
                            }
                            MidiMessage::Controller { controller, value } => {
                                let cc = u8::from(controller);
                                let v = u8::from(value);
                                let mut tracker = cc_tracker().lock().unwrap();
                                let now = Instant::now();
                                if cc < 32 {
                                    // MSB — stash and wait for the matching LSB
                                    tracker.last_msb.insert((ch, cc), (v, now));
                                } else if cc >= 32 && cc < 64 {
                                    let msb_cc = cc - 32;
                                    if let Some((msb, when)) =
                                        tracker.last_msb.remove(&(ch, msb_cc))
                                    {
                                        if now.duration_since(when) < Duration::from_millis(50) {
                                            let combined = ((msb as u16) << 7) | (v as u16);
                                            println!(
                                                "CC14 ch{} cc={} (0x{:02X}) val={}",
                                                ch, msb_cc, msb_cc, combined
                                            );
                                        } else {
                                            println!(
                                                "CC ch{} cc={} (0x{:02X}) val={} (orphan LSB)",
                                                ch, cc, cc, v
                                            );
                                        }
                                    } else {
                                        println!(
                                            "CC ch{} cc={} (0x{:02X}) val={} (orphan LSB)",
                                            ch, cc, cc, v
                                        );
                                    }
                                } else {
                                    println!("CC ch{} cc={} (0x{:02X}) val={}", ch, cc, cc, v);
                                }
                            }
                            MidiMessage::Aftertouch { key, vel } => {
                                println!("Aftertouch ch{} key={} vel={}", ch, key, vel);
                            }
                            other => {
                                println!("ch{}: {:?}", ch, other);
                            }
                        }
                    }
                    other => {
                        println!("{:?}", other);
                    }
                }
            } else {
                let hex: Vec<String> = message.iter().map(|b| format!("{:02X}", b)).collect();
                println!("(parse failed) [{}]", hex.join(" "));
            }
        },
        (),
    )?;

    println!(
        "\nListening on '{}', sending on '{}'",
        in_port_name, out_port_name
    );

    println!("Listening for events. Press enter to exit...");
    input.clear();
    stdin().read_line(&mut input)?;

    println!("Disabling 14-bit analogue output...");
    send_and_log(&mut out_conn, "14-bit analogue off", &[0xB6, 0x44, 0x00]);
    thread::sleep(Duration::from_millis(100));

    println!("Returning to standard mode...");
    send_and_log(&mut out_conn, "DAW mode off", &[0x9F, 0x0C, 0x00]);

    Ok(())
}
