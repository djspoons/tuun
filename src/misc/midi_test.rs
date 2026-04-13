use std::io::{Write, stdin, stdout};
use std::thread;
use std::time::Duration;

use midir::{Ignore, MidiInput, MidiOutput};
use midly::MidiMessage;
use midly::live::LiveEvent;

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
            let hex: Vec<String> = message.iter().map(|b| format!("{:02X}", b)).collect();
            print!("IN: [{}]", hex.join(" "));
            if let Ok(event) = LiveEvent::parse(message) {
                match event {
                    LiveEvent::Midi { channel, message } => {
                        let ch = u8::from(channel) + 1;
                        match message {
                            MidiMessage::NoteOn { key, vel } => {
                                println!(" -> NoteOn ch{} key={} vel={}", ch, key, vel);
                            }
                            MidiMessage::NoteOff { key, vel } => {
                                println!(" -> NoteOff ch{} key={} vel={}", ch, key, vel);
                            }
                            MidiMessage::Controller { controller, value } => {
                                println!(" -> CC ch{} cc={} val={}", ch, controller, value);
                            }
                            MidiMessage::Aftertouch { key, vel } => {
                                println!(" -> Aftertouch ch{} key={} vel={}", ch, key, vel);
                            }
                            other => {
                                println!(" -> ch{}: {:?}", ch, other);
                            }
                        }
                    }
                    other => {
                        println!(" -> {:?}", other);
                    }
                }
            } else {
                println!(" -> (parse failed)");
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

    println!("Returning to standard mode...");
    send_and_log(&mut out_conn, "DAW mode off", &[0x9F, 0x0C, 0x00]);

    Ok(())
}
