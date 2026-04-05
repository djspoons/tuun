use std::io::{Write, stdin, stdout};

use midir::{Ignore, MidiInput, MidiOutput};
use midly::MidiMessage;
use midly::live::LiveEvent;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let midi_out = MidiOutput::new("midir reading output")?;
    //midi_out.ignore(Ignore::None);

    // Get an output port (read from console if multiple are available)
    let out_ports = midi_out.ports();
    let out_port = match out_ports.len() {
        0 => return Err("no output port found".into()),
        1 => {
            println!(
                "Choosing the only available output port: {}",
                midi_out.port_name(&out_ports[0]).unwrap()
            );
            &out_ports[0]
        }
        _ => {
            println!("\nAvailable output ports:");
            for (i, p) in out_ports.iter().enumerate() {
                println!("{}: {}", i, midi_out.port_name(p).unwrap());
            }
            print!("Please select output port: ");
            stdout().flush()?;
            let mut input = String::new();
            stdin().read_line(&mut input)?;
            out_ports
                .get(input.trim().parse::<usize>()?)
                .ok_or("invalid output port selected")?
        }
    };

    let mut out_conn = midi_out.connect(out_port, "midir-output-port")?;
    println!("Changing to DAW mode...");
    out_conn.send(&[0x9F, 0x0C, 0x7F])?;

    let mut midi_in = MidiInput::new("midir reading input")?;
    midi_in.ignore(Ignore::None);

    // Get an input port (read from console if multiple are available)
    let in_ports = midi_in.ports();
    let in_port = match in_ports.len() {
        0 => return Err("no input port found".into()),
        1 => {
            println!(
                "Choosing the only available input port: {}",
                midi_in.port_name(&in_ports[0]).unwrap()
            );
            &in_ports[0]
        }
        _ => {
            println!("\nAvailable input ports:");
            for (i, p) in in_ports.iter().enumerate() {
                println!("{}: {}", i, midi_in.port_name(p).unwrap());
            }
            print!("Please select input port: ");
            stdout().flush()?;
            let mut input = String::new();
            stdin().read_line(&mut input)?;
            in_ports
                .get(input.trim().parse::<usize>()?)
                .ok_or("invalid input port selected")?
        }
    };

    println!("\nOpening input connection");
    let in_port_name = midi_in.port_name(in_port)?;

    // _conn_in needs to be a named parameter, because it needs to be kept alive until the end of the scope
    let _conn_in = midi_in.connect(
        in_port,
        "midir-read-input",
        move |_stamp, message, _| {
            if message != &[248] {
                //println!("{}: {:?} (len = {})", stamp, message, message.len());
                let event = LiveEvent::parse(message).unwrap();
                match event {
                    LiveEvent::Midi { channel, message } => match message {
                        MidiMessage::NoteOn { key, vel } => {
                            println!(
                                "Note On: {} on channel {} with velocity {}",
                                key, channel, vel
                            );
                        }
                        message => {
                            println!("Other message (on channel {}): {:?}", channel, message);
                        }
                    },
                    _ => {}
                }
            }
        },
        (),
    )?;

    println!(
        "Connection open, reading input from '{}' (press enter to exit) ...",
        in_port_name
    );

    let mut input = String::new();
    input.clear();
    stdin().read_line(&mut input)?; // wait for next enter key press

    println!("Closing connection, returning to standard mode");
    out_conn.send(&[0x9F, 0x0C, 0x00])?;

    Ok(())
}
