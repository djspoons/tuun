use midir::{Ignore, MidiInput, MidiInputConnection, MidiOutput, MidiOutputConnection};
//use midly::live::LiveEvent;
//use midly::MidiMessage;

use thiserror::Error;

pub struct Launchkey {
    daw_output_conn: MidiOutputConnection,
    daw_input_conn: MidiInputConnection<()>,
}

pub enum Event {
    NextTrack,
    PreviousTrack,
    MixerEncoderChange {
        parameter: u8, // 0-15 used
        change: f32,   // ?
    },
    PluginEncoderChange {
        parameter: u8, // 0-15 used
        change: f32,   // ?
    },
}

#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    InitError(#[from] midir::InitError),
    #[error(transparent)]
    OutputConnectError(#[from] midir::ConnectError<MidiOutput>),
    #[error(transparent)]
    InputConnectError(#[from] midir::ConnectError<MidiInput>),
    #[error(transparent)]
    SendError(#[from] midir::SendError),
    #[error("controller error: {0}")]
    Other(String),
}

impl Launchkey {
    pub fn new() -> Result<Self, Error> {
        // First, find the DAW input channel, connect it to our input, and put the controller into DAW mode
        let midi_output = MidiOutput::new("tuun sending output")?;
        let mut daw_output_conn;
        if let Some(daw_output_port) = midi_output.ports().iter().find(|p| {
            midi_output.port_name(p).unwrap_or("error".to_string())
                == "Launchkey MK4 37 DAW In".to_string()
        }) {
            daw_output_conn = midi_output.connect(daw_output_port, "tuun-daw-output-port")?;
            daw_output_conn.send(&[0x9F, 0x0C, 0x7F])?;
        } else {
            return Err(Error::Other(
                "couldn't find Launchkey DAW In port".to_string(),
            ));
        }

        // Second, set up the output channels for both "MIDI" and "DAW".
        let mut midi_input = MidiInput::new("tuun reading input")?;
        midi_input.ignore(Ignore::Time);
        let daw_input_conn;
        if let Some(daw_input_port) = midi_input.ports().iter().find(|p| {
            midi_input.port_name(p).unwrap_or("error".to_string())
                == "Launchkey MK4 37 DAW Out".to_string()
        }) {
            daw_input_conn = midi_input.connect(
                daw_input_port,
                "tuun-daw-input-port",
                |_stamp, _message, _| { /* XXX */ },
                (),
            )?;
        } else {
            return Err(Error::Other(
                "couldn't find Launchkey DAW Out port".to_string(),
            ));
        }
        // XXX set up MIDI port too

        Ok(Launchkey {
            daw_output_conn,
            daw_input_conn,
        })
    }
}

impl Drop for Launchkey {
    fn drop(&mut self) {
        match self.daw_output_conn.send(&[0x9F, 0x0C, 0x00]) {
            Err(e) => {
                println!("launchkey: failed to exit from DAW mode: {}", e);
            }
            _ => (),
        }
    }
}

impl Launchkey {
    pub fn dummy(&self) {
        _ = self.daw_input_conn;
    }
}
