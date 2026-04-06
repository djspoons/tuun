use std::sync::mpsc;

use midir::{Ignore, MidiInput, MidiInputConnection, MidiOutput, MidiOutputConnection};
use midly::MidiMessage;
use midly::live::LiveEvent;

use thiserror::Error;

pub struct Launchkey {
    daw_output_conn: MidiOutputConnection,
    daw_input_conn: MidiInputConnection<()>,
    midi_input_conn: MidiInputConnection<()>,

    pub events: mpsc::Receiver<Event>,
}

enum EncoderMode {
    Plugin,
    Mixer,
}

/// Responsible for decoding and forwarding messages received on the DAW connection.
struct DAWState {
    encoder_mode: EncoderMode,

    sender: mpsc::Sender<Event>,
}

#[derive(Debug)]
pub enum Event {
    NextTrack,
    PreviousTrack,
    NextTrackBank,
    PreviousTrackBank,
    PluginEncoderChange {
        index: u8,
        value: u8, // ranges from 0-127
    },
    /*
    MixerEncoderChange {
        index: u8,
        value: u8,     // ranges from 0-127
    },
    */
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

static ENCODER_OFFSET: u8 = 21;

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
            daw_output_conn.send(&[159, 12, 127])?;
        } else {
            return Err(Error::Other(
                "couldn't find Launchkey DAW In port".to_string(),
            ));
        }

        // Second, set up the output channels for both "DAW" and "MIDI".
        let (sender, receiver) = mpsc::channel();
        let mut daw_state = DAWState {
            encoder_mode: EncoderMode::Plugin,
            sender,
        };
        let mut midi_input = MidiInput::new("tuun reading DAW input")?;
        midi_input.ignore(Ignore::Time);
        let daw_input_conn;
        if let Some(daw_input_port) = midi_input.ports().iter().find(|p| {
            midi_input.port_name(p).unwrap_or("error".to_string())
                == "Launchkey MK4 37 DAW Out".to_string()
        }) {
            daw_input_conn = midi_input.connect(
                daw_input_port,
                "tuun-daw-input-port",
                move |stamp, message, data| daw_state.handle_message(stamp, message, data),
                (),
            )?;
        } else {
            return Err(Error::Other(
                "couldn't find Launchkey DAW Out port".to_string(),
            ));
        }

        let mut midi_input = MidiInput::new("tuun reading MIDI input")?;
        midi_input.ignore(Ignore::Time);
        let midi_input_conn;
        if let Some(midi_input_port) = midi_input.ports().iter().find(|p| {
            midi_input.port_name(p).unwrap_or("error".to_string())
                == "Launchkey MK4 37 MIDI Out".to_string()
        }) {
            midi_input_conn = midi_input.connect(
                midi_input_port,
                "tuun-midi-input-port",
                move |_stamp, message, _data| {
                    println!("Got message on MIDI connection: {:?}", message);
                },
                (),
            )?;
        } else {
            return Err(Error::Other(
                "couldn't find Launchkey midi Out port".to_string(),
            ));
        }

        Ok(Launchkey {
            daw_output_conn,
            daw_input_conn,
            midi_input_conn,

            events: receiver,
        })
    }

    pub fn update_encoder_state(&mut self, index: u8, value: u8) {
        let event = LiveEvent::Midi {
            channel: 15.into(),
            message: MidiMessage::Controller {
                controller: (index + ENCODER_OFFSET).into(),
                value: value.into(),
            },
        };
        let mut buf = Vec::new();
        event.write(&mut buf).unwrap();
        //println!("Sending event {:?} as {:?}", &event, &buf);
        if let Err(e) = self.daw_output_conn.send(&buf) {
            println!("launchkey: got error on send: {}", e);
        }
    }
}

impl Drop for Launchkey {
    fn drop(&mut self) {
        match self.daw_output_conn.send(&[159, 12, 0]) {
            Err(e) => {
                println!("launchkey: failed to exit from DAW mode: {}", e);
            }
            _ => (),
        }
    }
}

impl Launchkey {
    // XXX
    pub fn dummy(&self) {
        _ = self.daw_input_conn;
        _ = self.midi_input_conn;
    }
}

impl DAWState {
    pub fn dummy(&self) {
        _ = EncoderMode::Mixer;
        _ = self.encoder_mode;
    }
}

impl DAWState {
    fn handle_message(&mut self, _stamp: u64, message: &[u8], _info: &mut ()) {
        // XXX
        self.dummy();

        if let Some(event) = self.decode(message) {
            match self.sender.send(event) {
                Ok(()) => (),
                Err(e) => {
                    println!("Got error sending event: {}", e);
                }
            }
        }
    }

    fn decode(&mut self, message: &[u8]) -> Option<Event> {
        let event = LiveEvent::parse(message).unwrap();
        match event {
            LiveEvent::Midi { channel, message } => match message {
                MidiMessage::Controller { controller, value } => {
                    println!(
                        "On channel {}, got controller {} with value {}",
                        channel,
                        controller.as_int(),
                        value.as_int()
                    );
                    match (controller.as_int(), value.as_int()) {
                        // Navigation
                        (102, 127) => Some(Event::NextTrack),
                        (102, 0) => None,
                        (103, 127) => Some(Event::PreviousTrack),
                        (103, 0) => None,
                        (108, 127) => Some(Event::NextTrackBank),
                        (108, 0) => None,
                        (109, 127) => Some(Event::PreviousTrackBank),
                        (109, 0) => None,

                        // Encoders
                        (controller, value)
                            if controller >= ENCODER_OFFSET && controller < ENCODER_OFFSET + 8 =>
                        {
                            Some(Event::PluginEncoderChange {
                                index: controller - ENCODER_OFFSET,
                                value,
                            })
                        }
                        _ => {
                            println!(
                                "Ignoring message Controller({}, {}) on channel {}",
                                controller, value, channel
                            );
                            None
                        }
                    }
                }
                _ => {
                    println!("Ignoring message {:?} on channel {}", message, channel);
                    None
                }
            },
            _ => {
                println!("Ignoring event {:?}", event);
                None
            }
        }
    }
}
