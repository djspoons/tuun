//! The connection to a Launchkey MK4 in DAW mode.

use std::sync::{Arc, mpsc};

use crate::launchkey::protocol::{self, Color, DAWState, EncoderMode, Event, PadMode};
use crate::midi::{self, EventSink, Ignore, Input, Output};

pub struct Launchkey {
    daw_output: Output,
    daw_input: Input,

    pub events: mpsc::Receiver<Event>,
    /// Main-thread mirror of `DAWState::encoder_mode`, kept in sync by
    /// observing `Event::EncoderModeChanged`. Lets the runner decide what
    /// to push back without crossing the input-thread boundary.
    pub encoder_mode: EncoderMode,
    /// Main-thread mirror of `DAWState::pad_mode`.
    //
    // Kept in sync by observing `Event::PadModeChanged`.
    pub pad_mode: PadMode,
}

impl Launchkey {
    /// Connects to the Launchkey and puts it into DAW mode.
    ///
    /// Claims the output and input ports whose names contain "Launchkey MK4"
    /// and "DAW", and returns an error if either is missing. Decoded input
    /// arrives on [`Launchkey::events`]; `wake` is called on MIDI's input
    /// thread after each event is sent there.
    pub fn new(wake: impl Fn() + Send + Sync + 'static) -> Result<Self, midi::Error> {
        // The DAW ports are matched loosely so that every MK4 size and each
        // OS's port naming are claimed.
        const DAW_PORT: &[&str] = &["Launchkey MK4", "DAW"];

        // First, connect to the controller's "DAW In" port and put the controller into DAW mode.
        let mut daw_output = Output::open(DAW_PORT)?;
        // Enter DAW mode.
        daw_output.send(&protocol::daw_mode(true));

        // Second, read from the controller's "DAW" output port.
        let (events, receiver) = EventSink::channel(Arc::new(wake));
        let mut daw_state = DAWState::new();
        let daw_input = Input::open(DAW_PORT, Ignore::Time, move |message| {
            if let Some(event) = daw_state.decode(message) {
                events.send(event);
            }
        })?;
        log::info!(
            "Launchkey claimed output port {:?} and input port {:?}",
            daw_output.name(),
            daw_input.name()
        );

        let mut launchkey = Launchkey {
            daw_output,
            daw_input,

            events: receiver,
            encoder_mode: EncoderMode::Plugin,
            // Entering DAW mode resets pads to the DAW layout (see page 10).
            pad_mode: PadMode::DAW,
        };
        launchkey.set_encoder_relative_output();
        Ok(launchkey)
    }

    /// Returns the full names of the input ports the Launchkey claimed.
    pub fn claimed_inputs(&self) -> Vec<String> {
        vec![self.daw_input.name().to_string()]
    }

    /// Sends each of `messages` to the controller.
    fn send_all(&mut self, messages: Vec<Vec<u8>>) {
        for message in messages {
            self.daw_output.send(&message);
        }
    }

    /// (Re-)enables "DAW Encoder Relative output" on the controller. The
    /// device resets this when the user switches encoder mode (Plugin /
    /// Mixer), so it has to be re-asserted on every mode change
    /// to keep encoders emitting deltas instead of absolute positions.
    pub fn set_encoder_relative_output(&mut self) {
        self.daw_output
            .send(&protocol::encoder_relative_output(true));
    }

    pub fn set_daw_top_pad_color(&mut self, index: u8, red: u8, green: u8, blue: u8) {
        self.daw_output
            .send(&protocol::daw_top_pad_color(index, red, green, blue));
    }

    pub fn set_daw_bottom_pad_color(&mut self, index: u8, red: u8, green: u8, blue: u8) {
        self.daw_output
            .send(&protocol::daw_bottom_pad_color(index, red, green, blue));
    }

    pub fn set_launch_color(&mut self, color: Color) {
        self.daw_output.send(&protocol::launch_color(color));
    }

    /// Sets the brightness of the Record button's LED, from 0 (dim) to 127.
    pub fn set_record_brightness(&mut self, brightness: u8) {
        self.daw_output
            .send(&protocol::record_brightness(brightness));
    }

    /// Sets the brightness of the Loop button's LED, from 0 (dim) to 127.
    pub fn set_loop_brightness(&mut self, brightness: u8) {
        self.daw_output.send(&protocol::loop_brightness(brightness));
    }

    /// Sets the brightness of the Metronome button's LED, from 0 (dim) to 127.
    pub fn set_metronome_brightness(&mut self, brightness: u8) {
        self.daw_output
            .send(&protocol::metronome_brightness(brightness));
    }

    /// Shows `name` above `text` on the device screen.
    pub fn set_daw_mode_display(&mut self, name: &str, text: &str) {
        self.send_all(protocol::daw_mode_display(name, text));
    }

    pub fn set_encoder_display(&mut self, index: u8, name: &str, value: &str) {
        self.send_all(protocol::encoder_display(index, name, value));
    }
}

impl Drop for Launchkey {
    fn drop(&mut self) {
        // Revert encoders to absolute output before leaving DAW mode.
        self.daw_output
            .send(&protocol::encoder_relative_output(false));
        // Exit DAW mode.
        self.daw_output.send(&protocol::daw_mode(false));
    }
}
