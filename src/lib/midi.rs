//! MIDI ports on the host, through midir, and the MIDI keyboards that play keys.

use std::sync::{Arc, mpsc};
use std::time::Instant;

use midir::{MidiIO, MidiInput, MidiInputConnection, MidiOutput, MidiOutputConnection};
use midly::live;
use thiserror::Error;

use crate::actions::Action;

pub use midir::Ignore;

/// A function that wakes a receiver blocked on something other than a channel.
pub type Waker = Arc<dyn Fn() + Send + Sync>;

/// An error opening a MIDI port.
#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    Init(#[from] midir::InitError),
    #[error(transparent)]
    OutputConnect(#[from] midir::ConnectError<MidiOutput>),
    #[error(transparent)]
    InputConnect(#[from] midir::ConnectError<MidiInput>),
    #[error("no MIDI {direction} port name contains all of {parts:?}")]
    NotFound {
        direction: &'static str,
        parts: Vec<String>,
    },
}

/// The sending end of a channel whose receiver is woken by a [`Waker`].
///
/// Each event is sent, then `wake` is called so that a receiver blocked on
/// something other than the channel notices it.
pub struct EventSink<T> {
    sender: mpsc::Sender<T>,
    wake: Waker,
}

impl<T> EventSink<T> {
    /// Returns a sink that calls `wake` after each event it sends, and the
    /// sink's receiver.
    pub fn channel(wake: Waker) -> (EventSink<T>, mpsc::Receiver<T>) {
        let (sender, receiver) = mpsc::channel();
        (EventSink { sender, wake }, receiver)
    }

    /// Sends `event` on the channel, then wakes its receiver.
    ///
    /// If the receiver is gone, logs a warning and does not wake.
    pub fn send(&self, event: T) {
        match self.sender.send(event) {
            Ok(()) => (self.wake)(),
            Err(e) => log::warn!("Got error sending event: {}", e),
        }
    }
}

// Not derived, which would require `T: Clone`.
impl<T> Clone for EventSink<T> {
    fn clone(&self) -> Self {
        EventSink {
            sender: self.sender.clone(),
            wake: self.wake.clone(),
        }
    }
}

/// An open connection to a MIDI output port.
pub struct Output {
    connection: MidiOutputConnection,
    name: String,
}

impl Output {
    /// Connects to the first output port whose name contains all of `parts`.
    pub fn open(parts: &[&str]) -> Result<Output, Error> {
        let output = MidiOutput::new("tuun output")?;
        let Some((port, name)) = find_port(&output, |name| contains_all(name, parts)) else {
            return Err(not_found("output", parts));
        };
        let connection = output.connect(&port, "tuun-output")?;
        Ok(Output { connection, name })
    }

    /// Returns the full name of the connected port.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Sends `bytes` to the port, logging any error.
    pub fn send(&mut self, bytes: &[u8]) {
        if let Err(e) = self.connection.send(bytes) {
            log::warn!("Error sending {:02X?} to {}: {}", bytes, self.name, e);
        }
    }
}

/// An open connection to a MIDI input port; it is closed when dropped.
pub struct Input {
    _connection: MidiInputConnection<()>,
    name: String,
}

impl Input {
    /// Connects to the first input port whose name contains all of `parts`.
    ///
    /// Messages of the kinds in `ignore` are dropped; `callback` is called
    /// with each other message on midir's input thread.
    pub fn open(
        parts: &[&str],
        ignore: Ignore,
        callback: impl FnMut(&[u8]) + Send + 'static,
    ) -> Result<Input, Error> {
        connect_input(|name| contains_all(name, parts), ignore, callback)?
            .ok_or_else(|| not_found("input", parts))
    }

    /// Returns the full name of the connected port.
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// The MIDI keyboards open for playing notes.
pub struct Keyboards {
    /// The note actions played on every keyboard, stamped on arrival.
    pub events: mpsc::Receiver<Action>,
    _inputs: Vec<Input>,
    names: Vec<String>,
}

impl Keyboards {
    /// Opens each MIDI input port chosen by `exclude` and `only` as a MIDI
    /// keyboard.
    ///
    /// A port is chosen unless its full name is in `exclude` or contains
    /// "Through" or "Thru". Each note played on a keyboard arrives on
    /// [`Keyboards::events`] as the action [`note_action`] returns, and `wake`
    /// is called on midir's input thread after each one. A port that fails to
    /// open is logged and skipped.
    pub fn open(exclude: &[String], wake: Waker) -> Keyboards {
        let (sink, events) = EventSink::channel(wake);
        // TODO: Hot-plug is not handled; ports are scanned only here.
        let all_names: Vec<String> = match MidiInput::new("tuun listing inputs") {
            Ok(input) => ports(&input).into_iter().map(|(_, name)| name).collect(),
            Err(e) => {
                log::warn!("Couldn't list MIDI inputs: {}", e);
                Vec::new()
            }
        };
        let mut inputs = Vec::new();
        let mut names = Vec::new();
        for name in select_keyboards(&all_names, exclude) {
            let sink = sink.clone();
            let callback = move |bytes: &[u8]| {
                // Stamp on arrival rather than converting midir's own stamp: the
                // recorder compares note times against `Instant`s from the tracker.
                if let Some(action) = note_action(bytes, Instant::now()) {
                    sink.send(action);
                }
            };
            // SysEx, timing, and active sensing are never notes.
            match connect_input(|n| n == name, Ignore::All, callback) {
                Ok(Some(input)) => {
                    inputs.push(input);
                    names.push(name);
                }
                Ok(None) => log::warn!("MIDI input {} disappeared before it was opened", name),
                Err(e) => log::warn!("Couldn't open MIDI input {}: {}", name, e),
            }
        }
        Keyboards {
            events,
            _inputs: inputs,
            names,
        }
    }

    /// Returns the full names of the open keyboards' ports.
    pub fn names(&self) -> &[String] {
        &self.names
    }
}

/// Returns the action that plays the note in `bytes`, received at `stamp`.
///
/// Returns `None` if `bytes` is not a complete note on or note off message.
/// A note on with velocity zero is a note off. The channel is ignored, so a
/// note gives the same action on every channel.
pub fn note_action(bytes: &[u8], stamp: Instant) -> Option<Action> {
    let Ok(event) = live::LiveEvent::parse(bytes) else {
        log::debug!("Ignoring malformed MIDI message {:02X?}", bytes);
        return None;
    };
    let live::LiveEvent::Midi { message, .. } = event else {
        return None;
    };
    match message {
        midly::MidiMessage::NoteOn { key, vel } if vel.as_int() > 0 => Some(Action::NoteOn {
            key: key.as_int(),
            velocity: vel.as_int(),
            stamp,
        }),
        midly::MidiMessage::NoteOn { key, .. } | midly::MidiMessage::NoteOff { key, .. } => {
            Some(Action::NoteOff {
                key: key.as_int(),
                stamp,
            })
        }
        _ => None,
    }
}

/// Returns the names in `names` to open as MIDI keyboards, in order.
///
/// A name is chosen unless it is in `exclude` or contains "Through" or "Thru".
fn select_keyboards(names: &[String], exclude: &[String]) -> Vec<String> {
    names
        .iter()
        .filter(|name| !exclude.contains(name))
        .filter(|name| !name.contains("Through") && !name.contains("Thru"))
        .cloned()
        .collect()
}

/// Returns whether `name` contains every one of `parts`.
fn contains_all(name: &str, parts: &[&str]) -> bool {
    parts.iter().all(|part| name.contains(part))
}

/// Returns each port of `io` with its name, skipping ports whose name can't be read.
fn ports<T: MidiIO>(io: &T) -> Vec<(T::Port, String)> {
    io.ports()
        .into_iter()
        .filter_map(|port| {
            let name = io.port_name(&port).ok()?;
            Some((port, name))
        })
        .collect()
}

/// Returns the first port of `io` whose name satisfies `matches`, with that name.
fn find_port<T: MidiIO>(io: &T, matches: impl Fn(&str) -> bool) -> Option<(T::Port, String)> {
    ports(io).into_iter().find(|(_, name)| matches(name))
}

/// Connects `callback` to the first input port whose name satisfies `matches`.
///
/// Returns `Ok(None)` if no port matches.
fn connect_input(
    matches: impl Fn(&str) -> bool,
    ignore: Ignore,
    mut callback: impl FnMut(&[u8]) + Send + 'static,
) -> Result<Option<Input>, Error> {
    // Each connection needs its own client, since connecting consumes it.
    let mut input = MidiInput::new("tuun input")?;
    input.ignore(ignore);
    let Some((port, name)) = find_port(&input, matches) else {
        return Ok(None);
    };
    let connection = input.connect(&port, "tuun-input", move |_, bytes, _| callback(bytes), ())?;
    Ok(Some(Input {
        _connection: connection,
        name,
    }))
}

/// Returns the error for a missing `direction` port named by `parts`.
fn not_found(direction: &'static str, parts: &[&str]) -> Error {
    Error::NotFound {
        direction,
        parts: parts.iter().map(|part| part.to_string()).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn sink_wakes_only_after_a_sent_event() {
        let wakes = Arc::new(AtomicUsize::new(0));
        let (sink, receiver) = EventSink::channel(Arc::new({
            let wakes = wakes.clone();
            move || {
                wakes.fetch_add(1, Ordering::SeqCst);
            }
        }));

        sink.send("event");
        assert_eq!(wakes.load(Ordering::SeqCst), 1);
        assert_eq!(receiver.try_recv(), Ok("event"));

        // With the receiver gone the send fails, and nothing is woken.
        drop(receiver);
        sink.send("event");
        assert_eq!(wakes.load(Ordering::SeqCst), 1);
    }
}
