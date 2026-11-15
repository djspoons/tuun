//! The Launchkey MK4 as a control surface: its pads, encoders, buttons and
//! display, through its DAW ports.

mod device;
mod mapping;
mod protocol;

pub use device::Launchkey;
pub use mapping::{classify, update_launchkey_state};
pub use protocol::{Color, EncoderMode, Event, NUM_ENCODERS, PadMode};
