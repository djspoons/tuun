use std::sync::mpsc;

use midir::{Ignore, MidiInput, MidiInputConnection, MidiOutput, MidiOutputConnection};
use midly::MidiMessage;
use midly::live::LiveEvent;
use midly::num::u7;

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
    NextTrackDown,
    PreviousTrackDown,
    NextTrackBankDown,
    PreviousTrackBankDown,
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
    DAWTopPadDown {
        index: u8,
    },
    DAWBottomPadDown {
        index: u8,
    },

    PadFunctionDown,
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

const ENCODER_OFFSET: u8 = 21;
const NUM_ENCODERS: u8 = 8;

const ENCODER_UPDATE_CHANNEL: u8 = 15;

const DAW_PAD_TOP_ROW_OFFSET: u8 = 96;
const DAW_PAD_BOTTOM_ROW_OFFSET: u8 = 112;
const NUM_DAW_PADS_PER_ROW: u8 = 8;

const PAD_FUNCTION_OFFSET: u8 = 105;

// For SysEx messages
const STANDARD_SKU_PREFIX: [u8; 5] = [0, 32, 41, 2, 20];
const PAD_RGB_COLOR: [u8; 2] = [1, 67];

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

    fn send_event(&mut self, event: LiveEvent) {
        let mut buf = Vec::new();
        event.write(&mut buf).unwrap();
        //println!("Sending event {:?} as {:?}", &event, &buf);
        if let Err(e) = self.daw_output_conn.send(&buf) {
            println!("launchkey: got error on send: {}", e);
        }
    }

    pub fn update_encoder_state(&mut self, index: u8, value: u8) {
        self.send_event(LiveEvent::Midi {
            channel: ENCODER_UPDATE_CHANNEL.into(),
            message: MidiMessage::Controller {
                controller: (index + ENCODER_OFFSET).into(),
                value: value.into(),
            },
        });
    }

    pub fn set_daw_top_pad_color(&mut self, index: u8, red: u8, green: u8, blue: u8) {
        let pad_id = index + DAW_PAD_TOP_ROW_OFFSET;
        let mut buf = Vec::new();
        buf.extend(&STANDARD_SKU_PREFIX);
        buf.extend(&PAD_RGB_COLOR);
        buf.push(pad_id);
        buf.push(red.min(127));
        buf.push(green.min(127));
        buf.push(blue.min(127));
        let msg: Vec<u7> = buf.iter().map(|&x| u7::new(x & 0x7F)).collect();
        self.send_event(LiveEvent::Common(midly::live::SystemCommon::SysEx(&msg)));
    }

    pub fn set_daw_bottom_pad_color(&mut self, index: u8, red: u8, green: u8, blue: u8) {
        let pad_id = index + DAW_PAD_BOTTOM_ROW_OFFSET;
        let mut buf = Vec::new();
        buf.extend(&STANDARD_SKU_PREFIX);
        buf.extend(&PAD_RGB_COLOR);
        buf.push(pad_id);
        buf.push(red.min(127));
        buf.push(green.min(127));
        buf.push(blue.min(127));
        let msg: Vec<u7> = buf.iter().map(|&x| u7::new(x & 0x7F)).collect();
        self.send_event(LiveEvent::Common(midly::live::SystemCommon::SysEx(&msg)));
    }

    pub fn set_pad_function_color(&mut self, color: Color) {
        self.send_event(LiveEvent::Midi {
            channel: 0.into(),
            message: MidiMessage::Controller {
                controller: PAD_FUNCTION_OFFSET.into(),
                value: (color as u8).into(),
            },
        });
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
                        (102, 127) => Some(Event::NextTrackDown),
                        (102, 0) => None,
                        (103, 127) => Some(Event::PreviousTrackDown),
                        (103, 0) => None,
                        (108, 127) => Some(Event::NextTrackBankDown),
                        (108, 0) => None,
                        (109, 127) => Some(Event::PreviousTrackBankDown),
                        (109, 0) => None,

                        // Encoders
                        (controller, value)
                            if controller >= ENCODER_OFFSET
                                && controller < ENCODER_OFFSET + NUM_ENCODERS =>
                        {
                            Some(Event::PluginEncoderChange {
                                index: controller - ENCODER_OFFSET,
                                value,
                            })
                        }

                        // Other buttons
                        (PAD_FUNCTION_OFFSET, 127) => Some(Event::PadFunctionDown),
                        (PAD_FUNCTION_OFFSET, 0) => None,

                        _ => {
                            println!(
                                "Ignoring message Controller({}, {}) on channel {}",
                                controller, value, channel
                            );
                            None
                        }
                    }
                }
                MidiMessage::NoteOn { key, vel } => {
                    println!(
                        "On channel {}, got note-on for key {} with velocity {}",
                        channel,
                        key.as_int(),
                        vel.as_int()
                    );

                    if vel > 0 {
                        if key >= DAW_PAD_TOP_ROW_OFFSET
                            && key < DAW_PAD_TOP_ROW_OFFSET + NUM_DAW_PADS_PER_ROW
                        {
                            Some(Event::DAWTopPadDown {
                                index: key.as_int() - DAW_PAD_TOP_ROW_OFFSET,
                            })
                        } else if key >= DAW_PAD_BOTTOM_ROW_OFFSET
                            && key < DAW_PAD_BOTTOM_ROW_OFFSET + NUM_DAW_PADS_PER_ROW
                        {
                            Some(Event::DAWBottomPadDown {
                                index: key.as_int() - DAW_PAD_BOTTOM_ROW_OFFSET,
                            })
                        } else {
                            None
                        }
                    } else {
                        None
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

// Meaningful names for colors provided in the standard palette.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Color {
    /// RGB(97, 97, 97)
    Gray = 0,
    /// RGB(179, 179, 179)
    LightGray = 1,
    /// RGB(221, 221, 221)
    Silver = 2,
    /// RGB(255, 255, 255)
    White = 3,
    /// RGB(253, 179, 179)
    PaleRose = 4,
    /// RGB(251, 99, 95)
    BrightRed = 5,
    /// RGB(218, 98, 97)
    DustyRed = 6,
    /// RGB(179, 97, 98)
    DarkRose = 7,
    /// RGB(254, 243, 214)
    Cream = 8,
    /// RGB(254, 179, 92)
    Tangerine = 9,
    /// RGB(218, 139, 96)
    Rust = 10,
    /// RGB(179, 118, 95)
    Sienna = 11,
    /// RGB(255, 236, 158)
    Buttercup = 12,
    /// RGB(254, 254, 99)
    BrightYellow = 13,
    /// RGB(222, 223, 98)
    Olive = 14,
    /// RGB(179, 178, 98)
    DarkKhaki = 15,
    /// RGB(222, 254, 162)
    PaleChartreuse = 16,
    /// RGB(191, 255, 99)
    LimeGreen = 17,
    /// RGB(160, 221, 89)
    GrassGreen = 18,
    /// RGB(129, 178, 101)
    FernGreen = 19,
    /// RGB(195, 252, 180)
    PaleMint = 20,
    /// RGB(99, 254, 97)
    BrightGreen = 21,
    /// RGB(88, 222, 90)
    Emerald = 22,
    /// RGB(100, 178, 98)
    ForestGreen = 23,
    /// RGB(196, 254, 201)
    PaleShamrock = 24,
    /// RGB(93, 253, 138)
    SpringGreen = 25,
    /// RGB(94, 222, 119)
    KellyGreen = 26,
    /// RGB(96, 177, 110)
    HunterGreen = 27,
    /// RGB(202, 253, 207)
    PaleSeafoam = 28,
    /// RGB(94, 254, 202)
    MintGreen = 29,
    /// RGB(97, 221, 159)
    Jade = 30,
    /// RGB(99, 179, 130)
    SageGreen = 31,
    /// RGB(196, 253, 243)
    PaleAqua = 32,
    /// RGB(94, 255, 233)
    BrightCyan = 33,
    /// RGB(98, 221, 193)
    Teal = 34,
    /// RGB(101, 178, 150)
    DarkTeal = 35,
    /// RGB(199, 241, 254)
    PaleSky = 36,
    /// RGB(96, 241, 255)
    ElectricCyan = 37,
    /// RGB(95, 198, 219)
    SteelBlue = 38,
    /// RGB(102, 158, 178)
    SlateBlue = 39,
    /// RGB(196, 220, 255)
    PalePeriwinkle = 40,
    /// RGB(95, 200, 253)
    SkyBlue = 41,
    /// RGB(93, 163, 225)
    CeruleanBlue = 42,
    /// RGB(98, 128, 180)
    DenimBlue = 43,
    /// RGB(160, 139, 253)
    LightIndigo = 44,
    /// RGB(99, 94, 255)
    BrightBlue = 45,
    /// RGB(97, 96, 223)
    RoyalBlue = 46,
    /// RGB(97, 97, 181)
    NavyBlue = 47,
    /// RGB(202, 179, 253)
    PaleLavender = 48,
    /// RGB(159, 97, 255)
    BrightViolet = 49,
    /// RGB(129, 97, 220)
    DeepViolet = 50,
    /// RGB(119, 95, 178)
    DarkViolet = 51,
    /// RGB(251, 180, 251)
    PaleOrchid = 52,
    /// RGB(254, 96, 252)
    BrightMagenta = 53,
    /// RGB(221, 96, 218)
    DarkMagenta = 54,
    /// RGB(178, 94, 179)
    Plum = 55,
    /// RGB(255, 180, 217)
    PalePink = 56,
    /// RGB(250, 97, 194)
    HotPink = 57,
    /// RGB(219, 94, 162)
    Raspberry = 58,
    /// RGB(178, 96, 140)
    DarkRaspberry = 59,
    /// RGB(254, 117, 97)
    Vermilion = 60,
    /// RGB(232, 181, 97)
    Amber = 61,
    /// RGB(219, 194, 94)
    DarkGold = 62,
    /// RGB(161, 158, 100)
    OliveGreen = 63,
    /// RGB(101, 176, 96)
    MossGreen = 64,
    /// RGB(100, 178, 138)
    SeaGreen = 65,
    /// RGB(97, 138, 211)
    WedgwoodBlue = 66,
    /// RGB(96, 95, 255)
    ElectricBlue = 67,
    /// RGB(95, 175, 175)
    CadetTeal = 68,
    /// RGB(140, 97, 248)
    Amethyst = 69,
    /// RGB(206, 179, 195)
    PaleMauve = 70,
    /// RGB(138, 116, 129)
    Taupe = 71,
    /// RGB(253, 91, 95)
    Scarlet = 72,
    /// RGB(244, 254, 159)
    PaleLime = 73,
    /// RGB(242, 254, 98)
    LemonYellow = 74,
    /// RGB(209, 255, 100)
    Chartreuse = 75,
    /// RGB(123, 223, 99)
    LeafGreen = 76,
    /// RGB(96, 255, 204)
    Aquamarine = 77,
    /// RGB(95, 235, 251)
    BrightSkyBlue = 78,
    /// RGB(92, 159, 255)
    CornflowerBlue = 79,
    /// RGB(142, 96, 255)
    BlueViolet = 80,
    /// RGB(203, 97, 250)
    Orchid = 81,
    /// RGB(238, 138, 217)
    PinkLavender = 82,
    /// RGB(159, 118, 99)
    Mocha = 83,
    /// RGB(255, 161, 94)
    BrightOrange = 84,
    /// RGB(223, 253, 94)
    YellowGreen = 85,
    /// RGB(211, 253, 139)
    PaleGreen = 86,
    /// RGB(98, 254, 94)
    NeonGreen = 87,
    /// RGB(184, 250, 168)
    MintCream = 88,
    /// RGB(202, 248, 212)
    PaleSage = 89,
    /// RGB(184, 254, 249)
    LightCyan = 90,
    /// RGB(207, 230, 255)
    IceBlue = 91,
    /// RGB(159, 194, 245)
    BabyBlue = 92,
    /// RGB(212, 192, 246)
    Wisteria = 93,
    /// RGB(247, 143, 252)
    BrightOrchid = 94,
    /// RGB(253, 93, 203)
    Fuchsia = 95,
    /// RGB(251, 195, 96)
    GoldenOrange = 96,
    /// RGB(243, 238, 101)
    PaleGold = 97,
    /// RGB(230, 255, 99)
    LimeYellow = 98,
    /// RGB(220, 204, 91)
    DarkYellow = 99,
    /// RGB(182, 165, 94)
    Bronze = 100,
    /// RGB(97, 184, 118)
    MediumSeaGreen = 101,
    /// RGB(126, 200, 141)
    CeladonGreen = 102,
    /// RGB(130, 131, 163)
    CoolGray = 103,
    /// RGB(125, 139, 209)
    MutedPeriwinkle = 104,
    /// RGB(207, 171, 131)
    Tan = 105,
    /// RGB(225, 98, 94)
    Coral = 106,
    /// RGB(245, 175, 158)
    Salmon = 107,
    /// RGB(251, 184, 115)
    Apricot = 108,
    /// RGB(255, 244, 136)
    Canary = 109,
    /// RGB(232, 247, 162)
    PaleYellowGreen = 110,
    /// RGB(212, 236, 122)
    Pistachio = 111,
    /// RGB(128, 127, 161)
    StormGray = 112,
    /// RGB(249, 248, 214)
    Cornsilk = 113,
    /// RGB(221, 251, 223)
    Honeydew = 114,
    /// RGB(230, 229, 252)
    LightLavender = 115,
    /// RGB(228, 212, 252)
    PalePlum = 116,
    /// RGB(179, 179, 179)
    Ash = 117,
    /// RGB(212, 212, 212)
    LightSilver = 118,
    /// RGB(248, 254, 254)
    MintWhite = 119,
    /// RGB(237, 100, 95)
    Tomato = 120,
    /// RGB(171, 94, 97)
    Rosewood = 121,
    /// RGB(128, 245, 97)
    ParrotGreen = 122,
    /// RGB(98, 178, 94)
    DarkMoss = 123,
    /// RGB(242, 235, 97)
    Goldenrod = 124,
    /// RGB(179, 162, 98)
    DarkTan = 125,
    /// RGB(233, 191, 95)
    Honey = 126,
    /// RGB(198, 117, 96)
    Copper = 127,
}

impl Color {
    /// Convert a u8 index (0–127) to a `Color` variant.
    pub fn from_index(index: u8) -> Option<Self> {
        if index < 128 {
            // SAFETY: all values 0..128 are valid variants due to #[repr(u8)]
            Some(unsafe { std::mem::transmute(index) })
        } else {
            None
        }
    }

    /// Returns the RGB triple for this color.
    pub fn rgb(self) -> (u8, u8, u8) {
        match self {
            Color::Gray => (97, 97, 97),
            Color::LightGray => (179, 179, 179),
            Color::Silver => (221, 221, 221),
            Color::White => (255, 255, 255),
            Color::PaleRose => (253, 179, 179),
            Color::BrightRed => (251, 99, 95),
            Color::DustyRed => (218, 98, 97),
            Color::DarkRose => (179, 97, 98),
            Color::Cream => (254, 243, 214),
            Color::Tangerine => (254, 179, 92),
            Color::Rust => (218, 139, 96),
            Color::Sienna => (179, 118, 95),
            Color::Buttercup => (255, 236, 158),
            Color::BrightYellow => (254, 254, 99),
            Color::Olive => (222, 223, 98),
            Color::DarkKhaki => (179, 178, 98),
            Color::PaleChartreuse => (222, 254, 162),
            Color::LimeGreen => (191, 255, 99),
            Color::GrassGreen => (160, 221, 89),
            Color::FernGreen => (129, 178, 101),
            Color::PaleMint => (195, 252, 180),
            Color::BrightGreen => (99, 254, 97),
            Color::Emerald => (88, 222, 90),
            Color::ForestGreen => (100, 178, 98),
            Color::PaleShamrock => (196, 254, 201),
            Color::SpringGreen => (93, 253, 138),
            Color::KellyGreen => (94, 222, 119),
            Color::HunterGreen => (96, 177, 110),
            Color::PaleSeafoam => (202, 253, 207),
            Color::MintGreen => (94, 254, 202),
            Color::Jade => (97, 221, 159),
            Color::SageGreen => (99, 179, 130),
            Color::PaleAqua => (196, 253, 243),
            Color::BrightCyan => (94, 255, 233),
            Color::Teal => (98, 221, 193),
            Color::DarkTeal => (101, 178, 150),
            Color::PaleSky => (199, 241, 254),
            Color::ElectricCyan => (96, 241, 255),
            Color::SteelBlue => (95, 198, 219),
            Color::SlateBlue => (102, 158, 178),
            Color::PalePeriwinkle => (196, 220, 255),
            Color::SkyBlue => (95, 200, 253),
            Color::CeruleanBlue => (93, 163, 225),
            Color::DenimBlue => (98, 128, 180),
            Color::LightIndigo => (160, 139, 253),
            Color::BrightBlue => (99, 94, 255),
            Color::RoyalBlue => (97, 96, 223),
            Color::NavyBlue => (97, 97, 181),
            Color::PaleLavender => (202, 179, 253),
            Color::BrightViolet => (159, 97, 255),
            Color::DeepViolet => (129, 97, 220),
            Color::DarkViolet => (119, 95, 178),
            Color::PaleOrchid => (251, 180, 251),
            Color::BrightMagenta => (254, 96, 252),
            Color::DarkMagenta => (221, 96, 218),
            Color::Plum => (178, 94, 179),
            Color::PalePink => (255, 180, 217),
            Color::HotPink => (250, 97, 194),
            Color::Raspberry => (219, 94, 162),
            Color::DarkRaspberry => (178, 96, 140),
            Color::Vermilion => (254, 117, 97),
            Color::Amber => (232, 181, 97),
            Color::DarkGold => (219, 194, 94),
            Color::OliveGreen => (161, 158, 100),
            Color::MossGreen => (101, 176, 96),
            Color::SeaGreen => (100, 178, 138),
            Color::WedgwoodBlue => (97, 138, 211),
            Color::ElectricBlue => (96, 95, 255),
            Color::CadetTeal => (95, 175, 175),
            Color::Amethyst => (140, 97, 248),
            Color::PaleMauve => (206, 179, 195),
            Color::Taupe => (138, 116, 129),
            Color::Scarlet => (253, 91, 95),
            Color::PaleLime => (244, 254, 159),
            Color::LemonYellow => (242, 254, 98),
            Color::Chartreuse => (209, 255, 100),
            Color::LeafGreen => (123, 223, 99),
            Color::Aquamarine => (96, 255, 204),
            Color::BrightSkyBlue => (95, 235, 251),
            Color::CornflowerBlue => (92, 159, 255),
            Color::BlueViolet => (142, 96, 255),
            Color::Orchid => (203, 97, 250),
            Color::PinkLavender => (238, 138, 217),
            Color::Mocha => (159, 118, 99),
            Color::BrightOrange => (255, 161, 94),
            Color::YellowGreen => (223, 253, 94),
            Color::PaleGreen => (211, 253, 139),
            Color::NeonGreen => (98, 254, 94),
            Color::MintCream => (184, 250, 168),
            Color::PaleSage => (202, 248, 212),
            Color::LightCyan => (184, 254, 249),
            Color::IceBlue => (207, 230, 255),
            Color::BabyBlue => (159, 194, 245),
            Color::Wisteria => (212, 192, 246),
            Color::BrightOrchid => (247, 143, 252),
            Color::Fuchsia => (253, 93, 203),
            Color::GoldenOrange => (251, 195, 96),
            Color::PaleGold => (243, 238, 101),
            Color::LimeYellow => (230, 255, 99),
            Color::DarkYellow => (220, 204, 91),
            Color::Bronze => (182, 165, 94),
            Color::MediumSeaGreen => (97, 184, 118),
            Color::CeladonGreen => (126, 200, 141),
            Color::CoolGray => (130, 131, 163),
            Color::MutedPeriwinkle => (125, 139, 209),
            Color::Tan => (207, 171, 131),
            Color::Coral => (225, 98, 94),
            Color::Salmon => (245, 175, 158),
            Color::Apricot => (251, 184, 115),
            Color::Canary => (255, 244, 136),
            Color::PaleYellowGreen => (232, 247, 162),
            Color::Pistachio => (212, 236, 122),
            Color::StormGray => (128, 127, 161),
            Color::Cornsilk => (249, 248, 214),
            Color::Honeydew => (221, 251, 223),
            Color::LightLavender => (230, 229, 252),
            Color::PalePlum => (228, 212, 252),
            Color::Ash => (179, 179, 179),
            Color::LightSilver => (212, 212, 212),
            Color::MintWhite => (248, 254, 254),
            Color::Tomato => (237, 100, 95),
            Color::Rosewood => (171, 94, 97),
            Color::ParrotGreen => (128, 245, 97),
            Color::DarkMoss => (98, 178, 94),
            Color::Goldenrod => (242, 235, 97),
            Color::DarkTan => (179, 162, 98),
            Color::Honey => (233, 191, 95),
            Color::Copper => (198, 117, 96),
        }
    }

    /// Returns a human-readable name for this color.
    pub fn name(self) -> &'static str {
        match self {
            Color::Gray => "Gray",
            Color::LightGray => "Light Gray",
            Color::Silver => "Silver",
            Color::White => "White",
            Color::PaleRose => "Pale Rose",
            Color::BrightRed => "Bright Red",
            Color::DustyRed => "Dusty Red",
            Color::DarkRose => "Dark Rose",
            Color::Cream => "Cream",
            Color::Tangerine => "Tangerine",
            Color::Rust => "Rust",
            Color::Sienna => "Sienna",
            Color::Buttercup => "Buttercup",
            Color::BrightYellow => "Bright Yellow",
            Color::Olive => "Olive",
            Color::DarkKhaki => "Dark Khaki",
            Color::PaleChartreuse => "Pale Chartreuse",
            Color::LimeGreen => "Lime Green",
            Color::GrassGreen => "Grass Green",
            Color::FernGreen => "Fern Green",
            Color::PaleMint => "Pale Mint",
            Color::BrightGreen => "Bright Green",
            Color::Emerald => "Emerald",
            Color::ForestGreen => "Forest Green",
            Color::PaleShamrock => "Pale Shamrock",
            Color::SpringGreen => "Spring Green",
            Color::KellyGreen => "Kelly Green",
            Color::HunterGreen => "Hunter Green",
            Color::PaleSeafoam => "Pale Seafoam",
            Color::MintGreen => "Mint Green",
            Color::Jade => "Jade",
            Color::SageGreen => "Sage Green",
            Color::PaleAqua => "Pale Aqua",
            Color::BrightCyan => "Bright Cyan",
            Color::Teal => "Teal",
            Color::DarkTeal => "Dark Teal",
            Color::PaleSky => "Pale Sky",
            Color::ElectricCyan => "Electric Cyan",
            Color::SteelBlue => "Steel Blue",
            Color::SlateBlue => "Slate Blue",
            Color::PalePeriwinkle => "Pale Periwinkle",
            Color::SkyBlue => "Sky Blue",
            Color::CeruleanBlue => "Cerulean Blue",
            Color::DenimBlue => "Denim Blue",
            Color::LightIndigo => "Light Indigo",
            Color::BrightBlue => "Bright Blue",
            Color::RoyalBlue => "Royal Blue",
            Color::NavyBlue => "Navy Blue",
            Color::PaleLavender => "Pale Lavender",
            Color::BrightViolet => "Bright Violet",
            Color::DeepViolet => "Deep Violet",
            Color::DarkViolet => "Dark Violet",
            Color::PaleOrchid => "Pale Orchid",
            Color::BrightMagenta => "Bright Magenta",
            Color::DarkMagenta => "Dark Magenta",
            Color::Plum => "Plum",
            Color::PalePink => "Pale Pink",
            Color::HotPink => "Hot Pink",
            Color::Raspberry => "Raspberry",
            Color::DarkRaspberry => "Dark Raspberry",
            Color::Vermilion => "Vermilion",
            Color::Amber => "Amber",
            Color::DarkGold => "Dark Gold",
            Color::OliveGreen => "Olive Green",
            Color::MossGreen => "Moss Green",
            Color::SeaGreen => "Sea Green",
            Color::WedgwoodBlue => "Wedgwood Blue",
            Color::ElectricBlue => "Electric Blue",
            Color::CadetTeal => "Cadet Teal",
            Color::Amethyst => "Amethyst",
            Color::PaleMauve => "Pale Mauve",
            Color::Taupe => "Taupe",
            Color::Scarlet => "Scarlet",
            Color::PaleLime => "Pale Lime",
            Color::LemonYellow => "Lemon Yellow",
            Color::Chartreuse => "Chartreuse",
            Color::LeafGreen => "Leaf Green",
            Color::Aquamarine => "Aquamarine",
            Color::BrightSkyBlue => "Bright Sky Blue",
            Color::CornflowerBlue => "Cornflower Blue",
            Color::BlueViolet => "Blue Violet",
            Color::Orchid => "Orchid",
            Color::PinkLavender => "Pink Lavender",
            Color::Mocha => "Mocha",
            Color::BrightOrange => "Bright Orange",
            Color::YellowGreen => "Yellow Green",
            Color::PaleGreen => "Pale Green",
            Color::NeonGreen => "Neon Green",
            Color::MintCream => "Mint Cream",
            Color::PaleSage => "Pale Sage",
            Color::LightCyan => "Light Cyan",
            Color::IceBlue => "Ice Blue",
            Color::BabyBlue => "Baby Blue",
            Color::Wisteria => "Wisteria",
            Color::BrightOrchid => "Bright Orchid",
            Color::Fuchsia => "Fuchsia",
            Color::GoldenOrange => "Golden Orange",
            Color::PaleGold => "Pale Gold",
            Color::LimeYellow => "Lime Yellow",
            Color::DarkYellow => "Dark Yellow",
            Color::Bronze => "Bronze",
            Color::MediumSeaGreen => "Medium Sea Green",
            Color::CeladonGreen => "Celadon Green",
            Color::CoolGray => "Cool Gray",
            Color::MutedPeriwinkle => "Muted Periwinkle",
            Color::Tan => "Tan",
            Color::Coral => "Coral",
            Color::Salmon => "Salmon",
            Color::Apricot => "Apricot",
            Color::Canary => "Canary",
            Color::PaleYellowGreen => "Pale Yellow Green",
            Color::Pistachio => "Pistachio",
            Color::StormGray => "Storm Gray",
            Color::Cornsilk => "Cornsilk",
            Color::Honeydew => "Honeydew",
            Color::LightLavender => "Light Lavender",
            Color::PalePlum => "Pale Plum",
            Color::Ash => "Ash",
            Color::LightSilver => "Light Silver",
            Color::MintWhite => "Mint White",
            Color::Tomato => "Tomato",
            Color::Rosewood => "Rosewood",
            Color::ParrotGreen => "Parrot Green",
            Color::DarkMoss => "Dark Moss",
            Color::Goldenrod => "Goldenrod",
            Color::DarkTan => "Dark Tan",
            Color::Honey => "Honey",
            Color::Copper => "Copper",
        }
    }
}
