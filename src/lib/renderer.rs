use std::collections::HashMap;
use std::fmt;
use std::time::Duration;
use std::time::Instant;

use sdl2::Sdl;
use sdl2::pixels::Color;
use sdl2::render::{TextureCreator, TextureQuery};
use sdl2::ttf::{Font, Sdl2TtfContext};
use sdl2::video::WindowContext;

use realfft::RealFftPlanner;
use realfft::num_complex::{Complex, ComplexFloat};

use crate::metric::Metric;
use crate::optimizer;
use crate::parser;
use crate::slider;
use crate::tracker;
use crate::waveform;

// TODO: rename Program as Clip? Or make Clip a type of program?
// And the other type is Key(channel, key_number)?

pub type ProgramId = i32;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum WaveformId {
    // Beats are silent waveforms that are used to keep time. The bool tracks whether
    // it is an odd or even measure (false == odd).
    Beats(bool),
    Program(ProgramId),
}

impl WaveformId {
    pub fn is_beats(&self) -> bool {
        match self {
            WaveformId::Beats(_) => true,
            _ => false,
        }
    }
}

// These two functions allow for explicit conversion from index to id.

pub fn index_from_id(id: ProgramId) -> usize {
    return (id - 1) as usize;
}

pub fn id_from_index(index: usize) -> ProgramId {
    return (index + 1) as ProgramId;
}

#[derive(Clone, Debug, PartialEq)]
pub enum MarkId {
    TopLevel, // a mark for the whole Program
    Slider(String),
    UserDefined(u32),
}

impl fmt::Display for MarkId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MarkId::TopLevel => write!(f, "top-level"),
            MarkId::Slider(name) => write!(f, "slider({:?})", name),
            MarkId::UserDefined(id) => write!(f, "{:?}", id),
        }
    }
}

impl MarkId {
    /// Returns a 0-based index for color and position calculations, or None if not applicable.
    fn color_index(&self) -> Option<usize> {
        match self {
            MarkId::TopLevel => None,
            MarkId::UserDefined(id) => Some((*id as usize).saturating_sub(1)),
            MarkId::Slider(_) => None,
        }
    }
}

#[derive(Debug)]
// `active_program_index` should be the index of the active program in the `programs` array
pub enum Mode {
    Select {
        active_program_index: usize,
        message: String,
    },
    Edit {
        active_program_index: usize,
        // TODO unicode!!
        cursor_position: usize, // Cursor is located before the character this position
        errors: Vec<parser::Error>,
        message: String,
    },
    MoveSliders {
        active_program_index: usize, // Don't forget this
    },
    // The following are transient modes that are used to indicate an action should be
    // taken. They are used either when the action requires significant computation or
    // modifies the context.
    Play {
        active_program_index: usize,
        cursor_position: usize,
        program: Program,
        // After how many measures should this program repeat (if any)
        repeat_after_measures: Option<u32>,
    },
    LoadContext {
        active_program_index: usize,
    },
    LoadPrograms {
        active_program_index: usize,
    },
    Exit,
}

pub enum WaveformOrMode {
    Waveform(waveform::Waveform<MarkId>),
    Mode(Mode),
}

#[derive(Debug, Clone, Default)]
pub struct ProgramSliders {
    pub configs: Vec<slider::SliderConfig>,
    /// Normalized values in 0.0..1.0, parallel to configs
    pub normalized_values: Vec<f32>,
}

impl ProgramSliders {
    pub fn slider_display(&self) -> Vec<SliderDisplay> {
        self.configs
            .iter()
            .enumerate()
            .map(|(j, config)| {
                let norm = self.normalized_values[j];
                SliderDisplay {
                    label: config.label.clone(),
                    axis: if j == 0 { "X" } else { "Y" },
                    normalized_value: norm,
                    actual_value: config.min + norm * (config.max - config.min),
                }
            })
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct Program {
    pub text: String,
    pub id: ProgramId,
    pub sliders: ProgramSliders,
}

pub struct SliderDisplay {
    pub label: String,
    pub axis: &'static str, // "X" or "Y"
    pub normalized_value: f32,
    pub actual_value: f32,
}

impl fmt::Display for SliderDisplay {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}({}) = {:.3}",
            self.label, self.axis, self.actual_value
        )
    }
}

pub enum SliderEvent {
    UpdateSlider {
        id: WaveformId,
        slider: String,
        value: f32,
    },
    SetInitialValues(HashMap<(WaveformId, String), f32>),
}

fn make_texture<'a>(
    font: &Font<'a, 'static>,
    color: Color,
    texture_creator: &'a TextureCreator<WindowContext>,
    s: &str,
) -> sdl2::render::Texture<'a> {
    let surface = font
        .render(s)
        .blended(color)
        .map_err(|e| e.to_string())
        .unwrap();
    let texture = texture_creator
        .create_texture_from_surface(&surface)
        .map_err(|e| e.to_string())
        .unwrap();
    return texture;
}

const INACTIVE_COLOR: Color = Color::RGB(0x00, 0xFF, 0xFF);
const ACTIVE_COLOR: Color = Color::RGB(0x00, 0xFF, 0x00);
const EDIT_COLOR: Color = Color::RGB(0xFF, 0xFF, 0xFF);
const ERROR_COLOR: Color = Color::RGB(0xFF, 0x00, 0x00);

pub struct Renderer {
    pub video_subsystem: sdl2::VideoSubsystem,
    canvas: sdl2::render::Canvas<sdl2::video::Window>,

    tempo: u32,
    beats_per_measure: u32,

    pub width: u32,
    pub height: u32,
    prompt_width: u32,
    line_height: u32,
    nav_width: u32,

    last_message: String,
    samples: Vec<f32>,
    spectrum: Vec<Complex<f32>>,
}

pub struct Metrics {
    pub tracker_load: Metric<f32>,
    pub allocations_per_sample: Metric<f32>,
}

const FONT_PATH: &'static str = "/Library/Fonts/Arial Unicode.ttf";

impl Renderer {
    pub fn new(
        sdl_context: &Sdl,
        ttf_context: &Sdl2TtfContext,
        tempo: u32,
        beats_per_measure: u32,
    ) -> Renderer {
        let video_subsystem = sdl_context.video().unwrap();
        let display_mode = video_subsystem
            .current_display_mode(0)
            .map_err(|e| e.to_string())
            .unwrap();
        let width = 1500.min(display_mode.w) as u32;
        let height = 1000.min(display_mode.h - 64 /* menu bar, etc. */) as u32;
        //        let ttf_context = sdl2::ttf::init().map_err(|e| e.to_string()).unwrap();
        let window = video_subsystem
            .window("Tuun", width, height)
            .position_centered()
            .build()
            .map_err(|e| e.to_string())
            .unwrap();
        let canvas = window
            .into_canvas()
            .build()
            .map_err(|e| e.to_string())
            .unwrap();
        let texture_creator = canvas.texture_creator();
        let font = ttf_context.load_font(FONT_PATH, 48).unwrap();

        let prompt_texture = make_texture(&font, INACTIVE_COLOR, &texture_creator, " ▸ ");
        let TextureQuery {
            width: prompt_width,
            height: line_height,
            ..
        } = prompt_texture.query();
        let number_texture = make_texture(&font, INACTIVE_COLOR, &texture_creator, "① ");
        let TextureQuery {
            width: number_width,
            ..
        } = number_texture.query();
        let nav_width = prompt_width + number_width;

        Renderer {
            video_subsystem,
            canvas,
            tempo,
            beats_per_measure,
            width,
            height,
            prompt_width,
            line_height,
            nav_width,
            last_message: String::new(),
            samples: Vec::new(),
            spectrum: Vec::new(),
        }
    }

    pub fn render(
        &mut self,
        ttf_context: &Sdl2TtfContext,
        programs: &[Program],
        status: &tracker::Status<WaveformId, MarkId>,
        mode: &Mode,
        metrics: &mut Metrics,
    ) {
        // TODO so much clean-up
        let now = Instant::now();
        let mut current_beat = 0;
        for mark in status.marks.iter() {
            if let WaveformId::Beats(_) = mark.waveform_id {
                // XXX sometimes this doesn't match anything?
                if mark.start <= status.buffer_start
                    && mark.start + mark.duration > status.buffer_start
                    && let MarkId::UserDefined(beat) = mark.mark_id
                {
                    current_beat = beat;
                }
            }
        }
        if current_beat == 0 {
            println!(
                "No current beat found in marks at time {:?}: {:?}",
                status.buffer_start, status.marks
            );
        }
        let texture_creator = self.canvas.texture_creator();
        let font = ttf_context.load_font(FONT_PATH, 48).unwrap();
        let circle_font = ttf_context.load_font(FONT_PATH, 108).unwrap();

        self.canvas.set_draw_color(Color::RGB(0x00, 0x00, 0x00));
        self.canvas.clear();

        let mut y = 10;
        for (index, program) in programs.iter().enumerate() {
            let color = match (&mode, is_pending_program(&status, now, program.id)) {
                (_, true) => ACTIVE_COLOR,
                (
                    Mode::Edit {
                        active_program_index,
                        ..
                    },
                    _,
                ) if index == *active_program_index => EDIT_COLOR,
                _ => INACTIVE_COLOR,
            };
            if !is_pending_program(&status, now, program.id) || current_beat % 2 == 1 {
                let number = char::from_u32(0x31 + index as u32).unwrap().to_string();
                let number_texture = make_texture(&font, color, &texture_creator, &number);
                let TextureQuery {
                    width: number_width,
                    ..
                } = number_texture.query();
                self.canvas
                    .copy(
                        &number_texture,
                        None,
                        Some(sdl2::rect::Rect::new(
                            self.prompt_width as i32,
                            y,
                            number_width,
                            self.line_height,
                        )),
                    )
                    .unwrap();
            }
            let circle = char::from_u32(0x25EF).unwrap().to_string();
            let circle_texture =
                make_texture(&circle_font, ACTIVE_COLOR, &texture_creator, &circle);
            let TextureQuery {
                width: circle_width,
                height: circle_height,
                ..
            } = circle_texture.query();
            if is_active_program(status, now, program.id) {
                self.canvas
                    .copy(
                        &circle_texture,
                        None,
                        Some(sdl2::rect::Rect::new(
                            self.prompt_width as i32 - 20,
                            y - 38,
                            circle_width,
                            circle_height,
                        )),
                    )
                    .unwrap();
            }

            match *mode {
                Mode::Edit {
                    active_program_index,
                    cursor_position,
                    ref errors,
                    ..
                } => {
                    if active_program_index != index && !program.text.is_empty() {
                        let text_texture = make_texture(
                            &font,
                            INACTIVE_COLOR,
                            &texture_creator,
                            program.text.as_str(),
                        );
                        let TextureQuery {
                            width: text_width,
                            height: text_height,
                            ..
                        } = text_texture.query();
                        self.canvas
                            .copy(
                                &text_texture,
                                None,
                                Some(sdl2::rect::Rect::new(
                                    self.nav_width as i32,
                                    y,
                                    text_width,
                                    text_height,
                                )),
                            )
                            .unwrap();
                    } else if active_program_index == index {
                        // Loop over each character in program and check to see if it's in any of the error
                        // ranges
                        let mut x = self.nav_width as i32;
                        for (j, c) in program.text.chars().enumerate() {
                            let color = if errors.iter().any(|e| match e.range() {
                                Some(range) if range.contains(&j) => true,
                                _ => false,
                            }) {
                                ERROR_COLOR
                            } else {
                                EDIT_COLOR
                            };
                            if cursor_position == j {
                                self.draw_cursor(ttf_context, color, x, y, &texture_creator);
                            }
                            let char_texture =
                                make_texture(&font, color, &texture_creator, &c.to_string());
                            let TextureQuery {
                                width: char_width,
                                height: char_height,
                                ..
                            } = char_texture.query();
                            self.canvas
                                .copy(
                                    &char_texture,
                                    None,
                                    Some(sdl2::rect::Rect::new(x, y, char_width, char_height)),
                                )
                                .unwrap();
                            x += char_width as i32;
                        }
                        if cursor_position == program.text.len() {
                            // Draw the cursor at the end of the line
                            let color = if !errors.is_empty() {
                                ERROR_COLOR
                            } else {
                                EDIT_COLOR
                            };
                            self.draw_cursor(ttf_context, color, x, y, &texture_creator);
                        }
                    }
                }
                Mode::Select {
                    active_program_index,
                    ..
                }
                | Mode::MoveSliders {
                    active_program_index,
                    ..
                } => {
                    if active_program_index == index {
                        let color = match mode {
                            Mode::MoveSliders { .. } => ACTIVE_COLOR,
                            _ => INACTIVE_COLOR, // Select
                        };
                        let prompt_texture = make_texture(&font, color, &texture_creator, " ▸ ");
                        self.canvas
                            .copy(
                                &prompt_texture,
                                None,
                                Some(sdl2::rect::Rect::new(
                                    0,
                                    y,
                                    self.prompt_width,
                                    self.line_height,
                                )),
                            )
                            .unwrap();
                    }

                    if !program.text.is_empty() {
                        let text_texture = make_texture(
                            &font,
                            INACTIVE_COLOR,
                            &texture_creator,
                            program.text.as_str(),
                        );
                        let TextureQuery {
                            width: text_width,
                            height: text_height,
                            ..
                        } = text_texture.query();
                        self.canvas
                            .copy(
                                &text_texture,
                                None,
                                Some(sdl2::rect::Rect::new(
                                    self.nav_width as i32,
                                    y,
                                    text_width,
                                    text_height,
                                )),
                            )
                            .unwrap();
                    }
                }
                Mode::Play { .. }
                | Mode::LoadContext { .. }
                | Mode::LoadPrograms { .. }
                | Mode::Exit => (),
            }
            y += self.line_height as i32;
        }

        // Draw the sliders
        let slider_color = if let Mode::MoveSliders { .. } = mode {
            ACTIVE_COLOR
        } else {
            INACTIVE_COLOR
        };
        self.canvas.set_draw_color(slider_color);
        let slider_display: Vec<SliderDisplay> = match &mode {
            Mode::MoveSliders {
                active_program_index,
            }
            | Mode::Select {
                active_program_index,
                ..
            }
            | Mode::Edit {
                active_program_index,
                ..
            } => programs[*active_program_index].sliders.slider_display(),
            Mode::Play { .. }
            | Mode::LoadContext { .. }
            | Mode::LoadPrograms { .. }
            | Mode::Exit => Vec::new(),
        };
        // First slider: horizontal indicator at top
        if let Some(s) = slider_display.get(0) {
            self.canvas
                .fill_rect(sdl2::rect::Rect::new(
                    (self.width as f32 * s.normalized_value) as i32 - 3,
                    0,
                    6,
                    16,
                ))
                .unwrap();
        }
        // Second slider: vertical indicator at left
        if let Some(s) = slider_display.get(1) {
            self.canvas
                .fill_rect(sdl2::rect::Rect::new(
                    0,
                    self.height as i32 - (self.height as f32 * s.normalized_value) as i32 - 3,
                    16,
                    6,
                ))
                .unwrap();
        }

        // Save the new buffer if present
        if let Some(buffer) = &status.buffer {
            self.samples = buffer.clone();
            let mut planner = RealFftPlanner::<f32>::new();
            let fft = planner.plan_fft_forward(self.samples.len());
            let mut input = fft.make_input_vec();
            for (i, f) in self.samples.iter().enumerate() {
                input[i] = *f;
            }
            self.spectrum = fft.make_output_vec();
            if let Err(e) = fft.process(&mut input, &mut self.spectrum) {
                println!("Error processing FFT: {}", e);
            }
        }
        // Draw the most recent sample buffer
        if !self.samples.is_empty() {
            // Draw the waveform
            let x_scale = self.width as f32 / (self.samples.len() + 1) as f32;
            let waveform_height = self.height * 3 / 5;
            if self.samples.len() > 0 {
                self.canvas.set_draw_color(Color::RGB(0x00, 0xFF, 0x00));
                let mut last_y = (waveform_height as f32
                    - ((self.samples[0] + 1.0) * (waveform_height as f32 / 2.4)))
                    as i32;
                for (i, f) in self.samples.iter().enumerate() {
                    let x = (i as f32 * x_scale) as i32;
                    let y = (waveform_height as f32 - ((f + 1.0) * (waveform_height as f32 / 2.4)))
                        as i32;
                    if f.abs() <= 0.95 {
                        self.canvas.set_draw_color(Color::RGB(0x00, 0xFF, 0x00));
                    } else if f.abs() <= 1.0 {
                        self.canvas.set_draw_color(Color::RGB(0xFF, 0xDE, 0x21));
                    } else {
                        self.canvas.set_draw_color(Color::RGB(0xFF, 0x00, 0x00));
                    }
                    self.canvas
                        .draw_line((x, last_y as i32), (x + x_scale as i32, y))
                        .unwrap();
                    last_y = y;
                }
            }

            // Draw the spectrum
            if !self.spectrum.is_empty() {
                let spectrum_height = self.height - waveform_height;
                fn as_scalar(c: &Complex<f32>) -> f32 {
                    c.abs().log10()
                }
                let y_scale_max = -self
                    .spectrum
                    .iter()
                    .map(as_scalar)
                    .reduce(f32::max)
                    .unwrap();
                self.canvas.set_draw_color(Color::RGB(0xFF, 0x00, 0x00));
                let mut last_y = (waveform_height + 300) as i32; // i.e., 0
                for (i, c) in self.spectrum.iter().enumerate() {
                    let x = ((i * 10) as f32 * x_scale) as i32;
                    let y = (as_scalar(c) / y_scale_max * spectrum_height as f32
                        + (waveform_height + 300) as f32) as i32;
                    self.canvas.draw_line((x, last_y), (x + 9, y)).unwrap();
                    last_y = y;
                }
            }
        }

        // Draw the marks
        let mark_colors = vec![
            Color::RGB(0xE1, 0x77, 0xF9),
            Color::RGB(0xAD, 0xD8, 0xFF),
            Color::RGB(0xAC, 0xD8, 0xAA),
            Color::RGB(0xFF, 0xAD, 0xC3),
            Color::RGB(0xFF, 0xDC, 0x85),
        ];
        let marks_row_height = self.height as f32 / 4.0 / mark_colors.len() as f32;
        let marks_y_padding = 6.0;
        for even in [false, true] {
            let mut marks_start = Instant::now() + Duration::from_secs(1000); // TODO something better?
            let mut marks_duration = Duration::from_secs(0);
            let marks_width = self.width as f32 / 4.0;
            let marks_x_offset = if even {
                self.width as f32 / 2.0
            } else {
                self.width as f32 / 4.0
            };
            // Find the start of the first Beats waveform that's odd/even
            for mark in status.marks.iter() {
                if mark.waveform_id == WaveformId::Beats(even) {
                    if mark.mark_id == MarkId::TopLevel {
                        marks_start = marks_start.min(mark.start);
                        marks_duration = mark.duration;
                    }
                }
            }
            for mark in status.marks.iter().rev() {
                // Reverse so we draw earlier ones last
                if mark.waveform_id.is_beats() {
                    continue; // Skip beats
                }
                let color_idx = match mark.mark_id.color_index() {
                    None => continue, // Skip if not user-defined
                    Some(idx) => idx,
                };
                if mark.start > marks_start + marks_duration
                    || mark.start + mark.duration < marks_start
                {
                    continue; // Skip marks that don't start during the Beats waveform
                }
                if mark.start < now && mark.start + mark.duration >= now {
                    self.canvas
                        .set_draw_color(mark_colors[color_idx % mark_colors.len()]);
                } else {
                    let mut color = mark_colors[color_idx % mark_colors.len()];
                    color.r = color.r / 2;
                    color.g = color.g / 2;
                    color.b = color.b / 2;
                    self.canvas.set_draw_color(color);
                }
                let x = (mark.start - marks_start).as_secs_f32() / marks_duration.as_secs_f32()
                    * marks_width
                    + marks_x_offset;
                let y = color_idx as f32 * marks_row_height + 2.0 * self.height as f32 / 3.0;
                let width =
                    mark.duration.as_secs_f32() / marks_duration.as_secs_f32() * marks_width;
                let height = marks_row_height - marks_y_padding;
                self.canvas
                    .fill_rect(sdl2::rect::Rect::new(
                        x as i32,
                        y as i32,
                        width as u32,
                        height as u32,
                    ))
                    .unwrap();
            }
        }

        // Draw the message
        let mut message = match mode {
            Mode::Edit { message, .. } => message,
            Mode::Select { message, .. } => message,
            Mode::MoveSliders { .. } => {
                if slider_display.is_empty() {
                    &"No sliders configured".to_string()
                } else {
                    &slider_display
                        .iter()
                        .map(|s| format!("{}", s))
                        .collect::<Vec<_>>()
                        .join(", ")
                }
            }
            Mode::Play { .. }
            | Mode::LoadContext { .. }
            | Mode::LoadPrograms { .. }
            | Mode::Exit => "",
        };

        if !message.is_empty() && message != self.last_message {
            println!("{}", message);
        }
        self.last_message = message.to_string();
        if !message.is_empty() {
            if message.len() > 45 {
                message = &message[..45];
            }
            let message_texture = make_texture(&font, INACTIVE_COLOR, &texture_creator, message);
            let TextureQuery {
                width: message_width,
                height: message_height,
                ..
            } = message_texture.query();
            self.canvas
                .copy(
                    &message_texture,
                    None,
                    Some(sdl2::rect::Rect::new(
                        20,
                        self.height as i32 - message_height as i32 - 20,
                        message_width,
                        message_height,
                    )),
                )
                .unwrap();
        }

        // Draw the current beat
        let beat_font = ttf_context.load_font(FONT_PATH, 64).unwrap();
        let beat_texture = make_texture(
            &beat_font,
            ACTIVE_COLOR,
            &texture_creator,
            format!(
                "{} / {} ({} bpm)",
                current_beat, self.beats_per_measure, self.tempo
            )
            .as_str(),
        );
        let TextureQuery {
            width: beat_width,
            height: beat_height,
            ..
        } = beat_texture.query();
        self.canvas
            .copy(
                &beat_texture,
                None,
                Some(sdl2::rect::Rect::new(
                    self.width as i32 - beat_width as i32 - 20,
                    self.height as i32 - beat_height as i32 - 20,
                    beat_width,
                    beat_height,
                )),
            )
            .unwrap();

        // Draw some internal metrics
        let metrics_height = self.height / 2 - beat_height as u32 - 40;
        let metrics_width = 200;
        let x = (self.width - metrics_width - 20) as i32;
        let y = self.height as i32 / 2;
        self.draw_metric(
            ttf_context,
            &texture_creator,
            x,
            y,
            metrics_width,
            metrics_height,
            &mut metrics.tracker_load,
        );

        let x = (self.width - 2 * metrics_width - 20) as i32;
        let y = self.height as i32 / 2;
        self.draw_metric(
            ttf_context,
            &texture_creator,
            x,
            y,
            metrics_width,
            metrics_height,
            &mut metrics.allocations_per_sample,
        );

        self.canvas.present();
    }

    fn draw_cursor(
        self: &mut Renderer,
        ttf_context: &Sdl2TtfContext,
        color: Color,
        x: i32,
        y: i32,
        texture_creator: &TextureCreator<WindowContext>,
    ) {
        let font = ttf_context.load_font(FONT_PATH, 48).unwrap();
        let cursor_texture = make_texture(&font, color, &texture_creator, "‸");
        let TextureQuery {
            width: cursor_width,
            height: cursor_height,
            ..
        } = cursor_texture.query();
        self.canvas
            .copy(
                &cursor_texture,
                None,
                Some(sdl2::rect::Rect::new(
                    x - 7,
                    y + 5,
                    cursor_width,
                    cursor_height,
                )),
            )
            .unwrap();
    }

    fn draw_metric(
        self: &mut Renderer,
        ttf_context: &Sdl2TtfContext,
        texture_creator: &TextureCreator<WindowContext>,
        x: i32,
        y: i32,
        _width: u32,
        height: u32,
        metric: &mut Metric<f32>,
    ) {
        let font = ttf_context.load_font(FONT_PATH, 48).unwrap();
        let points: Vec<f32> = metric.iter().collect();
        if points.len() > 0 {
            let last_value_texture = make_texture(
                &font,
                Color::RGB(0x00, 0xFF, 0x00),
                texture_creator,
                format!("{:.2}", points.last().unwrap()).as_str(),
            );
            let TextureQuery {
                width: last_value_width,
                height: last_value_height,
                ..
            } = last_value_texture.query();
            self.canvas
                .copy(
                    &last_value_texture,
                    None,
                    Some(sdl2::rect::Rect::new(
                        x,
                        y,
                        last_value_width,
                        last_value_height,
                    )),
                )
                .unwrap();

            let mut last_y = y + height as i32 - (points[0] * height as f32) as i32;
            for (i, &point) in points.iter().enumerate() {
                if i == points.len() - 1 && point == 0.0 {
                    continue; // Skip the last point if it's zero
                }
                let value = y + height as i32 - (point * height as f32) as i32;
                // TODO these thresholds don't make sense for arbitrary metrics
                if point < 0.7 {
                    self.canvas.set_draw_color(Color::RGB(0x00, 0xFF, 0x00));
                } else if point < 0.9 {
                    self.canvas.set_draw_color(Color::RGB(0xFF, 0xDE, 0x21));
                } else {
                    self.canvas.set_draw_color(Color::RGB(0xFF, 0x00, 0x00));
                }
                self.canvas
                    .draw_line(
                        (x + (200 / points.len() as i32) * i as i32, last_y),
                        (x + (200 / points.len() as i32) * ((i + 1) as i32), value),
                    )
                    .unwrap();
                last_y = value;
            }
        }
    }
}

pub fn duration_from_beats(tempo: u32, beats: u64) -> Duration {
    Duration::from_secs_f32(beats as f32 * 60.0 / tempo as f32)
}

pub fn beats_waveform(
    tempo: u32,
    beats_per_measure: u32,
    context: &Vec<(String, parser::Expr<MarkId>)>,
) -> waveform::Waveform<MarkId> {
    let seconds_per_beat = duration_from_beats(tempo, 1);
    let mut ws = Vec::new();
    for i in 0..beats_per_measure {
        ws.push(format!(
            "0 | fin(time - {}) | seq(time - {}) | mark({})",
            seconds_per_beat.as_secs_f32(),
            seconds_per_beat.as_secs_f32(),
            i + 1
        ));
    }
    let expr = match parser::parse_program(&format!("<[{}]>", ws.join(", "))) {
        Ok(expr) => expr,
        Err(errors) => panic!("Error parsing beats waveform: {:?}", errors),
    };
    match parser::evaluate(&context, expr) {
        Ok(parser::Expr::Seq { waveform, .. }) => match *waveform {
            parser::Expr::Waveform(waveform) => waveform::Waveform::<MarkId>::Marked {
                id: MarkId::TopLevel,
                waveform: Box::new(optimizer::optimize(waveform)),
            },
            expr => panic!("Error creating beats waveform with seq, got {}", expr),
        },
        Ok(expr) => panic!("Error creating beats waveform, got {}", expr),
        Err(errors) => panic!("Error evaluating beats waveform: {:?}", errors),
    }
}

pub fn is_pending_program(
    status: &tracker::Status<WaveformId, MarkId>,
    now: Instant,
    program_id: ProgramId,
) -> bool {
    status.marks.iter().any(|w| {
        w.waveform_id == WaveformId::Program(program_id)
            && w.mark_id == MarkId::TopLevel
            && w.start > now
    })
}

pub fn is_active_program(
    status: &tracker::Status<WaveformId, MarkId>,
    now: Instant,
    program_id: ProgramId,
) -> bool {
    status.marks.iter().any(|w| {
        w.waveform_id == WaveformId::Program(program_id)
            && w.mark_id == MarkId::TopLevel
            && w.start <= now
    })
}

pub fn play_waveform_helper(
    context: &Vec<(String, parser::Expr<MarkId>)>,
    active_program_index: usize,
    cursor_position: usize,
    program: &Program,
) -> WaveformOrMode {
    match parser::parse_program(&program.text) {
        Ok(expr) => {
            println!("Parser returned: {}", &expr);
            let expr = slider::prepend_slider_bindings(
                &program.sliders.configs,
                &program.sliders.normalized_values,
                MarkId::Slider,
                expr,
            );
            match parser::evaluate(context, expr) {
                Ok(expr) => {
                    println!("parser::evaluate returned: {}", &expr);
                    use parser::Expr;
                    let mut waveform = match expr {
                        Expr::Waveform(waveform) => waveform,
                        Expr::Seq { waveform, .. } => match *waveform {
                            Expr::Waveform(waveform) => waveform,
                            _ => panic!("Got non-Waveform in seq after evaluate"),
                        },
                        _ => {
                            println!("Expression is not a waveform, cannot play: {:#?}", expr);
                            return WaveformOrMode::Mode(Mode::Edit {
                                active_program_index,
                                cursor_position,
                                errors: vec![parser::Error::new(
                                    "Expression is not a waveform".to_string(),
                                )],
                                message: format!("Not a waveform: {}", expr),
                            });
                        }
                    };
                    waveform = optimizer::optimize(waveform);
                    println!("optimizer::optimize returned: {}", &waveform);
                    return WaveformOrMode::Waveform(waveform);
                }
                Err(error) => {
                    // If there are errors, we stay in edit mode
                    println!("Errors while evaluating input: {:?}", error);
                    let message = format!("Error: {}", error.to_string());
                    return WaveformOrMode::Mode(Mode::Edit {
                        active_program_index,
                        cursor_position,
                        errors: vec![error],
                        message: message,
                    });
                }
            }
        }
        Err(errors) => {
            // If there are errors, we stay in edit mode
            println!("Errors while parsing input: {:?}", errors);
            let message = format!("Error: {}", errors[0].to_string());
            return WaveformOrMode::Mode(Mode::Edit {
                active_program_index,
                cursor_position,
                errors,
                message,
            });
        }
    }
}
