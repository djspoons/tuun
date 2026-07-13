use std::time::{Duration, Instant};

use sdl2::Sdl;
use sdl2::pixels::Color;
use sdl2::render::{TextureCreator, TextureQuery};
use sdl2::ttf::{Font, Sdl2TtfContext};
use sdl2::video::WindowContext;

use realfft::RealFftPlanner;
use realfft::num_complex::{Complex, ComplexFloat};

use crate::actions::{AppState, Mode};
use crate::ids::{MarkId, WaveformId};
use crate::launchkey;
use crate::metric::Metric;
use crate::programs::{PROGRAMS_PER_BANK, SliderDisplay};
use crate::tracker;

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
    texture_creator
        .create_texture_from_surface(&surface)
        .map_err(|e| e.to_string())
        .unwrap()
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
    sample_rate: u32,

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

const FONT_PATH: &str = "/Library/Fonts/Arial Unicode.ttf";

impl Renderer {
    pub fn new(
        sdl_context: &Sdl,
        ttf_context: &Sdl2TtfContext,
        tempo: u32,
        beats_per_measure: u32,
        sample_rate: u32,
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
            sample_rate,
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
        state: &AppState,
        status: &tracker::Status<WaveformId, MarkId>,
        metrics: &mut Metrics,
        encoder_mode: Option<launchkey::EncoderMode>,
    ) {
        // Alias the AppState fields the body uses so the rest of this
        // function (still mostly written against unbound locals) keeps
        // working unchanged.
        let programs = state.programs.programs();
        let mode = &state.mode;
        let active_program_index = state.active_program_index;
        let message = state.message.as_str();

        // TODO so much clean-up
        let now = Instant::now();
        let (current_beat, current_beat_start, current_beat_duration) =
            current_beat_info(now, status);
        let texture_creator = self.canvas.texture_creator();
        let font = ttf_context.load_font(FONT_PATH, 48).unwrap();
        let circle_font = ttf_context.load_font(FONT_PATH, 108).unwrap();

        self.canvas.set_draw_color(Color::RGB(0x00, 0x00, 0x00));
        self.canvas.clear();

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
            if !self.samples.is_empty() {
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
                        .draw_line((x, last_y), (x + x_scale as i32, y))
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

        let mut y = 10;
        let bank_start = active_program_index - (active_program_index % PROGRAMS_PER_BANK);
        for (i, program) in programs[bank_start..bank_start + PROGRAMS_PER_BANK]
            .iter()
            .enumerate()
        {
            let index = bank_start + i;
            let program_color = program
                .color()
                .map(|(r, g, b)| Color::RGB(r, g, b))
                .unwrap_or(INACTIVE_COLOR);
            let color = match (
                &mode,
                status.has_active_mark(now, WaveformId::Program(index), MarkId::TopLevel),
            ) {
                (_, true) => ACTIVE_COLOR,
                (Mode::Edit { .. }, _) if index == active_program_index => EDIT_COLOR,
                _ => program_color,
            };
            let number = char::from_u32(0x31 + i as u32).unwrap().to_string();
            let mut number_texture = make_texture(&font, color, &texture_creator, &number);
            if status.has_active_mark(now, WaveformId::Program(index), MarkId::TopLevel) {
                let intensity = (now
                    .duration_since(current_beat_start)
                    .div_duration_f32(current_beat_duration)
                    * u8::MAX as f32) as u8;
                number_texture.set_alpha_mod(u8::MAX - intensity);
            }
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
            if status.has_pending_mark(now, WaveformId::Program(index), MarkId::TopLevel) {
                let circle = char::from_u32(0x25EF).unwrap().to_string();
                let circle_texture =
                    make_texture(&circle_font, ACTIVE_COLOR, &texture_creator, &circle);
                let TextureQuery {
                    width: circle_width,
                    height: circle_height,
                    ..
                } = circle_texture.query();
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
                    cursor_position,
                    ref errors,
                    ..
                } => {
                    if active_program_index != index && !program.is_empty() {
                        let text_texture =
                            make_texture(&font, program_color, &texture_creator, program.text());
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
                        // ranges. `char_indices` yields byte offsets, matching
                        // both `cursor_position` and the errors' byte ranges.
                        let mut x = self.nav_width as i32;
                        for (j, c) in program.text().char_indices() {
                            let color = if errors.iter().any(
                                |e| matches!(&e.program_range, Some(range) if range.contains(&j)),
                            ) {
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
                        if cursor_position == program.text().len() {
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
                Mode::Select | Mode::MoveSliders | Mode::Keys => {
                    if active_program_index == index {
                        let color = match mode {
                            Mode::MoveSliders => ACTIVE_COLOR,
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

                    if !program.is_empty() {
                        let text_texture =
                            make_texture(&font, program_color, &texture_creator, program.text());
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
            }
            y += self.line_height as i32;
        }

        // Draw the sliders. Only the mouse-driven (no-launchkey) layout
        // gets the top/left edge bars; when the controller is connected
        // the encoder visualization below covers the same data more
        // completely.
        let slider_display: Vec<SliderDisplay> =
            programs[active_program_index].sliders().slider_display();
        if encoder_mode.is_none() {
            let slider_color = if let Mode::MoveSliders = mode {
                ACTIVE_COLOR
            } else {
                INACTIVE_COLOR
            };
            self.canvas.set_draw_color(slider_color);
            // First slider: horizontal indicator at top
            if let Some(s) = slider_display.first() {
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
        }

        // Draw the marks
        let programs_bottom = (20 + PROGRAMS_PER_BANK as i32 * self.line_height as i32) as f32;
        let beat_font = ttf_context.load_font(FONT_PATH, 64).unwrap();
        let beat_height = beat_font.size_of("0").unwrap().1 as f32;
        let marks_bottom = self.height as f32 - beat_height - 10.0;
        let marks_total_height = (marks_bottom - programs_bottom).max(0.0);
        let marks_row_height = marks_total_height / PROGRAMS_PER_BANK as f32;
        let marks_y_padding = 6.0;
        // We "round" some values below by sample_duration as Instant is a lot more precise.
        // For example, sometimes a program will start just before the measure it's supposed
        // to start in, so we move its start time slightly to account for this. Not sure I did
        // that exactly right, but this seems close enough for now.
        let sample_duration = Duration::from_secs_f64(1.0 / self.sample_rate as f64);
        for even in [false, true] {
            // Start with a point far into the future.
            let mut marks_start = Instant::now()
                + Duration::from_secs_f32(
                    (self.tempo as f32 / 60.0) / 4.0 * self.beats_per_measure as f32,
                ); // TODO something better?
            let mut marks_duration = Duration::from_secs(0);
            let marks_width = self.width as f32 / 4.0; // for one measure
            let marks_x_offset = if even {
                5.0 * self.width as f32 / 8.0
            } else {
                3.0 * self.width as f32 / 8.0
            };
            // Find the start of the first Beats waveform that's odd/even
            for mark in status.marks.iter() {
                if mark.waveform_id == WaveformId::Beats(even) && mark.mark_id == MarkId::TopLevel {
                    // We want to find the oldest even/odd beats in the marks.
                    if mark.start < marks_start {
                        marks_start = mark.start;
                        // Make the duration shorter by one sample to avoid spurious overlaps due to rounding.
                        marks_duration = mark.duration - sample_duration;
                    }
                }
            }
            for mark in status.marks.iter().rev() {
                // Reverse so we draw earlier ones last
                let program_index = match &mark.waveform_id {
                    WaveformId::Program(index) => *index,
                    _ => continue,
                };
                // Only draw Mark(1)
                if mark.mark_id != MarkId::UserDefined(1) {
                    continue;
                }
                // Skip marks that don't start during this measure.
                if mark.start < marks_start - sample_duration
                    || mark.start >= marks_start + marks_duration
                {
                    continue;
                }
                let row = program_index % PROGRAMS_PER_BANK;
                let mut program_color = programs[program_index]
                    .color()
                    .map(|(r, g, b)| Color::RGB(r, g, b))
                    .unwrap_or(INACTIVE_COLOR);
                program_color.a = 128;
                if now < mark.start || now >= mark.start + mark.duration {
                    program_color.r /= 2;
                    program_color.g /= 2;
                    program_color.b /= 2;
                }
                self.canvas.set_draw_color(program_color);
                let x = (mark.start - marks_start).as_secs_f32() / marks_duration.as_secs_f32()
                    * marks_width
                    + marks_x_offset;
                let y = row as f32 * marks_row_height + programs_bottom;
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

        // Encoder visualization: fader-style bars. Plugin mode shows the active program's
        // sliders (one bar per slider); Mixer mode shows level_db for every non-empty
        // programs in the bank.
        if let Some(encoder_mode) = encoder_mode {
            let bars_left = self.width as f32 / 8.0;
            let bars_right = 3.0 * self.width as f32 / 8.0;
            let bars_top = programs_bottom + 10.0;
            let bars_bottom = marks_bottom - 10.0;
            let slot_width = (bars_right - bars_left) / PROGRAMS_PER_BANK as f32;
            let bar_height = 6.0_f32;
            let track_top = bars_top + bar_height / 2.0;
            let track_bottom = bars_bottom - bar_height / 2.0;
            let track_span = track_bottom - track_top;
            let bar_width = slot_width * 0.6;

            for i in 0..PROGRAMS_PER_BANK {
                let slot_center = bars_left + (i as f32 + 0.5) * slot_width;
                let (normalized, color) = match encoder_mode {
                    launchkey::EncoderMode::Plugin => {
                        let program = &programs[active_program_index];
                        let Some(&normalized) = program.sliders().normalized_values().get(i) else {
                            continue;
                        };
                        let color = program
                            .color()
                            .map(|(r, g, b)| Color::RGB(r, g, b))
                            .unwrap_or(INACTIVE_COLOR);
                        (normalized, color)
                    }
                    launchkey::EncoderMode::Mixer => {
                        let Some(program) = programs.get(bank_start + i) else {
                            continue;
                        };
                        if program.is_empty() {
                            continue;
                        }
                        // level_db ∈ [-60, +6] → [0, 1].
                        let normalized = ((program.level_db() + 60.0) / 66.0).clamp(0.0, 1.0);
                        let color = program
                            .color()
                            .map(|(r, g, b)| Color::RGB(r, g, b))
                            .unwrap_or(INACTIVE_COLOR);
                        (normalized, color)
                    }
                };

                // Dim vertical track behind the bar.
                let track_color = Color::RGB(color.r / 3, color.g / 3, color.b / 3);
                self.canvas.set_draw_color(track_color);
                self.canvas
                    .draw_line(
                        (slot_center as i32, track_top as i32),
                        (slot_center as i32, track_bottom as i32),
                    )
                    .unwrap();

                // Fader bar.
                self.canvas.set_draw_color(color);
                let bar_center_y = track_bottom - normalized * track_span;
                self.canvas
                    .fill_rect(sdl2::rect::Rect::new(
                        (slot_center - bar_width / 2.0) as i32,
                        (bar_center_y - bar_height / 2.0) as i32,
                        bar_width as u32,
                        bar_height as u32,
                    ))
                    .unwrap();
            }
        }

        // Draw the message. MoveSliders shows the live slider values
        // instead of `state.message` — slider feedback is the whole point
        // of that mode, and replaces any status text while active.
        let dynamic_slider_message: String;
        let message: &str = match mode {
            Mode::MoveSliders => {
                dynamic_slider_message = if slider_display.is_empty() {
                    "No sliders configured".to_string()
                } else {
                    slider_display
                        .iter()
                        .map(|s| format!("{}", s))
                        .collect::<Vec<_>>()
                        .join(", ")
                };
                &dynamic_slider_message
            }
            _ => message,
        };

        if !message.is_empty() && message != self.last_message {
            println!("{}", message);
        }
        self.last_message = message.to_string();
        // The status line shows only the message's first line; any further
        // lines (e.g. error snippets) are context for the terminal echo
        // above.
        let mut message = message.lines().next().unwrap_or("");
        if !message.is_empty() {
            // Truncate to 45 chars, not bytes — a byte slice can panic
            // mid-character on multi-byte input.
            if let Some((limit, _)) = message.char_indices().nth(45) {
                message = &message[..limit];
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
        let metrics_height = self.height / 2 - beat_height - 40;
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
        let cursor_texture = make_texture(&font, color, texture_creator, "‸");
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

    #[allow(clippy::too_many_arguments)]
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
        let default_color = Color::RGBA(0, 255, 0, 128);
        let points: Vec<f32> = metric.iter().collect();
        if !points.is_empty() {
            let last_value_texture = make_texture(
                &font,
                default_color,
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
                    self.canvas.set_draw_color(default_color);
                } else if point < 0.9 {
                    self.canvas.set_draw_color(Color::RGBA(127, 222, 33, 128));
                } else {
                    self.canvas.set_draw_color(Color::RGBA(127, 0, 0, 128));
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

pub fn current_beat_info(
    now: Instant,
    status: &tracker::Status<WaveformId, MarkId>,
) -> (u32, Instant, Duration) {
    let mut current_beat = 0;
    let mut current_beat_start = now;
    let mut current_beat_duration = Duration::from_secs(1);
    for mark in status.marks.iter() {
        if let WaveformId::Beats(_) = mark.waveform_id {
            // XXX sometimes this doesn't match anything?
            if mark.start <= status.buffer_start
                && mark.start + mark.duration > status.buffer_start
                && let MarkId::UserDefined(beat) = mark.mark_id
            {
                current_beat = beat;
                current_beat_start = mark.start;
                current_beat_duration = mark.duration;
            }
        }
    }
    if current_beat == 0 {
        println!(
            "No current beat found in marks at time {:?}: {:?}",
            status.buffer_start, status.marks
        );
    }
    (current_beat, current_beat_start, current_beat_duration)
}
