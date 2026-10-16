use std::time::{Duration, Instant};

use sdl2::Sdl;
use sdl2::pixels::Color;
use sdl2::render::{TextureCreator, TextureQuery};
use sdl2::ttf::{Font, Sdl2TtfContext};
use sdl2::video::WindowContext;

use realfft::RealFftPlanner;
use realfft::num_complex::{Complex, ComplexFloat};

use crate::actions::{AppState, Mode};
use crate::ids::{MarkId, WaveformId, WaveformSelector};
use crate::launchkey;
use crate::metric::Metric;
use crate::programs::{PROGRAMS_PER_BANK, SliderDisplay};
use crate::recorder::{Finish, Phase};
use crate::tracker;

fn make_texture<'t>(
    font: &Font<'_, 'static>,
    color: Color,
    texture_creator: &'t TextureCreator<WindowContext>,
    s: &str,
) -> sdl2::render::Texture<'t> {
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
const ERROR_COLOR: Color = Color::RGB(0xFF, 0xDE, 0x21);
const RECORD_COLOR: Color = Color::RGB(0xFF, 0x00, 0x00);

pub struct Renderer<'ttf> {
    pub video_subsystem: sdl2::VideoSubsystem,
    canvas: sdl2::render::Canvas<sdl2::video::Window>,

    font: Font<'ttf, 'static>,
    circle_font: Font<'ttf, 'static>,
    beat_font: Font<'ttf, 'static>,

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
    /// The time spent rendering a frame, as a share of [`FRAME_BUDGET`].
    ///
    /// Values above 1.0 mean the frame took longer than its budget.
    pub renderer_load: Metric<f32>,
}

/// The time between frames at 60 Hz.
pub const FRAME_BUDGET: Duration = Duration::from_nanos(1_000_000_000 / 60);

const FONT_PATH: &str = "/Library/Fonts/Arial Unicode.ttf";

impl<'ttf> Renderer<'ttf> {
    pub fn new(
        sdl_context: &Sdl,
        ttf_context: &'ttf Sdl2TtfContext,
        tempo: u32,
        beats_per_measure: u32,
        sample_rate: u32,
    ) -> Renderer<'ttf> {
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
        let circle_font = ttf_context.load_font(FONT_PATH, 108).unwrap();
        let beat_font = ttf_context.load_font(FONT_PATH, 64).unwrap();

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
            font,
            circle_font,
            beat_font,
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
        // Alpha for things that fade over the beat: opaque at its start.
        let beat_fade = u8::MAX
            - (beat_phase(now, current_beat_start, current_beat_duration) * u8::MAX as f32) as u8;
        let texture_creator = self.canvas.texture_creator();
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
            let sounding = status.has_active_mark(
                now,
                &WaveformSelector::ProgramVoices(index),
                &MarkId::TopLevel,
            );
            let take_phase = match mode {
                Mode::Record { take, .. } if index == active_program_index => Some(take.phase()),
                _ => None,
            };
            // The take's red wins over the sounding green for the active slot.
            let (color, alpha) = match (take_phase, mode, sounding) {
                (Some(Phase::Recording | Phase::Finishing(_)), _, _) => (RECORD_COLOR, beat_fade),
                (_, _, true) => (ACTIVE_COLOR, beat_fade),
                (_, Mode::Edit { .. }, _) if index == active_program_index => (EDIT_COLOR, u8::MAX),
                _ => (program_color, u8::MAX),
            };
            let number = char::from_u32(0x31 + i as u32).unwrap().to_string();
            let mut number_texture = make_texture(&self.font, color, &texture_creator, &number);
            number_texture.set_alpha_mod(alpha);
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
            // The Finishing (play) circle stands in for the pending mark the
            // write will queue.
            let circle = match take_phase {
                Some(Phase::Armed) => Some((RECORD_COLOR, u8::MAX)),
                Some(Phase::Finishing(Finish::Play)) => Some((ACTIVE_COLOR, u8::MAX)),
                Some(Phase::Recording | Phase::Finishing(Finish::Silent)) => None,
                None => status
                    .has_pending_mark(
                        now,
                        &WaveformSelector::ProgramVoices(index),
                        &MarkId::TopLevel,
                    )
                    .then_some((ACTIVE_COLOR, u8::MAX)),
            };
            if let Some((circle_color, circle_alpha)) = circle {
                self.draw_circle(&texture_creator, circle_color, circle_alpha, y);
            }

            match *mode {
                Mode::Edit {
                    cursor_position,
                    ref errors,
                    ..
                } => {
                    if active_program_index != index && !program.is_empty() {
                        let text_texture = make_texture(
                            &self.font,
                            program_color,
                            &texture_creator,
                            program.text(),
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
                        // Draw the text in runs of one color, split where it
                        // enters or leaves an error range. `char_indices`
                        // yields byte offsets, matching both `cursor_position`
                        // and the errors' byte ranges.
                        let text = program.text();
                        let color_at = |j: usize| {
                            if errors.iter().any(
                                |e| matches!(&e.program_range, Some(range) if range.contains(&j)),
                            ) {
                                ERROR_COLOR
                            } else {
                                EDIT_COLOR
                            }
                        };
                        // The byte offset where each run starts, and its color.
                        let mut runs: Vec<(usize, Color)> = Vec::new();
                        for (j, _) in text.char_indices() {
                            let color = color_at(j);
                            if runs.last().is_none_or(|&(_, last)| last != color) {
                                runs.push((j, color));
                            }
                        }
                        for (k, &(start, color)) in runs.iter().enumerate() {
                            let end = runs.get(k + 1).map_or(text.len(), |&(next, _)| next);
                            // Measure the whole prefix, not the earlier runs, so
                            // that runs and the cursor keep the font's kerning.
                            let x = self.nav_width as i32 + self.text_width(&text[..start]);
                            let run_texture = make_texture(
                                &self.font,
                                color,
                                &texture_creator,
                                &text[start..end],
                            );
                            let TextureQuery {
                                width: run_width,
                                height: run_height,
                                ..
                            } = run_texture.query();
                            self.canvas
                                .copy(
                                    &run_texture,
                                    None,
                                    Some(sdl2::rect::Rect::new(x, y, run_width, run_height)),
                                )
                                .unwrap();
                        }
                        if let Some(before_cursor) = text.get(..cursor_position) {
                            let color = if cursor_position < text.len() {
                                color_at(cursor_position)
                            } else if !errors.is_empty() {
                                ERROR_COLOR
                            } else {
                                EDIT_COLOR
                            };
                            let x = self.nav_width as i32 + self.text_width(before_cursor);
                            self.draw_cursor(color, x, y, &texture_creator);
                        }
                    }
                }
                Mode::Select | Mode::MoveSliders | Mode::Keys | Mode::Record { .. } => {
                    if active_program_index == index && !matches!(mode, Mode::Record { .. }) {
                        let color = match mode {
                            Mode::MoveSliders => ACTIVE_COLOR,
                            _ => INACTIVE_COLOR, // Select
                        };
                        let prompt_texture =
                            make_texture(&self.font, color, &texture_creator, " ▸ ");
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
                        let text_texture = make_texture(
                            &self.font,
                            program_color,
                            &texture_creator,
                            program.text(),
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
        let beat_height = self.beat_font.size_of("0").unwrap().1 as f32;
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
                if mark.waveform_id == WaveformId::Beats(even)
                    && mark.mark_id == MarkId::TopLevel
                    && let Some(duration) = mark.duration
                {
                    // We want to find the oldest even/odd beats in the marks.
                    if mark.start < marks_start {
                        marks_start = mark.start;
                        // Make the duration shorter by one sample to avoid spurious overlaps due to rounding.
                        marks_duration = duration - sample_duration;
                    }
                }
            }
            for mark in status.marks.iter().rev() {
                // Reverse so we draw earlier ones last
                let program_index = match &mark.waveform_id {
                    WaveformId::Program(index) => *index,
                    WaveformId::Step { program, .. } => *program,
                    _ => continue,
                };
                // Only draw Mark(1), and only when its width is known.
                if mark.mark_id != MarkId::UserDefined(1) {
                    continue;
                }
                let Some(duration) = mark.duration else {
                    continue;
                };
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
                if now < mark.start || now >= mark.start + duration {
                    program_color.r /= 2;
                    program_color.g /= 2;
                    program_color.b /= 2;
                }
                self.canvas.set_draw_color(program_color);
                let x = (mark.start - marks_start).as_secs_f32() / marks_duration.as_secs_f32()
                    * marks_width
                    + marks_x_offset;
                let y = row as f32 * marks_row_height + programs_bottom;
                let width = duration.as_secs_f32() / marks_duration.as_secs_f32() * marks_width;
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
            let message_texture =
                make_texture(&self.font, INACTIVE_COLOR, &texture_creator, message);
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
            &self.beat_font,
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
            &texture_creator,
            x,
            y,
            metrics_width,
            metrics_height,
            &mut metrics.allocations_per_sample,
        );

        let x = (self.width - 3 * metrics_width - 20) as i32;
        let y = self.height as i32 / 2;
        self.draw_metric(
            &texture_creator,
            x,
            y,
            metrics_width,
            metrics_height,
            &mut metrics.renderer_load,
        );

        self.canvas.present();

        // This frame's load is drawn with the next frame.
        metrics
            .renderer_load
            .set(now.elapsed().div_duration_f32(FRAME_BUDGET));
    }

    /// Draws the large circle around the slot number on the row at `y`.
    fn draw_circle(
        &mut self,
        texture_creator: &TextureCreator<WindowContext>,
        color: Color,
        alpha: u8,
        y: i32,
    ) {
        let circle = char::from_u32(0x25EF).unwrap().to_string();
        let mut circle_texture = make_texture(&self.circle_font, color, texture_creator, &circle);
        circle_texture.set_alpha_mod(alpha);
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

    /// Returns the width of `s` in the text font, in pixels.
    fn text_width(&self, s: &str) -> i32 {
        // Empty text has no width; skip the call into SDL_ttf.
        if s.is_empty() {
            return 0;
        }
        self.font.size_of(s).unwrap().0 as i32
    }

    fn draw_cursor(
        &mut self,
        color: Color,
        x: i32,
        y: i32,
        texture_creator: &TextureCreator<WindowContext>,
    ) {
        let cursor_texture = make_texture(&self.font, color, texture_creator, "‸");
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
        &mut self,
        texture_creator: &TextureCreator<WindowContext>,
        x: i32,
        y: i32,
        _width: u32,
        height: u32,
        metric: &mut Metric<f32>,
    ) {
        let default_color = Color::RGBA(0, 255, 0, 128);
        let points: Vec<f32> = metric.iter().collect();
        if !points.is_empty() {
            let last_value_texture = make_texture(
                &self.font,
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

/// Returns how far `now` is through the beat starting at `beat_start`, from 0
/// at its start to 1 at its end.
///
/// # Example
///
/// ```
/// use std::time::{Duration, Instant};
/// use tuun::renderer::beat_phase;
///
/// let start = Instant::now();
/// let beat = Duration::from_millis(500);
/// assert_eq!(beat_phase(start + Duration::from_millis(125), start, beat), 0.25);
/// assert_eq!(beat_phase(start + Duration::from_secs(2), start, beat), 1.0);
/// ```
pub fn beat_phase(now: Instant, beat_start: Instant, beat_duration: Duration) -> f32 {
    now.duration_since(beat_start)
        .div_duration_f32(beat_duration)
        .min(1.0)
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
            if let Some(duration) = mark.duration
                && mark.start <= status.buffer_start
                && mark.start + duration > status.buffer_start
                && let MarkId::UserDefined(beat) = mark.mark_id
            {
                current_beat = beat;
                current_beat_start = mark.start;
                current_beat_duration = duration;
            }
        }
    }
    if current_beat == 0 {
        panic!(
            "No current beat found in marks at time {:?}: {:?}",
            status.buffer_start, status.marks
        );
    }
    (current_beat, current_beat_start, current_beat_duration)
}
