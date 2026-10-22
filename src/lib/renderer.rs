use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::time::{Duration, Instant};

use sdl2::Sdl;
use sdl2::pixels::Color;
use sdl2::rect::Rect;
use sdl2::render::{BlendMode, TextureCreator, TextureQuery};
use sdl2::ttf::{Font, Sdl2TtfContext};
use sdl2::video::WindowContext;

use realfft::RealFftPlanner;
use realfft::num_complex::{Complex, ComplexFloat};

use crate::actions::{AppState, Mode};
use crate::ids::{ANCHOR_SIXTEENTH, MarkId, WaveformId, WaveformSelector};
use crate::launchkey;
use crate::metric::Metric;
use crate::player::{self, PlayedMarks};
use crate::programs::{PROGRAMS_PER_BANK, Program, ProgramSet, SliderDisplay};
use crate::recorder::{Finish, Phase};
use crate::scope;
use crate::trace_walk::Segment;
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
/// Scan's measure lines: `INACTIVE_COLOR` at about 30% brightness.
const SCAN_MEASURE_COLOR: Color = Color::RGB(0x00, 0x4C, 0x4C);
/// Scan's current columns, by clip level (none, warn, clip): Snapshot's
/// per-sample colors.
const SCAN_COLORS: [Color; 3] = [
    // Slightly lighter than text
    Color::RGB(0x00, 0xCC, 0x00),
    Color::RGB(0xFF, 0xDE, 0x21),
    Color::RGB(0xFF, 0x00, 0x00),
];
/// Scan's previous columns, by clip level: the current colors at about 30%
/// brightness.
const SCAN_PREVIOUS_COLORS: [Color; 3] = [
    Color::RGB(0x00, 0x4C, 0x00),
    Color::RGB(0x4C, 0x42, 0x0A),
    Color::RGB(0x4C, 0x00, 0x00),
];
/// Scan's hatch for dropped samples: dim red, and dimmer still in the previous
/// pass.
const SCAN_GAP_COLOR: Color = Color::RGB(0x59, 0x00, 0x00);
const SCAN_PREVIOUS_GAP_COLOR: Color = Color::RGB(0x1B, 0x00, 0x00);

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
    // Snapshot's samples and their spectrum, refreshed every `snapshot_refresh`.
    samples: Vec<f32>,
    spectrum: Vec<Complex<f32>>,
    snapshot_refresh: Duration,
    next_snapshot_refresh: Instant,
}

pub struct Metrics {
    pub tracker_load: Metric<f32>,
    /// The number of buffers the tracker allocated in one audio callback.
    pub allocations: Metric<f32>,
    /// The time from when a frame was due until it was presented, as a fraction
    /// of [`FRAME_BUDGET`].
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
        snapshot_refresh: Duration,
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
            snapshot_refresh,
            next_snapshot_refresh: Instant::now(),
        }
    }

    /// Draws and presents one frame that was due at `due`.
    ///
    /// Snapshot and the spectrum show `scope`'s latest samples as of their last
    /// refresh, at most one `snapshot_refresh` before `due`.
    pub fn render(
        &mut self,
        state: &AppState,
        status: &tracker::Status<WaveformId, MarkId>,
        scope: &scope::Scope,
        metrics: &mut Metrics,
        encoder_mode: Option<launchkey::EncoderMode>,
        due: Instant,
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
        let (current_beat, current_beat_start, current_beat_duration, even_measure) =
            current_beat_info(status, self.tempo, self.beats_per_measure);
        // Alpha for things that fade over the beat: opaque at its start.
        let beat_fade = u8::MAX
            - (beat_phase(now, current_beat_start, current_beat_duration) * u8::MAX as f32) as u8;
        let texture_creator = self.canvas.texture_creator();
        self.canvas.set_draw_color(Color::RGB(0x00, 0x00, 0x00));
        self.canvas.clear();

        if due >= self.next_snapshot_refresh {
            self.next_snapshot_refresh = due + self.snapshot_refresh;
            self.samples.clear();
            self.samples.extend(scope.snapshot());
            if !self.samples.is_empty() {
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
        }
        let scope_height = self.height * 3 / 5;
        match state.scope_view {
            scope::ScopeView::Snapshot => self.draw_snapshot(scope_height),
            scope::ScopeView::Scan => self.draw_scan(scope, scope_height),
        }
        self.draw_spectrum(scope_height);

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
            let sounding = status.has_active(now, &WaveformSelector::ProgramPlayback(index));
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
                    .has_pending(now, &WaveformSelector::ProgramPlayback(index))
                    .then_some((ACTIVE_COLOR, u8::MAX)),
            };
            if let Some((circle_color, circle_alpha)) = circle {
                self.draw_circle(&texture_creator, circle_color, circle_alpha, y);
            }

            // Mark highlights go behind the text.
            if highlights_row(mode, active_program_index, index) && !program.is_empty() {
                let lit = lit_ranges(
                    index,
                    program.text(),
                    &status.start_times,
                    &state.waveform_marks,
                    self.sample_rate,
                    now,
                );
                self.draw_highlights(program.text(), &lit, program_color, y);
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
        // Each measure is drawn one sample short, as Instant is a lot more precise than the
        // sample clock: a mark starting at the next measure then never draws a sliver at the
        // end of this one.
        let sample_duration = Duration::from_secs_f64(1.0 / self.sample_rate as f64);
        let marks_duration = player::duration_from_beats(self.tempo, self.beats_per_measure as u64)
            - sample_duration;
        let strip = strip_marks(
            &status.start_times,
            &state.waveform_marks,
            &state.programs,
            self.tempo,
            self.sample_rate,
        );
        // Each mark's rectangle, its colour, and whether it's sounding.
        let mut rects = Vec::new();
        for even in [false, true] {
            // The oldest odd/even Beats waveform's measure.
            let Some(marks_start) = status
                .start_times
                .iter()
                .filter(|(id, _)| *id == WaveformId::Beats(even))
                .map(|(_, start)| *start)
                .min()
            else {
                continue;
            };
            let marks_width = self.width as f32 / 4.0; // for one measure
            let marks_x_offset = if even {
                5.0 * self.width as f32 / 8.0
            } else {
                3.0 * self.width as f32 / 8.0
            };
            for mark in &strip {
                // A mark running into the next measure continues there.
                let Some(part) = mark.within(marks_start, marks_duration) else {
                    continue;
                };
                let program_index = mark.program;
                let row = program_index % PROGRAMS_PER_BANK;
                let mut program_color = programs[program_index]
                    .color()
                    .map(|(r, g, b)| Color::RGB(r, g, b))
                    .unwrap_or(INACTIVE_COLOR);
                let sounding = mark.start <= now && now < mark.start + mark.duration;
                if !sounding {
                    program_color.r /= 2;
                    program_color.g /= 2;
                    program_color.b /= 2;
                }
                let x = (part.start - marks_start).as_secs_f32() / marks_duration.as_secs_f32()
                    * marks_width
                    + marks_x_offset;
                let y = row as f32 * marks_row_height + programs_bottom;
                let width =
                    part.duration.as_secs_f32() / marks_duration.as_secs_f32() * marks_width;
                let height = marks_row_height - marks_y_padding;
                let rect = Rect::new(x as i32, y as i32, width as u32, height as u32);
                rects.push((sounding, program_color, rect));
            }
        }
        // Sounding marks go on top, so overlaps never hide where one starts or
        // ends.
        for sounding in [false, true] {
            for (_, color, rect) in rects.iter().filter(|(s, _, _)| *s == sounding) {
                self.canvas.set_draw_color(*color);
                self.canvas.fill_rect(*rect).unwrap();
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

        // Mark odd and even measures with two bars under the "N / M" part of
        // the beat, the current measure's bar lit.
        let beat_x = self.width as i32 - beat_width as i32 - 20;
        let baseline = self.height as i32 - beat_height as i32 - 20 + self.beat_font.ascent();
        let bars_width = self
            .beat_font
            .size_of(&format!("{} / {}", current_beat, self.beats_per_measure))
            .unwrap()
            .0;
        let bar_width = bars_width / 2 - 1;
        for (i, lit) in [!even_measure, even_measure].into_iter().enumerate() {
            self.canvas.set_draw_color(if lit {
                ACTIVE_COLOR
            } else {
                SCAN_PREVIOUS_COLORS[0]
            });
            self.canvas
                .fill_rect(Rect::new(
                    beat_x + i as i32 * (bars_width / 2 + 1) as i32,
                    baseline + 10,
                    bar_width,
                    6,
                ))
                .unwrap();
        }

        // Draw some internal metrics
        let metrics_height = self.height / 2 - beat_height - 40;
        let metrics_width = 200;
        let x = (self.width - metrics_width - 20) as i32;
        let y = self.height as i32 / 2;
        self.draw_metric(
            &texture_creator,
            x,
            y,
            metrics_height,
            &mut metrics.tracker_load,
            1.0,
        );

        let x = (self.width - 2 * metrics_width - 20) as i32;
        let y = self.height as i32 / 2;
        self.draw_metric(
            &texture_creator,
            x,
            y,
            metrics_height,
            &mut metrics.allocations,
            1000.0,
        );

        let x = (self.width - 3 * metrics_width - 20) as i32;
        let y = self.height as i32 / 2;
        self.draw_metric(
            &texture_creator,
            x,
            y,
            metrics_height,
            &mut metrics.renderer_load,
            1.0,
        );

        self.canvas.present();

        // This frame's load is drawn with the next frame.
        metrics
            .renderer_load
            .set_max(due.elapsed().div_duration_f32(FRAME_BUDGET));
    }

    /// Draws Snapshot's samples as a waveform in the top `waveform_height`
    /// pixels.
    fn draw_snapshot(&mut self, waveform_height: u32) {
        if self.samples.is_empty() {
            return;
        }
        let x_scale = self.width as f32 / (self.samples.len() + 1) as f32;
        self.canvas.set_draw_color(Color::RGB(0x00, 0xFF, 0x00));
        let mut last_y = (waveform_height as f32
            - ((self.samples[0] + 1.0) * (waveform_height as f32 / 2.4)))
            as i32;
        for (i, f) in self.samples.iter().enumerate() {
            let x = (i as f32 * x_scale) as i32;
            let y = (waveform_height as f32 - ((f + 1.0) * (waveform_height as f32 / 2.4))) as i32;
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

    /// Draws Scan's columns in the top `scope_height` pixels, one pixel column
    /// per column of `scope`.
    fn draw_scan(&mut self, scope: &scope::Scope, scope_height: u32) {
        let h = scope_height as f32;
        // Snapshot's amplitude mapping. Like Snapshot, levels below -1 extend
        // below the scope area.
        let y = |v: f32| (h - (v + 1.0) * h / 2.4) as i32;
        let gap_top = (scope_height / 20) as i32;
        let gap_height = scope_height * 9 / 10;

        // Rectangles are collected per color so that each color takes one
        // `fill_rects` call. Bars are indexed by clip level, as the colors are.
        let mut bars: [Vec<Rect>; 3] = Default::default();
        let mut previous_bars: [Vec<Rect>; 3] = Default::default();
        let mut gaps = Vec::new();
        let mut previous_gaps = Vec::new();
        let mut heads = Vec::new();
        let level = |clip| match clip {
            scope::ClipLevel::None => 0,
            scope::ClipLevel::Warn => 1,
            scope::ClipLevel::Clip => 2,
        };
        let mut columns = 0;
        for (column, scan_column) in scope.scan_columns().enumerate() {
            columns = column + 1;
            let x = column as i32;
            let bar = |min: f32, max: f32| {
                let top = y(max);
                Rect::new(x, top, 1, y(min).saturating_sub(top).max(1) as u32)
            };
            // The hatch is a line in every third column.
            let gap = (x % 3 == 0).then(|| Rect::new(x, gap_top, 1, gap_height));
            match scan_column {
                scope::ScanColumn::Empty => {}
                scope::ScanColumn::Current { min, max, clip } => {
                    bars[level(clip)].push(bar(min, max))
                }
                scope::ScanColumn::Previous { min, max, clip } => {
                    previous_bars[level(clip)].push(bar(min, max))
                }
                scope::ScanColumn::CurrentGap => gaps.extend(gap),
                scope::ScanColumn::PreviousGap => previous_gaps.extend(gap),
                scope::ScanColumn::Head => heads.push(Rect::new(x, 0, 2, scope_height)),
            }
        }
        let measure_lines = [0, columns as i32 / 2].map(|x| Rect::new(x, 0, 1, scope_height));

        let colored_bars = SCAN_PREVIOUS_COLORS
            .iter()
            .zip(&previous_bars)
            .chain(SCAN_COLORS.iter().zip(&bars))
            .map(|(color, rects)| (*color, rects.as_slice()));
        for (color, rects) in colored_bars.chain([
            (SCAN_PREVIOUS_GAP_COLOR, previous_gaps.as_slice()),
            (SCAN_GAP_COLOR, &gaps),
            (SCAN_MEASURE_COLOR, &measure_lines),
            (INACTIVE_COLOR, &heads),
        ]) {
            self.canvas.set_draw_color(color);
            self.canvas.fill_rects(rects).unwrap();
        }
    }

    /// Draws the spectrum of Snapshot's samples below the top `waveform_height`
    /// pixels.
    fn draw_spectrum(&mut self, waveform_height: u32) {
        if self.samples.is_empty() || self.spectrum.is_empty() {
            return;
        }
        let x_scale = self.width as f32 / (self.samples.len() + 1) as f32;
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

    /// Draws a block in `color` at half opacity behind each of `ranges`, byte
    /// ranges of `text` as drawn on the row at `y`.
    fn draw_highlights(&mut self, text: &str, ranges: &[Range<usize>], color: Color, y: i32) {
        const INSET: i32 = 4;
        let rects: Vec<Rect> = ranges
            .iter()
            .filter_map(|range| {
                // Measure the whole prefix, as the text's runs do, so blocks
                // keep the font's kerning.
                let x = self.text_width(text.get(..range.start)?);
                let end = self.text_width(text.get(..range.end)?);
                Some(Rect::new(
                    self.nav_width as i32 + x,
                    y + INSET,
                    (end - x).max(0) as u32,
                    (self.line_height as i32 - 2 * INSET) as u32,
                ))
            })
            .collect();
        self.canvas.set_blend_mode(BlendMode::Blend);
        self.canvas
            .set_draw_color(Color::RGBA(color.r, color.g, color.b, 128));
        self.canvas.fill_rects(&rects).unwrap();
        self.canvas.set_blend_mode(BlendMode::None);
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

    /// Draws `metric`'s latest value and a graph of its history at (`x`, `y`).
    ///
    /// A value of `full_scale` is drawn at the graph's full `height`; the
    /// line turns yellow at 0.7 × `full_scale` and red at 0.9 × `full_scale`.
    fn draw_metric(
        &mut self,
        texture_creator: &TextureCreator<WindowContext>,
        x: i32,
        y: i32,
        height: u32,
        metric: &mut Metric<f32>,
        full_scale: f32,
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

            let mut last_y = y + height as i32 - (points[0] / full_scale * height as f32) as i32;
            for (i, &point) in points.iter().enumerate() {
                if i == points.len() - 1 && point == 0.0 {
                    continue; // Skip the last point if it's zero
                }
                let level = point / full_scale;
                let value = y + height as i32 - (level * height as f32) as i32;
                // TODO these thresholds don't make sense for arbitrary metrics
                if level < 0.7 {
                    self.canvas.set_draw_color(default_color);
                } else if level < 0.9 {
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

/// One rectangle of the measure strip: a segment of a waveform a program
/// plays, placed on the clock.
#[derive(Clone, Debug, PartialEq)]
pub struct StripMark {
    /// The index of the program the waveform comes from.
    pub program: usize,
    pub start: Instant,
    pub duration: Duration,
}

impl StripMark {
    /// Returns the part of the mark that falls between `start` and
    /// `start + duration`, or `None` when no part of it does.
    ///
    /// # Example
    ///
    /// ```
    /// use std::time::{Duration, Instant};
    /// use tuun::renderer::StripMark;
    ///
    /// let start = Instant::now();
    /// let second = Duration::from_secs(1);
    /// let mark = StripMark { program: 0, start: start + second / 2, duration: second };
    /// let part = mark.within(start + second, second).unwrap();
    /// assert_eq!((part.start, part.duration), (start + second, second / 2));
    /// assert_eq!(mark.within(start + 2 * second, second), None);
    /// ```
    pub fn within(&self, start: Instant, duration: Duration) -> Option<StripMark> {
        let from = self.start.max(start);
        let to = (self.start + self.duration).min(start + duration);
        (from < to).then(|| StripMark {
            program: self.program,
            start: from,
            duration: to - from,
        })
    }
}

/// Returns the measure strip's marks for the waveforms that start at
/// `start_times`, with segments measured in samples at `sample_rate`.
///
/// A `Program` waveform has a mark for each segment with a duration in its
/// played marks in `waveform_marks`. A sequence's anchor has one for each such
/// segment of its program's step marks in `set`, at each step of the program's
/// sequence in `set`, placed at `tempo`. Segments with the same offset and
/// duration in one waveform's marks give one mark. Other waveforms have none.
pub fn strip_marks(
    start_times: &[(WaveformId, Instant)],
    waveform_marks: &HashMap<WaveformId, PlayedMarks>,
    set: &ProgramSet,
    tempo: u32,
    sample_rate: u32,
) -> Vec<StripMark> {
    let samples = |n: usize| Duration::from_secs_f64(n as f64 / sample_rate as f64);
    let mut strip = Vec::new();
    for (id, start) in start_times {
        // The waveform's program, its segments, and where they start from
        // the waveform's start.
        let (program, segments, starts) = match id {
            WaveformId::Program(program) => match waveform_marks.get(id) {
                Some(played) => (*program, &played.marks.segments, vec![Duration::ZERO]),
                None => continue,
            },
            WaveformId::Step {
                program,
                sixteenth: ANCHOR_SIXTEENTH,
            } => match set.program(*program).and_then(Program::sequence) {
                Some(sequence) => (
                    *program,
                    &sequence.step_marks.segments,
                    sequence
                        .step_beats()
                        .map(|beat| player::beat_offset(tempo, beat))
                        .collect(),
                ),
                None => continue,
            },
            _ => continue,
        };
        let mut drawn = HashSet::new();
        for segment in segments {
            let Some(duration) = segment.duration else {
                continue;
            };
            if !drawn.insert((segment.offset, duration)) {
                continue;
            }
            for step_start in &starts {
                strip.push(StripMark {
                    program,
                    start: *start + *step_start + samples(segment.offset),
                    duration: samples(duration),
                });
            }
        }
    }
    strip
}

/// Returns whether the row of the program at `index` draws mark highlights in
/// `mode`, whose active program is at `active_program_index`: every row does
/// but the one being edited.
pub fn highlights_row(mode: &Mode, active_program_index: usize, index: usize) -> bool {
    !(matches!(mode, Mode::Edit { .. }) && index == active_program_index)
}

/// Returns the byte ranges of `text`, the text of the program at `program`,
/// that are lit at `now`: in order, with overlapping and touching ranges
/// merged.
///
/// A mark of a waveform that started at `start_times` lights its range while
/// one of its segments, measured in samples at `sample_rate`, sounds. Only the
/// marks in `waveform_marks` played from the program and lit against `text`
/// count.
pub fn lit_ranges(
    program: usize,
    text: &str,
    start_times: &[(WaveformId, Instant)],
    waveform_marks: &HashMap<WaveformId, PlayedMarks>,
    sample_rate: u32,
    now: Instant,
) -> Vec<Range<usize>> {
    let samples = |n: usize| Duration::from_secs_f64(n as f64 / sample_rate as f64);
    let mut lit = Vec::new();
    for (id, start) in start_times {
        let Some(played) = waveform_marks.get(id) else {
            continue;
        };
        // Marks lit against other text would point at the wrong code.
        if played.program != program || *played.text != *text {
            continue;
        }
        let segments = &played.marks.segments;
        let sounding = |segment: &Segment| {
            let from = *start + samples(segment.offset);
            from <= now
                && segment
                    .duration
                    .is_none_or(|duration| now < from + samples(duration))
        };
        for mark in played.marks.marks.iter().chain(&played.extra) {
            if segments[mark.segments.clone()].iter().any(sounding) {
                lit.push(mark.text_range.clone());
            }
        }
    }
    lit.sort_by_key(|range| range.start);
    let mut merged: Vec<Range<usize>> = Vec::new();
    for range in lit {
        match merged.last_mut() {
            Some(last) if range.start <= last.end => last.end = last.end.max(range.end),
            _ => merged.push(range),
        }
    }
    merged
}

/// Returns the beat playing at `status.buffer_start`, at `tempo` beats per
/// minute and `beats_per_measure` beats to a measure: its number within the
/// measure, its start, its duration, and whether its measure is even.
///
/// The measure is the one whose Beats waveform started last by then. Before
/// any has started, it's the first beat of an odd measure starting then.
pub fn current_beat_info(
    status: &tracker::Status<WaveformId, MarkId>,
    tempo: u32,
    beats_per_measure: u32,
) -> (u32, Instant, Duration, bool) {
    let beat_duration = player::duration_from_beats(tempo, 1);
    let measure = status
        .start_times
        .iter()
        .filter_map(|(id, start)| match id {
            WaveformId::Beats(even) if *start <= status.buffer_start => Some((*start, *even)),
            _ => None,
        })
        .max_by_key(|(start, _)| *start);
    let Some((measure_start, even_measure)) = measure else {
        return (1, status.buffer_start, beat_duration, false);
    };
    // A measure whose samples run a little past the tempo's stays on its
    // last beat.
    let beat = (status
        .buffer_start
        .duration_since(measure_start)
        .div_duration_f32(beat_duration) as u32)
        .min(beats_per_measure.saturating_sub(1));
    (
        beat + 1,
        measure_start + beat_duration * beat,
        beat_duration,
        even_measure,
    )
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::programs::Evaluated;
    use crate::sequencer::Sequence;
    use crate::trace_walk::{self, WaveformMarks};
    use crate::waveform::Waveform;

    use super::*;

    const SAMPLE_RATE: u32 = 8000;
    const TEMPO: u32 = 60;

    fn segment(offset: usize, duration: Option<usize>) -> Segment {
        Segment { offset, duration }
    }

    fn samples(n: u64) -> Duration {
        Duration::from_secs_f64(n as f64 / SAMPLE_RATE as f64)
    }

    /// Returns the played marks of a waveform of program `program` whose
    /// segments are `segments`.
    fn played(program: usize, segments: Vec<Segment>) -> PlayedMarks {
        PlayedMarks {
            program,
            text: Arc::default(),
            marks: Arc::new(WaveformMarks {
                segments,
                marks: vec![],
            }),
            extra: vec![],
        }
    }

    /// Returns a set whose program 0 is a sequence with steps on `beats` and
    /// whose step waveform has `segments`.
    fn sequenced_set(beats: &[f32], segments: Vec<Segment>) -> ProgramSet {
        let (mut set, _) =
            ProgramSet::from_source("#{level_db=0}\nx = 1;\n".to_string(), Default::default())
                .expect("the fixture parses");
        let steps: Vec<(f32, Range<usize>)> = beats.iter().map(|&beat| (beat, 0..1)).collect();
        let sequence = Sequence {
            steps,
            step_waveform: Waveform::Const(0.0),
            step_marks: Arc::new(WaveformMarks {
                segments,
                marks: vec![],
            }),
        };
        set.record(
            0,
            Arc::default(),
            Ok((Some(Evaluated::Sequence(sequence)), vec![])),
        )
        .expect("a success records");
        set
    }

    #[test]
    fn strip_marks_place_a_program_waveforms_segments_once_each() {
        let start = Instant::now();
        let id = WaveformId::Program(3);
        let segments = vec![
            segment(0, Some(80)),
            segment(0, Some(80)),
            segment(100, None),
            segment(160, Some(40)),
        ];
        let waveform_marks = HashMap::from([(id.clone(), played(3, segments))]);
        let set = sequenced_set(&[], vec![]);
        let strip = strip_marks(&[(id, start)], &waveform_marks, &set, TEMPO, SAMPLE_RATE);
        assert_eq!(
            strip,
            vec![
                StripMark {
                    program: 3,
                    start,
                    duration: samples(80),
                },
                StripMark {
                    program: 3,
                    start: start + samples(160),
                    duration: samples(40),
                },
            ]
        );
    }

    #[test]
    fn strip_marks_place_a_sequences_steps_from_its_anchor() {
        let start = Instant::now();
        let anchor = WaveformId::Step {
            program: 0,
            sixteenth: ANCHOR_SIXTEENTH,
        };
        let set = sequenced_set(&[1.0, 2.5], vec![segment(8, Some(80))]);
        let strip = strip_marks(
            &[(anchor, start)],
            &HashMap::new(),
            &set,
            TEMPO,
            SAMPLE_RATE,
        );
        let starts: Vec<_> = strip.iter().map(|mark| mark.start - start).collect();
        assert_eq!(
            starts,
            vec![samples(8), Duration::from_secs_f32(1.5) + samples(8)]
        );
        assert!(
            strip
                .iter()
                .all(|mark| mark.program == 0 && mark.duration == samples(80))
        );
    }

    #[test]
    fn strip_marks_skip_steps_keys_and_waveforms_without_marks() {
        let start = Instant::now();
        let step = WaveformId::Step {
            program: 0,
            sixteenth: 0,
        };
        let key = WaveformId::Key(60);
        let waveform_marks = HashMap::from([
            (step.clone(), played(0, vec![segment(0, Some(80))])),
            (key.clone(), played(0, vec![segment(0, Some(80))])),
        ]);
        let set = sequenced_set(&[1.0], vec![segment(0, Some(80))]);
        let start_times = [
            (step, start),
            (key, start),
            (WaveformId::Program(1), start),
            (WaveformId::Beats(false), start),
        ];
        let strip = strip_marks(&start_times, &waveform_marks, &set, TEMPO, SAMPLE_RATE);
        assert_eq!(strip, vec![]);
    }

    const TEXT: &str = "_ = $440 * Qw;";

    fn mark(text_range: Range<usize>, segments: Range<usize>) -> trace_walk::Mark {
        trace_walk::Mark {
            text_range,
            segments,
        }
    }

    /// Returns the played marks of a waveform of program 0, lit against
    /// `TEXT`, with `segments`, `marks` and `extra` marks.
    fn played_with(
        segments: Vec<Segment>,
        marks: Vec<trace_walk::Mark>,
        extra: Vec<trace_walk::Mark>,
    ) -> PlayedMarks {
        PlayedMarks {
            program: 0,
            text: Arc::from(TEXT),
            marks: Arc::new(WaveformMarks { segments, marks }),
            extra,
        }
    }

    /// Returns the ranges of `TEXT` lit on program 0's row at `at` after
    /// `start`, with program 0's waveform `played` started at `start`.
    fn lit_at(played: PlayedMarks, start: Instant, at: Duration) -> Vec<Range<usize>> {
        let id = WaveformId::Program(0);
        let waveform_marks = HashMap::from([(id.clone(), played)]);
        lit_ranges(
            0,
            TEXT,
            &[(id, start)],
            &waveform_marks,
            SAMPLE_RATE,
            start + at,
        )
    }

    #[test]
    fn lit_ranges_light_a_mark_only_while_its_segment_sounds() {
        let start = Instant::now();
        let played = || played_with(vec![segment(80, Some(80))], vec![mark(4..8, 0..1)], vec![]);
        assert_eq!(lit_at(played(), start, samples(79)), vec![]);
        assert_eq!(lit_at(played(), start, samples(80)), vec![4..8]);
        assert_eq!(lit_at(played(), start, samples(159)), vec![4..8]);
        assert_eq!(lit_at(played(), start, samples(160)), vec![]);
    }

    #[test]
    fn lit_ranges_wait_for_a_pending_waveform_to_start() {
        let now = Instant::now();
        let id = WaveformId::Program(0);
        let played = played_with(vec![segment(0, Some(80))], vec![mark(4..8, 0..1)], vec![]);
        let waveform_marks = HashMap::from([(id.clone(), played)]);
        let start = now + samples(40);
        let lit = |at: Instant| {
            lit_ranges(
                0,
                TEXT,
                &[(id.clone(), start)],
                &waveform_marks,
                SAMPLE_RATE,
                at,
            )
        };
        assert_eq!(lit(now), vec![]);
        assert_eq!(lit(start), vec![4..8]);
    }

    #[test]
    fn lit_ranges_light_a_mark_during_the_segments_in_its_scope_only() {
        let start = Instant::now();
        let played = || {
            played_with(
                vec![segment(0, Some(80)), segment(80, Some(80))],
                // `$440` in the first segment, `*` in both.
                vec![mark(4..8, 0..1), mark(9..10, 0..2)],
                vec![],
            )
        };
        assert_eq!(lit_at(played(), start, samples(40)), vec![4..8, 9..10]);
        assert_eq!(lit_at(played(), start, samples(120)), vec![9..10]);
    }

    #[test]
    fn lit_ranges_go_dark_in_a_gap_between_segments() {
        let start = Instant::now();
        let played = || {
            played_with(
                vec![segment(0, Some(80)), segment(160, Some(80))],
                vec![mark(4..8, 0..2)],
                vec![],
            )
        };
        assert_eq!(lit_at(played(), start, samples(40)), vec![4..8]);
        assert_eq!(lit_at(played(), start, samples(120)), vec![]);
        assert_eq!(lit_at(played(), start, samples(200)), vec![4..8]);
    }

    #[test]
    fn lit_ranges_light_a_segment_without_a_duration_until_the_waveform_ends() {
        let start = Instant::now();
        let played = played_with(vec![segment(0, None)], vec![mark(4..8, 0..1)], vec![]);
        assert_eq!(lit_at(played, start, Duration::from_secs(3600)), vec![4..8]);
    }

    #[test]
    fn lit_ranges_merge_overlapping_and_touching_ranges() {
        let start = Instant::now();
        let played = played_with(
            vec![segment(0, Some(80))],
            vec![mark(9..13, 0..1), mark(4..8, 0..1), mark(4..10, 0..1)],
            vec![],
        );
        assert_eq!(lit_at(played, start, samples(0)), vec![4..13]);
    }

    #[test]
    fn lit_ranges_skip_marks_lit_against_other_text() {
        let start = Instant::now();
        let mut played = played_with(vec![segment(0, Some(80))], vec![mark(4..8, 0..1)], vec![]);
        played.text = Arc::from("_ = $330 * Qw;");
        assert_eq!(lit_at(played, start, samples(0)), vec![]);
    }

    #[test]
    fn lit_ranges_skip_other_programs_waveforms() {
        let start = Instant::now();
        let mut played = played_with(vec![segment(0, Some(80))], vec![mark(4..8, 0..1)], vec![]);
        played.program = 1;
        assert_eq!(lit_at(played, start, samples(0)), vec![]);
    }

    #[test]
    fn lit_ranges_light_a_steps_literal_only_during_its_own_hit() {
        let start = Instant::now();
        // `[1, 5] | on_beats(c)`: the steps share the step waveform's marks,
        // which light `c`, and each adds its literal.
        let text = "[1, 5] | on_beats(c)";
        let shared = Arc::new(WaveformMarks {
            segments: vec![segment(0, Some(80))],
            marks: vec![mark(18..19, 0..1)],
        });
        let step = |sixteenth: u8, literal: Range<usize>| {
            (
                WaveformId::Step {
                    program: 0,
                    sixteenth,
                },
                PlayedMarks {
                    program: 0,
                    text: Arc::from(text),
                    marks: shared.clone(),
                    extra: vec![mark(literal, 0..1)],
                },
            )
        };
        let (first, first_marks) = step(0, 1..2);
        let (second, second_marks) = step(16, 4..5);
        let waveform_marks =
            HashMap::from([(first.clone(), first_marks), (second.clone(), second_marks)]);
        let start_times = [(first, start), (second, start + samples(160))];
        let lit = |at: Duration| {
            lit_ranges(
                0,
                text,
                &start_times,
                &waveform_marks,
                SAMPLE_RATE,
                start + at,
            )
        };
        assert_eq!(lit(samples(40)), vec![1..2, 18..19]);
        assert_eq!(lit(samples(120)), vec![]);
        assert_eq!(lit(samples(200)), vec![4..5, 18..19]);
    }

    #[test]
    fn highlights_row_skips_only_the_row_being_edited() {
        let edit = Mode::Edit {
            cursor_position: 0,
            errors: vec![],
            completion: None,
        };
        assert!(!highlights_row(&edit, 2, 2));
        assert!(highlights_row(&edit, 2, 3));
        assert!(highlights_row(&Mode::Select, 2, 2));
        assert!(highlights_row(&Mode::Keys, 2, 2));
    }

    #[test]
    fn the_current_beat_comes_from_the_latest_beats_start() {
        let t0 = Instant::now();
        let second = Duration::from_secs(1);
        let status = |buffer_start, start_times| tracker::Status {
            buffer_start,
            marks: Vec::new(),
            start_times,
            tracker_load: None,
            allocations: None,
        };
        // At 60 bpm and 4 beats a measure: odd at t0, even at t0 + 4 s, and
        // the odd one's next pass pending at t0 + 8 s.
        let start_times = vec![
            (WaveformId::Beats(false), t0),
            (WaveformId::Beats(true), t0 + 4 * second),
            (WaveformId::Beats(false), t0 + 8 * second),
        ];
        let at = |seconds: f64| {
            current_beat_info(
                &status(t0 + Duration::from_secs_f64(seconds), start_times.clone()),
                TEMPO,
                4,
            )
        };
        assert_eq!(at(0.0), (1, t0, second, false));
        assert_eq!(at(2.5), (3, t0 + 2 * second, second, false));
        assert_eq!(at(5.0), (2, t0 + 5 * second, second, true));
        // A measure running past the tempo's length stays on its last beat.
        assert_eq!(
            current_beat_info(
                &status(t0 + Duration::from_secs_f64(4.5), start_times[..1].to_vec()),
                TEMPO,
                4
            ),
            (4, t0 + 3 * second, second, false)
        );
        // Before any Beats waveform starts, it's beat 1 of an odd measure.
        assert_eq!(
            current_beat_info(&status(t0, start_times[1..].to_vec()), TEMPO, 4),
            (1, t0, second, false)
        );
    }
}
