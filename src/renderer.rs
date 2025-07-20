use std::time::Instant;

use sdl2::pixels::Color;
use sdl2::render::{TextureCreator, TextureQuery};
use sdl2::ttf::{Font, Sdl2TtfContext};
use sdl2::video::WindowContext;
use sdl2::Sdl;

use realfft::num_complex::ComplexFloat;
use realfft::RealFftPlanner;

use crate::metric::Metric;
use crate::tracker::Status;
use crate::Mode;

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

const INACTIVE_COLOR: Color = Color::RGBA(0, 255, 255, 255);
const ACTIVE_COLOR: Color = Color::RGBA(0, 255, 0, 255);
const PENDING_COLOR: Color = Color::RGBA(255, 222, 33, 255);
const EDIT_COLOR: Color = Color::RGBA(255, 255, 255, 255);
const ERROR_COLOR: Color = Color::RGBA(255, 0, 0, 255);

pub struct Renderer {
    pub video_subsystem: sdl2::VideoSubsystem,
    canvas: sdl2::render::Canvas<sdl2::video::Window>,

    beats_per_minute: u32,
    beats_per_measure: u32,

    width: u32,
    height: u32,
    prompt_width: u32,
    line_height: u32,
    nav_width: u32,
}

pub struct Metrics {
    pub tracker_load: Metric<f32>,
}

const FONT_PATH: &'static str = "/Library/Fonts/Arial Unicode.ttf";

impl Renderer {
    pub fn new(
        sdl_context: &Sdl,
        ttf_context: &Sdl2TtfContext,
        beats_per_minute: u32,
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
            beats_per_minute,
            beats_per_measure,
            width,
            height,
            prompt_width,
            line_height,
            nav_width,
        }
    }

    pub fn render(
        &mut self,
        ttf_context: &Sdl2TtfContext,
        programs: &[String],
        status: &Status,
        mode: &Mode,
        metrics: &mut Metrics,
    ) {
        // TODO so much clean-up
        let texture_creator = self.canvas.texture_creator();
        let font = ttf_context.load_font(FONT_PATH, 48).unwrap();
        let mut bold_font = ttf_context.load_font(FONT_PATH, 48).unwrap();
        bold_font.set_style(sdl2::ttf::FontStyle::BOLD);

        self.canvas.set_draw_color(Color::RGB(0, 0, 0));
        self.canvas.clear();

        let mut y = 10;
        for (i, program) in programs.iter().enumerate() {
            let color = match (&mode, is_active(&status, i), is_pending(&status, i)) {
                (_, true, _) => ACTIVE_COLOR,
                (_, _, true) => PENDING_COLOR,
                (Mode::Edit { index, .. }, _, _) if i == *index => EDIT_COLOR,
                _ => INACTIVE_COLOR,
            };
            let number = char::from_u32(0x2460 + i as u32).unwrap().to_string();
            let number_texture = if is_active(&status, i) {
                make_texture(&bold_font, color, &texture_creator, &number)
            } else {
                make_texture(&font, color, &texture_creator, &number)
            };
            let TextureQuery {
                width: number_width,
                ..
            } = number_texture.query();
            match *mode {
                Mode::Edit { index, ref errors } => {
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
                    if i != index && !program.is_empty() {
                        let text_texture =
                            make_texture(&font, INACTIVE_COLOR, &texture_creator, program);
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
                    } else if i == index {
                        // Loop over each character in program and check to see if it's in any of the error
                        // ranges
                        let mut x = self.nav_width as i32;
                        for (j, c) in program.chars().enumerate() {
                            let color = if errors.iter().any(|e| match e.range() {
                                Some(range) if range.contains(&j) => true,
                                _ => false,
                            }) {
                                ERROR_COLOR
                            } else {
                                EDIT_COLOR
                            };
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
                        let color = if !errors.is_empty() {
                            ERROR_COLOR
                        } else {
                            EDIT_COLOR
                        };
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
                                Some(sdl2::rect::Rect::new(x, y, cursor_width, cursor_height)),
                            )
                            .unwrap();
                    }
                }
                Mode::Select { index } => {
                    if index == i {
                        let prompt_texture =
                            make_texture(&font, INACTIVE_COLOR, &texture_creator, " ▸ ");
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
                    if !program.is_empty() {
                        let text_texture =
                            make_texture(&font, INACTIVE_COLOR, &texture_creator, program);
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
                Mode::Exit => (),
            }
            y += self.line_height as i32;
        }

        match &status.buffer {
            Some(buffer) => {
                // Draw the waveform
                let x_scale = self.width as f32 / (buffer.len() + 1) as f32;
                let waveform_height = self.height * 3 / 5;
                if buffer.len() > 0 {
                    self.canvas.set_draw_color(Color::RGB(0, 255, 0));
                    let mut last_y = (buffer[0] * (waveform_height as f32 / 2.4)
                        + (waveform_height as f32 / 2.0))
                        as i32;
                    for (i, f) in buffer.iter().enumerate() {
                        let x = (i as f32 * x_scale) as i32;
                        let y = (f * (waveform_height as f32 / 2.4)
                            + (waveform_height as f32 / 2.0))
                            as i32;
                        self.canvas
                            .draw_line((x, last_y as i32), (x + x_scale as i32, y))
                            .unwrap();
                        last_y = y;
                    }
                }

                // Draw the spectra
                let mut planner = RealFftPlanner::<f32>::new();
                let fft = planner.plan_fft_forward(buffer.len());
                let mut input = fft.make_input_vec();
                for (i, f) in buffer.iter().enumerate() {
                    input[i] = *f;
                }
                let mut spectrum = fft.make_output_vec();
                if let Err(e) = fft.process(&mut input, &mut spectrum) {
                    println!("Error processing FFT: {}", e);
                } else {
                    let spectrum_height = self.height - waveform_height;
                    let y_scale = -(buffer.len() as f32).sqrt();
                    self.canvas.set_draw_color(Color::RGB(255, 0, 0));
                    let mut last_y = (waveform_height + 300) as i32;
                    for (i, f) in spectrum.iter().enumerate() {
                        let x = ((i * 10) as f32 * x_scale) as i32;
                        let y = (f.abs() / y_scale * (spectrum_height as f32 / 10.0)
                            + (waveform_height + 300) as f32)
                            as i32;
                        self.canvas.draw_line((x, last_y), (x + 9, y)).unwrap();
                        last_y = y;
                    }
                }
            }
            None => (),
        }

        // Draw the current beat
        let current_beat = if status.next_beat_start <= Instant::now() {
            status.current_beat + 1
        } else {
            status.current_beat
        } % self.beats_per_measure as u64
            + 1;
        let beat_font = ttf_context.load_font(FONT_PATH, 64).unwrap();
        let beat_texture = make_texture(
            &beat_font,
            ACTIVE_COLOR,
            &texture_creator,
            format!(
                "{} / {} ({} bpm)",
                current_beat, self.beats_per_measure, self.beats_per_minute
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
        let points: Vec<f32> = metrics.tracker_load.iter().collect();
        if points.len() > 0 {
            let mut last_y = y + metrics_height as i32 - (points[0] * metrics_height as f32) as i32;
            for (i, &load) in points.iter().enumerate() {
                if i == points.len() - 1 && load == 0.0 {
                    continue; // Skip the last point if it's zero
                }
                let value = y + metrics_height as i32 - (load * metrics_height as f32) as i32;
                if load < 0.7 {
                    self.canvas.set_draw_color(Color::RGB(0, 255, 0));
                } else if load < 0.9 {
                    self.canvas.set_draw_color(Color::RGB(255, 222, 33));
                } else {
                    self.canvas.set_draw_color(Color::RGB(255, 0, 0));
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

        self.canvas.present();
    }
}

fn is_pending(status: &Status, index: usize) -> bool {
    status
        .pending_waveforms
        .iter()
        .any(|w| w.id == index as u32)
}

fn is_active(status: &Status, index: usize) -> bool {
    status.active_waveforms.iter().any(|w| w.id == index as u32)
}
