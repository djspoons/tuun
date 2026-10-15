//! Times SDL_ttf font loading and text rendering without a window.
//!
//! Usage: font_timing <font-path> [iterations]
use sdl2::pixels::{Color, PixelFormatEnum};
use sdl2::render::{Canvas, TextureCreator};
use sdl2::surface::{Surface, SurfaceContext};
use sdl2::ttf::{Font, Sdl2TtfContext};
use std::time::{Duration, Instant};

const COLOR: Color = Color::RGB(0x00, 0xFF, 0xFF);

fn stats(name: &str, mut samples: Vec<Duration>) {
    samples.sort();
    let us = |d: Duration| d.as_secs_f64() * 1e6;
    let n = samples.len();
    println!(
        "{name:<44} n={n:<5} min={:>9.1}us med={:>9.1}us p99={:>9.1}us",
        us(samples[0]),
        us(samples[n / 2]),
        us(samples[(n * 99 / 100).min(n - 1)])
    );
}

struct Frame {
    small: Vec<String>,
    beat: String,
    edit_chars: Vec<String>,
}

fn frame(edit: bool) -> Frame {
    let line = "(sine 440 * adsr 0.01 0.1 0.5 0.2) |> lpf";
    let mut small: Vec<String> = (1..=8).map(|i| i.to_string()).collect();
    for i in 0..8 {
        if !(edit && i == 0) {
            small.push(format!("{}{}", &line[..38], i));
        }
    }
    small.push(" ▸ ".into());
    small.push("Loaded programs from session.tuun (8 slots)".into());
    small.push("0.12".into());
    small.push("3.45".into());
    let edit_chars = if edit {
        let mut v: Vec<String> = line.chars().map(|c| c.to_string()).collect();
        v.push("‸".into());
        v
    } else {
        vec![]
    };
    Frame {
        small,
        beat: "3 / 4 (120 bpm)".into(),
        edit_chars,
    }
}

struct Fonts<'ttf> {
    f48: Font<'ttf, 'static>,
    f64: Font<'ttf, 'static>,
    f108: Font<'ttf, 'static>,
}

fn load<'t>(ctx: &'t Sdl2TtfContext, path: &str, extra48: usize) -> Fonts<'t> {
    let f48 = ctx.load_font(path, 48).unwrap();
    let f108 = ctx.load_font(path, 108).unwrap();
    let f64 = ctx.load_font(path, 64).unwrap();
    // draw_metric x2 and draw_cursor each load another 48 pt font.
    for _ in 0..extra48 {
        drop(ctx.load_font(path, 48).unwrap());
    }
    Fonts { f48, f64, f108 }
}

fn draw(
    fonts: &Fonts,
    f: &Frame,
    tc: Option<&TextureCreator<SurfaceContext>>,
    canvas: &mut Option<Canvas<Surface>>,
) {
    let mut one = |font: &Font, s: &str| {
        let surf = font.render(s).blended(COLOR).unwrap();
        if let Some(tc) = tc {
            let t = tc.create_texture_from_surface(&surf).unwrap();
            if let Some(c) = canvas.as_mut() {
                let q = t.query();
                c.copy(&t, None, sdl2::rect::Rect::new(0, 0, q.width, q.height))
                    .unwrap();
            }
        }
    };
    for s in &f.small {
        one(&fonts.f48, s);
    }
    for s in &f.edit_chars {
        one(&fonts.f48, s);
    }
    one(&fonts.f64, &f.beat);
    one(&fonts.f108, "◯");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("font path").as_str();
    let iters: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(300);
    println!(
        "SDL {} / SDL_ttf {} / font {} ({} bytes)",
        sdl2::version::version(),
        sdl2::ttf::get_linked_version(),
        path,
        std::fs::metadata(path).unwrap().len()
    );
    let ctx = sdl2::ttf::init().unwrap();

    for size in [48u16, 64, 108] {
        let mut v = Vec::new();
        for _ in 0..iters {
            let t = Instant::now();
            let font = ctx.load_font(path, size).unwrap();
            v.push(t.elapsed());
            drop(font);
        }
        stats(&format!("load_font {size}pt"), v);
    }

    for (edit, label) in [(false, "select"), (true, "edit")] {
        let f = frame(edit);
        let n_strings = f.small.len() + f.edit_chars.len() + 2;
        let extra48 = if edit { 3 } else { 2 };
        for with_tex in [false, true] {
            let tag = format!("{label} ({n_strings} str{})", if with_tex { "+tex" } else { "" });
            let mut canvas = with_tex.then(|| {
                Surface::new(1920, 1080, PixelFormatEnum::ARGB8888)
                    .unwrap()
                    .into_canvas()
                    .unwrap()
            });
            let tc = canvas.as_ref().map(|c| c.texture_creator());
            // Fresh fonts each frame (as Renderer::render does now).
            let mut v = Vec::new();
            for _ in 0..iters {
                let t = Instant::now();
                let fonts = load(&ctx, path, extra48);
                draw(&fonts, &f, tc.as_ref(), &mut canvas);
                v.push(t.elapsed());
            }
            stats(&format!("{tag} fresh fonts/frame"), v);
            // Fonts loaded once, glyph caches warm.
            let fonts = load(&ctx, path, 0);
            draw(&fonts, &f, tc.as_ref(), &mut canvas);
            let mut v = Vec::new();
            for _ in 0..iters {
                let t = Instant::now();
                draw(&fonts, &f, tc.as_ref(), &mut canvas);
                v.push(t.elapsed());
            }
            stats(&format!("{tag} cached fonts"), v);
        }
    }
}
