# Measure render cost

Labels: wayfinder:task
Status: closed
Assignee: djspoons
Blocked by: none

## Question

How is the `render` step's 5–7 ms per pass ([Measure on the dev's machine](02-measure.md)) split, on the dev's machine? Break it into:
- loading fonts (`ttf_context.load_font`, about five or six per frame);
- rendering text to surfaces and creating textures from them (`create_texture_from_surface` in `src/lib/renderer.rs`), with the number of textures per frame;
- the rest of drawing;
- `canvas.present()`, and whether it waits for vsync (the canvas is built without `present_vsync`, but the platform's default may still sync).

Measure idle, with several programs playing, and in Edit mode on a long program. Add timing alongside `--show-loop-timing` if it can't show the split already.

Done when the numbers are recorded in this ticket's resolution. They decide whether text textures are worth caching (which meets a `TextureCreator` lifetime) and whether render fits within the 60 Hz frame budget that [The 10 ms status wait](04-status-wait.md) assumes. If it doesn't, handling input on its own thread comes back as a question.

## Instrumentation

`--show-loop-timing` now also shows, below the existing lines:
- `render driver: <name> (vsync on|off)`: SDL's render driver and whether its `present()` waits for vsync, from `SDL_RendererInfo`.
- `render fonts`, `render text`, `render other`, `render present`: the four parts of one render, in ms, as `current second (10 s max)` like the other lines. `render text` covers `font.render(..).blended(..)` plus `create_texture_from_surface`; `render other` is the time before `present()` less fonts and text; `render present` is `present()` alone. The `render` step line still covers the whole call.
- `text textures`: the most text textures one render created.

The timing display draws its own 15 lines as text textures, so with the flag on, `render text` and `text textures` include them.

Code: `RenderPart` and `LoopTiming::record_render_part` / `record_text_textures` in `src/lib/loop_timing.rs`; `FrameCost`, `load_font` and the timed `make_texture` in `src/lib/renderer.rs`. The `CLAUDE.md` checks pass. It has not been run with a display: the container has neither the display nor the macOS font.


## Measurements

Each value is `current second / 10 s max`, in ms, on the dev's machine.

| Line | Idle | Several programs playing | Edit mode, long program |
|---|---|---|---|
| render driver | metal, vsync off | metal, vsync off | metal, vsync off |
| render | 6.1 / 17.1 | 5.8 / 6.4 | 5.5 / 22.8 |
| render fonts | 0.8 / 1.5 | 0.7 / 1.1 | 0.6 / 3.1 |
| render text | 4.0 / 14.8 | 4.0 / 4.1 | 3.8 / 20.9 |
| render other | 1.2 / 2.7 | 1.2 / 1.4 | 1.0 / 3.1 |
| render present | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| text textures | 28 / 113 | 27 / 27 | 27 / 112 |

- **`present()` doesn't wait.** Metal runs with vsync off and `present()` rounds to 0.0 ms, so nothing paces rendering but the loop itself.
- **Text is about two thirds of an idle render** (4.0 of 6.1 ms), at about 0.14 ms per texture. Font loading is 0.8 ms.
- **15 of the 28 idle textures are the timing display itself,** so without the flag an idle frame makes about 13.
- Hypothesis (unverified): some of `render text` is glyph rasterization that a font kept across frames would cache, so keeping fonts ([Font loading per frame](05-font-loading.md)) may lower `render text` as well as `render fonts`.
- **Several programs playing looks like idle:** render 5.8 ms, of which text is 4.0 ms over 27 textures. Playing adds no measurable render cost; the idle maximums (17.1 ms, 113 textures) were one-off spikes not seen here.
- **Edit mode on a long program** (screenshot taken after leaving Edit mode, so the current second shows the Select-mode frame and the maximums show Edit mode): one render took 22.8 ms, 20.9 ms of it text, across 112 textures, about 0.19 ms each. That is more than a 60 Hz frame (16.7 ms). Leaving Edit mode took 220 ms in the SDL events step, matching the 233 ms in [Measure on the dev's machine](02-measure.md).
- **Conclusion:** text textures are the render cost that grows with what's on screen. In Edit mode on a long program they push a single render past the frame budget, so caching them is worth deciding; idle and playing frames (about 13 textures without the timing display) are not a problem.

