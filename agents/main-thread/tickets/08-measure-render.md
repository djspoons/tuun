# Measure render cost

Labels: wayfinder:task
Status: open
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

