# Font loading per frame

Labels: wayfinder:grilling
Status: closed
Assignee: djspoons
Blocked by: [Measure on the dev's machine](02-measure.md)

## Question

`Renderer::render` loads its fonts from disk on every frame (`ttf_context.load_font` in `src/lib/renderer.rs`). How much does that cost, and should the fonts be loaded once and kept?

## Resolution

**Decisions**

1. **Load once and keep.** Fonts are loaded once, when the `Renderer` is created, rather than five or six times per frame.
2. **`Renderer<'ttf>` holds them.** The ttf context is created before the `Renderer` and outlives the main loop, so the fonts borrow it through one lifetime parameter on `Renderer`. `render`, `draw_cursor` and `draw_metric` no longer take the context. Not chosen: a separate `Fonts<'ttf>` struct passed to `render` (a parameter on every call), or leaking the context with `Box::leak` (hides an intentional leak).
3. **Fonts only.** Caching text textures stays in the map's fog, under render cost beyond font loading, until it is measured on the dev's machine.
4. **The font path is out of scope.** The hard-coded macOS path (`/Library/Fonts/Arial Unicode.ttf`) is a portability question, not a latency one.
5. **Closed on container numbers.** The before-and-after check with `--show-loop-timing` on the dev's machine belongs to making the change.

**Measurements behind these decisions**

Headless on a 4-core Linux container (SDL 2.30.0, SDL_ttf 2.22.0), not the dev's machine, so only proportions transfer. Arial Unicode was not available; stand-ins were a 16.8 MB TrueType font, a 26.3 MB CFF font and a 1.8 MB TrueType font. The harness is in [research/font-timing](../research/font-timing/README.md).

- **Per frame:** 5 `load_font` calls (48 and 108 pt at the top of `render`, 64 pt for the beat font, 48 pt in each of the two `draw_metric` calls, even when a metric draws nothing); 6 in Edit mode (`draw_cursor`). About 22 strings are rendered in Select mode and about 63 in Edit mode, where the active line is drawn one character at a time. Nothing is cached across frames.

| One frame, release, median | Fonts loaded each frame | Fonts loaded once |
|---|---|---|
| Select mode, text only | 1.1–4.0 ms | 0.15–0.23 ms |
| Select mode, with texture upload | 2.1–4.7 ms | 0.7–1.0 ms |
| Edit mode, with texture upload | 2.2–5.5 ms (p99 10 ms) | 0.9–1.2 ms |

- One `load_font` takes about 33 µs median for the 16.8 MB TrueType font and about 0.4 ms for the CFF font, so for TrueType the loads are a small part of the gap. Hypothesis (unverified): the rest is the glyph cache, which each new `Font` starts empty.
- Debug builds measure about the same: the time is spent in the C libraries.
- Against the dev's 5–7 ms render step ([Measure on the dev's machine](02-measure.md)), keeping fonts is estimated to save 1–4 ms per frame.
- With fonts kept, creating text textures is the largest remaining text cost here (about 0.7 ms on a software renderer). Caching textures would borrow from the `TextureCreator`; sdl2's `unsafe_textures` feature avoids that.

## Measured after the change

Release build on the dev's machine, idle: `render` 5.6 / 8.4 ms (`current second / 10 s max`) against 6.1 / 17.1 before ([Measure render cost](08-measure-render.md)). `render text` is 3.7 / 5.4 against 4.0 / 14.8 before, for the same 13 textures without the timing display. The saving is about the 0.8 ms that `render fonts` used to show, the low end of the 1–4 ms estimate. The hypothesis that a kept font's glyph cache would also cut `render text` noticeably isn't borne out at idle: 4.0 to 3.7 ms.
