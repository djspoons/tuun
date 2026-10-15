# Caching text textures

Labels: wayfinder:grilling
Status: closed
Assignee: djspoons
Blocked by: none

## Question

Should the renderer keep text textures across frames instead of rendering every line to a new texture each frame, and if so, which text (Edit-mode program lines, the program list, status messages) and how are entries invalidated? [Measure render cost](08-measure-render.md) found about 0.14–0.19 ms per text texture, and a long program in Edit mode needs about 112 per frame, about 21 ms of text in a 22.8 ms render, more than a 60 Hz frame. Keeping textures meets a `TextureCreator` lifetime in `src/lib/renderer.rs`, in the same way that keeping fonts ([Font loading per frame](05-font-loading.md)) meets the TTF context's.

## Resolution

The cost is not a missing cache: in Edit mode the active program is drawn one texture per character (`src/lib/renderer.rs`), so that error ranges can be coloured and the cursor placed. That is most of the 112 textures. Every other line is already one texture.

- **Draw the active Edit-mode line in runs.** Split it where the colour changes (inside or outside an error range) and render each run as one texture. Place the cursor by measuring the text before it with `font.size_of`, which measures without drawing. A long line becomes 1–3 textures.
- **No cross-frame text cache for now.** With runs and fonts kept ([Font loading per frame](05-font-loading.md)), an ordinary frame makes about 13–15 textures, about 2–2.5 ms. A cache would bring a `TextureCreator` lifetime and invalidation for little gain.
- **Acceptance check:** re-measure Edit mode on the same long program with `--show-loop-timing`. A render should take under 16.7 ms, with text textures in the low teens to twenties.
- Hypothesis (unverified): drawing runs may also fix letter spacing, since characters drawn one by one lose the font's kerning.

