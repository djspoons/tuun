# Caching text textures

Labels: wayfinder:grilling
Status: open
Assignee: djspoons
Blocked by: none

## Question

Should the renderer keep text textures across frames instead of rendering every line to a new texture each frame, and if so, which text (Edit-mode program lines, the program list, status messages) and how are entries invalidated? [Measure render cost](08-measure-render.md) found about 0.14–0.19 ms per text texture, and a long program in Edit mode needs about 112 per frame, about 21 ms of text in a 22.8 ms render, more than a 60 Hz frame. Keeping textures meets a `TextureCreator` lifetime in `src/lib/renderer.rs`, in the same way that keeping fonts ([Font loading per frame](05-font-loading.md)) meets the TTF context's.
