# Font loading per frame

Labels: wayfinder:grilling
Status: open
Assignee:
Blocked by: [Measure on the dev's machine](02-measure.md)

## Question

`Renderer::render` loads its fonts from disk on every frame (`ttf_context.load_font` in `src/lib/renderer.rs`). How much does that cost, and should the fonts be loaded once and kept?
