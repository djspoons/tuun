# Measure render cost

Labels: wayfinder:task
Status: open
Assignee:
Blocked by: none

## Question

How is the `render` step's 5–7 ms per pass ([Measure on the dev's machine](02-measure.md)) split, on the dev's machine? Break it into:
- loading fonts (`ttf_context.load_font`, about five or six per frame);
- rendering text to surfaces and creating textures from them (`create_texture_from_surface` in `src/lib/renderer.rs`), with the number of textures per frame;
- the rest of drawing;
- `canvas.present()`, and whether it waits for vsync (the canvas is built without `present_vsync`, but the platform's default may still sync).

Measure idle, with several programs playing, and in Edit mode on a long program. Add timing alongside `--show-loop-timing` if it can't show the split already.

Done when the numbers are recorded in this ticket's resolution. They decide whether text textures are worth caching (which meets a `TextureCreator` lifetime) and whether render fits within the 60 Hz frame budget that [The 10 ms status wait](04-status-wait.md) assumes. If it doesn't, handling input on its own thread comes back as a question.
