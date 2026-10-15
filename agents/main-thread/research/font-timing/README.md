# Font timing harness

The measurements in [Font loading per frame](../../tickets/05-font-loading.md) were made with `font_timing.rs`, an example binary that needs no window. It times `load_font` at 48, 64 and 108 pt. It also times one frame's worth of text (Select mode, about 22 strings; Edit mode, about 63) with fonts loaded fresh each frame and with fonts loaded once, with and without texture upload through a software renderer on a surface canvas.

To run, in a scratch worktree:

```text
mkdir -p examples
cp agents/main-thread/research/font-timing/font_timing.rs examples/
cargo run --release --example font_timing -- <font-path> [iterations]
```

The native UI's font, Arial Unicode, was not available; the measurements used `wqy-zenhei.ttc` (16.8 MB, TrueType), `NotoSerifCJK-Regular.ttc` (26.3 MB, CFF, from the `fonts-noto-cjk` package) and `FreeSans.ttf` (1.8 MB).
