# Processing timing harness

The measurements in [Where program processing runs](../../tickets/06-evaluation.md) were made with these files. They time phases of processing outside the native UI, so they need no display or audio device.

- `instrumentation.diff`: adds a `prof` module (`src/lib/mod.rs`) and `Instant` accumulators to `environment.rs`, `eval.rs` and `programs.rs`. It applies to commit `e6e6508`.
- `eval_timing.rs`: the harness, an example binary (the repo has no `examples/` directory, so create one). It times reading and parsing the file, a check and evaluation of each program (cold and warm environment), a reload, `splice` with and without the write, and one program's `evaluate_and_record` three times in a row.
- `medium.tuun`, `large.tuun`: stand-in source files (26 and 51 programs).

To run, in a scratch worktree (don't commit the instrumentation):

```text
git apply agents/main-thread/research/processing-timing/instrumentation.diff
mkdir -p examples
cp agents/main-thread/research/processing-timing/eval_timing.rs examples/
cargo run --release --example eval_timing -- agents/main-thread/research/processing-timing/medium.tuun lib/v0
```

- Drop `--release` for a debug build.
- `ONE=<k>` times only the k-th non-empty program.
- `SKIP2=1` skips the per-program table and the cold reload.

Results print to stderr.
