# Processing timing harness

The measurements in [Where program processing runs](../../tickets/06-evaluation.md) were made with these files. They time phases of processing outside the native UI, so they need no display or audio device.

- `instrumentation.diff`: adds a `prof` module (`src/lib/mod.rs`) and `Instant` accumulators to `environment.rs`, `eval.rs` and `programs.rs`. It applies to commit `e6e6508`.
- `eval_timing.rs`: the harness, an example binary (the repo has no `examples/` directory, so create one). It times reading and parsing the file, a check and evaluation of each program (cold and warm environment), a reload, `splice` with and without the write, and one program's `evaluate_and_record` three times in a row.
- `medium.tuun`, `large.tuun`: stand-in source files (26 and 51 programs).
- `tracker_load.rs`: another example binary, which needs no instrumentation. For each named program it plays the program with `Start::Now` into a tracker, once without and once with precompute, and prints the tracker load per buffer (max, p95, mean). Run it from the repo root after copying it into `examples/`:
  `cargo run --release --example tracker_load -- agents/main-thread/research/processing-timing/medium.tuun lib/v0 A:6 A:8 "A:1 (harm_line)"`. Each name is a program's display name.

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

## The processing-thread port

The re-measurement in [Caching evaluated modules and bindings](../../tickets/07-evaluation-caching.md) used ports of these files to the API after the processing thread landed:

- `instrumentation-9d764f7.diff`: the same accumulators, applied to commit `9d764f7`.
- `eval_timing-9d764f7.rs`: the harness over `processing::load` and `processing::process`. It times reading and parsing the file; check, evaluation and `Environment::program_context` for each program (warm environment); a load (cold and warm, with precompute off and on); and processing the first, middle and last program three times each. It drops the splice section and `ONE` / `SKIP2`.

Run it as above, with these two files in place of the originals (copy the harness to `examples/eval_timing.rs`). The library prints a "Checking programs…" line to stdout for each check; send stdout to `/dev/null`.
