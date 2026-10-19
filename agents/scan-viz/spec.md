# Spec: Scope views (Snapshot and Scan)

Adds the **Scan** scope view next to **Snapshot** in the native SDL2 UI, and replaces how the tracker reports output samples to the UI. Terms follow `agents/CONTEXT.md`. Each decision below comes from a closed ticket in [tickets/](tickets/); the ticket is linked where its reasoning matters.

## Outcome

- The tracker pushes every callback's output samples to the UI through two lock-free `rtrb` rings. `Command::SendCurrentBuffer`, `Status.buffer` and the main loop's `next_buffer_refresh` are gone.
- **Snapshot** draws the latest 2048 output samples. It looks as it does today and redraws, with the spectrum, every `--snapshot-refresh-ms` (default 100).
- **Scan** draws the output's per-column min/max over two measures, aligned to measure boundaries, as a left-to-right sweep. Clipping colours a column yellow or red, and dropped samples show as a hatched gap.
- **Ctrl+S** toggles the scope view in every editor mode and shows "Scope view: Scan" or "Scope view: Snapshot". `--scope-view scan` starts in Scan; the default is Snapshot.
- The spectrum stays below the scope in both views.

Out of scope (from [map.md](map.md)): per-program scopes, the web/wasm UI, Snapshot triggering, freezing the scope, moving other key bindings to Ctrl, and auditing the main thread.

## 1. The sample path (`src/lib/tracker.rs`)

From [How the audio thread reports samples](tickets/01-sample-reporting.md).

### Types

Add `rtrb = "0.4"` to `Cargo.toml` and to the `native` feature's list (like `sdl2`; only `tracker` uses it). It allocates only in `RingBuffer::new`.

```rust
/// The placement on the clock of one callback's output samples.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SampleHeader {
    /// When the first of these samples plays.
    pub buffer_start: Instant,
    /// How many samples follow this header in the sample ring.
    pub written: usize,
    /// How many samples were dropped, because a ring was full, since the previous header.
    pub dropped: usize,
}

/// The audio thread's end of the sample and header rings.
pub struct SampleProducer {
    samples: rtrb::Producer<f32>,
    headers: rtrb::Producer<SampleHeader>,
    // Samples dropped since the last header pushed.
    dropped: usize,
}

/// The UI's end of the sample and header rings.
pub struct SampleConsumer {
    pub samples: rtrb::Consumer<f32>,
    pub headers: rtrb::Consumer<SampleHeader>,
}

/// Returns both ends of a new pair of rings holding `sample_capacity` samples and
/// `header_capacity` headers.
pub fn sample_rings(sample_capacity: usize, header_capacity: usize) -> (SampleProducer, SampleConsumer);
```

### Push (all or nothing, per callback)

`SampleProducer::push(&mut self, buffer_start: Instant, samples: &[f32])`:

1. If `self.headers.slots() >= 1` and `self.samples.slots() >= samples.len()`:
   push the samples with `push_entire_slice`, then push `SampleHeader { buffer_start, written: samples.len(), dropped: self.dropped }`, and reset `self.dropped` to 0.
2. Otherwise push nothing, and add `samples.len()` to `self.dropped` (saturating).

Both calls are allocation-free and lock-free. The newest data is dropped when full; nothing blocks.

**Visibility of samples once a header is seen.** Verified by reading `rtrb` 0.4.0's source (not by running it): `push` and `push_entire_slice` end with a `Release` store of the ring's tail, and `Consumer::pop`, `read_chunk` and `pop_entire_slice` refresh their cached tail with an `Acquire` load whenever it doesn't show enough items. The samples' writes and the sample ring's tail store come before the header push on the audio thread, so a UI thread that has popped a header will find at least `written` samples ready. A short comment in `SampleProducer::push` should say why the samples go first.

### Tracker changes

- `Tracker::new` takes a `SampleProducer` (one argument, which keeps it at seven parameters, clippy's limit).
- `callback` calls `self.sample_producer.push(buffer_start, out)` after `generate`, every callback, whether or not a scope is drawn. `samples_elapsed` advances as today, so the clock keeps counting dropped samples.
- Remove `Command::SendCurrentBuffer`, the `send_current_buffer` field, and `Status.buffer`, including the copy in `callback`. Update every `Status { .. }` literal (`src/main.rs`, `src/lib/effects.rs`, `src/lib/actions.rs`, `src/lib/player.rs`, and the tracker tests).
- Everything else in `Status` (marks, `tracker_load`, `allocations`, the `mpsc` send) is unchanged.

### Wiring in `src/main.rs`

- UI mode: `let (sample_producer, sample_consumer) = tracker::sample_rings(args.sample_rate as usize, SAMPLE_HEADER_CAPACITY);` before `open_playback`, with `const SAMPLE_HEADER_CAPACITY: usize = 1024;` (one second of samples; the header ring practically never fills first). Move the producer into the `open_playback` closure.
- Batch mode (`--ui false`): build rings the same way and leave the consumer undrained. Once full, pushes just count drops.

## 2. Scope state with no drawing (`src/lib/scope.rs`, new)

From [Testing the sample path and scope views](tickets/07-testing.md): everything that isn't SDL drawing lives here and is unit-tested. Register it in `src/lib/mod.rs` under `#[cfg(feature = "native")]`. Dependencies: `crate::tracker` (the consumer and header types, `Status`) and `crate::ids` (to find the measure origin).

### `ScopeView`

```rust
/// Which scope view the screen draws.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScopeView { #[default] Snapshot, Scan }
```

With `toggled()`, `display_name()` ("Snapshot" / "Scan"), and `FromStr` / `Display` using `snapshot` / `scan`, so clap can parse `--scope-view` the way it parses `environment::CheckMode`.

### `Scope`

Owned by `main`; holds the `SampleConsumer`, Snapshot's latest samples, and Scan's columns.

- `Scope::new(consumer, sample_rate, scan_window: Duration, scan_columns: usize)`. `scan_window` is two measures, `2 * beats_per_measure * 60 / tempo` seconds; `scan_columns` is the renderer's width in pixels.
- `drain(&mut self, measure_origin: Option<Instant>)`: while a header can be popped, pop it, then take exactly `header.written` samples with `read_chunk(header.written)`, iterate them, and commit. A missing sample breaks the invariant above, so treat it as a bug and panic with a message naming the invariant. For each chunk:
  - Snapshot: append the samples, keeping only the latest `SNAPSHOT_LEN = 2048` (a `VecDeque<f32>` with that capacity, popping from the front).
  - Scan: if `header.dropped > 0`, mark the interval `[buffer_start − dropped / sample_rate, buffer_start)` as a gap. Then fold each sample, at time `buffer_start + i / sample_rate`, into its column. With no `measure_origin`, Scan columns are left unchanged.
- `snapshot(&self) -> impl Iterator<Item = f32>`: the latest samples, oldest first, up to 2048. Fewer at startup.
- `scan_columns(&self) -> impl Iterator<Item = ScanColumn>`: one per column, left to right (see below).

### Scan columns

From [Scan's time axis](tickets/03-scan-time-axis.md) and [Scan look](tickets/04-scan-look.md).

- **Placement.** For a sample at time `t`, let `offset = (t − measure_origin)` in seconds, signed, as `f64`. Its *pass* is `floor(offset / window)` (an `i64`; negative before the origin) and its column is `floor(frac(offset / window) * columns)`. Because tempo and beats per measure are fixed for a run, each pass starts on a measure boundary and every measure has the same width.
- **Storage.** One slot per column: `Empty`, `Range { pass, min, max }` or `Gap { pass }`. A sample whose pass is newer than its slot's replaces the slot. One whose pass equals the slot's widens `min`/`max` (or replaces a `Gap` of that pass). One whose pass is older is ignored.
- **Write head.** The column and pass of the end of the newest sample drained. The head leads what is heard by the output latency, as Snapshot does today. Following the newest sample rather than `Instant::now()` means every column the head has passed holds current data and a single slot per column is enough.
- **Output.** `ScanColumn` is one of `Empty`, `Current { min, max, clip }`, `Previous { min, max, clip }`, `CurrentGap`, `PreviousGap`, or `Head` (the head's own column). Behind the head (columns `< head`), a slot of the head's pass is shown as current. Ahead of it (columns `> head`), a slot of the pass before is shown as previous. Anything else is `Empty`.
- **Clipping colour.** `pub fn clip_level(min: f32, max: f32) -> ClipLevel` returns `None` when the peak `max(|min|, |max|)` is at most 0.95, `Warn` when it is above 0.95 and at most 1.0, and `Clip` above 1.0. These are Snapshot's per-sample thresholds.

### Measure origin

`pub fn measure_origin(status: &tracker::Status<WaveformId, MarkId>) -> Option<Instant>` returns the `start` of any `WaveformId::Beats(false)` mark with `MarkId::TopLevel` (active or pending). Those starts differ by whole multiples of two measures (`Player::start_beats` repeats the odd Beats waveform every two measures), so any one gives the same windows. Call it on every drain, not once: a tracker clock resync shifts every mark (`Tracker::next_buffer_start`), and the origin should follow it.

The pass *numbers*, though, depend on which start is reported, and the reported start moves a window ahead mid-pass: status lists active marks before pending ones, and a repeating waveform's pending copy takes its next `start` when it activates. So the origin is this pass's start during the first measure and the next pass's start during the second. Measured against those, every sample is in pass 0 or −1, so columns are never replaced (only widened), and the first measure shows as `Empty` while the second is drawn. To prevent this, Scan numbers passes from the first origin it gets. Each later origin is moved by the whole number of windows that brings it nearest that origin, so pass numbers stay fixed and a resync shift of less than half a window is still followed.

## 3. Switching scope views (`src/lib/actions.rs`, `src/lib/sdl2_input.rs`, `src/main.rs`)

From [Switching scope views](tickets/05-switching-scope-views.md).

- `AppState` gets `pub scope_view: scope::ScopeView`, defaulting to Snapshot in `from_source`. `main` sets it from `--scope-view` after building the state.
- `Action::ToggleScopeView` flips it and returns `Effect::ShowMessage(format!("Scope view: {}", state.scope_view.display_name()))`, like `ToggleLaunchMode`. `refused_by_ownership` already lets unlisted actions through, so it works while a program is owned.
- `sdl2_input::InputHandler::classify_keydown`:
  - Keys mode: catch `Scancode::S` with Ctrl next to the Ctrl+C check, before the piano-key mapping (S is a piano key).
  - Every other mode: add `(_, Some(Scancode::S)) if ctrl => Some(vec![Action::ToggleScopeView])` next to the Ctrl+C arm, so it covers Select, Edit, Move Sliders and Record.
- No Launchkey binding.
- `Args` gains:
  - `#[arg(long, default_value_t = scope::ScopeView::Snapshot)] scope_view: scope::ScopeView`
  - `#[arg(long, default_value_t = 100)] snapshot_refresh_ms: u64`, with doc comments saying what each does.
- README: add "Ctrl + S - switch the scope view between Snapshot and Scan" to the key lists (Select mode, with a note that it works in every mode), and mention both flags where flags are described, or under "Keyboard Navigation" if there is no better place.

## 4. Main loop (`src/main.rs`)

- Remove `BUFFER_REFRESH_INTERVAL`, `next_buffer_refresh`, its initial `SendCurrentBuffer`, its block in the loop, and its term in the deadline: `let mut deadline = next_render;`.
- Right after the status drain, call `scope.drain(scope::measure_origin(&status))`. Like statuses, samples don't wake the loop; it already wakes at least once a frame to render.
- Pass `&scope` to `renderer.render`.

The loop's design comes from the main-thread work map (`agents/main-thread/` on `claude/main-thread`), now on `main`: it sleeps in `wait_event_timeout`, collects processing-thread replies, and renders at most once per `renderer::FRAME_BUDGET`. Keep these changes as small as above.

## 5. Drawing (`src/lib/renderer.rs`)

`Renderer` only draws what `Scope` gives it.

- `Renderer::new` takes `snapshot_refresh: Duration` and keeps `next_snapshot_refresh: Instant`.
- **Snapshot timer** (from [The spectrum in Scan](tickets/06-spectrum-in-scan.md)): at the start of `render`, if `due >= next_snapshot_refresh`, copy `scope.snapshot()` into `self.samples`, recompute `self.spectrum` from it (the FFT code that runs on `status.buffer` today), and set `next_snapshot_refresh = due + snapshot_refresh`. It runs in both views, so the spectrum keeps updating under Scan. It adds no deadline to the loop and may fire up to a frame late.
- **Scope area:** the top 3/5 of the window, as today. Split the existing body into `draw_snapshot` (today's waveform code, unchanged) and `draw_scan`, chosen by `state.scope_view`; then draw the spectrum (today's code, unchanged) in either view.
- **`draw_scan`**, one pixel column per `ScanColumn`:
  - Amplitude: linear, using Snapshot's mapping `y(v) = h − (v + 1) · h / 2.4` for scope height `h`, so a level looks the same in both views. A column is a vertical bar from `y(max)` to `y(min)`, at least 1 px tall. As in Snapshot, `y` isn't clamped, so a level below −1 extends below the scope area.
  - Colour, by the column's `ClipLevel`: Snapshot's per-sample colours, green `RGB(0x00, 0xFF, 0x00)` for `None`, `ERROR_COLOR` for `Warn` and `RECORD_COLOR` for `Clip`. Previous columns use the same colours at about 30% brightness (precomputed, so no blend mode is needed).
  - Gaps: a hatch, a vertical line every third column across 90% of the scope height, in dim red (dimmer still for previous gaps), as in the [prototype](prototypes/scan-look.html).
  - Measure lines only: dim cyan at column 0 and at the window's midpoint.
  - Write head: a 2 px line in `INACTIVE_COLOR` (cyan) at the head column.
  - Collect the rectangles per colour and draw each colour with one `canvas.fill_rects` call. One `draw_line` per column, every frame, may not fit the frame budget.
- Scan redraws on every render; only Snapshot and the spectrum wait for the timer.

## 6. Tests

From [Testing the sample path and scope views](tickets/07-testing.md). Follow `CLAUDE.md` § Testing: each unit test builds its own rings, headers and marks.

**`src/lib/tracker.rs`.** `test_tracker()` builds tiny rings with `sample_rings` and also returns the `SampleConsumer`. Tests call `callback` directly:
1. One callback pushes its samples, then a header with that callback's `buffer_start`, `written == out.len()` and `dropped == 0`. The samples match `out`.
2. With a sample ring too small for the next buffer, that callback pushes nothing. After the consumer drains, the next header reports `dropped` equal to the skipped buffer's length.
3. With a full header ring and room in the sample ring, nothing is pushed (the sample ring's slot count doesn't change), and the drop is counted.

Allocation-freedom of the push is checked by inspection, not by a test.

**`src/lib/scope.rs`.**
- `drain` reads exactly each header's samples: two chunks of different lengths come out in order and nothing is left in the sample ring.
- Snapshot keeps the latest 2048 samples, oldest first, across chunks longer and shorter than 2048. It holds fewer before 2048 have arrived.
- Placement: samples at `origin + window/4` land in column `columns/4` of pass 0. At `origin + window + window/4`, they land in the same column, pass 1. Before the origin, they land in pass −1.
- Several samples in one column widen `min`/`max`. A sample from a newer pass replaces the slot; one from an older pass is ignored.
- `scan_columns`: behind the head, current-pass slots are `Current`; ahead of it, previous-pass slots are `Previous`; older slots are `Empty`; the head's column is `Head`.
- A header with `dropped > 0` marks the columns in `[buffer_start − dropped / sample_rate, buffer_start)` as gaps.
- With `measure_origin` `None`, `drain` still updates Snapshot and leaves the Scan columns empty.
- `clip_level`: 0.95 → `None`, 0.96 → `Warn`, 1.0 → `Warn`, 1.01 → `Clip`, and a negative peak (`min = −1.01`) → `Clip`.
- `measure_origin` returns a `Beats(false)` / `TopLevel` mark's start, ignores `Beats(true)` and other marks, and returns `None` without one.
- `ScopeView` parses `snapshot` / `scan`, rejects anything else, and `toggled()` alternates.

**`src/lib/actions.rs`.** `ToggleScopeView` flips `scope_view` and shows the message, and is not refused in Edit or Record mode.

**`src/lib/sdl2_input.rs`.**
- Ctrl+S gives `ToggleScopeView` in Select, Edit, Move Sliders, Keys and Record modes.
- Plain S in Keys mode is still a note.
- Rename `record_mode_accepts_only_ctrl_c` to cover both Ctrl combinations, and add the Ctrl+S assertion.

**No automated tests for SDL drawing.**

## 7. Acceptance

Run the checks in `CLAUDE.md` § Build & Test Verification (`cargo build`, `cargo build --benches`, `cargo test`, `cargo fmt`, `cargo clippy`, and `cargo test --no-default-features --features wasm --lib wasm::`). Then, on the dev's machine:

- The tracker's `allocations` metric is unchanged from before.
- Render time and `renderer_load` stay within the 60 Hz frame budget with Scan showing.
- By eye:
  - an envelope sweeps across two measures, with the previous pass dimmed ahead of the head and lines at both measure starts;
  - clipping columns show in yellow and red;
  - Ctrl+S switches views in each editor mode and shows the status message, without typing an "s" in Edit mode or playing a note in Keys mode;
  - `--scope-view scan` starts in Scan;
  - `--snapshot-refresh-ms` changes how often Snapshot and the spectrum redraw, in both views.
- Gaps (the hatch for dropped samples) have no by-eye check: a source reload runs on the processing thread, so nothing easily stalls the main thread long enough to drop samples. The `scope.rs` tests (§6) cover them.

`--show-loop-timing` is not needed. If it is, bring it in from `claude/main-thread` as one commit and revert it afterwards; nothing here depends on it.

## Suggested commit order

1. Sample path: `rtrb`, `SampleHeader`/`SampleProducer`/`SampleConsumer`, push in `callback`, and its tracker tests. `Status.buffer` and `SendCurrentBuffer` stay for now, so the UI is unchanged.
2. `scope.rs` with `Scope` and its tests. `main` drains it, and Snapshot reads from it on the `--snapshot-refresh-ms` timer. Remove `Status.buffer`, `SendCurrentBuffer` and `next_buffer_refresh`.
3. `ScopeView`, `AppState.scope_view`, Ctrl+S, `--scope-view`, and their tests, with Scan drawing a placeholder (for example, only the head and measure lines).
4. `draw_scan` in full, plus the README.
