# Lock-free handoff from audio thread to UI in Rust

Research for [ticket 02](../tickets/02-lock-free-handoff.md). Gathered 2026-10-14.
This note lists options and what each one costs. It does not pick a design.

Sources were read directly: the Rust std source at tag `1.97.0` (the toolchain
this repo builds with), and the published `.crate` sources of each crate at the
versions below, as downloaded from `static.crates.io`. Line references point
into those files.

| Crate | Version read | docs.rs |
|---|---|---|
| `rtrb` | 0.4.0 | <https://docs.rs/rtrb/0.4.0> |
| `ringbuf` | 0.5.3 | <https://docs.rs/ringbuf/0.5.3> |
| `triple_buffer` | 9.0.0 | <https://docs.rs/triple_buffer/9.0.0> |
| `crossbeam-queue` | 0.3.14 | <https://docs.rs/crossbeam-queue/0.3.14> |
| `arc-swap` | 1.9.2 | <https://docs.rs/arc-swap/1.9.2> |
| `basedrop` | 0.1.3 | <https://docs.rs/basedrop/0.1.3> |

## How tuun does it today

- `Tracker::callback` (`src/lib/tracker.rs`, `impl AudioCallback for Tracker`)
  builds one `Status` on every audio callback. It sends it with
  `self.status_sender.send(status_to_send).unwrap()`. The sender comes from
  `mpsc::channel()`, which is unbounded.
- Building that `Status` can allocate on the audio thread in two ways that have
  nothing to do with the channel:
  - `status_to_send.marks.extend_from_slice(..)` grows a `Vec` that starts out
    as `Vec::new()`. It allocates whenever there are any marks.
  - When `send_current_buffer` is set, it runs
    `Vec::with_capacity(out.len())` and then `out.clone_into(&mut copy)`. That is
    one heap allocation the size of the audio buffer.
- `src/main.rs` sends `Command::SendCurrentBuffer` every
  `BUFFER_REFRESH_INTERVAL` (200 ms). Each loop it drains `status_receiver`
  with `try_recv`. If nothing has arrived yet, it waits with
  `recv_timeout(10 ms)`. Then it calls `apply_status`, which ends with
  `*status = new_status`.
- What I saw reading the code (verified by reading, not by running it): when
  one drain pulls in several statuses, only the last one survives
  `apply_status`. `renderer.rs` reads `status.buffer` only after the drain
  loop ends. So if the status carrying `buffer: Some(..)` is not the last one
  drained, a later status with `buffer: None` replaces it and that buffer is
  never drawn. In that case the clone was allocated for nothing.
- `size_of::<Status>()` is 80 bytes. I checked this by compiling a stand-in
  struct with the same field types on x86_64. `I` and `M` only appear inside
  `Vec<Mark>`, so they don't change the size.

## Does `std::sync::mpsc::Sender::send` allocate? Yes, now and then.

From std 1.97.0:

- `std::sync::mpsc` is now built on the `mpmc` module. The source says so
  (`library/std/src/sync/mpsc.rs:141`: "ported from the `crossbeam-channel`
  crate"). In `library/std/src/sync/mpmc/mod.rs:198`, `channel()` builds a
  `list::Channel`, and `sync_channel(cap)` builds an `array::Channel`
  (or a `zero::Channel` when `cap == 0`).
- `list::Channel` is a linked list of blocks. Each block holds
  `BLOCK_CAP = 31` slots (`mpmc/list.rs:24-26`). `Block::new()` is
  `Box::new_zeroed()` (`list.rs:73`). `start_send` allocates the next block
  ahead of time when it reaches the second-to-last slot
  (`list.rs:199-202`: "If we're going to have to install the next block,
  allocate it in advance"). It also allocates the first block on the first
  send (`list.rs:205-208`).
  - So one `send` in every 31 heap-allocates one block. For `Status` that is
    roughly `8 + 31 × (80 + 8)` ≈ 2.7 KB. Hypothesis (unverified): the exact
    size depends on padding inside `Slot`. I computed this figure and did not
    measure it.
  - Blocks are freed on the receiving side. `read` calls `Block::destroy`
    (`list.rs:383-388`), which drops the `Box`. A send never frees a block.
- After it writes the slot, `write` calls `self.receivers.notify()`
  (`list.rs:279`). `SyncWaker::notify` (`mpmc/waker.rs:171-183`) reads an
  atomic `is_empty` flag first. If a receiver is registered as waiting, it
  takes a `std::sync::Mutex` (`self.inner.lock()`) and unparks that thread,
  which is a futex wake syscall on Linux.
  - In tuun, a receiver is registered whenever the UI thread is inside
    `recv_timeout(10 ms)`. That happens every UI loop that finds the queue
    empty. So the audio thread will sometimes take a mutex and make a syscall.
    The waiting receiver also takes the same mutex in `register` and
    `unregister` (`waker.rs:152-165`). That means the audio thread can block
    for a moment behind the UI thread. By the strict definition, this is not
    lock-free.
- The doc for `Sender::send` (`mpsc.rs:591`) says "This method will never
  block the current thread." This means it never waits for space, because the
  channel is unbounded. It is not a real-time guarantee. The allocation and
  the mutex above both sit on that path.
- With one sender, the spin-waits in `start_send` (`backoff.spin_heavy()`
  when `offset == BLOCK_CAP`) only wait for *another* sender that is installing
  the next block. Tuun has one sender per channel, so the audio thread should
  not spin there. Hypothesis (unverified): I reached this by reading the code
  and did not trace every interleaving.
- When the UI falls behind: the channel is unbounded, so messages queue up and
  memory grows one block per 31 statuses for as long as the UI is behind. The
  sender is never told.
- `sync_channel(n)` with `try_send` (`mpmc/array.rs:309-314`) uses a buffer
  `Box<[Slot<T>]>` allocated once up front (`array.rs:70,104`). Sends don't
  allocate, and when the buffer is full it returns `TrySendError::Full(msg)`.
  It still calls the same `SyncWaker::notify`, so it still has the mutex and
  unpark path. The `Status` payload and its `Vec`s would still allocate.

## Candidates

### `rtrb`: real-time-safe SPSC ring buffer

- Crate docs (`src/lib.rs:1-20`): "A fixed-capacity buffer is allocated on
  construction. After that, no more memory is allocated (unless the type `T`
  does that internally). Reading from and writing into the ring buffer is
  *lock-free* and *wait-free*. All reading and writing functions return
  immediately."
- `Producer::push` (`lib.rs:333-344`) does a slot write, an index increment,
  and one `Release` store. It has no loop and makes no syscall.
- When full: "Attempts to write to a full buffer return an error; values
  inside the buffer are *not* overwritten." `push` returns
  `PushError::Full(value)`. The newest data is the data lost, unless the
  producer chooses otherwise. If `T` owns heap memory (`Vec<f32>`), dropping
  the rejected value on the audio thread frees that memory there.
- No blocking receive: "there is no way for the reading thread to wait for new
  data, other than trying repeatedly". The UI has to poll. Tuun's UI loop
  already polls every frame.
- Bulk `Copy` data: `push_entire_slice` / `push_partial_slice`
  (`chunks.rs:281,307`) and `write_chunk_uninit` (`chunks.rs:217`) let you
  stream raw `f32` samples, or `(min, max)` pairs, with no per-message
  allocation. `push_entire_slice` fails with `ChunkError` when there are too
  few free slots. The documented example is a stream of `f32` audio samples.
- Has single-producer/single-consumer endpoints only. Has no dependencies by
  default. Authors include Stjepan Glavina (crossbeam) and Matthias Geier.

### `ringbuf`: lock-free SPSC FIFO

- README: "Lock-free SPSC FIFO ring buffer with direct access to inner data …
  Lock-free operations - they succeed or fail immediately without blocking or
  waiting." `HeapRb` allocates its storage once. Splitting puts it in an `Arc`
  (`src/rb/shared.rs:159-165`). It also offers a `StaticRb` that uses no
  `alloc`.
- `try_push` returns `Err(item)` when full (`traits/producer.rs:57-60`).
  `push_slice` returns how many items it actually wrote. There are also
  `Read`/`Write` impls.
- Overwriting: `push_overwrite` / `push_slice_overwrite` exist on the
  `RingBuffer` trait (`traits/ring_buffer.rs:29,48`). They take `&mut self` of
  the *whole* ring buffer, and that trait is implemented only for
  `LocalRb`/`SharedRb` (`rb/local.rs:126`, `rb/shared.rs:142`). They are
  therefore not usable from a split producer on another thread. With a split
  producer you get the same "fail when full" behaviour as `rtrb`.
- Docs call it "lock-free". From the code shape (bounded index arithmetic, no
  retry loop), the push/pop calls look wait-free as well. Hypothesis
  (unverified): the crate does not claim wait-freedom.
- Has caching endpoints (`CachingProd`/`CachingCons`) and async/blocking
  wrapper crates.

### `triple_buffer`: latest-value handoff

- Crate docs (`src/lib.rs`): a "single producer thread is frequently updating a
  shared data block, and a single consumer thread wants to be able to read the
  latest available version". README: "nonblocking, and more precisely bounded
  wait-free." Memory is "3x payload".
- `Input::publish` (`lib.rs:320-349`) is a single `AtomicU8::swap` with
  `AcqRel` ordering. It returns `true` if it "overwrote unread data". Its
  counterpart `Output::update` (`lib.rs:516`) is also a single swap.
- When the UI falls behind, older frames are silently dropped and the UI always
  sees the latest one. There is no queue and memory does not grow. Every
  update between two UI reads is lost, which makes it a poor fit for a
  continuous scrolling scan that needs every slice. It fits "the most recent
  buffer" (today's oscilloscope and FFT view) or "the most recent N-slice
  window".
- Allocation: `Input::write(value)` (`lib.rs:247-253`) assigns
  `*self.input_buffer_mut() = value`. That drops the old contents of the input
  buffer on the producer thread, so with `T = Vec<f32>` the audio thread would
  free memory there. The in-place API (`input_buffer_mut()` + `publish()`, or
  `input_buffer_publisher()`) avoids that. With a fixed-size array or a `Vec`
  that keeps its capacity (`clear()` + `extend_from_slice`), publishing never
  allocates. The docs warn that the contents of the reused buffer are stale
  ("you should always reset the value before use").
- Depends on `crossbeam-utils` (for `CachePadded`).

### Other options worth knowing

- **`crossbeam_queue::ArrayQueue`**: a bounded MPMC queue based on Vyukov's
  bounded queue (`src/array_queue.rs:1-4`) with a buffer allocated up front.
  Its `force_push` (`array_queue.rs:293`) pushes and evicts the *oldest*
  element when full, returning it. Of the queues read here, it is the only one
  that offers "keep the newest N" from the producer side. The evicted value is
  returned to the producer and dropped there, so a `Vec` payload would be freed
  on the audio thread. Hypothesis (unverified): its CAS loops make it
  lock-free but not wait-free. I didn't find a claim either way in the docs I
  read.
- **`crossbeam-channel`**: std's `mpsc` is a port of it, so it behaves the
  same: unbounded = block allocation every 31 messages, bounded = no
  allocation, both use a mutex-guarded waker to wake blocked receivers.
- **`arc-swap`**: atomically swap an `Arc<T>` that a reader can load.
  `docs/performance.rs:5-11`: reads are lock-free and "most of the time" wait-free,
  and writers are lock-free. To publish a new value the audio thread would have
  to allocate a new `Arc`, and swapping can drop the previous `Arc` on the
  writer. So it fits UI → audio parameter handoff better than audio → UI data.
- **`basedrop`**: "Memory-management tools for real-time audio" (`src/lib.rs`).
  Its `Owned`/`Shared` work like `Box`/`Arc`, but dropping one queues the memory
  for a `Collector` on another thread instead of freeing it. This does not move
  data between threads. It solves the "dropping a `Vec` on the audio thread
  frees memory" problem that comes up with each queue above. Creating a node
  (`Node::alloc`, `collector.rs:54`) still allocates.
- **Return-the-buffer pool** (a pattern, not a crate): two SPSC rings, one
  carrying filled `Vec<f32>`s to the UI and one carrying empty ones back. After
  warm-up the audio thread neither allocates nor frees. Hypothesis
  (unverified): this is common practice in audio apps. I did not find a primary
  source that documents it as a named pattern.

## Comparison

| Option | Allocates on send? | Progress guarantee (sender) | UI falls behind |
|---|---|---|---|
| `mpsc::channel` (today) | Yes: one ~2.7 KB block per 31 sends, plus whatever the payload allocates | Not lock-free: `notify` takes a `Mutex` and unparks when a receiver is waiting | Queue grows without limit |
| `mpsc::sync_channel` + `try_send` | No (buffer allocated up front) | Same mutex/unpark path in `notify` | `Full`: newest is rejected |
| `rtrb` | No (allocated at construction) | Lock-free and wait-free (documented) | `Full`: newest is rejected; consumer polls |
| `ringbuf` (split) | No | Lock-free (documented) | `Err`: newest is rejected; overwrite only on an unsplit buffer |
| `crossbeam_queue::ArrayQueue::force_push` | No | Lock-free (hypothesis) | Oldest is evicted and returned to the producer |
| `triple_buffer` (in-place API) | No | Bounded wait-free (documented) | Only the latest value is kept; earlier ones are dropped |

In every queue row, "no allocation" covers only the queue's own memory. A
payload such as today's `Status` (with `marks: Vec<_>` and
`buffer: Option<Vec<f32>>`) still allocates when it is built, and still frees
memory wherever it is dropped. Plain `Copy` payloads avoid that: raw `f32`
samples, `(f32, f32)` min/max pairs, or fixed-size arrays.

## What this means for per-slice min/max

- A per-slice summary is small and `Copy`, for example `(min, max)` as two
  `f32`s per slice. A SPSC ring of those (`rtrb` / `ringbuf` slice push)
  carries every slice with no allocation. The UI decides how many slices of
  backlog to keep by sizing the ring.
- A triple buffer holding a fixed array of the last N slices keeps memory
  bounded and is wait-free. The cost is that the UI only sees snapshots, and
  any slices that scroll out between two UI reads are lost.
- Which of these is right depends on whether the scan view must show every
  slice. That is a design decision for another ticket.
