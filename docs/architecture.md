# Tuun Architecture


## Waveform Pipeline

Though Tuun waveforms can be created in ways that don't leverage the Tuun expression language, no implementations currently do so. As such, we'll describe the pipeline starting with a Tuun expression and ending with playback on an audio device. There are four main stages, each described in more detail below:

* Expression parsing and evaluation — converting strings into abstract syntax representing waveforms

* Abstract waveform optimizations — transformations of the waveform independent of audio device or sample rate that enable more efficient generation

* Sample generation — conversion of waveforms into audio samples

* Audio device interfacing — mixing multiple tracks into a single buffer for playback

Program text flows through four stages before reaching audio output:

```
  "$(440)" │     Parser           │   Optimizer    │  Generator    │   Tracker
      │    │ parse      evaluate  │    optimize*   │    generate   │
      └─String───► Expr ─────► Waveform ──────► Waveform ─────► Vec<f32> ─────►
           |                      │                │               │   
```
As described below, generation can occur both on a UI thread or on an audio thread. In the first case, it's called "precomputing" some or all of a waveform. This means generating the samples and replacing the waveform, in whole or in part, with a `Fixed` waveform.

### Expression Parsing and Evaluation

Tuun expressions are parsed using a [nom](https://docs.rs/nom/8.0.0/nom/)-based parser. Recall that Tuun expressions form a simple, call-by-value functional language. 

Once parsing is complete, Tuun expressions are evaluated. This means functions and `let` bindings are resolved as well as any arithmetic expressions on floating point numbers. Importantly `seq` and `\` are replaced with a combination of `Fin`, `Append`, and `Merge` as described in the [language overview](tuun-langs.md#sequencing). The result will be a value: either a floating point number, a string, a boolean, a function, waveform, or a seq waveform. In all but the last two cases, an error is presented to the user (since those values cannot be played). In the case of a seq waveform, an `unseq` is implicitly applied to extract the underlying waveform.

### Abstract Waveform Optimizations

Once a waveform has been produced, Tuun could move straight on to generating samples. However, there are often many aspects of the result of evaluating a Tuun expression that would lead to inefficient sample generation. That is, if we think of the waveform as a sort of assembly language program, there are many traditional compiler transformations that we can apply to improve performance.

For example, the optimizer folds constant waveforms and collapses nested `Fin`s to produce simpler waveforms that will generate the same results. It also anticipates strategies that the generator will use and, for example, moves constants to the right side of binary operators (so those operators can be implemented in place).

Note that these transformations occur without generating any samples or even knowing what the sample rate will be. Instead, they take advantage of algebraic properties of the waveforms, including the commutativity and associativity of addition and multiplication.

### Sample Generation

Once a waveform is in a suitable format, we can generate samples. Some types of waveforms maintain internal state that is specific to the implementation of the generator. For example, the `Sine` waveform maintains an accumulator that represents the current phase of the oscillator. At the beginning of generate, all waveforms are initialized with a new state.

Generation can occur in two places:

 * Pre-computation: as a further optimization before playback, waveforms are inspected to determine whether some or all of the samples can be determined in advance. That is, finite waveforms that don't depend on or cause interaction with the environment can be precomputed. Such a waveform is replaced with a `Fixed` waveform that contains the generated samples. In some cases (like a `Marked` waveform) an inner waveform may be replaced with a `Fixed` waveform, while the outer waveform is preserved. Once any pre-computation is complete, the waveform is reinitialized with a new state.

 * Online: when an audio device demands samples for playback, a generator provides the required samples, updating any internal state of the waveform.

### Audio Device Interfacing

The final component of Tuun, the tracker, manages the sets of active and pending waveforms, handles callbacks from the audio device, and uses the generator to create samples as necessary.

## Native App Architecture

The native Tuun app is a two-thread system: a **main thread** for UI and DSL evaluation, and an **audio callback thread** for real-time sample generation. They communicate via channels.

### System Overview

```
                    Main Thread                         Audio Thread
              ┌─────────────────────┐              ┌──────────────────┐
User ──────►  │  SDL2 Event Loop    │────────────► │  Tracker         │
  keyboard    │                     │  (Play,      │                  │
  mouse       │  1. Parse text      │   Stop,      │  1. Drain cmds   │
              │  2. Evaluate exprs  │   MoveSlider)│  2. Promote      │
              │  3. Optimize &      │              │     pending →    │
              │     precompute      │              │     active       │
              │  4. Send commands   │  Status      │  3. Generate &   │
              │  5. Render UI       │◄──────────── │     mix samples  │──► Audio Out
              │                     │  (marks,     │  4. Send status  │    (SDL2)
              └─────────────────────┘   buffer,    └──────────────────┘
                                        load)
```

