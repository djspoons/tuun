# Synthesis

Tuun can be used to explore different methods of synthesis. This page is far from an explanation or tutorial on these techniques, but might serve as a complement to books like __The Computer Music Tutorial__ by Curtis Roads. The examples here are not meant to be particularly accurate representations of the instruments they are named for; they are meant just to demonstrate these synthesis techniques and how they might be realized in Tuun.

## Additive

Additive synthesis is based on the idea of combining simple waveforms. These examples use sine waves.

### Harmonic

Harmonic overtones are integer multiples of the fundamental tone. These examples make use of the overtone function, defined in the standard context.
```
over = fn(freq) => fn(x) => $(freq*x) * (1/x),
```

Additive synthesis often requires many primitive waveforms to be combined; this example uses 20 per note!

<div class="container">
  <tuun-synth description="Additive synthesis (harmonic)" expanded>
    let
      odd = fn(freq) => map(over(freq), [1, 3, 5, 7]),
  
      // Envelope parameters
      a = 0.05, d = 0.1, r = 0.1, s_level = 0.75,

      organ = fn(dur, freq) => {append(
          map(amp(0.8), odd(freq)),
          map(amp(0.5), odd(3/2 * freq)),
          map(amp(0.3), odd(2 * freq)),
          map(amp(0.4), odd(3 * freq)),
          map(amp(0.2), odd(6 * freq))
        )} * (1 + 0.05 * $3)
        | ADSR(a, d, s_level, max(dur - (a + d + r), 0), r)
        | seq(time - dur),
    in
      <map(organ, [(Q, @48), (Q, @52), (W, @55)])>
  </tuun-synth>
</div>

### Inharmonic

Additive synthesis can also be used to create inharmonic sounds, including tuned percussion instruments. This example is based on parameters from [Jim Woodhouse's Euphonics site](https://euphonics.org/3-3-marimbas-and-xylophones/).

<div class="container">
  <tuun-synth description="Additive synthesis (inharmonic)" expanded>
    let
      bars = fn(dur, freq) => {map(over(freq), [1.0, 3.92, 9.24, 16.27, 24.22, 33.54, 42.97])}
        | ADSR(0, 0.1, 0.3, 0, 0.2) | seq(time - dur)
    in
      <map(bars, [(Q, @60), (Q, @64), (W, @67)])>
  </tuun-synth>
</div>

## Subtractive

Subtractive synthesis starts with a waveform with many component frequencies and then passes it through a filter to remove some of those frequencies. One example of such a starting waveform is a pulse wave.

<div class="container">
  <tuun-synth description="Pulse wave" expanded>
  let pulse_inst = fn(dur, freq) =>
    pulse(0.93, freq) | amp(0.2) | ADSR(0.01, 0, 1, dur, 0.01) | seq(time - dur)
  in
    <map(pulse_inst, [(Q, @60), (Q, @64), (W, @67)])>
  </tuun-synth>
</div>

In this example, the two pulse waves are combined and then passed through a low-pass filter (which removes higher frequencies). This example is based on one from [Welsh's Synthesizer Cookbook](https://synthesizer-cookbook.com/).

<div>
  <tuun-synth description="Subtractive synthesis" expanded>
    <script type="text/tuun">
      let harmonica = fn(dur, freq) =>
        let
          osc1 = pulse(0.93 + 0.05 * $(1.6), freq),
          osc2 = reset(osc1, pulse(0.7, add_cents(add_semitones(freq, 8), 7))),
          osc = 0.375 * osc1 + 0.5 * osc2,
          // Envelope
          a = 0.13, d = 0.33, r = 0.33, s = max(dur - (a + d + r), 0)
        in
          osc
          | lpf(0.5, 1900)
          | ADSR(a, d, 0.5, s, r)
          | seq(time - dur),
      in
        <map(harmonica, [(Q, @60), (Q, @64), (W, @67)])>
      </script>
  </tuun-synth>
</div>

## Phase modulation

As discussed in the [advanced uses of Sine](sine.md#advanced-synthesis), Tuun supports both frequency and phase modulation synthesis. Here, we'll use phase modulation, with `fc` as the frequency of the carrier and `fm` as the frequency of the modulator (both in hertz). Instead of the standard formula...
```
sine(2*pi * fc, I * $fm)
```
We'll flip the order of the operands in the multiplication in the phase parameter:
```
sine(2*pi * fc, $fm * I)
```
This will let us change the magnitude of the modulator over time by using a waveform for `I`. Since we know that the sine expression is infinite and has no offset, the length, offset, and magnitude of the modulator will be determined by `I`. That same envelope will be used for the magnitude the resulting waveform as well.

To create a synthesizer instrument using the function below, provide:
 * The maximum value of the index of modulation $I$
 * $D$, which determines the frequency of the modulator
 * Parameters to the envelope: attack duration, decay duration, sustain level, and release duration

This instrument is based on an example from [Chowning's original article on frequency modulation](https://web.eecs.umich.edu/~fessler/course/100/misc/chowning-73-tso.pdf).

<div class="container">
  <tuun-synth description="Phase modulation synthesis" expanded>
    <script type="text/tuun">
      let
        pm_synth = fn(I_max, D, a, d, s_level, r) => fn(dur, freq) =>
          let
            fc = freq,
            s = max(dur - (a + d + r), 0),
            envelope = ADSR(a, d, s_level, s, r),
            I = I_max | envelope,
            fm = D/2 * fc
          in
            sine(2*pi * fc, $fm * I) | envelope | seq(time - dur),
        
        brass = pm_synth(5, 1, 0.1, 0.1, 0.7, 0.1),
      in
        <[brass(Q, @60), brass(Q, @64), brass(W, @67)]>
    </script>
  </tuun-synth>
</div>

<!--
TODO an example based on noise?
TODO more DX7 style synth stuff ("algorithms" and "operators")
-->

<script type="module" src="tuun/tuun-synth.js"></script>
