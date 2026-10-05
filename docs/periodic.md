# Periodic Waveforms

Tuun supports a single primitive periodic waveform, written as `Phase`, which describes the behavior of an abstract oscillator. It is the only primitive waveform which repeats in a non-trivial way and does so without depending on another periodic waveform. Like many other waveform combinators, `Phase` transforms a pair of input waveforms into an output waveform.

`Phase` can be used with `Sin` and `Alt` to make other kinds of periodic waveforms, including sine, square, and sawtooth waves. `Phase` is implemented through a form of direct digital synthesis. (More on that below!) We'll focus generating sine waves using `Phase` first and then briefly touch on the related functions that appear in the Tuun expression language below.

## Basic Usage

The output of `Phase` is similar to a sawtooth wave in the interval $[0, 1)$. During each cycle, the output starts at zero and increases across that range until the end of the cycle, at which point it repeats. These values represent how far through the cycle we are at each point in time, so $0.25$ represents the point one quarter through the cycle, $0.5$ represents the point halfway through, and so on.

<!-- TODO add a picture -->

`Phase` takes two parameters: the first is a waveform that determines the frequency of these cycles, measured in cycles per second (hertz): it determines both the length of the cycle and the slope of the output. The second parameter is an offset waveform; its samples are added to each sample in the output before ensuring that the output still falls in the range $[0,1)$.
```
Phase(frequency, offset)
```
These input waveforms represent the *instantaneous* frequency and offset. That is, they determine the frequency and offset at each point in time.

As a first example, a sine wave with a frequency of 440 Hz can be written as follows (since multiplying by $2\pi$ converts cycles into radians).
```
Sin(Const(2 * PI) * Phase(Const(440), Const(0)))
```

That waveform generates the following audio:

<div class="container">
  <tuun-synth description="440 Hz sine wave" open='["std"]' expression="sin(2 * pi * phase(440, 0))" />
</div>

To generate a waveform based on cosine instead of sine, use the offset parameter and the fact that $\cos(\theta) = \sin(\theta + \pi/2)$.

| Tuun waveform                                          | Mathematical equivalent
|---                                                     |---
| `Sin(Const(2 * PI) * Phase(Const(f), Const(1 / 4)))`   | $s(t) = \cos(2\pi f t)$


In general, in the case where both parameters are constant waveforms, `Phase` and `Sin` produce a waveform defined by the following equation:

| Tuun waveform                                        | Mathematical equivalent
|---                                                   |---
| `Sin(Const(2 * PI) * Phase(Const(f), Const(c)))`     | $s(t) = \sin(2\pi f t + \phi)$ where $\phi = 2\pi c$

In other words, when passed a constant frequency and constant offset, `Phase` and `Sin` together generate a sine wave whose amplitude at each point in time is determined by a formula familiar to many high school students.

Time is implicit in Tuun waveforms, so `Phase`, `Sin`, and other Tuun waveforms like `Const` generate a sequence of samples, starting at $t_0 = 0$ and followed by one sample every $\Delta t = 1/f_s$ seconds (where $f_s$ is the sampling frequency).

As an aside, another common use of `Sin` is to generate scalar parameters. In this case, you may see waveforms like the following:
```
Sin(Fixed([2 * PI * 0.1]))
```
Since the length of `Sin` is determined by that of its input, this waveform will generate just one sample. If you need to use `Sin` to compute the sine of a single angle measured in radians, you can write something like this:

| Tuun waveform                  | Mathematical equivalent
|---                             |---
| `Sin(Fixed([c]))`              | $s(0) = \sin(c)$ and $s(t)$ is undefined for $t > 0$


It may be tempting to use `Sin` together with `Time` instead of `Phase`. That is, sine is defined over the whole range of real numbers, not just the interval $[0,2\pi)$, so perhaps an expression like `2 * pi * 440 * time` could be used as the input to `Sin`? Unfortunately, floating point numbers are not the real numbers, and as they get large the gaps between them will introduce audible artifacts in the sound. (To save you time, the example below adds a large offset to the parameter: try different values to hear different effects.)

<div class="container">
  <tuun-synth description="Sin without Phase" open='["std"]' expression="sin(2 * pi * 440 * time + pow(2, 24))" />
</div>


## Dynamic Frequency and Phase

Passing a non-constant waveform as the first parameter to `Phase` will result in a waveform whose frequency changes over time. That is, each sample of the frequency waveform represents the instantaneous frequency at that time. For example, the following waveform generates a sine wave whose frequency starts at 0 Hz and then increases by 500 Hz every second (a frequency "sweep").
```
Sin(Const(2 * PI) * Phase(500 * Time, Const(0)))
```
Which you can listen to here:
<div class="container">
  <tuun-synth description="Frequency sweep" open='["std"]'
    expression="sin(2 * pi * phase(500 * time, 0)) | fin(time - 20)"
  />
</div>

As is made explicit in these examples, the argument to $\sin$ is a phase. Since phase is determined by integrating frequency, Tuun must integrate the first parameter of `Phase` to determine the output at each point in time. In the case above, we can determine the phase at each time $t$ by computing the value of the following definite integral:

$$
2\pi \int_0^t 500 \tau  d\tau = 2\pi \frac{500 t^2}{2} = 2\pi \cdot 250 t^2
$$

Which leads to the following equivalence:

| Tuun waveform                                       | Mathematical equivalent
|---                                                  |---
| `Sin(Const(2 * PI) * Phase(500 * Time, Const(0)))`  | $s(t) = \sin(2\pi \cdot 250 t^2)$


You might imagine that this is *also* equivalent to the following Tuun waveform, which uses a phase offset that depends on time instead of the frequency parameter.

```
Sin(Const(2 * PI) * Phase(Const(0), Const(250) * Time * Time))
```

This waveform will have audible artifacts after a few seconds, much more quickly than the other naive example above. This is because the $250 t^2$ term will become quite large, and (as above) Tuun's 32-bit representation of samples is not accurate enough to represent these numbers without introducing significant errors. (Listen for the sidebands that become audible after about 11 seconds.)
<div class="container">
  <tuun-synth description="Frequency sweep using phase offset" open='["std"]'
    expression="sin(2 * pi * phase(0, 250 * time * time)) | fin(time - 20)"
  />
</div>

You should avoid using Tuun waveforms (and especially intermediate waveforms that are passed to `Sin`) whose values exceed about 10,000 whenever possible. `Phase` helps by computing its result in the interval $[0,1)$ and therefore keeping intermediate values smaller. However, you still need to use the frequency parameter whenever possible: it will also save you the trouble of determining the integral analytically — which may be difficult in some cases — and will often result in fewer occurrences of the `Time` primitive.

In general, for a frequency waveform `f` and a offset waveform `c` (both of which may vary with time), `Phase` and `Sin` produce something like the following equation.

| Tuun waveform                          | Mathematical "equivalent"
|---                                     |---
| ```Sin(Const(2 * PI) * Phase(f, c))``` | $s(t) = \sin \left( 2\pi \int_0^t f(\tau) d\tau + \phi(t) \right)$ where $\phi(t) = 2\pi c(t)$

This is a generalization of the equivalence above and makes clear the role that `Phase` plays in integrating the frequency. Of course, Tuun is not computing that integral exactly; it's approximating it as described below.

## Accumulation

Since Tuun is generating discrete samples, you can think of the implementation of `Phase` as an approximation using a sum of the previous instantaneous frequencies. For example, here's a translation of the above equation into discrete time using a rectangular approximation (a left Riemann sum to be precise).

$$
s[t_n] = \sin \left( \left(2\pi \sum_{i=0}^{n-1} f[t_i] \Delta t\right) + 2\pi c[t_n]\right)
$$

Substituting $\Delta t = 1/f_s$ we have the following:
$$
s[t_n] = \sin \left( \left(2\pi \sum_{i=0}^{n-1} \frac{f[t_i]}{f_s}\right) + 2\pi c[t_n]\right)
\\[1em]
 = \sin \left( 2\pi \left[ \left(\sum_{i=0}^{n-1} \frac{f[t_i]}{f_s}\right) + c[t_n] \right] \right)
$$

We then introduce a term $p[t]$ for the the `Phase` waveform, and remember that the phase must always fall in the interval $[0,1)$.

$$
s[t_n] = \sin ( 2\pi p[t_n])
$$

$$
p[t_n] = \left( \sum_{i=0}^{n-1} \frac{f[t_i]}{f_s} + c[t_n] \right) \bmod 1
$$

We also factor out the sum as a term $a[t]$...

$$
p[t_n] = \left( a[t_n] + c[t_n] \right) \bmod 1
$$

... so that it can be written as a recurrence:

$$
a[t_0] = 0
\\[1em]
a[t_n] = a[t_{n-1}] + \frac{f[t_{n-1}]}{f_s} \enspace \text{ (for n = 1, 2, 3, ...)}
$$

In other words, at each step, we compute a new phase based on:

 * The accumulated phase
 * The frequency at the previous sample divided by $f_s$, that is, change in phase per sample
 * The phase offset at that time

Those equations translate more or less directly to the Rust code that implements `Phase`:
```
let mut accumulator = 0.0;
for i in (0..n) {
    let mut sample = accumulator + offset[i];
    // Take the result "mod 1"
    sample -= sample.floor();
    // Watch out for rounding and keep the interval half open
    out[i] = if sample >= 1.0 { 0.0 } else { sample };
    let incr = frequency[i] / sample_frequency;
    accumulator = accumulator + incr;
}
// Make the sure accumulator also doesn't grow without bound
accumulator -= accumulator.floor();
```

When combined with `Sin`, this is often called (software) *direct digital synthesis* (DDS) or more specifically the *numerically controlled oscillator* (NCO) portion of DDS.

## Expression Syntax

Because $\sin$ is used in several different ways in sound and music synthesis, it appears in several forms in Tuun expressions. Analogous to other waveform combinators, `sin` maps directly to the waveform primitive.

Since `Sin` is so often used together with `Phase`, Tuun's std library includes a shorthand for this pattern:
```tuun
sine = fn(w, phi) =>
  sin(2 * pi * phase(w / (2 * pi), phi / (2 * pi)));
```
The parameters to `sine` are given in radians per second and radians, respectively.

Finally, there are many cases where the phase offset is $0$.

```tuun
$ = fn(freq_hz) => sin(2 * pi * phase(freq_hz, 0));
```

In most cases, you should use `sine` or `$`, with the exception being an argument that will be a constant waveform. In summary:

| Expression      | Description
|---------------- |--------------
| `sin(p)`        | General form: use constants or `Phase` (argument in radians)
| `sine(w, p)`    | `Phase` is built in; both arguments in radians
| `$f`            | Sine wave with frequency `f` measured in hertz and zero phase offset

## Advanced Synthesis

We conclude with two more sophisticated examples using `Sin` and `Phase`. For clarity, they are written using the `sine` function defined above.

### FM synthesis

Frequency modulation (FM) synthesis is a technique for producing rich tones by varying the frequency $w_c$ of carrier oscillator using a second oscillator (the "modulator") whose frequency $w_m$ is also in the audible range. This results in a tone with frequency components $w_c + i w_m$ for $i >= 0$. The formula for FM synthesis is usually presented as follows:

$$
s_\text{FM}(t) = \sin(w_c t + I \sin(w_m t))
$$

Where $I$ is the index of modulation. Confusingly, this "index" is not an integer, but instead a continuous "dial" that controls the strength of the sideband frequency components. When $I = 0$ there is no modulation, and as $I$ increases, the number and strength of sidebands generally increases.

The *frequency* of an FM tone is given as:

$$
w_\text{FM}(t) = w_c + I w_m \cos(w_m t)
$$

You can double check that this is indeed the integral of the argument to $\sin$ above. In some presentations, $I w_m$ is written as $d$, the maximum deviation from the carrier signal, but here we'll continue with $I$.

We can implement this directly in Tuun, again remembering that $\cos(\theta) = \sin(\theta + \pi/2)$, and using one of the helper functions from above.
```
sine(w_c + I * w_m * sine(w_m, pi / 2), 0)
```

The following is an example of an FM tone where the index of modulation `I` is controlled by the slider below. Notice how the number of harmonics generally increases as `I` increases, and some harmonics fade in and out over time.
<div class="container">
  <tuun-synth description="FM synthesis tone (index of modulation)" open='["std"]' sliders='["index:0:0:10"]'>
    let
      f_c = 440,
      w_c = 2 * pi * f_c,

      D = 1,
      f_m = D/2 * f_c,
      w_m = 2 * pi * f_m,

      I = index,
    in
      sine(w_c + I * w_m * sine(w_m, pi / 2), 0)
  </tuun-synth>
</div>

Though phase offset is difficult to perceive audibly in general, the choice of the phase offset *in the modulator* can have significant effects on the relative strength of the sideband frequencies. (See "The Effect of Modulator Phase on Timbres in FM Synthesis." John A. Bate, in _Computer Music Journal_, Vol. 14 (1990) for a discussion of this and other variations of FM synthesis.) In the following, the index of modulation `I` is held constant while the modulator phase varies over time.

<div class="container">
  <tuun-synth description="FM synthesis tone (modulator phase)" open='["std"]' sliders='["phi:1.57:0:1.57"]'>
    let
      f_c = 440,
      w_c = 2 * pi * f_c,

      D = 1,
      f_m = D/2 * f_c,
      w_m = 2 * pi * f_m,

      I = 5,
    in
      sine(w_c + I * w_m * sine(w_m, phi), 0)
  </tuun-synth>
</div>

### PM Synthesis

Since phase offset parameter to `Phase` can also vary with time, Tuun offers another way of writing the original FM formula. Since that formula of the form $\sin(w t + p(t))$, we can treat the modulator as a change in the phase offset rather than a change in the frequency. That leads to the following implementation:
```
sine(w_c, I * sine(w_m, 0))
```
Technically this is *phase* modulation (PM) synthesis rather than frequency modulation, but when the modulator is a sinusoid, they produce the same results. Many implementations of FM synthesis use phase modulation since there are cases where it produces better results.

One case where they are *not* equivalent is where the modulator has a non-zero DC offset (that is, where its average value over time is not zero). In general, the DC offset of the first parameter of `Phase` determines the frequency that we perceive; if the modulator has a non-zero DC offset, it will cause this frequency to shift.

An example of a modulator with a non-zero DC offset is a pulse wave. FM will accumulate this offset, leading to a shift in pitch. In the examples below, a fader can be used to add in a pure tone at the carrier frequency. In the FM case, this pure tone will be highly dissonant. PM, on the other hand, handles this case without changes in pitch (and no dissonance with the additional pure tone). Note, however, that FM and PM have very different timbres with this modulator.

(Here `pulse(width, freq_hz)` is a function that returns a pulse wave with the given width and frequency and is defined in the standard context. A width of 0.0 yields a square wave, while a width of 0.5 yields a pulse with 25% duty cycle.)

First, FM with pulse modulator: `sine(w_c + I * w_m * pulse(0.5, w_m / (2 * pi)), 0)`

<!-- TODO use level in db instead of amplitude -->
<div class="container">
  <tuun-synth description="FM synthesis tone with pulse modulator (with pure tone)" open='["std"]' sliders='["pure_tone_amplitude:0:0:1"]'>
    <script type="text/tuun">
      let
        f_c = 440,
        w_c = 2 * pi * f_c,

        D = 1,
        f_m = D/2 * f_c,
        w_m = 2 * pi * f_m,

        I = 6,
      in
        {[sine(w_c + I * w_m * pulse(0.5, f_m), 0) * 0.5,
          pure_tone_amplitude * $f_c]}
    </script>
  </tuun-synth>
</div>

Second, PM with pulse modulator: `sine(w_c, I * pulse(0.5, w_m / (2 * pi)))`

<div class="container">
  <tuun-synth description="PM synthesis tone with pulse modulator (with pure tone)" open='["std"]' sliders='["pure_tone_amplitude:0:0:1"]'>
    <script type="text/tuun">
      let
        f_c = 440,
        w_c = 2 * pi * f_c,

        D = 1,
        f_m = D/2 * f_c,

        I = 6,
      in
        {[sine(w_c, I * pulse(0.5, f_m)) * 0.5,
          pure_tone_amplitude * $f_c]}
    </script>
  </tuun-synth>
</div>

<script type="module" src="tuun/tuun-synth.js"></script>
