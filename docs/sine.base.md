# Sine Waves

Tuun supports a single primitive periodic waveform, a sine wave, written as `Sine`. It is the only primitive waveform which repeats in a non-trivial way and does so without depending on another periodic waveform. Like other waveform combinators, `Sine` transforms a pair of input waveforms into an output waveform. While the shape of `Sine` cannot be changed, it can be used with `Alt` and `Res` to make many other kinds of periodic waveforms (for example, square and sawtooth waves). `Sine` is implemented through a form of direct digital synthesis (more on that below). 

## Basic Usage

`Sine` takes two parameters: the first is a waveform that determines the angular frequency of the output waveform, measured in radians/second; the second is a waveform that determines the angular phase offset of the output waveform, also measured in radians. These waveforms represent the *instantaneous* frequency and phase offset.
```
Sine(angular_frequency, angular_phase_offset)
```

As a simple example, a sine wave with a frequency of 440 Hz can be written as follows (since hertz can be converted into radians by multiplying by $2\pi$).
```
Sine(Const(2 * PI * 440), Const(0))
```

That waveform generates the following audio:

```tuun -C context.tuun
$440 | fin(time - 3) | capture("01-sin-440hz")
```

To generate a waveform based on cosine instead of sine, use the phase offset parameter and the fact that $\cos(\theta) = \sin(\theta + \pi/2)$.

| Tuun waveform                      | Mathematical equivalent
|---                                 |---
| `Sine(Const(w), Const(PI / 2))`    | $s(t) = \cos(w t)$


In general, in the case where both parameters are constant waveforms, `Sine` produces a waveform defined by the following equation:

| Tuun waveform                 | Mathematical equivalent
|---                            |---
| `Sine(Const(2 * PI * f), Const(c))`     | $s(t) = \sin(2\pi f t + c)$

In other words, when passed a constant frequency and constant offset, `Sine` generates a sine wave whose amplitude at each point in time is determined by a formula familiar to many high school students.

Time is implicit in the output of Tuun waveforms, so `Sine` and other Tuun waveforms like `Const` generate a sequence of samples, starting at $t_0 = 0$ and followed by one sample every $\Delta t = 1/f_s$ seconds (where $f_s$ is the sampling frequency).

Another common use of `Sine` is to generate parameters to other waveforms, like `Filter`. In this case, you may see waveforms like the following.

```
Sine(Const(0), Fixed([2 * PI * 0.1]))
```

Since the length of `Sine` is determined by the shorter of its two parameters, this waveform will generate exactly one sample.

If you need to use `Sine` to compute the sine of a single angle measured in radians, you can write something like this:

| Tuun waveform                  | Mathematical equivalent
|---                             |---
| `Sine(Const(0), Fixed[c]))`     | $s(0) = \sin(c)$ and $s(t)$ is undefined for $t > 0$


## Dynamic Frequency and Phase

Passing a non-constant waveform as the first parameter to `Sine` will result in a waveform whose frequency changes over time. That is, each sample of the frequency waveform represents the instantaneous frequency at that time. For example, the following waveform generates a sine wave whose frequency starts at 0 Hz and then increases by 500 Hz every second:
```
Sine(Const(2 * PI * 500) * Time, Const(0))
```
Which you can listen to here:
```tuun -C context.tuun
sin(2*pi*500*time, 0) | fin(time - 20) | capture("02-sweep-as-frequency")
```

Recall that when we write the mathematical function $\sin$, its argument is a phase (or angle). Phase is determined by integrating frequency, so Tuun must integrate the first parameter of `Sine` to determine the phase at each point in time. In the case above, we can determine the phase at each time $t$ by computing the value of the following the integral:
 
 $$
 \int_0^t 2\pi 1000 \tau  d\tau = \frac{2\pi 1000 t^2}{2} = 1000 \pi t^2
 $$

Which leads to the following equivalence:
| Tuun waveform                                              | Mathematical equivalent
|---                                                         |---
| `Sine(Const(2 * PI * 1000) * Time, Const(0))`           | $s(t) = \sin(1000 \pi t^2)$


You might imagine that this is *also* equivalent to the following Tuun waveform, which uses a phase offset that depends on time instead of the frequency parameter.

```
Sine(Const(0), Const(1000 * PI) * Time * Time)
```

And it *is* equivalent... but only up to the point of numeric accuracy, which is this case is not very good: that waveform will have audible artifacts after a few seconds. This is because the $1000 \pi t^2$ term will become quite large, and Tuun's 32-bit representation of samples is not accurate enough to represent these numbers without introducing significant errors. (Listen for the side bands that become audible after about 11 seconds.)
```tuun -C context.tuun
sin(0, pi*500*time*time) | fin(time - 20) | capture("03-sweep-as-phase")
```

You should avoid using Tuun waveforms (especially intermediate waveforms that are passed to `Sine`) whose values exceed about 10,000 whenever possible. In this case, that means using the version of the waveform with fewer uses of the `Time` primitive. This will often save you the trouble of determining the integral and produce better sounding results as well.

In general, for a frequency waveform `w` and a phase offset waveform `p` (both of which may vary with time), `Sine` produces something like the following equation.


| Tuun waveform                          | Mathematical "equivalent"
|---                                     |---
| ```Sine(w, p)```                        | $s(t) = \sin \left( \left(\int_0^t w(\tau) d\tau\right) + p(t)\right)$


Of course, Tuun is not computing that integral exactly; it's approximating it as described below.

## Accumulation

Since Tuun is generating discrete samples, you can think of the implementation of `Sine` as an approximation using a sum of the instantaneous frequencies and phase offsets. For example, here's a simple translation of the above equation into discrete time using a rectangular approximation:
$$
s[t_n] = \sin \left( \left(\sum_{i=0}^n w[t_i] \Delta t\right) + p[t_n]\right) = \sin \left( \left(\sum_{i=0}^n \frac{w[t_i]}{f_s}\right) + p[t_n]\right)
$$

To see how Tuun computes that efficiently, let's introduce an intermediate term: the accumulated phase $a$, sometimes just called "the accumulator."

$$
s[t_n] = \sin(a[t_n] + p[t_n])
$$

Notice how both operands to $+$ are phases and are measured in radians. We can the write a recurrence for the accumulated phase as follows:

$$
a[t_0] = 0
\\[1em]
a[t_n] = a[t_{n-1}] + \frac{w[t_n]}{2f_s} \enspace \text{ (for n = 1, 2, 3, ...)}
$$

That is, at each step, we compute a new phase based on:

 * The previous accumulated phase
 * The angular frequency at that time divided by $f_s$, aka change in phase per sample

 We use the accumulated phase together with the phase offset at that time as the argument to $\sin$.

Those equations translate more or less directly to the Rust code that implements `Sine`:
```
let mut accumulator = 0.0;
for i in (1..n) {
    out[i] = (accumulator + phase_offset[i-1]).sin();
    let phase_inc = frequency[i] / sample_frequency;
    accumulator = (accumulator + phase_inc).rem_euclid(consts::TAU);
}
```

This is often called (software) *direct digital synthesis* (DDS) or more specifically the *numerically controlled oscillator* (NCO) portion of DDS.

## Expression Syntax

Tuun expressions make use of several built-in and pre-defined functions for common uses of $\sin$.

| Expression      | Waveform                 | Description    
|---------------- |--------------------------|--------------  
| `sin(w, p)`     | `Sine(\|w\|, \|p\|)`     | General form, angular frequency (periodic case)
| `sin(p)`        | `Sine(Const(0), \|p\|)`  | Zero frequency ($\sin$ of an angle or other non-periodic case)
| `$e`            | `Sine(Const(2 * PI) * \|e\|, Const(0))` | Frequency measured in hertz (zero phase offset)

where `|e|` is the waveform translation of `e`.

## Advanced Synthesis

We conclude with two more sophisticated examples using `Sine`.

### FM synthesis

Frequency modulation (FM) synthesis is a technique for producing rich tones by varying the frequency $w_c$ of carrier oscillator using a second oscillator (the "modulator") whose frequency $w_m$ is also in the audible range. This results in a tone with frequency components $w_c + i w_m$ for $i >= 0$. The formula for FM synthesis is usually presented as follows:
$$
s_\text{FM}(t) = \sin(w_c t + I \sin(w_m t) )
$$
Where $I$ is the index of modulation. Confusingly, this "index" is not an integer, but instead more of a continuous "dial" that controls the strength of the sideband frequency components. When $I = 0$ there is no modulation, and as $I$ increases, the number and strength of side bands generally increases.

The frequency of an FM tone is given as:
$$
w_\text{FM}(t) = w_c + I w_m \cos(w_m t)
$$
You can double check that this is indeed the integral of the argument to $\sin$ above. In some presentations, $I w_m$ is written as $d$, the maximum deviation from the carrier signal.

We can implement this directly in Tuun, again remembering that $\cos(\theta) = \sin(\theta + \pi/2)$.
```
Sine(w_c + I * w_m * Sine(w_m, PI / 2), 0)
```
(Eliding the `Const` primitive here and below for clarity.)

Here is an example of an FM tone where the index of modulation `I` starts at 0 and increases by 1 every two seconds. Notice how the number of harmonics generally increases, and some harmonics fade in and out over time.
```tuun -C context.tuun
let fc = 440, I = linear(0, 0.5), D = 1, fm = D/2 * fc in sin(2*pi*(fc + (I * fm * sin(2*pi*fm, pi/2))), 0) | fin(time - 20) | capture("04-linear-index-fm")
```

Though in general phase offset is difficult to perceive audibly, the choice of the phase offset *in the modulator* can have significant effects on the relative strength of the sideband frequencies. (See "The Effect of Modulator Phase on Timbres in FM Synthesis." John A. Bate, in _Computer Music Journal_, Vol. 14 (1990) for a discussion of this and other variations of FM synthesis.) In this case, the index of modulation `I` is held constant.

```tuun -C context.tuun
let fc = 440, I = 6, D = 1, fm = D/2 * fc in sin(2*pi*(fc + (I * fm * sin(2*pi*fm, linear(pi/2,pi/8)))), 0) | fin(time - 10) | capture("05-linear-phase-fm")
```

### PM Synthesis

The phase offset parameter to `Sine` can also vary with time, and so Tuun offers another way of writing the original FM formula. Since that formula of the form $\sin(w t + p(t))$, we can treat the modulator as a change in the phase rather than a change in the frequency. That leads to the following implementation:
```
Sine(w_c, I * Sine(w_m, 0))
```
Technically is *phase* modulation (PM) synthesis rather than frequency modulation, but in cases where the modulator is a sinusoid, they produce the same results. Many implementations of FM synthesis are actually phase modulation as it produces better results in some cases.

One case where they are *not* equivalent is where the modulator has a non-zero DC offset (that is, where its average value over time is not zero). An example of a modulator with a non-zero DC offset is a pulse wave. FM will accumulate this offset over time, leading to a shift in pitch. (The examples below are played first without and then with a pure tone at the carrier frequency to demonstrate this shift: notice the beats in the second FM example.) PM, on the other hand, handles this case without changes in pitch. Note, however, that FM and PM have very different timbres with this modulator.

First, FM with pulse modulator:

  * `Sine(w_c + I * w_m * pulse(0.5, w_m), 0)`
```tuun -C context.tuun
let fc = 440, I = 6, D = 1, fm = D/2 * fc in {[$fc | fin(time - 5), sin(2*pi*(fc + (I * fm * pulse(0.5, fm))), 0) | fin(time - 5) | capture("06-fm-pulse-modulator")]} | capture("07-fm-pulse-modulator-with-pure")
```

Second, PM with pulse modulator:

  * `Sine(w_c, I * pulse(0.5, w_m))`
```tuun -C context.tuun
let fc = 440, I = 6, D = 1, fm = D/2 * fc in {[$fc | fin(time - 5), sin(2*pi*fc, I * pulse(0.5, fm)) | fin(time - 5) | capture("08-pm-pulse-modulator")]} | capture("09-pm-pulse-modulator-with-pure")
```

(Here `pulse(width, freq)` is shorthand for a pulse wave with the given width and frequency. A width of 1.0 yields a square wave, while a width of 0.5 yields a pulse with 25% duty cycle.)




