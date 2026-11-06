# Impulse Response Filters

Tuun's `Filter` waveform implements feed-forward and feedback filters, also known as finite and infinite impulse response filters. It takes an input waveform and two sets of coefficients, called the feed-forward coefficients and the feedback coefficients. (`Filter` should not be confused with more generic usage of the word "filter" to mean any transformation of a waveform.)

## Feed-Forward Filters

Feed-forward filters process the samples of an input waveform by taking a weighted sum of the most recent $K$ samples from that waveform. Feed-forward filters are also called finite impulse response, since as long as the input waveform eventually goes to $0.0$, the output of the filter will as well. They can also be understood as a *convolution* of the input and the coefficients, as can be seen below.

To start with a simple example, the following is a filter that averages the four most recent samples; that is, it's moving average with a window size of four. This requires four coefficients.

```
Filter(w, [Const(0.25), Const(0.25), Const(0.25), Const(0.25)], [])
```

The value of the $n^\text{th}$ sample (at time $t_n$) is the sum of products:

$$
y[t_n] = 0.25 \cdot w[t_n] + 0.25 \cdot w
[t_{n-1}] + 0.25 \cdot w[t_{n-2}] + 0.25 \cdot w[t_{n-3}]
$$

Input samples before the first one are 0.0, so the first three output samples average fewer than four input samples:

$$
y[t_0] = 0.25 \cdot w[t_0] + 0.25 \cdot 0 + 0.25 \cdot 0 + 0.25 \cdot 0 = 0.25 \cdot w[t_0]
$$

Each output sample depends only on the current and earlier input samples: that is, the filter is *causal*. A sound passed through it never starts before the sound itself would have.

A feed-forward filter with $K$ *constant* coefficients can be written as the following waveform:

```
Filter(w, [Const(b_0), Const(b_1), ..., Const(b_K)], [])
```

The output of that filter at time $t_n$ is defined as follows:

$$
y[t_n] = \sum_{i=0}^{K-1} b_i \cdot w[t_{n - i}]
$$

where $w[t_{n - i}] = 0$ when $n - i < 0$.

<!--
TODO example with weighted average filter
-->

The filter coefficients can also be arbitrary waveforms whose values vary over time. More generally, a feed-forward filter with $K$ coefficients looks like this:

```
Filter(w, [b_0, b_1, ..., b_K], [])
```

The output of which is defined by the following formula:

$$
y[t_n] = \sum_{i=0}^{K-1} b_i[t_n] \cdot w[t_{n - i}]
$$

Notice that the filter reads one sample from each coefficient for each output sample, and that each coefficient's sample is taken at the time of the output, not of the input sample it multiplies. 

<!-- 
TODO example where coefficients change over time
-->

While the values of the coefficients can change, the *number* of coefficients is fixed.

## Feedback Filters

The Tuun `Filter` waveform can also be used to create feedback filters. In addition to feed-forward coefficients, these filters also have feedback coefficients.

```
Filter(w, [b_0, b_1, ..., b_K], [a_1, ..., a_J])
```

Feedback coefficients are multiplied together with previously generated *output* samples and then subtracted from the result. If there are $J$ feedback coefficients, Tuun uses $J$ values equal to $0.0$ to bootstrap the filter.

$$
y[t_n] = \sum_{i=0}^{K-1} b_i[t_n] \cdot w[t_{n - i}] - \sum_{i=1}^J a_i[t_n] \cdot y[t_{n - i}]
$$

The feedback coefficients are labeled starting with $a_1$ with the assumption that the coefficient $a_0$ for $y[t_n]$ is equal to $1$. If you are implementing a feedback filter where $a_0 \ne 1$, simply divide all of the other coefficients by $a_0$, as in the low-pass filter example below.

It's very easy to provide feedback parameters which cause the output to diverge! In general, even if the input samples go to 0.0, there's no guarantee that the output samples will. As such, feedback filters are sometimes called *infinite* impulse response filters.

### Exponential Decay

A simple example that uses a feedback filter is an exponential decay envelope. Exponential decay is useful in creating percussion instruments where you want a short, intense burst of sound followed by a fast but smooth drop in amplitude. You can create exponential decay using the exponential function `exp` but it can be computationally cheaper to do so with `filter` instead.

 In this example, we create an envelope by combining an impulse with a filter that has a single feedback parameter. You can control `f` with the slider below. (Due to rounding in the display, it will sometimes show "1.00" when the actual value is slightly less than 1.0.) The slider is limited to values less than 1.0, since the filter output diverges when `f` is greater than 1.0.

<div class="container">
  <tuun-synth
    open='["std"]'
    sliders='["f:0.5:fn(x) => 1 - pow(10000, -(x+1)/2)"]'>
  let
    env = append(fixed([1]), 0) | filter([1], [-f])
  in
    reset($(1/2), triangle(60) * env | fin(time - 1))
  </tuun-synth>
</div>

The envelope in this example implements the following equations:

$$
y[t_0] = 1
\\[1em]
y[t_n] = f \cdot y[t_{n-1}]
$$

That is, each output sample is a constant factor times the previous output sample. This is equivalent to:

$$
y[t_n] = f^n \cdot y[t_{0}] = f^n
$$

Since there is only one feed-forward and one feedback parameter, and they are both constant, we can simplify the equation above. Here $x[t_n]$ is the input to the filter (the impulse). 

$$
y[t_n] = w[t_n] + f \cdot y[t_{n - 1}]
$$

Recall that $y[t_n]$ for $n <$ 0 is equal to 0.0. The first five output values are given for $f =$ 0.995 in the table below.


$t_n$ | $x[t_n]$ | $y[t_n]$
|---  |--------  |---------
0     | 1.0      | 1.00000
1     | 0.0      | 0.99500
2     | 0.0      | 0.99002
3     | 0.0      | 0.98507
4     | 0.0      | 0.98015

### Random Constant Waveform

An even simpler example of a feedback filter is a sample-and-hold register. This filter takes the first sample from its input and repeats it forever.
```
Filter(_, [Fixed([1])], [Const(-1)])
```
(`Filter` zero extends its coefficient waveforms, so the feed-forward coefficient is equivalent to `Fixed([1]) & Const(0)`.)

This implements the following equations:

$$
y[t_0] = 1 \cdot x[t_0] = x[t_0]
\\[1em]
y[t_n] = 0 \cdot x[t_n] + 1 \cdot y[t_{n-1}] = x[t_0]
$$

The example below uses a sample-and-hold filter and `Noise` to generate a random constant waveform. As an argument to `Alt`, this means that only one of the two waveforms will be used each time the `Alt` is repeated: you'll hear one of the two notes each time, but which one you hear is selected randomly.
<div>
  <tuun-synth open='["std"]'>
    <script type="text/tuun">
      let
        hold = fn() => filter([fixed([1])], [-1]),
        rand = fn() => noise | hold(),
      in
        reset($(1/2), alt(rand(), $220, $330) | fin(time - 1))
    </script>
  </tuun-synth>
</div>
(One last aside: neither `hold` nor `rand` needs to be defined as a thunk since there are no effects in their bodies. The randomness of `Noise` occurs not during evaluation in Tuun's expression language, but when generating samples: `Noise` is a value in the expression language, but nonetheless will generate _different_ samples each time it's played.)

### Biquad Low-Pass Filter

`Filter` can also be used to implement much more sophisticated transformations, including low-pass, high-pass, and band-pass filters. In these cases, determining feedback coefficients can be a subtle art, and Tuun's library includes several functions that can help. The following example uses a biquad feedback filter described in [Robert Bristow-Johnson's Cookbook](https://webaudio.github.io/Audio-EQ-Cookbook/Audio-EQ-Cookbook.txt). This filter attenuates frequencies above `f_c` (the "center frequency") and passes through those below it.

<div class="container">
  <tuun-synth
    open='["std"]'
    sliders='["frequency:0.5:fn(x) => 100 * pow(100, x)"]'>
    let
      // RBJ Cookbook low-pass filter
      lpf = fn(Q, f_c) =>
        let
          w0 = 2*pi*f_c/sample_rate,
          alpha = sin(w0)/(2*Q),
          b0 = (1 - cos(w0))/2,
          b1 =  1 - cos(w0),
          b2 = (1 - cos(w0))/2,
          a0 =  1 + alpha,
          a1 = -2*cos(w0),
          a2 =  1 - alpha,
        in
          filter([b0/a0, b1/a0, b2/a0], [a1/a0, a2/a0]),
      in
        0.5 * square(220) | lpf(0.707, frequency)
  </tuun-synth>
</div>


## Notes on Length

Each Tuun `Filter` output waveform is the same length as its input. The output ends with the last input sample, so with $K$ feed-forward coefficients, the response to the last $K-1$ input samples is cut short. (To hear all of it, extend the input with zeros, for example by appending a waveform that is $K-1$ samples of 0.0.) In most cases the coefficient waveforms will be infinite; if they are not, they are extended with zeros to match the length of the input.

Finally, because the length of a `Filter` waveform is independent of the *values* of the coefficients, an *infinite* impulse response filter can have a *finite* length. Similarly, a *finite* impulse response filter can have an *infinite* length (though the output will still eventually go to 0.0).

<script type="module" src="tuun/tuun-synth.js"></script>
