# Impulse Response Filters

Tuun's `Filter` waveform implements feed-forward and feedback filters, also known as finite and infinite impulse response filters. It takes an input waveform and two sets of coefficients, called the feed-forward coefficients and the feedback coefficients. (`Filter` should not be confused with more generic usage of the word "filter" to mean any transformation of a waveform.)

## Feed-Forward Filters

Feed-forward filters process the samples of an input waveform by taking a weighted sum of the most recent $K$ samples from that waveform. Feed-forward filters are also called finite impulse response, since as long as the input waveform eventually goes to $0.0$, the output of the filter will as well. They can also be understood as a *convolution* of the input and the coefficients, as can be seen below.

To start with a simple example, the following is a filter that averages the four most recent samples; that is, it's moving average with a window size of four. This requires four coefficients.

```
Filter(w, [Const(0.25), Const(0.25), Const(0.25), Const(0.25)], [])
```

The value of the $n^\text{th}$ sample (at time $t_n$) is usually written as the sum of products:

$$
y[t_n] = 0.25 \cdot w[t_n] + 0.25 \cdot w
[t_{n-1}] + 0.25 \cdot w[t_{n-2}] + 0.25 \cdot w[t_{n-3}]
$$

However, Tuun implements this without negative indexes or padding the beginning of the input:

$$
y[t_n] = 0.25 \cdot w[t_{n+3}] + 0.25 \cdot w[t_{n+2}] + 0.25 \cdot w[t_{n+1}] + 0.25 \cdot w[t_n]
$$

This means that the input to the filter `w` is delayed by three samples in this example. `Filter` will not use the input sample for $t_{n+3}$ until $t_n$.

If the input waveform to this filter has a finite length, Tuun will extend it by three samples (all zeros), so if $w[t_n]$ is the last input sample, the following defines the last output sample.

$$
y[t_n] = 0.25 \cdot 0 + 0.25 \cdot 0 + 0.25 \cdot 0 + 0.25 \cdot w[t_n] = 0.25 \cdot w[t_n]
$$

That is, the last sample in the output only depends on one sample from the input.

A feed-forward filter with $K$ *constant* coefficients can be written as the following waveform:

```
Filter(w, [Const(b_0), Const(b_1), ..., Const(b_K)], [])
```

The output of that filter at time $t_n$ is defined as follows:

$$
y[t_n] = \sum_{i=0}^{K-1} b_i \cdot w[t_{n + (K - 1 - i)}]
$$

If there are $K$ feed-forward coefficients, then the input is delayed by $K-1$ samples, and finite inputs are zero-extended by $K-1$ samples at the end.

<!--
TODO example with weighted average filter
-->

The filter coefficients can also be arbitrary waveforms whose values vary over time. More generally, a feed-forward filter with $K$ coefficients looks like this:

```
Filter(w, [b_0, b_1, ..., b_K], [])
```

The output of which is defined by the following formula:

$$
y[t_n] = \sum_{i=0}^{K-1} b_i[t_n] \cdot w[t_{n + (K - 1 - i)}]
$$

Notice that the filter reads one sample from each coefficient for each output sample. 

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
y[t_n] = \sum_{i=0}^{K-1} b_i[t_n] \cdot w[t_{n + (K - 1 - i)}] - \sum_{i=1}^J a_i[t_n] \cdot y[t_{n - i}]
$$

The feedback coefficients are labeled starting with $a_1$ with the assumption that the coefficient $a_0$ for $y[t_n]$ is equal to $1$. If you are implementing a feedback filter where $a_0 \ne 1$, simply divide all of the other coefficients by $a_0$, as in the low-pass filter example below.

Note that it's very easy to provide feedback parameters which cause the output to diverge! In general, even if the input samples go to $0.0$, there's no guarantee that the output samples will. As such, feedback filters are sometimes called *infinite* impulse response filters.

Determining feedback coefficients can be a subtle art, and Tuun's library includes several functions that can help. The following example uses a feedback filter defined in [Robert Bristow-Johnson's Cookbook](https://webaudio.github.io/Audio-EQ-Cookbook/Audio-EQ-Cookbook.txt).

<div class="container">
  <tuun-synth
    sliders='["frequency:100:10000:1000"]'>
    let
      // RBJ Cookbook low-pass filter
      lpf = fn(Q, fc) =>
        let
          w0 = 2*pi*fc/sample_rate,
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
        square(220) | lpf(0.707, frequency)
  </tuun-synth>
</div>



## A Note on Length

Tuun `Filter` waveforms are always the same length as their input. As noted above, finite waveforms are extended by $K-1$ samples (where $K$ is the number of feed-forward coefficients) since the first $K-1$ input samples are consumed for the first output sample.

Recall that the length of the sum (or difference) of two waveforms is the *maximum* of their two lengths. Therefore, the length of a `Filter` waveform should be the maximum length of any element of the sums. Each of these elements is itself a product whose length should be the *minimum* of their respective lengths. However, in most cases the coefficient waveforms will be infinite, and tracking the lengths of each coefficient waveform would add significant complexity to the implementation. As such, Tuun assumes that the coefficient waveforms are at least as long as the input waveform and extends them with $0.0$ if they are not.

Finally, note that because the length of a `Filter` waveform is independent of the coefficients (including their values), even an *infinite* impulse response filter could have a *finite* length. Similarly, a *finite* impulse response filter could have an *infinite* length.

<script type="module" src="tuun/tuun-synth.js"></script>
