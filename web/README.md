# Tuun Web Synthesizer

This directory contains the WebAssembly build of the Tuun music synthesizer, allowing it to run in web browsers.

## Quick Start

1. Build the WASM package from the project root:
   ```bash
   ./build-wasm.sh
   ```

2. Start a local web server:
   ```bash
   cd web && python3 -m http.server 8080
   ```

3. Open http://localhost:8080 in your browser

## JavaScript API

The WASM module exposes a JavaScript API for integrating Tuun into web applications.

### Basic Usage

```javascript
import init, { WasmGenerator } from './pkg/tuun.js';

// Initialize the WASM module
await init();

// Create a generator with a sample rate
const generator = new WasmGenerator(44100);

// Parse an expression into a waveform
const waveform = generator.parse("$440 * Qw");

// Generate audio samples
const samples = generator.generate(waveform, 4096);
// samples is a Float32Array that can be used with Web Audio API
```

### API Reference

#### `WasmGenerator`

The main interface for parsing expressions and generating audio samples.

**Constructor:**
```javascript
new WasmGenerator(sampleRate: number): WasmGenerator
```
- `sampleRate`: Audio sample rate in Hz (e.g., 44100, 48000)
- Throws an error if initialization fails

**Methods:**

##### `parse(expression: string): WasmWaveform`
Parses a Tuun expression and returns a waveform.
- `expression`: The Tuun expression string to parse
- Returns: A `WasmWaveform` object
- Throws: Parse error message if the expression is invalid

Example:
```javascript
const wf = generator.parse("sin(440 * 2 * pi * t)");
```

##### `parse_and_optimize(expression: string): WasmWaveform`
Parses and optimizes a Tuun expression.
- Performs constant folding and algebraic simplification
- Returns: An optimized `WasmWaveform` object

Example:
```javascript
const wf = generator.parse_and_optimize("2 * 220"); // Optimizes to 440
```

##### `generate(waveform: WasmWaveform, numSamples: number): Float32Array`
Generates audio samples from a waveform.
- `waveform`: The waveform to generate from (will be mutated to track state)
- `numSamples`: Number of samples to generate
- Returns: Float32Array of audio samples in range [-1.0, 1.0]

Example:
```javascript
const samples = generator.generate(waveform, 4096);
```

##### `WasmGenerator.get_builtins(): string` (static)
Returns a string describing all available builtin functions and constants.

Example:
```javascript
const help = WasmGenerator.get_builtins();
console.log(help);
```

**Properties:**

##### `sample_rate: number` (read-only)
The sample rate of the generator.

#### `WasmWaveform`

Represents a parsed waveform ready for audio generation. This object maintains internal state for time-dependent waveforms.

**Methods:**

##### `toString(): string`
Returns a debug string representation of the waveform.

## Web Audio Integration

Here's a complete example of integrating Tuun with the Web Audio API:

```javascript
import init, { WasmGenerator } from './pkg/tuun.js';

class AudioPlayer {
    constructor(sampleRate = 44100) {
        this.generator = null;
        this.audioContext = null;
        this.sampleRate = sampleRate;
    }

    async initialize() {
        await init();
        this.generator = new WasmGenerator(this.sampleRate);
        this.audioContext = new AudioContext({ sampleRate: this.sampleRate });
    }

    async play(expression, duration = 2.0) {
        // Parse the expression
        const waveform = this.generator.parse(expression);

        // Calculate samples needed
        const numSamples = Math.floor(duration * this.sampleRate);

        // Generate samples
        const samples = this.generator.generate(waveform, numSamples);

        // Create audio buffer
        const audioBuffer = this.audioContext.createBuffer(
            1, // mono
            samples.length,
            this.sampleRate
        );

        // Copy samples to audio buffer
        audioBuffer.getChannelData(0).set(samples);

        // Create and connect source
        const source = this.audioContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(this.audioContext.destination);

        // Play
        source.start();
    }
}

// Usage
const player = new AudioPlayer(44100);
await player.initialize();
await player.play("$440 * Qw", 2.0); // Play quarter wave for 2 seconds
```

## Expression Language

Tuun expressions support:

### Constants
- `pi` - Mathematical constant π (3.14159...)
- `e` - Mathematical constant e (2.71828...)
- `t` - Time variable (in seconds)

### Waveform Constructors
- `$freq` - Creates a waveform at the given frequency (e.g., `$440`)
- `Qw` - Quarter wave
- `Hw` - Half wave
- `Ww` - Whole wave

### Math Functions
- `sin(x)`, `cos(x)`, `tan(x)` - Trigonometric functions
- `abs(x)` - Absolute value
- `sqrt(x)` - Square root
- `floor(x)`, `ceil(x)`, `round(x)` - Rounding
- `min(a, b)`, `max(a, b)` - Min/max
- `pow(base, exp)` - Power
- `ln(x)`, `log2(x)`, `log10(x)` - Logarithms

### Signal Processing
- `noise()` - White noise generator
- `lpf(cutoff, waveform)` - Low-pass filter
- `hpf(cutoff, waveform)` - High-pass filter
- `delay(time, waveform)` - Time delay
- `mix(w1, w2, ...)` - Mix multiple waveforms

### Operators
- `+`, `-`, `*`, `/` - Arithmetic
- `%` - Modulo
- `^` - Exponentiation

### Example Expressions

```javascript
// Sine wave at 440Hz (A4 note)
"sin(440 * 2 * pi * time)"

// Quieter sine wave
"sin(440 * 2 * pi * time) * 0.5"

// White noise
"noise * 0.1"

// Higher frequency sine (880Hz - A5)
"sin(880 * 2 * pi * time)"

// Two tones combined
"sin(440 * 2 * pi * time) + sin(554 * 2 * pi * time)"

// Vibrato effect (frequency modulation)
"sin((440 + sin(5 * 2 * pi * time) * 20) * 2 * pi * time)"

// Amplitude modulation
"sin(440 * 2 * pi * time) * (1 + sin(2 * 2 * pi * time)) / 2"
```

**Note:** Advanced features like `$`, `Qw`, `Hw`, `Ww` require loading the full context.tuun file,
which is not currently supported in the web version. The examples above use only the core builtins.

## Browser Compatibility

The web version requires:
- Chrome 66+ or Edge 79+
- Firefox 60+
- Safari 11.1+

All browsers must support:
- WebAssembly
- Web Audio API
- ES6 modules

## Known Limitations

Compared to the native version, the web version currently lacks:

1. **Interactive sliders** - No mouse-based X/Y parameter control yet
2. **File I/O** - Cannot save/load waveforms to files
3. **Real-time visualization** - No waveform or spectrum display
4. **Multi-program management** - Single expression only

These features may be added in future updates.

## Development

To modify the web interface:

1. Edit [index.html](index.html), [app.js](app.js), or [style.css](style.css)
2. Refresh your browser (no rebuild needed unless you changed Rust code)

To modify the Rust WASM interface:

1. Edit [../src/lib/wasm.rs](../src/lib/wasm.rs)
2. Rebuild: `./build-wasm.sh`
3. Refresh your browser

## Troubleshooting

**"Failed to initialize" error:**
- Make sure you've built the WASM package with `./build-wasm.sh`
- Check that the `pkg/` directory exists and contains `tuun_bg.wasm` and `tuun.js`
- Ensure you're serving via HTTP, not opening the file directly (file:// protocol doesn't work)

**No sound:**
- Check browser console for errors
- Ensure your browser's audio isn't muted
- Try clicking the page first (some browsers require user interaction before audio)

**Parse errors:**
- Check the expression syntax matches the Tuun language
- Review the "Available Functions & Constants" section in the UI
- Look at the example expressions for reference

## License

Same as the main Tuun project.
