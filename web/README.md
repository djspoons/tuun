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
import init, { Tuun } from './pkg/tuun.js';

// Initialize the WASM module
await init();

// Create a Tuun instance with a sample rate
const tuun = new Tuun(44100);

// Parse an expression into a waveform
const waveform = tuun.parse("sin(440, 0)");

// Generate audio samples
const samples = tuun.generate(waveform, 4096);
// samples is a Float32Array that can be used with Web Audio API
```

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
