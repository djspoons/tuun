import { initSync, Tuun } from './pkg/tuun.js';

class TuunProcessor extends AudioWorkletProcessor {
    constructor(options) {
        super();
        const { wasmModule, sampleRate, tempo } = options.processorOptions;
        this.waveform = null;
        this.playing = false;

        // Initialize WASM synchronously from the pre-compiled module
        initSync({module: wasmModule});
        this.tuun = new Tuun(sampleRate || 44100, tempo || 120);
        this.port.postMessage({ type: 'ready' });

        this.port.onmessage = (event) => {
            switch (event.data.type) {
                case 'play':
                    this._play(event.data.expression);
                    break;
                case 'stop':
                    this._stop();
                    break;
                case 'slider':
                    this.tuun.set_slider_value(event.data.name, event.data.value);
                    break;
            }
        };
    }

    _play(expression) {
        this._stop();
        try {
            this.waveform = this.tuun.parse(expression);
            this.playing = true;
        } catch (e) {
            this.port.postMessage({ type: 'error', message: e.toString() });
        }
    }

    _stop() {
        this.playing = false;
        if (this.waveform) {
            this.waveform.free();
            this.waveform = null;
        }
    }

    process(inputs, outputs) {
        const output = outputs[0][0];

        if (!this.playing || !this.waveform) {
            output.fill(0);
            return true;
        }

        try {
            const samples = this.tuun.generate(this.waveform, output.length);

            /* TODO remove this?
            if (samples.length === 0) {
                output.fill(0);
                this._stop();
                this.port.postMessage({ type: 'ended' });
                return true;
            }
            */

            output.set(samples.subarray(0, Math.min(samples.length, output.length)));

            if (samples.length < output.length) {
                output.fill(0, samples.length);
                this._stop();
                this.port.postMessage({ type: 'ended' });
            }
        } catch (e) {
            output.fill(0);
            this._stop();
            this.port.postMessage({ type: 'error', message: 'Generation failed: ' + e });
        }

        return true;
    }
}

registerProcessor('tuun-processor', TuunProcessor);
