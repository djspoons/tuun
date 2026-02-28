import { initSync, Tuun } from './pkg/tuun.js';

class TuunProcessor extends AudioWorkletProcessor {
    constructor(options) {
        super();
        const { wasmModule, sampleRate, bufferSize } = options.processorOptions;
        this.bufferSize = bufferSize || 1024;
        this.waveform = null;
        this.playing = false;
        this.finished = false;
        this.buffer = null;
        this.bufferOffset = 0;

        // Initialize WASM synchronously from the pre-compiled module
        initSync({module: wasmModule});
        this.tuun = new Tuun(sampleRate || 44100);
        this.port.postMessage({ type: 'ready' });

        this.port.onmessage = (event) => {
            switch (event.data.type) {
                case 'play':
                    this._play(event.data.expression);
                    break;
                case 'stop':
                    this._stop();
                    break;
            }
        };
    }

    _play(expression) {
        this._stop();
        try {
            this.waveform = this.tuun.parse(expression);
            this.playing = true;
            this.finished = false;
            this.buffer = null;
            this.bufferOffset = 0;
        } catch (e) {
            this.port.postMessage({ type: 'error', message: e.toString() });
        }
    }

    _stop() {
        this.playing = false;
        this.finished = false;
        if (this.waveform) {
            this.waveform.free();
            this.waveform = null;
        }
        this.buffer = null;
        this.bufferOffset = 0;
    }

    process(inputs, outputs) {
        const output = outputs[0][0];

        if (!this.playing || !this.waveform) {
            output.fill(0);
            return true;
        }

        try {
            let written = 0;

            while (written < output.length) {
                if (!this.buffer || this.bufferOffset >= this.buffer.length) {
                    if (this.finished) {
                        output.fill(0, written);
                        this._stop();
                        this.port.postMessage({ type: 'ended' });
                        return true;
                    }

                    this.buffer = this.tuun.generate(this.waveform, this.bufferSize);
                    this.bufferOffset = 0;

                    if (this.buffer.length < this.bufferSize) {
                        this.finished = true;
                    }

                    if (this.buffer.length === 0) {
                        output.fill(0, written);
                        this._stop();
                        this.port.postMessage({ type: 'ended' });
                        return true;
                    }
                }

                const available = this.buffer.length - this.bufferOffset;
                const needed = output.length - written;
                const count = Math.min(available, needed);

                output.set(
                    this.buffer.subarray(this.bufferOffset, this.bufferOffset + count),
                    written
                );

                this.bufferOffset += count;
                written += count;
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
