import { initSync, Tuun } from './pkg/tuun.js';

class TuunProcessor extends AudioWorkletProcessor {
    constructor(options) {
        super();
        const { wasmModule, sampleRate, tempo } = options.processorOptions;
        this.playing = false;

        // Initialize WASM synchronously from the pre-compiled module
        initSync({module: wasmModule});
        this.tuun = new Tuun(sampleRate || 44100, tempo || 120);
        this.port.postMessage({ type: 'ready' });

        this.port.onmessage = (event) => {
            //console.log("tuun-processor: got '" + event.data.type + "' message with data: " + JSON.stringify(event.data));
            switch (event.data.type) {
                case 'play':
                    try {
                        this.tuun.parse(
                            event.data.expression,
                            event.data.sliders,
                        );
                        this.playing = true;
                    } catch (e) {
                        this.port.postMessage({ type: 'error', message: e.toString() });
                    }
                    break;
                case 'stop':
                    this.tuun.stop();
                    this.playing = false;
                    break;
                case 'update_sliders':
                    for (const [name, value] of Object.entries(event.data.values)) {
                        this.tuun.update_slider(name, value);
                    }
                    break;
            }
        };
    }

    process(inputs, outputs) {
        const output = outputs[0][0];
        if (!this.playing) {
            output.fill(0);
            return true;
        }

        try {
            const more = this.tuun.process(output);

            if (!more) {
                this.tuun.stop();
                this.playing = false;
                this.port.postMessage({ type: 'ended' });
            }
        } catch (e) {
            output.fill(0);
            this.tuun.stop();
            this.playing = false;
            this.port.postMessage({ type: 'error', message: 'Generation failed: ' + e });
        }

        return true;
    }
}

registerProcessor('tuun-processor', TuunProcessor);
