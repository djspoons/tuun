class TuunSynth {
    constructor(bufferSize = 1024) {
        this.audioContext = null;
        this.workletNode = null;
        this.isPlaying = false;
        this.bufferSize = bufferSize;
    }

    async initialize(sampleRate = 44100) {
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: sampleRate
        });

        const wasmResponse = await fetch('./pkg/tuun_bg.wasm');
        const wasmBytes = await wasmResponse.arrayBuffer();
        const wasmModule = await WebAssembly.compile(wasmBytes);

        await this.audioContext.audioWorklet.addModule('tuun-processor.js');

        this.workletNode = new AudioWorkletNode(this.audioContext, 'tuun-processor', {
            processorOptions: { wasmModule, bufferSize: this.bufferSize, sampleRate }
        });

        // Wait for the worklet to finish WASM init
        await new Promise((resolve, reject) => {
            this.workletNode.port.onmessage = (event) => {
                if (event.data.type === 'ready') resolve();
                else if (event.data.type === 'error') reject(new Error(event.data.message));
            };
        });

        // Replace with ongoing handler
        this.workletNode.port.onmessage = (event) => {
            switch (event.data.type) {
                case 'ended':
                    this.isPlaying = false;
                    updatePlayButton();
                    break;
                case 'error':
                    this.isPlaying = false;
                    showError(event.data.message);
                    updatePlayButton();
                    break;
            }
        };

        this.workletNode.connect(this.audioContext.destination);
    }

    async play(expression) {
        if (this.audioContext.state === 'suspended') {
            await this.audioContext.resume();
        }
        this.workletNode.port.postMessage({ type: 'play', expression });
        this.isPlaying = true;
    }

    stop() {
        if (this.workletNode) {
            this.workletNode.port.postMessage({ type: 'stop' });
        }
        this.isPlaying = false;
    }

    async changeSampleRate(newSampleRate) {
        this.stop();
        if (this.workletNode) {
            this.workletNode.disconnect();
            this.workletNode = null;
        }
        if (this.audioContext) { await this.audioContext.close(); }
        await this.initialize(newSampleRate);
    }
}

const DEFAULT_SAMPLE_RATE = 44100;
const DEFAULT_BUFFER_SIZE = 1024;

let synth = null;
let expressionInput, playToggle, sampleRateSelect, errorDiv;

function getSampleRate() {
    return sampleRateSelect ? parseInt(sampleRateSelect.value) || DEFAULT_SAMPLE_RATE : DEFAULT_SAMPLE_RATE;
}

function getBufferSize() {
    const container = document.querySelector('.editor-section');
    if (container && container.dataset.bufferSize) {
        const size = parseInt(container.dataset.bufferSize);
        if (size >= 128 && size <= 16384) {
            return size;
        }
    }
    return DEFAULT_BUFFER_SIZE;
}

function showError(message) {
    errorDiv.textContent = message;
    errorDiv.classList.remove('hidden');
}

function hideError() {
    errorDiv.classList.add('hidden');
}

function updatePlayButton() {
    if (synth && synth.isPlaying) {
        playToggle.textContent = '\u23F9';
        playToggle.classList.add('playing');
    } else {
        playToggle.textContent = '\u25B6';
        playToggle.classList.remove('playing');
    }
}

async function handlePlayToggle() {
    if (synth.isPlaying) {
        synth.stop();
        updatePlayButton();
        hideError();
    } else {
        const expression = expressionInput.value.trim();
        if (!expression) { showError('Please enter an expression'); return; }

        try {
            hideError();
            await synth.play(expression);
            updatePlayButton();
        } catch (error) {
            showError(error.message || 'Error playing audio');
            updatePlayButton();
        }
    }
}

async function handleSampleRateChange() {
    try {
        await synth.changeSampleRate(getSampleRate());
    } catch (error) {
        showError('Failed to change sample rate');
    }
}

async function initApp() {
    try {
        expressionInput = document.getElementById('expression');
        playToggle = document.getElementById('play-toggle');
        errorDiv = document.getElementById('error');

        if (!expressionInput || !playToggle || !errorDiv) {
            console.error('Missing required DOM elements');
            return;
        }

        sampleRateSelect = document.getElementById('sample-rate');

        const container = document.querySelector('.editor-section');
        if (container && container.hasAttribute('data-hide-controls')) {
            document.querySelectorAll('.control-group').forEach(el => el.style.display = 'none');
        }

        synth = new TuunSynth(getBufferSize());
        await synth.initialize(getSampleRate());

        playToggle.addEventListener('click', handlePlayToggle);
        if (sampleRateSelect) {
            sampleRateSelect.addEventListener('change', handleSampleRateChange);
        }

        expressionInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                handlePlayToggle();
            }
        });

        updatePlayButton();
    } catch (error) {
        console.error('Failed to initialize:', error);
        if (errorDiv) { showError('Failed to initialize: ' + error.message); }
    }
}

document.addEventListener('DOMContentLoaded', initApp);
