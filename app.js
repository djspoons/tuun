import init, { WasmGenerator } from './pkg/tuun.js';

class TuunSynth {
    constructor() {
        this.generator = null;
        this.audioContext = null;
        this.currentSource = null;
        this.isPlaying = false;
    }

    async initialize(sampleRate = 44100) {
        await init();
        this.generator = new WasmGenerator(sampleRate);
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: sampleRate
        });
    }

    async play(expression, duration) {
        this.stop();

        try {
            const waveform = this.generator.parse(expression);
            const numSamples = Math.floor(duration * this.audioContext.sampleRate);
            const samples = this.generator.generate(waveform, numSamples);

            const audioBuffer = this.audioContext.createBuffer(1, samples.length, this.audioContext.sampleRate);
            audioBuffer.getChannelData(0).set(samples);

            this.currentSource = this.audioContext.createBufferSource();
            this.currentSource.buffer = audioBuffer;
            this.currentSource.connect(this.audioContext.destination);

            this.currentSource.onended = () => {
                this.isPlaying = false;
                this.currentSource = null;
                updatePlayButton();
            };

            if (this.audioContext.state === 'suspended') {
                await this.audioContext.resume();
            }

            this.currentSource.start();
            this.isPlaying = true;
        } catch (error) {
            this.isPlaying = false;
            throw error;
        }
    }

    stop() {
        if (this.currentSource) {
            try {
                this.currentSource.stop();
            } catch (e) {
                // Already stopped
            }
            this.currentSource = null;
        }
        this.isPlaying = false;
    }

    async changeSampleRate(newSampleRate) {
        this.stop();
        if (this.audioContext) {
            await this.audioContext.close();
        }
        await this.initialize(newSampleRate);
    }
}

const DEFAULT_DURATION = 2;
const DEFAULT_SAMPLE_RATE = 44100;

let synth = null;
let expressionInput, playToggle, durationInput, sampleRateSelect, errorDiv;

function getDuration() {
    return durationInput ? parseFloat(durationInput.value) || DEFAULT_DURATION : DEFAULT_DURATION;
}

function getSampleRate() {
    return sampleRateSelect ? parseInt(sampleRateSelect.value) || DEFAULT_SAMPLE_RATE : DEFAULT_SAMPLE_RATE;
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
        playToggle.textContent = '⏹ Stop';
        playToggle.classList.add('playing');
    } else {
        playToggle.textContent = '▶ Play';
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
        if (!expression) {
            showError('Please enter an expression');
            return;
        }

        try {
            hideError();
            await synth.play(expression, getDuration());
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
        // Get required DOM elements
        expressionInput = document.getElementById('expression');
        playToggle = document.getElementById('play-toggle');
        errorDiv = document.getElementById('error');

        if (!expressionInput || !playToggle || !errorDiv) {
            console.error('Missing required DOM elements (expression, play-toggle, or error)');
            return;
        }

        // Get optional control elements
        durationInput = document.getElementById('duration');
        sampleRateSelect = document.getElementById('sample-rate');

        // Hide controls if data-hide-controls is set on the container
        const container = document.querySelector('.editor-section');
        if (container && container.hasAttribute('data-hide-controls')) {
            document.querySelectorAll('.control-group').forEach(el => el.style.display = 'none');
        }

        synth = new TuunSynth();
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
        if (errorDiv) {
            showError('Failed to initialize: ' + error.message);
        }
    }
}

// Always wait for DOM to be ready
document.addEventListener('DOMContentLoaded', initApp);
