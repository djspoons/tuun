import { initSync, Tuun, sliderValue, sliderPosition } from './pkg/tuun.js';

// ─── Shared Runtime (singleton across all <tuun-synth> instances) ─

class TuunRuntime {
    constructor() {
        this.wasmModule = null;
        this.audioContext = null;
        this.tuun = null;
        this.activeInstance = null;
        this._initPromise = null;
        this._sampleRate = null;
        this._tempo = null;
        // An instance with no audio, for reading the sliders of expressions.
        this._checker = null;

        // Start loading the WASM module immediately so components can read
        // their sliders before anything plays.
        this._wasmReady = this._loadWasm();
    }

    async _loadWasm() {
        const wasmUrl = new URL('./pkg/tuun_bg.wasm', import.meta.url);
        const resp = await fetch(wasmUrl);
        const bytes = await resp.arrayBuffer();
        this.wasmModule = await WebAssembly.compile(bytes);
        initSync({ module: this.wasmModule });
    }

    async ensureInitialized(sampleRate = 44100, tempo = 120) {
        const needsInit = this._sampleRate !== sampleRate
            || this._tempo !== tempo
            || !this._initPromise;

        if (needsInit) {
            this._sampleRate = sampleRate;
            this._tempo = tempo;
            this._initPromise = this._doInit(sampleRate, tempo).catch(err => {
                this._initPromise = null;
                throw err;
            });
        }

        return this._initPromise;
    }

    async _doInit(sampleRate, tempo) {
        await this._wasmReady;

        if (this.audioContext) {
            this.stopActive();
            await this.audioContext.close();
        }

        this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate
        });
        this.tuun = new Tuun(sampleRate, tempo);

        const processorUrl = new URL('./tuun-processor.js', import.meta.url);
        await this.audioContext.audioWorklet.addModule(processorUrl);
    }

    // Returns the sliders `expression` reaches, as `{label, default, min, max,
    // curve}` objects in the order it first reaches them, evaluating it without
    // audio. Throws the expression's error.
    async sliders(expression, opens, uses) {
        await this._wasmReady;
        // TODO this reads sliders at tempo 120, so a default computed from
        // `tempo` shows the wrong value until the expression plays.
        if (!this._checker) this._checker = new Tuun(44100, 120);
        return JSON.parse(this._checker.sliders(expression, opens, uses));
    }

    // Checks that `expression` installs, with its sliders starting at `values`
    // (a JSON object of label to value), and returns the sliders it reaches as
    // `sliders` does. Throws the expression's error.
    install(expression, values, opens, uses) {
        const sliders = JSON.parse(this.tuun.install(expression, values, opens, uses));
        // The worklet plays it; this copy discards its state.
        this.tuun.stop();
        return sliders;
    }

    createWorkletNode() {
        return new AudioWorkletNode(this.audioContext, 'tuun-processor', {
            processorOptions: {
                wasmModule: this.wasmModule,
                sampleRate: this._sampleRate,
                tempo: this._tempo
            }
        });
    }

    setPlaying(instance) {
        if (this.activeInstance && this.activeInstance !== instance) {
            this.activeInstance.stop();
        }
        this.activeInstance = instance;
    }

    stopActive() {
        if (this.activeInstance) {
            this.activeInstance.stop();
            this.activeInstance = null;
        }
    }

    clearActive(instance) {
        if (this.activeInstance === instance) {
            this.activeInstance = null;
        }
    }
}

const runtime = new TuunRuntime();

// ─── Shadow DOM Styles ───────────────────────────────────────────

const STYLES = `
:host {
    display: block;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

.tuun-synth {
    background: white;
    border-radius: 8px;
    padding: 12px;
    margin: 0.8em;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.main-row {
    display: flex;
    gap: 8px;
    align-items: baseline;
}

.description {
    flex: 1;
    font-size: 14px;
    color: #333;
    display: flex;
    align-items: center;
}

.editor-row {
    display: flex;
    gap: 8px;
    align-items: start;
    margin-top: 8px;
    padding-left: 40px;
}

textarea {
    flex: 1;
    padding: 8px 12px;
    border: 1px solid #ddd;
    border-radius: 4px;
    font-family: 'Monaco', 'Menlo', monospace;
    field-sizing: content;
    font-size: 14px;
    resize: vertical;
    transition: border-color 0.2s;
}

textarea:focus {
    outline: none;
    border-color: #667eea;
}

.main-row textarea {
    flex: 1;
}

.play-button,
.reset-button,
.code-toggle {
    width: 32px;
    min-height: 37px;
    padding: 0;
    border: none;
    border-radius: 4px;
    font-size: 16px;
    font-weight: 600;
    line-height: 1;
    cursor: pointer;
    transition: background 0.2s;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}

.play-button {
    background: #667eea;
    color: white;
}

.play-button:hover {
    background: #5568d3;
}

.play-button.playing {
    background: #dc3545;
}

.play-button.playing:hover {
    background: #c82333;
}

.reset-button {
    background: #e9ecef;
    color: #495057;
    font-size: 1.4em;
}

.reset-button:hover:not(:disabled) {
    background: #dee2e6;
}

.reset-button:disabled {
    opacity: 0.35;
    cursor: default;
}

.code-toggle {
    background: none;
    font-size: 1.5em;
    color: #999;
    transform: rotate(270deg);
    transition: transform 0.2s, color 0.2s;
}

.code-toggle:hover {
    color: #555;
    background: none;
}

.code-toggle.active {
    transform: rotate(90deg);
    color: #667eea;
    background: none;
}

.code-toggle.active:hover {
    color: #5568d3;
    background: none;
}

.controls {
    display: flex;
    gap: 12px;
    align-items: center;
    margin-top: 8px;
    padding-left: 40px;
    font-size: 14px;
}

.controls label {
    display: flex;
    align-items: center;
    gap: 6px;
    font-weight: 500;
    color: #666;
}

select {
    padding: 4px 6px;
    border: 1px solid #ddd;
    border-radius: 4px;
    font-size: 14px;
    background: white;
}

select:focus {
    outline: none;
    border-color: #667eea;
}

input[type="number"] {
    width: 70px;
    padding: 4px 6px;
    border: 1px solid #ddd;
    border-radius: 4px;
    font-size: 14px;
}

input[type="number"]:focus {
    outline: none;
    border-color: #667eea;
}

.error {
    margin-top: 8px;
    padding: 8px 10px;
    background: #fee;
    border-left: 3px solid #f44;
    border-radius: 4px;
    color: #c33;
    font-size: 13px;
    font-family: monospace;
}

.sliders {
    display: flex;
    flex-direction: column;
    gap: 6px;
    margin-top: 8px;
    padding-left: 40px;
}

.slider-row {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 14px;
}

.slider-row label {
    min-width: 60px;
    font-weight: 500;
    color: #666;
}

.slider-row input[type="range"] {
    flex: 1;
    accent-color: #667eea;
}

.slider-row .slider-value {
    min-width: 50px;
    text-align: right;
    font-family: 'Monaco', 'Menlo', monospace;
    font-size: 13px;
    color: #555;
}

.hidden {
    display: none !important;
}
`;

// ─── <tuun-synth> Custom Element ─────────────────────────────────

class TuunSynthElement extends HTMLElement {
    static observedAttributes = ['expression', 'description', 'controls', 'expanded', 'tempo', 'open', 'use'];

    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
        this._isPlaying = false;
        this._expanded = false;
        this._originalExpression = '';
        this._workletNode = null;
        // One control per slider the expression reached when it last played
        // (or, before that, the original expression): `{label, default, min,
        // max, curve, value}`, with the value in the slider's own units.
        this._sliders = [];
        // The controls for the original expression, at their defaults, which
        // the reset button restores.
        this._originalSliders = [];
    }

    async connectedCallback() {
        const script = this.querySelector('script[type="text/tuun"]');
        const bodyText = script ? script.textContent : this.textContent;
        this._originalExpression = this._dedent(bodyText) || this.getAttribute('expression') || '';
        this._expanded = !this.getAttribute('description') || this.hasAttribute('expanded');
        this._originalSliders = await this._slidersOf(this._originalExpression);
        this._sliders = this._originalSliders.map(s => ({ ...s }));
        this._render();
        this._bindEvents();
    }

    disconnectedCallback() {
        if (this._workletNode) {
            this._workletNode.disconnect();
            this._workletNode = null;
        }
        runtime.clearActive(this);
    }

    attributeChangedCallback(name, oldValue, newValue) {
        if (!this.shadowRoot.querySelector('.tuun-synth')) return;

        if (name === 'expression') {
            this._originalExpression = newValue || '';
            const ta = this.shadowRoot.querySelector('textarea');
            if (ta) ta.value = newValue || '';
            this._slidersOf(this._originalExpression).then(sliders => {
                this._originalSliders = sliders;
                this._resetSliders();
                this._updateResetButton();
            });
        } else if (name === 'description') {
            // Layout changes — re-render
            this._expanded = !newValue;
            this._render();
            this._bindEvents();
        } else if (name === 'controls') {
            const ctrl = this.shadowRoot.querySelector('.controls');
            if (ctrl) ctrl.classList.toggle('hidden', !this.hasAttribute('controls'));
        }
    }

    _render() {
        const expression = this._escapeHtml(this._originalExpression);
        const description = this.getAttribute('description') || '';
        const hasDescription = !!description;
        const hasControls = this.hasAttribute('controls');

        if (hasDescription) {
            this.shadowRoot.innerHTML = `
                <style>${STYLES}</style>
                <div class="tuun-synth">
                    <div class="main-row">
                        <button class="play-button" title="Play/Stop">&#x25B6;</button>
                        <span class="description">${this._escapeHtml(description)}</span>
                        <button class="code-toggle ${this._expanded ? 'active' : ''}" title="Show/hide expression">&#10095;</button>
                    </div>
                    <div class="editor-row ${this._expanded ? '' : 'hidden'}">
                        <textarea rows="1">${expression}</textarea>
                        <button class="reset-button" title="Reset expression and sliders" disabled>&#x21BA;</button>
                    </div>
                    <div class="sliders"></div>
                    <div class="controls ${hasControls ? '' : 'hidden'}">
                        <label>Tempo:
                            <input type="number" class="tempo" value="${this._getTempo()}" min="1" max="999" step="1">
                        </label>
                        <label>Sample Rate:
                            <select class="sample-rate">
                                <option value="22050">22050 Hz</option>
                                <option value="44100" selected>44100 Hz</option>
                                <option value="48000">48000 Hz</option>
                            </select>
                        </label>
                    </div>
                    <div class="error hidden"></div>
                </div>`;
        } else {
            this.shadowRoot.innerHTML = `
                <style>${STYLES}</style>
                <div class="tuun-synth">
                    <div class="main-row">
                        <button class="play-button" title="Play/Stop">&#x25B6;</button>
                        <textarea rows="1">${expression}</textarea>
                        <button class="reset-button" title="Reset expression and sliders" disabled>&#x21BA;</button>
                    </div>
                    <div class="sliders"></div>
                    <div class="controls ${hasControls ? '' : 'hidden'}">
                        <label>Sample Rate:
                            <select class="sample-rate">
                                <option value="22050">22050 Hz</option>
                                <option value="44100" selected>44100 Hz</option>
                                <option value="48000">48000 Hz</option>
                            </select>
                        </label>
                        <label>Tempo:
                            <input type="number" class="tempo" value="${this._getTempo()}" min="1" max="999" step="1">
                        </label>
                    </div>
                    <div class="error hidden"></div>
                </div>`;
        }
    }

    _bindEvents() {
        const playBtn = this.shadowRoot.querySelector('.play-button');
        const codeToggle = this.shadowRoot.querySelector('.code-toggle');
        const textarea = this.shadowRoot.querySelector('textarea');
        const resetBtn = this.shadowRoot.querySelector('.reset-button');
        const sampleRate = this.shadowRoot.querySelector('.sample-rate');

        playBtn.addEventListener('click', () => this._handlePlayToggle());

        if (codeToggle) {
            codeToggle.addEventListener('click', () => this._toggleExpanded());
        }

        if (textarea) {
            textarea.addEventListener('input', () => this._updateResetButton());
            textarea.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && (e.altKey || e.ctrlKey || e.metaKey)) {
                    e.preventDefault();
                    this._handlePlayToggle();
                    playBtn.focus();
                }
            });
        }

        if (resetBtn) {
            resetBtn.addEventListener('click', () => {
                if (textarea) {
                    textarea.value = this._originalExpression;
                }
                this._resetSliders();
                this._hideError();
                this._updateResetButton();
            });
        }

        if (sampleRate) {
            sampleRate.addEventListener('change', () => this._handleSampleRateChange());
        }

        const tempo = this.shadowRoot.querySelector('.tempo');
        if (tempo) {
            tempo.addEventListener('change', () => this._handleTempoChange());
        }

        this._drawSliders();
        this._updateResetButton();
    }

    _toggleExpanded() {
        this._expanded = !this._expanded;
        const editorRow = this.shadowRoot.querySelector('.editor-row');
        const codeToggle = this.shadowRoot.querySelector('.code-toggle');
        if (editorRow) editorRow.classList.toggle('hidden', !this._expanded);
        if (codeToggle) codeToggle.classList.toggle('active', this._expanded);
    }

    async _handlePlayToggle() {
        if (this._isPlaying) {
            this.stop();
            return;
        }

        const textarea = this.shadowRoot.querySelector('textarea');
        const expression = textarea
            ? textarea.value.trim()
            : (this.getAttribute('expression') || '').trim();

        if (!expression) {
            this._showError('Please enter an expression');
            return;
        }

        try {
            this._hideError();
            await this._play(expression);
        } catch (error) {
            this._showError(error.message || String(error) || 'Error playing audio');
            this._updatePlayButton();
        }
    }

    async _play(expression) {
        expression = this._stripComments(expression);
        await runtime.ensureInitialized(this._getSampleRate(), this._getTempo());
        const values = JSON.stringify(Object.fromEntries(this._sliders.map(s => [s.label, s.value])));
        const opens = this._getOpens();
        const uses = this._getUses();
        // First, check that the expression installs (parses and
        // evaluates) before involving the worklet.
        this._updateSliders(runtime.install(expression, values, opens, uses));
        // Claim our status as the active instance.
        runtime.setPlaying(this);

        await this._ensureWorkletNode();

        if (runtime.audioContext.state === 'suspended') {
            await runtime.audioContext.resume();
        }

        // Send a message to the processor to start playing.
        this._workletNode.port.postMessage({
            type: 'play',
            expression,
            values,
            opens,
            uses,
        });
        this._isPlaying = true;
        this._updatePlayButton();
    }

    // The `open` attribute is a JSON array of dotted module paths to bring
    // into scope (e.g. `open='["std"]'`). Passed through verbatim to
    // `Tuun.parse`; an absent or empty attribute means no opens.
    _getOpens() {
        const attr = this.getAttribute('open');
        return attr && attr.trim() ? attr : '[]';
    }

    // The `use` attribute is a JSON array of dotted module paths to bind as
    // module values (e.g. `use='["synth.pm"]'`); each is bound to its last
    // path component and accessed as `<name>.<binding>`. Passed through
    // verbatim; an absent or empty attribute means no uses.
    _getUses() {
        const attr = this.getAttribute('use');
        return attr && attr.trim() ? attr : '[]';
    }

    async _ensureWorkletNode() {
        if (this._workletNode && this._workletNode.context === runtime.audioContext) {
            return;
        }

        if (this._workletNode) {
            this._workletNode.disconnect();
        }

        this._workletNode = runtime.createWorkletNode();

        await new Promise((resolve, reject) => {
            this._workletNode.port.onmessage = (event) => {
                if (event.data.type === 'ready') resolve();
                else if (event.data.type === 'error') reject(new Error(event.data.message));
            };
        });

        this._workletNode.port.onmessage = (event) => {
            switch (event.data.type) {
                case 'ended':
                    this._isPlaying = false;
                    this._updatePlayButton();
                    runtime.clearActive(this);
                    break;
                case 'error':
                    this._isPlaying = false;
                    this._showError(event.data.message);
                    this._updatePlayButton();
                    runtime.clearActive(this);
                    break;
            }
        };

        this._workletNode.connect(runtime.audioContext.destination);
    }

    stop() {
        if (this._workletNode) {
            this._workletNode.port.postMessage({ type: 'stop' });
        }
        if (this._batchTimerId) {
            clearTimeout(this._batchTimerId);
            this._batchTimerId = null;
        }
        if (this._pendingSliderUpdates) {
            this._pendingSliderUpdates.clear();
        }
        this._isPlaying = false;
        this._updatePlayButton();
        runtime.clearActive(this);
    }

    _getSampleRate() {
        const select = this.shadowRoot.querySelector('.sample-rate');
        return select ? parseInt(select.value) || 44100 : 44100;
    }

    async _handleSampleRateChange() {
        this.stop();
        if (this._workletNode) {
            this._workletNode.disconnect();
            this._workletNode = null;
        }
        try {
            await runtime.ensureInitialized(this._getSampleRate(), this._getTempo());
        } catch (error) {
            this._showError('Failed to change sample rate');
        }
    }

    _getTempo() {
        const input = this.shadowRoot.querySelector('.tempo');
        if (input) return parseFloat(input.value) || 120;
        const attr = this.getAttribute('tempo');
        return attr ? parseFloat(attr) || 120 : 120;
    }

    async _handleTempoChange() {
        this.stop();
        if (this._workletNode) {
            this._workletNode.disconnect();
            this._workletNode = null;
        }
        try {
            await runtime.ensureInitialized(this._getSampleRate(), this._getTempo());
        } catch (error) {
            this._showError('Failed to change tempo');
        }
    }

    _updatePlayButton() {
        const btn = this.shadowRoot.querySelector('.play-button');
        if (!btn) return;
        if (this._isPlaying) {
            btn.textContent = '\u23F9';
            btn.classList.add('playing');
        } else {
            btn.textContent = '\u25B6';
            btn.classList.remove('playing');
        }
    }

    _updateResetButton() {
        const btn = this.shadowRoot.querySelector('.reset-button');
        if (!btn) return;
        const textarea = this.shadowRoot.querySelector('textarea');
        const textChanged = !!textarea && textarea.value !== this._originalExpression;
        const slidersChanged = JSON.stringify(this._sliders) !== JSON.stringify(this._originalSliders);
        btn.disabled = !textChanged && !slidersChanged;
    }

    // Redraws the controls for the original expression, at their defaults, and
    // moves any that are playing there.
    _resetSliders() {
        this._sliders = this._originalSliders.map(s => ({ ...s }));
        this._drawSliders();
        this._sliders.forEach(slider => this._queueSliderUpdate(slider));
    }

    // Replaces the controls with one for each slider in `sliders`, as
    // `runtime.install` returns them. A label still reached keeps its value,
    // clamped to its range; a new one starts at its default. Redraws only if
    // the controls changed.
    _updateSliders(sliders) {
        const values = new Map(this._sliders.map(s => [s.label, s.value]));
        const next = sliders.map(s => ({
            ...s,
            value: values.has(s.label)
                ? Math.min(Math.max(values.get(s.label), s.min), s.max)
                : s.default,
        }));
        const changed = JSON.stringify(next) !== JSON.stringify(this._sliders);
        this._sliders = next;
        if (changed) this._drawSliders();
    }

    // Returns the controls for the sliders `expression` reaches, each at its
    // default, or none if it doesn't evaluate.
    async _slidersOf(expression) {
        try {
            const sliders = await runtime.sliders(
                this._stripComments(expression), this._getOpens(), this._getUses());
            return sliders.map(s => ({ ...s, value: s.default }));
        } catch (e) {
            return [];
        }
    }

    // Draws a control for each of `this._sliders`. The range input moves
    // over positions from 0 to 1, which follow the slider's curve.
    _drawSliders() {
        const container = this.shadowRoot.querySelector('.sliders');
        if (!container) return;
        container.classList.toggle('hidden', !this._sliders.length);
        container.innerHTML = this._sliders.map(s => `
            <div class="slider-row">
                <label>${this._escapeHtml(s.label)}</label>
                <input type="range" min="0" max="1" step="any" value="${sliderPosition(s.min, s.max, s.curve, s.value)}" data-label="${this._escapeHtml(s.label)}">
                <span class="slider-value">${this._formatValue(s)}</span>
            </div>`).join('');
        container.querySelectorAll('input[type="range"]').forEach(input => {
            input.addEventListener('input', () => {
                const slider = this._sliders.find(s => s.label === input.dataset.label);
                if (!slider) return;
                slider.value = sliderValue(slider.min, slider.max, slider.curve, parseFloat(input.value));
                input.nextElementSibling.textContent = this._formatValue(slider);
                this._queueSliderUpdate(slider);
                this._updateResetButton();
            });
        });
    }

    // Returns the slider's value with about three significant digits for
    // the largest value in its range.
    _formatValue(slider) {
        const largest = Math.max(Math.abs(slider.min), Math.abs(slider.max));
        const intDigits = Math.max(1, String(Math.floor(largest)).length);
        return slider.value.toFixed(Math.max(0, 3 - intDigits));
    }

    // Sends the slider's value to the playing worklet, batched with other
    // moves.
    _queueSliderUpdate(slider) {
        if (!this._isPlaying || !this._workletNode) return;
        if (!this._pendingSliderUpdates) this._pendingSliderUpdates = new Map();
        this._pendingSliderUpdates.set(slider.label, slider.value);
        if (!this._batchTimerId) {
            this._batchTimerId = setTimeout(() => this._flushSliderUpdates(), 46);
        }
    }

    _showError(message) {
        const el = this.shadowRoot.querySelector('.error');
        if (el) {
            el.textContent = message;
            el.classList.remove('hidden');
        }
    }

    _hideError() {
        const el = this.shadowRoot.querySelector('.error');
        if (el) el.classList.add('hidden');
    }

    _stripComments(text) {
        return text.replace(/\/\/.*$/gm, '');
    }

    _dedent(text) {
        const lines = text.split('\n');
        // Drop empty leading/trailing lines
        while (lines.length && !lines[0].trim()) lines.shift();
        while (lines.length && !lines[lines.length - 1].trim()) lines.pop();
        if (!lines.length) return '';
        // Strip the first line's leading whitespace from all lines
        const indent = lines[0].match(/^(\s*)/)[1];
        if (!indent) return lines.join('\n');
        return lines.map(l => l.startsWith(indent) ? l.slice(indent.length) : l).join('\n');
    }

    _flushSliderUpdates() {
        if (this._pendingSliderUpdates && this._pendingSliderUpdates.size && this._workletNode) {
            this._workletNode.port.postMessage({
                type: 'update_sliders',
                values: Object.fromEntries(this._pendingSliderUpdates),
            });
            this._pendingSliderUpdates.clear();
        }
        this._batchTimerId = null;
    }

    _escapeHtml(text) {
        const el = document.createElement('span');
        el.textContent = text;
        return el.innerHTML;
    }
}

customElements.define('tuun-synth', TuunSynthElement);
