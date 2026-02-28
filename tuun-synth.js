import { initSync, Tuun } from './pkg/tuun.js';

// ─── Shared Runtime (singleton across all <tuun-synth> instances) ─

class TuunRuntime {
    constructor() {
        this.wasmModule = null;
        this.audioContext = null;
        this.tuun = null;
        this.activeInstance = null;
        this._initPromise = null;
        this._sampleRate = null;
    }

    async ensureInitialized(sampleRate = 44100) {
        if (this._sampleRate === sampleRate && this._initPromise) {
            return this._initPromise;
        }
        this._sampleRate = sampleRate;
        this._initPromise = this._doInit(sampleRate).catch(err => {
            this._initPromise = null;
            throw err;
        });
        return this._initPromise;
    }

    async _doInit(sampleRate) {
        if (!this.wasmModule) {
            const wasmUrl = new URL('./pkg/tuun_bg.wasm', import.meta.url);
            const resp = await fetch(wasmUrl);
            const bytes = await resp.arrayBuffer();
            this.wasmModule = await WebAssembly.compile(bytes);
            initSync({ module: this.wasmModule });
        }

        if (this.audioContext) {
            this.stopActive();
            await this.audioContext.close();
        }

        this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate
        });
        this.tuun = new Tuun(sampleRate);

        const processorUrl = new URL('./tuun-processor.js', import.meta.url);
        await this.audioContext.audioWorklet.addModule(processorUrl);
    }

    parse(expression) {
        const wf = this.tuun.parse(expression);
        wf.free();
    }

    createWorkletNode() {
        return new AudioWorkletNode(this.audioContext, 'tuun-processor', {
            processorOptions: {
                wasmModule: this.wasmModule,
                bufferSize: 1024,
                sampleRate: this._sampleRate
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

.hidden {
    display: none !important;
}
`;

// ─── <tuun-synth> Custom Element ─────────────────────────────────

class TuunSynthElement extends HTMLElement {
    static observedAttributes = ['expression', 'description', 'controls', 'expanded'];

    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
        this._isPlaying = false;
        this._expanded = false;
        this._originalExpression = '';
        this._workletNode = null;
    }

    connectedCallback() {
        const script = this.querySelector('script[type="text/tuun"]');
        const bodyText = script ? script.textContent : this.textContent;
        this._originalExpression = this._dedent(bodyText) || this.getAttribute('expression') || '';
        this._expanded = !this.getAttribute('description') || this.hasAttribute('expanded');
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
            this._updateResetButton();
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
                        <button class="reset-button" title="Reset expression" disabled>&#x21BA;</button>
                    </div>
                    <div class="controls ${hasControls ? '' : 'hidden'}">
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
                        <button class="reset-button" title="Reset expression" disabled>&#x21BA;</button>
                    </div>
                    <div class="controls ${hasControls ? '' : 'hidden'}">
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
                    this._hideError();
                    this._updateResetButton();
                }
            });
        }

        if (sampleRate) {
            sampleRate.addEventListener('change', () => this._handleSampleRateChange());
        }

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
        const sampleRate = this._getSampleRate();
        await runtime.ensureInitialized(sampleRate);
        runtime.parse(expression);
        runtime.setPlaying(this);

        await this._ensureWorkletNode();

        if (runtime.audioContext.state === 'suspended') {
            await runtime.audioContext.resume();
        }

        this._workletNode.port.postMessage({ type: 'play', expression });
        this._isPlaying = true;
        this._updatePlayButton();
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
            await runtime.ensureInitialized(this._getSampleRate());
        } catch (error) {
            this._showError('Failed to change sample rate');
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
        const textarea = this.shadowRoot.querySelector('textarea');
        if (btn && textarea) {
            btn.disabled = textarea.value === this._originalExpression;
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

    _escapeHtml(text) {
        const el = document.createElement('span');
        el.textContent = text;
        return el.innerHTML;
    }
}

customElements.define('tuun-synth', TuunSynthElement);
