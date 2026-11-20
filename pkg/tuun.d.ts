/* tslint:disable */
/* eslint-disable */
/**
 * Returns the value at `position`, from 0 at `min` to 1 at `max`, along a
 * slider with the curve named `curve` (see
 * [`slider::SliderSpec::denormalize`]).
 */
export function sliderValue(min: number, max: number, curve: string, position: number): number;
/**
 * Returns the position of `value` along a slider over `min` to `max` with
 * the curve named `curve`, the inverse of [`slider_value`].
 */
export function sliderPosition(min: number, max: number, curve: string, value: number): number;
/**
 * Initializes the WASM module.
 * This is called automatically when you import the module.
 */
export function main(): void;
/**
 * WebAssembly interface for the Tuun synthesizer.
 *
 * Owns the currently-playing waveform.
 */
export class Tuun {
  free(): void;
  /**
   * Returns whether a waveform is currently playing.
   */
  is_playing(): boolean;
  /**
   * Moves the slider `name` of the playing waveform to `value`, clamped to
   * its range, ramping from its last value. Does nothing if nothing is
   * playing or the waveform doesn't reach that slider.
   */
  update_slider(name: string, value: number): void;
  /**
   * Creates a new Tuun instance with the specified sample rate and tempo.
   *
   * # Arguments
   * * `sample_rate` - The audio sample rate in Hz (e.g., 44100)
   * * `tempo` - The tempo in beats per minute (e.g., 120)
   */
  constructor(sample_rate: number, tempo: number);
  /**
   * Drops the current waveform.
   */
  stop(): void;
  /**
   * Installs an expression as the current waveform: parses it, evaluates
   * it under the opened modules, and stores the resulting waveform for
   * [`Wasm::process`] to render. Returns the sliders it reaches, as
   * [`Wasm::sliders`] does.
   *
   * `value_json` is a JSON object mapping slider labels to the values they
   * start at, for example `{"volume": 0.5, "cutoff": 2000}`. A value
   * outside its slider's range is clamped; a slider with no value starts
   * at its default, and a label the expression doesn't reach is ignored.
   * Pass `"{}"` for every slider's default.
   *
   * `open_json` is a JSON array of dotted module paths to bring into scope
   * before evaluating, e.g. `["std", "foo.bar"]`. Each entry behaves like an
   * `open` binding at the top of the expression. Pass `"[]"` for no opens.
   *
   * `use_json` is a JSON array of dotted module paths to bind as module
   * values, e.g. `["synth.pm"]`. Each entry behaves like a `use` binding
   * at the top of the expression: the module is bound to its last path
   * component and its bindings are reached by `.name` projection. Pass
   * `"[]"` for no uses.
   *
   * # Examples
   * ```javascript
   * tuun.install("$440", "{}", '["std"]', "[]");
   * tuun.install("std.square(220) * 0.3", "{}", "[]", '["std"]');
   * ```
   */
  install(expression: string, value_json: string, open_json: string, use_json: string): string;
  /**
   * Generates audio samples from the current waveform. Updates the internal
   * state of the waveform so that the next call to `generate()` will continue
   * from the point at which this call left off.
   *
   * # Arguments
   * * `out` - A buffer to fill with samples
   *
   * # Returns
   * A boolean indicating whether or not the current waveform will generate any
   * more samples
   *
   * # Examples
   * ```javascript
   * tuun.install("$440", "{}", '["std"]', "[]");
   * const done = tuun.process(output);
   * ```
   */
  process(out: Float32Array): boolean;
  /**
   * Returns the sliders `expression` reaches, as a JSON array of `{label,
   * default, min, max, curve}` objects in the order the expression first
   * reaches their calls, without installing it.
   *
   * `open_json` and `use_json` are as for [`Wasm::install`]. Returns the
   * error [`Wasm::install`] would.
   *
   * # Examples
   * ```javascript
   * tuun.sliders('$220 * slider("level", 0.5)', '["std"]', "[]");
   * // '[{"label":"level","default":0.5,"min":0,"max":1,"curve":"linear"}]'
   * ```
   */
  sliders(expression: string, open_json: string, use_json: string): string;
  /**
   * Returns the current sample rate.
   */
  readonly sample_rate: number;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_tuun_free: (a: number, b: number) => void;
  readonly sliderPosition: (a: number, b: number, c: number, d: number, e: number) => [number, number, number];
  readonly sliderValue: (a: number, b: number, c: number, d: number, e: number) => [number, number, number];
  readonly tuun_install: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => [number, number, number, number];
  readonly tuun_is_playing: (a: number) => number;
  readonly tuun_new: (a: number, b: number) => [number, number, number];
  readonly tuun_process: (a: number, b: number, c: number, d: any) => number;
  readonly tuun_sample_rate: (a: number) => number;
  readonly tuun_sliders: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number, number, number];
  readonly tuun_stop: (a: number) => void;
  readonly tuun_update_slider: (a: number, b: number, c: number, d: number) => void;
  readonly main: () => void;
  readonly __wbindgen_free: (a: number, b: number, c: number) => void;
  readonly __wbindgen_malloc: (a: number, b: number) => number;
  readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
  readonly __wbindgen_export_3: WebAssembly.Table;
  readonly __externref_table_dealloc: (a: number) => void;
  readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;
/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
*
* @returns {InitOutput}
*/
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
