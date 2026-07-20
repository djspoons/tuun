/* tslint:disable */
/* eslint-disable */
/**
 * Parses a slider config string like `["volume:0.5:0:1", "freq:0.5:fn(x) => 100 * pow(100, x)"]`
 * and returns a JSON array of slider objects.
 *
 * Linear sliders: `{ type: "linear", label, initial_value, min, max }`
 * User-defined sliders: `{ type: "user-defined", label, normalized_initial_value, function_source, initial_value, value_at_0, value_at_1 }`
 *
 * Returns an error string if parsing fails.
 */
export function parseSliders(input: string): string;
/**
 * Evaluates a user-defined slider function at a given normalized value.
 *
 * For example, `evaluateSlider("fn(x) => 100 * pow(100, x)", 0.5)` returns ~1000.
 */
export function evaluateSlider(function_source: string, normalized_value: number): number;
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
   * Creates a new Tuun instance with the specified sample rate and tempo.
   *
   * # Arguments
   * * `sample_rate` - The audio sample rate in Hz (e.g., 44100)
   * * `tempo` - The tempo in beats per minute (e.g., 120)
   */
  constructor(sample_rate: number, tempo: number);
  /**
   * Installs an expression as the current waveform: parses it, evaluates
   * it under the slider bindings and opened modules, and stores the
   * resulting waveform for [`Wasm::process`] to render.
   *
   * `slider_json` is a JSON object mapping slider names to initial values,
   * for example, `{"volume": 0.5, "cutoff": 2000}`. Pass `"{}"` for no
   * sliders.
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
   * tuun.install("sine(2764, 0)", "{}", "[]", "[]");
   * tuun.install("$440", "{}", '["std"]', "[]");
   * tuun.install("std.square(220) * 0.3", "{}", "[]", '["std"]');
   * ```
   */
  install(expression: string, slider_json: string, open_json: string, use_json: string): void;
  /**
   * Drops the current waveform.
   */
  stop(): void;
  /**
   * Updates a slider value in the current waveform.
   *
   * Builds a linear ramp from the last value to the new value and
   * substitutes it into the playing waveform.
   */
  update_slider(name: string, value: number): void;
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
   * Returns whether a waveform is currently playing.
   */
  is_playing(): boolean;
  /**
   * Returns the current sample rate.
   */
  readonly sample_rate: number;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_tuun_free: (a: number, b: number) => void;
  readonly tuun_new: (a: number, b: number) => [number, number, number];
  readonly tuun_install: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => [number, number];
  readonly tuun_stop: (a: number) => void;
  readonly tuun_update_slider: (a: number, b: number, c: number, d: number) => void;
  readonly tuun_process: (a: number, b: number, c: number, d: any) => number;
  readonly tuun_is_playing: (a: number) => number;
  readonly tuun_sample_rate: (a: number) => number;
  readonly parseSliders: (a: number, b: number) => [number, number, number, number];
  readonly evaluateSlider: (a: number, b: number, c: number) => [number, number, number];
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
