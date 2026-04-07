/* tslint:disable */
/* eslint-disable */
/**
 * Parses a slider config string like `["volume:0.5:0:1", "cutoff:2000:200:8000"]`
 * and returns a JSON array of slider objects.
 *
 * Each object has: `{ label, initial_value, min, max }`
 *
 * Returns an error string if parsing fails.
 */
export function parseSliders(input: string): string;
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
   * Parses an expression with slider bindings and prepares for playback.
   *
   * `slider_json` is a JSON object mapping slider names to initial values,
   * for example, `{"volume": 0.5, "cutoff": 2000}`.
   * Pass `"{}"` for no sliders.
   *
   * # Examples
   * ```javascript
   * const waveform = tuun.parse("sine(2764, 0)", "{}");
   * ```
   */
  parse(expression: string, slider_json: string): void;
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
   * tuun.parse("$440", "{}");
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
  readonly tuun_parse: (a: number, b: number, c: number, d: number, e: number) => [number, number];
  readonly tuun_stop: (a: number) => void;
  readonly tuun_update_slider: (a: number, b: number, c: number, d: number) => void;
  readonly tuun_process: (a: number, b: number, c: number, d: any) => number;
  readonly tuun_is_playing: (a: number) => number;
  readonly tuun_sample_rate: (a: number) => number;
  readonly parseSliders: (a: number, b: number) => [number, number, number, number];
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
