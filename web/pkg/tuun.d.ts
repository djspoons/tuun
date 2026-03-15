/* tslint:disable */
/* eslint-disable */
/**
 * Initializes the WASM module.
 * This is called automatically when you import the module.
 */
export function main(): void;
/**
 * WebAssembly interface for the Tuun synthesizer.
 *
 * Provides parsing, optimization, and audio generation from Tuun expressions.
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
   * Parses a Tuun expression and returns a WasmWaveform.
   *
   * # Arguments
   * * `expression` - The Tuun expression string to parse
   *
   * # Returns
   * A WasmWaveform that can be used with `generate()`, or an error string
   *
   * # Example
   * ```javascript
   * const waveform = tuun.parse("sine(2764, 0)");
   * ```
   */
  parse(expression: string): WasmWaveform;
  set_slider_value(name: string, value: number): void;
  /**
   * Generates audio samples from a waveform. Updates the internal state
   * of the waveform so that the next call to `generate()` will continue
   * from the point at which this call left off.
   *
   * # Arguments
   * * `waveform` - The WasmWaveform to generate from
   * * `desired` - The number of samples to generate
   *
   * # Returns
   * A Float32Array of audio samples
   *
   * # Example
   * ```javascript
   * const samples = tuun.generate(waveform, 4096);
   * // samples is a Float32Array that can be used with Web Audio API
   * ```
   */
  generate(waveform: WasmWaveform, desired: number): Float32Array;
  /**
   * Returns the current sample rate.
   */
  readonly sample_rate: number;
}
/**
 * A waveform that can be used to generate audio samples.
 *
 * This wraps the internal Waveform type and maintains state between
 * calls to generate().
 */
export class WasmWaveform {
  private constructor();
  free(): void;
  /**
   * Returns a string representation of the waveform (for debugging).
   */
  toString(): string;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_tuun_free: (a: number, b: number) => void;
  readonly tuun_new: (a: number, b: number) => [number, number, number];
  readonly tuun_parse: (a: number, b: number, c: number) => [number, number, number];
  readonly tuun_set_slider_value: (a: number, b: number, c: number, d: number) => void;
  readonly tuun_generate: (a: number, b: number, c: number) => [number, number];
  readonly tuun_sample_rate: (a: number) => number;
  readonly __wbg_wasmwaveform_free: (a: number, b: number) => void;
  readonly wasmwaveform_toString: (a: number) => [number, number];
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
