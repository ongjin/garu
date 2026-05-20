import { GaruBase } from './core.js';
import type { LoadOptions } from './core.js';

export type {
  POS,
  Token,
  AnalyzeResult,
  AnalyzeOptions,
  NounsOptions,
  LoadOptions,
  ModelInfo,
  NormalizeOptions,
  Segment,
} from './core.js';
export { normalizeText, splitSentences } from './normalize.js';

/**
 * Browser-targeted Garu. Resolves WASM and model assets via `fetch` and
 * `new URL(..., import.meta.url)`. Has no Node-only imports (`fs/promises`,
 * `url`, `path`), so bundlers (Vite, Webpack, Rollup) emit a clean
 * browser-only chunk.
 */
export class Garu extends GaruBase {
  static async load(options?: LoadOptions): Promise<Garu> {
    // @ts-ignore dynamic WASM import
    const wasmModule = await import('../pkg/garu_wasm.js');
    await wasmModule.default();

    let modelBytes: Uint8Array;
    if (options?.modelData) {
      modelBytes = new Uint8Array(options.modelData);
    } else if (options?.modelUrl) {
      const response = await fetch(options.modelUrl);
      if (!response.ok) {
        throw new Error(
          `Failed to fetch model from ${options.modelUrl}: ${response.status} ${response.statusText}`,
        );
      }
      modelBytes = new Uint8Array(await response.arrayBuffer());
    } else {
      const url = new URL('../models/base.gmdl', import.meta.url).href;
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(
          `Failed to fetch model from ${url}: ${response.status} ${response.statusText}`,
        );
      }
      modelBytes = new Uint8Array(await response.arrayBuffer());
    }

    const wasmInstance = new wasmModule.GaruWasm(modelBytes);
    return new Garu(wasmInstance, modelBytes.byteLength);
  }
}
