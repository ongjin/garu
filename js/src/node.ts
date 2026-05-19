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
 * Node-targeted Garu. Resolves WASM and model bytes via `fs/promises`.
 * Browser code paths must not import this entry — use `garu-ko` directly
 * (resolves via package.json conditional exports) or import from
 * `garu-ko/browser`.
 */
export class Garu extends GaruBase {
  static async load(options?: LoadOptions): Promise<Garu> {
    // @ts-ignore dynamic WASM import
    const wasmModule = await import('../pkg/garu_wasm.js');
    const { readFile } = await import('fs/promises');
    const { fileURLToPath } = await import('url');
    const { join, dirname } = await import('path');
    const dir = dirname(fileURLToPath(import.meta.url));
    const wasmBytes = await readFile(join(dir, '..', 'pkg', 'garu_wasm_bg.wasm'));
    await wasmModule.default(wasmBytes);

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
      const buf = await readFile(join(dir, '..', 'models', 'base.gmdl'));
      modelBytes = new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
    }

    const cnnBuf = await readFile(join(dir, '..', 'models', 'cnn2.bin'));
    const cnnBytes = new Uint8Array(cnnBuf.buffer, cnnBuf.byteOffset, cnnBuf.byteLength);

    const wasmInstance = new wasmModule.GaruWasm(modelBytes, cnnBytes);
    return new Garu(wasmInstance, modelBytes.byteLength + cnnBytes.byteLength);
  }
}
