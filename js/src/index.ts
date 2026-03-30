export type POS =
  | 'NNG' | 'NNP' | 'NNB' | 'NR' | 'NP'
  | 'VV' | 'VA' | 'VX' | 'VCP' | 'VCN'
  | 'MAG' | 'MAJ' | 'MM'
  | 'IC'
  | 'JKS' | 'JKC' | 'JKG' | 'JKO' | 'JKB' | 'JKV' | 'JKQ' | 'JX' | 'JC'
  | 'EP' | 'EF' | 'EC' | 'ETN' | 'ETM'
  | 'XPN' | 'XSN' | 'XSV' | 'XSA' | 'XR'
  | 'SF' | 'SP' | 'SS' | 'SE' | 'SO' | 'SW' | 'SH' | 'SL' | 'SN';

export interface Token {
  text: string;
  pos: POS;
  start: number;
  end: number;
  score?: number;
}

export interface AnalyzeResult {
  tokens: Token[];
  score: number;
  elapsed: number;
}

export interface AnalyzeOptions {
  topN?: number;
}

export interface LoadOptions {
  modelData?: ArrayBuffer;
  modelUrl?: string;
}

export interface ModelInfo {
  version: string;
  size: number;
  accuracy: number;
}

const isNode =
  typeof process !== 'undefined' &&
  process.versions != null &&
  process.versions.node != null;

const EMPTY_RESULT: AnalyzeResult = Object.freeze({
  tokens: [],
  score: 0,
  elapsed: 0,
});

export class Garu {
  private _wasm: any;
  private _loaded: boolean;
  private _modelSize: number;

  private constructor(wasmInstance: any, modelSize: number) {
    this._wasm = wasmInstance;
    this._loaded = true;
    this._modelSize = modelSize;
  }

  /**
   * Load the WASM module and model data, returning a ready-to-use Garu instance.
   *
   * @param options.modelData - Provide model bytes directly as an ArrayBuffer
   * @param options.modelUrl  - Fetch model from this URL
   * If neither is provided, the model is fetched from the default CDN URL.
   */
  static async load(options?: LoadOptions): Promise<Garu> {
    // Dynamic import of the WASM glue module and initialise it
    // @ts-ignore dynamic WASM import
    const wasmModule = await import('../pkg/garu_wasm.js');
    if (isNode) {
      const { readFile } = await import('fs/promises');
      const { fileURLToPath } = await import('url');
      const { join, dirname } = await import('path');
      const dir = dirname(fileURLToPath(import.meta.url));
      const wasmBytes = await readFile(join(dir, '..', 'pkg', 'garu_wasm_bg.wasm'));
      await wasmModule.default(wasmBytes);
    } else {
      await wasmModule.default();
    }

    // Resolve model bytes
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
    } else if (isNode) {
      const { readFile } = await import('fs/promises');
      const { fileURLToPath } = await import('url');
      const { join, dirname } = await import('path');
      const dir = dirname(fileURLToPath(import.meta.url));
      const buf = await readFile(join(dir, '..', 'models', 'base.gmdl'));
      modelBytes = new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
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

    // Construct the WASM analyzer
    const wasmInstance = new wasmModule.GaruWasm(modelBytes);
    return new Garu(wasmInstance, modelBytes.byteLength);
  }

  /**
   * Analyze Korean text, returning morphological tokens with scores.
   *
   * When `options.topN` is greater than 1, returns an array of N-best results.
   * Otherwise returns a single AnalyzeResult.
   *
   * Note: topN > 1 is not yet fully supported and may return fewer results.
   */
  analyze(text: string, options?: AnalyzeOptions): AnalyzeResult | AnalyzeResult[] {
    if (!this._loaded) {
      throw new Error('Garu instance has been destroyed');
    }
    const topN = options?.topN ?? 1;

    if (topN > 1) {
      if (text === '') {
        return [{ ...EMPTY_RESULT, tokens: [] }];
      }
      return this._wasm.analyze_topn(text, topN) as AnalyzeResult[];
    }

    if (text === '') {
      return { ...EMPTY_RESULT, tokens: [] };
    }
    return this._wasm.analyze(text) as AnalyzeResult;
  }

  /**
   * Quick tokenisation — returns an array of surface-form strings.
   */
  tokenize(text: string): string[] {
    if (!this._loaded) {
      throw new Error('Garu instance has been destroyed');
    }
    if (text === '') {
      return [];
    }
    return this._wasm.tokenize(text) as string[];
  }

  /**
   * Extract nouns (NNG, NNP) from text.
   */
  nouns(text: string): string[] {
    if (!this._loaded) {
      throw new Error('Garu instance has been destroyed');
    }
    if (text === '') {
      return [];
    }
    const result = this._wasm.analyze(text) as AnalyzeResult;
    return result.tokens
      .filter((t) => t.pos === 'NNG' || t.pos === 'NNP')
      .map((t) => t.text);
  }

  /**
   * Whether the WASM analyzer is loaded and ready.
   */
  isLoaded(): boolean {
    return this._loaded;
  }

  /**
   * Return metadata about the loaded model.
   */
  modelInfo(): ModelInfo {
    return {
      version: this._wasm.constructor.version(),
      size: this._modelSize,
      accuracy: 0.953,
    };
  }

  /**
   * Free the WASM instance and mark this Garu as unloaded.
   */
  destroy(): void {
    if (this._wasm) {
      this._wasm.free();
      this._wasm = null;
    }
    this._loaded = false;
  }
}
