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

const DEFAULT_MODEL_URL =
  'https://cdn.jsdelivr.net/npm/garu@latest/models/base.gmdl';

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
    const wasmModule = await import('../../pkg/garu_wasm.js');
    await wasmModule.default();

    // Resolve model bytes
    let modelBytes: Uint8Array;

    if (options?.modelData) {
      modelBytes = new Uint8Array(options.modelData);
    } else {
      const url = options?.modelUrl ?? DEFAULT_MODEL_URL;
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(
          `Failed to fetch model from ${url}: ${response.status} ${response.statusText}`,
        );
      }
      const buffer = await response.arrayBuffer();
      modelBytes = new Uint8Array(buffer);
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
   */
  analyze(text: string, options?: AnalyzeOptions): AnalyzeResult | AnalyzeResult[] {
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
    if (text === '') {
      return [];
    }
    return this._wasm.tokenize(text) as string[];
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
      accuracy: 0.8,
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
