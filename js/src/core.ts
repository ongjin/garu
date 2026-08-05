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

export { normalizeText, splitSentences } from './normalize.js';
export type { NormalizeOptions, Segment } from './normalize.js';

export interface AnalyzeOptions {
  topN?: number;
}

export interface NounsOptions {
  includeSL?: boolean;
}

/** A domain word registered at runtime via `addUserWord` / `addUserWords`. */
export interface UserWord {
  surface: string;
  pos: POS;
  /** Dictionary-scale frequency; higher wins over a split analysis. Default 5000. */
  freq?: number;
}

export interface LoadOptions {
  modelData?: ArrayBuffer;
  modelUrl?: string;
  /** 분석 결과의 자모 형태소(ETM 'ㄴ' 등)를 U+11xx 결합 자모로 정규화.
   *  기본값 false (gold v15k 다수가 호환 자모 U+3130-318F를 사용). canonical 출력이 필요하면 true. */
  normalizeJamo?: boolean;
}

export interface ModelInfo {
  version: string;
  size: number;
  accuracy: number;
}

export const EMPTY_RESULT: AnalyzeResult = Object.freeze({
  tokens: [],
  score: 0,
  elapsed: 0,
});

/**
 * Shared analyzer instance. The browser/node entry points subclass this and
 * provide their own `static load()` that resolves WASM and model bytes via
 * environment-appropriate APIs (fetch vs fs).
 */
export class GaruBase {
  protected _wasm: any;
  protected _loaded: boolean;
  protected _modelSize: number;

  protected constructor(wasmInstance: any, modelSize: number) {
    this._wasm = wasmInstance;
    this._loaded = true;
    this._modelSize = modelSize;
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
   * Set `options.includeSL` to also include foreign tokens (SL) like "AI", "BM25".
   */
  nouns(text: string, options?: NounsOptions): string[] {
    if (!this._loaded) {
      throw new Error('Garu instance has been destroyed');
    }
    if (text === '') {
      return [];
    }
    const result = this._wasm.analyze(text) as AnalyzeResult;
    const includeSL = options?.includeSL ?? false;
    return result.tokens
      .filter((t) => t.pos === 'NNG' || t.pos === 'NNP' || (includeSL && t.pos === 'SL'))
      .map((t) => t.text);
  }

  /**
   * Register a domain word so it is recognised as a single token.
   *
   * No model rebuild is needed — the word is injected into the lattice and
   * competes with the built-in dictionary on the same cost scale. A higher
   * `freq` makes it more likely to win over a split analysis (default 5000).
   * A word cannot span whitespace.
   */
  addUserWord(surface: string, pos: POS, freq?: number): void {
    if (!this._loaded) {
      throw new Error('Garu instance has been destroyed');
    }
    this._wasm.add_user_word(surface, pos, freq);
  }

  /**
   * Register several domain words at once.
   */
  addUserWords(words: UserWord[]): void {
    for (const w of words) {
      this.addUserWord(w.surface, w.pos, w.freq);
    }
  }

  /**
   * Remove every registered user word, restoring the built-in behaviour.
   */
  clearUserWords(): void {
    if (!this._loaded) {
      throw new Error('Garu instance has been destroyed');
    }
    this._wasm.clear_user_words();
  }

  /**
   * Number of registered user words.
   */
  userWordCount(): number {
    if (!this._loaded) {
      throw new Error('Garu instance has been destroyed');
    }
    return this._wasm.user_word_count() as number;
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
      accuracy: 0.960,
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
