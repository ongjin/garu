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

export interface ModelInfo {
  version: string;
  size: number;
  accuracy: number;
}

export class Garu {
  private constructor() {}

  static async load(): Promise<Garu> {
    throw new Error('Not implemented');
  }

  analyze(_text: string, _options?: AnalyzeOptions): AnalyzeResult | AnalyzeResult[] {
    throw new Error('Not implemented');
  }

  tokenize(_text: string): string[] {
    throw new Error('Not implemented');
  }

  isLoaded(): boolean {
    throw new Error('Not implemented');
  }

  modelInfo(): ModelInfo {
    throw new Error('Not implemented');
  }

  destroy(): void {
    throw new Error('Not implemented');
  }
}
