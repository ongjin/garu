import { Garu } from 'garu-ko';
import type { AnalyzeResult, POS } from 'garu-ko';

/**
 * POS tags kept by default. Captures content-bearing morphemes
 * (nouns, verb/adjective stems, foreign words, numerals, roots) and drops
 * particles, endings, and other functional pieces.
 */
const DEFAULT_POS: ReadonlyArray<POS> = [
  'NNG', // general noun
  'NNP', // proper noun
  'NR',  // number
  'VV',  // verb stem
  'VA',  // adjective stem
  'SL',  // foreign word (Latin)
  'SH',  // Chinese characters
  'SN',  // digits
  'XR',  // root
];

export interface CreateTokenizerOptions {
  /** Reuse an existing Garu instance. If omitted, a new one is loaded. */
  garu?: Garu;
  /** POS tags to include. Defaults to content-bearing morphemes. */
  posFilter?: Iterable<string>;
  /** Tokens to drop (already lowercased). */
  stopwords?: Iterable<string>;
  /** Lowercase output. Default: true. */
  lowercase?: boolean;
}

export interface GaruOramaTokenizer {
  language: string;
  normalizationCache: Map<string, string>;
  tokenize: (raw: string, language?: string, prop?: string) => string[];
}

/**
 * Create an Orama-compatible tokenizer that segments Korean text using Garu.
 *
 * Pass the result to Orama's `components.tokenizer`:
 *
 * ```ts
 * const db = await create({
 *   schema: { title: 'string', body: 'string' },
 *   components: { tokenizer: await createTokenizer() }
 * })
 * ```
 */
export async function createTokenizer(
  options: CreateTokenizerOptions = {},
): Promise<GaruOramaTokenizer> {
  const garu = options.garu ?? (await Garu.load());
  const posSet = new Set<string>(options.posFilter ?? DEFAULT_POS);
  const stopSet = new Set<string>(options.stopwords ?? []);
  const lowercase = options.lowercase !== false;

  return {
    language: 'korean',
    normalizationCache: new Map<string, string>(),
    tokenize(raw: string): string[] {
      if (!raw) return [];
      const result = garu.analyze(raw) as AnalyzeResult;
      const out: string[] = [];
      for (const t of result.tokens) {
        if (!posSet.has(t.pos)) continue;
        const text = lowercase ? t.text.toLowerCase() : t.text;
        if (stopSet.has(text)) continue;
        out.push(text);
      }
      return out;
    },
  };
}

export { DEFAULT_POS };
