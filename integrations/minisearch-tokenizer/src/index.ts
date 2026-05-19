import { Garu } from 'garu-ko';
import type { AnalyzeResult, POS } from 'garu-ko';

/**
 * POS tags kept by default — content-bearing morphemes only.
 * Particles, endings, auxiliaries, etc. are dropped.
 */
const DEFAULT_POS: ReadonlyArray<POS> = [
  'NNG', 'NNP', 'NR',
  'VV', 'VA',
  'SL', 'SH', 'SN',
  'XR',
];

export interface CreateTokenizerOptions {
  /** Reuse an existing Garu instance. */
  garu?: Garu;
  /** POS tags to keep. Defaults to content-bearing morphemes. */
  posFilter?: Iterable<string>;
  /** Tokens to drop, matched after lowercasing. */
  stopwords?: Iterable<string>;
  /** Lowercase output. Default: true. */
  lowercase?: boolean;
}

export type TokenizeFn = (text: string, fieldName?: string) => string[];

/**
 * Build a tokenize function for MiniSearch.
 *
 * ```ts
 * const tokenize = await createTokenizer()
 * const ms = new MiniSearch({ fields: ['title', 'body'], tokenize })
 * ```
 *
 * Use the same tokenize at search time (MiniSearch defaults to that) or pass a
 * different one via `searchOptions.tokenize`.
 */
export async function createTokenizer(
  options: CreateTokenizerOptions = {},
): Promise<TokenizeFn> {
  const garu = options.garu ?? (await Garu.load());
  const posSet = new Set<string>(options.posFilter ?? DEFAULT_POS);
  const stopSet = new Set<string>(options.stopwords ?? []);
  const lowercase = options.lowercase !== false;

  return (text: string, _fieldName?: string): string[] => {
    if (!text) return [];
    const result = garu.analyze(text) as AnalyzeResult;
    const out: string[] = [];
    for (const t of result.tokens) {
      if (!posSet.has(t.pos)) continue;
      const v = lowercase ? t.text.toLowerCase() : t.text;
      if (stopSet.has(v)) continue;
      out.push(v);
    }
    return out;
  };
}

/**
 * Build a `processTerm` function for MiniSearch that drops stopwords and
 * lowercases. Useful when combined with the default whitespace tokenizer
 * if you only want term filtering, not full morphological splitting.
 */
export function createProcessTerm(stopwords: Iterable<string> = []) {
  const stopSet = new Set<string>(stopwords);
  return (term: string): string | null => {
    const lower = term.toLowerCase();
    if (stopSet.has(lower)) return null;
    return lower;
  };
}

export { DEFAULT_POS };
