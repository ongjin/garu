/** Optional local sentence verifier. Import from `garu-ko/sentences`. */
import { SENTENCE_MODEL } from './sentence-data.js';
import { SENTENCE_MODEL as FAST_MODEL } from './sentence-fast-data.js';
import { loadSentenceVerifier, splitVerified } from './sentence-verifier.js';
import type { SentenceAnalyzer, SentenceVerifier } from './sentence-verifier.js';
import type { Segment } from './normalize.js';

export type { SentenceAnalyzer } from './sentence-verifier.js';

const LINE_BREAK = /\r\n|[\n\r\u2028\u2029]/gu;

let model: SentenceVerifier | undefined;
let fastModel: SentenceVerifier | undefined;

/**
 * Fast offline splitting; supplying an analyzer opts into the slower morphology verifier.
 * Every line break ends a sentence; join hard-wrapped lines before calling.
 */
export function splitSentences(text: string, analyzer?: SentenceAnalyzer): Segment[] {
  const lines = splitLines(text);
  if (lines.length === 1) return splitLine(text, analyzer);

  const segments: Segment[] = [];
  for (const line of lines) {
    if (!line.text.trim()) continue;
    for (const segment of splitLine(line.text, analyzer)) {
      segments.push({ text: segment.text, offset: line.offset + segment.offset });
    }
  }
  return segments.length ? segments : [{ text: text.trim(), offset: 0 }];
}

function splitLine(text: string, analyzer?: SentenceAnalyzer): Segment[] {
  if (!analyzer) {
    fastModel ??= loadSentenceVerifier(FAST_MODEL);
    return splitVerified(text, fastModel);
  }
  model ??= loadSentenceVerifier(SENTENCE_MODEL);
  return splitVerified(text, model, analyzer);
}

/** Lines with Unicode codepoint offsets into the original text. */
function splitLines(text: string): Segment[] {
  const lines: Segment[] = [];
  let start = 0;
  let offset = 0;
  for (const match of text.matchAll(LINE_BREAK)) {
    const line = text.slice(start, match.index);
    lines.push({ text: line, offset });
    offset += [...line].length + match[0].length;
    start = match.index + match[0].length;
  }
  lines.push({ text: text.slice(start), offset });
  return lines;
}
