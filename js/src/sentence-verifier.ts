import { restoreSpacing, sentenceCandidates, splitAtBoundaries } from './boundaries.js';
import type { SentenceCandidate } from './boundaries.js';
import type { Segment } from './normalize.js';

const SPACE = /\s/u;
const PUNCTUATION = /^[\p{P}\p{S}]*$/u;
const PUNCT_TAGS = new Set(['SF', 'SP', 'SS', 'SE', 'SO', 'SW']);
const VERB_TAGS = new Set(['VV', 'VA', 'VX', 'VCP', 'VCN']);
const OPEN_QUOTES = '“‘「『';
const CLOSE_QUOTES = '”’」』';
const OPEN_BRACKETS = '([{';
const CLOSE_BRACKETS = ')]}';

/** Stable training/inference ABI. All values are binary except score and capped depth. */
export const FEATURE_NAMES = [
  'baseScore', 'leftTerminal', 'leftComma', 'insideQuote', 'bracketDepth',
  'tokenBoundary', 'tokenCrossing', 'morphUnreliable', 'leftEF', 'leftEC',
  'leftETM_ETN', 'rightVerb', 'rightParticle', 'rightNoun', 'leftReporting', 'rightReporting',
] as const;

interface SentenceAnalysis {
  tokens: { text: string; pos: string; start: number; end: number }[];
}

export interface SentenceAnalyzer {
  analyze(text: string): SentenceAnalysis | SentenceAnalysis[];
}

interface MappedToken {
  pos: string;
  start: number;
  end: number;
}

function quoteStates(chars: string[]): [number[], number[]] {
  const quotes = [0];
  const brackets = [0];
  let quoteDepth = 0;
  let bracketDepth = 0;
  const straight = new Set<string>();
  for (const char of chars) {
    if (OPEN_QUOTES.includes(char)) quoteDepth++;
    if (CLOSE_QUOTES.includes(char)) quoteDepth = Math.max(0, quoteDepth - 1);
    if (char === '"' || char === "'") {
      if (straight.has(char)) straight.delete(char);
      else straight.add(char);
    }
    if (OPEN_BRACKETS.includes(char)) bracketDepth++;
    if (CLOSE_BRACKETS.includes(char)) bracketDepth = Math.max(0, bracketDepth - 1);
    quotes.push(Number(quoteDepth > 0 || straight.size > 0));
    brackets.push(Math.min(bracketDepth, 3));
  }
  return [quotes, brackets];
}

function contextFeatures(
  compact: string[], candidates: readonly SentenceCandidate[],
): number[][] {
  const [quotes, brackets] = quoteStates(compact);
  return candidates.map(candidate => {
    const i = candidate.compactOffset;
    const before = compact.slice(Math.max(0, i - 12), i).join('');
    const after = compact.slice(i, i + 12).join('');
    return [
      candidate.score,
      Number(/[.!?…]["'”’」』)\]}]*$/u.test(before)),
      Number(/[,，;:]["'”’」』)\]}]*$/u.test(before)),
      quotes[i] ?? 0,
      brackets[i] ?? 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0,
      Number(/(?:다고|라고|냐고|자고)$/u.test(before)),
      Number(/^["'”’」』)\]}]*(?:라고|다고|냐고|자고|고)/u.test(after)),
    ];
  });
}

/** Context-only verifier features; the nine morphology slots remain zero. */
export function extractContextFeatures(
  text: string, candidates: readonly SentenceCandidate[],
): number[][] {
  if (candidates.length === 0) return [];
  return contextFeatures([...text].filter(char => !SPACE.test(char)), candidates);
}

/** Analyze restored spacing once; map codepoint spans back to compact input indices. */
export function extractSentenceFeatures(
  text: string, candidates: readonly SentenceCandidate[], analyzer: SentenceAnalyzer,
): number[][] {
  if (candidates.length === 0) return [];
  const restored = restoreSpacing(text);
  const chars = [...restored];
  const compact: string[] = [];
  const prefix = [0];
  for (const char of chars) {
    if (!SPACE.test(char)) compact.push(char);
    prefix.push(compact.length);
  }
  const features = contextFeatures(compact, candidates);
  const tokens: MappedToken[] = [];
  let invalid = false;
  const analysis = analyzer.analyze(restored);
  if (Array.isArray(analysis)) throw new Error('Sentence verification requires a single analysis');
  for (const token of analysis.tokens) {
    if (!Number.isInteger(token.start) || !Number.isInteger(token.end)
        || token.start < 0 || token.end > chars.length || token.start >= token.end
        || prefix[token.start] === prefix[token.end]) {
      invalid = true;
      continue;
    }
    tokens.push({ pos: token.pos, start: prefix[token.start], end: prefix[token.end] });
  }

  return candidates.map((candidate, index) => {
    const i = candidate.compactOffset;
    let left: MappedToken | undefined;
    let right: MappedToken | undefined;
    let crossing = false;
    let boundary = false;
    for (const token of tokens) {
      if (token.start < i && i < token.end) crossing = true;
      if (token.end === i || token.start === i) boundary = true;
      if (PUNCT_TAGS.has(token.pos)) continue;
      if (token.end <= i && (!left || token.end >= left.end)) left = token;
      if (token.start >= i && (!right || token.start < right.start)) right = token;
    }
    // Gaps containing words indicate missing coverage; punctuation between tokens is allowed.
    const reliable = !invalid && !crossing && left !== undefined && right !== undefined
      && PUNCTUATION.test(compact.slice(left.end, i).join(''))
      && PUNCTUATION.test(compact.slice(i, right.start).join(''));
    const leftPos = reliable ? left!.pos : '';
    const rightPos = reliable ? right!.pos : '';
    const values = features[index];
    values.splice(5, 9,
      Number(boundary),
      Number(crossing),
      Number(!reliable),
      Number(leftPos === 'EF'),
      Number(leftPos === 'EC'),
      Number(leftPos === 'ETM' || leftPos === 'ETN'),
      Number(VERB_TAGS.has(rightPos)),
      Number(rightPos.startsWith('J')),
      Number(rightPos === 'NNG' || rightPos === 'NNP' || rightPos === 'NNB' || rightPos === 'NP'),
    );
    return values;
  });
}

export interface Q4Tensor {
  rows: number;
  cols: number;
  /** Flattened row-major nibbles, low nibble first; signed value = nibble - 8. */
  data: string;
  /** One little-endian float32 scale per row. */
  scales: string;
  /** Optional little-endian float32 bias per row, for dense tensors only. */
  bias?: string;
}

export interface SentenceModelData {
  version: 1;
  /** Number of compact Unicode codepoints on each side of a boundary. */
  context: number;
  embeddingSize: number;
  featureNames: string[];
  /** IDs are array index + 2; ID0 is padding and ID1 is unknown. */
  vocabulary: string[];
  threshold: number;
  sourceHash: string;
  tensors: { embedding: Q4Tensor; hidden: Q4Tensor; output: Q4Tensor };
}

interface DecodedTensor {
  rows: number;
  cols: number;
  values: Float32Array;
  bias: Float32Array;
}

export interface SentenceVerifier {
  readonly context: number;
  readonly embeddingSize: number;
  readonly threshold: number;
  readonly vocabulary: ReadonlyMap<string, number>;
  readonly embedding: DecodedTensor;
  readonly hidden: DecodedTensor;
  readonly output: DecodedTensor;
}

function decodeBytes(encoded: string): Uint8Array {
  const raw = atob(encoded);
  return Uint8Array.from(raw, char => char.charCodeAt(0));
}

function readFloats(encoded: string, length: number): Float32Array {
  const bytes = decodeBytes(encoded);
  if (bytes.length !== length * 4) throw new Error('Invalid sentence model float32 length');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const values = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    values[i] = view.getFloat32(i * 4, true);
    if (!Number.isFinite(values[i])) throw new Error('Non-finite sentence model value');
  }
  return values;
}

function decodeTensor(tensor: Q4Tensor): DecodedTensor {
  const { rows, cols } = tensor;
  if (!Number.isSafeInteger(rows) || rows <= 0 || !Number.isSafeInteger(cols) || cols <= 0
      || !Number.isSafeInteger(rows * cols)) throw new Error('Invalid sentence model dimensions');
  const bytes = decodeBytes(tensor.data);
  if (bytes.length !== Math.ceil(rows * cols / 2)) throw new Error('Invalid sentence model Q4 length');
  const scales = readFloats(tensor.scales, rows);
  if (scales.some(value => value < 0)) throw new Error('Negative sentence model scale');
  const values = new Float32Array(rows * cols);
  for (let i = 0; i < values.length; i++) {
    const quantized = ((bytes[i >> 1] >> ((i & 1) * 4)) & 15) - 8;
    values[i] = quantized * scales[Math.floor(i / cols)];
    if (!Number.isFinite(values[i])) throw new Error('Non-finite sentence model weight');
  }
  const bias = tensor.bias === undefined ? new Float32Array(rows) : readFloats(tensor.bias, rows);
  return { rows, cols, values, bias };
}

/** Decode explicit local model data once. No model weights or analyzer are imported. */
export function loadSentenceVerifier(data: SentenceModelData): SentenceVerifier {
  if (data.version !== 1 || !Number.isSafeInteger(data.context) || data.context <= 0
      || !Number.isSafeInteger(data.embeddingSize) || data.embeddingSize <= 0
      || !Number.isFinite(data.threshold) || typeof data.sourceHash !== 'string') {
    throw new Error('Invalid sentence model metadata');
  }
  if (data.featureNames.length !== FEATURE_NAMES.length
      || FEATURE_NAMES.some((name, i) => name !== data.featureNames[i])) {
    throw new Error('Incompatible sentence model feature order');
  }
  const vocabulary = new Map<string, number>();
  for (const char of data.vocabulary) {
    if (typeof char !== 'string' || [...char].length !== 1 || vocabulary.has(char)) {
      throw new Error('Invalid sentence model vocabulary');
    }
    vocabulary.set(char, vocabulary.size + 2);
  }
  const { embedding, hidden, output } = data.tensors;
  if (embedding.rows !== vocabulary.size + 2 || embedding.cols !== data.embeddingSize
      || embedding.bias !== undefined
      || hidden.cols !== data.context * 2 * data.embeddingSize + FEATURE_NAMES.length
      || output.rows !== 1 || output.cols !== hidden.rows) {
    throw new Error('Incompatible sentence model tensor dimensions');
  }
  const decoded = decodeTensor(embedding);
  if (decoded.values.subarray(0, data.embeddingSize).some(value => value !== 0)) {
    throw new Error('Sentence model padding embedding must be zero');
  }
  return {
    context: data.context, embeddingSize: data.embeddingSize, threshold: data.threshold,
    vocabulary, embedding: decoded, hidden: decodeTensor(hidden), output: decodeTensor(output),
  };
}

/** Return logits for supplied candidates; context is flattened position-major. */
export function scoreSentenceCandidates(
  text: string, candidates: readonly SentenceCandidate[], features: readonly (readonly number[])[],
  model: SentenceVerifier,
): number[] {
  if (features.length !== candidates.length) throw new Error('Sentence feature count mismatch');
  if (candidates.length === 0) return [];
  const compact = [...text].filter(char => !SPACE.test(char));
  const ids = compact.map(char => model.vocabulary.get(char) ?? 1);
  const input = new Float32Array(model.hidden.cols);
  const hidden = new Float32Array(model.hidden.rows);
  return candidates.map((candidate, index) => {
    const boundary = candidate.compactOffset;
    if (!Number.isInteger(boundary) || boundary < 1 || boundary >= compact.length) {
      throw new Error('Sentence candidate outside input');
    }
    const row = features[index];
    if (row.length !== FEATURE_NAMES.length || row.some(value => !Number.isFinite(value))) {
      throw new Error('Invalid sentence feature vector');
    }
    let cursor = 0;
    for (let i = boundary - model.context; i < boundary + model.context; i++) {
      const offset = (ids[i] ?? 0) * model.embeddingSize;
      for (let j = 0; j < model.embeddingSize; j++) input[cursor++] = model.embedding.values[offset + j];
    }
    input.set(row, cursor);
    for (let i = 0; i < model.hidden.rows; i++) {
      let value = model.hidden.bias[i];
      const offset = i * model.hidden.cols;
      for (let j = 0; j < input.length; j++) value += input[j] * model.hidden.values[offset + j];
      hidden[i] = Math.max(0, value);
    }
    let score = model.output.bias[0];
    for (let i = 0; i < hidden.length; i++) score += hidden[i] * model.output.values[i];
    return score;
  });
}

/** A strict veto: retain existing candidates only when their score exceeds threshold. */
export function verifySentenceCandidates(
  text: string, candidates: readonly SentenceCandidate[], features: readonly (readonly number[])[],
  model: SentenceVerifier,
): SentenceCandidate[] {
  const scores = scoreSentenceCandidates(text, candidates, features, model);
  return candidates.filter((_, i) => scores[i] > model.threshold);
}

/** Keep balanced quotations as top-level units, without trusting unmatched quote state. */
export function keepOutsideQuotes(
  text: string, candidates: readonly SentenceCandidate[],
): SentenceCandidate[] {
  if (candidates.length === 0) return [];
  const chars = [...text];
  const changes = new Int32Array(chars.length + 1);
  const stack: { family: number; start: number; previous: number }[] = [];
  const active: number[] = Array(OPEN_QUOTES.length).fill(-1);
  // Straight double/single quotes share families with curved double/single quotes.
  const straight = '"\'';
  const singleFamily = OPEN_QUOTES.indexOf('‘');
  const latinLetter = /^\p{Script=Latin}$/u;
  for (let i = 0; i < chars.length; i++) {
    const char = chars[i];
    if ((char === "'" || char === '’') && latinLetter.test(chars[i - 1] ?? '')
        && latinLetter.test(chars[i + 1] ?? '')) continue;
    // A possessive apostrophe cannot open a quote, but may close an already open one.
    if (char === "'" && active[singleFamily] < 0 && latinLetter.test(chars[i - 1] ?? '')
        && SPACE.test(chars[i + 1] ?? '')) continue;
    let family = OPEN_QUOTES.indexOf(char);
    let closes = false;
    if (family < 0) {
      family = CLOSE_QUOTES.indexOf(char);
      closes = family >= 0;
    }
    if (family < 0) {
      family = straight.indexOf(char);
      if (family < 0) continue;
      closes = stack[stack.length - 1]?.family === family;
    }
    if (!closes) {
      stack.push({ family, start: i, previous: active[family] });
      active[family] = stack.length - 1;
      continue;
    }

    const match = active[family];
    if (match < 0) continue;
    const balanced = match === stack.length - 1;
    const start = stack[match].start;
    // Crossed directional marks invalidate unfinished frames instead of protecting their tail.
    while (stack.length > match) {
      const frame = stack.pop()!;
      active[frame.family] = frame.previous;
    }
    if (!balanced) continue;
    changes[start + 1]++;
    changes[i + 1]--;
  }
  const protectedAt = new Uint8Array(chars.length + 1);
  let depth = 0;
  for (let i = 0; i < protectedAt.length; i++) {
    depth += changes[i];
    protectedAt[i] = Number(depth > 0);
  }
  return candidates.filter(candidate => !protectedAt[candidate.offset]);
}

/** Split original text using only existing candidates retained by the explicit verifier. */
export function splitVerified(
  text: string, model: SentenceVerifier, analyzer?: SentenceAnalyzer,
): Segment[] {
  const candidates = keepOutsideQuotes(text, sentenceCandidates(text));
  const features = analyzer
    ? extractSentenceFeatures(text, candidates, analyzer)
    : extractContextFeatures(text, candidates);
  const retained = verifySentenceCandidates(text, candidates, features, model);
  return splitAtBoundaries(text, snapBoundaries(text, retained.map(candidate => candidate.offset)));
}

// Trailing laughter, emoticons, tildes and repeated marks belong to the preceding sentence.
const TRAILING = /[\s~!?.…^;*♡♥★☆♪♬”’)\]」』\u1100-\u11FF\u3131-\u318E\u2600-\u27BF\uFE0F\u200D\u{1F000}-\u{1FAFF}]/u;
const HANGUL_SYLLABLE = /[가-힣]/u;
// Spaced input: at least one word gap per ten Hangul syllables.
const SPACED_RATIO = 10;

/** Move boundaries past trailing ornaments; in spaced input, drop word-internal boundaries. */
function snapBoundaries(text: string, offsets: number[]): number[] {
  const chars = [...text];
  let hangul = 0;
  let gaps = 0;
  for (let i = 0; i < chars.length; i++) {
    if (HANGUL_SYLLABLE.test(chars[i])) hangul++;
    else if (SPACE.test(chars[i]) && i > 0 && !SPACE.test(chars[i - 1])) gaps++;
  }
  const spaced = gaps * SPACED_RATIO >= hangul;

  const out: number[] = [];
  for (let offset of offsets) {
    while (offset < chars.length && TRAILING.test(chars[offset])) offset++;
    if (offset >= chars.length || out[out.length - 1] === offset) continue;
    if (spaced && offset > 0 && !TRAILING.test(chars[offset - 1])) continue;
    out.push(offset);
  }
  return out;
}
