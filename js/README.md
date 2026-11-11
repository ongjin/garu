# garu-ko

**Browser-native Korean morphological analyzer.** No server required.

- **1.34MB model** bundled in npm package (no CDN needed)
- **447KB WASM** engine (182KB gzipped) -- runs in any modern browser
- **F1 96.18%** on 9k human-verified gold testset (ep_norm), **F1 92.32%** on a fixed 2025 spoken-language set
- **~1ms** inference per sentence
- **Offline-ready** -- works without network
- **[Live Demo](https://garu.zerry.co.kr)** -- try it in your browser

## Live demos

Try it in the browser — every page below runs the analyzer 100% client-side via WASM.

- **[Morphological analyzer](https://garu.zerry.co.kr)** — interactive POS tagging with the full Sejong tagset
- **[Use cases](https://garu.zerry.co.kr/use-cases)** — copy-paste recipes for search indexing, keyword extraction, chatbot preprocessing, and offline PWAs
- **[Orama integration](https://garu.zerry.co.kr/orama)** — side-by-side comparison of Orama's default tokenizer vs. `garu-orama-tokenizer`
- **[MiniSearch integration](https://garu.zerry.co.kr/minisearch)** — same comparison for MiniSearch

## Quick Start

```bash
npm install garu-ko
```

```typescript
import { Garu } from 'garu-ko';

const garu = await Garu.load();

// Morphological analysis
const result = garu.analyze('배가 아파서 약을 먹었다');
console.log(result.tokens);
// [
//   { text: '배',   pos: 'NNG', start: 0, end: 2 },
//   { text: '가',   pos: 'JKS', start: 0, end: 2 },
//   { text: '아프', pos: 'VA',  start: 3, end: 6 },
//   { text: '어서', pos: 'EC',  start: 3, end: 6 },
//   { text: '약',   pos: 'NNG', start: 7, end: 9 },
//   { text: '을',   pos: 'JKO', start: 7, end: 9 },
//   { text: '먹',   pos: 'VV',  start: 10, end: 13 },
//   { text: '었',   pos: 'EP',  start: 10, end: 13 },
//   { text: '다',   pos: 'EF',  start: 10, end: 13 },
// ]

// Simple tokenization
const tokens = garu.tokenize('나는 학교에 간다');
// ['나', '는', '학교', '에', '간다']

garu.destroy(); // free WASM memory
```

## Custom Model

```typescript
// Load from custom URL
const garu = await Garu.load({ modelUrl: '/models/custom.gmdl' });

// Load from ArrayBuffer
const res = await fetch('/models/custom.gmdl');
const garu = await Garu.load({ modelData: await res.arrayBuffer() });
```

## Unspaced Korean text

Import `garu-ko/text` to infer missing spaces and sentence boundaries. These synchronous utilities use bundled JavaScript weights and need no `Garu.load()`, additional WASM load, CDN or server calls. The boundary weights belong to this optional entry point; morphological analysis does not apply restoration automatically.

```typescript
import { restoreSpacing, splitUnspaced } from 'garu-ko/text';

restoreSpacing('나는밥을먹었다.');
// '나는 밥을 먹었다.'

restoreSpacing('나는밥을먹었다.', { mode: 'balanced' });

splitUnspaced('안녕하세요반갑습니다');
// [
//   { text: '안녕하세요', offset: 0 },
//   { text: '반갑습니다', offset: 5 },
// ]
```

- `restoreSpacing(text, { mode? })` inserts spaces while preserving original characters and existing whitespace. The default `conservative` mode may leave some missing spaces unresolved. `balanced` recovers more boundaries but can split words too aggressively.
- `splitUnspaced(text)` returns original substrings and their starting `offset`, without inserting spaces. Segment edges are trimmed. Offsets count Unicode code points in the original input, so they can differ from JavaScript UTF-16 indices.

These are boundary estimates. Ambiguous expressions and quotations can still produce missed or extra splits. Separate URLs from surrounding Korean text with whitespace, especially when the URL contains Unicode. A URL at the start of input or after whitespace is preserved up to the next whitespace.

### Context-verified sentence splitting

`garu-ko/sentences` preserves balanced quotations as single top-level units and verifies proposed boundaries using a small character-context model. Multiple grammatical sentences inside a quotation remain together.

```typescript
import { splitSentences } from 'garu-ko/sentences';

splitSentences('안녕하세요반갑습니다');
// [{ text: '안녕하세요', offset: 0 }, { text: '반갑습니다', offset: 5 }]
splitSentences('“안녕하세요반갑습니다”');
// [{ text: '“안녕하세요반갑습니다”', offset: 0 }]
splitSentences('가방에 쏙! 파우치에 쏙! 너무 좋아요ㅎㅎ\n다음에 또 살게요');
// [{ text: '가방에 쏙!', offset: 0 }, { text: '파우치에 쏙!', offset: 7 },
//  { text: '너무 좋아요ㅎㅎ', offset: 15 }, { text: '다음에 또 살게요', offset: 24 }]
```

No morphological analyzer is needed. The Q4 model ships in the package and inference runs offline. Passing an analyzer as `splitSentences(text, garu)` retains the slower morphology-verification path. Results retain original substrings and Unicode codepoint offsets, trimming segment edges. The existing text utilities and default analyzer keep their behavior and do not import this model.

Every line break ends a sentence; join hard-wrapped lines (for example text copied from a PDF) before calling. Trailing laughter, emoji and tildes stay with the preceding sentence, and spaced input is never split inside a word.

On the inspected, fixed NIKL2025 news set of 9,355 paragraphs, the default call's boundary F1 is 96.56% after whitespace deletion, 96.92% after terminal punctuation deletion as well, and 99.57% with the original spacing. On the held-out Instagram split of the NIKL online posting corpus 2025, paragraph boundary F1 is 77.89% against NIKL labels. Unpunctuated conversational text remains weak: 47.8% on a reviewed sample of 300. On Apple M4 Node, loaded mean latency was about 0.020ms per news paragraph with P95 about 0.06ms (median of seven rounds, 1,250 inputs averaging82.5 codepoints). This is not a per-input guarantee; import and first-call costs are separate. Browser latency was not remeasured for this version. The quote rule was added after inspecting fixed-set errors; reference conventions that split within quotations produce lower scores. See the repository research history for exact evaluation and latency scope.

## API

### `Garu.load(options?): Promise<Garu>`

Initialize WASM and load model. Uses bundled model by default.

| Option | Type | Description |
|---|---|---|
| `modelData` | `ArrayBuffer` | Provide model bytes directly |
| `modelUrl` | `string` | Fetch model from URL |

### `garu.analyze(text, options?): AnalyzeResult`

Returns morphological tokens with POS tags (Sejong tagset).

```typescript
interface Token {
  text: string;   // surface form
  pos: POS;       // POS tag
  start: number;  // eojeol start offset
  end: number;    // eojeol end offset
}
```

Set `options.topN > 1` to get N-best results as an array. Note: topN > 1 is not yet fully supported and may return fewer results.

### `garu.nouns(text, options?): string[]`

Extract nouns (NNG, NNP) from text. Set `options.includeSL` to also include foreign tokens (SL) like "AI", "BM25".

```js
garu.nouns('인공지능 기술이 발전했다');
// ["인공", "지능", "기술", "발전"]

garu.nouns('AI 기술이 발전했다', { includeSL: true });
// ["AI", "기술", "발전"]
```

### `garu.tokenize(text): string[]`

Returns surface-form strings only. Lightweight alternative to `analyze()`.

### `garu.addUserWord(surface, pos, freq?): void`

Register a domain word at runtime — **no model rebuild required**. The word is injected into the lattice and competes with the built-in dictionary on the same cost scale.

```js
garu.nouns('인공지능 기술이 발전했다');
// ["인공", "지능", "기술", "발전"]

garu.addUserWord('인공지능', 'NNG');
garu.nouns('인공지능 기술이 발전했다');
// ["인공지능", "기술", "발전"]
```

`freq` uses the same scale as the built-in dictionary — higher values are more likely to beat a split analysis (default 5000). A user word cannot span whitespace.

Useful for product names, organisations and neologisms that a general corpus does not cover, especially when indexing them as single search terms.

### `garu.addUserWords(words): void`

Register several words at once.

```js
garu.addUserWords([
  { surface: '전세사기', pos: 'NNG' },
  { surface: '탄소중립', pos: 'NNG', freq: 20000 },
]);
```

### `garu.clearUserWords(): void` / `garu.userWordCount(): number`

Remove every registered user word, or count them. With none registered, analysis is identical to using the built-in dictionary alone.

### `garu.destroy(): void`

Free WASM memory. Instance is unusable after this call.

## Integrations

Drop-in tokenizers for popular JS search libraries:

- **[`garu-orama-tokenizer`](https://www.npmjs.com/package/garu-orama-tokenizer)** — Korean tokenization for [Orama](https://github.com/oramasearch/orama)
- **[`garu-minisearch-tokenizer`](https://www.npmjs.com/package/garu-minisearch-tokenizer)** — Korean tokenization for [MiniSearch](https://github.com/lucaong/minisearch)

Both solve the same problem: default tokenizers don't handle Korean particles or verb inflections, so `"먹다"` never matches `"먹었다"` and `"학교"` misses `"학교에"`. These adapters run morphological analysis so the inflections fall off before indexing.

```ts
import { create, insert, search } from '@orama/orama'
import { createTokenizer } from 'garu-orama-tokenizer'

const db = await create({
  schema: { title: 'string' },
  components: { tokenizer: await createTokenizer() }
})
await insert(db, { title: '학교에서 점심을 먹었다' })

await search(db, { term: '먹다' })  // ← matches
```

## FAQ

**What is garu-ko?**
garu-ko (가루/Garu) is a browser-native Korean morphological analyzer. A 1.34MB model and a 447KB WASM engine run entirely in the browser, so it segments Korean text, tags parts of speech, extracts nouns, and tokenizes with no server and no network.

**Does it need a server or API?**
No. It runs 100% client-side via WebAssembly. After the initial load there is no backend call, so it works offline and inside browser extensions, service-worker PWAs, and intranet apps.

**How accurate is it?**
F1 96.18% on a 9,000-sentence human-verified gold testset (ep_norm), and F1 92.32% on a fixed 2025 spoken-language evaluation set.

**Which POS tagset does it use?**
The Sejong tagset (42 tags) from the National Institute of Korean Language — NNG/NNP for nouns, VV/VA for verbs/adjectives, particles, endings, and symbols.

**Does it use a neural network?**
No. It combines a codebook, an eojeol cache, sentence-level N-best trigram Viterbi, deterministic post-processing rules, and a linear reranking perceptron — not a neural network — so output is deterministic and reproducible.

**Where does it run besides the browser?**
The same `npm install garu-ko` works in browser ESM, Node.js 18+, Bun, and Deno with one API.

**Is it free for commercial use?**
The engine is licensed under Apache-2.0, which permits commercial use. Bundled dictionary data also carries Apache-2.0 and CC BY-SA 2.0 KR terms, including attribution and applicable ShareAlike requirements. See the [dictionary data notices](models/NOTICE.md).

## Acknowledgments

The model uses frequency statistics from the **NIKL Morpheme-Tagged Corpus (v1.1)** provided by the National Institute of Korean Language (국립국어원), together with reviewed dictionary entries. The bulk additions come from **mecab-ko-dic**, **한국어기초사전**, and **우리말샘**. Original corpus paragraphs and dictionary definitions are not shipped. Source versions, entry identifiers and license notices accompany the model.

## License

Engine and package code: [Apache-2.0](LICENSE), with project attribution in [NOTICE](NOTICE). Imported dictionary data: [source-specific terms and attribution](models/NOTICE.md).
