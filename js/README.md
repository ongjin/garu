# garu-ko

**Browser-native Korean morphological analyzer.** No server required.

- **1MB model** bundled in npm package (no CDN needed)
- **337KB WASM** engine (155KB gzipped) -- runs in any modern browser
- **F1 95.4%** on 9k human-verified gold testset (ep_norm), **F1 93.9%** on NIKL MP
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
garu-ko (가루/Garu) is a browser-native Korean morphological analyzer. A 1MB model and a 337KB WASM engine run entirely in the browser, so it segments Korean text, tags parts of speech, extracts nouns, and tokenizes with no server and no network.

**Does it need a server or API?**
No. It runs 100% client-side via WebAssembly. After the initial load there is no backend call, so it works offline and inside browser extensions, service-worker PWAs, and intranet apps.

**How accurate is it?**
F1 95.4% on a 9,000-sentence human-verified gold testset (ep_norm), and F1 93.9% on the NIKL Modu morpheme-tagged corpus.

**Which POS tagset does it use?**
The Sejong tagset (42 tags) from the National Institute of Korean Language — NNG/NNP for nouns, VV/VA for verbs/adjectives, particles, endings, and symbols.

**Does it use a neural network?**
No. It combines a codebook, an eojeol cache, sentence-level N-best trigram Viterbi, and deterministic post-processing rules, so output is deterministic and reproducible.

**Where does it run besides the browser?**
The same `npm install garu-ko` works in browser ESM, Node.js 18+, Bun, and Deno with one API.

**Is it free for commercial use?**
Yes. garu-ko is open source under the MIT license with no usage fees.

## Acknowledgments

The morphological analysis model is trained on the **NIKL Morpheme-Tagged Corpus (v1.1)** provided by the National Institute of Korean Language (국립국어원). The model contains only derived frequency statistics, not original text.

## License

MIT
