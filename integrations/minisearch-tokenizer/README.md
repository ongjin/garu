# garu-minisearch-tokenizer

Korean tokenizer for [MiniSearch](https://github.com/lucaong/minisearch), powered by [garu-ko](https://www.npmjs.com/package/garu-ko).

MiniSearch is delightful for small-to-medium full-text search, but its default tokenizer splits on whitespace and punctuation. Korean doesn't put particles in front of spaces — `"학교에"`, `"학교를"`, and `"학교가"` all look like single words to MiniSearch, and none of them match a query for `"학교"`. This package replaces the tokenizer with real morphological analysis so the index sees `"학교"` once, regardless of which particle is glued to it.

The whole library is basically one function: `(text) => string[]`. The work happens inside `garu-ko`.

## Install

```bash
npm i minisearch garu-minisearch-tokenizer
```

## Use

```ts
import MiniSearch from 'minisearch'
import { createTokenizer } from 'garu-minisearch-tokenizer'

const tokenize = await createTokenizer()

const ms = new MiniSearch({
  fields: ['title', 'body'],
  tokenize,
})

ms.addAll([
  { id: 1, title: '학교에서 점심을 먹었다', body: '' },
  { id: 2, title: '오늘 뭐 먹지', body: '' },
])

ms.search('먹다')
// → both documents come back. the verb stem "먹" was indexed,
// so any inflection of 먹다 finds it.
```

By default MiniSearch uses the same tokenizer for both indexing and search, which is what you want here. If you need a different one at query time (rare for Korean) pass it via `searchOptions.tokenize`.

## What it keeps

Default filter is content-bearing morphemes only:

| kept | examples |
|---|---|
| NNG / NNP | 사과, 학교, 서울 |
| VV / VA stems | 먹, 가, 빠르, 예쁘 |
| SL | AI, BM25, Korean |
| SH | 漢 |
| NR / SN | 하나, 둘, 2024 |
| XR | (rare) |

Dropped: particles, endings, auxiliaries, punctuation. Lowercased so case doesn't matter.

Tweak it if you want:

```ts
await createTokenizer({ posFilter: ['NNG', 'NNP'] })           // nouns only
await createTokenizer({ stopwords: ['것', '수', '때', '거'] })  // drop common noise
```

## Sharing a Garu instance

The model is ~1.9MB. Don't load it twice if you already have an instance lying around:

```ts
import { Garu } from 'garu-ko'
import { createTokenizer } from 'garu-minisearch-tokenizer'

const garu = await Garu.load()
const tokenize = await createTokenizer({ garu })
```

## Options

```ts
interface CreateTokenizerOptions {
  garu?: Garu                  // reuse an existing instance
  posFilter?: Iterable<string> // POS tags to keep (Sejong tagset)
  stopwords?: Iterable<string> // tokens to drop (post-lowercase)
  lowercase?: boolean          // default: true
}
```

## Stopword-only mode

If you don't want full morphological splitting and just want stopword filtering on top of MiniSearch's default tokenizer, there's a helper:

```ts
import MiniSearch from 'minisearch'
import { createProcessTerm } from 'garu-minisearch-tokenizer'

const ms = new MiniSearch({
  fields: ['title', 'body'],
  processTerm: createProcessTerm(['것', '수', '때']),
})
```

This skips Garu entirely. Smaller bundle, less power. Useful when your Korean is well-spaced and you just need to drop a few junk terms.

## Things to keep in mind

- Tokenization is **synchronous after load**. MiniSearch's `tokenize` contract is sync, so this fits naturally.
- Indexed tokens are **morpheme stems**, not surface inflected forms. `"먹었다"` → indexed as `"먹"`. Search for `"먹다"` matches; search for `"먹었"` does not.
- For noisy text (chat, typos, jamo abbreviations like `ㄱㅅ`), run `normalizeText` from `garu-ko` over the content before `addAll`.
- The model + WASM together are about **1.9MB**. If that's a problem for your bundle, MiniSearch's default tokenizer is probably the right call.

## Sibling packages

- [`garu-ko`](https://www.npmjs.com/package/garu-ko) — the morphological analyzer
- [`garu-orama-tokenizer`](https://www.npmjs.com/package/garu-orama-tokenizer) — same idea for [Orama](https://github.com/oramasearch/orama)

## License

MIT
