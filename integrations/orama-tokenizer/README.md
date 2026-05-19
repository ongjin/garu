# garu-orama-tokenizer

A Korean tokenizer for [Orama](https://github.com/oramasearch/orama), backed by [garu-ko](https://www.npmjs.com/package/garu-ko). Runs everywhere Orama runs — browser, Node, edge — because the analyzer itself is a 1.9MB WASM blob.

If you've tried building Korean search with Orama's default tokenizer, you already know what's wrong. `"먹었다"` doesn't match `"먹는다"`. Searching for `"학교"` misses `"학교에"`, `"학교를"`, `"학교가"`. Particles aren't word boundaries, but the default splitter treats them like they are. This package fixes that by running real morphological analysis instead of whitespace splitting.

## Install

```bash
npm i @orama/orama garu-orama-tokenizer
```

## Use

```ts
import { create, insert, search } from '@orama/orama'
import { createTokenizer } from 'garu-orama-tokenizer'

const db = await create({
  schema: { title: 'string', body: 'string' },
  components: {
    tokenizer: await createTokenizer(),
  },
})

await insert(db, { title: '학교에서 점심을 먹었다', body: '' })
await insert(db, { title: '오늘 뭐 먹지', body: '' })

const res = await search(db, { term: '먹다' })
// matches both. the verb stem "먹" is what got indexed;
// the inflections "-었다" and "-지" never make it into the index.
```

## What it indexes

By default the tokenizer keeps content-bearing morphemes and drops everything else.

**Kept:** nouns (`NNG`/`NNP`), verb stems (`VV`), adjective stems (`VA`), foreign words (`SL`), numbers (`NR`/`SN`), 한자 (`SH`), roots (`XR`).

**Dropped:** particles like `은/는/이/가/을/를`, endings like `-다/-었/-어서`, auxiliaries, punctuation, and pretty much everything else that carries grammar instead of meaning.

Output is lowercased so an English term inside a Korean document still matches case-insensitively. POS tags are from the standard Sejong tagset — same as `garu-ko`.

Override the defaults whenever the default isn't what you want:

```ts
// nouns only
await createTokenizer({ posFilter: ['NNG', 'NNP'] })

// drop a few common dependent nouns that act like noise
await createTokenizer({ stopwords: ['것', '수', '때', '거'] })
```

## Sharing a Garu instance

The WASM model is ~1.9MB. If your app already loaded `garu-ko` somewhere else, hand the instance over and skip the second load:

```ts
import { Garu } from 'garu-ko'
import { createTokenizer } from 'garu-orama-tokenizer'

const garu = await Garu.load()
const tokenizer = await createTokenizer({ garu })
```

## Options

```ts
interface CreateTokenizerOptions {
  garu?: Garu                  // reuse an existing instance
  posFilter?: Iterable<string> // POS tags to keep (Sejong tagset)
  stopwords?: Iterable<string> // tokens to drop (matched after lowercase)
  lowercase?: boolean          // default: true
}
```

## A few things worth knowing

- **Tokenization is sync after load.** Orama's contract allows async (`Promise<string[]>`), but we return synchronous arrays — every `tokenize()` call is sub-millisecond.
- **You get stems, not surface forms.** `"먹었다"` is indexed as `"먹"`. Searching `"먹다"` works because the same tokenizer runs on the query. Searching `"먹었"` will not — that's an inflected form, not a morpheme.
- **No stemming pass.** Korean doesn't need English-style stemming once you have the morpheme. The morpheme *is* the lemma for our purposes.
- **For chat / OCR / dialect**, run `normalizeText` from `garu-ko` over the text before insert. Otherwise typos and jamo abbreviations (`ㄱㅅ`, `ㅊㅋ`) leak through as noise.

## Bundle size

The tokenizer itself is ~2KB. The heavy bit is `garu-ko`:

- ~93KB WASM glue
- 1.2MB codebook model
- 408KB CNN reranker (gzipped)

Total ~1.9MB. That's smaller than most JS frameworks and *much* smaller than the Java/Python alternatives (Kiwi, MeCab-ko sit at 40–50MB and don't run in browsers anyway).

## Sibling packages

- [`garu-ko`](https://www.npmjs.com/package/garu-ko) — the analyzer
- [`garu-minisearch-tokenizer`](https://www.npmjs.com/package/garu-minisearch-tokenizer) — same idea for [MiniSearch](https://github.com/lucaong/minisearch)

## License

MIT
