# garu-ko

**Browser-native Korean morphological analyzer.** No server required.

- **2.2MB model** bundled in npm package (no CDN needed)
- **93KB WASM** engine -- runs in any modern browser
- **F1 95.3%** on NIKL MP benchmark (vs. Kiwi 87.9%)
- **< 1ms** inference per sentence
- **Offline-ready** -- works without network

## Comparison

|  | Kiwi | MeCab-ko | garu-ko |
|---|---|---|---|
| Model size | ~40MB | ~50MB | **2.2MB** |
| npm package | No | No | **Yes** |
| F1 (NIKL MP) | 87.9% | ~85% | **95.3%** |
| Browser support | Impractical | No | **Yes** |

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

Set `options.topN > 1` to get N-best results as an array.

### `garu.tokenize(text): string[]`

Returns surface-form strings only. Lightweight alternative to `analyze()`.

### `garu.destroy(): void`

Free WASM memory. Instance is unusable after this call.

## License

MIT
