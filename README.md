# 가루 (Garu)

**Ultra-lightweight Korean morphological analyzer for the web (~5MB)**

텍스트를 곱게 갈아주는 초경량 한국어 형태소 분석기.

---

## Comparison

|                   | Kiwi WASM       | Garu            |
|-------------------|-----------------|-----------------|
| Model size        | ~40MB           | ~5MB            |
| Load time         | 2-5s            | < 500ms         |
| Mobile web        | Impractical     | Fully supported |
| Offline support   | Difficult       | Built-in        |

Garu achieves its small footprint through INT8 quantized BiLSTM weights, a LOUDS-compressed trie dictionary, and a compact binary model format.

---

## Quick Start

```bash
npm install garu
```

```typescript
import { Garu } from 'garu';

// Load the analyzer (fetches the model from CDN by default)
const garu = await Garu.load();

// Morphological analysis
const result = garu.analyze('아버지가방에들어가신다');
console.log(result.tokens);
// [{ text: '아버지', pos: 'NNG', start: 0, end: 3 }, ...]

// Quick tokenization (surface forms only)
const tokens = garu.tokenize('나는 학교에 간다');
console.log(tokens);
// ['나', '는', '학교', '에', '간다']

// Free resources when done
garu.destroy();
```

### Loading with custom model

```typescript
// From a custom URL
const garu = await Garu.load({ modelUrl: '/models/base.gmdl' });

// From an ArrayBuffer
const response = await fetch('/models/base.gmdl');
const modelData = await response.arrayBuffer();
const garu = await Garu.load({ modelData });
```

---

## API Reference

### `Garu.load(options?): Promise<Garu>`

Static factory method. Initializes the WASM module and loads the model.

| Option      | Type          | Description                              |
|-------------|---------------|------------------------------------------|
| `modelData` | `ArrayBuffer` | Provide model bytes directly             |
| `modelUrl`  | `string`      | Fetch model from a custom URL            |

If neither option is provided, the model is fetched from the default CDN.

### `garu.analyze(text, options?): AnalyzeResult | AnalyzeResult[]`

Performs morphological analysis on the input text.

- `text` — Korean text to analyze.
- `options.topN` — When greater than 1, returns an array of N-best results.

Returns an `AnalyzeResult`:

```typescript
interface AnalyzeResult {
  tokens: Token[];  // Array of morphological tokens
  score: number;    // Path score from CRF decoder
  elapsed: number;  // Processing time in milliseconds
}

interface Token {
  text: string;     // Surface form
  pos: POS;         // Part-of-speech tag (Sejong tagset)
  start: number;    // Start offset in the input
  end: number;      // End offset in the input
  score?: number;   // Token-level score
}
```

### `garu.tokenize(text): string[]`

Returns an array of surface-form strings. A lightweight alternative to `analyze()` when you only need the segmented text.

### `garu.isLoaded(): boolean`

Returns `true` if the WASM analyzer is initialized and ready.

### `garu.modelInfo(): ModelInfo`

Returns metadata about the loaded model.

```typescript
interface ModelInfo {
  version: string;  // Model version
  size: number;     // Model size in bytes
  accuracy: number; // Reported accuracy
}
```

### `garu.destroy(): void`

Frees the WASM instance and releases memory. The instance cannot be used after calling this method.

---

## Development

### Prerequisites

- Rust toolchain (stable)
- `wasm-pack`
- Node.js >= 18
- Python >= 3.10 (for training)

### Project Structure

```
crates/
  garu-core/    # Rust core: jamo, trie, BiLSTM, CRF, model loader
  garu-wasm/    # WASM bindings via wasm-bindgen
js/             # TypeScript API wrapper (npm package)
training/       # Python training pipeline
models/         # Compiled model files (.gmdl)
```

### Build

```bash
# Build the Rust core library
cargo build --release

# Build the WASM package
wasm-pack build crates/garu-wasm --target web --out-dir ../../js/pkg

# Build the TypeScript wrapper
cd js && npx tsc

# Or build everything at once
cd js && npm run build
```

### Test

```bash
# Rust tests
cargo test

# JS/TS tests
cd js && npm test
```

### Training Pipeline

The training pipeline uses PyTorch and exports models in safetensors format, which are then converted to the compact `.gmdl` binary format.

```bash
cd training
pip install -r requirements.txt

# Preprocess corpus
python preprocess.py --input <corpus> --output data/

# Train the model
python train.py --data data/ --output checkpoints/

# Evaluate
python evaluate.py --checkpoint checkpoints/best.pt --data data/

# Export to safetensors, then to .gmdl
python export.py --checkpoint checkpoints/best.pt --output ../models/base.gmdl
```

---

## License

MIT
