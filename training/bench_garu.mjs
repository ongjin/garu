// Garu throughput benchmark.
// Usage: node training/bench_garu.mjs <gold_jsonl> <n_sentences>
// Output: JSON to stdout with load_ms, warm_ms, total_ms, sentences, chars.

import { readFileSync } from 'fs';
import { performance } from 'perf_hooks';
import { Garu } from '../js/dist/node.js';

const [, , goldPath, nArg] = process.argv;
if (!goldPath) {
  console.error('usage: node bench_garu.mjs <gold_jsonl> [n]');
  process.exit(1);
}
const n = nArg ? parseInt(nArg, 10) : 1000;

const lines = readFileSync(goldPath, 'utf8').trim().split('\n').slice(0, n);
const sentences = lines.map((l) => JSON.parse(l).text);
const totalChars = sentences.reduce((s, t) => s + t.length, 0);

const t0 = performance.now();
const garu = await Garu.load();
const tLoad = performance.now() - t0;

// warm-up (50 sentences)
for (const s of sentences.slice(0, Math.min(50, sentences.length))) {
  garu.analyze(s);
}

const tStart = performance.now();
for (const s of sentences) {
  garu.analyze(s);
}
const tEnd = performance.now();

console.log(
  JSON.stringify({
    tool: 'garu',
    load_ms: tLoad,
    total_ms: tEnd - tStart,
    sentences: sentences.length,
    chars: totalChars,
    sentences_per_sec: (sentences.length / (tEnd - tStart)) * 1000,
    chars_per_sec: (totalChars / (tEnd - tStart)) * 1000,
  }),
);
