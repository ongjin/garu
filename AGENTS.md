# AGENTS.md

## Purpose

Garu is a browser-first, lightweight Korean morphological analyzer. The core research goal is to keep the analyzer practical for client-side use while approaching server-grade analyzers in accuracy.

Primary constraints:

- Runs fully offline in the browser through WASM.
- Ships model files inside the npm package; do not require CDN or server calls.
- Keep inference cheap: prefer lookup tables, Viterbi, compact rules, caches, and small quantized models.
- Treat model size, latency, and F1 as a single tradeoff. Accuracy gains that make browser delivery impractical are not acceptable by default.

## Source Priority

- `CLAUDE.md` is the current operational snapshot for architecture, file map, commands, and local rules.
- `docs/paper.md` is the research narrative: why earlier approaches failed, how each optimization helped, and what open research directions remain.
- Code, model artifacts, and fresh benchmark output override both documents when they disagree.
- If reporting metrics externally, rerun or cite the exact benchmark command and dataset split. The paper and `CLAUDE.md` contain different historical metric snapshots.

## Current Architecture

Pipeline:

```text
input text
  -> sentence-level lattice construction
     - FST content-word dictionary
     - suffix codebook
     - smart eojeol cache as low-cost arcs
     - typo-tolerant OOV arcs
  -> trigram N-best Viterbi decoding
  -> postprocessing rules
     - VCP split
     - VX/JC/JKC/NNB/XSN/XPN/MM/XSV/XSA corrections
  -> CNN reranking / agreement scoring
  -> token output
```

Important model pieces:

- `codebook.gmdl` / `js/models/base.gmdl`: GMDL binary model with FST dictionary, suffix codebook, sparse trigram costs, word-bigram adjustments, parameters, and smart eojeol cache.
- `cnn2.bin`: small int8 1D CNN used for reranking and POS correction.
- `crates/garu-core/src/model.rs`: analyzer orchestration.
- `crates/garu-core/src/codebook.rs`: lattice construction, Viterbi, cache injection, and postprocessing.
- `crates/garu-core/src/cnn.rs`: quantized CNN inference.
- `crates/garu-core/src/trie.rs`: FST dictionary and multi-POS packing.
- `crates/garu-core/src/types.rs`: Sejong POS tags.
- `crates/garu-wasm/src/lib.rs`: WASM bindings.
- `js/src/index.ts`: public JS/TS API.

## Research Lessons

Do not restart from approaches already shown to be poor fits unless the goal is explicitly to reproduce or disprove them:

- BiLSTM distillation failed because matrix-heavy inference is not browser-friendly.
- Jamo-level sequence labeling failed because it lengthened sequences and destroyed useful syllable-level signals.
- Ambiguity-table expansion hurt accuracy by adding too many noisy candidates.
- Self-learning frequency loops did not converge reliably.
- Perceptron POS reranking produced only tiny gains because many remaining errors are segmentation errors.
- Decomposition preference penalties increased over-segmentation.
- Cost-scale normalization did not improve over tuned parameters.

Strong historical wins:

- Switching from Kiwi-derived output to NIKL gold-derived codebook patterns was the largest improvement.
- Lowering `morpheme_penalty` made multi-morpheme suffix patterns usable.
- Removing noisy Wikipedia NNP entries improved both size and accuracy.
- Multi-POS FST, sparse trigram quantization, and word-bigram costs gave small but clean gains.
- Smart eojeol cache broke the codebook-only ceiling by caching high-correction-value words, not merely frequent words.
- Contextual postprocessing rules gave free accuracy without growing the model.
- Sentence-level Viterbi allows cached analyses to be overridden by stronger sentence context.
- N-best Viterbi plus CNN agreement scoring is the main current path for better ambiguity handling.

## Research Priorities

Prefer work that targets known residual error classes:

- Segmentation errors: the dominant remaining error source. Investigate Viterbi top-N candidates, CNN reranking, and candidate generation changes that can actually change boundaries.
- POS ambiguity: continue using sentence context, word-bigram rules, CNN confidence, and targeted postprocessing.
- OOV and neologisms: explore syllable-pattern or CNN-assisted POS inference without adding large neural inference cost.
- Typos and spacing noise: improve typo arcs, noisy training, and pre/postprocessing while measuring clean-text regressions.
- Domain robustness: compare same-domain, split, and cross-domain NIKL results. Do not optimize only one split.
- Model-size discipline: record size deltas for every model or dictionary change.

## Experiment Protocol

Before changing the analyzer, identify:

- The exact error type being targeted.
- Baseline F1 and model size.
- The expected mechanism of improvement.
- The rollback criterion if accuracy, size, or speed regresses.

After changing the analyzer, check at least the focused failure examples and one aggregate benchmark. For broad model changes, compare:

- Gold testset F1.
- NIKL MP F1 when the dataset is available.
- Model gzip size.
- WASM/package impact if the change touches public delivery.

When analyzing dataset errors, preserve exact sentence references and describe whether the issue is segmentation, POS tagging, OOV handling, spacing, or punctuation classification.

## Common Commands

```bash
# Rebuild model
python3 training/build_codebook_model.py

# Rust tests
cargo test

# WASM build
wasm-pack build crates/garu-wasm --target web --out-dir ../../js/pkg

# NIKL MP benchmark, requires ~/Downloads/NIKL_MP(v1.1)/
python3 training/eval_nikl_mp.py --n 2000

# Gold testset evaluation
python3 training/gold_testset/eval_f1.py
```

Use focused commands first when iterating. Run broader tests before claiming a general improvement.

## Local Rules

- Do not put AI/Claude-related text in commit messages.
- Use git email `dydwls140@naver.com` when committing.
- Do not add speculative design or planning documents to the repo unless the user explicitly asks for that artifact.
- Keep changes narrowly scoped to the current research question.
- Do not overwrite untracked model artifacts or generated files unless the task requires regenerating them.
- If metrics or file sizes are mentioned in docs, verify whether they reflect the current artifacts before treating them as current facts.

