# Character-boundary experiments

These scripts train optional spacing and sentence-boundary predictors. They do
not change Garu's morphological model or its normal analysis path. Python needs
NumPy, SciPy and Brotli. The TypeScript export test also needs Node and the dependencies installed by `npm ci` in `js/`. Keep all corpora, generated examples and checkpoints
outside the repository.

```sh
BOUNDARY_LAB=$(mktemp -d)
python3 training/boundaries/boundary.py prepare \
  --out "$BOUNDARY_LAB/pilot" --train-per-source 15000 --eval-per-source 1500
python3 training/boundaries/speech_variant.py \
  "$BOUNDARY_LAB/pilot" "$BOUNDARY_LAB/speech"
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 training/boundaries/boundary.py train \
  --out "$BOUNDARY_LAB/speech" --bits 18 --iterations 160 \
  --speech-weight 10 --sentence-threshold macro
python3 -m unittest discover -s training/boundaries -p 'test_*.py'
```

`prepare --fixed-eval <previous-output>` retains the previous development and
test records while increasing training coverage. `--train-per-source 1000000000`
uses every eligible training document. Source SHA256, sentence IDs, document IDs,
exclusions and normalization are recorded in `manifest.json` and the generated
JSONL files. Do not add those JSONL files to Git.

Document IDs determine the split. Fixed gold, the existing reranker development
texts and 2025 SX are excluded from training, including whitespace and terminal
punctuation deletion variants. Underscores in training text become spaces.
Runtime inference preserves underscores and all existing whitespace; it inserts
spaces only. Offsets and hash inputs are Unicode code points. Runtime whitespace
matches JavaScript `\s`, and quantized integer weights are summed before one
multiplication by the head scale.

`report.json` contains development-selected thresholds and fixed-set metrics.
`export.json` and `weights.b64` describe the int8 weights and hash contract for
JavaScript. `infer.py` accepts JSON strings on stdin and emits restored JSON
strings, one per line. `BoundaryModel.restore_many()` supports batch experiments.

The spacing head uses the full eligible training pool. The sentence head was first
a 100,000-source-sentence model and, since research history #83, adds Instagram
supervision (see "Instagram sentence head" below). Both use 2^19 buckets, the
original 13 n-gram features, L2=1e-5 and speech weight10. Recreate the shared evaluation split and both training pools before
selecting heads:

```sh
python3 training/boundaries/boundary.py prepare --out "$BOUNDARY_LAB/clean" --train-per-source 15000 --eval-per-source 1500
python3 training/boundaries/boundary.py prepare --out "$BOUNDARY_LAB/large_raw" --train-per-source 50000 --fixed-eval "$BOUNDARY_LAB/clean"
python3 training/boundaries/speech_variant.py "$BOUNDARY_LAB/large_raw" "$BOUNDARY_LAB/large19"
python3 training/boundaries/boundary.py prepare --out "$BOUNDARY_LAB/full_raw" --train-per-source 1000000000 --fixed-eval "$BOUNDARY_LAB/clean"
python3 training/boundaries/speech_variant.py "$BOUNDARY_LAB/full_raw" "$BOUNDARY_LAB/full19"
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 training/boundaries/boundary.py train --out "$BOUNDARY_LAB/large19" --bits 19 --iterations 160 --speech-weight 10 --sentence-threshold macro
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 training/boundaries/boundary.py train --out "$BOUNDARY_LAB/full19" --bits 19 --iterations 160 --speech-weight 10 --sentence-threshold macro
```

These commands require the original NIKL2021/2025 files plus
`training/rerank/data/dev_texts.txt` and the gold JSONL, with the hashes recorded
in the retained manifests. Optimizer libraries and hardware can affect a
retrained weight file; compare metrics and hashes instead of assuming identity.
The checked-in TypeScript weights are the adopted runtime artifact.

Compatible heads can be selected independently without retraining. Selection
checks the complete hash contract, quantized weights and common dev/test hashes:

```sh
python3 training/boundaries/export.py select \
  --spaces "$BOUNDARY_LAB/full19" --sentences "$BOUNDARY_LAB/large19" \
  --out "$BOUNDARY_LAB/selected"
python3 - "$BOUNDARY_LAB/selected/export.json" <<'PYCODE'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
metadata = json.loads(path.read_text())
metadata['heads']['spaces']['conservative_threshold'] = 1.0
path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
PYCODE
python3 training/boundaries/export.py typescript \
  --model "$BOUNDARY_LAB/selected" --out "$BOUNDARY_LAB/boundary-data.ts"
python3 training/boundaries/export.py refs \
  --model "$BOUNDARY_LAB/selected" --out /path/to/reference-archive
```

The selection preserves each head's balanced threshold. Any conservative wrapper
threshold is a separate later choice. `source-manifest.json` records head/model,
dataset and source hashes. The reference archive stores compressed IDs, lengths
and verification hashes; it includes neither source text nor model weights.
TypeScript output explicitly declares `BOUNDARY_WEIGHTS: string` to avoid a large
literal type in declaration files. It also emits `scoreBoundary`, an unrolled FNV
sum specialized to the exported feature list and dimensions. This preserves the
integer scores without runtime `eval` or `Function` compilation. Re-export the
scorer together with the weights. Selection and reference archive destinations
must be empty to preserve existing annotations and later threshold edits.
The TypeScript exporter retains optional `heads.spaces.conservative_threshold`
metadata. Python `BoundaryModel.restore_many()` defaults to the balanced threshold;
the product API may default to conservative mode. Pass an explicit threshold when
using Python to reproduce that product policy. Protection gates in the JavaScript
wrapper are also separate from the Python raw-model inference.

The 2021 sentence examples join up to three consecutive source records. Their
order is original, but the inserted whitespace is synthetic. The two variants
of one block are correlated observations. NX sentence supervision requires
terminal punctuation; optional SX supervision requires every constituent
utterance's final non-punctuation POS to be EF. That is a weak completeness
filter, not independently reviewed sentence gold. A single SX record can contain
several complete sentences, so its internal boundaries can remain mislabeled.

The internal-label follow-up is recorded in `review_labels_20261020.json`.
Adding SF-backed internal boundaries, masking uncertain EF positions, and deleting
the new boundaries' terminal punctuation did not establish a training benefit over
a matched threshold-only control. Those weights and threshold changes were rejected;
the runtime above remains unchanged. A source EF is not automatically a sentence
boundary: reported speech, postposed phrases and fillers require separate review.
Reviewed source IDs, codepoint offsets and text hashes are retained under
`~/workspace/data/korean_text_boundaries/20261020-label-review/`. The training-candidate
review and held-out review are separate files; do not move the held-out annotations
into training. They are exploratory single-pass annotations, not independently
adjudicated gold. Reconstruct source text from the licensed original and verify its
SHA256 before using a reference. The temporary fitting scripts and rejected models
were removed after recording results; their historical paths are not runnable commands.
`comparison_20261020.json` separately records a frozen-input reference comparison
with Kiwi0.23.0/CoNg. Its sentence-boundary F1 on the same9,355 natural paragraphs
is91.24% after whitespace deletion and92.33% after terminal deletion as well.
Kiwi's training lineage was not audited, so this is not a controlled training
comparison or a claim about the accuracy ceiling. It does not change Garu's runtime.

For an external check, `eval_natural.py <model-output> --training <training-output>`
aligns 2025 NX `sentence.original_form` exactly to the original `paragraph.form`,
rejects alignment failures and training overlaps, and evaluates whitespace
removal with and without terminal punctuation. It uses the already selected
threshold. News quotation conventions and incomplete utterances still require
separate interpretation. If fixed-set results have been inspected across model
experiments, describe them as exploratory fixed-set validation, not an untouched
final test.

Evaluate the compiled public JavaScript API, including its protection gates and
default conservative mode, separately from Python raw-model measurements:

```sh
python3 training/boundaries/eval_api.py \
  --model "$BOUNDARY_LAB/selected" --javascript js/dist/text.js \
  --out "$BOUNDARY_LAB/public-js-evaluation.json"
```

This checks the exact-aligned 2025 NX paragraphs and common development/test
spacing records. Partial-spacing input keeps every other gold space. Normal-input
preservation describes explicit `restoreSpacing` calls, not the unchanged default
analyzer. The report records runtime/source hashes, original-character invariants
and frozen thresholds; it contains no corpus text and does not retune the model.

For the adopted default, set `heads.spaces.conservative_threshold` in the selected
`export.json` to1.0 before emitting TypeScript. It was selected from thresholds
0/0.5/1/1.5/2 using the frozen base analyzer on the existing 6,000-sentence dev;
`review_20261020.json` records the separate boundary-F1 and morphology objectives.
`eval_api.py` evaluates the actual JavaScript wrapper, including protection gates,
Unicode positions and conservative/balanced modes; use its `--help` for paths.

## Instagram sentence head

The adopted sentence head (research history #83) adds Instagram paragraphs from the
NIKL online posting corpus 2025 to the `large19` 2021 pool. Document buckets
`sha256(id)[:8] mod 10` are 0-1 final test, 2-3 dev and 4-9 train. Those sentence
records are punctuation-driven, so only positions after terminal punctuation
(optionally followed by laughter, emoticons or closing brackets) carry loss, with
weight3; all other Instagram positions have weight0. Training on every position
taught the head not to split unpunctuated news. The threshold stays at the 2021
value -0.75; dev macro selection preferred -0.5, and -0.75 was kept after viewing
the fixed news set. The spacing head is unchanged.

```sh
SNS_LAB=$(mktemp -d)
python3 training/boundaries/sns_sentences.py prepare --out "$SNS_LAB/sns"
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 training/boundaries/sns_sentences.py train \
  --base "$BOUNDARY_LAB/large19" --sns "$SNS_LAB/sns" --out "$SNS_LAB/head"
python3 training/boundaries/sns_sentences.py apply --npz "$SNS_LAB/head/sentences.npz" --ts js/src/boundary-data.ts
```

Retraining reproduced the adopted head byte for byte (SHA-256
`685e44c82b4ffeed4c00289204ea865b309a078ba34942bff5e10bc7deb901a4`; combined
weights `c8a4aa2bc5a85130e01204bfd09d4612340eadd1f3b4dfd811bcaf991eb380c9`).
`apply` replaces only the sentence half and its scale/threshold in the generated
file. `splitUnspaced` uses the same head.

`garu-ko/sentences` adds three deterministic steps after verification: every line
break ends a sentence, a boundary followed by trailing ornaments (whitespace,
`~!?.…^`, jamo, emoji, closing brackets) moves past them and disappears at the end
of input, and in spaced input (at least one word gap per ten Hangul syllables) a
boundary inside a word is dropped unless it follows punctuation or an ornament.

## Optional context verifier

`garu-ko/sentences` adds balanced-quotation preservation and a small context
verifier to the frozen sentence candidates. Its default `splitSentences(text)`
uses character context without loading a morphological analyzer. Passing a loaded
analyzer retains the full morphology verifier. Original substrings/codepoint offsets remain intact.
The original `garu-ko/text` functions do not import the new model.

Training additionally needs PyTorch. Install workspace JS dependencies and build
the JS package before mining, so its relative WASM/model assets resolve normally.
All generated examples, features and alternative weights belong outside the repo:

```sh
npm ci
npm run build:js --workspace=garu-ko
VERIFIER_LAB=$(mktemp -d)
python3 training/boundaries/prepare_verifier.py --out "$VERIFIER_LAB/data"
python3 training/boundaries/mine_verifier.py --dataset "$VERIFIER_LAB/data/train.jsonl" --out "$VERIFIER_LAB/train-features.jsonl" --runtime js/dist --random-controls
python3 training/boundaries/mine_verifier.py --dataset "$VERIFIER_LAB/data/dev.jsonl" --out "$VERIFIER_LAB/dev-features.jsonl" --runtime js/dist
python3 training/boundaries/mine_verifier.py --dataset "$VERIFIER_LAB/data/test.jsonl" --out "$VERIFIER_LAB/test-features.jsonl" --runtime js/dist
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 training/boundaries/verifier.py --lab "$VERIFIER_LAB" --modes full structured no_morph easy
python3 training/boundaries/export_verifier.py --model "$VERIFIER_LAB/models/full.json" --out "$VERIFIER_LAB/sentence-data.ts"
python3 -m unittest discover -s training/boundaries -p 'test_*.py'
```

The fixed previous boundary references under
`~/workspace/data/korean_text_boundaries/20261020/` are required exclusions, in
addition to original NIKL corpora, gold and the legacy reranker dev texts. The
preparer verifies NX sentence numbering within paragraph IDs; original2021
paragraph text is absent. Source sentences are joined with synthetic whitespace.
SX requires one final EF, no earlier SF and no quotation marks; terminal
punctuation is optional. These SX labels remain weak supervision.

Mining uses the same compiled JS feature extractor as inference. The easy-negative
control matches each source row's hard-negative count and uses the same positive
positions. Vocabularies use training only, while quantized epoch and threshold
selection use development macro F1. Missing first-stage candidates count as FN.
Feature caches are accepted only when source and feature-file SHA256 match.
The explicit embedding mask prevents the observed PyTorch2.8 MPS padding-gradient
defect; zero padding and finite inference are checked before model delivery.

For frozen end-to-end comparisons, the evaluator prepares `natural.jsonl` using
exact paragraph alignment and `reviewed.jsonl` using the retained held-out IDs
and offsets after verifying original hashes. The reviewed set is small and not
independent expert gold. Neither file is a training input:

```sh
python3 training/boundaries/eval_verifier.py --lab "$VERIFIER_LAB" --runtime js/dist
```

`review_verifier_20261020.json` records all measured paths. On the inspected2025NX
set, the frozen full verifier before the quote rule reached85.56/86.06% F1;
quote protection alone reached94.17/95.01%; the combined API reached95.73/96.30%
for whitespace-only / whitespace-plus-terminal deletion. Thus most of the gain
comes from the explicit quote policy. Original2021 labels sometimes split inside
balanced quotations and score lower under this API. Keep raw reference metrics
and the post-selection caveat; do not describe these numbers as universal95%
accuracy or as a neural-only gain.

## Fast sentence path

The default uses context24, embedding8 and hidden32, with the nine morphology
feature slots zeroed in both training and inference. The selected full and fast
models remain separate; importing the sentence entry parses both weight modules,
but only the requested model is decoded. Loaded latency excludes that import and
first-call cost.

To reproduce the fast model, use a fresh lab and the same exclusions:

```sh
FAST_LAB=$(mktemp -d)
python3 training/boundaries/prepare_verifier.py --out "$FAST_LAB/data"
python3 training/boundaries/mine_verifier.py --dataset "$FAST_LAB/data/train.jsonl" --out "$FAST_LAB/train-features.jsonl" --runtime js/dist --features context --random-controls
python3 training/boundaries/mine_verifier.py --dataset "$FAST_LAB/data/dev.jsonl" --out "$FAST_LAB/dev-features.jsonl" --runtime js/dist --features context
python3 training/boundaries/mine_verifier.py --dataset "$FAST_LAB/data/test.jsonl" --out "$FAST_LAB/test-features.jsonl" --runtime js/dist --features context
python3 training/boundaries/verifier.py --lab "$FAST_LAB" --context 24 --embedding 8 --hidden 32 --modes no_morph
python3 training/boundaries/export_verifier.py --model "$FAST_LAB/models/no_morph.json" --out "$FAST_LAB/sentence-fast-data.ts"
```

The same-seed reproduction selected the same epoch/threshold and dev/test counts;
MPS produced small scale/bias differences, so byte-identical retraining is not
assumed. The generated `js/src/sentence-fast-data.ts` is the canonical artifact.
The selected model is trained on gold/weak-label candidates, not on outputs from
the full verifier. Test references are never used to select the checkpoint.

Reconstruct the exact-aligned natural paragraphs and measure the compiled default:

```sh
python3 - "$FAST_LAB" <<'PYCODE'
import json, sys
from pathlib import Path
sys.path.insert(0, 'training/boundaries')
from eval_natural import load_natural
lab = Path(sys.argv[1])
rows, info = load_natural([lab / 'data/train.jsonl'],
    Path.home() / 'workspace/data/nikl_mp_2025/NXMP2502512313.json')
(lab / 'natural.json').write_text(json.dumps(rows, ensure_ascii=False))
(lab / 'natural-scope.json').write_text(json.dumps(info, ensure_ascii=False, indent=2))
PYCODE
node training/boundaries/bench_sentences.mjs js/dist "$FAST_LAB/natural.json" "$FAST_LAB/latency.json"
```

An optional fourth argument supplies a frozen baseline `dist` directory with its
sibling WASM/model assets. The harness rotates paths, warms50 inputs, and times
seven passes over1,250 inputs averaging82.5 codepoints. It separately reports
individual-input P95 and startup costs, and evaluates all9,355 paragraphs in both
variants without changing labels. See `review_speed_20261020.json` for the actual
Node/Chrome measurements, model selection, raw2021 references and quote-policy
scope. The default's fixed news F1 is95.91/96.28%; the loaded mean meets45µs on the
measured M4, while Node P95 exceeds45µs. Neither result is an all-input guarantee.

Retained original-free references, matched negative positions and the selected
artifact lineage are under `~/workspace/data/korean_sentence_verifier/20261020/`.
The adopted model lives in `js/src/sentence-data.ts`; discarded checkpoints,
duplicate inputs and temporary worktrees are removed after application checks.
