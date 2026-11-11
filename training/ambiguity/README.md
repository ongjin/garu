# Context-dependent POS corrections

Section16 holds58 learned eojeol alternatives and220 context conditions. At final
inference, exactly one POS may change when the whole eojeol form/POS/span matches
the original Section13 cache. Explicit user words take precedence. This does not
add lattice candidates or alter other token fields.

The scripts require original NIKL2021 files, existing reranker train/dev data,
the fixed gold and2025 spoken files for text-only overlap exclusion, and NumPy/
Brotli through the shared evaluators. Source and split hashes are in
`review_20261020.json` and `~/workspace/data/korean_context_pos/20261020/`.

To recreate the immediately preceding dictionary-only baseline from the adopted
model, remove Section16 and recompress the unchanged remaining sections:

```sh
python3 - <<'PYCODE'
import sys, struct, hashlib
from pathlib import Path
import brotli
sys.path.insert(0, 'training/ambiguity')
from context_cache import sections
parts = sections(Path('js/models/base.gmdl').read_bytes())
raw = b'GMDL' + struct.pack('<I', 3)
for kind, blob in parts.items():
    if kind != 16:
        raw += struct.pack('<BI', kind, len(blob)) + blob
model = brotli.compress(raw, quality=11)
assert hashlib.sha256(model).hexdigest() == '5c67af7801cf444229a007dc67b41d486c5a1b80542b235e1b5fc8c0c795686d'
Path('/tmp/garu-context-baseline.gmdl').write_bytes(model)
PYCODE
AMBIGUITY_LAB=$(mktemp -d)
python3 training/ambiguity/prepare.py --out "$AMBIGUITY_LAB" --model /tmp/garu-context-baseline.gmdl
cargo build --release --example analyze_batch
python3 training/ambiguity/probe.py --out "$AMBIGUITY_LAB" --binary target/release/examples/analyze_batch --baseline-binary target/release/examples/analyze_batch --model /tmp/garu-context-baseline.gmdl
python3 -m unittest discover -s training/ambiguity -p 'test_*.py'
```

The same current binary can evaluate baseline and candidate models because absent
Section16 leaves the old behavior unchanged. Historical latency comparisons used
the previous source binary separately. The codec retains a reserved−1.0 arc-cost
field for byte-identical model lineage; final POS correction ignores this field.
The reusable probe evaluates one model and does not sweep that inactive field.

The model builder preserves `training/codebook_data/context_pos.bin`. Section14
training must use `Analyzer::analyze_topn_for_rerank`, which is now used by the
tracked `dump_topk` example. It shares the production reranker preprocessing and
excludes final context-POS correction and display-only jamo normalization.
Public `analyze_topn` includes final corrections. Revalidate the learned context
rules if the underlying cache or analyzer conventions change.

## Section 17: one-POS rules over final eojeol outputs

`context_rules.py` mines rules from the current production output instead of the
Section13 cache. A rule stores the eojeol surface, its exact final form/POS
sequence, one position and the replacement POS, plus up to four neighbor guards
(1-3 codepoint prefix/suffix of the previous/next word, the exact word, or a
sentence edge). An n-codepoint guard only matches neighbors with at least n
codepoints. Runtime order is Section16, then Section17; user words block both.

```sh
LAB=$(mktemp -d)
python3 training/ambiguity/prepare.py --out "$LAB" --model js/models/base.gmdl
cargo build --release --example analyze_batch
for s in train dev; do
  python3 -c "import json,sys; [print(json.loads(l)['text']) for l in open(sys.argv[1])]" "$LAB/$s.jsonl" > "$LAB/$s.txt"
  GARU_MODEL=<model without Section17> target/release/examples/analyze_batch "$LAB/$s.txt" --json-spans > "$LAB/$s.base.jsonl"
done
python3 training/ambiguity/context_rules.py mine "$LAB"
python3 training/ambiguity/context_rules.py select "$LAB"
python3 training/ambiguity/context_rules.py export "$LAB/selected.json" js/models/base.gmdl out.gmdl --blob training/codebook_data/context_rules.bin
```

Mining uses the output of the model without Section17. Selection compares 32
settings on the document-split dev only; fixed sets are judged afterwards with
gate v2 (`docs/wiki/benchmarks.md`). The adopted record is `review_20261102.json`.
It re-implements research-history #75 from its written description and is not
the same rule file.

Development and fixed-set totals improved, but some individual sentences
regressed. The rejected lattice-injection pilot and the selected final-POS
intervention are recorded separately in the research history and review JSON.

### Spoken filler `그` (appended rule, research history #83)

NIKL 2021 SX transcribes the filler as `그~`, so the mined rules only cover that
form. NIKL 2025 has no tildes, and bare `그` was always MM. `spoken_geu.py` strips
tildes from the SX part of the `prepare.py` split, mines neighbor guards for the
single rule (그, [그/MM]) to IC on train, and picks the grid setting with the best
dev SX gain among settings whose dev NX F1 does not fall. It then appends the rule
to the existing Section 17 blob. Guards are sorted before the greedy choice, so
reruns are identical regardless of the Python hash seed.

```sh
LAB=$(mktemp -d)
python3 training/ambiguity/prepare.py --out "$LAB/orig" --model js/models/base.gmdl
python3 training/ambiguity/spoken_geu.py strip "$LAB/orig" "$LAB/strip"
for s in train dev; do
  GARU_MODEL=<model without the 그 rule> target/release/examples/analyze_batch "$LAB/strip/$s.txt" --json-spans > "$LAB/strip/$s.base.jsonl"
done
python3 training/ambiguity/spoken_geu.py mine "$LAB/strip"
python3 training/ambiguity/spoken_geu.py append "$LAB/strip/rule_그.json" <model without the 그 rule> out.gmdl --blob training/codebook_data/context_rules.bin
```

The adopted rule has 32 guards (precision 0.8, support 3, documents 2). The
`--blob` output is the Section 17 source of truth; the model builder then reproduces
the shipped model byte for byte. Re-mining Section 17 with
`context_rules.py` drops this rule, so append it again and re-run gate v2.
