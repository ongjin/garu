"""Tune analyzer parameters by grid search.

Modifies Section 10 (parameters) in the GMDL file and re-runs benchmark.
"""
import json
import os
import random
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
MODEL_PATH = ROOT / "models" / "codebook.gmdl"

POS_TAGS = [
    "NNG", "NNP", "NNB", "NR", "NP",
    "VV", "VA", "VX", "VCP", "VCN",
    "MAG", "MAJ", "MM", "IC",
    "JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ", "JX", "JC",
    "EP", "EF", "EC", "ETN", "ETM",
    "XPN", "XSN", "XSV", "XSA", "XR",
    "SF", "SP", "SS", "SE", "SO", "SW", "SH", "SL", "SN",
]
POS_SET = set(POS_TAGS)
NIKL_DIR = Path.home() / "Downloads" / "NIKL_MP(v1.1)"


def normalize_pos(tag):
    if tag in POS_SET:
        return tag
    NIKL_MAP = {
        'MMD': 'MM', 'MMN': 'MM', 'MMA': 'MM',
        'NA': 'NNG', 'NAP': 'NNG', 'NF': 'NNG', 'NV': 'VV',
    }
    if tag in NIKL_MAP:
        return NIKL_MAP[tag]
    base = tag.split('-')[0]
    if base in POS_SET:
        return base
    return 'SW'


def load_nikl_sentences(max_n=2000):
    sentences = []
    for fname in ["NXMP1902008040.json", "SXMP1902008031.json"]:
        path = NIKL_DIR / fname
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        for doc in data["document"]:
            for sent in doc["sentence"]:
                text = sent["form"]
                if not text or len(text) < 5 or len(text) > 200:
                    continue
                morphemes = []
                for m in sent["morpheme"]:
                    form = m["form"]
                    label = normalize_pos(m["label"])
                    if form.strip():
                        morphemes.append((form, label))
                if morphemes:
                    sentences.append((text, morphemes))
    random.seed(42)
    if len(sentences) > max_n:
        sentences = random.sample(sentences, max_n)
    return sentences


def patch_params(model_bytes, mp, op, lb, sc):
    """Patch Section 10 parameters in GMDL model bytes."""
    result = bytearray(model_bytes)
    pos = 8
    while pos < len(result):
        section_type = result[pos]
        section_len = int.from_bytes(result[pos+1:pos+5], 'little')
        if section_type == 10:
            data_start = pos + 5
            struct.pack_into("<f", result, data_start, mp)
            struct.pack_into("<f", result, data_start + 4, op)
            struct.pack_into("<f", result, data_start + 8, lb)
            struct.pack_into("<f", result, data_start + 12, sc)
            break
        pos += 5 + section_len
    return bytes(result)


def run_and_score(model_bytes, sentences):
    """Write model to temp file, run analyzer, compute F1."""
    with tempfile.NamedTemporaryFile(suffix='.gmdl', delete=False) as mf:
        mf.write(model_bytes)
        model_path = mf.name

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        for text, _ in sentences:
            f.write(text + '\n')
        input_path = f.name

    env = os.environ.copy()
    env['GARU_MODEL'] = model_path
    result = subprocess.run(
        ["cargo", "run", "--release", "--example", "analyze_batch", "--", input_path],
        cwd=str(ROOT), capture_output=True, text=True, timeout=300, env=env,
    )
    os.unlink(input_path)
    os.unlink(model_path)

    if result.returncode != 0:
        return 0.0

    analyses = []
    current = []
    for line in result.stdout.strip().split('\n'):
        line = line.strip()
        if line == '---':
            analyses.append(current)
            current = []
        elif line == '[]':
            analyses.append([])
        elif '\t' in line:
            form, pos = line.split('\t', 1)
            current.append((form, pos))
    if current:
        analyses.append(current)

    total_match = 0
    total_pred = 0
    total_gold = 0
    for i in range(min(len(analyses), len(sentences))):
        pred_set = set((f, t) for f, t in analyses[i] if f.strip())
        gold_set = set((f, t) for f, t in sentences[i][1] if f.strip())
        total_match += len(pred_set & gold_set)
        total_pred += len(pred_set)
        total_gold += len(gold_set)
    P = total_match / max(total_pred, 1)
    R = total_match / max(total_gold, 1)
    F = 2 * P * R / max(P + R, 1e-10)
    return F


def main():
    print("Loading sentences...")
    sentences = load_nikl_sentences(2000)
    print(f"  {len(sentences)} sentences")

    print("Loading base model...")
    with open(MODEL_PATH, 'rb') as f:
        base_model = f.read()

    # Current best: mp=0.25, op=4.0, lb=1.5, sc=3.5
    mp_values = [0.15, 0.20, 0.25, 0.30, 0.35]
    op_values = [3.5, 4.0, 4.5]
    lb_values = [1.0, 1.25, 1.5, 1.75, 2.0]
    sc_values = [3.0, 3.5, 4.0]

    best_f1 = 0
    best_params = None
    results = []

    total = len(mp_values) * len(op_values) * len(lb_values) * len(sc_values)
    done = 0

    for mp in mp_values:
        for op in op_values:
            for lb in lb_values:
                for sc in sc_values:
                    done += 1
                    patched = patch_params(base_model, mp, op, lb, sc)
                    f1 = run_and_score(patched, sentences)
                    results.append((f1, mp, op, lb, sc))
                    if f1 > best_f1:
                        best_f1 = f1
                        best_params = (mp, op, lb, sc)
                        print(f"  [{done}/{total}] NEW BEST: F1={f1:.4f} mp={mp} op={op} lb={lb} sc={sc}")
                    elif done % 25 == 0:
                        print(f"  [{done}/{total}] F1={f1:.4f} mp={mp} op={op} lb={lb} sc={sc}")

    print(f"\n{'='*60}")
    print(f"  BEST: F1={best_f1:.4f}")
    print(f"  Params: mp={best_params[0]} op={best_params[1]} lb={best_params[2]} sc={best_params[3]}")
    print(f"{'='*60}")

    results.sort(key=lambda x: -x[0])
    print(f"\nTop 10 configurations:")
    for f1, mp, op, lb, sc in results[:10]:
        print(f"  F1={f1:.4f}  mp={mp}  op={op}  lb={lb}  sc={sc}")


if __name__ == "__main__":
    main()
