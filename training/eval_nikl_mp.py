"""Evaluate on NIKL MP (모두의 말뭉치) benchmark.

Compares both Garu (Rust) and Kiwi against NIKL MP gold annotations.

Usage:
    python training/eval_nikl_mp.py [--n 2000]
"""

import json
import os
import random
import subprocess
import sys
import tempfile
from collections import defaultdict, Counter
from pathlib import Path

NIKL_DIR = Path.home() / "Downloads" / "NIKL_MP(v1.1)"

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


def normalize_pos(tag):
    """Normalize POS tag to our 42-tag set."""
    if tag in POS_SET:
        return tag
    # NIKL-specific tags
    NIKL_MAP = {
        'MMD': 'MM', 'MMN': 'MM', 'MMA': 'MM',  # 관형사 세분류 → MM
        'NA': 'NNG', 'NAP': 'NNG',               # 분석불능 → NNG
        'NF': 'NNG',                               # 명사추정 → NNG
        'NV': 'VV',                                # 용언추정 → VV
    }
    if tag in NIKL_MAP:
        return NIKL_MAP[tag]
    base = tag.split('-')[0]
    if base in POS_SET:
        return base
    for p, d in [('V', 'VV'), ('N', 'NNG'), ('J', 'JX'), ('E', 'EF'), ('X', 'XR')]:
        if tag.startswith(p):
            return d
    return 'SW'


def load_nikl_sentences(max_n=2000):
    """Load sentences with gold morpheme annotations from NIKL MP."""
    sentences = []
    for fname in ["NXMP1902008040.json", "SXMP1902008031.json"]:
        path = NIKL_DIR / fname
        if not path.exists():
            print(f"Warning: {path} not found")
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


def run_rust_analyzer(sentences):
    """Run Rust analyzer on sentences."""
    root = Path(__file__).parent.parent

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        for text, _ in sentences:
            f.write(text + '\n')
        input_path = f.name

    result = subprocess.run(
        ["cargo", "run", "--release", "--example", "analyze_batch", "--", input_path],
        cwd=str(root), capture_output=True, text=True, timeout=300,
    )
    os.unlink(input_path)

    if result.returncode != 0:
        print(f"Rust analyzer failed: {result.stderr[:500]}", file=sys.stderr)
        return None

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
    return analyses


def run_kiwi_analyzer(sentences):
    """Run Kiwi on sentences."""
    from kiwipiepy import Kiwi
    kw = Kiwi()
    results = []
    for text, _ in sentences:
        tokens = [(t.form, normalize_pos(t.tag)) for t in kw.tokenize(text) if t.form.strip()]
        results.append(tokens)
    return results


def compute_f1(predictions, gold_sentences):
    """Compute morpheme-level F1."""
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    total_match = 0
    total_pred = 0
    total_gold = 0

    for i in range(min(len(predictions), len(gold_sentences))):
        pred_set = set((f, t) for f, t in predictions[i] if f.strip())
        gold_set = set((f, t) for f, t in gold_sentences[i][1] if f.strip())

        matched = pred_set & gold_set
        total_match += len(matched)
        total_pred += len(pred_set)
        total_gold += len(gold_set)

        for item in pred_set:
            if item in gold_set:
                tp[item[1]] += 1
            else:
                fp[item[1]] += 1
        for item in gold_set:
            if item not in pred_set:
                fn[item[1]] += 1

    P = total_match / max(total_pred, 1)
    R = total_match / max(total_gold, 1)
    F = 2 * P * R / max(P + R, 1e-10)
    return P, R, F, tp, fp, fn


def print_results(name, P, R, F, tp, fp, fn):
    print(f'\n{"=" * 60}')
    print(f'  {name}')
    print(f'  Precision: {P:.1%}  Recall: {R:.1%}  F1: {F:.1%}')
    print(f'{"=" * 60}\n')

    all_tags = sorted(set(list(tp) + list(fp) + list(fn)),
                      key=lambda t: -(fp[t] + fn[t]))
    print(f'{"POS":<8}{"Prec":>8}{"Rec":>8}{"F1":>8}{"TP":>6}{"FP":>6}{"FN":>6}')
    print('-' * 52)
    for tag in all_tags[:25]:
        t, f_p, f_n = tp[tag], fp[tag], fn[tag]
        p = t / max(t + f_p, 1)
        r = t / max(t + f_n, 1)
        f = 2 * p * r / max(p + r, 1e-10)
        print(f'{tag:<8}{p:>8.1%}{r:>8.1%}{f:>8.1%}{t:>6}{f_p:>6}{f_n:>6}')


def main():
    n = 2000
    if len(sys.argv) > 2 and sys.argv[1] == '--n':
        n = int(sys.argv[2])

    print(f"Loading NIKL MP sentences (max {n})...")
    sentences = load_nikl_sentences(n)
    print(f"  Loaded {len(sentences)} sentences")

    # Garu (Rust)
    print("\nRunning Garu (Rust)...")
    garu_results = run_rust_analyzer(sentences)
    if garu_results:
        P, R, F, tp, fp, fn = compute_f1(garu_results, sentences)
        print_results(f"Garu vs NIKL MP ({len(sentences)} sentences)", P, R, F, tp, fp, fn)
        garu_f1 = F
    else:
        garu_f1 = 0

    # Kiwi
    print("\nRunning Kiwi...")
    kiwi_results = run_kiwi_analyzer(sentences)
    P, R, F, tp, fp, fn = compute_f1(kiwi_results, sentences)
    print_results(f"Kiwi vs NIKL MP ({len(sentences)} sentences)", P, R, F, tp, fp, fn)
    kiwi_f1 = F

    # Summary
    print(f'\n{"=" * 60}')
    print(f'  SUMMARY (NIKL MP {len(sentences)} sentences)')
    print(f'  Garu:  F1 = {garu_f1:.1%}')
    print(f'  Kiwi:  F1 = {kiwi_f1:.1%}')
    print(f'  Gap:   {(kiwi_f1 - garu_f1)*100:.1f}%p')
    print(f'{"=" * 60}')


if __name__ == "__main__":
    main()
