"""Evaluate analyze_v2 (sentence-level Viterbi) on NIKL MP benchmark.

Compares v1 (current) and v2 (sentence-level) on same dataset.
"""
import json
import os
import random
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

NIKL_DIR = Path.home() / "Downloads" / "NIKL_MP(v1.1)"
ROOT = Path(__file__).parent.parent

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
    for p, d in [('V', 'VV'), ('N', 'NNG'), ('J', 'JX'), ('E', 'EF'), ('X', 'XR')]:
        if tag.startswith(p):
            return d
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


def run_analyzer(sentences, example_name):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        for text, _ in sentences:
            f.write(text + '\n')
        input_path = f.name

    result = subprocess.run(
        ["cargo", "run", "--release", "--example", example_name, "--", input_path],
        cwd=str(ROOT), capture_output=True, text=True, timeout=600,
    )
    os.unlink(input_path)

    if result.returncode != 0:
        print(f"Analyzer failed: {result.stderr[:500]}", file=sys.stderr)
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


def compute_f1(predictions, gold_sentences):
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
    for tag in all_tags[:20]:
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

    # v1 (current)
    print("\nRunning v1 (current, eojeol-level)...")
    v1_results = run_analyzer(sentences, "analyze_batch")
    if v1_results:
        P, R, F, tp, fp, fn = compute_f1(v1_results, sentences)
        print_results(f"v1 (eojeol-level) — {len(sentences)} sentences", P, R, F, tp, fp, fn)
        v1_f1 = F
    else:
        v1_f1 = 0

    # v2 (sentence-level Viterbi)
    print("\nRunning v2 (sentence-level Viterbi)...")
    v2_results = run_analyzer(sentences, "analyze_batch_v2")
    if v2_results:
        P, R, F, tp, fp, fn = compute_f1(v2_results, sentences)
        print_results(f"v2 (sentence-level) — {len(sentences)} sentences", P, R, F, tp, fp, fn)
        v2_f1 = F
    else:
        v2_f1 = 0

    # Compare
    print(f'\n{"=" * 60}')
    print(f'  COMPARISON (NIKL MP {len(sentences)} sentences)')
    print(f'  v1 (eojeol-level):    F1 = {v1_f1:.1%}')
    print(f'  v2 (sentence-level):  F1 = {v2_f1:.1%}')
    print(f'  Delta:                {(v2_f1 - v1_f1)*100:+.2f}%p')
    print(f'{"=" * 60}')

    # Detailed diff: sentences where v2 is better/worse
    if v1_results and v2_results:
        better = 0
        worse = 0
        for i in range(min(len(v1_results), len(v2_results), len(sentences))):
            v1_set = set((f, t) for f, t in v1_results[i] if f.strip())
            v2_set = set((f, t) for f, t in v2_results[i] if f.strip())
            gold_set = set((f, t) for f, t in sentences[i][1] if f.strip())
            v1_match = len(v1_set & gold_set)
            v2_match = len(v2_set & gold_set)
            if v2_match > v1_match:
                better += 1
            elif v2_match < v1_match:
                worse += 1
        print(f'\n  v2 better: {better} sentences')
        print(f'  v2 worse:  {worse} sentences')
        print(f'  Same:      {len(sentences) - better - worse} sentences')


if __name__ == "__main__":
    main()
