"""Detailed error analysis for Garu on NIKL MP.

Outputs:
1. Per-POS error breakdown (FP/FN patterns)
2. Most common mismatches (pred_form/pred_pos vs gold_form/gold_pos)
3. Segmentation vs POS confusion error classification
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


def run_rust_analyzer(sentences):
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


def analyze_errors(predictions, gold_sentences):
    # FP: in pred but not in gold
    # FN: in gold but not in pred
    fp_patterns = Counter()  # (form, pred_pos) that shouldn't be there
    fn_patterns = Counter()  # (form, gold_pos) that should be there
    confusion = Counter()    # (form, pred_pos, gold_pos) - same form, wrong POS
    segmentation_errors = []  # sentences with different morpheme count

    total_fp = 0
    total_fn = 0
    total_match = 0

    # Per-POS: what POS does it get confused with?
    pos_confusion_fp = defaultdict(Counter)  # pred_pos -> Counter of gold_pos it displaced
    pos_confusion_fn = defaultdict(Counter)  # gold_pos -> Counter of pred_pos that appeared instead

    for i in range(min(len(predictions), len(gold_sentences))):
        pred_list = predictions[i]
        gold_list = gold_sentences[i][1]
        text = gold_sentences[i][0]

        pred_set = set((f, t) for f, t in pred_list if f.strip())
        gold_set = set((f, t) for f, t in gold_list if f.strip())

        matched = pred_set & gold_set
        total_match += len(matched)

        fps = pred_set - gold_set
        fns = gold_set - pred_set
        total_fp += len(fps)
        total_fn += len(fns)

        for (form, pos) in fps:
            fp_patterns[(form, pos)] += 1
        for (form, pos) in fns:
            fn_patterns[(form, pos)] += 1

        # Detect same-form POS confusion
        fp_forms = {f: p for f, p in fps}
        fn_forms = {f: p for f, p in fns}
        for form in set(fp_forms) & set(fn_forms):
            confusion[(form, fp_forms[form], fn_forms[form])] += 1
            pos_confusion_fp[fp_forms[form]][fn_forms[form]] += 1
            pos_confusion_fn[fn_forms[form]][fp_forms[form]] += 1

    # Classify errors
    same_form_count = sum(confusion.values())
    pure_fp = total_fp - same_form_count
    pure_fn = total_fn - same_form_count

    print(f"\n{'='*70}")
    print(f"  ERROR ANALYSIS")
    print(f"  Total matches: {total_match}, FP: {total_fp}, FN: {total_fn}")
    print(f"  POS confusion (same form, wrong POS): {same_form_count}")
    print(f"  Segmentation FP (extra morphemes): {pure_fp}")
    print(f"  Segmentation FN (missing morphemes): {pure_fn}")
    print(f"{'='*70}\n")

    # Top FP patterns
    print("TOP 30 FALSE POSITIVES (predicted but wrong):")
    print(f"{'Form':<12}{'POS':<8}{'Count':>6}")
    print('-' * 30)
    for (form, pos), cnt in fp_patterns.most_common(30):
        print(f"{form:<12}{pos:<8}{cnt:>6}")

    print(f"\nTOP 30 FALSE NEGATIVES (missed by Garu):")
    print(f"{'Form':<12}{'POS':<8}{'Count':>6}")
    print('-' * 30)
    for (form, pos), cnt in fn_patterns.most_common(30):
        print(f"{form:<12}{pos:<8}{cnt:>6}")

    print(f"\nTOP 30 POS CONFUSIONS (same form, wrong POS):")
    print(f"{'Form':<12}{'Pred':<8}{'Gold':<8}{'Count':>6}")
    print('-' * 38)
    for (form, pred_pos, gold_pos), cnt in confusion.most_common(30):
        print(f"{form:<12}{pred_pos:<8}{gold_pos:<8}{cnt:>6}")

    print(f"\nPOS CONFUSION MATRIX (pred→gold, top pairs):")
    print(f"{'Pred':<8}{'Gold':<8}{'Count':>6}")
    print('-' * 24)
    pairs = Counter()
    for pred_pos, gold_counts in pos_confusion_fp.items():
        for gold_pos, cnt in gold_counts.items():
            pairs[(pred_pos, gold_pos)] += cnt
    for (pred, gold), cnt in pairs.most_common(25):
        print(f"{pred:<8}{gold:<8}{cnt:>6}")

    # Per-POS error summary
    print(f"\nPER-POS ERROR SUMMARY:")
    print(f"{'POS':<8}{'FP':>6}{'FN':>6}{'Confusion':>10}{'Net':>6}")
    print('-' * 40)
    all_pos = sorted(set([p for _, p in fp_patterns] + [p for _, p in fn_patterns]),
                     key=lambda p: -(sum(c for (_, pp), c in fp_patterns.items() if pp == p) +
                                     sum(c for (_, pp), c in fn_patterns.items() if pp == p)))
    for pos in all_pos[:20]:
        fp = sum(c for (_, p), c in fp_patterns.items() if p == pos)
        fn_c = sum(c for (_, p), c in fn_patterns.items() if p == pos)
        conf = sum(pos_confusion_fp.get(pos, {}).values())
        print(f"{pos:<8}{fp:>6}{fn_c:>6}{conf:>10}{fp-fn_c:>6}")


def main():
    n = 2000
    if len(sys.argv) > 2 and sys.argv[1] == '--n':
        n = int(sys.argv[2])

    print(f"Loading NIKL MP sentences (max {n})...")
    sentences = load_nikl_sentences(n)
    print(f"  Loaded {len(sentences)} sentences")

    print("\nRunning Garu (Rust)...")
    results = run_rust_analyzer(sentences)
    if results:
        analyze_errors(results, sentences)


if __name__ == "__main__":
    main()
