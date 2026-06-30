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

NIKL_DIR = Path(os.environ.get("NIKL_MP_DIR", str(Path.home() / "workspace" / "data" / "nikl_mp_2021")))

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
    for path in sorted(NIKL_DIR.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        for doc in data["document"]:
            for sent in doc["sentence"]:
                # 2025 marks intra-token spaces of joined compounds with '_'
                # (사업_분야); feed natural text so analyzers don't emit '_/SW'.
                text = sent["form"].replace("_", " ")
                if not text or len(text) < 5 or len(text) > 200:
                    continue
                morphemes = []
                for m in (sent.get("MP") or sent.get("morpheme") or []):
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


def run_mecab_analyzer(sentences):
    """Run Mecab on sentences."""
    import mecab
    mc = mecab.MeCab()
    results = []
    for text, _ in sentences:
        try:
            raw = mc.pos(text)
        except Exception:
            results.append([])
            continue
        tokens = []
        for form, tag in raw:
            base = tag.split("+")[0] if "+" in tag else tag
            if form.strip():
                tokens.append((form, normalize_pos(base)))
        results.append(tokens)
    return results


KKMA_TO_SEJONG = {
    "OH": "SH", "OL": "SL", "ON": "SN", "NNM": "NNB",
}


def run_kkma_analyzer(sentences):
    """Run Kkma on sentences."""
    from konlpy.tag import Kkma
    kk = Kkma()
    results = []
    for text, _ in sentences:
        try:
            raw = kk.pos(text)
        except Exception:
            results.append([])
            continue
        tokens = []
        for form, tag in raw:
            base = tag.split("+")[0] if "+" in tag else tag
            mapped = KKMA_TO_SEJONG.get(base, base)
            if form.strip():
                tokens.append((form, normalize_pos(mapped)))
        results.append(tokens)
    return results


def run_komoran_analyzer(sentences):
    """Run Komoran on sentences."""
    from konlpy.tag import Komoran
    km = Komoran()
    results = []
    for text, _ in sentences:
        try:
            raw = km.pos(text)
        except Exception:
            results.append([])
            continue
        tokens = []
        for form, tag in raw:
            base = tag.split("+")[0] if "+" in tag else tag
            if form.strip():
                tokens.append((form, normalize_pos(base)))
        results.append(tokens)
    return results


def normalize_for_2025(morphs):
    """Reconcile NIKL 형태분석 2025's coarser segmentation convention.

    2025 merges productive 명사/어근 + 파생접미사(XSV/XSA) into a single 용언
    (방문/NNG + 하/XSV → 방문하/VV), whereas 2021/Sejong (and Garu/Kiwi) split them.
    2025 also joins spaced compound nouns into one token via '_' (사업_분야/NNG);
    we un-join those so granularity matches the split analyzers.
    Applied symmetrically to both gold and prediction so the comparison is
    convention-neutral: on 2021 both sides split→both merge identically (수치 불변),
    on 2025 it lifts Garu's split output to match the merged gold.
    """
    # 0) un-join 2025's '_'-joined compounds (사업_분야/NNG → 사업/NNG + 분야/NNG).
    #    Garu is fed '_'-free text so never produces these → no-op on predictions.
    expanded = []
    for f, p in morphs:
        if "_" in f:
            expanded.extend((part, p) for part in f.split("_") if part)
        else:
            expanded.append((f, p))
    morphs = expanded

    out = []
    i = 0
    n = len(morphs)
    while i < n:
        if i + 1 < n:
            f0, p0 = morphs[i]
            f1, p1 = morphs[i + 1]
            if p0 in ("NNG", "NNP", "XR") and p1 in ("XSV", "XSA"):
                out.append((f0 + f1, "VV" if p1 == "XSV" else "VA"))
                i += 2
                continue
        out.append(morphs[i])
        i += 1
    return out


def compute_f1(predictions, gold_sentences, normalize=None):
    """Compute morpheme-level F1."""
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    total_match = 0
    total_pred = 0
    total_gold = 0

    for i in range(min(len(predictions), len(gold_sentences))):
        pred_list = predictions[i]
        gold_list = gold_sentences[i][1]
        if normalize is not None:
            pred_list = normalize(pred_list)
            gold_list = normalize(gold_list)
        pred_set = set((f, t) for f, t in pred_list if f.strip())
        gold_set = set((f, t) for f, t in gold_list if f.strip())

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


RUNNERS = {
    "garu":    run_rust_analyzer,
    "kiwi":    run_kiwi_analyzer,
    "mecab":   run_mecab_analyzer,
    "kkma":    run_kkma_analyzer,
    "komoran": run_komoran_analyzer,
}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--analyzers", default=",".join(RUNNERS.keys()))
    ap.add_argument("--verbose-pos", action="store_true",
                    help="POS별 breakdown 출력")
    ap.add_argument("--norm-2025", action="store_true",
                    help="2025 분절 컨벤션 정규화(명사+XSV/XSA 병합)를 채점에 적용")
    args = ap.parse_args()
    normalize = normalize_for_2025 if args.norm_2025 else None

    selected = [a.strip() for a in args.analyzers.split(",") if a.strip()]
    for a in selected:
        if a not in RUNNERS:
            sys.exit(f"Unknown analyzer: {a}")

    print(f"Loading NIKL MP sentences (max {args.n})...")
    sentences = load_nikl_sentences(args.n)
    print(f"  Loaded {len(sentences)} sentences")

    f1_results = {}
    for a in selected:
        print(f"\nRunning {a}...", flush=True)
        pred = RUNNERS[a](sentences)
        if pred is None:
            print(f"  {a}: failed")
            f1_results[a] = (0, 0, 0, {}, {}, {})
            continue
        P, R, F, tp, fp, fn = compute_f1(pred, sentences, normalize=normalize)
        f1_results[a] = (P, R, F, tp, fp, fn)
        if args.verbose_pos:
            print_results(f"{a.capitalize()} vs NIKL MP ({len(sentences)} sentences)",
                          P, R, F, tp, fp, fn)

    print(f'\n{"=" * 60}')
    print(f'  SUMMARY (NIKL MP {len(sentences)} sentences{", norm-2025" if normalize else ""})')
    print(f'  {"analyzer":<10} {"Prec":>8} {"Rec":>8} {"F1":>8}')
    print(f'  {"-"*36}')
    for a in selected:
        P, R, F, *_ = f1_results[a]
        print(f'  {a:<10} {P:>8.4f} {R:>8.4f} {F:>8.4f}')
    print(f'{"=" * 60}')


if __name__ == "__main__":
    main()
