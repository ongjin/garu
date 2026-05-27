"""5개 분석기 F1 비교: Garu, Kiwi, Mecab, Kkma, Komoran vs Gold

기본: ep_norm(jamo/모음조화/EP축약/태그 정규화) 적용 후 비교.
--no-norm 플래그로 끄면 raw 비교.
--analyzers 로 사용할 분석기 부분 선택 가능 (예: garu,kiwi).
"""
import argparse, json, os, subprocess, sys
from kiwipiepy import Kiwi
import mecab

BASE = os.path.dirname(__file__)
ROOT = os.path.join(BASE, "..", "..")
sys.path.insert(0, os.path.join(BASE, "expand"))
from ep_norm import normalize_ep_morphemes

# Kkma POS → 세종 매핑 (training/ensemble/pos_normalize.py와 동일)
KKMA_TO_SEJONG = {
    "OH": "SH", "OL": "SL", "ON": "SN", "NNM": "NNB",
}

def load_gold():
    records = []
    with open(os.path.join(BASE, "gold_testset.jsonl")) as f:
        for line in f:
            records.append(json.loads(line))
    return records

def run_garu(texts):
    txt_path = os.path.join(BASE, "_eval_input.txt")
    with open(txt_path, "w") as f:
        for t in texts:
            f.write(t + "\n")
    model = os.path.join(ROOT, "js/models/base.gmdl")
    binary = os.path.join(ROOT, "target/release/examples/analyze_batch")
    result = subprocess.run([binary, txt_path, "--json"],
                          capture_output=True, text=True,
                          env={**os.environ, "GARU_MODEL": model})
    os.remove(txt_path)
    return [json.loads(l) for l in result.stdout.strip().split("\n")]

def run_kiwi(texts):
    kw = Kiwi()
    results = []
    for t in texts:
        r = kw.analyze(t)
        tokens = [[m.form, m.tag.replace("-I","").replace("-R","")] for m in r[0][0]] if r else []
        results.append(tokens)
    return results

def run_mecab(texts):
    mc = mecab.MeCab()
    results = []
    for t in texts:
        try:
            raw = mc.pos(t)
        except Exception:
            results.append([])
            continue
        tokens = []
        for form, tag in raw:
            if "+" in tag:
                tokens.append([form, tag.split("+")[0]])
            else:
                tokens.append([form, tag])
        results.append(tokens)
    return results


def run_kkma(texts):
    from konlpy.tag import Kkma
    kk = Kkma()
    results = []
    for t in texts:
        try:
            raw = kk.pos(t)
        except Exception:
            results.append([])
            continue
        tokens = []
        for form, tag in raw:
            base = tag.split("+")[0] if "+" in tag else tag
            mapped = KKMA_TO_SEJONG.get(base, base)
            tokens.append([form, mapped])
        results.append(tokens)
    return results


def run_komoran(texts):
    from konlpy.tag import Komoran
    km = Komoran()
    results = []
    for t in texts:
        try:
            raw = km.pos(t)
        except Exception:
            results.append([])
            continue
        tokens = []
        for form, tag in raw:
            base = tag.split("+")[0] if "+" in tag else tag
            tokens.append([form, base])
        results.append(tokens)
    return results

def _maybe_norm(token_lists, apply_norm):
    if not apply_norm:
        return token_lists
    return [normalize_ep_morphemes(s) for s in token_lists]

def compute_f1(pred, gold):
    """Token-level F1 (form+POS pair matching). 호출자가 사전 정규화 책임."""
    tp = fp = fn = 0
    for p_tokens, g_tokens in zip(pred, gold):
        p_set = {}
        for form, pos in p_tokens:
            key = (form, pos)
            p_set[key] = p_set.get(key, 0) + 1
        g_set = {}
        for form, pos in g_tokens:
            key = (form, pos)
            g_set[key] = g_set.get(key, 0) + 1
        all_keys = set(list(p_set.keys()) + list(g_set.keys()))
        for k in all_keys:
            pc = p_set.get(k, 0)
            gc = g_set.get(k, 0)
            matched = min(pc, gc)
            tp += matched
            fp += pc - matched
            fn += gc - matched
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return prec, rec, f1

ALL_ANALYZERS = ["garu", "kiwi", "mecab", "kkma", "komoran"]
RUNNERS = {
    "garu": run_garu,
    "kiwi": run_kiwi,
    "mecab": run_mecab,
    "kkma": run_kkma,
    "komoran": run_komoran,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-norm", action="store_true", help="ep_norm 비적용 (raw 비교)")
    ap.add_argument("--analyzers", default=",".join(ALL_ANALYZERS),
                    help=f"Comma-separated subset of {','.join(ALL_ANALYZERS)}")
    args = ap.parse_args()
    apply_norm = not args.no_norm
    selected = [a.strip() for a in args.analyzers.split(",") if a.strip()]
    for a in selected:
        if a not in RUNNERS:
            sys.exit(f"Unknown analyzer: {a}")

    records = load_gold()
    texts = [r["text"] for r in records]
    gold_raw = [r["morphemes"] for r in records]

    preds_raw = {}
    for a in selected:
        print(f"Running {a}... (norm={apply_norm})", flush=True)
        preds_raw[a] = RUNNERS[a](texts)

    gold = _maybe_norm(gold_raw, apply_norm)
    preds = {a: _maybe_norm(v, apply_norm) for a, v in preds_raw.items()}

    print(f"\n=== F1 Score (vs Gold Testset, n={len(texts)}) — norm={apply_norm} ===\n")
    print(f"{'Analyzer':<10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 42)
    overall_f1 = {}
    for a in selected:
        p, r, f1 = compute_f1(preds[a], gold)
        overall_f1[a] = f1
        print(f"{a.capitalize():<10} {p:>10.4f} {r:>10.4f} {f1:>10.4f}")

    print("\n=== Domain별 F1 ===\n")
    domains = {}
    for i, rec in enumerate(records):
        d = rec["domain"]
        if d not in domains:
            domains[d] = {"preds": {a: [] for a in selected}, "gold": []}
        for a in selected:
            domains[d]["preds"][a].append(preds[a][i])
        domains[d]["gold"].append(gold[i])

    # 헤더
    header = f"  {'domain':<10}"
    for a in selected:
        header += f"  {a:>8}"
    print(header)
    for d in sorted(domains.keys()):
        row = f"  {d:<10}"
        for a in selected:
            _, _, fa = compute_f1(domains[d]["preds"][a], domains[d]["gold"])
            row += f"  {fa:>8.4f}"
        print(row)

if __name__ == "__main__":
    main()
