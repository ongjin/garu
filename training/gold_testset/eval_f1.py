"""4개 분석기 F1 비교: Garu, Kiwi, Claude, Mecab vs Gold

기본: ep_norm(jamo/모음조화/EP축약/태그 정규화) 적용 후 비교.
--no-norm 플래그로 끄면 raw 비교.
"""
import argparse, json, os, subprocess, sys
from kiwipiepy import Kiwi
import mecab

BASE = os.path.dirname(__file__)
ROOT = os.path.join(BASE, "..", "..")
sys.path.insert(0, os.path.join(BASE, "expand"))
from ep_norm import normalize_ep_morphemes

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
        tokens = []
        for form, tag in mc.pos(t):
            if "+" in tag:
                tokens.append([form, tag.split("+")[0]])
            else:
                tokens.append([form, tag])
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-norm", action="store_true", help="ep_norm 비적용 (raw 비교)")
    args = ap.parse_args()
    apply_norm = not args.no_norm

    records = load_gold()
    texts = [r["text"] for r in records]
    gold_raw = [r["morphemes"] for r in records]

    print(f"Running Garu... (norm={apply_norm})", flush=True)
    garu_raw = run_garu(texts)
    print("Running Kiwi...", flush=True)
    kiwi_raw = run_kiwi(texts)
    print("Running Mecab...", flush=True)
    mec_raw = run_mecab(texts)

    gold = _maybe_norm(gold_raw, apply_norm)
    garu = _maybe_norm(garu_raw, apply_norm)
    kiwi = _maybe_norm(kiwi_raw, apply_norm)
    mec = _maybe_norm(mec_raw, apply_norm)

    print(f"\n=== F1 Score (vs Gold Testset) — norm={apply_norm} ===\n")
    print(f"{'Analyzer':<10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 42)
    for name, pred in [("Garu", garu), ("Kiwi", kiwi), ("Mecab", mec)]:
        p, r, f1 = compute_f1(pred, gold)
        print(f"{name:<10} {p:>10.4f} {r:>10.4f} {f1:>10.4f}")

    print("\n=== Domain별 F1 ===\n")
    domains = {}
    for i, rec in enumerate(records):
        d = rec["domain"]
        if d not in domains:
            domains[d] = {"pred_garu": [], "pred_kiwi": [], "gold": []}
        domains[d]["pred_garu"].append(garu[i])
        domains[d]["pred_kiwi"].append(kiwi[i])
        domains[d]["gold"].append(gold[i])
    for d in sorted(domains.keys()):
        pg, _, fg = compute_f1(domains[d]["pred_garu"], domains[d]["gold"])
        pk, _, fk = compute_f1(domains[d]["pred_kiwi"], domains[d]["gold"])
        delta = fg - fk
        print(f"  {d:<10} Garu={fg:.4f}  Kiwi={fk:.4f}  Δ={delta:+.4f}")

if __name__ == "__main__":
    main()
