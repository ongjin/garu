"""Run Garu on problem cases and compute pass/fail + token F1.

Usage: python3 training/eval_problem_cases.py <testset.jsonl>
"""
import json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def run_garu(texts):
    txt = "/tmp/_garu_eval.txt"
    with open(txt, "w") as f:
        for t in texts:
            f.write(t + "\n")
    binary = os.path.join(ROOT, "target/release/examples/analyze_batch")
    model = os.path.join(ROOT, "js/models/base.gmdl")
    cnn = os.path.join(ROOT, "models/cnn2.bin")
    result = subprocess.run(
        [binary, txt, "--json"],
        capture_output=True, text=True,
        env={**os.environ, "GARU_MODEL": model, "GARU_CNN": cnn}
    )
    return [json.loads(l) for l in result.stdout.strip().split("\n")]

def f1(pred, gold):
    tp = fp = fn = 0
    for p, g in zip(pred, gold):
        pm, gm = {}, {}
        for form, pos in p:
            pm[(form, pos)] = pm.get((form, pos), 0) + 1
        for form, pos in g:
            gm[(form, pos)] = gm.get((form, pos), 0) + 1
        for k in set(list(pm) + list(gm)):
            matched = min(pm.get(k, 0), gm.get(k, 0))
            tp += matched
            fp += pm.get(k, 0) - matched
            fn += gm.get(k, 0) - matched
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    return prec, rec, 2*prec*rec/(prec+rec) if prec+rec else 0

def main():
    path = sys.argv[1]
    records = [json.loads(l) for l in open(path)]
    texts = [r["text"] for r in records]
    gold = [r["morphemes"] for r in records]
    pred = run_garu(texts)
    p, r, f = f1(pred, gold)
    exact = sum(1 for pp, gg in zip(pred, gold) if pp == gg)
    print(f"File: {path}")
    print(f"N={len(records)}  Exact={exact}/{len(records)}  P={p:.4f} R={r:.4f} F1={f:.4f}")
    print("\n--- Failures ---")
    for t, pp, gg in zip(texts, pred, gold):
        if pp != gg:
            print(f"  {t}")
            print(f"    gold: {gg}")
            print(f"    pred: {pp}")

if __name__ == "__main__":
    main()
