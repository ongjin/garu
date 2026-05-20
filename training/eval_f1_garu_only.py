"""Garu-only F1 eval — fast, no Kiwi/Mecab subprocess.
Usage: python3 training/eval_f1_garu_only.py [binary_name]
       binary_name defaults to analyze_batch
"""
import json, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "training" / "gold_testset" / "gold_testset.jsonl"
MODEL = ROOT / "js" / "models" / "base.gmdl"
CNN = ROOT / "js" / "models" / "cnn2.bin"


def compute_f1(pred, gold):
    tp = fp = fn = 0
    for p_tokens, g_tokens in zip(pred, gold):
        p_set, g_set = {}, {}
        for form, pos in p_tokens:
            p_set[(form, pos)] = p_set.get((form, pos), 0) + 1
        for form, pos in g_tokens:
            g_set[(form, pos)] = g_set.get((form, pos), 0) + 1
        for k in set(list(p_set.keys()) + list(g_set.keys())):
            pc = p_set.get(k, 0); gc = g_set.get(k, 0)
            m = min(pc, gc)
            tp += m; fp += pc - m; fn += gc - m
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    return prec, rec, f1


def main():
    binary_name = sys.argv[1] if len(sys.argv) > 1 else "analyze_batch"
    binary = ROOT / "target" / "release" / "examples" / binary_name
    if not binary.exists():
        sys.exit(f"binary not found: {binary} — build with cargo build --release --example {binary_name}")

    records = [json.loads(l) for l in GOLD.open()]
    texts = [r["text"] for r in records]
    gold = [r["morphemes"] for r in records]

    inp = ROOT / "training" / "_eval_input.txt"
    inp.write_text("\n".join(texts))

    env = {**os.environ, "GARU_MODEL": str(MODEL), "GARU_CNN": str(CNN)}
    r = subprocess.run([str(binary), str(inp), "--json"], capture_output=True, text=True, env=env)
    inp.unlink()
    if r.returncode != 0:
        sys.exit(f"binary failed: {r.stderr}")

    pred = [json.loads(l) for l in r.stdout.strip().split("\n")]
    p, rec, f1 = compute_f1(pred, gold)
    print(f"{binary_name:30} P={p:.4f} R={rec:.4f} F1={f1:.4f}  ({len(pred)} sentences)")

if __name__ == "__main__":
    main()
