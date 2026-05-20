"""Per-domain F1 breakdown for 3 variants on gold testset."""
import json, os, subprocess, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "training" / "gold_testset" / "gold_testset.jsonl"
MODEL = ROOT / "js" / "models" / "base.gmdl"
CNN = ROOT / "js" / "models" / "cnn2.bin"


def run(binary_name, texts):
    binary = ROOT / "target" / "release" / "examples" / binary_name
    inp = ROOT / "training" / f"_eval_input_{os.getpid()}_{binary_name}.txt"
    inp.write_text("\n".join(texts))
    env = {**os.environ, "GARU_MODEL": str(MODEL), "GARU_CNN": str(CNN)}
    r = subprocess.run([str(binary), str(inp), "--json"], capture_output=True, text=True, env=env)
    inp.unlink()
    if r.returncode != 0:
        sys.exit(f"binary failed: {r.stderr[:300]}")
    return [json.loads(l) for l in r.stdout.strip().split("\n")]


def f1(pred, gold):
    tp = fp = fn = 0
    for p_tokens, g_tokens in zip(pred, gold):
        ps, gs = Counter(), Counter()
        for f, p in p_tokens: ps[(f, p)] += 1
        for f, p in g_tokens: gs[(f, p)] += 1
        for k in set(list(ps) + list(gs)):
            pc, gc = ps[k], gs[k]
            m = min(pc, gc)
            tp += m; fp += pc - m; fn += gc - m
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    return prec, rec, (2 * prec * rec / (prec + rec) if (prec + rec) else 0)


def main():
    records = [json.loads(l) for l in GOLD.open()]
    texts = [r["text"] for r in records]
    gold = [r["morphemes"] for r in records]
    domains = [r["domain"] for r in records]

    by_domain = defaultdict(list)
    for i, d in enumerate(domains):
        by_domain[d].append(i)

    print("Running 3 variants on 5000 gold sentences...", file=sys.stderr)
    preds = {
        "force_cnn": run("analyze_batch_force_cnn", texts),
        "gated":     run("analyze_batch", texts),
        "skip":      run("analyze_batch_skip", texts),
    }

    print(f"\n{'Domain':<10} {'Count':>6}  {'force_cnn':>10}  {'gated':>10}  {'skip':>10}  {'Δgate':>8}  {'Δskip':>8}")
    print("-" * 80)
    for d in ["뉴스", "일상", "SNS", "기술", "문학", "엣지케이스"]:
        idx = by_domain.get(d, [])
        if not idx: continue
        g_subset = [gold[i] for i in idx]
        results = {}
        for name, pred in preds.items():
            p_subset = [pred[i] for i in idx]
            _, _, fv = f1(p_subset, g_subset)
            results[name] = fv
        d_gate = results["gated"] - results["force_cnn"]
        d_skip = results["skip"] - results["force_cnn"]
        print(f"{d:<10} {len(idx):>6}  {results['force_cnn']:>10.4f}  {results['gated']:>10.4f}  {results['skip']:>10.4f}  {d_gate:>+8.4f}  {d_skip:>+8.4f}")

    # Overall
    print("-" * 80)
    overall = {}
    for name, pred in preds.items():
        _, _, fv = f1(pred, gold)
        overall[name] = fv
    print(f"{'OVERALL':<10} {len(records):>6}  {overall['force_cnn']:>10.4f}  {overall['gated']:>10.4f}  {overall['skip']:>10.4f}  "
          f"{overall['gated']-overall['force_cnn']:>+8.4f}  {overall['skip']-overall['force_cnn']:>+8.4f}")


if __name__ == "__main__":
    main()
