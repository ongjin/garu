"""구어 gold에서 `~/SO`가 등장하는 sentence 추출 후 Garu 분석 결과 비교.

목적: 현재 Garu가 `~/SO`를 잘 잡는지, 못 잡는지 판단.
"""
import json, os, sys, subprocess, tempfile

BASE = os.path.dirname(os.path.abspath(__file__))
GOLD = os.path.join(BASE, "gold_testset/gold_testset.jsonl")

def load_gold():
    out = []
    with open(GOLD) as f:
        for line in f:
            out.append(json.loads(line))
    return out

def run_garu(texts):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for t in texts:
            f.write(t + "\n")
        p = f.name
    bin_path = os.path.join(BASE, "..", "target/release/examples/analyze_batch")
    model = os.path.join(BASE, "..", "js/models/base.gmdl")
    r = subprocess.run([bin_path, p, "--json"], capture_output=True, text=True,
                       env={**os.environ, "GARU_MODEL": model})
    os.unlink(p)
    return [json.loads(l) for l in r.stdout.strip().split("\n")]

def main():
    gold = load_gold()
    so_records = [(i, r) for i, r in enumerate(gold)
                  if any(m[1] == "SO" and "~" in m[0] for m in r["morphemes"])]
    print(f"Total gold sentences: {len(gold)}")
    print(f"Sentences with ~/SO in gold: {len(so_records)}")
    if not so_records:
        print("No ~/SO in gold — Track C는 의미 없음, skip.")
        return

    texts = [r["text"] for _, r in so_records]
    pred = run_garu(texts)

    hit = miss = 0
    miss_samples = []
    for (idx, gold_r), pred_tokens in zip(so_records, pred):
        gold_so = sum(1 for m in gold_r["morphemes"] if m[1] == "SO" and "~" in m[0])
        pred_so = sum(1 for f, p in pred_tokens if p == "SO" and "~" in f)
        if pred_so >= gold_so:
            hit += 1
        else:
            miss += 1
            if len(miss_samples) < 10:
                miss_samples.append({
                    "text": gold_r["text"],
                    "gold_so": [m for m in gold_r["morphemes"] if m[1] == "SO"],
                    "pred_relevant": [(f, p) for f, p in pred_tokens if p == "SO" or "~" in f],
                })

    print(f"\nResult: hit={hit}/{len(so_records)} ({100*hit/len(so_records):.1f}%), miss={miss}")
    print(f"\nDecision threshold: hit rate ≥ 90% → skip Task 12. Current: {100*hit/len(so_records):.1f}%")
    if miss_samples:
        print(f"\nFirst {len(miss_samples)} miss samples:")
        for s in miss_samples:
            print(f"\n  TEXT: {s['text']}")
            print(f"  GOLD SO: {s['gold_so']}")
            print(f"  PRED:    {s['pred_relevant']}")

if __name__ == "__main__":
    main()
