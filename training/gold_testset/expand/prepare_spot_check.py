"""auto 라벨 5% 샘플링 → spot-check용 batch JSON 작성."""
import argparse
import json
import random
from pathlib import Path


def sample_auto(records: list[dict], ratio: float = 0.05, seed: int = 42) -> list[dict]:
    auto = [r for r in records if r.get("confidence") == "auto"]
    rng = random.Random(seed)
    rng.shuffle(auto)
    k = max(1, int(len(auto) * ratio))
    return auto[:k]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--ratio", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=50)
    args = p.parse_args()

    records = [json.loads(l) for l in open(args.input)]
    sample = sample_auto(records, ratio=args.ratio)
    print(f"Sampled {len(sample)} auto records ({args.ratio*100:.0f}%)")

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    n_batches = (len(sample) + args.batch_size - 1) // args.batch_size
    for i in range(n_batches):
        chunk = sample[i*args.batch_size : (i+1)*args.batch_size]
        cases = [
            {"id": i*args.batch_size + j, "text": r["text"], "label_morphemes": r["morphemes"]}
            for j, r in enumerate(chunk)
        ]
        with open(out_path / f"spot_batch_{i:03d}.json", "w") as f:
            json.dump({"cases": cases}, f, ensure_ascii=False, indent=2)
    print(f"Created {n_batches} spot-check batches in {out_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main()
    else:
        recs = [{"text": f"s_{i}", "morphemes": [], "confidence": "auto"} for i in range(100)]
        recs += [{"text": f"x_{i}", "morphemes": [], "confidence": "reviewed"} for i in range(20)]
        sample = sample_auto(recs, ratio=0.05, seed=42)
        assert len(sample) == 5
        assert all(r["confidence"] == "auto" for r in sample)
        sample2 = sample_auto(recs, ratio=0.10, seed=42)
        assert len(sample2) == 10
        print("OK")
