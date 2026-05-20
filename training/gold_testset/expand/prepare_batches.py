"""pairs.jsonl의 disagreement만 골라서 batch JSON 파일로 split.

각 batch 파일은 Claude 서브에이전트가 한 번에 처리할 단위.
"""
import argparse
import json
import os
from pathlib import Path


def split_disagreements_to_batches(pairs: list[dict], out_dir: str,
                                    batch_size: int = 50) -> int:
    disagreements = [p for p in pairs if not p["agree"]]
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    n_batches = (len(disagreements) + batch_size - 1) // batch_size
    for i in range(n_batches):
        chunk = disagreements[i*batch_size : (i+1)*batch_size]
        cases = [
            {"id": i*batch_size + j, "text": p["text"], "garu": p["garu"], "kiwi": p["kiwi"]}
            for j, p in enumerate(chunk)
        ]
        with open(out_path / f"batch_{i:03d}.json", "w") as f:
            json.dump({"cases": cases}, f, ensure_ascii=False, indent=2)
    return n_batches


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="pairs.jsonl")
    p.add_argument("--out-dir", required=True, help="batch JSON 출력 디렉토리")
    p.add_argument("--batch-size", type=int, default=50)
    args = p.parse_args()

    pairs = [json.loads(l) for l in open(args.input)]
    n = split_disagreements_to_batches(pairs, args.out_dir, args.batch_size)
    n_disagree = sum(1 for p in pairs if not p["agree"])
    n_agree = sum(1 for p in pairs if p["agree"])
    print(f"Input pairs: {len(pairs)} (agree {n_agree}, disagree {n_disagree})")
    print(f"Created {n} batches of size ≤{args.batch_size} in {args.out_dir}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main()
    else:
        pairs = [
            {"text": f"sentence_{i}", "garu": [], "kiwi": [], "agree": (i % 3 == 0)}
            for i in range(20)
        ]
        n_batches = split_disagreements_to_batches(pairs, "/tmp/_test_batches", batch_size=5)
        assert n_batches == 3
        for i in range(n_batches):
            path = f"/tmp/_test_batches/batch_{i:03d}.json"
            assert os.path.exists(path)
            data = json.load(open(path))
            assert "cases" in data
            assert len(data["cases"]) <= 5
            os.unlink(path)
        os.rmdir("/tmp/_test_batches")
        print("OK")
