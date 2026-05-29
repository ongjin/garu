"""still_suspicious의 미해소 어절을 Claude 검토용 청크 JSONL로 분할.

각 줄 = 검토 단위 1건:
  {"uid": "<문장idx>_<어절idx>", "text": <문장>, "surface": <어절>,
   "candidates": [{"analyzers": [...], "morphemes": [[s,p],...]},...]}

청크 파일은 chunk_dir/chunk_000.jsonl ... 으로 분할 (기본 200건/청크).

사용:
    $GARU_ENSEMBLE_PYTHON training/cleansing/claude_review_prep.py \\
        --still training/gold_testset/gold_still_suspicious.jsonl \\
        --chunk-dir training/gold_testset/claude_chunks \\
        --chunk-size 200
"""
import argparse
import json
from pathlib import Path


def build_items(still_path: Path):
    items = []
    for sent_idx, line in enumerate(open(still_path)):
        d = json.loads(line)
        for se in d["suspicious_eojeols"]:
            if se.get("resolved") is not None:
                continue
            items.append({
                "uid": f"{sent_idx}_{se['index']}",
                "text": d["text"],
                "surface": se["surface"],
                "candidates": se["candidates"],
            })
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--still", required=True, type=Path)
    ap.add_argument("--chunk-dir", required=True, type=Path)
    ap.add_argument("--chunk-size", type=int, default=200)
    args = ap.parse_args()

    items = build_items(args.still)
    args.chunk_dir.mkdir(parents=True, exist_ok=True)
    n_chunks = 0
    for i in range(0, len(items), args.chunk_size):
        chunk = items[i:i + args.chunk_size]
        p = args.chunk_dir / f"chunk_{n_chunks:03d}.jsonl"
        with open(p, "w") as f:
            for it in chunk:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        n_chunks += 1
    print(f"{len(items)} items → {n_chunks} chunks in {args.chunk_dir}")


if __name__ == "__main__":
    main()
