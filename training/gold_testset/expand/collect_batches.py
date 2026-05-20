"""pairs + 서브에이전트 batch 결과 → 통합 judged.jsonl."""
import argparse
import json
from pathlib import Path


def collect(pairs_path: str, batches_dir: str) -> list[dict]:
    """pairs.jsonl 읽고, agree는 auto, disagree는 batch_NNN_out.json에서 매칭."""
    pairs = [json.loads(l) for l in open(pairs_path)]
    disagreements = [(i, p) for i, p in enumerate(pairs) if not p["agree"]]

    bdir = Path(batches_dir)
    id_to_result = {}
    for out_file in sorted(bdir.glob("batch_*_out.json")):
        with open(out_file) as f:
            results = json.load(f)
        for r in results:
            id_to_result[r["id"]] = r

    out = []
    for orig_idx, pair in enumerate(pairs):
        if pair["agree"]:
            out.append({
                **pair,
                "morphemes": pair["garu"],
                "confidence": "auto",
                "source": "garu+kiwi(ep_norm)",
            })
        else:
            di_index = next(di for di, (idx, _) in enumerate(disagreements) if idx == orig_idx)
            if di_index not in id_to_result:
                print(f"[missing] disagreement id={di_index} text={pair['text'][:30]}")
                out.append({**pair, "morphemes": pair["garu"], "confidence": "reviewed",
                           "source": "missing-batch-result"})
                continue
            r = id_to_result[di_index]
            morphemes = r.get("morphemes")
            if morphemes is None:
                if r.get("choice") == "garu":
                    morphemes = pair["garu"]
                elif r.get("choice") == "kiwi":
                    morphemes = pair["kiwi"]
                else:
                    morphemes = pair["garu"]
            out.append({
                **pair,
                "morphemes": morphemes,
                "confidence": "reviewed",
                "source": f"garu+kiwi+claude(선택:{r['choice']})",
                "reason": r.get("reason", ""),
            })
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", required=True)
    p.add_argument("--batches-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--domain", help="domain 필드 강제 부여")
    args = p.parse_args()

    judged = collect(args.pairs, args.batches_dir)
    if args.domain:
        for r in judged:
            r["domain"] = args.domain

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for r in judged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_auto = sum(1 for r in judged if r["confidence"] == "auto")
    n_reviewed = sum(1 for r in judged if r["confidence"] == "reviewed")
    print(f"Wrote {len(judged)} to {args.output}")
    print(f"  auto: {n_auto}, reviewed: {n_reviewed}")


if __name__ == "__main__":
    import sys, tempfile, os, json
    if len(sys.argv) > 1:
        main()
    else:
        pairs = [
            {"text": "A", "garu": [["A","NNG"]], "kiwi": [["A","NNG"]], "agree": True},
            {"text": "B", "garu": [["B","NNG"]], "kiwi": [["B","VV"]], "agree": False},
        ]
        batch_out = [
            {"id": 0, "choice": "kiwi", "morphemes": [["B","VV"]], "reason": "VV 옳음"}
        ]
        with tempfile.TemporaryDirectory() as td:
            pairs_path = os.path.join(td, "pairs.jsonl")
            with open(pairs_path, "w") as f:
                for p in pairs:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")
            batches_dir = os.path.join(td, "batches")
            os.makedirs(batches_dir)
            with open(os.path.join(batches_dir, "batch_000.json"), "w") as f:
                json.dump({"cases": [{"id": 0, "text": "B", "garu": pairs[1]["garu"], "kiwi": pairs[1]["kiwi"]}]}, f, ensure_ascii=False)
            with open(os.path.join(batches_dir, "batch_000_out.json"), "w") as f:
                json.dump(batch_out, f, ensure_ascii=False)
            judged = collect(pairs_path, batches_dir)
            assert len(judged) == 2
            assert judged[0]["confidence"] == "auto"
            assert judged[1]["confidence"] == "reviewed"
            assert judged[1]["morphemes"] == [["B","VV"]]
            assert "선택:kiwi" in judged[1]["source"]
        print("OK")
