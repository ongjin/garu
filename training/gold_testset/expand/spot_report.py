"""spot check batch 결과 → 리포트."""
import argparse
import json
from pathlib import Path


def build_report(batches_dir: str) -> dict:
    bdir = Path(batches_dir)
    id_to_input = {}
    for inp_file in sorted(bdir.glob("spot_batch_*.json")):
        if "_out" in inp_file.name:
            continue
        with open(inp_file) as f:
            data = json.load(f)
        for c in data["cases"]:
            id_to_input[c["id"]] = c

    disagrees = []
    n_total = 0
    for out_file in sorted(bdir.glob("spot_batch_*_out.json")):
        with open(out_file) as f:
            results = json.load(f)
        for r in results:
            n_total += 1
            if r["verdict"] != "OK":
                inp = id_to_input.get(r["id"], {})
                disagrees.append({
                    "text": inp.get("text", "?"),
                    "label_morphemes": inp.get("label_morphemes"),
                    "correct_morphemes": r.get("correct_morphemes"),
                    "reason": r.get("reason", ""),
                })
    return {
        "n_sampled": n_total,
        "n_disagree": len(disagrees),
        "disagree_rate": len(disagrees) / n_total if n_total else 0,
        "cases": disagrees,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batches-dir", required=True)
    p.add_argument("--output", required=True, help="report .md")
    args = p.parse_args()

    rep = build_report(args.batches_dir)
    gate = "PASS" if rep["disagree_rate"] <= 0.02 else "FAIL (>2%)"
    md = f"""# Spot Check Report

- 샘플 수: {rep['n_sampled']}
- 불일치 수: {rep['n_disagree']}
- 불일치율: {rep['disagree_rate']*100:.2f}%
- 게이트: {gate}

## 불일치 케이스
"""
    for c in rep["cases"]:
        md += f"\n### {c['text']}\n"
        md += f"- 라벨: `{c['label_morphemes']}`\n"
        md += f"- 제안: `{c['correct_morphemes']}`\n"
        md += f"- 근거: {c['reason']}\n"
    Path(args.output).write_text(md)
    print(f"Wrote {args.output} (disagree rate: {rep['disagree_rate']*100:.2f}%)")


if __name__ == "__main__":
    import sys, tempfile, os, json
    if len(sys.argv) > 1:
        main()
    else:
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "spot_batch_000_out.json"), "w") as f:
                json.dump([
                    {"id": 0, "verdict": "OK"},
                    {"id": 1, "verdict": "wrong", "correct_morphemes": [["X","NNG"]], "reason": "오답"},
                    {"id": 2, "verdict": "OK"},
                ], f, ensure_ascii=False)
            with open(os.path.join(td, "spot_batch_000.json"), "w") as f:
                json.dump({"cases": [
                    {"id": 0, "text": "a", "label_morphemes": []},
                    {"id": 1, "text": "b", "label_morphemes": [["B","VV"]]},
                    {"id": 2, "text": "c", "label_morphemes": []},
                ]}, f, ensure_ascii=False)
            rep = build_report(td)
            assert rep["n_sampled"] == 3
            assert rep["n_disagree"] == 1
            assert abs(rep["disagree_rate"] - 1/3) < 0.01
        print("OK")
