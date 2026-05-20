"""신규 + 기존 5K 머지 + challenge 분리."""
import argparse
import json
from pathlib import Path

GOLD_DIR = Path(__file__).resolve().parent.parent
EXISTING_PATH = GOLD_DIR / "gold_testset.jsonl"
MAIN_OUT = GOLD_DIR / "gold_testset.jsonl"
CHALLENGE_OUT = GOLD_DIR / "gold_challenge.jsonl"


def merge_records(existing: list[dict], new: list[dict]) -> list[dict]:
    seen = {r["text"] for r in existing}
    out = list(existing)
    for r in new:
        if r["text"] in seen:
            continue
        out.append(r)
        seen.add(r["text"])
    return out


def split_challenge(records: list[dict]) -> tuple[list[dict], list[dict]]:
    main = [r for r in records if r.get("domain") != "엣지케이스"]
    challenge = [r for r in records if r.get("domain") == "엣지케이스"]
    return main, challenge


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--new", nargs="+", required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    existing = [json.loads(l) for l in open(EXISTING_PATH)]
    new = []
    for path in args.new:
        for line in open(path):
            r = json.loads(line)
            keep_keys = ["text", "morphemes", "domain", "confidence", "source", "reason", "edge_category"]
            new.append({k: r[k] for k in keep_keys if k in r})

    merged = merge_records(existing, new)
    main_recs, challenge_recs = split_challenge(merged)
    print(f"merged total: {len(merged)}")
    print(f"  main: {len(main_recs)}, challenge: {len(challenge_recs)}")

    if args.dry_run:
        print("(dry-run, not writing)")
        return

    with open(MAIN_OUT, "w") as f:
        for r in main_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(CHALLENGE_OUT, "w") as f:
        for r in challenge_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {MAIN_OUT} and {CHALLENGE_OUT}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main()
    else:
        existing = [{"text": "A", "morphemes": [["A","NNG"]], "domain": "뉴스", "confidence": "manual"}]
        new = [
            {"text": "A", "morphemes": [["A","NNG"]], "domain": "뉴스", "confidence": "auto"},
            {"text": "B", "morphemes": [["B","NNG"]], "domain": "뉴스", "confidence": "auto"},
        ]
        merged = merge_records(existing, new)
        assert len(merged) == 2
        assert merged[0]["text"] == "A"
        assert merged[1]["text"] == "B"

        rec = [{"text": "X", "domain": "뉴스"}, {"text": "Y", "domain": "엣지케이스"}]
        main_r, challenge_r = split_challenge(rec)
        assert [r["text"] for r in main_r] == ["X"]
        assert [r["text"] for r in challenge_r] == ["Y"]
        print("OK")
