"""search_train gold 출력 중 정답과 같은(ep_norm) 문장 번호를 고른다.

    python3 training/search/verify_gold.py ROWS.jsonl GOLD.jsonl > VERIFIED.txt
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gold_testset" / "expand"))
from ep_norm import normalize_ep_morphemes  # noqa: E402


def main(rows_path, gold_path):
    found = verified = total = 0
    with open(rows_path, encoding="utf-8") as rows, open(gold_path, encoding="utf-8") as golds:
        for row_line, gold_line in zip(rows, golds):
            row, path = json.loads(row_line), json.loads(gold_line)
            total += 1
            if path["tokens"] is None:
                continue
            found += 1
            got = normalize_ep_morphemes([list(t) for t in path["tokens"]])
            if got == normalize_ep_morphemes([list(m) for m in row["morphemes"]]):
                verified += 1
                print(path["i"])
    print(f"rows {total} path {found} verified {verified}", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
