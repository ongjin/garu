"""kowikitext.txt에서 분석용 한국어 문장 N개 sample.

필터:
- 길이 ≥ 20자 (헤더 "생애.", "어린 시절." 등 제거)
- 한글 비율 ≥ 50% (수식/리스트/메타데이터 라인 제거)
- 마침표/물음표/느낌표로 끝나는 문장
- simhash dedup (이미 풀에 들어간 crawl 문장과 중복 회피용은 sub-100K에선 부담이라 생략)

출력: JSONL ({"text": str}) — analyze_corpus.py 입력 형식과 동일.
"""
import argparse
import json
import random
import re
from pathlib import Path


HANGUL_RE = re.compile(r"[가-힣]")
END_PUNCT_RE = re.compile(r"[.!?]$")


def korean_ratio(s: str) -> float:
    if not s:
        return 0.0
    return len(HANGUL_RE.findall(s)) / len(s)


def is_valid_sentence(s: str, min_len: int, min_hangul: float) -> bool:
    s = s.strip()
    if len(s) < min_len:
        return False
    if not END_PUNCT_RE.search(s):
        return False
    if korean_ratio(s) < min_hangul:
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="training/kowikitext.txt")
    ap.add_argument("--output", required=True)
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--min-len", type=int, default=20)
    ap.add_argument("--min-hangul", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    valid_lines: list[str] = []
    total = 0
    rejected = 0
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            s = line.strip()
            if not is_valid_sentence(s, args.min_len, args.min_hangul):
                rejected += 1
                continue
            valid_lines.append(s)
            if total % 200000 == 0:
                print(f"  scanned {total}, valid {len(valid_lines)}, rejected {rejected}")

    print(f"Total scanned: {total}, valid: {len(valid_lines)}, rejected: {rejected}")

    if len(valid_lines) <= args.n:
        sample = valid_lines
    else:
        sample = random.sample(valid_lines, args.n)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for s in sample:
            f.write(json.dumps({"text": s}, ensure_ascii=False) + "\n")
    print(f"Wrote {len(sample)} sentences to {out_path}")


if __name__ == "__main__":
    main()
