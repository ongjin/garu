#!/usr/bin/env python3
"""
표준국어대사전 등재 여부 자동 검증 스크립트.
입력:  training/temp_compound_candidates.jsonl
출력:  training/temp_compound_verified.jsonl  (stdict 필드 추가)
"""

import json
import re
import sys
import time
import urllib.request
import urllib.parse
from pathlib import Path

INPUT  = Path(__file__).parent / "temp_compound_candidates.jsonl"
OUTPUT = Path(__file__).parent / "temp_compound_verified.jsonl"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "ko-KR,ko;q=0.9",
}

ZERO_RESULT_PATTERN = re.compile(r"찾기 결과 \(총\s+0 개\)")
SLEEP_SEC = 1.2   # 요청 간격 (초)
TIMEOUT   = 10    # HTTP 타임아웃 (초)
FLUSH_EVERY = 50  # N개마다 flush


def fetch_html(word: str):
    """stdict 검색 결과 HTML 반환. 실패 시 None."""
    encoded = urllib.parse.quote(word)
    url = (
        f"https://stdict.korean.go.kr/search/searchResult.do"
        f"?pageSize=10&searchKeyword={encoded}"
    )
    req = urllib.request.Request(url, headers=HEADERS)
    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception as e:
            if attempt == 0:
                print(f"  [재시도] {word}: {e}", file=sys.stderr)
                time.sleep(2)
            else:
                print(f"  [실패] {word}: {e}", file=sys.stderr)
    return None


def judge(html: str) -> str:
    """'listed' | 'unlisted' 판정."""
    if ZERO_RESULT_PATTERN.search(html):
        return "unlisted"
    return "listed"


def check_word(word: str) -> str:
    """'listed' | 'unlisted' | 'unknown'."""
    html = fetch_html(word)
    if html is None:
        return "unknown"
    return judge(html)


# ── sanity test ──────────────────────────────────────────────────────────────

SANITY = [
    ("주유비",     "unlisted"),
    ("교통비",     "listed"),
    ("식비",       "listed"),
    # 프레임워크는 외래어라 미리 예측 불가. 결과만 출력.
]

print("=== Sanity check ===", file=sys.stderr)
sanity_ok = True
for word, expected in SANITY:
    result = check_word(word)
    mark = "✓" if result == expected else "✗ FAIL"
    print(f"  {word}: {result} (expected {expected}) {mark}", file=sys.stderr)
    if result != expected:
        sanity_ok = False
    time.sleep(SLEEP_SEC)

# 프레임워크는 정답 불명 → 결과만 출력
fw_result = check_word("프레임워크")
print(f"  프레임워크: {fw_result} (no expectation)", file=sys.stderr)
time.sleep(SLEEP_SEC)

if not sanity_ok:
    print("\n[STOP] Sanity check 실패 — 검색 로직 확인 필요.", file=sys.stderr)
    sys.exit(1)

print("Sanity check 통과. 본 작업 시작.\n", file=sys.stderr)

# ── main loop ────────────────────────────────────────────────────────────────

candidates = [json.loads(line) for line in INPUT.read_text().splitlines() if line.strip()]
total = len(candidates)
results = []

out_f = OUTPUT.open("w", encoding="utf-8")

for i, item in enumerate(candidates, 1):
    surface = item["surface"]
    status = check_word(surface)
    item["stdict"] = status
    results.append(item)

    line = json.dumps(item, ensure_ascii=False)
    out_f.write(line + "\n")

    if i % FLUSH_EVERY == 0:
        out_f.flush()
        listed_so_far   = sum(1 for r in results if r["stdict"] == "listed")
        unlisted_so_far = sum(1 for r in results if r["stdict"] == "unlisted")
        unknown_so_far  = sum(1 for r in results if r["stdict"] == "unknown")
        print(
            f"[{i}/{total}] listed={listed_so_far} "
            f"unlisted={unlisted_so_far} unknown={unknown_so_far}",
            file=sys.stderr,
        )

    time.sleep(SLEEP_SEC)

out_f.flush()
out_f.close()

# ── 최종 집계 ──────────────────────────────────────────────────────────────

listed   = [r for r in results if r["stdict"] == "listed"]
unlisted = [r for r in results if r["stdict"] == "unlisted"]
unknown  = [r for r in results if r["stdict"] == "unknown"]

print(f"\n=== 결과 ===", file=sys.stderr)
print(f"총 {total}개  |  등재: {len(listed)}  |  미등재: {len(unlisted)}  |  미상: {len(unknown)}", file=sys.stderr)
print(f"출력 파일: {OUTPUT}", file=sys.stderr)

# stdout 에 집계 + 미등재 상위 30개
print(f"총 {total}개  |  등재: {len(listed)}  |  미등재: {len(unlisted)}  |  미상: {len(unknown)}")
print(f"\n미등재 상위 30개 (빈도순):")
print(f"{'순위':>4}  {'단어':<14}  {'빈도':>5}  분리")
print("-" * 55)
for rank, r in enumerate(sorted(unlisted, key=lambda x: -x["freq"])[:30], 1):
    split_str = "+".join(f"{m}/{t}" for m, t in r["split"])
    print(f"{rank:>4}  {r['surface']:<14}  {r['freq']:>5}  {split_str}")
