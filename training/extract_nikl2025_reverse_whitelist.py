"""Build the 하/되 병합·분리 whitelist used by reverse_2025_to_2021.py.

Learns, from the NIKL MP **2021** gold, which 하/되/시키/받 표면을 어휘화 용언으로
'병합 유지'하고 (대하/위하/좋아하…) 어떤 명사가 생산적 하/되를 취해 '분리'되는지를
(방문/NNG+하/XSV…) 코퍼스 통계로 결정한다. 결과는 두 집합으로 커밋된다:

  keep_merged : 병합 유지 표면 — 2021이 단일 VV/VA 형태소로 유지하며, 그 빈도가
                동일 어기의 명사+하/되 분리 빈도 이상인 경우 (mc >= max(sc,1)).
  split_nouns : 생산적 하/되/시키/받을 취하는 어기명사 (content_dict로 커버 안 되는
                83종 어근 포함 — reverse 변환의 분리 판정 fallback).

2021 코퍼스 원문은 커밋하지 않는다 — 이 스크립트로 재생성한다.

Usage:
    NIKL_MP_DIR=~/workspace/data/nikl_mp_2021 python3 training/extract_nikl2025_reverse_whitelist.py
    → training/nikl2025_reverse_whitelist.json 갱신
"""
import json, os, sys
from collections import Counter
from pathlib import Path

NIKL = Path(os.environ.get("NIKL_MP_DIR", str(Path.home() / "workspace" / "data" / "nikl_mp_2021")))
OUT = Path(__file__).parent / "nikl2025_reverse_whitelist.json"
VERBALIZERS = ["시키", "당하", "받", "되", "하"]  # longest-first


def base_of(surface):
    for s in VERBALIZERS:
        if surface.endswith(s) and len(surface) > len(s):
            return surface[:-len(s)]
    return None


def main():
    merged = Counter()      # single VV/VA surface ending in 하/되/... (lexicalized)
    split_noun = Counter()  # noun immediately before productive 하/되/시키/받 XSV/XSA
    for path in sorted(NIKL.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        for doc in data["document"]:
            if doc is None:
                continue
            for sent in (doc.get("sentence") or []):
                ms = sent.get("morpheme") or sent.get("MP") or []
                for i, m in enumerate(ms):
                    form, label = m.get("form", ""), m.get("label", "")
                    if label in ("VV", "VA") and len(form) >= 2 and form[-1] in "하되":
                        merged[form] += 1
                    if label in ("XSV", "XSA") and i > 0:
                        pf, pl = ms[i - 1].get("form", ""), ms[i - 1].get("label", "")
                        if pl in ("NNG", "NNP", "XR"):
                            split_noun[pf] += 1
        print(f"  scanned {path.name}", file=sys.stderr)

    # keep_merged: lexicalized surface whose merged count dominates the split count
    keep_merged = sorted(
        f for f, c in merged.items()
        if base_of(f) is not None and c > 0 and c >= max(split_noun.get(base_of(f), 0), 1)
    )
    split_nouns = sorted(split_noun.keys())

    json.dump({"keep_merged": keep_merged, "split_nouns": split_nouns},
              open(OUT, "w"), ensure_ascii=False, indent=0)
    print(f"keep_merged={len(keep_merged)}  split_nouns={len(split_nouns)}  → {OUT}")


if __name__ == "__main__":
    main()
