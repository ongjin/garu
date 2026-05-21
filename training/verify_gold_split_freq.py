#!/usr/bin/env python3
"""
71개 content_dict 제거 단어에 대해 gold testset 양방향 공존 빈도 확인.
각 단어 X에 대해:
  - single: gold에서 (X, NNG) 단일 형태소로 등장한 횟수
  - split:  gold에서 A+B == X 인 인접 NNG+NNG 쌍으로 등장한 횟수
분류: 단일 우세 / 분해 우세 / ambiguous / 무영향
"""

import json
from pathlib import Path

REPO = Path(__file__).parent.parent
TSV_PATH = REPO / "training/temp_compound_sources.tsv"
GOLD_PATH = REPO / "training/gold_testset/gold_testset.jsonl"
BACKUP_PATH = REPO / "training/codebook_data/content_dict.txt.bak.predisambig"
DICT_PATH = REPO / "training/codebook_data/content_dict.txt"

# ── Step 1: 71개 단어 추출 ────────────────────────────────────────────────────
targets = {}  # surface -> split_form (from TSV)
with open(TSV_PATH) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("surface"):
            continue
        parts = line.split("\t")
        surface, freq, in_cache, in_dict = parts[0], parts[1], parts[2], parts[3]
        split_form = parts[5] if len(parts) > 5 else ""
        if in_dict == "Y":
            targets[surface] = split_form

assert len(targets) == 71, f"Expected 71, got {len(targets)}"
print(f"[Step 1] 대상 단어 {len(targets)}개 추출 완료\n")

# ── Step 2: Gold testset 양방향 빈도 카운트 ───────────────────────────────────
single_count = {w: 0 for w in targets}
split_count  = {w: 0 for w in targets}

with open(GOLD_PATH) as f:
    for line in f:
        obj = json.loads(line)
        morphs = obj["morphemes"]  # list of [surface, tag]
        n = len(morphs)

        for i, (surf, tag) in enumerate(morphs):
            # single NNG
            if tag == "NNG" and surf in targets:
                single_count[surf] += 1

        # split: adjacent NNG+NNG where A+B == target
        for i in range(n - 1):
            s1, t1 = morphs[i]
            s2, t2 = morphs[i + 1]
            if t1 == "NNG" and t2 == "NNG":
                combined = s1 + s2
                if combined in targets:
                    split_count[combined] += 1

# ── Step 3: 분류 ─────────────────────────────────────────────────────────────
revert_words   = []  # single>0, split==0  → dict 복원
keep_words     = []  # split>0, single==0  → 분해 유지
ambiguous_words = [] # both>0              → 별도 검토
no_data_words  = []  # both==0             → 데이터 없음

for word in sorted(targets):
    s = single_count[word]
    sp = split_count[word]
    if s > 0 and sp == 0:
        revert_words.append((word, s, sp))
    elif sp > 0 and s == 0:
        keep_words.append((word, s, sp))
    elif s > 0 and sp > 0:
        ambiguous_words.append((word, s, sp))
    else:
        no_data_words.append((word, s, sp))

print("=" * 60)
print(f"분류 결과 (총 {len(targets)}개)")
print("=" * 60)
print(f"  단일 우세 (revert) : {len(revert_words):3d}개")
print(f"  분해 우세 (keep)   : {len(keep_words):3d}개")
print(f"  ambiguous          : {len(ambiguous_words):3d}개")
print(f"  무영향 (no data)   : {len(no_data_words):3d}개")
print()

print("── 단일 우세 (revert 대상) ──────────────────────────────")
for w, s, sp in sorted(revert_words, key=lambda x: -x[1]):
    print(f"  {w:20s}  single={s:3d}  split={sp}")
print()

print("── 분해 우세 (유지) ─────────────────────────────────────")
for w, s, sp in sorted(keep_words, key=lambda x: -x[2]):
    print(f"  {w:20s}  single={s:3d}  split={sp}")
print()

print("── ambiguous ────────────────────────────────────────────")
for w, s, sp in sorted(ambiguous_words, key=lambda x: -(x[1]+x[2])):
    print(f"  {w:20s}  single={s:3d}  split={sp}")
print()

print("── 무영향 (gold 미출현) ─────────────────────────────────")
for w, s, sp in sorted(no_data_words):
    print(f"  {w:20s}  single={s}  split={sp}")
print()

# ── Step 4: revert 실행 ───────────────────────────────────────────────────────
if not revert_words:
    print("[Step 4] revert 대상 없음. content_dict.txt 수정 불필요.")
else:
    revert_set = {w for w, s, sp in revert_words}

    # 백업에서 해당 단어들의 NNG 라인 추출
    lines_to_add = []
    with open(BACKUP_PATH) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[0] in revert_set and parts[1] == "NNG":
                lines_to_add.append(line.rstrip("\n"))

    print(f"[Step 4] 백업에서 추출한 NNG 라인: {len(lines_to_add)}개")

    # 현재 content_dict.txt에 없는 라인만 추가
    with open(DICT_PATH) as f:
        existing = set(f.read().splitlines())

    new_lines = [l for l in lines_to_add if l not in existing]
    print(f"[Step 4] 추가할 신규 라인: {len(new_lines)}개")

    if new_lines:
        with open(DICT_PATH, "a") as f:
            for l in new_lines:
                f.write(l + "\n")

        # 정렬 (content_dict.txt는 정렬 필요 여부 확인 후)
        with open(DICT_PATH) as f:
            all_lines = f.read().splitlines()
        all_lines_sorted = sorted(set(all_lines))
        with open(DICT_PATH, "w") as f:
            f.write("\n".join(all_lines_sorted) + "\n")

        print(f"[Step 4] content_dict.txt 갱신 완료 (정렬 포함). 추가된 라인: {len(new_lines)}개")
        for l in sorted(new_lines):
            print(f"  + {l}")
    else:
        print("[Step 4] 추가할 신규 라인 없음 (이미 존재).")

print("\n[Done] verify_gold_split_freq.py 완료.")
