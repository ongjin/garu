#!/usr/bin/env python3
"""
임시 합성명사 후보 추출 스크립트.

Garu가 단일 NNG로 처리하지만 Kiwi는 NNG+NNG로 분리하는 어절을 수집한다.
출력: training/temp_compound_candidates.jsonl
"""

import json
import re
import subprocess
import sys
import tempfile
import os
import random
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
MODEL_PATH = REPO_ROOT / "models" / "codebook.gmdl"
ANALYZE_BATCH = REPO_ROOT / "target" / "release" / "examples" / "analyze_batch"
KOWIKI_PATH = Path(__file__).parent / "kowikitext.txt"
GOLD_PATH = Path(__file__).parent / "gold_testset" / "gold_testset.jsonl"
OUTPUT_PATH = Path(__file__).parent / "temp_compound_candidates.jsonl"

KOWIKI_SAMPLE_SIZE = 50_000  # 줄 수
MIN_FREQ = 2


def load_kowiki_sentences(path: Path, n: int) -> list[str]:
    """kowikitext에서 첫 n줄을 읽어 비어있지 않은 줄만 반환."""
    lines = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
            if len(lines) >= n:
                break
    return lines


def load_gold_sentences(path: Path) -> list[str]:
    """gold_testset.jsonl에서 text 필드 추출."""
    sentences = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            sentences.append(obj["text"])
    return sentences


def sentences_to_eojeols(sentences: list[str]) -> list[str]:
    """
    문장 리스트에서 어절(공백 분리) 중 한글 포함된 것만 추출, 중복 제거.
    백슬래시 등 JSON 직렬화를 깨는 문자가 포함된 어절은 제외.
    """
    seen = set()
    result = []
    for sent in sentences:
        for tok in sent.split():
            # 한글 포함 어절만, 백슬래시 제외 (Garu JSON 직렬화 버그 회피)
            if re.search(r"[가-힣]", tok) and "\\" not in tok:
                if tok not in seen:
                    seen.add(tok)
                    result.append(tok)
    return result


def run_garu(eojeols: list[str]) -> list[list[tuple[str, str]]]:
    """
    analyze_batch로 어절 리스트를 분석한다.
    반환: 어절별 [(surface, pos), ...] 리스트
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", encoding="utf-8", delete=False) as f:
        tmp_path = f.name
        for eojeol in eojeols:
            f.write(eojeol + "\n")

    try:
        env = os.environ.copy()
        env["GARU_MODEL"] = str(MODEL_PATH)
        result = subprocess.run(
            [str(ANALYZE_BATCH), tmp_path, "--json"],
            capture_output=True,
            text=True,
            env=env,
        )
        if result.returncode != 0:
            print(f"[ERROR] analyze_batch failed:\n{result.stderr}", file=sys.stderr)
            sys.exit(1)

        lines = result.stdout.strip().split("\n")
        parsed = []
        for line in lines:
            line = line.strip()
            if not line:
                parsed.append([])
                continue
            tokens = json.loads(line)  # [["surface","POS"], ...]
            parsed.append([(t[0], t[1]) for t in tokens])
        return parsed
    finally:
        os.unlink(tmp_path)


def run_kiwi(eojeols: list[str]) -> list[list[tuple[str, str]]]:
    """
    kiwipiepy로 어절 리스트를 분석한다.
    반환: 어절별 [(surface, pos), ...] 리스트
    """
    try:
        from kiwipiepy import Kiwi
    except ImportError:
        print("[ERROR] kiwipiepy not installed. Run: pip3 install kiwipiepy", file=sys.stderr)
        sys.exit(1)

    # stderr 경고(Quantization) 억제는 불필요—그냥 허용
    kiwi = Kiwi()

    results = []
    # kiwi.analyze는 리스트로 배치 처리 가능
    # 반환: _ResIter → 각 항목은 list of (token_list, score), 첫 번째가 최선 분석
    batch_results = kiwi.analyze(eojeols)
    for item in batch_results:
        # item: [(token_list, score), ...]
        tokens_raw, _score = item[0]
        tokens = [(t.form, t.tag if isinstance(t.tag, str) else t.tag.name) for t in tokens_raw]
        results.append(tokens)
    return results


def extract_nng_prefix(morphemes: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """
    어절 morphemes에서 앞쪽 NNG 연속 구간을 반환.
    예: [(주유, NNG),(비, NNG),(가, JKS)] → [(주유, NNG),(비, NNG)]
        [(주유비, NNG),(가, JKS)] → [(주유비, NNG)]
    """
    prefix = []
    for surf, pos in morphemes:
        if pos == "NNG":
            prefix.append((surf, pos))
        else:
            break
    return prefix


def is_single_nng(morphemes: list[tuple[str, str]]) -> bool:
    """어절의 NNG prefix가 정확히 하나인가."""
    prefix = extract_nng_prefix(morphemes)
    return len(prefix) == 1


def is_double_nng(morphemes: list[tuple[str, str]]) -> tuple[bool, str, str]:
    """
    어절의 NNG prefix가 정확히 둘이고 그 surface 합이 뭔가 확인.
    반환: (해당 여부, part_a, part_b)
    """
    prefix = extract_nng_prefix(morphemes)
    if len(prefix) == 2:
        return True, prefix[0][0], prefix[1][0]
    return False, "", ""


def find_candidates(
    eojeols: list[str],
    garu_results: list[list[tuple[str, str]]],
    kiwi_results: list[list[tuple[str, str]]],
) -> dict[str, dict]:
    """
    Garu=단일NNG, Kiwi=NNG+NNG인 어절 후보를 freq 집계해 반환.
    반환: {surface: {"split": [["A","NNG"],["B","NNG"]], "freq": N}}
    """
    candidates: dict[str, dict] = {}
    freq: dict[str, int] = defaultdict(int)
    splits: dict[str, list] = {}

    for eojeol, g_morphs, k_morphs in zip(eojeols, garu_results, kiwi_results):
        if not g_morphs or not k_morphs:
            continue

        # Garu: 단일 NNG prefix
        if not is_single_nng(g_morphs):
            continue

        garu_nng_surface = g_morphs[0][0]

        # Kiwi: NNG+NNG prefix
        ok, part_a, part_b = is_double_nng(k_morphs)
        if not ok:
            continue

        kiwi_nng_surface = part_a + part_b

        # Garu NNG surface == Kiwi NNG+NNG surface
        if garu_nng_surface != kiwi_nng_surface:
            continue

        # 뒤쪽 조사 부분도 일치하는지 확인 (선택적 검증)
        garu_rest = [m for m in g_morphs if m[0] != garu_nng_surface or m == g_morphs[0]]
        # 간단하게: 후보 단어 surface 기준으로 집계
        surface = garu_nng_surface
        freq[surface] += 1
        if surface not in splits:
            splits[surface] = [[part_a, "NNG"], [part_b, "NNG"]]

    result = {}
    for surface, count in freq.items():
        if count >= MIN_FREQ:
            result[surface] = {"split": splits[surface], "freq": count}

    return result


def sanity_check_juyu_bi(
    eojeols: list[str],
    garu_results: list[list[tuple[str, str]]],
    kiwi_results: list[list[tuple[str, str]]],
) -> bool:
    """주유비가 후보로 검출되는지 확인. 없으면 False 반환."""
    for eojeol, g, k in zip(eojeols, garu_results, kiwi_results):
        if not g or g[0][0] != "주유비":
            continue
        if not is_single_nng(g):
            continue
        ok, a, b = is_double_nng(k)
        if ok and a + b == "주유비":
            return True
    return False


def inject_sanity_fixtures() -> tuple[list[str], list[list[tuple[str,str]]], list[list[tuple[str,str]]]]:
    """주유비가 소스에 없을 때를 대비한 fixture."""
    fixtures = ["주유비가", "주유비도", "주유비를"]
    garu_res = run_garu(fixtures)
    kiwi_res = run_kiwi(fixtures)
    return fixtures, garu_res, kiwi_res


def main():
    print("[1/5] 텍스트 소스 로딩...", file=sys.stderr)
    sentences = []
    if KOWIKI_PATH.exists():
        wiki_sents = load_kowiki_sentences(KOWIKI_PATH, KOWIKI_SAMPLE_SIZE)
        sentences.extend(wiki_sents)
        print(f"  kowikitext: {len(wiki_sents):,}줄", file=sys.stderr)
    else:
        print(f"  [WARN] {KOWIKI_PATH} 없음, 건너뜀", file=sys.stderr)

    if GOLD_PATH.exists():
        gold_sents = load_gold_sentences(GOLD_PATH)
        sentences.extend(gold_sents)
        print(f"  gold_testset: {len(gold_sents):,}문장", file=sys.stderr)
    else:
        print(f"  [WARN] {GOLD_PATH} 없음, 건너뜀", file=sys.stderr)

    print("[2/5] 어절 추출...", file=sys.stderr)
    eojeols = sentences_to_eojeols(sentences)
    print(f"  고유 한글 어절: {len(eojeols):,}", file=sys.stderr)

    print("[3/5] Garu 분석 중...", file=sys.stderr)
    garu_results = run_garu(eojeols)
    print(f"  완료 ({len(garu_results):,}개)", file=sys.stderr)

    print("[4/5] Kiwi 분석 중...", file=sys.stderr)
    kiwi_results = run_kiwi(eojeols)
    print(f"  완료 ({len(kiwi_results):,}개)", file=sys.stderr)

    print("[5/5] 불일치 어절 검출...", file=sys.stderr)
    candidates = find_candidates(eojeols, garu_results, kiwi_results)

    # Sanity check: 주유비 검출 여부
    found_juyu = sanity_check_juyu_bi(eojeols, garu_results, kiwi_results)
    if not found_juyu:
        print("  [SANITY] 주유비 미검출 — fixture 주입 후 재확인", file=sys.stderr)
        fix_eojeols, fix_garu, fix_kiwi = inject_sanity_fixtures()
        found_juyu2 = sanity_check_juyu_bi(fix_eojeols, fix_garu, fix_kiwi)
        if found_juyu2:
            print("  [SANITY] fixture로 주유비 검출 확인 (소스 텍스트에 미포함이었을 뿐)", file=sys.stderr)
            # fixture 결과도 candidate에 병합
            fix_candidates = find_candidates(fix_eojeols, fix_garu, fix_kiwi)
            for surf, info in fix_candidates.items():
                if surf not in candidates:
                    candidates[surf] = info
                else:
                    candidates[surf]["freq"] += info["freq"]
        else:
            print("  [SANITY FAIL] fixture로도 주유비 미검출 — 스크립트 로직 확인 필요", file=sys.stderr)
    else:
        print("  [SANITY OK] 주유비 검출 확인", file=sys.stderr)

    # 빈도 내림차순 정렬
    sorted_candidates = sorted(candidates.items(), key=lambda x: -x[1]["freq"])

    # 출력 파일 기록
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for surface, info in sorted_candidates:
            obj = {"surface": surface, "split": info["split"], "freq": info["freq"]}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # 통계
    total_eojeols = len(eojeols)
    mismatch_eojeols = sum(
        1 for eojeol, g, k in zip(eojeols, garu_results, kiwi_results)
        if g and k and is_single_nng(g) and is_double_nng(k)[0]
        and g[0][0] == is_double_nng(k)[1] + is_double_nng(k)[2]  # noqa: W503 (readability)
    )
    unique_candidates = len(sorted_candidates)

    print(f"\n=== 결과 ===", file=sys.stderr)
    print(f"  분석한 고유 어절 수: {total_eojeols:,}", file=sys.stderr)
    print(f"  불일치(Garu NNG / Kiwi NNG+NNG) 어절 수 (freq≥1): {mismatch_eojeols:,}", file=sys.stderr)
    print(f"  빈도 ≥ {MIN_FREQ} 고유 후보 단어 수: {unique_candidates:,}", file=sys.stderr)
    print(f"  출력 파일: {OUTPUT_PATH}", file=sys.stderr)

    print(f"\n상위 30개 후보:", file=sys.stderr)
    print(f"{'순위':>4}  {'단어':>12}  {'빈도':>5}  Kiwi 분리", file=sys.stderr)
    print("-" * 50, file=sys.stderr)
    for i, (surface, info) in enumerate(sorted_candidates[:30], 1):
        split_str = "+".join(p[0] for p in info["split"])
        print(f"{i:>4}  {surface:>12}  {info['freq']:>5}  {split_str}", file=sys.stderr)

    # stdout에도 JSON summary 출력 (파이프 활용 용이하게)
    summary = {
        "total_eojeols": total_eojeols,
        "mismatch_eojeols": mismatch_eojeols,
        "unique_candidates": unique_candidates,
        "output_path": str(OUTPUT_PATH),
        "top30": [
            {"surface": s, "split": info["split"], "freq": info["freq"]}
            for s, info in sorted_candidates[:30]
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
