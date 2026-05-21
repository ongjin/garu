"""4분석기 합의 점수 계산.

(surface, normalized_pos) 키별로 분석기 i가 단일 형태소로 인식했는지 0/1 vote.
POS 부분 불일치는 별도 후보로 분리 (스펙 섹션 2.5 — 정보 보존).
Garu는 합의 분모 외, in_garu_dict 메타만 표시.
"""
from typing import Iterable

ANALYZERS = ("kiwi", "mecab", "kkma", "komoran")


def _single_morpheme_keys(tokens: Iterable[tuple[str, str]]) -> set[tuple[str, str]]:
    """분석기가 단일 형태소로 인식한 (surface, pos) 키 집합.

    동일 surface가 여러 번 등장하면 한 번만 카운트 (단일 형태소 set 의미상).
    """
    return {(s, p) for s, p in tokens}


def compute_votes(
    analyses: dict[str, list[tuple[str, str]]],
    garu_analysis: list[tuple[str, str]],
) -> dict[tuple[str, str], dict]:
    """4분석기 분석 결과로부터 후보별 vote dict 생성.

    Args:
        analyses: {analyzer_name: [(surface, pos), ...]}, analyzer_name ∈ ANALYZERS
        garu_analysis: Garu의 단일 문장 분석 결과 (dedup용)

    Returns:
        {(surface, pos): {"votes": {kiwi, mecab, kkma, komoran}, "in_garu_dict": bool}}
        모든 분석기에서 1번이라도 단일로 등장한 (surface, pos)가 키.
    """
    per_analyzer_keys = {a: _single_morpheme_keys(analyses.get(a, [])) for a in ANALYZERS}
    garu_keys = _single_morpheme_keys(garu_analysis)

    all_keys: set[tuple[str, str]] = set()
    for s in per_analyzer_keys.values():
        all_keys |= s

    result: dict[tuple[str, str], dict] = {}
    for key in all_keys:
        votes = {a: (1 if key in per_analyzer_keys[a] else 0) for a in ANALYZERS}
        result[key] = {
            "votes": votes,
            "in_garu_dict": key in garu_keys,
        }
    return result


def total_votes(info: dict) -> int:
    """info["votes"]의 합 (0-4 범위)."""
    return sum(info["votes"].values())
