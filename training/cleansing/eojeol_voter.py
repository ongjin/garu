"""어절별 5분석기 다수결 voting.

다수결 단위: morpheme 시퀀스 = [(surface, POS), ...] 전체 동일성.
임계값: 3+ 분석기가 같은 시퀀스 → 정상, 그 외 → 의심.

동률(2-2-1, 1-1-1-1-1 등) 케이스도 3/5+ 미달이므로 자동 의심.
빈 출력은 빈 시퀀스 키 `()` 로 그룹화 — 실패 분석기끼리 단일 그룹을 형성한다 (downstream 노이즈 식별에 유용).

morpheme 표현은 JSON-friendly한 list[list[str]]로 반환.
"""
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional


@dataclass
class EojeolVoteResult:
    surface: str
    status: str           # "normal" | "suspicious"
    agree: int            # 다수파 분석기 수 (1~5)
    morphemes: Optional[list]  # 정상일 때 채택된 시퀀스 (list[list[str]])
    candidates: list[dict]  # [{"analyzers": [...], "morphemes": [...]}, ...] (의심일 때 raw 후보)


def _seq_key(seq) -> tuple:
    """morpheme 시퀀스를 해시 가능한 키로 변환.

    입력은 list[tuple|list[str, str]] 모두 허용.
    """
    return tuple((s, p) for s, p in seq)


def vote_eojeol(
    surface: str,
    analyses: dict,
    threshold: int = 3,
) -> EojeolVoteResult:
    """5분석기 분석 결과로부터 어절 voting 수행.

    Args:
        surface: 어절 원형 (디버그/저장용).
        analyses: {analyzer_name: [(surface, pos), ...]}. 5분석기 입력 가정.
        threshold: 다수결 임계 (기본 3 = 3/5+).
    """
    groups: dict = defaultdict(list)
    for name, seq in analyses.items():
        groups[_seq_key(seq)].append(name)

    # 가장 큰 그룹
    best_key = max(groups, key=lambda k: len(groups[k]))
    best_size = len(groups[best_key])

    if best_size >= threshold:
        return EojeolVoteResult(
            surface=surface,
            status="normal",
            agree=best_size,
            morphemes=[list(t) for t in best_key],
            candidates=[],
        )

    # 의심: 모든 후보 그룹 보존 (큰 그룹부터)
    cands = []
    for key, names in sorted(groups.items(), key=lambda x: -len(x[1])):
        cands.append({
            "analyzers": sorted(names),
            "morphemes": [list(t) for t in key],
        })
    return EojeolVoteResult(
        surface=surface,
        status="suspicious",
        agree=best_size,
        morphemes=None,
        candidates=cands,
    )
