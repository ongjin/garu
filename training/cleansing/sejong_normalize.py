"""의심 어절 후보를 세종 표준형으로 정규화하고 수렴을 판정.

규칙 (보수적 — 확신 없으면 변형하지 않아 수렴 실패 → Claude 폴백):
  - unify_ep: EP(았/었/였) 모음조화 통일 (하→였, 양성→았, 음성→었)
  - is_merge_of / candidates_converge: 받침 단순결합 입도 정규화 + 수렴 판정

false convergence(틀린 자동 결정)를 만들지 않는 것이 최우선.
"""
from typing import Optional

from hangul_jamo import decompose_syllable, attach_jongseong

# 양성모음 (모음조화상 '았' 선택). ㅏ, ㅗ 계열.
_POSITIVE_VOWELS = set("ㅏㅑㅗㅛㅘㅙ")  # ㅘ/ㅙ = ㅗ 계열 양성 복합모음 (봐→봤)


def _last_vowel(surface: str) -> Optional[str]:
    """surface의 마지막 한글 음절 중성(모음). 없으면 None."""
    for ch in reversed(surface):
        d = decompose_syllable(ch)
        if d is not None:
            return d[1]
    return None


def unify_ep(morphemes: list) -> list:
    """EP(았/었/였)를 세종 모음조화 표준형으로 통일. 새 리스트 반환.

    어간(직전 형태소)이 '하' → '였'.
    그 외: 어간 끝 모음 양성(ㅏㅑㅗㅛ) → '았', 음성 → '었'.
    EP가 첫 형태소거나 어간 모음 판별 불가 시 변경하지 않음.
    """
    out = [list(m) for m in morphemes]
    for i, (s, p) in enumerate(out):
        if p == "EP" and s in ("았", "었", "였"):
            if i == 0:
                continue
            stem = out[i - 1][0]
            if stem == "하":
                out[i][0] = "였"
            else:
                v = _last_vowel(stem)
                if v is None:
                    continue
                out[i][0] = "았" if v in _POSITIVE_VOWELS else "었"
    return out


def _combine_two(stem_surface: str, ending_surface: str) -> Optional[str]:
    """어간 표면 + 어미 표면을 받침 단순결합. 결합 불가 시 None.

    어미가 단일 종성자모(ㄴ/ㄹ/ㅁ 등)로 시작하면 어간 끝음절에 종성으로 붙임.
    예: ("하","ㄴ다") → "한다", ("하","ㄹ") → "할".
    모음축약/불규칙(가+았=갔)은 처리하지 않음 → None (보수적).
    """
    if not ending_surface:
        return None
    head = ending_surface[0]
    rest = ending_surface[1:]
    # head가 종성 가능 자모일 때만 결합 시도
    attached = attach_jongseong(stem_surface[-1], head) if stem_surface else None
    if attached is None:
        return None
    return stem_surface[:-1] + attached + rest


def is_merge_of(fine: list, coarse: list) -> bool:
    """fine(더 잘게 분해)의 마지막 두 형태소를 받침결합하면 coarse가 되는가.

    조건: len(coarse) == len(fine) - 1, 앞부분(surface+pos) 동일,
          fine 마지막 둘의 표면 결합 == coarse 마지막 표면.
    POS는 결합 시 coarse 마지막 POS가 fine 어간 POS와 일치하지 않아도 됨
    (분석기가 합친 형태에 다른 POS를 줄 수 있으므로 표면 일치만 요구).
    """
    if len(coarse) != len(fine) - 1:
        return False
    # 앞부분 동일 (surface + pos)
    for a, b in zip(fine[:-2], coarse[:-1]):
        if a[0] != b[0] or a[1] != b[1]:
            return False
    combined = _combine_two(fine[-2][0], fine[-1][0])
    if combined is None:
        return False
    return combined == coarse[-1][0]


def _key(morphemes: list) -> tuple:
    return tuple((m[0], m[1]) for m in morphemes)


def candidates_converge(candidates: list) -> Optional[list]:
    """후보들을 세종 정규화 후 모두 동일하면 그 시퀀스, 아니면 None.

    1. 각 후보 EP 통일
    2. 통일 후 전부 동일 → 그 시퀀스
    3. 아니면 최대분해 후보가 나머지 모두의 병합 원형인지 검증 (입도)
       - 나머지 각각이 maximal과 동일하거나 maximal의 병합형이면 maximal 채택
    """
    normed = [unify_ep(c["morphemes"]) for c in candidates]
    keys = {_key(m) for m in normed}
    if len(keys) == 1:
        return normed[0]

    # 최대 형태소 수 후보 (동률이면 첫 번째)
    maximal = max(normed, key=len)
    for other in normed:
        if _key(other) == _key(maximal):
            continue
        if not is_merge_of(maximal, other):
            return None
    return maximal
