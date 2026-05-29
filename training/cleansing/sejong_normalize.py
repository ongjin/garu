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
