"""의심 어절 후보를 세종 표준형으로 정규화하고 수렴을 판정.

규칙 (보수적 — 확신 없으면 변형하지 않아 수렴 실패 → Claude 폴백):
  - unify_ep: EP(았/었/였) 모음조화 통일 (하→였, 양성→았, 음성→었)
  - candidates_converge: EP 통일 후 완전 일치 시 수렴 (그 외 → None, Claude 폴백)

false convergence(틀린 자동 결정)를 만들지 않는 것이 최우선.
"""
from typing import Optional

from hangul_jamo import decompose_syllable

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


def _key(morphemes: list) -> tuple:
    return tuple((m[0], m[1]) for m in morphemes)


def candidates_converge(candidates: list) -> Optional[list]:
    """후보들을 EP 통일 후 모두 동일하면 그 시퀀스, 아니면 None.

    EP 이형태(았/었/였)만 다른 후보는 세종 표준형으로 수렴(안전 — 같은 형태소
    시퀀스의 철자 정규화). 입도 차이·POS 충돌 등 그 외 모든 불일치는 None을
    반환해 Claude 검토로 넘긴다. (입도 정규화는 noun↔verb 동형이의에서
    false convergence를 만들어 제거됨.)
    """
    normed = [unify_ep(c["morphemes"]) for c in candidates]
    keys = {_key(m) for m in normed}
    if len(keys) == 1:
        return normed[0]
    return None
