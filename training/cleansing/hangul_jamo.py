"""한글 음절 ↔ 자모 분해/조합 (유니코드 산술). 외부 의존성 없음.

세종 입도 정규화에서 어간+어미 받침 결합 검증에 사용.
예: "하"(받침 없음) + 종성 "ㄴ" → "한".
"""
CHO = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
JUNG = list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
JONG = [""] + list("ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")

_BASE = 0xAC00
_N_JUNG = 21
_N_JONG = 28


def decompose_syllable(ch: str):
    """완성형 한글 음절 → (초성, 중성, 종성). 한글 아니면 None.
    종성 없으면 종성은 빈 문자열."""
    code = ord(ch) - _BASE
    if 0 <= code < 11172:
        cho = code // (_N_JUNG * _N_JONG)
        jung = (code % (_N_JUNG * _N_JONG)) // _N_JONG
        jong = code % _N_JONG
        return (CHO[cho], JUNG[jung], JONG[jong])
    return None


def compose_syllable(cho: str, jung: str, jong: str = "") -> str:
    """(초성, 중성, 종성) → 완성형 음절."""
    ci = CHO.index(cho)
    ji = JUNG.index(jung)
    ki = JONG.index(jong)
    return chr(_BASE + (ci * _N_JUNG + ji) * _N_JONG + ki)


def attach_jongseong(syllable: str, jong: str):
    """받침 없는 음절에 종성 자모를 결합. 결합 불가 시 None.

    - syllable이 완성형 한글이 아니면 None
    - 이미 종성이 있으면 None
    - jong이 종성 가능 자모가 아니면 None
    """
    d = decompose_syllable(syllable)
    if d is None:
        return None
    cho, jung, existing_jong = d
    if existing_jong != "":
        return None
    if jong not in JONG or jong == "":
        return None
    return compose_syllable(cho, jung, jong)
