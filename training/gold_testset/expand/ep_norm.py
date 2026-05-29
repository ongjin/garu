"""형태소 표기 정규화 (비교용 키 생성).

비교 키 만들 때만 적용. 원본 morphemes 자체는 보존.
- EP 축약 분해: 했=하+았, 였=이+었
- Jamo 통일: 호환 자모 → 결합 자모 (ㄴ→ᆫ 등)
- 모음조화 통일 EC/EP: 었→았, 어→아 등 (음성 → 양성으로 통일)
- 태그 하위분류 통일: SSO/SSC → SS 등
"""

CONTRACT_MAP = {
    ("했", "XSV+EP"): [("하", "XSV"), ("았", "EP")],
    ("였", "VCP+EP"): [("이", "VCP"), ("었", "EP")],
}

# 호환 자모 (U+3130~U+318F) → 결합 자모 종성 (U+11A8~U+11FF)
COMPAT_TO_COMBINING = {
    "ㄱ": "ᆨ", "ㄲ": "ᆩ", "ㄳ": "ᆪ", "ㄴ": "ᆫ", "ㄵ": "ᆬ",
    "ㄶ": "ᆭ", "ㄷ": "ᆮ", "ㄹ": "ᆯ", "ㄺ": "ᆰ", "ㄻ": "ᆱ",
    "ㄼ": "ᆲ", "ㄽ": "ᆳ", "ㄾ": "ᆴ", "ㄿ": "ᆵ", "ㅀ": "ᆶ",
    "ㅁ": "ᆷ", "ㅂ": "ᆸ", "ㅄ": "ᆹ", "ㅅ": "ᆺ", "ㅆ": "ᆻ",
    "ㅇ": "ᆼ", "ㅈ": "ᆽ", "ㅊ": "ᆾ", "ㅋ": "ᆿ", "ㅌ": "ᇀ",
    "ㅍ": "ᇁ", "ㅎ": "ᇂ",
}

# 모음조화 시작 음절 매핑 (어/었/워/웠 등) — EC/EP/EF 한정
# 여/였: 하다 여불규칙 (하+여→해, 하+였→했). 하+아/하+았와 동치로 통일.
VOWEL_HARMONY_PREFIX = {
    "어": "아", "었": "았", "워": "와", "웠": "왔",
    "여": "아", "였": "았",
}

# 태그 하위분류 통일
TAG_NORM = {
    "SSO": "SS", "SSC": "SS",  # 묶음표 열림/닫힘
    "SS": "SS",
}


def _convert_leading_jamo(surface: str) -> str:
    """첫 글자가 호환 자모면 결합 자모로 치환."""
    if surface and surface[0] in COMPAT_TO_COMBINING:
        return COMPAT_TO_COMBINING[surface[0]] + surface[1:]
    return surface


def _apply_vowel_harmony(surface: str, pos: str) -> str:
    """EC/EP/EF의 모음조화 prefix를 양성으로 통일."""
    if pos not in ("EC", "EP", "EF"):
        return surface
    if surface and surface[0] in VOWEL_HARMONY_PREFIX:
        return VOWEL_HARMONY_PREFIX[surface[0]] + surface[1:]
    return surface


def _norm_surface(surface: str, pos: str) -> str:
    surface = _convert_leading_jamo(surface)
    surface = _apply_vowel_harmony(surface, pos)
    return surface


def _norm_pos(pos: str) -> str:
    return TAG_NORM.get(pos, pos)


def normalize_ep_morphemes(morphemes: list) -> list:
    """비교용 정규화된 morpheme 리스트 반환. 원본 morphemes 변경 안 함."""
    out = []
    for surface, pos in morphemes:
        key = (surface, pos)
        if key in CONTRACT_MAP:
            for s, p in CONTRACT_MAP[key]:
                out.append([_norm_surface(s, p), _norm_pos(p)])
        else:
            out.append([_norm_surface(surface, pos), _norm_pos(pos)])
    return out


if __name__ == "__main__":
    # 기존 EP 축약 테스트
    a = [["하", "XSV"], ["았", "EP"], ["다", "EF"]]
    b = [["했", "XSV+EP"], ["다", "EF"]]
    assert normalize_ep_morphemes(a) == normalize_ep_morphemes(b)

    c = [["이", "VCP"], ["었", "EP"], ["다", "EF"]]
    d = [["였", "VCP+EP"], ["다", "EF"]]
    assert normalize_ep_morphemes(c) == normalize_ep_morphemes(d)

    e = [["밥", "NNG"], ["을", "JKO"], ["먹", "VV"], ["다", "EF"]]
    assert normalize_ep_morphemes(e) == e

    # 자모 통일 (ㄴ↔ᆫ)
    j1 = [["크", "VA"], ["ㄴ", "ETM"]]
    j2 = [["크", "VA"], ["ᆫ", "ETM"]]
    assert normalize_ep_morphemes(j1) == normalize_ep_morphemes(j2)

    # 자모 통일 (ㄹ↔ᆯ)
    j3 = [["벌어지", "VV"], ["ㄹ", "ETM"]]
    j4 = [["벌어지", "VV"], ["ᆯ", "ETM"]]
    assert normalize_ep_morphemes(j3) == normalize_ep_morphemes(j4)

    # 모음조화 EP (았↔었)
    v1 = [["하", "XSV"], ["았", "EP"], ["다", "EF"]]
    v2 = [["하", "XSV"], ["었", "EP"], ["다", "EF"]]
    assert normalize_ep_morphemes(v1) == normalize_ep_morphemes(v2)

    # 모음조화 EC (아↔어)
    v3 = [["통하", "VV"], ["아", "EC"]]
    v4 = [["통하", "VV"], ["어", "EC"]]
    assert normalize_ep_morphemes(v3) == normalize_ep_morphemes(v4)

    # 하다 여불규칙 EP (았↔였): 하+았 == 하+였
    h1 = [["하", "XSV"], ["았", "EP"], ["다", "EF"]]
    h2 = [["하", "XSV"], ["였", "EP"], ["다", "EF"]]
    assert normalize_ep_morphemes(h1) == normalize_ep_morphemes(h2)

    # 하다 여불규칙 EC (아↔여): 하+아 == 하+여
    h3 = [["대립하", "VV"], ["아", "EC"]]
    h4 = [["대립하", "VV"], ["여", "EC"]]
    assert normalize_ep_morphemes(h3) == normalize_ep_morphemes(h4)

    # 태그 하위분류 (SS↔SSC)
    t1 = [["”", "SS"]]
    t2 = [["”", "SSC"]]
    assert normalize_ep_morphemes(t1) == normalize_ep_morphemes(t2)

    # 한꺼번에: ㄴ/ㄹ 자모 + 모음조화 + SSC
    big1 = [["가", "VV"], ["ㄴ", "ETM"], ["수", "NNB"], ["있", "VA"], ["었", "EP"], ["다", "EF"], ["”", "SS"]]
    big2 = [["가", "VV"], ["ᆫ", "ETM"], ["수", "NNB"], ["있", "VA"], ["았", "EP"], ["다", "EF"], ["”", "SSC"]]
    assert normalize_ep_morphemes(big1) == normalize_ep_morphemes(big2)

    # 다음절 어미 leading jamo + 모음조화: ㄴ다 / ㅂ니다 / 어야 / 어서
    m1 = [["하", "VV"], ["ㄴ다", "EF"]]
    m2 = [["하", "VV"], ["ᆫ다", "EF"]]
    assert normalize_ep_morphemes(m1) == normalize_ep_morphemes(m2)

    n1 = [["하", "VV"], ["ㅂ니다", "EF"]]
    n2 = [["하", "VV"], ["ᆸ니다", "EF"]]
    assert normalize_ep_morphemes(n1) == normalize_ep_morphemes(n2)

    o1 = [["가", "VV"], ["아야", "EC"]]
    o2 = [["가", "VV"], ["어야", "EC"]]
    assert normalize_ep_morphemes(o1) == normalize_ep_morphemes(o2)

    p1 = [["가", "VV"], ["아서", "EC"]]
    p2 = [["가", "VV"], ["어서", "EC"]]
    assert normalize_ep_morphemes(p1) == normalize_ep_morphemes(p2)

    # SSO도 SS와 매치
    so1 = [["“", "SS"]]
    so2 = [["“", "SSO"]]
    assert normalize_ep_morphemes(so1) == normalize_ep_morphemes(so2)

    print("OK")
