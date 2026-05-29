"""5분석기 POS schema → 세종 42태그 정규화.

스펙: .specs/2026-05-21-dict-expansion-design.md 섹션 2.3.
"""

KKMA_TO_SEJONG = {
    "OH": "SH",   # 한자
    "OL": "SL",   # 외국어
    "ON": "SN",   # 숫자
    "NNM": "NNB", # 단위명사 → 의존명사
    "EFN": "EF",  # 평서형 종결어미
    "EFI": "EF",  # 의문형 종결어미
    "EFQ": "EF",  # 의문형 종결어미 (other)
    "EFO": "EF",  # 명령형 종결어미
    "EFA": "EF",  # 청유형 종결어미
    "EFR": "EF",  # 존칭형 종결어미
    "EPT": "EP",  # 시제 선어말어미
    "EPH": "EP",  # 주체높임 선어말어미
    "EPP": "EP",  # 공손 선어말어미
    "ECD": "EC",  # 종속적 연결어미
    "ECE": "EC",  # 대등적 연결어미
    "ECS": "EC",  # 보조적 연결어미
    "ETD": "ETM", # 관형사형 전성어미
    "JKM": "JKB", # 부사격 조사
    "VXV": "VX",  # 보조동사
    "VXA": "VX",  # 보조형용사
    "MDT": "MM",  # 지시 관형사
    "MDN": "MM",  # 수 관형사
    "UN":  "UNKNOWN",
}

KIWI_TO_SEJONG = {
    "SSO": "SS",  # opening bracket
    "SSC": "SS",  # closing bracket
}

MECAB_TO_SEJONG = {
    "SY":   "SW",  # symbol
    "SSO":  "SS",
    "SSC":  "SS",
    "SC":   "SP",  # 쉼표
    "NNBC": "NNB", # 단위성 의존명사
}

def normalize_pos(tag: str, source: str) -> str:
    """단일 POS 태그를 세종 표준으로 매핑. 매핑 없으면 그대로 반환."""
    if source == "kkma":
        return KKMA_TO_SEJONG.get(tag, tag)
    if source == "kiwi":
        return KIWI_TO_SEJONG.get(tag, tag)
    if source == "mecab":
        return MECAB_TO_SEJONG.get(tag, tag)
    return tag

def split_compound_pos(tag: str) -> list[str]:
    """Mecab의 'EP+EF' 같은 컴파운드 태그를 리스트로 분리."""
    if "+" in tag:
        return tag.split("+")
    return [tag]
