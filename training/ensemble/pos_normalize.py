"""5분석기 POS schema → 세종 42태그 정규화.

스펙: .specs/2026-05-21-dict-expansion-design.md 섹션 2.3.
"""

KKMA_TO_SEJONG = {
    "OH": "SH",   # 한자
    "OL": "SL",   # 외국어
    "ON": "SN",   # 숫자
    "NNM": "NNB", # 단위명사 → 의존명사
}

def normalize_pos(tag: str, source: str) -> str:
    """단일 POS 태그를 세종 표준으로 매핑. 매핑 없으면 그대로 반환."""
    if source == "kkma":
        return KKMA_TO_SEJONG.get(tag, tag)
    return tag

def split_compound_pos(tag: str) -> list[str]:
    """Mecab의 'EP+EF' 같은 컴파운드 태그를 리스트로 분리."""
    if "+" in tag:
        return tag.split("+")
    return [tag]
