"""HTML → 한국어 문장 리스트 추출.

trafilatura로 본문 추출 → 문장 분리 → 한국어 필터.
필터: 한글 비율 ≥60%, 길이 5-200자.
"""
import re

import trafilatura

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|(?<=[다요죠])\s+(?=[가-힣])")
_HANGUL_CHAR = re.compile(r"[가-힣]")


def is_korean_sentence(s: str, min_len: int = 5, max_len: int = 200,
                       min_korean_ratio: float = 0.6) -> bool:
    """한국어 문장 필터: 길이 + 한글 비율."""
    if not s or len(s) < min_len or len(s) > max_len:
        return False
    hangul_count = len(_HANGUL_CHAR.findall(s))
    return (hangul_count / len(s)) >= min_korean_ratio


def extract_sentences(html_or_text: str) -> list[str]:
    """HTML 본문 추출 → 문장 분리 → 한국어 필터."""
    text = trafilatura.extract(html_or_text) if "<" in html_or_text else html_or_text
    if not text:
        return []
    sents = _SENT_SPLIT.split(text)
    cleaned: list[str] = []
    for s in sents:
        s = s.strip()
        if is_korean_sentence(s):
            cleaned.append(s)
    return cleaned
