"""Surface 정규화: NFC 자모 정규화 + ep_norm (eval_f1과 동일 함수 재사용).

스펙: .specs/2026-05-21-dict-expansion-design.md 섹션 2.4.
"""
import os, sys, unicodedata

_EP_NORM_DIR = os.path.join(
    os.path.dirname(__file__), "..", "gold_testset", "expand"
)
sys.path.insert(0, _EP_NORM_DIR)
from ep_norm import normalize_ep_morphemes  # noqa: E402


def nfc_normalize(text: str) -> str:
    """호환자모/결합자모 NFC 정규화."""
    return unicodedata.normalize("NFC", text)


def normalize_token_list(tokens: list[list[str]]) -> list[list[str]]:
    """[(surface, pos), ...] 토큰 리스트에 ep_norm 적용."""
    return list(normalize_ep_morphemes(tokens))
