"""hangul_jamo 단위 테스트."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hangul_jamo import decompose_syllable, compose_syllable, attach_jongseong


def test_decompose_basic():
    assert decompose_syllable("한") == ("ㅎ", "ㅏ", "ㄴ")
    assert decompose_syllable("하") == ("ㅎ", "ㅏ", "")
    assert decompose_syllable("값") == ("ㄱ", "ㅏ", "ㅄ")


def test_decompose_non_hangul():
    assert decompose_syllable("A") is None
    assert decompose_syllable(".") is None


def test_compose_basic():
    assert compose_syllable("ㅎ", "ㅏ", "ㄴ") == "한"
    assert compose_syllable("ㅎ", "ㅏ", "") == "하"


def test_compose_roundtrip():
    for ch in "한글형태소분석기":
        cho, jung, jong = decompose_syllable(ch)
        assert compose_syllable(cho, jung, jong) == ch


def test_attach_jongseong_to_open_syllable():
    # "하"(받침없음) + 종성 ㄴ → "한"
    assert attach_jongseong("하", "ㄴ") == "한"
    assert attach_jongseong("가", "ㄹ") == "갈"


def test_attach_jongseong_fails_on_closed_syllable():
    # 이미 종성이 있으면 결합 불가 → None
    assert attach_jongseong("한", "ㄴ") is None


def test_attach_jongseong_non_jongseong_char():
    # 종성으로 쓸 수 없는 자모 → None
    assert attach_jongseong("하", "ㅏ") is None
