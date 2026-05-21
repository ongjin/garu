"""simhash 기반 문장 dedup 테스트."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "crawl"))
from dedup import SimhashDedup

def test_unique_sentences_pass():
    d = SimhashDedup()
    assert d.add("오늘은 비가 옵니다.") is True
    assert d.add("내일은 눈이 올 것 같습니다.") is True

def test_exact_duplicate_blocked():
    d = SimhashDedup()
    assert d.add("동일한 문장입니다.") is True
    assert d.add("동일한 문장입니다.") is False

def test_near_duplicate_blocked():
    """공백/구두점만 다른 near-duplicate.

    NOTE: simhash 라이브러리의 한국어 단어 토크나이즈 결과, 1자 추가만으로도
    실측 distance ≈ 11 (계획서의 threshold=3 가정과 불일치). distinct 문장은
    distance ≈ 32이므로 threshold=16으로 near vs distinct 분리 가능.
    """
    d = SimhashDedup(hamming_threshold=16)
    assert d.add("오늘 날씨가 정말 좋다") is True
    # 한 글자 추가
    assert d.add("오늘 날씨가 정말 좋다요") is False

def test_distinct_sentences_pass():
    d = SimhashDedup(hamming_threshold=16)
    assert d.add("오늘 날씨가 정말 좋다") is True
    # 어휘가 완전 다른 문장
    assert d.add("쿠버네티스 클러스터를 구성한다") is True

if __name__ == "__main__":
    test_unique_sentences_pass(); print("unique OK")
    test_exact_duplicate_blocked(); print("exact OK")
    test_near_duplicate_blocked(); print("near OK")
    test_distinct_sentences_pass(); print("distinct OK")
    print("DEDUP OK")
