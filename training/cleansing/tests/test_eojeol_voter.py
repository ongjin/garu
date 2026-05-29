"""eojeol_voter 단위 테스트.

다양한 voting 시나리오:
- 5/5 일치 → 정상
- 4/5 일치 (1개 다름) → 정상
- 3/5 일치 (2-2-1 또는 3-1-1) → 정상
- 2/5 (2-2-1, 2-1-1-1) → 의심
- 1/5 (1-1-1-1-1) → 의심

주의: 구현은 morpheme을 [list, list, ...] 형태(JSON-friendly)로 반환한다.
테스트도 list-of-lists로 비교한다 (option b).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eojeol_voter import vote_eojeol, EojeolVoteResult


def _as_lists(seq):
    """tuple-of-tuples → list-of-lists 정규화 (JSON-friendly 비교용)."""
    return [list(t) for t in seq]


def test_full_agreement_5_of_5():
    """5분석기 모두 같은 morpheme 시퀀스 → 정상, agree=5."""
    seq = [("정부", "NNG"), ("가", "JKS")]
    analyses = {
        "mecab": seq, "kkma": seq, "komoran": seq, "kiwi": seq, "garu": seq,
    }
    r = vote_eojeol("정부가", analyses)
    assert r.status == "normal"
    assert r.agree == 5
    assert r.morphemes == _as_lists(seq)


def test_partial_agreement_3_of_5():
    """3분석기 같은 시퀀스, 2개 다름 → 정상, agree=3."""
    seq_a = [("내년", "NNG")]
    seq_b = [("내년", "MAG")]
    analyses = {
        "mecab": seq_a, "kkma": seq_a, "komoran": seq_a,
        "kiwi": seq_b, "garu": seq_b,
    }
    r = vote_eojeol("내년", analyses)
    assert r.status == "normal"
    assert r.agree == 3
    assert r.morphemes == _as_lists(seq_a)


def test_tie_2_2_1_suspicious():
    """2-2-1 동률 → 의심."""
    seq_a = [("어절", "NNG")]
    seq_b = [("어절", "NNP")]
    seq_c = [("어절", "VV")]
    analyses = {
        "mecab": seq_a, "kkma": seq_a,
        "komoran": seq_b, "kiwi": seq_b,
        "garu": seq_c,
    }
    r = vote_eojeol("어절", analyses)
    assert r.status == "suspicious"
    assert r.agree == 2
    assert r.morphemes is None
    # 분석기 그룹별로 묶여 있어야 함
    groups = {tuple(tuple(m) for m in cand["morphemes"]): set(cand["analyzers"])
              for cand in r.candidates}
    assert groups[tuple(tuple(m) for m in seq_a)] == {"mecab", "kkma"}
    assert groups[tuple(tuple(m) for m in seq_b)] == {"komoran", "kiwi"}
    assert groups[tuple(tuple(m) for m in seq_c)] == {"garu"}


def test_all_different_1_1_1_1_1():
    """모두 다른 답 → 의심, agree=1."""
    analyses = {
        "mecab":   [("a", "NNG")],
        "kkma":    [("b", "NNG")],
        "komoran": [("c", "NNG")],
        "kiwi":    [("d", "NNG")],
        "garu":    [("e", "NNG")],
    }
    r = vote_eojeol("xxx", analyses)
    assert r.status == "suspicious"
    assert r.agree == 1
    assert len(r.candidates) == 5


def test_empty_output_treated_as_distinct():
    """빈 출력은 별도 후보로 취급 (다른 분석기와 합쳐지지 않음)."""
    seq = [("정상", "NNG")]
    analyses = {
        "mecab": seq, "kkma": seq, "komoran": seq, "kiwi": seq,
        "garu": [],  # 빈 출력
    }
    r = vote_eojeol("정상", analyses)
    assert r.status == "normal"
    assert r.agree == 4


def test_surface_pos_both_must_match():
    """surface 같지만 POS 다르면 다른 후보."""
    analyses = {
        "mecab":   [("하", "VV")],
        "kkma":    [("하", "VV")],
        "komoran": [("하", "XSV")],
        "kiwi":    [("하", "XSV")],
        "garu":    [("하", "VX")],
    }
    r = vote_eojeol("하", analyses)
    # 가장 큰 그룹은 VV(2) 또는 XSV(2). 어느 쪽이든 2 < 3 → 의심.
    assert r.status == "suspicious"
    assert r.agree == 2
