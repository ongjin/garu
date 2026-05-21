"""Agreement counter 테스트.

스펙 섹션 2.5: 동일 surface에 다른 POS 분석되면 별도 후보로 분리.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ensemble"))
from agreement import compute_votes

def test_full_agreement_4_of_4():
    # 4분석기 모두 (러닝메이트, NNG) 단일 형태소로 인식
    analyses = {
        "kiwi":    [("러닝메이트", "NNG")],
        "mecab":   [("러닝메이트", "NNG")],
        "kkma":    [("러닝메이트", "NNG")],
        "komoran": [("러닝메이트", "NNG")],
    }
    garu_analysis = [("러닝", "NNG"), ("메이트", "NNG")]  # Garu는 분할
    votes = compute_votes(analyses, garu_analysis=garu_analysis)
    key = ("러닝메이트", "NNG")
    assert key in votes
    assert votes[key]["votes"] == {"kiwi": 1, "mecab": 1, "kkma": 1, "komoran": 1}
    assert votes[key]["in_garu_dict"] is False  # Garu가 단일로 인식 안 했으므로

def test_partial_pos_disagreement_splits():
    # Kiwi/Mecab: NNG, Komoran: NNP, Kkma: NNG → 2개 후보로 분리
    analyses = {
        "kiwi":    [("리듬게임", "NNG")],
        "mecab":   [("리듬게임", "NNG")],
        "kkma":    [("리듬게임", "NNG")],
        "komoran": [("리듬게임", "NNP")],
    }
    votes = compute_votes(analyses, garu_analysis=[])
    nng_key = ("리듬게임", "NNG")
    nnp_key = ("리듬게임", "NNP")
    assert nng_key in votes and nnp_key in votes
    assert votes[nng_key]["votes"] == {"kiwi": 1, "mecab": 1, "kkma": 1, "komoran": 0}
    assert votes[nnp_key]["votes"] == {"kiwi": 0, "mecab": 0, "kkma": 0, "komoran": 1}

def test_split_analysis_no_vote():
    # 분석기가 분할하면 vote=0 (단일 형태소로 인식 안 함)
    analyses = {
        "kiwi":    [("데이터베이스", "NNG")],
        "mecab":   [("데이터", "NNG"), ("베이스", "NNG")],
        "kkma":    [("데이터베이스", "NNG")],
        "komoran": [("데이터", "NNG"), ("베이스", "NNG")],
    }
    votes = compute_votes(analyses, garu_analysis=[])
    key = ("데이터베이스", "NNG")
    assert votes[key]["votes"] == {"kiwi": 1, "mecab": 0, "kkma": 1, "komoran": 0}

def test_garu_dedup_flag_when_single():
    # Garu가 단일 형태소로 인식 → in_garu_dict=True (이미 dict 보유 신호)
    analyses = {
        "kiwi":    [("러닝메이트", "NNG")],
        "mecab":   [("러닝메이트", "NNG")],
        "kkma":    [("러닝메이트", "NNG")],
        "komoran": [("러닝메이트", "NNG")],
    }
    garu_analysis = [("러닝메이트", "NNG")]
    votes = compute_votes(analyses, garu_analysis=garu_analysis)
    assert votes[("러닝메이트", "NNG")]["in_garu_dict"] is True

def test_total_votes_helper():
    from agreement import total_votes
    info = {"votes": {"kiwi": 1, "mecab": 1, "kkma": 0, "komoran": 1}}
    assert total_votes(info) == 3

if __name__ == "__main__":
    test_full_agreement_4_of_4(); print("4/4 OK")
    test_partial_pos_disagreement_splits(); print("POS split OK")
    test_split_analysis_no_vote(); print("morph split no-vote OK")
    test_garu_dedup_flag_when_single(); print("garu dedup OK")
    test_total_votes_helper(); print("total OK")
    print("ALL OK")
