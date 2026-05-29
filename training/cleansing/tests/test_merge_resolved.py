"""merge_resolved smoke 테스트."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from merge_resolved import merge_still_sentence, normalize_morphemes


def test_merge_still_sentence():
    # 문장: eojeol_votes 1개(정상) + suspicious 2개 (1개 rule-resolved, 1개 claude)
    sent = {
        "text": "그는 그 한다", "domain": "x", "vote_status": "suspicious",
        "eojeol_votes": [{"surface":"그는","agree":5,"morphemes":[["그","NP"],["는","JX"]]}],
        "suspicious_eojeols": [
            {"index":1,"surface":"그","agree":2,
             "candidates":[{"analyzers":["kiwi"],"morphemes":[["그","MM"]]}]},  # claude 대상
            {"index":2,"surface":"한다","agree":2,
             "candidates":[{"analyzers":["mecab"],"morphemes":[["한다","VV"]]}],
             "resolved":[["하","VV"],["ㄴ다","EF"]]},  # rule resolved
        ],
    }
    decisions = {"0_1": [["그","MM"]]}
    out = merge_still_sentence(sent, 0, decisions)
    assert out["vote_status"] == "normal"
    # 어절 순서: 그는(0) 그(1) 한다(2)
    assert out["morphemes"] == [["그","NP"],["는","JX"],["그","MM"],["하","VV"],["ㄴ다","EF"]]


def test_normalize_jamo_combining_to_compat():
    # 결합 자모(ᆫ U+11AB) → 호환 자모(ㄴ U+3134)
    assert normalize_morphemes([["오","VV"],["ᆫ다","EF"]]) == [["오","VV"],["ㄴ다","EF"]]
    assert normalize_morphemes([["사","VV"],["ᆸ니다","EF"]]) == [["사","VV"],["ㅂ니다","EF"]]


def test_normalize_ha_ep_to_yeot():
    # 하 + EP(았/었/였) → 였
    assert normalize_morphemes([["하","XSV"],["었","EP"],["다","EF"]]) == [["하","XSV"],["였","EP"],["다","EF"]]
    assert normalize_morphemes([["하","XSV"],["았","EP"],["다","EF"]]) == [["하","XSV"],["였","EP"],["다","EF"]]


def test_normalize_ha_ec_to_yeo():
    # 하 + EC(아/어/여) → 여
    assert normalize_morphemes([["하","XSV"],["어","EC"]]) == [["하","XSV"],["여","EC"]]


def test_normalize_non_ha_ep_untouched():
    # 하가 아닌 어간 뒤 EP는 건드리지 않음 (오르+았 그대로)
    assert normalize_morphemes([["오르","VV"],["았","EP"],["다","EF"]]) == [["오르","VV"],["았","EP"],["다","EF"]]
    # 일반 형태소 불변
    assert normalize_morphemes([["정부","NNG"],["가","JKS"]]) == [["정부","NNG"],["가","JKS"]]
