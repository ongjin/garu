"""sejong_normalize 단위 테스트."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sejong_normalize import unify_ep


def test_ep_after_ha_becomes_yeot():
    # 하 + EP → 였 (았/었 모두 였로)
    assert unify_ep([["하","XSV"],["았","EP"],["다","EF"]]) == [["하","XSV"],["였","EP"],["다","EF"]]
    assert unify_ep([["하","XSV"],["었","EP"],["다","EF"]]) == [["하","XSV"],["였","EP"],["다","EF"]]


def test_ep_positive_vowel_becomes_at():
    # 보(ㅗ 양성) + EP → 았
    assert unify_ep([["보","VV"],["었","EP"],["다","EF"]]) == [["보","VV"],["았","EP"],["다","EF"]]
    # 가(ㅏ 양성) + EP → 았
    assert unify_ep([["가","VV"],["었","EP"],["다","EF"]]) == [["가","VV"],["았","EP"],["다","EF"]]


def test_ep_negative_vowel_becomes_eot():
    # 먹(ㅓ 음성) + EP → 었
    assert unify_ep([["먹","VV"],["았","EP"],["다","EF"]]) == [["먹","VV"],["었","EP"],["다","EF"]]
    # 주(ㅜ 음성) + EP → 었
    assert unify_ep([["주","VV"],["았","EP"],["다","EF"]]) == [["주","VV"],["었","EP"],["다","EF"]]


def test_ep_no_change_when_already_correct():
    assert unify_ep([["가","VV"],["았","EP"],["다","EF"]]) == [["가","VV"],["았","EP"],["다","EF"]]


def test_ep_first_morpheme_no_stem():
    # EP가 첫 형태소면 어간 없음 → 변경 안 함 (방어)
    assert unify_ep([["았","EP"],["다","EF"]]) == [["았","EP"],["다","EF"]]


def test_non_ep_untouched():
    assert unify_ep([["정부","NNG"],["가","JKS"]]) == [["정부","NNG"],["가","JKS"]]


def test_returns_new_list_not_mutate():
    src = [["하","XSV"],["았","EP"]]
    out = unify_ep(src)
    assert src == [["하","XSV"],["았","EP"]]  # 원본 불변


def test_ep_positive_compound_vowel_becomes_at():
    # 봐(ㅘ = ㅗ+ㅏ 양성 복합모음) + EP → 았 (봤다 = 봐+았+다)
    assert unify_ep([["봐","VV"],["었","EP"],["다","EF"]]) == [["봐","VV"],["았","EP"],["다","EF"]]


from sejong_normalize import is_merge_of, candidates_converge


def test_is_merge_of_jongseong_attach():
    # fine=[하/VV, ㄴ다/EF] 의 마지막 둘 결합 → "한다" == coarse 마지막
    fine = [["하","VV"],["ㄴ다","EF"]]
    coarse = [["한다","VV"]]
    assert is_merge_of(fine, coarse) is True


def test_is_merge_of_with_prefix():
    # 앞부분 동일 + 마지막 결합. 말+하+ㄴ다 vs 말+한다
    fine = [["말","NNG"],["하","VV"],["ㄴ다","EF"]]
    coarse = [["말","NNG"],["한다","VV"]]
    assert is_merge_of(fine, coarse) is True


def test_is_merge_of_rieul_etm():
    # 하 + ㄹ(ETM) → 할
    fine = [["하","VV"],["ㄹ","ETM"]]
    coarse = [["할","VV"]]
    assert is_merge_of(fine, coarse) is True


def test_is_merge_of_false_when_vowel_contraction():
    # 가+았 → 갔 은 받침결합 아님(모음축약) → False (Claude로 폴백)
    fine = [["가","VV"],["았","EP"]]
    coarse = [["갔","VV"]]
    assert is_merge_of(fine, coarse) is False


def test_is_merge_of_false_when_prefix_differs():
    fine = [["먹","VV"],["ㄴ다","EF"]]
    coarse = [["한다","VV"]]
    assert is_merge_of(fine, coarse) is False


def test_converge_ep_only():
    # 했다: EP통일 후 모두 하+였+다 로 수렴
    cands = [
        {"analyzers": ["kiwi"], "morphemes": [["하","XSV"],["었","EP"],["다","EF"]]},
        {"analyzers": ["garu"], "morphemes": [["하","XSV"],["았","EP"],["다","EF"]]},
    ]
    assert candidates_converge(cands) == [["하","XSV"],["였","EP"],["다","EF"]]


def test_converge_granularity():
    # 한다: 입도 차이 → 최대분해(하+ㄴ다)로 수렴
    cands = [
        {"analyzers": ["kiwi"], "morphemes": [["하","VV"],["ㄴ다","EF"]]},
        {"analyzers": ["mecab"], "morphemes": [["한다","VV"]]},
    ]
    assert candidates_converge(cands) == [["하","VV"],["ㄴ다","EF"]]


def test_converge_none_on_pos_conflict():
    # 그/MM vs 그/VA: 정규화로 안 모임 → None (Claude)
    cands = [
        {"analyzers": ["kiwi","kkma"], "morphemes": [["그","MM"]]},
        {"analyzers": ["garu"], "morphemes": [["그","VA"]]},
    ]
    assert candidates_converge(cands) is None


def test_converge_none_on_vowel_contraction():
    # 갔다 vs 가+았+다 (+EF/EC 어미 충돌까지) → None
    cands = [
        {"analyzers": ["kiwi"], "morphemes": [["가","VV"],["았","EP"],["다","EF"]]},
        {"analyzers": ["mecab"], "morphemes": [["갔","VV"],["다","EC"]]},
    ]
    assert candidates_converge(cands) is None
