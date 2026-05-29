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
