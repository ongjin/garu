"""layer filter 테스트."""
import sys, os, json, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "layer_gates"))
from filter import LayerConfig, filter_candidates


def _make_record(surface="러닝메이트", pos="NNG",
                 votes=None, in_garu_dict=False,
                 frequency=10, source_domains=("news",)):
    if votes is None:
        votes = {"kiwi": 1, "mecab": 1, "kkma": 1, "komoran": 1}
    return {
        "surface": surface, "normalized_pos": pos, "votes": votes,
        "in_garu_dict": in_garu_dict, "frequency": frequency,
        "source_domains": list(source_domains),
    }


def test_excludes_in_garu_dict():
    records = [_make_record(in_garu_dict=True)]
    cfg = LayerConfig(name="t", required_pos={"NNG"}, min_votes=3, min_freq=5)
    assert filter_candidates(records, cfg) == []


def test_excludes_below_min_votes():
    records = [_make_record(votes={"kiwi": 1, "mecab": 1, "kkma": 0, "komoran": 0})]
    cfg = LayerConfig(name="t", required_pos={"NNG"}, min_votes=3, min_freq=5)
    assert filter_candidates(records, cfg) == []


def test_excludes_below_min_freq():
    records = [_make_record(frequency=3)]
    cfg = LayerConfig(name="t", required_pos={"NNG"}, min_votes=3, min_freq=5)
    assert filter_candidates(records, cfg) == []


def test_excludes_wrong_pos():
    records = [_make_record(pos="VV")]
    cfg = LayerConfig(name="t", required_pos={"NNG"}, min_votes=3, min_freq=5)
    assert filter_candidates(records, cfg) == []


def test_excludes_wrong_domain():
    records = [_make_record(source_domains=("blog",))]
    cfg = LayerConfig(name="t", required_pos={"NNG"}, min_votes=3, min_freq=5,
                      allowed_domains={"news"})
    assert filter_candidates(records, cfg) == []


def test_passes_qualifying_candidate():
    records = [_make_record(frequency=10, votes={"kiwi":1,"mecab":1,"kkma":1,"komoran":0})]
    cfg = LayerConfig(name="t", required_pos={"NNG"}, min_votes=3, min_freq=5)
    out = filter_candidates(records, cfg)
    assert len(out) == 1
    assert out[0]["surface"] == "러닝메이트"


def test_min_surface_length():
    records = [_make_record(surface="게"), _make_record(surface="러닝메이트")]
    cfg = LayerConfig(name="t", required_pos={"NNG"}, min_votes=3, min_freq=5,
                      min_surface_len=2)
    out = filter_candidates(records, cfg)
    assert len(out) == 1
    assert out[0]["surface"] == "러닝메이트"


def test_stoplist():
    records = [_make_record(surface="기자"), _make_record(surface="러닝메이트")]
    cfg = LayerConfig(name="t", required_pos={"NNG"}, min_votes=3, min_freq=5,
                      surface_stoplist={"기자", "오늘"})
    out = filter_candidates(records, cfg)
    assert len(out) == 1
    assert out[0]["surface"] == "러닝메이트"


if __name__ == "__main__":
    test_excludes_in_garu_dict(); print("in_garu OK")
    test_excludes_below_min_votes(); print("min_votes OK")
    test_excludes_below_min_freq(); print("min_freq OK")
    test_excludes_wrong_pos(); print("pos OK")
    test_excludes_wrong_domain(); print("domain OK")
    test_passes_qualifying_candidate(); print("pass OK")
    test_min_surface_length(); print("len OK")
    test_stoplist(); print("stoplist OK")
    print("FILTER OK")
