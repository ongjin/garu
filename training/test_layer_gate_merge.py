"""content_dict.txt 머지 테스트."""
import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "layer_gates"))
from merge import append_entries_to_dict, parse_dict_line

CURRENT_DICT = """""	NNP	4
%포인트	NNG	663
가나	NNG	120
"""

CANDIDATES = [
    {"surface": "러닝메이트", "normalized_pos": "NNG", "frequency": 50},
    {"surface": "쿠버네티스", "normalized_pos": "NNP", "frequency": 30},
]


def test_parse_dict_line():
    assert parse_dict_line("가나\tNNG\t120") == ("가나", "NNG", 120)
    assert parse_dict_line('""\tNNP\t4') == ('""', "NNP", 4)


def test_append_skips_duplicates():
    with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False) as f:
        f.write(CURRENT_DICT)
        path = f.name
    try:
        cands = CANDIDATES + [{"surface": "가나", "normalized_pos": "NNG", "frequency": 999}]
        added = append_entries_to_dict(cands, path)
        assert added == 2
        content = open(path).read()
        assert "러닝메이트\tNNG\t50" in content
        assert "쿠버네티스\tNNP\t30" in content
        assert content.count("가나") == 1
    finally:
        os.unlink(path)


def test_append_preserves_existing():
    with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False) as f:
        f.write(CURRENT_DICT)
        path = f.name
    try:
        append_entries_to_dict(CANDIDATES, path)
        content = open(path).read()
        assert "%포인트\tNNG\t663" in content
        assert '""\tNNP\t4' in content
    finally:
        os.unlink(path)


if __name__ == "__main__":
    test_parse_dict_line(); print("parse OK")
    test_append_skips_duplicates(); print("skip_dup OK")
    test_append_preserves_existing(); print("preserve OK")
    print("MERGE OK")
