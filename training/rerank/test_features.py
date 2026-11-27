"""어절 경계 feature: crates/garu-core/src/rerank.rs의 Section 14 ver 3과 같은 문자열."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import extract  # noqa: E402

SPACED = [["활성", "NNG", 0, 2], ["화", "XSN", 3, 6], ["학제", "NNG", 3, 6]]
JOINED = [["활성", "NNG", 0, 5], ["화", "XSN", 0, 5], ["학제", "NNG", 0, 5]]
SPACING = {"sb:NNG|XSN", "spw:활성|XSN", "sw:화/XSN"}


def test_spacing_features_at_eojeol_boundary():
    spaced = extract(SPACED, 0.0, 1)
    assert SPACING <= spaced.keys()
    assert "pw:활성|XSN" in spaced
    assert not SPACING & extract(JOINED, 0.0, 1).keys()
