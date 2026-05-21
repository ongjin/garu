"""eval_f1.compute_f1이 ep_norm을 적용하면 ㄴ/ᆫ 동치를 인식하는지 검증."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "expand"))

from ep_norm import normalize_ep_morphemes


def test_jamo_compat_vs_combining_match_after_norm():
    pred = [[("크", "VA"), ("ㄴ", "ETM"), ("것", "NNB")]]
    gold = [[("크", "VA"), ("ᆫ", "ETM"), ("것", "NNB")]]
    pred_norm = [normalize_ep_morphemes(s) for s in pred]
    gold_norm = [normalize_ep_morphemes(s) for s in gold]
    assert pred_norm == gold_norm, f"pred={pred_norm} gold={gold_norm}"


def test_vowel_harmony_match_after_norm():
    pred = [[("통하", "VV"), ("아", "EC")]]
    gold = [[("통하", "VV"), ("어", "EC")]]
    pred_norm = [normalize_ep_morphemes(s) for s in pred]
    gold_norm = [normalize_ep_morphemes(s) for s in gold]
    assert pred_norm == gold_norm


def test_ep_contract_match_after_norm():
    pred = [[("하", "XSV"), ("았", "EP"), ("다", "EF")]]
    gold = [[("했", "XSV+EP"), ("다", "EF")]]
    pred_norm = [normalize_ep_morphemes(s) for s in pred]
    gold_norm = [normalize_ep_morphemes(s) for s in gold]
    assert pred_norm == gold_norm


def test_raw_mismatch_without_norm():
    # ep_norm 호출 없으면 ㄴ ↔ ᆫ 불일치
    pred = [("크", "VA"), ("ㄴ", "ETM")]
    gold = [("크", "VA"), ("ᆫ", "ETM")]
    assert pred != gold  # raw level mismatch


if __name__ == "__main__":
    test_jamo_compat_vs_combining_match_after_norm()
    test_vowel_harmony_match_after_norm()
    test_ep_contract_match_after_norm()
    test_raw_mismatch_without_norm()
    print("OK")
