"""reverse_2025_to_2021.reverse_convert 단위테스트.

2025 거친 골드를 2021/Sejong 입도로 내리는 역변환. 각 규칙(R0 un-join, R1 하/되/시키/받
분리·어휘화 유지, R3 인용 clitic 재병합)과 의도적 제외(적/성/화 XSN, 동사 걸친 _ 복합어)를
검증. 마지막으로 normalize_for_2025(reverse(g)) == normalize_for_2025(g) 역함수 성질.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from reverse_2025_to_2021 import reverse_convert as R
from eval_nikl_mp import normalize_for_2025


# ── R0) '_' 결합 복합명사 un-join ─────────────────────────────────────────
def test_unjoin_nominal():
    assert R([["사업_분야", "NNG"], ["를", "JKO"]]) == \
        [["사업", "NNG"], ["분야", "NNG"], ["를", "JKO"]]

def test_unjoin_three_parts():
    assert R([["학창_시절", "NNG"], ["에", "JKB"]]) == \
        [["학창", "NNG"], ["시절", "NNG"], ["에", "JKB"]]

def test_unjoin_excludes_verb_compound():
    # 동사 걸친 '_' 복합어는 분해하지 않고 '_'만 제거(제외 규칙): 음주/VV 오태깅 방지.
    assert R([["음주_운전하", "VV"], ["면", "EC"]]) == [["음주운전하", "VV"], ["면", "EC"]]


# ── R1) 명사 + 하/되/시키/당하/받 병합용언 분리 ─────────────────────────────
def test_ha_split_vv():
    assert R([["방문하", "VV"], ["ㄴ다", "EF"]]) == \
        [["방문", "NNG"], ["하", "XSV"], ["ㄴ다", "EF"]]

def test_ha_split_va():
    assert R([["건강하", "VA"], ["게", "EC"]]) == \
        [["건강", "NNG"], ["하", "XSA"], ["게", "EC"]]

def test_doe_split():
    assert R([["공개되", "VV"], ["었", "EP"], ["고", "EC"]]) == \
        [["공개", "NNG"], ["되", "XSV"], ["었", "EP"], ["고", "EC"]]

def test_siki_split():
    assert R([["공부시키", "VV"], ["었", "EP"]]) == \
        [["공부", "NNG"], ["시키", "XSV"], ["었", "EP"]]


# ── R1) 어휘화 용언은 병합 유지 (keep_merged) ──────────────────────────────
def test_keep_lexicalized_daeha():
    assert R([["대하", "VV"], ["ㄴ", "ETM"]]) == [["대하", "VV"], ["ㄴ", "ETM"]]

def test_keep_lexicalized_wiha():
    assert R([["위하", "VV"], ["아서", "EC"]]) == [["위하", "VV"], ["아서", "EC"]]

def test_keep_lexicalized_joaha():
    assert R([["좋아하", "VV"], ["는", "ETM"]]) == [["좋아하", "VV"], ["는", "ETM"]]

def test_keep_native_verb():
    # 만들/모르 등 자·복합용언은 하/되로 안 끝나거나 어기가 명사 아님 → 유지.
    assert R([["만들", "VV"], ["ㄴ", "ETM"]]) == [["만들", "VV"], ["ㄴ", "ETM"]]
    assert R([["모르", "VV"], ["는", "ETM"]]) == [["모르", "VV"], ["는", "ETM"]]


# ── 제외) 적/성/화 파생접미사(XSN) 분리는 하지 않음 ────────────────────────
def test_xsn_not_split():
    assert R([["사회적", "NNG"], ["이", "VCP"]]) == [["사회적", "NNG"], ["이", "VCP"]]
    assert R([["가능성", "NNG"], ["이", "JKS"]]) == [["가능성", "NNG"], ["이", "JKS"]]


# ── R3) 간접인용 clitic 재병합 ─────────────────────────────────────────────
def test_quote_merge_go():
    assert R([["가", "VV"], ["ㄴ다", "EF"], ["고", "JKQ"]]) == [["가", "VV"], ["ㄴ다고", "EC"]]

def test_quote_merge_myeo():
    assert R([["가", "VV"], ["ㄴ다", "EF"], ["며", "EC"]]) == [["가", "VV"], ["ㄴ다며", "EC"]]

def test_quote_merge_neun():
    assert R([["가", "VV"], ["ㄴ다", "EF"], ["는", "ETM"]]) == [["가", "VV"], ["ㄴ다는", "ETM"]]

def test_quote_jkq_rago():
    assert R([["직업", "NNG"], ["이", "VCP"], ["라", "EF"], ["고", "JKQ"]]) == \
        [["직업", "NNG"], ["이", "VCP"], ["라고", "EC"]]


# ── 역함수 성질: normalize_for_2025(reverse(g)) == normalize_for_2025(g) ────
def test_invertibility_samples():
    samples = [
        [["생각하", "VV"], ["았", "EP"], ["다", "EF"]],
        [["공개되", "VV"], ["었", "EP"], ["고", "EC"]],
        [["사업_분야", "NNG"], ["를", "JKO"]],
        [["가", "VV"], ["ㄴ다", "EF"], ["고", "JKQ"]],
        [["사회적", "NNG"], ["이", "VCP"]],
        [["대하", "VV"], ["ㄴ", "ETM"]],
    ]
    for g in samples:
        assert {(f, p) for f, p in normalize_for_2025(R(g))} == \
               {(f, p) for f, p in normalize_for_2025(g)}, g


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
