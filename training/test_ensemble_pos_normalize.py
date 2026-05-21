"""POS 정규화 테스트. 5분석기의 POS schema 차이를 세종 42태그로 통일."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ensemble"))
from pos_normalize import normalize_pos, split_compound_pos

def test_kkma_hanja_to_sejong():
    # Kkma의 OH(한자), OL(영문) → 세종 SH, SL
    assert normalize_pos("OH", source="kkma") == "SH"
    assert normalize_pos("OL", source="kkma") == "SL"

def test_kkma_nnm_to_nnb():
    # Kkma NNM(단위명사) → 세종 NNB(의존명사)
    assert normalize_pos("NNM", source="kkma") == "NNB"

def test_passthrough_for_sejong_native():
    # 세종 표준 태그는 그대로
    for tag in ["NNG", "NNP", "VV", "VA", "MAG", "MM", "JKS"]:
        assert normalize_pos(tag, source="kiwi") == tag
        assert normalize_pos(tag, source="mecab") == tag

def test_mecab_compound_split():
    # Mecab "EP+EF" 같은 컴파운드 → 첫 태그만 (또는 분리)
    parts = split_compound_pos("EP+EF")
    assert parts == ["EP", "EF"]
    # 단일 태그는 길이 1 리스트
    assert split_compound_pos("NNG") == ["NNG"]

def test_unknown_tag_returns_itself():
    # 매핑 없는 태그는 그대로 통과 (caller가 후속 처리)
    assert normalize_pos("ZZ", source="kkma") == "ZZ"

if __name__ == "__main__":
    test_kkma_hanja_to_sejong()
    test_kkma_nnm_to_nnb()
    test_passthrough_for_sejong_native()
    test_mecab_compound_split()
    test_unknown_tag_returns_itself()
    print("OK")
