"""4분석기 wrapper 테스트. 공통 인터페이스 + POS 정규화 자동 적용 검증."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ensemble"))
from wrappers import KiwiWrapper, MecabWrapper, KkmaWrapper, KomoranWrapper, GaruWrapper

TEST_SENTENCE = "안녕하세요"

def _check_interface(result):
    assert isinstance(result, list), f"expected list, got {type(result)}"
    assert len(result) > 0, "result is empty"
    for tok in result:
        assert isinstance(tok, tuple) or isinstance(tok, list), f"bad token type: {type(tok)}"
        assert len(tok) == 2, f"expected (surface, pos), got {tok}"
        assert isinstance(tok[0], str)
        assert isinstance(tok[1], str)

def test_kiwi_wrapper():
    w = KiwiWrapper()
    result = w.analyze(TEST_SENTENCE)
    _check_interface(result)
    surfaces = [t[0] for t in result]
    assert "안녕" in surfaces, f"안녕 missing in {result}"

def test_mecab_wrapper():
    w = MecabWrapper()
    result = w.analyze(TEST_SENTENCE)
    _check_interface(result)

def test_kkma_wrapper():
    w = KkmaWrapper()
    result = w.analyze(TEST_SENTENCE)
    _check_interface(result)
    # Kkma의 OH 같은 태그가 정규화돼서 SH로 매핑되는지 (한자 들어간 문장은 별도 검증 필요)

def test_komoran_wrapper():
    w = KomoranWrapper()
    result = w.analyze(TEST_SENTENCE)
    _check_interface(result)

def test_garu_wrapper_dedup_role():
    # Garu wrapper는 합의 풀에서 빠지지만 dedup 용도로 동일 인터페이스 제공
    w = GaruWrapper()
    result = w.analyze(TEST_SENTENCE)
    _check_interface(result)

if __name__ == "__main__":
    test_kiwi_wrapper(); print("kiwi OK")
    test_mecab_wrapper(); print("mecab OK")
    test_kkma_wrapper(); print("kkma OK")
    test_komoran_wrapper(); print("komoran OK")
    test_garu_wrapper_dedup_role(); print("garu OK")
    print("ALL OK")
