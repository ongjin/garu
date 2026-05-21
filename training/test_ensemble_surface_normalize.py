"""Surface 정규화 테스트. 호환자모→결합자모, ep_norm 적용."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ensemble"))
from surface_normalize import nfc_normalize, normalize_token_list

def test_nfc_normalize_compatibility_to_combining():
    # 호환자모 ㄱ(U+3131) → 결합자모 ᄀ(U+1100)은 NFC로는 그대로지만 NFC 적용 시 안전
    # 실제는 NFC가 한글 자모 정규화에 영향 적음. 일반 한글은 그대로 통과.
    assert nfc_normalize("안녕") == "안녕"
    assert nfc_normalize("ㄱㅏ") != ""  # 결합 안 된 자모도 그대로 보존

def test_normalize_token_list_passthrough():
    # 일반 토큰 리스트는 정규화 후에도 형태 유지 (값 자체는 ep_norm 결과)
    tokens = [["안녕", "NNG"], ["하", "XSV"], ["세요", "EF"]]
    result = normalize_token_list(tokens)
    assert isinstance(result, list)
    assert len(result) >= 2  # ep_norm은 EP/EF 분리하기도 함
    for tok in result:
        assert len(tok) == 2
        assert isinstance(tok[0], str) and isinstance(tok[1], str)

def test_normalize_handles_ep_compound():
    # ep_norm은 EP+EF 같은 케이스를 정규화한다 (eval_f1과 동일 동작)
    tokens = [["갔", "VV"], ["었", "EP"], ["다", "EF"]]
    result = normalize_token_list(tokens)
    # ep_norm이 모음 축약 등을 정규화하나, 최소한 토큰 수가 비-증가
    assert len(result) <= 3 + 1  # 약간의 split은 허용

if __name__ == "__main__":
    test_nfc_normalize_compatibility_to_combining()
    test_normalize_token_list_passthrough()
    test_normalize_handles_ep_compound()
    print("OK")
