"""Naver API 자격증명 환경변수 존재 확인. 실제 API 호출은 안 함."""
import os

def test_naver_client_id_set():
    cid = os.environ.get("NAVER_CLIENT_ID")
    assert cid, "NAVER_CLIENT_ID not set in environment"
    assert len(cid) >= 10, f"NAVER_CLIENT_ID too short: {len(cid)}"

def test_naver_client_secret_set():
    cs = os.environ.get("NAVER_CLIENT_SECRET")
    assert cs, "NAVER_CLIENT_SECRET not set in environment"
    assert len(cs) >= 5, f"NAVER_CLIENT_SECRET too short: {len(cs)}"

if __name__ == "__main__":
    test_naver_client_id_set(); print("client_id OK")
    test_naver_client_secret_set(); print("client_secret OK")
    print("AUTH OK")
