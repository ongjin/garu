"""JVM 기반 분석기 (Kkma, Komoran) 구동 회귀 테스트.

ARM Mac + macOS 15 (Darwin 25.x) 환경에서 Xcode 번들 Python (/usr/bin/python3)
은 `com.apple.security.cs.allow-jit` entitlement 가 없어 JPype 가 JVM 을 임베드
할 때 SIGBUS (BUS_ADRALN, CodeHeap::allocate) 로 크래시한다 (JDK 17/21/25 모두).
Plan A Task 1 (2026-05-21) 진단으로 Homebrew Python (entitlement 미적용)
사용으로 우회.

이 테스트는 환경 변수 GARU_ENSEMBLE_PYTHON (또는 fallback 으로 Homebrew
python3.14, 그래도 없으면 sys.executable) 에서 Kkma/Komoran 인스턴스화가
정상 종료되는지 검증한다.
"""
import os
import subprocess
import sys


def _ensemble_python() -> str:
    """크래시 없이 JPype + JVM 을 임베드할 수 있는 Python 실행 파일."""
    candidates = [
        os.environ.get("GARU_ENSEMBLE_PYTHON"),
        "/opt/homebrew/bin/python3.14",
        "/opt/homebrew/bin/python3",
        sys.executable,
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return sys.executable


PYTHON = _ensemble_python()


def test_kkma_basic():
    out = subprocess.run(
        [PYTHON, "-c",
         "from konlpy.tag import Kkma; k = Kkma(); print(k.pos('안녕하세요'))"],
        capture_output=True, text=True, timeout=60,
    )
    assert out.returncode == 0, f"Kkma crashed: stderr={out.stderr[:500]}"
    assert "안녕" in out.stdout, f"unexpected output: {out.stdout}"


def test_komoran_basic():
    out = subprocess.run(
        [PYTHON, "-c",
         "from konlpy.tag import Komoran; k = Komoran(); print(k.pos('안녕하세요'))"],
        capture_output=True, text=True, timeout=60,
    )
    assert out.returncode == 0, f"Komoran crashed: stderr={out.stderr[:500]}"
    assert "안녕" in out.stdout, f"unexpected output: {out.stdout}"


if __name__ == "__main__":
    print(f"using python: {PYTHON}")
    test_kkma_basic()
    test_komoran_basic()
    print("OK")
