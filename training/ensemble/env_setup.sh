# Ensemble (Kiwi + Mecab + Kkma + Komoran + Garu) 실행 환경 설정.
#
# 사용법:
#   source training/ensemble/env_setup.sh
#   python3 training/test_ensemble_jvm.py
#
# 배경 (2026-05-21 진단):
#   ARM Mac + macOS 15 (Darwin 25.x)에서 Xcode 번들 /usr/bin/python3 는
#   `com.apple.security.cs.allow-jit` entitlement 가 없어 JPype 가 JVM 을
#   임베드할 때 SIGBUS (BUS_ADRALN, CodeHeap::allocate) 로 크래시. JDK 17/21/25
#   전부 동일 — JDK 문제 아님. Homebrew Python (entitlement 없음 = 제약 없음)
#   에서는 정상 동작.
#
#   • 작동: /opt/homebrew/bin/python3.14
#   • 크래시: /usr/bin/python3 (Xcode), /Applications/Xcode.app/.../python3.9

# Homebrew Python 강제 사용
export GARU_ENSEMBLE_PYTHON="/opt/homebrew/bin/python3.14"

# JAVA_HOME 은 KoNLPy 가 자체 탐색하므로 비워둬도 무방하나, 명시 시 21 권장
# (17/21/25 모두 Homebrew Python 위에서 동작 확인. 21 LTS 가 무난.)
if [ -z "$JAVA_HOME" ]; then
  _JH=$(/usr/libexec/java_home -v 21 2>/dev/null)
  if [ -n "$_JH" ]; then
    export JAVA_HOME="$_JH"
  fi
fi

echo "GARU_ENSEMBLE_PYTHON=$GARU_ENSEMBLE_PYTHON"
echo "JAVA_HOME=$JAVA_HOME"

# ---- Plan B: Naver Search API 자격증명 ----
# 사용자가 https://developers.naver.com/apps 에서 발급해 ~/.zshrc 에 export.
# 본 스크립트는 키를 echo 하지 않음 (보안). 미설정 시 경고만.
if [ -z "$NAVER_CLIENT_ID" ] || [ -z "$NAVER_CLIENT_SECRET" ]; then
  echo "warn: NAVER_CLIENT_ID/SECRET unset — Naver API crawl will fail" >&2
fi
