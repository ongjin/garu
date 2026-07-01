"""불규칙 활용 과거형·존댓말 복원 회귀 테스트.

Garu가 불규칙 어간의 과거형(었)·존댓말(으시)에서 어절을 NNG로 붕괴시키지 않고
표준대사전 기본형 어간 + 어미로 복원하는지 확인. build_codebook_model.py의
augment_irregular_conjugations 3-패치(ㅂ 었 / ㅅ 분기 / ㅡ탈락 었 버그) 검증.

실행: python3 training/test_irregular_restore.py
(현재 모델 js/models/base.gmdl 을 GARU_MODEL로 씀 — 리빌드 후 재실행.)
"""
import os, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent

# (문장, 타깃어절, 반드시 포함해야 할 (form,pos) 형태소)
# ep_norm 관례상 과거 어미는 밝은모음도 었/EP 로 기록됨(코드북 컨벤션).
PAST_CASES = [
    ("겨울이라 날씨가 무척 추웠다", "추웠다", ("춥", "VA")),       # ㅂ 단음절
    ("가방이 너무 무거웠다", "무거웠다", ("무겁", "VA")),          # ㅂ 다음절
    ("머리가 심하게 아팠다", "아팠다", ("아프", "VA")),           # ㅡ탈락
    ("일이 너무 바빴다", "바빴다", ("바쁘", "VA")),              # ㅡ탈락
    ("그들이 새 집을 지었다", "지었다", ("짓", "VV")),           # ㅅ
    ("요리사가 국을 저었다", "저었다", ("젓", "VV")),           # ㅅ
]

HONORIFIC_CASES = [
    ("할머니 손이 참 고우시다", "고우시다", ("곱", "VA")),        # ㅂ + 으시
    ("선생님이 매일 길을 걸으신다", "걸으신다", ("걷", "VV")),     # ㄷ + 으시
    ("어머니가 국을 저으신다", "저으신다", ("젓", "VV")),         # ㅅ + 으시
    ("할머니는 마음이 참 고우셨다", "고우셨다", ("곱", "VA")),     # ㅂ + 으시 + 었(과거존대)
]

# 회귀 가드: 정칙 활용·계사는 불규칙 과대생성에 영향받지 않아야 한다.
GUARD_CASES = [
    ("길이 너무 좁았다", "좁았다", ("좁", "VA")),               # 정칙 ㅂ (조왔다 아님)
    ("방이 좀 좁으시다", "좁으시다", ("좁", "VA")),             # 정칙 ㅂ 존대
    ("어제는 추운 날이었다", "날이었다", ("이", "VCP")),         # 계사 이었다 (잇었다 아님)
]


def analyze(sentences):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        for s in sentences:
            f.write(s + "\n")
        path = f.name
    env = dict(os.environ, GARU_MODEL="js/models/base.gmdl")
    out = subprocess.run(
        ["cargo", "run", "-q", "--release", "--example", "analyze_batch", "--", path],
        cwd=str(ROOT), capture_output=True, text=True, env=env, timeout=300,
    )
    os.unlink(path)
    if out.returncode != 0:
        sys.exit(f"analyzer failed: {out.stderr[:400]}")
    analyses, cur = [], []
    for line in out.stdout.strip().split("\n"):
        line = line.rstrip()
        if line == "---":
            analyses.append(cur); cur = []
        elif line == "[]":
            pass
        elif "\t" in line:
            f_, p_ = line.split("\t", 1)
            cur.append((f_, p_))
    if cur:
        analyses.append(cur)
    return analyses


def run(cases, label):
    analyses = analyze([c[0] for c in cases])
    fails = []
    for (sent, target, need), morphs in zip(cases, analyses):
        forms = [f for f, _ in morphs]
        # 붕괴: 타깃 어절이 통째 한 토큰으로 남았는가
        collapsed = target in forms
        has_stem = need in morphs
        ok = has_stem and not collapsed
        status = "OK " if ok else "FAIL"
        print(f"  [{status}] {target:8} → need {need}  got {morphs}")
        if not ok:
            fails.append(target)
    print(f"{label}: {len(cases)-len(fails)}/{len(cases)} pass")
    return fails


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    fails = []
    if which in ("all", "past"):
        print("=== 과거형 (ㅂ/ㅅ/ㅡ탈락 + 었) ===")
        fails += run(PAST_CASES, "past")
    if which in ("all", "hon"):
        print("=== 존댓말 (불규칙 + 으시) ===")
        fails += run(HONORIFIC_CASES, "honorific")
    if which in ("all", "guard"):
        print("=== 회귀 가드 (정칙·계사 불변) ===")
        fails += run(GUARD_CASES, "guard")
    sys.exit(1 if fails else 0)
