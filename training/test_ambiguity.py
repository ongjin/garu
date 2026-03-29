"""Test ambiguous Korean sentences that require cross-eojeol context understanding.

Categories:
1. 동형이의어 (same surface, different meaning depending on context)
2. 품사 모호성 (same word, different POS depending on role)
3. 분절 모호성 (same string, different segmentation)
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent

# (sentence, expected_key_morphemes_as_tuples)
# We only check KEY morphemes that are ambiguous — not the entire analysis
TEST_CASES = [
    # === 1. "나는" 동형이의어 ===
    ("나는 하늘을 나는 새를 보았다.",
     {"나는₂": [("나", "VV"), ("는", "ETM")]},
     "두 번째 '나는'은 날다(VV)+는(ETM)"),

    ("나는 밥을 먹었다.",
     {"나는₁": [("나", "NP"), ("는", "JX")]},
     "'나는'은 대명사+조사"),

    # === 2. "있는" — VA vs VX ===
    ("재미있는 영화를 봤다.",
     {"있는": [("있", "VA"), ("는", "ETM")]},
     "'있는'은 형용사 (재미있다)"),

    ("먹고 있는 사람이 많다.",
     {"있는": [("있", "VX"), ("는", "ETM")]},
     "'있는'은 보조용언 (먹고 있다)"),

    # === 3. "한" — MM vs VV+ETM ===
    ("한 사람이 왔다.",
     {"한": [("한", "MM")]},
     "'한'은 관형사 (하나의)"),

    ("내가 한 일이 많다.",
     {"한": [("하", "VV"), ("ㄴ", "ETM")]},
     "'한'은 하다(VV)+ㄴ(ETM)"),

    # === 4. "그" — MM vs NP vs IC ===
    ("그 사람이 왔다.",
     {"그": [("그", "MM")]},
     "'그'는 관형사"),

    ("그는 학생이다.",
     {"그": [("그", "NP")]},
     "'그'는 대명사"),

    # === 5. "이" — MM vs VCP vs NP vs JKS ===
    ("이 책이 좋다.",
     {"이₁": [("이", "MM")]},
     "첫 번째 '이'는 관형사"),

    ("이것은 책이다.",
     {"이다": [("이", "VCP"), ("다", "EF")]},
     "'이다'의 '이'는 서술격조사(VCP)"),

    # === 6. "되" — VV vs XSV ===
    ("문제가 해결되었다.",
     {"되": [("되", "XSV")]},
     "'되'는 접미사(XSV) — 해결+되다"),

    ("밥이 되었다.",
     {"되": [("되", "VV")]},
     "'되'는 동사(VV) — 밥이 되다"),

    # === 7. "는" — JX vs ETM ===
    ("나는 간다.",
     {"는": [("는", "JX")]},
     "'는'은 보조사(JX)"),

    ("가는 길에 만났다.",
     {"는": [("는", "ETM")]},
     "'는'은 관형형어미(ETM)"),

    # === 8. 복합 모호성 ===
    ("그는 그 사건이 그렇게 된 것을 알고 있었다.",
     {"그₁": [("그", "NP")], "그₂": [("그", "MM")]},
     "첫 '그'는 대명사, 두 번째 '그'는 관형사"),

    ("사과가 맛있는 이 가게는 유명하다.",
     {"있는": [("있", "VA"), ("는", "ETM")], "이": [("이", "MM")]},
     "'있는'은 형용사, '이'는 관형사"),

    ("나는 나는 것이 무섭다.",
     {"나는₁": [("나", "NP"), ("는", "JX")], "나는₂": [("나", "VV"), ("는", "ETM")]},
     "첫 '나는'은 대명사, 두 번째 '나는'은 동사"),
]


def run_analyzer(sentences, example_name):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        for sent, _, _ in sentences:
            f.write(sent + '\n')
        input_path = f.name

    result = subprocess.run(
        ["cargo", "run", "--release", "--example", example_name, "--", input_path],
        cwd=str(ROOT), capture_output=True, text=True, timeout=300,
    )
    os.unlink(input_path)

    if result.returncode != 0:
        print(f"Failed: {result.stderr[:300]}", file=sys.stderr)
        return None

    analyses = []
    current = []
    for line in result.stdout.strip().split('\n'):
        line = line.strip()
        if line == '---':
            analyses.append(current)
            current = []
        elif line == '[]':
            analyses.append([])
        elif '\t' in line:
            form, pos = line.split('\t', 1)
            current.append((form, pos))
    if current:
        analyses.append(current)
    return analyses


def check_result(analysis, expected_morphs, description):
    """Check if key expected morphemes appear in the analysis."""
    analysis_set = set((f, p) for f, p in analysis)

    passed = True
    details = []
    for key, expected in expected_morphs.items():
        exp_set = set((f, p) for f, p in expected)
        found = exp_set.issubset(analysis_set)
        if not found:
            passed = False
            details.append(f"  MISS: {key} expected {expected}")
            # Find what was actually produced for this form
            for ef, ep in expected:
                actual = [(f, p) for f, p in analysis if f == ef]
                if actual:
                    details.append(f"    got: {actual}")

    return passed, details


def main():
    print("=" * 70)
    print("  AMBIGUITY STRESS TEST")
    print("=" * 70)

    for version, example in [("v1", "analyze_batch"), ("v2", "analyze_batch_v2")]:
        print(f"\n{'='*70}")
        print(f"  {version}")
        print(f"{'='*70}")

        results = run_analyzer(TEST_CASES, example)
        if not results:
            print("  FAILED TO RUN")
            continue

        pass_count = 0
        fail_count = 0

        for i, (sent, expected, desc) in enumerate(TEST_CASES):
            if i >= len(results):
                break
            analysis = results[i]
            passed, details = check_result(analysis, expected, desc)

            if passed:
                print(f"  PASS: {desc}")
                pass_count += 1
            else:
                print(f"  FAIL: {desc}")
                print(f"    문장: {sent}")
                morphs_str = " ".join(f"{f}/{p}" for f, p in analysis)
                print(f"    분석: {morphs_str}")
                for d in details:
                    print(d)
                fail_count += 1

        print(f"\n  결과: {pass_count} PASS, {fail_count} FAIL / {pass_count + fail_count} total")


if __name__ == "__main__":
    main()
