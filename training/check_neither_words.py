"""Step 2: 20개 neither 단어 Garu vs Kiwi vs Gold 3방향 분석 비교.

temp_compound_sources.tsv에서 in_cache=N AND in_dict=N 필터로 neither 단어 추출,
Garu, Kiwi, 골드 테스트셋 비교 후 action(패치/skip) 결정.

결론: gold 양방향 공존 확인 후 무차별 패치 금지 (project_guuh_weakness.md 규칙).

Usage:
    python training/check_neither_words.py
"""
import json
import os
import subprocess
import tempfile
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
SOURCES_TSV = os.path.join(BASE, "temp_compound_sources.tsv")
GOLD_JSONL = os.path.join(BASE, "gold_testset/gold_testset.jsonl")
GOLD_CHALLENGE = os.path.join(BASE, "gold_testset/gold_challenge.jsonl")
ANALYZE_BIN = os.path.join(ROOT, "target/release/examples/analyze_batch")
MODEL = os.path.join(ROOT, "models/codebook.gmdl")


def load_neither_words():
    """Load words with in_cache=N AND in_dict=N from TSV."""
    words = []
    with open(SOURCES_TSV) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            surface, freq, in_cache, in_dict = parts[0], parts[1], parts[2], parts[3]
            if surface == "surface":  # header
                continue
            if in_cache == "N" and in_dict == "N":
                words.append(surface)
    return words


def load_gold_frequencies(words):
    """Count single vs split occurrences in gold for each word."""
    results = defaultdict(lambda: {"single": 0, "split": 0})
    for gold_file in [GOLD_JSONL, GOLD_CHALLENGE]:
        if not os.path.exists(gold_file):
            continue
        with open(gold_file) as f:
            for line in f:
                d = json.loads(line)
                morphemes = d["morphemes"]
                for w in words:
                    # Check single token
                    for m in morphemes:
                        if m[0] == w:
                            results[w]["single"] += 1
                    # Check split across consecutive tokens
                    for i in range(len(morphemes)):
                        for j in range(i + 2, min(i + 5, len(morphemes) + 1)):
                            joined = "".join(m[0] for m in morphemes[i:j])
                            if joined == w:
                                results[w]["split"] += 1
    return results


def run_garu(words):
    """Run Garu analyze_batch on words, return dict word→morphs."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for w in words:
            f.write(w + "\n")
        tmp = f.name
    try:
        env = os.environ.copy()
        env["GARU_MODEL"] = MODEL
        result = subprocess.run(
            [ANALYZE_BIN, tmp],
            capture_output=True, text=True, env=env
        )
        output = result.stdout
    finally:
        os.unlink(tmp)

    word_results = {}
    current_morphs = []
    idx = 0
    for line in output.splitlines():
        if line == "---":
            if idx < len(words):
                word_results[words[idx]] = current_morphs
            idx += 1
            current_morphs = []
        elif "\t" in line:
            form, pos = line.split("\t", 1)
            current_morphs.append((form, pos))
    return word_results


def run_kiwi(words):
    """Run Kiwi on words, return dict word→morphs."""
    try:
        import kiwipiepy
        kiwi = kiwipiepy.Kiwi()
    except ImportError:
        print("WARNING: kiwipiepy not installed, Kiwi analysis skipped")
        return {}

    results = {}
    for w in words:
        tokens = kiwi.tokenize(w)
        morphs = [(t.form, t.tag) for t in tokens]
        results[w] = morphs
    return results


def decide_action(word, garu_morphs, kiwi_morphs, gold_freq):
    """Decide action based on Garu, Kiwi, and gold frequency data."""
    garu_single = len(garu_morphs) == 1

    if not garu_single:
        return "skip", "Garu already splits"

    kiwi_single = len(kiwi_morphs) == 1 if kiwi_morphs else True
    if kiwi_single:
        return "skip", "both single (loanword/no split needed)"

    gold_single = gold_freq.get("single", 0)
    gold_split = gold_freq.get("split", 0)
    total_gold = gold_single + gold_split

    if total_gold == 0:
        # Not in gold — follow Kiwi (safe to patch if Kiwi splits)
        return "patch", "not in gold, Kiwi splits"

    if gold_single > 0 and gold_split > 0:
        # Ambiguous: unconditional cache would help half and hurt half
        return "skip", f"gold ambiguous (single={gold_single}, split={gold_split})"

    if gold_split > 0 and gold_single == 0:
        # Gold always split → patch
        return "patch", f"gold always split (n={gold_split})"

    # gold_single > 0, gold_split == 0 → single is correct
    return "skip", f"gold always single (n={gold_single})"


def main():
    words = load_neither_words()
    print(f"Neither words (in_cache=N, in_dict=N): {len(words)}")
    print(", ".join(words))
    print()

    print("Running Garu...")
    garu_results = run_garu(words)

    print("Running Kiwi...")
    kiwi_results = run_kiwi(words)

    print("Loading gold frequencies...")
    gold_freqs = load_gold_frequencies(words)

    print()
    header = f"{'word':<16} {'garu':<28} {'kiwi':<28} {'gold_s/sp':<12} {'action':<10} note"
    print(header)
    print("-" * 115)

    patch_targets = []
    skip_count = 0

    for w in words:
        garu_morphs = garu_results.get(w, [])
        kiwi_morphs = kiwi_results.get(w, [])
        gf = gold_freqs.get(w, {"single": 0, "split": 0})

        garu_str = "+".join(f"{f}/{p}" for f, p in garu_morphs) if garu_morphs else "?"
        kiwi_str = "+".join(f"{f}/{p}" for f, p in kiwi_morphs) if kiwi_morphs else "?"
        gold_str = f"{gf['single']}/{gf['split']}"

        action, note = decide_action(w, garu_morphs, kiwi_morphs, gf)
        print(f"{w:<16} {garu_str:<28} {kiwi_str:<28} {gold_str:<12} {action:<10} {note}")

        if action == "patch":
            patch_targets.append((w, kiwi_morphs))
        else:
            skip_count += 1

    print()
    print(f"Patch targets ({len(patch_targets)}):")
    for w, morphs in patch_targets:
        ms = "+".join(f"{f}/{p}" for f, p in morphs)
        print(f"  {w} → {ms}")
    print(f"Skip: {skip_count}")

    return patch_targets


if __name__ == "__main__":
    main()
