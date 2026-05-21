"""구어 gold에서 Garu가 over-segment하는 어절(그렇/그러/거/그것 등) 추출.

방법: gold v15k 구어 도메인에서 어절(공백 분리) 단위로
1) 어절에 TARGET_STEMS 중 하나가 포함되는지 검사
2) 그 어절을 Garu에 단독 입력해 현재 분석 얻기
3) gold의 해당 어절 분석과 다르면 후보로 저장
4) (gold_analysis, eojeol) 쌍을 빈도순 집계해 ≥2회 등장만 남김
"""
import json, os, subprocess, sys, tempfile
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
GOLD = os.path.join(BASE, "gold_testset/gold_testset.jsonl")
ROOT = os.path.join(BASE, "..")
OUT = os.path.join(BASE, "codebook_data/guuh_overseg_candidates.jsonl")

TARGET_STEMS = {"그렇", "그러", "이렇", "이러", "저렇", "저러",
                "그것", "이것", "저것", "그거", "이거", "저거"}


def load_gold_with_eojeol_align():
    """For each 구어 record, attempt to align morphemes to eojeols via surface concatenation."""
    out = []
    with open(GOLD) as f:
        for line in f:
            r = json.loads(line)
            if r.get("domain") != "구어":
                continue
            text = r["text"]
            morphs = [tuple(m) for m in r["morphemes"]]
            eojeols = text.split()
            # Greedy alignment: assign morphs to eojeols left-to-right by concatenated surface
            alignments = []
            mi = 0
            for ej in eojeols:
                taken = []
                surface_built = ""
                # Strip leading/trailing punctuation for matching
                ej_strip = ej.strip(".,!?:;\"'()[]《》「」『』【】…·—–−~")
                while mi < len(morphs):
                    m_surf, m_pos = morphs[mi]
                    # Try adding this morph
                    candidate = surface_built + m_surf
                    # accept if candidate is prefix of ej or ej_strip
                    if ej.startswith(candidate) or ej_strip.startswith(candidate):
                        taken.append(morphs[mi])
                        surface_built = candidate
                        mi += 1
                        if surface_built == ej or surface_built == ej_strip:
                            break
                    else:
                        break
                if taken:
                    alignments.append((ej, taken))
            if alignments:
                out.append({"text": text, "alignments": alignments})
    return out


def run_garu_single(eojeols):
    """Run Garu on each eojeol independently."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for ej in eojeols:
            f.write(ej + "\n")
        p = f.name
    bin_path = os.path.join(ROOT, "target/release/examples/analyze_batch")
    model = os.path.join(ROOT, "js/models/base.gmdl")
    r = subprocess.run([bin_path, p, "--json"], capture_output=True, text=True,
                       env={**os.environ, "GARU_MODEL": model})
    os.unlink(p)
    results = []
    for line in r.stdout.strip().split("\n"):
        line = line.strip()
        if line:
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                results.append([])
    return results


def main():
    print("Loading 구어 gold with eojeol alignment...")
    records = load_gold_with_eojeol_align()
    print(f"  Records: {len(records)}")

    # Collect candidate eojeols
    candidate_eojeols = set()
    eojeol_to_gold = defaultdict(lambda: defaultdict(int))  # eojeol → (gold_morphs_tuple → count)
    for rec in records:
        for ej, gold_morphs in rec["alignments"]:
            if any(s in ej for s in TARGET_STEMS):
                candidate_eojeols.add(ej)
                eojeol_to_gold[ej][tuple(gold_morphs)] += 1
    print(f"  Candidate eojeols (with target stem): {len(candidate_eojeols)}")

    eojeol_list = sorted(candidate_eojeols)
    print(f"  Running Garu on {len(eojeol_list)} eojeols...")
    garu_preds = run_garu_single(eojeol_list)

    if len(garu_preds) != len(eojeol_list):
        print(f"  WARNING: expected {len(eojeol_list)} predictions, got {len(garu_preds)}")

    garu_map = {}
    for i, ej in enumerate(eojeol_list):
        if i < len(garu_preds):
            garu_map[ej] = tuple(tuple(t) for t in garu_preds[i])
        else:
            garu_map[ej] = ()

    # Find disagreements
    out_records = []
    for ej, gold_counts in eojeol_to_gold.items():
        best_gold, best_count = max(gold_counts.items(), key=lambda kv: kv[1])
        if best_count < 2:
            continue  # require ≥2 occurrences
        pred = garu_map.get(ej, ())
        if pred == best_gold:
            continue  # Garu already correct
        out_records.append({
            "eojeol": ej,
            "gold_analysis": [list(m) for m in best_gold],
            "garu_pred": [list(m) for m in pred],
            "freq": best_count,
        })
    out_records.sort(key=lambda r: -r["freq"])

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nExtracted {len(out_records)} candidates → {OUT}")
    print(f"\nTop 20:")
    for r in out_records[:20]:
        gold_str = " + ".join(f"{m[0]}/{m[1]}" for m in r["gold_analysis"])
        pred_str = " + ".join(f"{m[0]}/{m[1]}" for m in r["garu_pred"])
        print(f"  {r['eojeol']:20s} (freq={r['freq']:3d})")
        print(f"    gold: {gold_str}")
        print(f"    pred: {pred_str}")


if __name__ == "__main__":
    main()
