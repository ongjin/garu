"""9K 골드셋 어절 단위 다수결 voting 메인 파이프라인.

입력: jsonl ({"text", "domain", ...})
출력:
  - voted.jsonl: 모든 어절이 다수결 통과한 정상 문장
  - suspicious.jsonl: 어절 1개 이상 의심인 문장 + 5분석기 raw 후보

사용:
    source training/ensemble/env_setup.sh
    $GARU_ENSEMBLE_PYTHON training/cleansing/run_voting.py \\
        --input training/gold_testset/gold_testset.jsonl \\
        --voted training/gold_testset/gold_voted.jsonl \\
        --suspicious training/gold_testset/gold_suspicious.jsonl
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training" / "ensemble"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wrappers import KiwiWrapper, MecabWrapper, KkmaWrapper, KomoranWrapper, GaruWrapper
from eojeol_voter import vote_eojeol


def _build_analyzers():
    return {
        "mecab":   MecabWrapper(),
        "kkma":    KkmaWrapper(),
        "komoran": KomoranWrapper(),
        "kiwi":    KiwiWrapper(),
        "garu":    GaruWrapper(),
    }


def process_one_sentence(text: str, analyzers: dict) -> dict:
    """한 문장을 어절로 쪼개고 어절별 voting 수행. 결과 dict 반환."""
    eojeols = text.split()
    eojeol_votes_normal = []
    suspicious_eojeols = []
    for idx, eoj in enumerate(eojeols):
        analyses = {}
        for name, a in analyzers.items():
            try:
                analyses[name] = a.analyze(eoj)
            except Exception:
                analyses[name] = []
        r = vote_eojeol(eoj, analyses)
        if r.status == "normal":
            eojeol_votes_normal.append({
                "surface": eoj,
                "agree": r.agree,
                "morphemes": r.morphemes,
            })
        else:
            suspicious_eojeols.append({
                "index": idx,
                "surface": eoj,
                "agree": r.agree,
                "candidates": r.candidates,
            })

    if not suspicious_eojeols:
        flat_morphs = []
        for ev in eojeol_votes_normal:
            flat_morphs.extend(ev["morphemes"])
        return {
            "vote_status": "normal",
            "morphemes": flat_morphs,
            "eojeol_votes": eojeol_votes_normal,
        }
    return {
        "vote_status": "suspicious",
        "eojeol_votes": eojeol_votes_normal,
        "suspicious_eojeols": suspicious_eojeols,
    }


def process_sentences(input_path: Path, voted_path: Path, suspicious_path: Path) -> dict:
    """jsonl 파일을 처리해 두 출력 파일에 기록. 통계 dict 반환."""
    print(f"Loading analyzers...", flush=True)
    analyzers = _build_analyzers()

    stats = {"total": 0, "normal": 0, "suspicious": 0, "by_domain": {}}
    start = time.time()

    with open(input_path) as fin, \
         open(voted_path, "w") as fv, \
         open(suspicious_path, "w") as fs:
        for line in fin:
            d = json.loads(line)
            stats["total"] += 1
            result = process_one_sentence(d["text"], analyzers)

            # 원본 메타데이터 유지하되 기존 morphemes / source / confidence 제거
            out = {"text": d["text"], "domain": d.get("domain", "unknown")}
            out.update(result)

            domain = out["domain"]
            stats["by_domain"].setdefault(domain, {"total": 0, "normal": 0, "suspicious": 0})
            stats["by_domain"][domain]["total"] += 1

            if result["vote_status"] == "normal":
                stats["normal"] += 1
                stats["by_domain"][domain]["normal"] += 1
                fv.write(json.dumps(out, ensure_ascii=False) + "\n")
            else:
                stats["suspicious"] += 1
                stats["by_domain"][domain]["suspicious"] += 1
                fs.write(json.dumps(out, ensure_ascii=False) + "\n")

            if stats["total"] % 100 == 0:
                elapsed = time.time() - start
                rate = stats["total"] / elapsed
                print(
                    f"  [{stats['total']}] normal={stats['normal']} "
                    f"suspicious={stats['suspicious']} ({rate:.1f} sent/s)",
                    flush=True,
                )

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s. {stats['total']} sentences.")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--voted", required=True, type=Path)
    ap.add_argument("--suspicious", required=True, type=Path)
    args = ap.parse_args()
    stats = process_sentences(args.input, args.voted, args.suspicious)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
