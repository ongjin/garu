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


def process_one_sentence(text: str, analyzers: dict, errors: dict) -> dict:
    """한 문장을 어절로 쪼개고 어절별 voting 수행. 결과 dict 반환.

    errors: {analyzer_name: count} — 누적 카운트, 첫 3건만 stderr 로그.
    """
    eojeols = text.split()
    eojeol_votes_normal = []
    suspicious_eojeols = []

    # Garu만 어절별 subprocess → 문장당 1회 배치로 묶음 (~10x).
    # 다른 4분석기는 in-process 호출이라 어절별 그대로 유지.
    garu = analyzers["garu"]
    try:
        garu_results = garu.analyze_batch(eojeols)
    except Exception as e:
        garu_results = [[] for _ in eojeols]
        errors["garu"] = errors.get("garu", 0) + 1
        if errors["garu"] <= 3:
            print(f"  WARN garu batch failed on sentence: {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)

    for idx, eoj in enumerate(eojeols):
        analyses = {}
        for name, a in analyzers.items():
            if name == "garu":
                analyses["garu"] = garu_results[idx]
                continue
            try:
                analyses[name] = a.analyze(eoj)
            except Exception as e:
                analyses[name] = []
                errors[name] = errors.get(name, 0) + 1
                if errors[name] <= 3:
                    print(f"  WARN {name} failed on {eoj!r}: {type(e).__name__}: {e}",
                          file=sys.stderr, flush=True)
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
    if voted_path.exists() or suspicious_path.exists():
        raise SystemExit(
            f"Output files already exist:\n"
            f"  {voted_path}\n  {suspicious_path}\n"
            f"Delete them and re-run, or choose different paths."
        )
    print("Loading analyzers...", flush=True)
    analyzers = _build_analyzers()

    stats = {"total": 0, "normal": 0, "suspicious": 0, "by_domain": {}}
    errors: dict[str, int] = {}
    bad_lines = 0
    start = time.time()

    with open(input_path) as fin, \
         open(voted_path, "w") as fv, \
         open(suspicious_path, "w") as fs:
        for line in fin:
            try:
                d = json.loads(line)
                text = d["text"]
            except (json.JSONDecodeError, KeyError) as e:
                bad_lines += 1
                print(f"  WARN skipping malformed line {stats['total'] + bad_lines}: {type(e).__name__}",
                      file=sys.stderr, flush=True)
                continue
            stats["total"] += 1
            result = process_one_sentence(text, analyzers, errors)

            # 원본 메타데이터 유지하되 기존 morphemes / source / confidence 제거
            out = {"text": text, "domain": d.get("domain", "unknown")}
            # result keys (vote_status / morphemes / eojeol_votes / suspicious_eojeols) must not
            # collide with text/domain; see process_one_sentence return shape.
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
                fv.flush()
                fs.flush()

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s. {stats['total']} sentences.")
    stats["analyzer_errors"] = errors
    stats["bad_lines"] = bad_lines
    if errors:
        print(f"\nAnalyzer error counts: {errors}", file=sys.stderr, flush=True)
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
