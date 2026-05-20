"""전체 F1 리포트: 도메인별 + 신뢰도별 + challenge."""
import json
import sys
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

from eval_f1 import run_garu, run_kiwi, run_mecab, compute_f1


def group_by(records, key):
    out = defaultdict(list)
    for r in records:
        out[r.get(key, "unknown")].append(r)
    return out


def report_for_subset(name, records):
    if not records:
        return f"## {name}\n(empty)\n\n"
    texts = [r["text"] for r in records]
    gold = [r["morphemes"] for r in records]
    garu = run_garu(texts)
    kiwi = run_kiwi(texts)
    mec = run_mecab(texts)

    out = f"## {name} (n={len(records)})\n\n"
    out += "| 분석기 | Precision | Recall | F1 |\n"
    out += "|--------|-----------|--------|----|\n"
    for label, pred in [("Garu", garu), ("Kiwi", kiwi), ("Mecab", mec)]:
        p, r, f = compute_f1(pred, gold)
        out += f"| {label} | {p:.4f} | {r:.4f} | {f:.4f} |\n"
    return out + "\n"


def main():
    main_path = BASE / "gold_testset.jsonl"
    challenge_path = BASE / "gold_challenge.jsonl"

    main_records = [json.loads(l) for l in open(main_path)]
    challenge_records = [json.loads(l) for l in open(challenge_path)] if challenge_path.exists() else []

    out = f"# F1 Report (gold v15k)\n\n총 메인: {len(main_records)} / challenge: {len(challenge_records)}\n\n"

    out += "# 전체 (메인 + challenge)\n"
    out += report_for_subset("Overall", main_records + challenge_records)
    out += "# 메인 골드만\n"
    out += report_for_subset("Main", main_records)
    out += "# Challenge set만\n"
    out += report_for_subset("Challenge", challenge_records)

    out += "# 도메인별 (메인)\n"
    by_domain = group_by(main_records, "domain")
    for domain in ["뉴스", "일상", "구어", "기술", "문학", "SNS", "의료"]:
        if domain in by_domain:
            out += report_for_subset(f"도메인: {domain}", by_domain[domain])

    out += "# 신뢰도별 (메인)\n"
    by_conf = group_by(main_records, "confidence")
    for conf in ["manual", "reviewed", "auto"]:
        if conf in by_conf:
            out += report_for_subset(f"신뢰도: {conf}", by_conf[conf])

    report_path = BASE / "F1_report_v15k.md"
    report_path.write_text(out)
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
