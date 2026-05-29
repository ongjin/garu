"""Stage 1 수렴 결과 리포트 + 잔여 미해소 패턴 분포.

사용:
    $GARU_ENSEMBLE_PYTHON training/cleansing/resolve_report.py \\
        --resolved training/gold_testset/gold_resolved.jsonl \\
        --still training/gold_testset/gold_still_suspicious.jsonl \\
        --out training/cleansing/resolve_report.md
"""
import argparse
import json
from collections import Counter
from pathlib import Path


def report(resolved_path: Path, still_path: Path, out_path: Path):
    resolved = [json.loads(l) for l in open(resolved_path)]
    still = [json.loads(l) for l in open(still_path)]

    remaining_eoj = 0
    resolved_eoj_in_still = 0
    pattern = Counter()
    for d in still:
        for se in d["suspicious_eojeols"]:
            if se.get("resolved") is not None:
                resolved_eoj_in_still += 1
                continue
            remaining_eoj += 1
            cands = sorted(se["candidates"], key=lambda c: -len(c["analyzers"]))
            a = "+".join(f"{s}/{p}" for s, p in cands[0]["morphemes"])
            b = "+".join(f"{s}/{p}" for s, p in cands[1]["morphemes"]) if len(cands) > 1 else ""
            pattern[(a, b)] += 1

    # resolved 문장의 의심 어절 수는 resolved_by=phase2_rule 기록 시 함께 저장됨
    resolved_eoj_from_full = sum(d.get("resolved_eojeol_count", 1) for d in resolved)
    resolved_eoj_total = resolved_eoj_from_full + resolved_eoj_in_still

    lines = []
    lines.append("# 골드셋 클렌징 Phase 2 — Stage 1 (규칙 수렴) Report\n")
    lines.append(f"- 입력 의심 문장: {len(resolved) + len(still)}")
    lines.append(f"- 완전 해소 문장(resolved): {len(resolved)}")
    lines.append(f"- 잔여 문장(still): {len(still)}")
    lines.append(f"- 규칙 해소 어절: {resolved_eoj_total}")
    lines.append(f"- Claude로 넘길 어절: {remaining_eoj}\n")
    lines.append("## 잔여 미해소 패턴 Top 30 (Claude 입력)\n")
    lines.append("| 빈도 | 최다그룹 | 차순위그룹 |")
    lines.append("|------|----------|-----------|")
    for (a, b), c in pattern.most_common(30):
        lines.append(f"| {c} | {a} | {b} |")
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Report written: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolved", required=True, type=Path)
    ap.add_argument("--still", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    report(args.resolved, args.still, args.out)


if __name__ == "__main__":
    main()
