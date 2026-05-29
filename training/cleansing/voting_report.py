"""voting 결과로부터 통계 리포트(Markdown) 생성.

도메인별 의심율, 어절 agreement(5/4/3) 분포, 분석기별 동의 패턴 등.

사용:
    $GARU_ENSEMBLE_PYTHON training/cleansing/voting_report.py \
        --voted training/gold_testset/gold_voted.jsonl \
        --suspicious training/gold_testset/gold_suspicious.jsonl \
        --out training/cleansing/voting_report.md
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def _load(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def report(voted_path: Path, suspicious_path: Path, out_path: Path):
    voted = _load(voted_path)
    susp = _load(suspicious_path)

    domain_total = Counter()
    domain_susp = Counter()
    for d in voted:
        domain_total[d["domain"]] += 1
    for d in susp:
        domain_total[d["domain"]] += 1
        domain_susp[d["domain"]] += 1

    # 어절 agreement 분포 (정상 문장의 어절들만)
    agree_dist = Counter()
    for d in voted:
        for ev in d["eojeol_votes"]:
            agree_dist[ev["agree"]] += 1
    # 의심 문장의 정상 어절들도
    for d in susp:
        for ev in d.get("eojeol_votes", []):
            agree_dist[ev["agree"]] += 1
    # 의심 어절들의 best agree
    for d in susp:
        for se in d["suspicious_eojeols"]:
            agree_dist[se["agree"]] += 1

    # 분석기별 동의 패턴: 어떤 분석기 조합이 다수파였나
    # (정상 어절 중 agree==3,4인 케이스만 의미. agree==5는 모두 동의)
    analyzer_in_majority = Counter()
    total_majority_eojeols = 0

    # voted.eojeol_votes는 morphemes만 있고 어떤 분석기였는지 정보 없음
    # 정확한 분석은 suspicious.eojeol_votes에서도 알 수 없음 (구조 동일)
    # → 이 통계는 candidates가 있는 의심 어절에서만 가능. skip.

    lines = []
    lines.append("# 골드셋 클렌징 Phase 1 — Voting Report\n")
    lines.append(f"**총 문장:** {len(voted) + len(susp)}")
    lines.append(f"- 정상: {len(voted)} ({len(voted)/(len(voted)+len(susp))*100:.1f}%)")
    lines.append(f"- 의심: {len(susp)} ({len(susp)/(len(voted)+len(susp))*100:.1f}%)\n")

    lines.append("## 도메인별 의심율\n")
    lines.append("| 도메인 | 정상 | 의심 | 의심율 |")
    lines.append("|--------|------|------|--------|")
    for domain, total in sorted(domain_total.items(), key=lambda x: -x[1]):
        s = domain_susp.get(domain, 0)
        normal = total - s
        rate = s / total * 100 if total else 0
        lines.append(f"| {domain} | {normal} | {s} | {rate:.1f}% |")

    lines.append("\n## 어절 agreement 분포\n")
    total_eojeols = sum(agree_dist.values())
    lines.append("| agree | 어절 수 | 비율 |")
    lines.append("|-------|---------|------|")
    for k in sorted(agree_dist.keys(), reverse=True):
        v = agree_dist[k]
        lines.append(f"| {k}/5 | {v} | {v/total_eojeols*100:.1f}% |")

    lines.append("\n## 의심 어절 — agree 분포\n")
    susp_agree = Counter()
    for d in susp:
        for se in d["suspicious_eojeols"]:
            susp_agree[se["agree"]] += 1
    total_susp_eoj = sum(susp_agree.values())
    lines.append(f"전체 의심 어절: {total_susp_eoj}\n")
    lines.append("| 최대 동의 | 어절 수 | 비율 |")
    lines.append("|-----------|---------|------|")
    for k in sorted(susp_agree.keys(), reverse=True):
        v = susp_agree[k]
        lines.append(f"| {k}/5 | {v} | {v/total_susp_eoj*100:.1f}% |")

    out_path.write_text("\n".join(lines) + "\n")
    print(f"Report written: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voted", required=True, type=Path)
    ap.add_argument("--suspicious", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    report(args.voted, args.suspicious, args.out)


if __name__ == "__main__":
    main()
