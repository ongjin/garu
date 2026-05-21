"""candidates_pool.jsonl 통계 + Plan B abandonment criterion 체크.

스펙 섹션 5.3.1: "5분석기 합의 통과 후보 < 30K → 게이트 너무 strict, 재설계 필요"
"""
import argparse
import json
from collections import Counter


def stats(pool_path: str):
    total = 0
    pos_dist = Counter()
    vote_dist = Counter()
    domain_dist = Counter()
    in_garu = 0
    freq_dist = []
    for line in open(pool_path):
        r = json.loads(line)
        total += 1
        pos_dist[r["normalized_pos"]] += 1
        vote_total = sum(r["votes"].values())
        vote_dist[vote_total] += 1
        for d in r["source_domains"]:
            domain_dist[d] += 1
        if r["in_garu_dict"]:
            in_garu += 1
        freq_dist.append(r["frequency"])

    print(f"=== candidates_pool stats ===")
    print(f"Total unique (surface, pos): {total}")
    print(f"In Garu dict already: {in_garu} ({100*in_garu/total:.1f}%)")
    print(f"\nVote distribution (0=no analyzer, 4=all 4 agree):")
    for v in sorted(vote_dist.keys()):
        print(f"  {v}/4: {vote_dist[v]} ({100*vote_dist[v]/total:.1f}%)")
    print(f"\nPOS distribution (top 10):")
    for p, c in pos_dist.most_common(10):
        print(f"  {p}: {c}")
    print(f"\nDomain distribution:")
    for d, c in domain_dist.most_common():
        print(f"  {d}: {c}")
    if freq_dist:
        freq_dist.sort()
        print(f"\nFrequency stats:")
        print(f"  min={freq_dist[0]}, median={freq_dist[len(freq_dist)//2]}, "
              f"max={freq_dist[-1]}, >=5: {sum(1 for f in freq_dist if f >= 5)}, "
              f">=10: {sum(1 for f in freq_dist if f >= 10)}")

    print(f"\n=== Abandonment criterion (스펙 5.3.1) ===")
    consensus_ge_3 = sum(c for v, c in vote_dist.items() if v >= 3)
    print(f"3/4 이상 합의 후보 수: {consensus_ge_3}")
    if consensus_ge_3 < 30000:
        print(f"⚠️  ABANDONMENT CRITERION TRIGGERED: < 30K — Plan C 진입 전 후보 풀 확장 필요")
        print(f"    조치 옵션:")
        print(f"    (a) pilot_queries.json target 증가 후 재크롤")
        print(f"    (b) Plan B-2로 본격 크롤 (1-5M 문장)")
        print(f"    (c) abandonment criterion 완화 (스펙 수정)")
    else:
        print(f"✅ Plan C 진입 가능 — 30K 충족")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="training/codebook_data/candidates_pool.jsonl")
    args = ap.parse_args()
    stats(args.pool)


if __name__ == "__main__":
    main()
