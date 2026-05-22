"""v1 vs v2 candidates_pool 비교: 도메인 추가 후 yield 변화 확인.

스펙 5.3.1 abandonment criterion + layer별 통과 추정.
"""
import argparse
import json
from collections import Counter, defaultdict


LAYER_GATES = {
    # name: (required_pos, min_votes, min_freq, allowed_domains, min_len, stoplist)
    "L2-News-NNG": ({"NNG"}, 3, 10, {"news"}, 2,
                    {"기자","오늘","어제","내일","올해","작년","내년",
                     "그것","이것","저것","이거","저거","그거","여기","저기","거기"}),
    "L2-News-NNP": ({"NNP"}, 4, 3, {"news"}, 2,
                    {"오늘","어제","내일","올해","작년","기자","씨","님","분",
                     "그","이","저","그것","이것","저것","1세","2세","3세","전투"}),
    "L3-Guueh":    ({"NNG","MAG"}, 4, 10, {"blog"}, 2,
                    {"와","헐","ㅎ","ㅋ","ㅠ","ㅜ","정말","진짜","완전","너무","엄청"}),
    "L4-VV-VA":    ({"VV","VA"}, 4, 10, None, 1, set()),
    "L5-Compound": ({"NNG"}, 4, 10, None, 3, set()),
    # 신규: wiki 활용 가능한 layer (NNP 제외)
    "L6-Wiki-NNG-strict": ({"NNG"}, 4, 10, {"wiki"}, 2, set()),
    "L6-Wiki-NNG-3of4":   ({"NNG"}, 3, 10, {"wiki"}, 2, set()),
}


def load_pool(path):
    out = []
    with open(path) as f:
        for line in f:
            out.append(json.loads(line))
    return out


def filter_count(pool, cfg):
    req_pos, min_votes, min_freq, allowed_domains, min_len, stoplist = cfg
    count = 0
    for r in pool:
        if r["normalized_pos"] not in req_pos: continue
        if r["in_garu_dict"]: continue
        if sum(r["votes"].values()) < min_votes: continue
        if r["frequency"] < min_freq: continue
        if allowed_domains is not None:
            if not (set(r["source_domains"]) & allowed_domains): continue
        if len(r["surface"]) < min_len: continue
        if r["surface"] in stoplist: continue
        count += 1
    return count


def summary(name, pool):
    n = len(pool)
    in_dict = sum(1 for r in pool if r["in_garu_dict"])
    by_votes = Counter(sum(r["votes"].values()) for r in pool)
    by_domain = Counter()
    for r in pool:
        for d in r["source_domains"]:
            by_domain[d] += 1
    consensus_ge_3 = sum(c for v, c in by_votes.items() if v >= 3)
    non_dict_4v_f10 = sum(1 for r in pool
                          if not r["in_garu_dict"]
                          and sum(r["votes"].values()) == 4
                          and r["frequency"] >= 10)
    non_dict_3v_f10 = sum(1 for r in pool
                          if not r["in_garu_dict"]
                          and sum(r["votes"].values()) >= 3
                          and r["frequency"] >= 10)
    return {
        "name": name,
        "total": n,
        "in_dict": in_dict,
        "non_dict_4v_f10": non_dict_4v_f10,
        "non_dict_3v_f10": non_dict_3v_f10,
        "consensus_ge_3": consensus_ge_3,
        "by_domain": dict(by_domain),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", default="training/codebook_data/candidates_pool.jsonl")
    ap.add_argument("--v2", default="training/codebook_data/candidates_pool_v2.jsonl")
    args = ap.parse_args()

    p1 = load_pool(args.v1); p2 = load_pool(args.v2)
    s1 = summary("v1 (news+blog)", p1)
    s2 = summary("v2 (+wiki)", p2)

    print("\n=== 풀 요약 비교 ===")
    print(f"{'metric':30s} {'v1':>15s} {'v2':>15s} {'delta':>10s}")
    for k in ("total","in_dict","non_dict_4v_f10","non_dict_3v_f10","consensus_ge_3"):
        delta = s2[k] - s1[k]
        print(f"{k:30s} {s1[k]:>15d} {s2[k]:>15d} {delta:>+10d}")

    print("\n=== 도메인 분포 ===")
    domains = sorted(set(list(s1['by_domain'].keys()) + list(s2['by_domain'].keys())))
    print(f"{'domain':15s} {'v1':>15s} {'v2':>15s}")
    for d in domains:
        print(f"{d:15s} {s1['by_domain'].get(d,0):>15d} {s2['by_domain'].get(d,0):>15d}")

    print("\n=== Layer 게이트 통과 예상 (필터만, runner의 도메인 게이트 측정은 별도) ===")
    print(f"{'layer':30s} {'v1':>10s} {'v2':>10s} {'delta':>10s}")
    for name, cfg in LAYER_GATES.items():
        c1 = filter_count(p1, cfg); c2 = filter_count(p2, cfg)
        print(f"{name:30s} {c1:>10d} {c2:>10d} {c2-c1:>+10d}")


if __name__ == "__main__":
    main()
