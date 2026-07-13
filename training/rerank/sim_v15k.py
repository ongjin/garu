"""학습된 weights를 v15k 테스트셋에 시뮬레이션 (평가 전용 — 튜닝 금지).

production 근사: final이 후보군에 없으면(analyze_inner early-return 경로) final 유지,
있으면 perceptron argmax로 교체. Phase 2 Rust 통합이 정밀 측정.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE.parent / "gold_testset/expand"))
from ep_norm import normalize_ep_morphemes  # noqa: E402
from features import extract, hash_feats  # noqa: E402


def norm(tokens):
    return normalize_ep_morphemes([[t[0], t[1]] for t in tokens])


def counts(pred, gold_counter):
    p = Counter((f, t) for f, t in pred)
    tp = sum((p & gold_counter).values())
    return np.array([tp, sum(p.values()) - tp, sum(gold_counter.values()) - tp])


def micro(agg):
    tp, fp, fn = agg
    return 2 * tp / (2 * tp + fp + fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", type=Path)
    ap.add_argument("--scratch", type=Path, required=True,
                    help="final.jsonl / top20.jsonl / gold 덤프 디렉토리")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--margin", type=float, default=0.0,
                    help="s(pick)-s(production후보) > margin 일 때만 override")
    args = ap.parse_args()

    w = np.load(args.weights)["w"]
    gold_recs = [json.loads(l) for l in
                 open(BASE.parent / "gold_testset/gold_testset.jsonl")]
    finals = [json.loads(l) for l in open(args.scratch / "final.jsonl")]
    dumps = [json.loads(l) for l in open(args.scratch / "top20.jsonl")]

    agg_base = np.zeros(3, np.int64)
    agg_sim = np.zeros(3, np.int64)
    dom_base = defaultdict(lambda: np.zeros(3, np.int64))
    dom_sim = defaultdict(lambda: np.zeros(3, np.int64))
    changed = improved = worsened = kept_final = 0

    for rec, final, dump in zip(gold_recs, finals, dumps):
        assert dump["text"] == rec["text"]
        domain = rec.get("domain", "?")
        gold_counter = Counter((f, t) for f, t in norm(rec["morphemes"]))
        final_raw = [[t[0], t[1]] for t in final]
        base_c = counts(norm(final_raw), gold_counter)

        cands = dump["candidates"][:args.k]
        cand_raws = [[[t[0], t[1]] for t in c["tokens"]] for c in cands]
        if final_raw not in cand_raws:
            sim_c = base_c  # early-return 경로 근사: 재순위 미적용
            kept_final += 1
        else:
            i_final = cand_raws.index(final_raw)
            base_score = cands[0]["score"]
            scores = []
            for i, (c, raw) in enumerate(zip(cands, cand_raws)):
                ids, vals = hash_feats(extract(raw, c["score"] - base_score, i + 1))
                scores.append(float(np.dot(w[ids], vals)))
            best_i = int(np.argmax(scores))
            if scores[best_i] - scores[i_final] <= args.margin:
                best_i = i_final
            sim_c = counts(norm(cand_raws[best_i]), gold_counter)
            if cand_raws[best_i] != final_raw:
                changed += 1
                bf, sf = micro(base_c), micro(sim_c)
                if sf > bf + 1e-9:
                    improved += 1
                elif sf < bf - 1e-9:
                    worsened += 1

        agg_base += base_c
        agg_sim += sim_c
        dom_base[domain] += base_c
        dom_sim[domain] += sim_c

    print(f"baseline (production): F1={micro(agg_base):.4f}")
    print(f"simulated (rerank k={args.k}): F1={micro(agg_sim):.4f} "
          f"({micro(agg_sim)-micro(agg_base):+.4f})")
    print(f"kept_final(early-return)={kept_final} changed={changed} "
          f"improved={improved} worsened={worsened}")
    print()
    for d in sorted(dom_base):
        b, s = micro(dom_base[d]), micro(dom_sim[d])
        print(f"{d:<10} base={b:.4f} sim={s:.4f} ({s-b:+.4f})")


if __name__ == "__main__":
    main()
