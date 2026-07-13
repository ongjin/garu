"""Averaged perceptron 재순위 학습.

입력: <data_dir>/{train,dev}.jsonl (gold) + {train,dev}_top10.jsonl (dump_topk k=10)
- feature: raw 후보 토큰에서 추출 (features.extract) — Rust production과 동일 입력
- 타깃: ep_norm 정규화 후 문장 F1 최대 후보 (동률 시 낮은 rank)
- 전처리 캐시: <data_dir>/cache_{split}.npz (feature/counts 재계산 방지)
출력: <data_dir>/weights.npz (dev micro F1 최고 에폭의 averaged weights)
"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE.parent / "gold_testset/expand"))
from ep_norm import normalize_ep_morphemes  # noqa: E402
from features import extract, hash_feats  # noqa: E402


def norm(tokens):
    return normalize_ep_morphemes([[t[0], t[1]] for t in tokens])


def prf_counts(pred, gold_counter):
    p = Counter((f, t) for f, t in pred)
    tp = sum((p & gold_counter).values())
    return tp, sum(p.values()) - tp, sum(gold_counter.values()) - tp


def sent_f1(tp, fp, fn):
    d = 2 * tp + fp + fn
    return 2 * tp / d if d else 1.0


def preprocess(data_dir, split):
    cache = data_dir / f"cache_{split}.npz"
    if cache.exists():
        z = np.load(cache)
        return {k: z[k] for k in z.files}

    gold_recs = [json.loads(l) for l in open(data_dir / f"{split}.jsonl")]
    dumps = [json.loads(l) for l in open(data_dir / f"{split}_top10.jsonl")]
    assert len(gold_recs) == len(dumps)

    all_ids, all_vals = [], []
    cand_feat_offsets = [0]  # per-candidate feature range
    sent_cand_offsets = [0]  # per-sentence candidate range
    counts = []  # (tp, fp, fn) per candidate
    srcs = []
    t0 = time.time()
    for i, (rec, dump) in enumerate(zip(gold_recs, dumps)):
        assert dump["text"] == rec["text"], f"line {i} mismatch"
        gold_counter = Counter((f, t) for f, t in norm(rec["morphemes"]))
        cands = dump["candidates"]
        base_score = cands[0]["score"] if cands else 0.0
        for rank, cand in enumerate(cands):
            raw = [[t[0], t[1]] for t in cand["tokens"]]
            counts.append(prf_counts(norm(raw), gold_counter))
            ids, vals = hash_feats(extract(raw, cand["score"] - base_score, rank + 1))
            all_ids.extend(ids)
            all_vals.extend(vals)
            cand_feat_offsets.append(len(all_ids))
        sent_cand_offsets.append(sent_cand_offsets[-1] + len(cands))
        srcs.append(1 if rec.get("src") == "SX" else 0)
        if (i + 1) % 10000 == 0:
            print(f"  {split} preprocess {i+1}/{len(gold_recs)} ({time.time()-t0:.0f}s)")

    out = {
        "ids": np.array(all_ids, dtype=np.int32),
        "vals": np.array(all_vals, dtype=np.float32),
        "feat_off": np.array(cand_feat_offsets, dtype=np.int64),
        "sent_off": np.array(sent_cand_offsets, dtype=np.int64),
        "counts": np.array(counts, dtype=np.int32),
        "srcs": np.array(srcs, dtype=np.int8),
    }
    np.savez_compressed(cache, **out)
    print(f"  {split}: {len(gold_recs)} sents, {len(counts)} cands, "
          f"{len(all_ids)} feat entries ({time.time()-t0:.0f}s)")
    return out


def cand_scores(w, data, s):
    """문장 s의 후보별 w·φ."""
    c0, c1 = data["sent_off"][s], data["sent_off"][s + 1]
    f0, f1 = data["feat_off"][c0], data["feat_off"][c1]
    if c0 == c1:
        return None, c0
    prod = w[data["ids"][f0:f1]] * data["vals"][f0:f1]
    bounds = data["feat_off"][c0:c1] - f0
    return np.add.reduceat(prod, bounds), c0


def evaluate(w, data):
    """averaged weights로 dev 평가: pick/rank1/oracle micro F1."""
    agg = {"pick": np.zeros(3, np.int64), "rank1": np.zeros(3, np.int64),
           "oracle": np.zeros(3, np.int64)}
    n_sents = len(data["sent_off"]) - 1
    for s in range(n_sents):
        scores, c0 = cand_scores(w, data, s)
        if scores is None:
            continue
        cnt = data["counts"][c0:c0 + len(scores)]
        f1s = np.array([sent_f1(*c) for c in cnt])
        agg["pick"] += cnt[int(np.argmax(scores))]
        agg["rank1"] += cnt[0]
        agg["oracle"] += cnt[int(np.argmax(f1s))]
    out = {}
    for k, (tp, fp, fn) in agg.items():
        out[k] = 2 * tp / (2 * tp + fp + fn)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", type=Path)
    ap.add_argument("--epochs", type=int, default=8)
    args = ap.parse_args()

    from features import DIM
    train = preprocess(args.data_dir, "train")
    dev = preprocess(args.data_dir, "dev")

    n_train = len(train["sent_off"]) - 1
    # 학습 대상: 후보 간 F1 분산이 있는 문장만
    f1_cache = []
    trainable = []
    for s in range(n_train):
        c0, c1 = train["sent_off"][s], train["sent_off"][s + 1]
        f1s = np.array([sent_f1(*c) for c in train["counts"][c0:c1]])
        f1_cache.append(f1s)
        if c1 - c0 >= 2 and f1s.max() - f1s.min() > 1e-9:
            trainable.append(s)
    print(f"trainable sentences: {len(trainable)}/{n_train}")

    w = np.zeros(DIM, dtype=np.float64)
    wa = np.zeros(DIM, dtype=np.float64)
    step = 0
    rng = np.random.default_rng(42)
    best = (-1.0, None)

    for epoch in range(args.epochs):
        order = rng.permutation(len(trainable))
        updates = 0
        for oi in order:
            s = trainable[oi]
            step += 1
            scores, c0 = cand_scores(w, train, s)
            f1s = f1_cache[s]
            pick = int(np.argmax(scores))
            oracle = int(np.argmax(f1s))
            if f1s[pick] < f1s[oracle] - 1e-9:
                updates += 1
                for idx, sign in [(oracle, 1.0), (pick, -1.0)]:
                    f0 = train["feat_off"][c0 + idx]
                    f1_ = train["feat_off"][c0 + idx + 1]
                    ids = train["ids"][f0:f1_]
                    delta = sign * train["vals"][f0:f1_]
                    np.add.at(w, ids, delta)
                    np.add.at(wa, ids, step * delta)
        w_avg = (w - wa / step).astype(np.float32)
        ev = evaluate(w_avg, dev)
        marker = ""
        if ev["pick"] > best[0]:
            best = (ev["pick"], w_avg.copy())
            marker = "  <-- best"
        print(f"epoch {epoch+1}: updates={updates}  dev pick={ev['pick']:.4f} "
              f"rank1={ev['rank1']:.4f} oracle={ev['oracle']:.4f}{marker}")

    np.savez_compressed(args.data_dir / "weights.npz", w=best[1])
    print(f"saved weights.npz (dev pick={best[0]:.4f})")


if __name__ == "__main__":
    main()
