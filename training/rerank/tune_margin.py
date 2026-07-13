"""dev에서 확신 마진 τ 튜닝: s(pick) - s(rank1) > τ 일 때만 override.

v15k는 건드리지 않는다 — 마진은 여기(dev)서만 고른다.
"""
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from train_perceptron import cand_scores, preprocess, sent_f1  # noqa: E402


def main():
    data_dir = Path(sys.argv[1])
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    w = np.load(data_dir / "weights.npz")["w"]
    dev = preprocess(data_dir, "dev")
    n = len(dev["sent_off"]) - 1

    # 문장별 (scores, counts) 1회 계산 (top_k개로 절단)
    per_sent = []
    for s in range(n):
        scores, c0 = cand_scores(w, dev, s)
        if scores is None:
            continue
        scores = scores[:top_k]
        per_sent.append((scores, dev["counts"][c0:c0 + len(scores)]))

    print(f"{'tau':>6} {'devF1':>8} {'overrides':>9} {'improved':>8} {'worsened':>8}")
    for tau in [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]:
        agg = np.zeros(3, np.int64)
        overrides = improved = worsened = 0
        for scores, cnt in per_sent:
            pick = int(np.argmax(scores))
            if pick != 0 and scores[pick] - scores[0] > tau:
                agg += cnt[pick]
                overrides += 1
                a, b = sent_f1(*cnt[pick]), sent_f1(*cnt[0])
                if a > b + 1e-9:
                    improved += 1
                elif a < b - 1e-9:
                    worsened += 1
            else:
                agg += cnt[0]
        tp, fp, fn = agg
        f1 = 2 * tp / (2 * tp + fp + fn)
        print(f"{tau:>6.1f} {f1:>8.4f} {overrides:>9} {improved:>8} {worsened:>8}")


if __name__ == "__main__":
    main()
