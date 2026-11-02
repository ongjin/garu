"""Section 18 적용 여부를 정하는 구어/문어 판별기.

문장(입력 전체)의 문자 1~3-gram을 FNV-1a로 해시해 나이브 베이즈 로그비를 더한다.
점수 > 임계값이면 구어로 보고 탐색 비용 보정을 적용한다. Rust 구현은
crates/garu-core/src/search.rs `Router`이며 해시·n-gram 규칙이 같아야 한다.

    python3 training/search/router.py TRAIN.jsonl DEV.jsonl OUT.npz [--dim-log2 15]

TRAIN/DEV는 training/ambiguity/prepare.py 형식({text, src}). src=SX가 구어다.
"""
import argparse
import json

import numpy as np

FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
U64 = (1 << 64) - 1
I8_MAX = 127
# Rust char::is_whitespace와 같은 집합 (str.strip 기본값은 U+001C–U+001F도 지운다).
RUST_WHITESPACE = "".join(map(chr, [0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20, 0x85, 0xA0, 0x1680,
                                    *range(0x2000, 0x200B), 0x2028, 0x2029, 0x202F, 0x205F,
                                    0x3000]))


def fnv(data):
    h = FNV_OFFSET
    for b in data:
        h ^= b
        h = (h * FNV_PRIME) & U64
    return h


def features(text, dim_log2):
    mask = (1 << dim_log2) - 1
    s = "^" + text.strip(RUST_WHITESPACE) + "$"
    out = set()
    for n in (1, 2, 3):
        for i in range(len(s) - n + 1):
            out.add(fnv(bytes([n]) + s[i:i + n].encode("utf-8")) & mask)
    return sorted(out)


def load(path, dim_log2):
    X, y = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            X.append(features(row["text"], dim_log2))
            y.append(1 if row["src"] == "SX" else 0)
    return X, np.array(y)


def train(X, y, dim_log2, alpha=1.0):
    counts = np.zeros((2, 1 << dim_log2))
    for f, label in zip(X, y):
        counts[label, f] += 1
    spoken = (counts[1] + alpha) / (counts[1].sum() + alpha * counts.shape[1])
    written = (counts[0] + alpha) / (counts[0].sum() + alpha * counts.shape[1])
    ratio = np.log(spoken / written)
    qscale = float(np.abs(ratio).max()) / I8_MAX
    quantized = np.round(ratio / qscale).astype(np.int8)
    bias = float(np.log(y.mean() / (1 - y.mean())))
    return quantized, qscale, bias


def score(text, quantized, qscale, bias, dim_log2):
    """Rust와 같게: 정수 합 × f32 qscale + f32 bias (f64 연산)."""
    total = int(quantized[features(text, dim_log2)].astype(np.int64).sum())
    return total * float(np.float32(qscale)) + float(np.float32(bias))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train")
    ap.add_argument("dev")
    ap.add_argument("out")
    ap.add_argument("--dim-log2", type=int, default=15)
    args = ap.parse_args()
    X, y = load(args.train, args.dim_log2)
    quantized, qscale, bias = train(X, y, args.dim_log2)
    Xd, yd = load(args.dev, args.dim_log2)
    q, b = float(np.float32(qscale)), float(np.float32(bias))
    pred = np.array([int(quantized[f].astype(np.int64).sum()) * q + b > 0 for f in Xd])
    print(f"dev acc {(pred == yd).mean():.4f} spoken recall {pred[yd == 1].mean():.4f} "
          f"written→spoken {pred[yd == 0].mean():.4f}")
    np.savez(args.out, quantized=quantized, qscale=qscale, bias=bias, dim_log2=args.dim_log2)


if __name__ == "__main__":
    main()
