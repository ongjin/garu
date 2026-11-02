"""search_train 가중치(bucket\\tweight) → GMDL Section 18.

    python3 training/search/export.py WEIGHTS.tsv IN.gmdl OUT.gmdl --scale 0.5 [--top N]
        [--router ROUTER.npz --threshold 0] [--blob B]

포맷은 crates/garu-core/src/search.rs `SearchCost::from_bytes`와 같다:
[GSCW\\x01][dim_log2 u8][scale f32][count u32][qscale f32][bucket 차분 LEB128][i16 가중치].
`--top N`은 절댓값이 큰 N개만 남긴다(크기 조절). 양자화 후 0이 되는 가중치는 버린다.
`--router`(training/search/router.py 산출물)를 주면 v2(`GSCW\x02`)로 판별기를 덧붙여
구어로 판별된 입력에만 보정을 적용한다.
"""
import argparse
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ambiguity"))
from context_cache import sections  # noqa: E402

MAGIC = b"GSCW\x01"
MAGIC_GATED = b"GSCW\x02"
DIM_LOG2 = 20
I16_MAX = 32767


def varint(n):
    out = bytearray()
    while True:
        chunk = n & 0x7F
        n >>= 7
        out.append(chunk | 0x80 if n else chunk)
        if not n:
            return bytes(out)


def encode_router(router_npz, threshold):
    import numpy as np
    z = np.load(router_npz)
    quantized = z["quantized"].astype(np.int8)
    dim_log2 = int(z["dim_log2"])
    assert len(quantized) == 1 << dim_log2
    return (struct.pack("<Bfff", dim_log2, float(z["qscale"]), float(z["bias"]), threshold)
            + quantized.tobytes())


def encode(weights, scale):
    peak = max((abs(w) for _, w in weights), default=0.0)
    qscale = peak / I16_MAX if peak else 1.0
    kept = [(b, round(w / qscale)) for b, w in sorted(weights)]
    kept = [(b, q) for b, q in kept if q]
    out = bytearray(MAGIC + struct.pack("<BfIf", DIM_LOG2, scale, len(kept), qscale))
    prev = 0
    for b, _ in kept:
        out += varint(b - prev)
        prev = b
    out += b"".join(struct.pack("<h", q) for _, q in kept)
    return bytes(out)


def read_weights(path):
    rows = []
    with open(path) as f:
        for line in f:
            bucket, weight = line.split("\t")
            rows.append((int(bucket), float(weight)))
    return rows


def export(weights_path, in_model, out_model, scale, blob_path=None, top=None,
           router=None, threshold=0.0):
    import brotli
    weights = read_weights(weights_path)
    if top:
        weights = sorted(weights, key=lambda bw: (-abs(bw[1]), bw[0]))[:top]
    blob = encode(weights, scale)
    if router:
        blob = MAGIC_GATED + blob[len(MAGIC):] + encode_router(router, threshold)
    parts = sections(Path(in_model).read_bytes())
    parts[18] = blob
    raw = b"GMDL" + struct.pack("<I", 3) + b"".join(
        struct.pack("<BI", kind, len(data)) + data for kind, data in parts.items())
    Path(out_model).write_bytes(brotli.compress(raw, quality=11))
    if blob_path:
        Path(blob_path).write_bytes(blob)
    return len(blob), Path(out_model).stat().st_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("in_model")
    ap.add_argument("out_model")
    ap.add_argument("--scale", type=float, required=True)
    ap.add_argument("--top", type=int)
    ap.add_argument("--router")
    ap.add_argument("--threshold", type=float, default=0.0)
    ap.add_argument("--blob")
    args = ap.parse_args()
    raw, model = export(args.weights, args.in_model, args.out_model, args.scale, args.blob, args.top,
                        args.router, args.threshold)
    print(f"section18: {raw:,}B raw, model {model:,}B brotli")


if __name__ == "__main__":
    main()
