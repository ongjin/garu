"""weights.npz → GMDL Section 14 주입 패처.

usage: export_weights.py <weights.npz> <in.gmdl> <out.gmdl> [--margin 4.0]

Section 14 포맷 (crates/garu-core/src/rerank.rs와 동일):
  [ver u8=2][dim_log2 u8][margin f32][count u32][scale f32]
  [bucket 차분 varint × count][양자화 가중치 i16 × count]
bucket 차분 + int16 양자화(step = max|w|/32767)로 ver=1 대비 모델 -220KB.
입력 gmdl은 raw/brotli 모두 허용, 출력은 brotli q=11 (기존 section 14는 교체).
"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import DIM  # noqa: E402


def varint(n):
    out = bytearray()
    while True:
        chunk = n & 0x7F
        n >>= 7
        out.append(chunk | 0x80 if n else chunk)
        if not n:
            return bytes(out)


def pack_section14(buckets, weights, dim_log2, margin):
    """오름차순 bucket + 가중치 → ver=2 blob."""
    buckets = np.asarray(buckets, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float32)
    assert len(buckets) == len(weights)
    assert np.all(np.diff(buckets) > 0), "bucket은 강한 오름차순이어야 함"
    peak = float(np.abs(weights).max()) if len(weights) else 0.0
    scale = peak / 32767.0 if peak > 0 else 1.0
    blob = bytearray(struct.pack("<BBfIf", 2, dim_log2, margin, len(buckets), scale))
    deltas = np.diff(np.concatenate([[0], buckets]))
    blob += b"".join(varint(int(d)) for d in deltas)
    blob += np.round(weights / scale).astype("<i2").tobytes()
    return bytes(blob)


def build_section14(npz_path, margin):
    w = np.load(npz_path)["w"]
    assert len(w) == DIM
    nz = np.nonzero(w)[0]  # np.nonzero는 오름차순
    return pack_section14(nz, w[nz], DIM.bit_length() - 1, margin)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", type=Path)
    ap.add_argument("in_gmdl", type=Path)
    ap.add_argument("out_gmdl", type=Path)
    ap.add_argument("--margin", type=float, default=4.0)
    ap.add_argument("--blob", type=Path, default=None,
                    help="Section 14 raw blob도 저장 (build_codebook_model passthrough용)")
    args = ap.parse_args()

    raw = args.in_gmdl.read_bytes()
    if raw[:4] != b"GMDL":
        import brotli
        raw = brotli.decompress(raw)
    assert raw[:4] == b"GMDL" and struct.unpack("<I", raw[4:8])[0] == 3

    out = bytearray(raw[:8])
    pos = 8
    while pos < len(raw):
        stype = raw[pos]
        slen = struct.unpack("<I", raw[pos + 1:pos + 5])[0]
        if stype != 14:  # 기존 section 14는 버리고 교체
            out += raw[pos:pos + 5 + slen]
        pos += 5 + slen

    sec = build_section14(args.weights, args.margin)
    if args.blob:
        args.blob.write_bytes(sec)
    out += struct.pack("<BI", 14, len(sec)) + sec

    import brotli
    compressed = brotli.compress(bytes(out), quality=11)
    args.out_gmdl.write_bytes(compressed)
    print(f"section14: {len(sec):,}B ({(len(sec))/1e6:.2f}MB raw), "
          f"model {len(raw):,}→{len(out):,}B raw, {len(compressed):,}B brotli")


if __name__ == "__main__":
    main()
