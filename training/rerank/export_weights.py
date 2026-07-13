"""weights.npz → GMDL Section 14 주입 패처.

usage: export_weights.py <weights.npz> <in.gmdl> <out.gmdl> [--margin 4.0]

Section 14 포맷 (crates/garu-core/src/rerank.rs와 동일):
  [ver u8=1][dim_log2 u8][margin f32][count u32][(bucket u32, weight f32)×count]
입력 gmdl은 raw/brotli 모두 허용, 출력은 brotli q=11 (기존 section 14는 교체).
"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import DIM  # noqa: E402


def build_section14(npz_path, margin):
    w = np.load(npz_path)["w"]
    assert len(w) == DIM
    nz = np.nonzero(w)[0]
    blob = bytearray(struct.pack("<BBfI", 1, DIM.bit_length() - 1, margin, len(nz)))
    for b in nz:  # np.nonzero는 오름차순
        blob += struct.pack("<If", int(b), float(w[b]))
    return bytes(blob)


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
