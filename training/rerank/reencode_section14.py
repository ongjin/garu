"""기존 Section 14 blob(ver=1)을 ver=2로 재인코딩하고 gmdl에 주입.

weights.npz 없이도 동작한다 — blob 자체가 완전한 sparse 가중치이기 때문.
가중치는 int16 양자화(step = max|w|/32767)되므로 비트 동일은 아니지만,
9,000문장 재채점 실측에서 재순위 argmax 플립 0건이었다.

usage: reencode_section14.py <section14.bin> <gmdl> [<gmdl> ...]
  blob은 제자리에서 ver=2로 교체되고, 지정한 gmdl들의 section 14도 함께 갱신된다.
"""
import struct
import sys
from pathlib import Path

import brotli
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_weights import pack_section14  # noqa: E402


def unpack_v1(data):
    ver, dim_log2 = data[0], data[1]
    if ver != 1:
        raise SystemExit(f"ver=1 blob이 아님 (ver={ver})")
    margin = struct.unpack("<f", data[2:6])[0]
    count = struct.unpack("<I", data[6:10])[0]
    arr = np.frombuffer(data[10:10 + count * 8],
                        dtype=np.dtype([("b", "<u4"), ("w", "<f4")]))
    return arr["b"], arr["w"], dim_log2, margin


def patch_gmdl(path, section):
    raw = path.read_bytes()
    compressed = raw[:4] != b"GMDL"
    if compressed:
        raw = brotli.decompress(raw)
    assert raw[:4] == b"GMDL" and struct.unpack("<I", raw[4:8])[0] == 3

    out = bytearray(raw[:8])
    pos = 8
    while pos < len(raw):
        stype = raw[pos]
        slen = struct.unpack("<I", raw[pos + 1:pos + 5])[0]
        if stype != 14:
            out += raw[pos:pos + 5 + slen]
        pos += 5 + slen
    out += struct.pack("<BI", 14, len(section)) + section

    before = len(path.read_bytes())
    data = brotli.compress(bytes(out), quality=11) if compressed else bytes(out)
    path.write_bytes(data)
    print(f"  {path}: {before:,} → {len(data):,}B ({len(data) - before:+,})")


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    blob_path = Path(sys.argv[1])
    buckets, weights, dim_log2, margin = unpack_v1(blob_path.read_bytes())
    section = pack_section14(buckets, weights, dim_log2, margin)
    print(f"section14: ver1 {10 + len(buckets) * 8:,}B → ver2 {len(section):,}B "
          f"({len(buckets):,} features, margin {margin})")
    blob_path.write_bytes(section)
    for p in sys.argv[2:]:
        patch_gmdl(Path(p), section)


if __name__ == "__main__":
    main()
