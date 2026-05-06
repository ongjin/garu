"""Append manual word-bigram rules to word_bigrams.bin.

Rules derived from research on ㄹ+의존명사 patterns (갈만한데/볼만한데/갈만해 family).
These penalize specific (word, prev_pos, target_pos) combinations that cause
wrong winners in N-best Viterbi.

Idempotent: checks for existing rules with same (word, prev_pos, target_pos)
and skips duplicates.
"""
import struct
from pathlib import Path

ROOT = Path(__file__).parent.parent
BIN = ROOT / "training" / "codebook_data" / "word_bigrams.bin"

POS_TAGS = [
    "NNG", "NNP", "NNB", "NR", "NP",
    "VV", "VA", "VX", "VCP", "VCN",
    "MAG", "MAJ", "MM", "IC",
    "JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ", "JX", "JC",
    "EP", "EF", "EC", "ETN", "ETM",
    "XPN", "XSN", "XSV", "XSA", "XR",
    "SF", "SP", "SS", "SE", "SO", "SW", "SH", "SL", "SN",
]
P = {p: i for i, p in enumerate(POS_TAGS)}
BOS = 255

PENALTY = 5.0  # cost units

MANUAL_RULES = [
    ("한데", P["JX"],  P["NNG"], +PENALTY),  # 볼/올/들+만/JX+한데/NNG → penalize
    ("한데", P["ETM"], P["NNG"], +PENALTY),  # ETM 직후 한데/NNG 차단
    ("만해", P["ETM"], P["NNP"], +PENALTY),  # ㄹ/ETM+만해/NNP(한용운 필명) 차단
    ("갈마", BOS,      P["NNP"], +PENALTY),  # 갈만한데의 갈마/NNP 인명 차단
    ("있",   P["NNB"], P["VV"],  +PENALTY),  # 수/것/리/NNB + 있/VV → 있/VX 유도
    ("들",   BOS,      P["XSN"], +PENALTY),  # 문두 들/XSN 차단: 들만한데 → 들/VV+ㄹ
    ("없",   P["VV"],  P["VA"],  +3.0),      # 갈리/VV+없/VA보다 ㄹ+리/NNB+없 선호
    # 방향부사 뒤 "가" → VV 유도 (저리가, 빨리가 etc.)
    ("가",   P["MAG"], P["JKS"], +PENALTY),  # MAG+가/JKS 차단 → VV 선호
    # 방향부사 뒤 "와" → JKB/JC 차단 (이리와, 빨리와 etc.)
    ("와",   P["MAG"], P["JKB"], +PENALTY),  # MAG+와/JKB 차단
    ("와",   P["MAG"], P["JC"],  +PENALTY),  # MAG+와/JC 차단
]


def parse(data):
    if len(data) < 4:
        return []
    n = int.from_bytes(data[:4], "little")
    out = []
    pos = 4
    for _ in range(n):
        wlen = int.from_bytes(data[pos:pos+2], "little"); pos += 2
        word = data[pos:pos+wlen].decode("utf-8"); pos += wlen
        prev_pb = data[pos]; target_pb = data[pos+1]; bq = struct.unpack("b", data[pos+2:pos+3])[0]
        pos += 3
        out.append((word, prev_pb, target_pb, bq))
    return out


def encode(entries):
    buf = bytearray()
    buf.extend(struct.pack("<I", len(entries)))
    for word, pp, tp, bq in entries:
        wb = word.encode("utf-8")
        buf.extend(struct.pack("<H", len(wb)))
        buf.extend(wb)
        buf.extend(struct.pack("B", pp))
        buf.extend(struct.pack("B", tp))
        buf.extend(struct.pack("b", bq))
    return bytes(buf)


def main():
    data = BIN.read_bytes()
    entries = parse(data)
    existing_keys = {(w, pp, tp) for w, pp, tp, _ in entries}
    before = len(entries)

    added = 0
    for word, prev_pos, target_pos, pen in MANUAL_RULES:
        key = (word, prev_pos, target_pos)
        # Convert penalty to storage: runtime bonus = bq / 25.0, positive = cost up.
        # Script convention (build_context_bigrams.py): bq = int(-bonus * 25), i.e.,
        # script "bonus" variable = preference (positive = preferred = cheaper).
        # For a penalty, we want runtime bonus > 0, so bq > 0, so script bonus < 0.
        # Direct encode: bq = int(pen * 25), clamped to i8.
        bq = max(-128, min(127, int(pen * 25)))
        if key in existing_keys:
            # Update existing entry to our penalty (keep rule list clean)
            for i, (w, pp, tp, old_bq) in enumerate(entries):
                if (w, pp, tp) == key:
                    entries[i] = (w, pp, tp, bq)
                    break
            print(f"  UPDATE: {word:<6} prev={POS_TAGS[prev_pos] if prev_pos<len(POS_TAGS) else 'BOS'} target={POS_TAGS[target_pos]} bq={bq} (runtime={bq/25:.2f})")
        else:
            entries.append((word, prev_pos, target_pos, bq))
            added += 1
            print(f"  ADD:    {word:<6} prev={POS_TAGS[prev_pos] if prev_pos<len(POS_TAGS) else 'BOS'} target={POS_TAGS[target_pos]} bq={bq} (runtime={bq/25:.2f})")

    out = encode(entries)
    BIN.write_bytes(out)
    print(f"\nBefore: {before} entries")
    print(f"After:  {len(entries)} entries (+{added} new)")
    print(f"Wrote {BIN}")


if __name__ == "__main__":
    main()
