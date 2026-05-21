"""In-place patching: 구어 over-segmentation 후보 어절을 기존 캐시에 추가.

기존 eojeol_cache.bin을 읽어 파싱, 새 어절 추가, 재직렬화.
기존 엔트리는 절대 수정하지 않음 (큐레이션 보존).

캐시 바이너리 포맷 (v2, sub-version=2):
  [u32 marker=0xFFFFFFFF]
  [u8 sub_version=2]
  [u32 string_table_len]
  [string_table bytes]
  [u16 num_strings]
  [u32 × (num_strings+1) string offsets]
  [u32 num_entries]
  for each entry:
    [u8 eojeol_len][eojeol bytes][u8 num_morphs]
    for each morph:
      [u16 form_idx][u8 pos_byte]
"""
import json, os, struct

BASE = os.path.dirname(os.path.abspath(__file__))
CACHE_BIN = os.path.join(BASE, "codebook_data/eojeol_cache.bin")
CANDIDATES = os.path.join(BASE, "codebook_data/guuh_overseg_candidates.jsonl")

POS_TAGS = [
    "NNG", "NNP", "NNB", "NR", "NP",
    "VV", "VA", "VX", "VCP", "VCN",
    "MAG", "MAJ", "MM", "IC",
    "JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ", "JX", "JC",
    "EP", "EF", "EC", "ETN", "ETM",
    "XPN", "XSN", "XSV", "XSA", "XR",
    "SF", "SP", "SS", "SE", "SO", "SW", "SH", "SL", "SN",
]
POS_TO_BYTE = {p: i for i, p in enumerate(POS_TAGS)}
BYTE_TO_POS = {i: p for i, p in enumerate(POS_TAGS)}


def parse_cache(data):
    """Parse eojeol_cache.bin (v2 format) → (forms_list, entries).
    entries: list of (eojeol_str, [(form_str, pos_str), ...])
    """
    assert len(data) >= 5
    marker = struct.unpack_from("<I", data, 0)[0]
    assert marker == 0xFFFFFFFF, f"Expected marker 0xFFFFFFFF, got 0x{marker:08X}"
    sub_version = data[4]
    pos = 5

    # String table
    st_len = struct.unpack_from("<I", data, pos)[0]
    pos += 4
    string_table = data[pos:pos + st_len]
    pos += st_len

    num_strings = struct.unpack_from("<H", data, pos)[0]
    pos += 2

    string_offsets = []
    if sub_version >= 2:
        for _ in range(num_strings + 1):
            off = struct.unpack_from("<I", data, pos)[0]
            string_offsets.append(off)
            pos += 4
    else:
        for _ in range(num_strings + 1):
            off = struct.unpack_from("<H", data, pos)[0]
            string_offsets.append(off)
            pos += 2

    forms = []
    for i in range(num_strings):
        s = string_table[string_offsets[i]:string_offsets[i + 1]].decode("utf-8")
        forms.append(s)

    # Entries
    num_entries = struct.unpack_from("<I", data, pos)[0]
    pos += 4

    entries = []
    for _ in range(num_entries):
        elen = data[pos]
        pos += 1
        eojeol = data[pos:pos + elen].decode("utf-8")
        pos += elen
        nm = data[pos]
        pos += 1
        morphs = []
        for _ in range(nm):
            form_idx = struct.unpack_from("<H", data, pos)[0]
            pos += 2
            pos_byte = data[pos]
            pos += 1
            morphs.append((forms[form_idx], BYTE_TO_POS.get(pos_byte, "SW")))
        entries.append((eojeol, morphs))

    return forms, entries


def serialize_cache(entries):
    """Serialize entries list → bytes (v2 format).
    entries: list of (eojeol_str, [(form_str, pos_str), ...])
    """
    # Build string table from all forms
    all_forms = set()
    for _, morphs in entries:
        for form, _ in morphs:
            all_forms.add(form)
    sorted_forms = sorted(all_forms, key=lambda s: s.encode("utf-8"))
    form_to_index = {f: i for i, f in enumerate(sorted_forms)}

    string_table = bytearray()
    string_offsets = []
    for form in sorted_forms:
        string_offsets.append(len(string_table))
        string_table.extend(form.encode("utf-8"))
    string_offsets.append(len(string_table))  # sentinel

    buf = bytearray()
    buf.extend(struct.pack("<I", 0xFFFFFFFF))  # marker
    buf.extend(struct.pack("B", 2))            # sub-version 2

    buf.extend(struct.pack("<I", len(string_table)))
    buf.extend(string_table)
    buf.extend(struct.pack("<H", len(sorted_forms)))
    for off in string_offsets:
        buf.extend(struct.pack("<I", off))

    buf.extend(struct.pack("<I", len(entries)))
    for eojeol, morphs in entries:
        eojeol_bytes = eojeol.encode("utf-8")
        buf.extend(struct.pack("B", len(eojeol_bytes)))
        buf.extend(eojeol_bytes)
        buf.extend(struct.pack("B", len(morphs)))
        for form, pos_str in morphs:
            buf.extend(struct.pack("<H", form_to_index[form]))
            buf.extend(struct.pack("B", POS_TO_BYTE.get(pos_str, 0)))

    return bytes(buf)


def main():
    print(f"Reading cache: {CACHE_BIN}")
    with open(CACHE_BIN, "rb") as f:
        data = f.read()
    size_before = len(data)
    print(f"  Size: {size_before:,} bytes")

    forms, entries = parse_cache(data)
    print(f"  Entries: {len(entries)}")
    print(f"  String table forms: {len(forms)}")

    # Existing eojeol set
    existing_eojeols = {ej for ej, _ in entries}
    print(f"  Existing unique eojeols: {len(existing_eojeols)}")

    # Load candidates
    print(f"\nReading candidates: {CANDIDATES}")
    candidates = []
    with open(CANDIDATES) as f:
        for line in f:
            candidates.append(json.loads(line))
    print(f"  Candidates: {len(candidates)}")

    # Add new entries
    added = 0
    skipped = 0
    for cand in candidates:
        ej = cand["eojeol"]
        gold = [(m[0], m[1]) for m in cand["gold_analysis"]]
        if ej in existing_eojeols:
            print(f"  SKIP (already in cache): {ej}")
            skipped += 1
            continue
        # Validate POS tags
        valid = True
        for form, pos_str in gold:
            if pos_str not in POS_TO_BYTE:
                print(f"  SKIP (unknown POS {pos_str}): {ej}")
                valid = False
                break
        if not valid:
            skipped += 1
            continue
        entries.append((ej, gold))
        existing_eojeols.add(ej)
        gold_str = " + ".join(f"{f}/{p}" for f, p in gold)
        print(f"  ADD: {ej:20s} (freq={cand['freq']}) → {gold_str}")
        added += 1

    print(f"\nAdded: {added}, Skipped: {skipped}")

    if added == 0:
        print("Nothing to add, exiting.")
        return

    # Reserialize
    new_data = serialize_cache(entries)
    size_after = len(new_data)
    print(f"\nSize: {size_before:,} → {size_after:,} bytes (delta: {size_after - size_before:+,})")

    with open(CACHE_BIN, "wb") as f:
        f.write(new_data)
    print(f"Written: {CACHE_BIN}")

    # Verify round-trip
    forms2, entries2 = parse_cache(new_data)
    assert len(entries2) == len(entries), f"Round-trip mismatch: {len(entries2)} != {len(entries)}"
    print(f"Round-trip OK: {len(entries2)} entries")


if __name__ == "__main__":
    main()
