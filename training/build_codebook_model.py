"""Build GMDL v3 codebook model from extracted data.

Reads codebook_data/ and produces models/codebook.gmdl in GMDL v3 binary format.

Usage:
    python training/build_codebook_model.py
"""
import json
import math
import struct
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "training" / "codebook_data"
OUT_PATH = ROOT / "models" / "codebook.gmdl"

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
NUM_POS = len(POS_TAGS)  # 42


def write_section(buf: bytearray, section_type: int, data: bytes):
    """Append a section header (type u8 + len u32) and data to buf."""
    buf.extend(struct.pack("B", section_type))
    buf.extend(struct.pack("<I", len(data)))
    buf.extend(data)


def pos_byte(tag: str) -> int:
    """Map POS tag string to byte. Falls back to NNP(1) for unknown tags."""
    return POS_TO_BYTE.get(tag, 1)


def build_content_dict_v1(dict_path: Path) -> bytes:
    """Build Dict v1 (Trie serialization) format from content_dict.txt.

    Format matches Rust Dict::to_bytes() / Dict::from_bytes_v1():
      GARU [4B] + version=1 [u32] + word_count [u32]
      For each word (sorted UTF-8):
        word_len [u16] + word_bytes + num_entries [u8]=1
        num_morphemes [u8]=1 + text_len [u16] + text_bytes + pos [u8] + score [f32]
    """
    # Parse content dict: group entries by word
    word_entries = {}  # {word: [(tag, freq), ...]}
    max_freq = 0
    with open(dict_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            word, tag, freq_str = parts[0], parts[1], parts[2]
            freq = int(freq_str)
            if freq > max_freq:
                max_freq = freq
            if word not in word_entries:
                word_entries[word] = []
            word_entries[word].append((tag, freq))

    if max_freq == 0:
        max_freq = 1

    # Sort by UTF-8 bytes for reproducibility
    sorted_words = sorted(word_entries.keys(), key=lambda w: w.encode("utf-8"))

    buf = bytearray()
    buf.extend(b"GARU")
    buf.extend(struct.pack("<I", 1))  # version 1
    buf.extend(struct.pack("<I", len(sorted_words)))

    for word in sorted_words:
        entries = word_entries[word]
        word_bytes = word.encode("utf-8")
        buf.extend(struct.pack("<H", len(word_bytes)))
        buf.extend(word_bytes)
        buf.extend(struct.pack("B", len(entries)))  # num_entries

        for tag, freq in entries:
            buf.extend(struct.pack("B", 1))  # num_morphemes = 1
            # morpheme text = word itself
            buf.extend(struct.pack("<H", len(word_bytes)))
            buf.extend(word_bytes)
            buf.extend(struct.pack("B", pos_byte(tag)))
            # score = -log(freq / max_freq), clamped >= 0
            score = -math.log(max(freq, 1) / max_freq) if freq < max_freq else 0.0
            buf.extend(struct.pack("<f", score))

    return bytes(buf), max_freq


def build_suffix_codebook(codebook_path: Path) -> tuple[bytes, int]:
    """Build Section 7: Suffix Codebook.

    Format:
      num_entries [u32]
      For each entry (sorted by surface UTF-8):
        surface_len [u16] + surface_bytes
        num_analyses [u16]
        For each analysis:
          freq [u32] + num_morphemes [u8]
          For each morpheme: form_len [u16] + form_bytes + pos_byte [u8]
    """
    with open(codebook_path, "r", encoding="utf-8") as f:
        codebook = json.load(f)

    # Sort entries by surface UTF-8
    sorted_surfaces = sorted(codebook.keys(), key=lambda s: s.encode("utf-8"))

    max_suffix_freq = 0
    buf = bytearray()
    buf.extend(struct.pack("<I", len(sorted_surfaces)))

    for surface in sorted_surfaces:
        analyses = codebook[surface]
        surface_bytes = surface.encode("utf-8")
        buf.extend(struct.pack("<H", len(surface_bytes)))
        buf.extend(surface_bytes)
        buf.extend(struct.pack("<H", len(analyses)))

        for analysis in analyses:
            freq = analysis["freq"]
            if freq > max_suffix_freq:
                max_suffix_freq = freq
            morphemes = analysis["morphemes"]
            buf.extend(struct.pack("<I", freq))
            buf.extend(struct.pack("B", len(morphemes)))
            for form, tag in morphemes:
                form_bytes = form.encode("utf-8")
                buf.extend(struct.pack("<H", len(form_bytes)))
                buf.extend(form_bytes)
                buf.extend(struct.pack("B", pos_byte(tag)))

    return bytes(buf), max_suffix_freq


def build_trigram_costs(costs_path: Path) -> bytes:
    """Build Section 8: Dense trigram cost table.

    Format:
      num_pos [u32] = 42
      default_cost [f32]
      trigram_data [42*42*42 f32] — 0.0 means use bigram backoff
      bigram_data [42*42 f32] — default_cost for missing bigrams
    """
    with open(costs_path, "r", encoding="utf-8") as f:
        costs = json.load(f)

    default_cost = costs.get("default_cost", 15.0)
    trigrams = costs.get("trigram", {})
    bigrams = costs.get("bigram", {})

    # Initialize arrays
    trigram_data = [0.0] * (NUM_POS * NUM_POS * NUM_POS)
    bigram_data = [default_cost] * (NUM_POS * NUM_POS)

    # Fill bigrams
    for key, cost in bigrams.items():
        parts = key.split(",")
        if len(parts) != 2:
            continue
        i = POS_TO_BYTE.get(parts[0].strip())
        j = POS_TO_BYTE.get(parts[1].strip())
        if i is not None and j is not None:
            bigram_data[i * NUM_POS + j] = cost

    # Fill trigrams
    for key, cost in trigrams.items():
        parts = key.split(",")
        if len(parts) != 3:
            continue
        i = POS_TO_BYTE.get(parts[0].strip())
        j = POS_TO_BYTE.get(parts[1].strip())
        k = POS_TO_BYTE.get(parts[2].strip())
        if i is not None and j is not None and k is not None:
            trigram_data[i * NUM_POS * NUM_POS + j * NUM_POS + k] = cost

    buf = bytearray()
    buf.extend(struct.pack("<I", NUM_POS))
    buf.extend(struct.pack("<f", default_cost))
    for v in trigram_data:
        buf.extend(struct.pack("<f", v))
    for v in bigram_data:
        buf.extend(struct.pack("<f", v))

    return bytes(buf)


def build_word_frequencies(max_freq: int, max_suffix_freq: int) -> bytes:
    """Build Section 9: Word frequency metadata."""
    buf = bytearray()
    buf.extend(struct.pack("<I", max_freq))
    buf.extend(struct.pack("<I", max_suffix_freq))
    return bytes(buf)


def build_analyzer_params() -> bytes:
    """Build Section 10: Analyzer parameters."""
    buf = bytearray()
    buf.extend(struct.pack("<f", 3.0))   # morpheme_penalty
    buf.extend(struct.pack("<f", 15.0))  # oov_penalty
    buf.extend(struct.pack("<f", 2.5))   # length_bonus
    buf.extend(struct.pack("<f", 4.0))   # single_char_content_penalty
    return bytes(buf)


def main():
    print("Building GMDL v3 codebook model...")
    print(f"  Input:  {DATA_DIR}")
    print(f"  Output: {OUT_PATH}")
    print()

    # Section 6: Content dict (Dict v1 format)
    print("Building content dict (Dict v1)...")
    dict_data, max_freq = build_content_dict_v1(DATA_DIR / "content_dict.txt")
    print(f"  Content dict: {len(dict_data):,} bytes, max_freq={max_freq}")

    # Section 7: Suffix codebook
    print("Building suffix codebook...")
    codebook_data, max_suffix_freq = build_suffix_codebook(DATA_DIR / "suffix_codebook.json")
    print(f"  Suffix codebook: {len(codebook_data):,} bytes, max_suffix_freq={max_suffix_freq}")

    # Section 8: Trigram cost table
    print("Building trigram cost table...")
    trigram_data = build_trigram_costs(DATA_DIR / "trigram_costs.json")
    print(f"  Trigram costs: {len(trigram_data):,} bytes")

    # Section 9: Word frequencies
    freq_data = build_word_frequencies(max_freq, max_suffix_freq)

    # Section 10: Analyzer parameters
    params_data = build_analyzer_params()

    # Assemble GMDL v3
    print()
    print("Assembling GMDL v3...")
    buf = bytearray()
    buf.extend(b"GMDL")
    buf.extend(struct.pack("<I", 3))  # version 3

    write_section(buf, 6, dict_data)
    write_section(buf, 7, codebook_data)
    write_section(buf, 8, trigram_data)
    write_section(buf, 9, freq_data)
    write_section(buf, 10, params_data)

    # Write output
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "wb") as f:
        f.write(buf)

    total = len(buf)
    print()
    print("=== Section Size Report ===")
    print(f"  Header:              8 bytes")
    print(f"  Section 6 (dict):    {len(dict_data):>10,} bytes  ({len(dict_data)/total*100:.1f}%)")
    print(f"  Section 7 (suffix):  {len(codebook_data):>10,} bytes  ({len(codebook_data)/total*100:.1f}%)")
    print(f"  Section 8 (trigram): {len(trigram_data):>10,} bytes  ({len(trigram_data)/total*100:.1f}%)")
    print(f"  Section 9 (freq):    {len(freq_data):>10,} bytes  ({len(freq_data)/total*100:.1f}%)")
    print(f"  Section 10 (params): {len(params_data):>10,} bytes  ({len(params_data)/total*100:.1f}%)")
    print(f"  Section headers:     {5 * 5:>10,} bytes")
    print(f"  ---")
    print(f"  Total:               {total:>10,} bytes  ({total/1024/1024:.2f} MB)")
    print()
    print(f"Output written to: {OUT_PATH}")


if __name__ == "__main__":
    main()
