"""Build GMDL v3 codebook model from extracted data.

Reads codebook_data/ and produces models/codebook.gmdl in GMDL v3 binary format.

Usage:
    python training/build_codebook_model.py
"""
import gzip
import json
import math
import struct
import subprocess
import tempfile
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


def build_content_dict_fst(dict_path: Path) -> tuple[bytes, int]:
    """Build Dict v2 (FST) format from content_dict.txt using the Rust build-dict tool.

    Reads content_dict.txt (word, pos_tag, freq), keeps highest-freq POS per word,
    shells out to `cargo run --release --bin build-dict` to produce FST binary.
    Returns (dict_bytes, max_freq).
    """
    # Parse content dict: keep highest-freq POS per word, filter by min freq
    MIN_CONTENT_FREQ = 5
    best = {}  # {word: (tag, freq)}
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
            if freq < MIN_CONTENT_FREQ:
                continue
            if freq > max_freq:
                max_freq = freq
            if word not in best or freq > best[word][1]:
                best[word] = (tag, freq)

    if max_freq == 0:
        max_freq = 1

    # Load wiki NNP titles
    wiki_path = ROOT / "training" / "kowiki-titles.gz"
    nnp_byte = pos_byte("NNP")  # = 1
    wiki_count = 0
    if wiki_path.exists():
        with gzip.open(wiki_path, "rt", encoding="utf-8") as f:
            for line in f:
                title = line.strip()
                if not title or title == "page_title":
                    continue
                # Clean: replace underscores with spaces, then take as-is
                title = title.replace("_", " ")
                # Skip if too long or too short
                if len(title) < 2 or len(title) > 50:
                    continue
                # Only keep pure-ASCII titles (English proper nouns)
                # Mixed Korean+English titles bloat the FST significantly
                if not all(c.isascii() for c in title):
                    continue
                if not any(c.isalpha() for c in title):
                    continue
                # Skip titles with problematic chars
                if any(c in title for c in '\t\n\r\x00'):
                    continue
                # Don't overwrite content dict entries (they have better POS)
                if title not in best:
                    best[title] = ("NNP", 1)
                    wiki_count += 1
        print(f"  Wiki NNP entries added: {wiki_count:,}")
    else:
        print(f"  Warning: {wiki_path} not found, skipping wiki NNP")

    # Sort by UTF-8 bytes and write temp input for build-dict
    sorted_words = sorted(best.keys(), key=lambda w: w.encode("utf-8"))

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "dict_input.txt"
        output_path = Path(tmpdir) / "dict_output.bin"

        with open(input_path, "w", encoding="utf-8") as f:
            for word in sorted_words:
                tag, freq = best[word]
                pb = pos_byte(tag) if isinstance(tag, str) else tag
                f.write(f"{word}\t{pb}\t{freq}\n")

        # Run build-dict from repo root
        result = subprocess.run(
            ["cargo", "run", "--release", "--bin", "build-dict", "--",
             str(input_path), str(output_path)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print("build-dict stderr:", result.stderr)
            raise RuntimeError(f"build-dict failed with code {result.returncode}")

        # Print build-dict info
        for line in result.stderr.strip().splitlines():
            print(f"  [build-dict] {line}")

        dict_bytes = output_path.read_bytes()

    return dict_bytes, max_freq


MIN_SUFFIX_FREQ = 50


def augment_contractions(codebook: dict) -> dict:
    """Add single-char contracted syllable entries derived from multi-char entries,
    plus hardcoded entries for contractions not in the data."""

    # Part A: derive from existing multi-char entries
    derived = {}  # {char: {morph_tuple: freq}}
    content_pos = {"VV", "VA", "VX", "VCP", "VCN", "NNG", "NNP", "NNB", "NR", "NP", "MAG", "MAJ", "MM", "IC", "XR"}

    for surface, analyses in codebook.items():
        if len(surface) < 2:
            continue
        first_char = surface[0]
        # Skip if first char is not Hangul
        if not ('\uAC00' <= first_char <= '\uD7A3'):
            continue
        # Skip if first char already in codebook
        if first_char in codebook:
            continue

        for a in analyses:
            morphs = a["morphemes"]
            if not isinstance(morphs[0], list):
                continue
            # Find prefix: all morphemes up to and including first EP
            prefix = []
            for m in morphs:
                prefix.append(tuple(m))
                if m[1] == "EP":
                    break
            # Must have at least 2 morphemes ending with EP
            if len(prefix) < 2 or prefix[-1][1] != "EP":
                continue

            key = tuple(prefix)
            if first_char not in derived:
                derived[first_char] = {}
            derived[first_char][key] = derived[first_char].get(key, 0) + a["freq"]

    # Part B: hardcoded contraction table for entries not derivable from data
    # These are contracted syllables where stem + 었/았 merge into one syllable
    hardcoded = {
        "셨": [[["시", "EP"], ["었", "EP"]], 10000],   # honorific past
        "녔": [[["니", "VV"], ["었", "EP"]], 100],
        "렸": [[["리", "VV"], ["었", "EP"]], 100],
        "몄": [[["미", "VV"], ["었", "EP"]], 100],
        "텄": [[["터", "VV"], ["었", "EP"]], 100],
        "켰": [[["키", "VV"], ["었", "EP"]], 500],
        "빴": [[["빠", "VV"], ["았", "EP"]], 100],
        "꼈": [[["끼", "VV"], ["었", "EP"]], 100],
        "팠": [[["파", "VV"], ["았", "EP"]], 100],
        "잤": [[["자", "VV"], ["았", "EP"]], 500],
        "뤘": [[["루", "VV"], ["었", "EP"]], 50],
        "챘": [[["채", "VV"], ["었", "EP"]], 100],
        "맸": [[["매", "VV"], ["었", "EP"]], 100],
        "댔": [[["대", "VV"], ["었", "EP"]], 100],
        "랐": [[["라", "VV"], ["았", "EP"]], 100],
        "뺐": [[["빼", "VV"], ["었", "EP"]], 100],
        "샜": [[["새", "VV"], ["었", "EP"]], 50],
    }

    # Merge into codebook
    added = 0
    for char, morph_freqs in derived.items():
        if char not in codebook:
            entries = []
            for morph_tuple, freq in sorted(morph_freqs.items(), key=lambda x: -x[1]):
                if freq >= 10:  # minimum threshold
                    entries.append({"morphemes": [list(m) for m in morph_tuple], "freq": freq})
            if entries:
                codebook[char] = entries
                added += 1

    for char, (morphs, freq) in hardcoded.items():
        if char not in codebook:
            codebook[char] = [{"morphemes": morphs, "freq": freq}]
            added += 1
        # If it exists but doesn't have this analysis, add it
        else:
            existing_keys = set()
            for a in codebook[char]:
                existing_keys.add(tuple(tuple(m) for m in a["morphemes"]))
            new_key = tuple(tuple(m) for m in morphs)
            if new_key not in existing_keys:
                codebook[char].append({"morphemes": morphs, "freq": freq})

    print(f"  Contraction augmentation: {added} new entries added")
    return codebook


def build_suffix_codebook(codebook_path: Path, min_freq: int = MIN_SUFFIX_FREQ) -> tuple[bytes, int]:
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

    # Augment with contraction entries
    codebook = augment_contractions(codebook)

    # Count total entries before filtering
    total_before = sum(len(analyses) for analyses in codebook.values())

    # Filter analyses by freq >= min_freq, drop surfaces with no remaining analyses
    filtered = {}
    for surface, analyses in codebook.items():
        kept = [a for a in analyses if a["freq"] >= min_freq]
        if kept:
            filtered[surface] = kept

    total_after = sum(len(analyses) for analyses in filtered.values())
    print(f"  Suffix filter: {total_before:,} → {total_after:,} entries (freq >= {min_freq})")

    # Sort entries by surface UTF-8
    sorted_surfaces = sorted(filtered.keys(), key=lambda s: s.encode("utf-8"))

    max_suffix_freq = 0
    buf = bytearray()
    buf.extend(struct.pack("<I", len(sorted_surfaces)))

    for surface in sorted_surfaces:
        analyses = filtered[surface]
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

    # Section 6: Content dict (Dict v2 FST format)
    print("Building content dict (Dict v2 FST)...")
    dict_data, max_freq = build_content_dict_fst(DATA_DIR / "content_dict.txt")
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
