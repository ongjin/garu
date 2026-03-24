"""Sejong corpus -> training data converter.

Decomposes Korean text into Jamo IDs matching the Rust jamo.rs implementation,
parses Sejong-format POS-tagged lines, and produces Jamo-level BIO labels
for BiLSTM+CRF training.
"""

import json
import re
import sys
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# POS tagset — must match Rust Pos enum order (39 tags, #[repr(u8)])
# ---------------------------------------------------------------------------

POS_TAGS: List[str] = [
    "NNG", "NNP", "NNB", "NR", "NP",
    "VV", "VA", "VX", "VCP", "VCN",
    "MAG", "MAJ", "MM",
    "IC",
    "JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ", "JX", "JC",
    "EP", "EF", "EC", "ETN", "ETM",
    "XPN", "XSN", "XSV", "XSA", "XR",
    "SF", "SP", "SS", "SE", "SO", "SW", "SH", "SL", "SN",
]

POS_TO_IDX = {tag: idx for idx, tag in enumerate(POS_TAGS)}

# ---------------------------------------------------------------------------
# Special token IDs — must match Rust jamo.rs constants
# ---------------------------------------------------------------------------

PAD: int = 0
UNK: int = 1
SPACE: int = 2
NUM: int = 3
LATIN: int = 4
PUNCT: int = 5
JAMO_OFFSET: int = 6

# Jamo compatibility block: U+3131 .. U+3163 (51 characters)
JAMO_COMPAT_START: int = 0x3131
JAMO_COMPAT_END: int = 0x3163
JAMO_COUNT: int = 51
VOCAB_SIZE: int = JAMO_OFFSET + JAMO_COUNT  # 57

# Hangul syllable block constants
S_BASE: int = 0xAC00
V_COUNT: int = 21
T_COUNT: int = 28
N_COUNT: int = V_COUNT * T_COUNT  # 588

# Leading consonants in compatibility Jamo order
LEADS: List[str] = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")

# Vowels in compatibility Jamo order
VOWELS: List[str] = list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")

# Trailing consonants (index 0 = no tail)
TAILS: List[Optional[str]] = [
    None,
    "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ",
    "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ",
    "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ",
    "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ",
    "ㅌ", "ㅍ", "ㅎ",
]

# ---------------------------------------------------------------------------
# Jamo decomposition — matching Rust implementation
# ---------------------------------------------------------------------------


def decompose(ch: str) -> Optional[Tuple[str, str, Optional[str]]]:
    """Decompose a Hangul syllable (U+AC00..U+D7A3) into (lead, vowel, tail).

    Returns None if the character is not a precomposed Hangul syllable.
    """
    code = ord(ch)
    if code < S_BASE or code > 0xD7A3:
        return None
    offset = code - S_BASE
    l_idx = offset // N_COUNT
    v_idx = (offset % N_COUNT) // T_COUNT
    t_idx = offset % T_COUNT
    return (LEADS[l_idx], VOWELS[v_idx], TAILS[t_idx])


def _jamo_to_id(ch: str) -> int:
    """Map a compatibility Jamo character (U+3131..U+3163) to its vocab ID."""
    code = ord(ch)
    if JAMO_COMPAT_START <= code <= JAMO_COMPAT_END:
        return JAMO_OFFSET + (code - JAMO_COMPAT_START)
    return UNK


def char_to_jamo_ids(ch: str) -> List[int]:
    """Encode a single character into a list of vocab IDs.

    - Hangul syllables are decomposed into 2-3 Jamo IDs.
    - Standalone compatibility Jamo characters are mapped directly.
    - Space -> SPACE, digit -> NUM, ASCII letter -> LATIN,
      ASCII punctuation -> PUNCT, else -> UNK.
    """
    result = decompose(ch)
    if result is not None:
        lead, vowel, tail = result
        ids = [_jamo_to_id(lead), _jamo_to_id(vowel)]
        if tail is not None:
            ids.append(_jamo_to_id(tail))
        return ids

    code = ord(ch)
    if JAMO_COMPAT_START <= code <= JAMO_COMPAT_END:
        return [_jamo_to_id(ch)]
    if ch == " ":
        return [SPACE]
    if ch.isdigit() and ord(ch) < 128:
        return [NUM]
    if ch.isalpha() and ord(ch) < 128:
        return [LATIN]
    if 0x21 <= ord(ch) <= 0x7E and not ch.isalnum():
        return [PUNCT]
    return [UNK]


def encode_text(text: str) -> List[int]:
    """Encode a text string into a sequence of vocab IDs (matching Rust encode)."""
    ids: List[int] = []
    for ch in text:
        ids.extend(char_to_jamo_ids(ch))
    return ids


# ---------------------------------------------------------------------------
# Sejong corpus parsing
# ---------------------------------------------------------------------------


def parse_sejong_line(line: str) -> List[Tuple[str, str]]:
    """Parse a Sejong-format POS-tagged string into (surface, tag) pairs.

    Format examples:
        "나/NP+는/JX 학교/NNG+에/JKB"
        "먹/VV+었/EP+다/EF+./SF"

    Each eojeol (space-separated) contains morphemes joined by '+'.
    Each morpheme is 'surface/TAG'.

    Returns list of (surface, tag) tuples. Spaces between eojeols are NOT
    included in the morpheme list but will be handled by the BIO labeller.
    """
    morphemes: List[Tuple[str, str]] = []
    eojeols = line.strip().split()
    for eojeol in eojeols:
        parts = eojeol.split("+")
        for part in parts:
            # Find last '/' — surface may contain '/'
            slash_idx = part.rfind("/")
            if slash_idx <= 0:
                continue
            surface = part[:slash_idx]
            tag = part[slash_idx + 1:]
            if tag in POS_TO_IDX:
                morphemes.append((surface, tag))
    return morphemes


# ---------------------------------------------------------------------------
# BIO label generation
# ---------------------------------------------------------------------------


def morphemes_to_bio_labels(
    text: str,
    morphemes: List[Tuple[str, str]],
) -> Optional[List[Tuple[str, str]]]:
    """Generate Jamo-level BIO labels from the original text and its morphemes.

    Each Jamo position gets a label like 'B-NNG', 'I-NNG', or 'O'.
    Space characters get the 'O' label.

    Returns a list of (jamo_id_str, bio_label) pairs, or None if alignment fails.
    """
    # Reconstruct the surface text from morphemes to align with the original
    # The original text has spaces between eojeols. Morphemes are contiguous
    # within an eojeol.
    jamo_ids = encode_text(text)
    labels: List[str] = []

    char_pos = 0  # position in original text characters
    for morph_surface, morph_tag in morphemes:
        # Skip spaces in the original text
        while char_pos < len(text) and text[char_pos] == " ":
            # Space produces 1 jamo (SPACE)
            labels.append("O")
            char_pos += 1

        # Align morpheme surface with the original text
        morph_jamo_count = 0
        for ch in morph_surface:
            if char_pos >= len(text):
                return None  # alignment failure
            if text[char_pos] != ch:
                return None  # alignment failure
            morph_jamo_count += len(char_to_jamo_ids(ch))
            char_pos += 1

        # Assign B/I labels for this morpheme
        for j in range(morph_jamo_count):
            if j == 0:
                labels.append(f"B-{morph_tag}")
            else:
                labels.append(f"I-{morph_tag}")

    # Handle trailing spaces
    while char_pos < len(text):
        if text[char_pos] == " ":
            labels.append("O")
            char_pos += 1
        else:
            # Unaligned trailing characters
            for _ in char_to_jamo_ids(text[char_pos]):
                labels.append("O")
            char_pos += 1

    if len(labels) != len(jamo_ids):
        return None  # length mismatch

    return list(zip([str(x) for x in jamo_ids], labels))


# ---------------------------------------------------------------------------
# Corpus preprocessing
# ---------------------------------------------------------------------------


def preprocess_corpus(input_path: str, output_path: str) -> None:
    """Convert a Sejong-format corpus file to JSONL training data.

    Input format (tab-separated, one sentence per line):
        <sentence_text>\\t<tagged_text>

    Where tagged_text is Sejong format: "나/NP+는/JX 학교/NNG+에/JKB ..."

    Output: JSONL where each line is:
        {"ids": [int, ...], "labels": ["B-NNG", "I-NNG", ...]}
    """
    num_ok = 0
    num_fail = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) < 2:
                num_fail += 1
                continue

            raw_text = parts[0].strip()
            tagged_text = parts[1].strip()

            if not raw_text or not tagged_text:
                num_fail += 1
                continue

            morphemes = parse_sejong_line(tagged_text)
            if not morphemes:
                num_fail += 1
                continue

            result = morphemes_to_bio_labels(raw_text, morphemes)
            if result is None:
                num_fail += 1
                continue

            ids = [int(x) for x, _ in result]
            labels = [lbl for _, lbl in result]

            record = {"ids": ids, "labels": labels}
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            num_ok += 1

    print(f"Preprocessed {num_ok} sentences ({num_fail} skipped) -> {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input_corpus> <output_jsonl>")
        sys.exit(1)
    preprocess_corpus(sys.argv[1], sys.argv[2])
