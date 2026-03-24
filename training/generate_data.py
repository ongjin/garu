"""Generate training data from kowikitext using Kiwi morphological analyzer.

Reads raw Korean text from kowikitext corpus, runs Kiwi analyzer to get
morpheme annotations, then outputs JSONL in the format expected by train.py.

Usage:
    python3 generate_data.py <kowikitext_file> <output.jsonl> [--max-sentences N]
"""

import json
import sys
from pathlib import Path

from kiwipiepy import Kiwi

# Must match Rust jamo.rs and preprocess.py
HANGUL_BASE = 0xAC00
HANGUL_END = 0xD7A3
VOWEL_COUNT = 21
TAIL_COUNT = 28

LEADS = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
VOWELS = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
TAILS = [
    "",
    "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ",
    "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ",
    "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ",
    "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ",
    "ㅌ", "ㅍ", "ㅎ",
]

PAD_ID = 0
UNK_ID = 1
SPACE_ID = 2
NUM_ID = 3
LATIN_ID = 4
PUNCT_ID = 5
JAMO_OFFSET = 6
JAMO_COMPAT_START = 0x3131

POS_TAGS = [
    "NNG", "NNP", "NNB", "NR", "NP",
    "VV", "VA", "VX", "VCP", "VCN",
    "MAG", "MAJ", "MM",
    "IC",
    "JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ", "JX", "JC",
    "EP", "EF", "EC", "ETN", "ETM",
    "XPN", "XSN", "XSV", "XSA", "XR",
    "SF", "SP", "SS", "SE", "SO", "SW", "SH", "SL", "SN",
]
POS_SET = set(POS_TAGS)

# BIO labels matching train.py
BIO_LABELS = []
for tag in POS_TAGS:
    BIO_LABELS.append(f"B-{tag}")
    BIO_LABELS.append(f"I-{tag}")
BIO_LABELS.append("O")


def decompose(ch):
    code = ord(ch)
    if HANGUL_BASE <= code <= HANGUL_END:
        offset = code - HANGUL_BASE
        lead = offset // (VOWEL_COUNT * TAIL_COUNT)
        vowel = (offset % (VOWEL_COUNT * TAIL_COUNT)) // TAIL_COUNT
        tail = offset % TAIL_COUNT
        return LEADS[lead], VOWELS[vowel], TAILS[tail] if tail > 0 else None
    return None


def char_to_jamo_ids(ch):
    result = decompose(ch)
    if result:
        lead, vowel, tail = result
        ids = [
            JAMO_OFFSET + (ord(lead) - JAMO_COMPAT_START),
            JAMO_OFFSET + (ord(vowel) - JAMO_COMPAT_START),
        ]
        if tail:
            ids.append(JAMO_OFFSET + (ord(tail) - JAMO_COMPAT_START))
        return ids
    elif ch == " ":
        return [SPACE_ID]
    elif ch.isdigit():
        return [NUM_ID]
    elif ch.isascii() and ch.isalpha():
        return [LATIN_ID]
    elif ch in ".,!?;:()[]{}\"'-/":
        return [PUNCT_ID]
    else:
        return [UNK_ID]


def char_jamo_count(ch):
    result = decompose(ch)
    if result:
        _, _, tail = result
        return 3 if tail else 2
    return 1


def encode_text(text):
    ids = []
    for ch in text:
        ids.extend(char_to_jamo_ids(ch))
    return ids


def text_to_bio_labels(text, tokens):
    """Convert Kiwi tokens to Jamo-level BIO labels.

    Args:
        text: original text string
        tokens: list of Kiwi Token objects (form, tag, start, end)

    Returns:
        list of BIO label strings, one per Jamo position
    """
    # Build char-level labels first
    char_labels = ["O"] * len(text)

    for token in tokens:
        tag = token.tag
        # Map Kiwi tags to our POS set
        # Kiwi uses same Sejong tagset but may have extra subtags
        if tag not in POS_SET:
            # Try base tag (e.g., "NNG" from "NNG-something")
            base = tag.split("-")[0]
            if base in POS_SET:
                tag = base
            elif tag.startswith("V"):
                tag = "VV"
            elif tag.startswith("N"):
                tag = "NNG"
            elif tag.startswith("J"):
                tag = "JX"
            elif tag.startswith("E"):
                tag = "EF"
            elif tag.startswith("X"):
                tag = "XR"
            else:
                tag = "SW"

        start = token.start
        end = token.end

        # Clamp to text bounds
        start = max(0, min(start, len(text)))
        end = max(start, min(end, len(text)))

        for i in range(start, end):
            if i == start:
                char_labels[i] = f"B-{tag}"
            else:
                char_labels[i] = f"I-{tag}"

    # Expand char labels to Jamo labels
    jamo_labels = []
    for i, ch in enumerate(text):
        n_jamo = char_jamo_count(ch)
        label = char_labels[i]
        if label.startswith("B-"):
            jamo_labels.append(label)
            tag = label[2:]
            for _ in range(n_jamo - 1):
                jamo_labels.append(f"I-{tag}")
        elif label.startswith("I-"):
            for _ in range(n_jamo):
                jamo_labels.append(label)
        else:
            for _ in range(n_jamo):
                jamo_labels.append("O")

    return jamo_labels


def process_corpus(input_path, output_path, max_sentences=None):
    kw = Kiwi()
    count = 0
    skipped = 0

    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:

        for line_num, line in enumerate(f_in):
            if max_sentences and count >= max_sentences:
                break

            text = line.strip()

            # Skip empty lines, headers, very short/long text
            if not text:
                continue
            if text.startswith("="):
                continue
            if len(text) < 5 or len(text) > 300:
                continue

            try:
                tokens = kw.tokenize(text)
            except Exception:
                skipped += 1
                continue

            if not tokens:
                skipped += 1
                continue

            jamo_ids = encode_text(text)
            bio_labels = text_to_bio_labels(text, tokens)

            # Sanity check: lengths must match
            if len(jamo_ids) != len(bio_labels):
                skipped += 1
                continue

            # Validate all labels are in BIO_LABELS
            valid = all(lbl in BIO_LABELS or lbl == "O" for lbl in bio_labels)
            if not valid:
                skipped += 1
                continue

            json.dump(
                {"ids": jamo_ids, "labels": bio_labels},
                f_out,
                ensure_ascii=False,
            )
            f_out.write("\n")
            count += 1

            if count % 10000 == 0:
                print(f"  Processed {count:,} sentences (skipped {skipped:,})...")

    print(f"Done: {count:,} sentences written, {skipped:,} skipped")
    return count


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input_text> <output.jsonl> [--max-sentences N]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    max_sentences = None
    if "--max-sentences" in sys.argv:
        idx = sys.argv.index("--max-sentences")
        max_sentences = int(sys.argv[idx + 1])

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    if max_sentences:
        print(f"Max sentences: {max_sentences:,}")

    process_corpus(input_path, output_path, max_sentences)
