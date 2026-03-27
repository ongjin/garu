"""Extract codebook data from Kiwi-annotated kowikitext.

Outputs:
  - codebook_data/suffix_codebook.json
  - codebook_data/content_dict.txt  (sorted, tab-separated: word\tpos_tag)
  - codebook_data/trigram_costs.json
  - codebook_data/stats.json

Usage:
    python training/extract_codebook.py
"""

import json
import math
import os
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

from kiwipiepy import Kiwi

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "training" / "codebook_data"
OUT_DIR.mkdir(exist_ok=True)

POS_TAGS = [
    "NNG", "NNP", "NNB", "NR", "NP",
    "VV", "VA", "VX", "VCP", "VCN",
    "MAG", "MAJ", "MM", "IC",
    "JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ", "JX", "JC",
    "EP", "EF", "EC", "ETN", "ETM",
    "XPN", "XSN", "XSV", "XSA", "XR",
    "SF", "SP", "SS", "SE", "SO", "SW", "SH", "SL", "SN",
]
POS_SET = set(POS_TAGS)
POS_TO_IDX = {p: i for i, p in enumerate(POS_TAGS)}
NUM_POS = len(POS_TAGS)

# Content POS tags (open class — go in dictionary)
CONTENT_POS = {"NNG", "NNP", "NNB", "NR", "NP", "VV", "VA", "VX", "VCP", "VCN", "MAG", "MAJ", "MM", "IC", "XR"}
# Functional POS tags (closed class — go in suffix codebook)
FUNC_POS = {"JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ", "JX", "JC",
            "EP", "EF", "EC", "ETN", "ETM", "XPN", "XSN", "XSV", "XSA"}

def normalize_pos(tag):
    """Map Kiwi POS tag to our 42-tag set."""
    if tag in POS_SET:
        return tag
    base = tag.split('-')[0]
    if base in POS_SET:
        return base
    for p, d in [('V', 'VV'), ('N', 'NNG'), ('J', 'JX'), ('E', 'EF'), ('X', 'XR')]:
        if tag.startswith(p):
            return d
    return 'SW'


def load_sentences(max_sentences=1_200_000):
    """Load kowikitext sentences."""
    txt_path = ROOT / "training" / "kowikitext.txt"
    if not txt_path.exists():
        parquet_path = ROOT / "training" / "data.parquet"
        if not parquet_path.exists():
            url = "https://huggingface.co/datasets/heegyu/kowikitext/resolve/refs%2Fconvert%2Fparquet/20221001/train/0000.parquet"
            print(f"Downloading kowikitext...")
            urllib.request.urlretrieve(url, parquet_path)
        import pandas as pd
        df = pd.read_parquet(parquet_path)
        with open(txt_path, 'w') as f:
            for t in df['text'].tolist():
                f.write(t + '\n')
        del df

    sentences = []
    with open(txt_path) as f:
        for line in f:
            text = line.strip()
            if not text or text.startswith('=') or len(text) < 5 or len(text) > 200:
                continue
            sentences.append(text)
            if len(sentences) >= max_sentences:
                break
    print(f"Loaded {len(sentences):,} sentences")
    return sentences


def main():
    sentences = load_sentences()
    kw = Kiwi()

    # Accumulators
    suffix_patterns = defaultdict(lambda: defaultdict(int))  # surface -> {tuple(POS...): freq}
    content_words = defaultdict(lambda: defaultdict(int))     # surface -> {POS: freq}
    trigram_counts = defaultdict(int)                          # (pos1, pos2, pos3) -> count
    bigram_counts = defaultdict(int)                           # (pos1, pos2) -> count
    total_morphemes = 0

    t0 = time.time()
    for si, sent in enumerate(sentences):
        try:
            tokens = kw.tokenize(sent)
        except Exception:
            continue

        # Build POS sequence for trigram counting
        pos_seq = ["<BOS>", "<BOS>"]
        for tok in tokens:
            pos = normalize_pos(tok.tag)
            pos_seq.append(pos)

            # Classify: content word or functional morpheme
            surface = tok.form
            if not surface:
                continue

            if pos in CONTENT_POS:
                content_words[surface][pos] += 1
            elif pos in FUNC_POS:
                suffix_patterns[surface][(pos,)] += 1

        pos_seq.append("<EOS>")

        # Count trigrams
        for i in range(2, len(pos_seq)):
            trigram_counts[(pos_seq[i-2], pos_seq[i-1], pos_seq[i])] += 1
            bigram_counts[(pos_seq[i-1], pos_seq[i])] += 1
            total_morphemes += 1

        # Also extract multi-morpheme suffix patterns from token spans
        # E.g., "었다" spanning EP+EF, "에서" as JKB
        # Look at consecutive functional morphemes
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            pos = normalize_pos(tok.tag)
            if pos in FUNC_POS:
                # Collect consecutive functional morphemes
                func_start = i
                func_morphemes = []
                combined_surface = ""
                while i < len(tokens) and normalize_pos(tokens[i].tag) in FUNC_POS:
                    p = normalize_pos(tokens[i].tag)
                    func_morphemes.append(p)
                    combined_surface += tokens[i].form
                    i += 1
                if len(func_morphemes) >= 1 and combined_surface:
                    key = tuple(func_morphemes)
                    suffix_patterns[combined_surface][key] += 1
            else:
                i += 1

        # Extract contracted surface forms (entire word = content + functional)
        # E.g., "갔다" → the whole surface from start of VV to end of EF
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            pos = normalize_pos(tok.tag)
            if pos in CONTENT_POS and tok.tag not in ("SF", "SP", "SS", "SE", "SO", "SW", "SH", "SL", "SN"):
                # Check if next tokens are functional (forming a contracted word)
                j = i + 1
                func_parts = []
                while j < len(tokens):
                    next_pos = normalize_pos(tokens[j].tag)
                    if next_pos in FUNC_POS:
                        func_parts.append(next_pos)
                        j += 1
                    else:
                        break
                if func_parts:
                    # Get the surface span from input text
                    start_char = tok.start
                    end_char = tokens[j-1].end if j-1 < len(tokens) else tok.end
                    span_surface = sent[start_char:end_char]
                    if span_surface and len(span_surface) <= 10:
                        all_pos = [pos] + func_parts
                        suffix_patterns[span_surface][tuple(all_pos)] += 1
                i = j
            else:
                i += 1

        if (si + 1) % 100000 == 0:
            elapsed = time.time() - t0
            print(f"  {si+1:,} sentences ({elapsed:.0f}s)")

    elapsed = time.time() - t0
    print(f"Processing done: {si+1:,} sentences in {elapsed:.0f}s")

    # --- Save suffix codebook ---
    MIN_FREQ = 10
    codebook = {}
    for surface, analyses in suffix_patterns.items():
        entries = []
        for pos_tuple, freq in analyses.items():
            if freq >= MIN_FREQ:
                entries.append({"morphemes": list(pos_tuple), "freq": freq})
        if entries:
            codebook[surface] = sorted(entries, key=lambda e: -e["freq"])

    with open(OUT_DIR / "suffix_codebook.json", "w") as f:
        json.dump(codebook, f, ensure_ascii=False, indent=2)
    print(f"Suffix codebook: {len(codebook)} patterns (freq >= {MIN_FREQ})")

    # --- Save content dictionary ---
    content_list = []
    for surface, pos_counts in content_words.items():
        best_pos = max(pos_counts, key=pos_counts.get)
        total = sum(pos_counts.values())
        if total >= 3:  # minimum frequency
            content_list.append((surface, best_pos, total))

    content_list.sort(key=lambda x: x[0].encode('utf-8'))
    with open(OUT_DIR / "content_dict.txt", "w") as f:
        for surface, pos, freq in content_list:
            f.write(f"{surface}\t{pos}\t{freq}\n")
    print(f"Content dictionary: {len(content_list)} words")

    # --- Save trigram costs ---
    # Convert counts to -log probabilities with Laplace smoothing
    trigram_costs = {}
    alpha = 0.01  # smoothing
    for (p1, p2, p3), count in trigram_counts.items():
        if p1 in ("<BOS>", "<EOS>") or p2 in ("<BOS>", "<EOS>") or p3 in ("<BOS>", "<EOS>"):
            continue
        if p1 not in POS_SET or p2 not in POS_SET or p3 not in POS_SET:
            continue
        bg_count = bigram_counts.get((p1, p2), 0)
        prob = (count + alpha) / (bg_count + alpha * NUM_POS)
        cost = -math.log(max(prob, 1e-10))
        key = f"{p1},{p2},{p3}"
        trigram_costs[key] = round(cost, 4)

    # Also save bigram costs for backoff
    bigram_costs_out = {}
    unigram_total = sum(v for k, v in bigram_counts.items()
                        if k[0] not in ("<BOS>", "<EOS>") and k[1] not in ("<BOS>", "<EOS>"))
    for (p1, p2), count in bigram_counts.items():
        if p1 in ("<BOS>", "<EOS>") or p2 in ("<BOS>", "<EOS>"):
            continue
        if p1 not in POS_SET or p2 not in POS_SET:
            continue
        prob = (count + alpha) / (unigram_total + alpha * NUM_POS)
        cost = -math.log(max(prob, 1e-10))
        bigram_costs_out[f"{p1},{p2}"] = round(cost, 4)

    with open(OUT_DIR / "trigram_costs.json", "w") as f:
        json.dump({"trigram": trigram_costs, "bigram": bigram_costs_out,
                    "default_cost": 15.0}, f)
    print(f"Trigram costs: {len(trigram_costs)} entries")
    print(f"Bigram costs: {len(bigram_costs_out)} entries")

    # --- Stats ---
    stats = {
        "sentences_processed": si + 1,
        "total_morphemes": total_morphemes,
        "codebook_patterns": len(codebook),
        "content_words": len(content_list),
        "trigram_entries": len(trigram_costs),
        "bigram_entries": len(bigram_costs_out),
        "min_freq": MIN_FREQ,
    }
    with open(OUT_DIR / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats: {json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    main()
