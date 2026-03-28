"""Extract codebook data from NIKL MP gold annotations.

Instead of extracting from Kiwi output (which has different segmentation standards),
this extracts content dict, suffix codebook, and trigram costs directly from
NIKL MP gold morpheme annotations.

Can also merge with existing kowikitext-based data for better coverage.

Usage:
    python training/extract_nikl_codebook.py [--merge]
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

NIKL_DIR = Path.home() / "Downloads" / "NIKL_MP(v1.1)"
OUT_DIR = Path(__file__).parent / "codebook_data"

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

NIKL_MAP = {'MMD': 'MM', 'MMN': 'MM', 'MMA': 'MM', 'NA': 'NNG', 'NAP': 'NNG', 'NF': 'NNG', 'NV': 'VV'}

CONTENT_POS = {"NNG", "NNP", "NNB", "NR", "NP", "VV", "VA", "VX", "VCP", "VCN",
               "MAG", "MAJ", "MM", "IC", "XR"}
FUNC_POS = {"JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ", "JX", "JC",
            "EP", "EF", "EC", "ETN", "ETM", "XPN", "XSN", "XSV", "XSA"}


def normalize_pos(tag):
    if tag in POS_SET: return tag
    if tag in NIKL_MAP: return NIKL_MAP[tag]
    base = tag.split('-')[0]
    if base in POS_SET: return base
    return 'SW'


def load_nikl_sentences():
    """Load ALL sentences from NIKL MP."""
    sentences = []
    for fname in ["NXMP1902008040.json", "SXMP1902008031.json"]:
        path = NIKL_DIR / fname
        if not path.exists():
            print(f"  Warning: {path} not found")
            continue
        with open(path) as f:
            data = json.load(f)
        count = 0
        for doc in data["document"]:
            if doc is None: continue
            for sent in (doc.get("sentence") or []):
                text = sent.get("form", "")
                if not text: continue
                morphemes = []
                for m in (sent.get("morpheme") or []):
                    form = m.get("form", "").strip()
                    label = normalize_pos(m.get("label", ""))
                    if form and label:
                        morphemes.append((form, label))
                if morphemes:
                    sentences.append((text, morphemes))
                    count += 1
        print(f"  {fname}: {count} sentences")
    return sentences


def extract_codebook(sentences):
    """Extract content dict, suffix codebook, and trigram costs from sentences."""
    content_words = defaultdict(lambda: defaultdict(int))  # surface → {POS: freq}
    suffix_patterns = defaultdict(lambda: defaultdict(int))  # surface → {tuple(morphemes): freq}
    trigram_counts = defaultdict(int)
    bigram_counts = defaultdict(int)
    total_morphemes = 0

    for text, morphemes in sentences:
        prev_pos = "<BOS>"
        prev_prev_pos = "<BOS>"

        # Process morphemes: identify content words and suffix patterns
        # Group morphemes by their word_id-like grouping (consecutive functional after content)
        i = 0
        while i < len(morphemes):
            form, pos = morphemes[i]
            total_morphemes += 1

            # Trigram counts
            trigram_counts[(prev_prev_pos, prev_pos, pos)] += 1
            bigram_counts[(prev_pos, pos)] += 1

            if pos in CONTENT_POS:
                content_words[form][pos] += 1
                # Check if followed by functional morphemes (suffix pattern)
                j = i + 1
                func_seq = []
                while j < len(morphemes) and morphemes[j][1] in FUNC_POS:
                    func_seq.append(morphemes[j])
                    j += 1

                if func_seq:
                    # Build suffix surface and morpheme list
                    suffix_surface = "".join(f for f, _ in func_seq)
                    suffix_key = tuple((f, p) for f, p in func_seq)
                    suffix_patterns[suffix_surface][suffix_key] += 1

                    # Also record sub-patterns (1-morpheme, 2-morpheme, etc.)
                    for k in range(1, len(func_seq)):
                        sub_surface = "".join(f for f, _ in func_seq[:k])
                        sub_key = tuple((f, p) for f, p in func_seq[:k])
                        suffix_patterns[sub_surface][sub_key] += 1

            elif pos in FUNC_POS:
                # Standalone functional morpheme
                suffix_patterns[form][((form, pos),)] += 1

            # Also handle symbol POS
            elif pos in ("SF", "SP", "SS", "SE", "SO", "SW", "SH", "SL", "SN"):
                # Don't add to content dict or suffix
                pass
            else:
                content_words[form][pos] += 1

            prev_prev_pos = prev_pos
            prev_pos = pos
            i += 1

        # EOS
        trigram_counts[(prev_prev_pos, prev_pos, "<EOS>")] += 1
        bigram_counts[(prev_pos, "<EOS>")] += 1

    print(f"  Total morphemes: {total_morphemes:,}")
    print(f"  Content words: {len(content_words):,}")
    print(f"  Suffix patterns: {len(suffix_patterns):,}")
    print(f"  Trigram keys: {len(trigram_counts):,}")

    return content_words, suffix_patterns, trigram_counts, bigram_counts


def merge_with_kiwi_data(nikl_content, nikl_suffix, nikl_trigram, nikl_bigram,
                          kiwi_weight=1.0, nikl_weight=3.0):
    """Merge NIKL gold data with existing Kiwi-based codebook data."""
    # Load existing Kiwi-based data
    existing_cd = defaultdict(lambda: defaultdict(int))
    cd_path = OUT_DIR / "content_dict.txt"
    with open(cd_path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                existing_cd[parts[0]][parts[1]] += int(int(parts[2]) * kiwi_weight)

    existing_cb = json.load(open(OUT_DIR / "suffix_codebook.json"))
    existing_costs = json.load(open(OUT_DIR / "trigram_costs.json"))

    # Merge content dict: add NIKL entries with weight
    for word, pos_freqs in nikl_content.items():
        for pos, freq in pos_freqs.items():
            existing_cd[word][pos] += int(freq * nikl_weight)

    # Merge suffix codebook
    for surface, pattern_freqs in nikl_suffix.items():
        if surface not in existing_cb:
            existing_cb[surface] = []
        existing_keys = set()
        for a in existing_cb[surface]:
            key = tuple(tuple(m) if isinstance(m, list) else (m,) for m in a["morphemes"])
            existing_keys.add(key)

        for morpheme_tuple, freq in pattern_freqs.items():
            key = morpheme_tuple
            if key not in existing_keys:
                existing_cb[surface].append({
                    "morphemes": [list(m) for m in morpheme_tuple],
                    "freq": int(freq * nikl_weight)
                })
                existing_keys.add(key)
            else:
                # Add freq to existing
                for a in existing_cb[surface]:
                    a_key = tuple(tuple(m) if isinstance(m, list) else (m,) for m in a["morphemes"])
                    if a_key == key:
                        a["freq"] += int(freq * nikl_weight)
                        break

    # Merge trigram costs (convert from counts to -log prob)
    # Add NIKL trigram counts to existing
    existing_tg = existing_costs.get("trigram", {})
    existing_bg = existing_costs.get("bigram", {})

    # For trigrams, we need to re-compute from combined counts
    # Just add NIKL costs as weighted averages
    for (p1, p2, p3), count in nikl_trigram.items():
        if p1 not in POS_SET or p2 not in POS_SET or p3 not in POS_SET:
            continue
        key = f"{p1},{p2},{p3}"
        nikl_cost = -math.log(max(count, 1) / max(sum(
            c for (a, b, _), c in nikl_trigram.items() if a == p1 and b == p2), 1))
        if key in existing_tg:
            # Weighted average
            existing_tg[key] = (existing_tg[key] * kiwi_weight + nikl_cost * nikl_weight) / (kiwi_weight + nikl_weight)
        else:
            existing_tg[key] = nikl_cost

    for (p1, p2), count in nikl_bigram.items():
        if p1 not in POS_SET or p2 not in POS_SET:
            continue
        key = f"{p1},{p2}"
        nikl_cost = -math.log(max(count, 1) / max(sum(
            c for (a, _), c in nikl_bigram.items() if a == p1), 1))
        if key in existing_bg:
            existing_bg[key] = (existing_bg[key] * kiwi_weight + nikl_cost * nikl_weight) / (kiwi_weight + nikl_weight)
        else:
            existing_bg[key] = nikl_cost

    return existing_cd, existing_cb, {"trigram": existing_tg, "bigram": existing_bg, "default_cost": 15.0}


def save_data(content_dict, suffix_codebook, trigram_costs):
    """Save merged data back to codebook_data/."""
    # Content dict
    with open(OUT_DIR / "content_dict.txt", "w") as f:
        for word in sorted(content_dict.keys()):
            # Keep highest-freq POS
            best_pos = max(content_dict[word].items(), key=lambda x: x[1])
            f.write(f"{word}\t{best_pos[0]}\t{best_pos[1]}\n")

    # Suffix codebook
    with open(OUT_DIR / "suffix_codebook.json", "w") as f:
        json.dump(suffix_codebook, f, ensure_ascii=False)

    # Trigram costs
    with open(OUT_DIR / "trigram_costs.json", "w") as f:
        json.dump(trigram_costs, f, ensure_ascii=False)


def main():
    merge_mode = "--merge" in sys.argv

    print("Loading NIKL MP sentences...")
    sentences = load_nikl_sentences()
    print(f"  Total: {len(sentences)} sentences\n")

    print("Extracting codebook from NIKL gold...")
    nikl_cd, nikl_cb, nikl_tg, nikl_bg = extract_codebook(sentences)

    if merge_mode:
        print("\nMerging with existing Kiwi-based data (weight: kiwi=1, nikl=3)...")
        content_dict, suffix_cb, trigram_costs = merge_with_kiwi_data(
            nikl_cd, nikl_cb, nikl_tg, nikl_bg,
            kiwi_weight=1.0, nikl_weight=3.0
        )
    else:
        print("\nUsing NIKL-only data (no merge)...")
        content_dict = nikl_cd
        suffix_cb = {}
        for surface, pattern_freqs in nikl_cb.items():
            suffix_cb[surface] = [
                {"morphemes": [list(m) for m in mt], "freq": freq}
                for mt, freq in pattern_freqs.items()
            ]
        # Build trigram costs from counts
        trigram_costs = {"trigram": {}, "bigram": {}, "default_cost": 15.0}
        # group by (p1,p2) for normalization
        tg_totals = defaultdict(int)
        for (p1, p2, p3), c in nikl_tg.items():
            if p1 in POS_SET and p2 in POS_SET and p3 in POS_SET:
                tg_totals[(p1, p2)] += c
        for (p1, p2, p3), c in nikl_tg.items():
            if p1 in POS_SET and p2 in POS_SET and p3 in POS_SET:
                total = tg_totals.get((p1, p2), 1)
                cost = -math.log(max(c, 1) / max(total, 1))
                trigram_costs["trigram"][f"{p1},{p2},{p3}"] = cost
        bg_totals = defaultdict(int)
        for (p1, p2), c in nikl_bg.items():
            if p1 in POS_SET and p2 in POS_SET:
                bg_totals[p1] += c
        for (p1, p2), c in nikl_bg.items():
            if p1 in POS_SET and p2 in POS_SET:
                total = bg_totals.get(p1, 1)
                cost = -math.log(max(c, 1) / max(total, 1))
                trigram_costs["bigram"][f"{p1},{p2}"] = cost

    print("\nSaving codebook data...")
    save_data(content_dict, suffix_cb, trigram_costs)
    print("  Done!")

    # Stats
    cd_count = sum(1 for _ in open(OUT_DIR / "content_dict.txt"))
    cb = json.load(open(OUT_DIR / "suffix_codebook.json"))
    cb_count = sum(len(v) for v in cb.values())
    print(f"\n  Content dict: {cd_count:,} entries")
    print(f"  Suffix codebook: {len(cb):,} surfaces, {cb_count:,} analyses")


if __name__ == "__main__":
    main()
