"""Build context-aware word bigram rules from NIKL MP data.

For ambiguous eojeols (same surface, multiple analyses), generates rules:
(eojeol_surface, prev_last_pos) → preferred_analysis with bonus

These are added to the word_bigrams.bin and help disambiguate homographs
like "한" (MM vs VV+ETM) and "나는" (NP+JX vs VV+ETM).

Output: training/codebook_data/word_bigrams.bin (appends to existing)
"""
import json
import struct
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "training" / "codebook_data"
NIKL_DIR = Path.home() / "Downloads" / "NIKL_MP(v1.1)"

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
POS_TO_BYTE = {p: i for i, p in enumerate(POS_TAGS)}
NIKL_MAP = {'MMD':'MM','MMN':'MM','MMA':'MM','NA':'NNG','NAP':'NNG','NF':'NNG','NV':'VV'}

def normalize_pos(t):
    if t in POS_SET: return t
    if t in NIKL_MAP: return NIKL_MAP[t]
    b = t.split('-')[0]
    return b if b in POS_SET else 'SW'


def main():
    # Collect (eojeol, prev_pos) → analysis distribution from NIKL
    context_dist = defaultdict(lambda: defaultdict(Counter))
    # eojeol → all analyses
    eojeol_analyses = defaultdict(Counter)

    for fname in ["NXMP1902008040.json", "SXMP1902008031.json"]:
        p = NIKL_DIR / fname
        if not p.exists(): continue
        d = json.load(open(p))
        for doc in d["document"]:
            if not doc: continue
            for sent in (doc.get("sentence") or []):
                words = {w["id"]: w["form"] for w in (sent.get("word") or [])}
                morphemes = sent.get("morpheme") or []
                word_morphs = defaultdict(list)
                for m in morphemes:
                    wid = m.get("word_id")
                    form = m.get("form", "").strip()
                    label = normalize_pos(m.get("label", ""))
                    if wid and form:
                        word_morphs[wid].append((form, label))

                sorted_wids = sorted(words.keys())
                for idx, wid in enumerate(sorted_wids):
                    wform = words[wid]
                    morphs = word_morphs.get(wid, [])
                    if not morphs: continue
                    analysis_key = tuple(morphs)
                    eojeol_analyses[wform][analysis_key] += 1

                    if idx > 0:
                        prev_wid = sorted_wids[idx - 1]
                        prev_morphs = word_morphs.get(prev_wid, [])
                        if prev_morphs:
                            prev_last_pos = prev_morphs[-1][1]
                            context_dist[wform][prev_last_pos][analysis_key] += 1

    # Find ambiguous eojeols (multiple analyses with significant counts)
    ambig_eojeols = {}
    for eojeol, analyses in eojeol_analyses.items():
        if len(analyses) < 2: continue
        top2 = analyses.most_common(2)
        total = sum(analyses.values())
        if total < 100: continue
        # At least 10% for second analysis
        if top2[1][1] / total >= 0.10:
            ambig_eojeols[eojeol] = analyses

    print(f"Ambiguous eojeols (≥2 analyses, ≥100 occurrences): {len(ambig_eojeols)}")

    # Generate context rules for ambiguous eojeols
    rules = []  # (eojeol, prev_pos, first_morph_form, first_morph_pos, bonus)

    for eojeol, analyses in ambig_eojeols.items():
        total = sum(analyses.values())
        default_analysis = analyses.most_common(1)[0][0]  # most common = default

        for prev_pos, ctx_analyses in context_dist[eojeol].items():
            ctx_total = sum(ctx_analyses.values())
            if ctx_total < 30: continue  # need sufficient evidence

            ctx_best = ctx_analyses.most_common(1)[0]
            ctx_best_analysis, ctx_best_count = ctx_best
            ctx_pct = ctx_best_count / ctx_total

            # If context-preferred analysis differs from global default
            if ctx_best_analysis != default_analysis and ctx_pct >= 0.50:
                # Generate a bonus rule
                first_form, first_pos = ctx_best_analysis[0]
                # Bonus proportional to how much context shifts preference
                global_pct = analyses.get(ctx_best_analysis, 0) / total
                shift = ctx_pct - global_pct
                if shift > 0.15:  # meaningful shift
                    bonus = min(shift * 8.0, 5.0)  # scale to cost units
                    rules.append((eojeol, prev_pos, first_form, first_pos, bonus))

    print(f"Generated {len(rules)} context bigram rules")
    for eojeol, prev_pos, form, pos, bonus in sorted(rules, key=lambda x: -x[4])[:20]:
        print(f"  {prev_pos:>6} + {eojeol:<8} → {form}/{pos} (bonus={bonus:.2f})")

    # Encode as word_bigrams.bin format
    # Format: num_entries [u32], then for each:
    #   word_len [u16], word_utf8, prev_pos [u8], target_pos [u8], bonus_q [i8]
    # We need to match the existing format in build_codebook_model.py

    # Load existing word bigrams
    existing_path = DATA_DIR / "word_bigrams.bin"
    if existing_path.exists():
        existing_data = existing_path.read_bytes()
        existing_count = int.from_bytes(existing_data[:4], 'little')
        print(f"\nExisting word bigrams: {existing_count} entries")
    else:
        existing_data = b'\x00\x00\x00\x00'
        existing_count = 0

    # Append new rules
    buf = bytearray(existing_data[4:])  # skip count header
    new_count = existing_count

    for eojeol, prev_pos, form, pos, bonus in rules:
        # The word bigram format uses the FIRST morpheme's form as the key
        word_bytes = form.encode('utf-8')
        prev_pb = POS_TO_BYTE.get(prev_pos, 0)
        target_pb = POS_TO_BYTE.get(pos, 0)
        bonus_q = max(-128, min(127, int(-bonus * 25)))  # negative = cheaper = preferred
        buf.extend(struct.pack("<H", len(word_bytes)))
        buf.extend(word_bytes)
        buf.extend(struct.pack("B", prev_pb))
        buf.extend(struct.pack("B", target_pb))
        buf.extend(struct.pack("b", bonus_q))
        new_count += 1

    # Write with new count
    out = struct.pack("<I", new_count) + bytes(buf)
    existing_path.write_bytes(out)
    print(f"Written {new_count} total word bigram entries ({new_count - existing_count} new)")


if __name__ == "__main__":
    main()
