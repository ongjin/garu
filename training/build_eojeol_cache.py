"""Build smart eojeol cache from NIKL MP gold annotations.

Uses word_id to correctly align morphemes to eojeols.
Runs Garu's Viterbi (without cache) on eojeols, compares against gold,
and selects top-N eojeols where caching would correct the most errors.

Outputs: training/codebook_data/eojeol_cache.bin
"""
import json
import os
import struct
import subprocess
import sys
import tempfile
from collections import defaultdict, Counter
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


def normalize_pos(tag):
    if tag in POS_SET:
        return tag
    NIKL_MAP = {
        'MMD': 'MM', 'MMN': 'MM', 'MMA': 'MM',
        'NA': 'NNG', 'NAP': 'NNG', 'NF': 'NNG', 'NV': 'VV',
    }
    if tag in NIKL_MAP:
        return NIKL_MAP[tag]
    base = tag.split('-')[0]
    if base in POS_SET:
        return base
    for p, d in [('V', 'VV'), ('N', 'NNG'), ('J', 'JX'), ('E', 'EF'), ('X', 'XR')]:
        if tag.startswith(p):
            return d
    return 'SW'


def load_eojeol_gold():
    """Load eojeol→morphemes mapping using word_id alignment from NIKL MP."""
    eojeol_analyses = defaultdict(Counter)  # eojeol_form → Counter of analysis tuples

    for fname in ["NXMP1902008040.json", "SXMP1902008031.json"]:
        path = NIKL_DIR / fname
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)

        for doc in data["document"]:
            if doc is None:
                continue
            for sent in (doc.get("sentence") or []):
                words = {w["id"]: w["form"] for w in (sent.get("word") or [])}
                morphemes = sent.get("morpheme") or []

                # Group morphemes by word_id
                word_morphs = defaultdict(list)
                for m in morphemes:
                    wid = m.get("word_id")
                    form = m.get("form", "").strip()
                    label = normalize_pos(m.get("label", ""))
                    if wid and form and label:
                        word_morphs[wid].append((form, label))

                # Map each word to its morpheme analysis
                for wid, word_form in words.items():
                    if wid not in word_morphs:
                        continue
                    morphs = word_morphs[wid]
                    if not morphs:
                        continue
                    key = tuple(morphs)
                    eojeol_analyses[word_form][key] += 1

    # For each eojeol, pick the most common analysis
    gold = {}
    for eojeol, analyses in eojeol_analyses.items():
        best_analysis, best_count = analyses.most_common(1)[0]
        total = sum(analyses.values())
        # Require consistency: most common ≥50%
        if best_count / total >= 0.5 and len(eojeol.strip()) > 0:
            gold[eojeol] = (list(best_analysis), best_count, total)

    return gold


def run_garu(eojeols):
    """Run Garu on eojeols."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        for eojeol in eojeols:
            f.write(eojeol + '\n')
        input_path = f.name

    env = os.environ.copy()
    result = subprocess.run(
        ["cargo", "run", "--release", "--example", "analyze_batch", "--", input_path],
        cwd=str(ROOT), capture_output=True, text=True, timeout=600, env=env,
    )
    os.unlink(input_path)

    if result.returncode != 0:
        print(f"Analyzer failed: {result.stderr[:500]}", file=sys.stderr)
        return None

    analyses = []
    current = []
    for line in result.stdout.strip().split('\n'):
        line = line.strip()
        if line == '---':
            analyses.append(current)
            current = []
        elif line == '[]':
            analyses.append([])
        elif '\t' in line:
            form, pos = line.split('\t', 1)
            current.append((form, pos))
    if current:
        analyses.append(current)
    return analyses


def compute_correction_value(pred_morphs, gold_morphs):
    """Compute how many morpheme errors caching would fix."""
    pred_set = set(pred_morphs)
    gold_set = set(gold_morphs)
    fp = len(pred_set - gold_set)
    fn_c = len(gold_set - pred_set)
    return fp + fn_c


def main():
    max_cache_entries = int(sys.argv[1]) if len(sys.argv) > 1 else 5000

    print(f"Loading NIKL MP eojeol gold (using word_id alignment)...")
    gold = load_eojeol_gold()
    print(f"  Unique eojeols with gold analysis: {len(gold)}")

    # Get eojeols to analyze
    eojeol_list = sorted(gold.keys())
    print(f"\nRunning Garu on {len(eojeol_list)} eojeols (no cache)...")

    # Process in batches
    batch_size = 5000
    all_results = []
    for batch_start in range(0, len(eojeol_list), batch_size):
        batch = eojeol_list[batch_start:batch_start + batch_size]
        results = run_garu(batch)
        if results is None:
            print("Failed to run analyzer!")
            return
        all_results.extend(results)
        if (batch_start // batch_size) % 20 == 0:
            print(f"  Processed {min(batch_start + batch_size, len(eojeol_list))}/{len(eojeol_list)}")

    # Compute correction value for each eojeol
    corrections = []
    for i, eojeol in enumerate(eojeol_list):
        if i >= len(all_results):
            break
        pred = tuple(all_results[i])
        gold_morphs, gold_count, gold_total = gold[eojeol]
        gold_tuples = tuple((f, p) for f, p in gold_morphs)

        cv = compute_correction_value(pred, gold_tuples)
        if cv > 0:
            weighted_cv = cv * gold_total
            corrections.append((eojeol, gold_morphs, weighted_cv, cv, gold_total))

    corrections.sort(key=lambda x: -x[2])
    print(f"\n  Eojeols needing correction: {len(corrections)}")
    print(f"  Selecting top {max_cache_entries} by weighted correction value")

    selected = corrections[:max_cache_entries]

    # Show top 20
    print(f"\n  Top 20 cache entries:")
    for eojeol, morphs, wcv, cv, freq in selected[:20]:
        morphs_str = " + ".join(f"{f}/{p}" for f, p in morphs)
        print(f"    {eojeol:15s} (freq={freq:4d}, cv={cv}) → {morphs_str}")

    # Build string table for morpheme forms
    all_forms = set()
    for _, morphs, _, _, _ in selected:
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

    print(f"\n  String table: {len(sorted_forms)} unique forms, {len(string_table):,} bytes")

    # Build compact binary cache (v1 format)
    buf = bytearray()
    buf.extend(struct.pack("<I", 0xFFFFFFFF))  # format marker
    buf.extend(struct.pack("B", 1))            # sub-version 1

    # String table
    buf.extend(struct.pack("<I", len(string_table)))
    buf.extend(string_table)
    buf.extend(struct.pack("<H", len(sorted_forms)))
    for off in string_offsets:
        buf.extend(struct.pack("<H", off))

    # Entries
    buf.extend(struct.pack("<I", len(selected)))
    for eojeol, morphs, _, _, _ in selected:
        eojeol_bytes = eojeol.encode("utf-8")
        buf.extend(struct.pack("B", len(eojeol_bytes)))
        buf.extend(eojeol_bytes)
        buf.extend(struct.pack("B", len(morphs)))
        for form, pos in morphs:
            buf.extend(struct.pack("<H", form_to_index[form]))
            buf.extend(struct.pack("B", POS_TO_BYTE.get(pos, 0)))

    out_path = DATA_DIR / "eojeol_cache.bin"
    with open(out_path, "wb") as f:
        f.write(buf)

    print(f"\n  Written {len(selected)} entries to {out_path} ({len(buf):,} bytes)")


if __name__ == "__main__":
    main()
