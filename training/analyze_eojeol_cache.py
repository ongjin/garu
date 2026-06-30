"""Analyze eojeol cache: per-entry contribution, distribution, pattern grouping.

Steps:
1. Parse training/codebook_data/eojeol_cache.bin (v2) → list of (eojeol, morphs)
2. Load NIKL MP gold → frequency per eojeol
3. Build cacheless GMDL (strip section 13 from base.gmdl)
4. Run Garu (cacheless) on all cached eojeols → predictions
5. For each entry compute correction_value, weighted_cv (freq * cv)
6. Print distribution + cumulative contribution curve + bottom-N pattern grouping
7. Write CSV: rank, eojeol, freq, cv, weighted_cv, gold, pred
"""
import csv
import gzip
import json
import os
import struct
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "training" / "codebook_data"
NIKL_DIR = Path(os.environ.get("NIKL_MP_DIR", str(Path.home() / "workspace" / "data" / "nikl_mp_2021")))
CACHE_BIN = DATA_DIR / "eojeol_cache.bin"
SOURCE_GMDL = ROOT / "models" / "codebook.gmdl"
OUTPUT_CSV = DATA_DIR / "cache_analysis.csv"

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


def normalize_pos(tag: str) -> str:
    if tag in POS_SET:
        return tag
    nikl_map = {
        "MMD": "MM", "MMN": "MM", "MMA": "MM",
        "NA": "NNG", "NAP": "NNG", "NF": "NNG", "NV": "VV",
    }
    if tag in nikl_map:
        return nikl_map[tag]
    base = tag.split("-")[0]
    if base in POS_SET:
        return base
    for p, d in [("V", "VV"), ("N", "NNG"), ("J", "JX"), ("E", "EF"), ("X", "XR")]:
        if tag.startswith(p):
            return d
    return "SW"


# ---------------------------------------------------------------------------
# 1. Parse eojeol_cache.bin (v2 format)
# ---------------------------------------------------------------------------
def parse_cache_bin(path: Path):
    data = path.read_bytes()
    marker = struct.unpack_from("<I", data, 0)[0]
    assert marker == 0xFFFFFFFF, f"Expected v1+ marker, got {marker:#x}"
    sub_ver = data[4]
    pos = 5
    st_len = struct.unpack_from("<I", data, pos)[0]; pos += 4
    string_table = data[pos:pos + st_len]; pos += st_len
    num_strings = struct.unpack_from("<H", data, pos)[0]; pos += 2
    off_fmt = "<I" if sub_ver >= 2 else "<H"
    off_size = 4 if sub_ver >= 2 else 2
    offsets = []
    for _ in range(num_strings + 1):
        offsets.append(struct.unpack_from(off_fmt, data, pos)[0])
        pos += off_size
    forms = [string_table[offsets[i]:offsets[i + 1]].decode("utf-8")
             for i in range(num_strings)]

    n_entries = struct.unpack_from("<I", data, pos)[0]; pos += 4
    entries = []
    for _ in range(n_entries):
        elen = data[pos]; pos += 1
        eojeol = data[pos:pos + elen].decode("utf-8"); pos += elen
        nm = data[pos]; pos += 1
        morphs = []
        for _ in range(nm):
            form_idx = struct.unpack_from("<H", data, pos)[0]; pos += 2
            pos_byte = data[pos]; pos += 1
            tag = POS_TAGS[pos_byte] if pos_byte < len(POS_TAGS) else "SW"
            morphs.append((forms[form_idx], tag))
        entries.append((eojeol, morphs))
    return entries


# ---------------------------------------------------------------------------
# 2. NIKL gold frequency
# ---------------------------------------------------------------------------
def load_eojeol_frequencies():
    freq = Counter()
    gold_analyses = defaultdict(Counter)
    for path in sorted(NIKL_DIR.glob("*.json")):
        with open(path) as f:
            doc_json = json.load(f)
        for doc in doc_json["document"]:
            if doc is None:
                continue
            for sent in (doc.get("sentence") or []):
                words = {w["id"]: w["form"] for w in (sent.get("word") or [])}
                wmorphs = defaultdict(list)
                for m in (sent.get("MP") or sent.get("morpheme") or []):
                    wid = m.get("word_id")
                    form = m.get("form", "").strip()
                    label = normalize_pos(m.get("label", ""))
                    if wid and form and label:
                        wmorphs[wid].append((form, label))
                for wid, wform in words.items():
                    if wid not in wmorphs:
                        continue
                    freq[wform] += 1
                    gold_analyses[wform][tuple(wmorphs[wid])] += 1
    return freq, gold_analyses


# ---------------------------------------------------------------------------
# 3. Cacheless GMDL: strip section 13
# ---------------------------------------------------------------------------
def build_cacheless_gmdl(src_path: Path, dst_path: Path):
    raw = src_path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    assert raw[:4] == b"GMDL", "Not a GMDL file"
    version = struct.unpack_from("<I", raw, 4)[0]
    out = bytearray(raw[:8])
    pos = 8
    stripped = False
    while pos < len(raw):
        sid = raw[pos]
        slen = struct.unpack_from("<I", raw, pos + 1)[0]
        section_end = pos + 5 + slen
        if sid == 13:
            stripped = True
            print(f"  Stripped section 13 (cache): {slen:,} bytes", file=sys.stderr)
        else:
            out.extend(raw[pos:section_end])
        pos = section_end
    dst_path.write_bytes(gzip.compress(bytes(out), 9))
    if not stripped:
        print("  WARN: section 13 not found in source GMDL", file=sys.stderr)


# ---------------------------------------------------------------------------
# 4. Run Garu (cacheless) on eojeols
# ---------------------------------------------------------------------------
def run_garu_cacheless(eojeols, model_path: Path):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        for e in eojeols:
            f.write(e + "\n")
        in_path = f.name

    env = os.environ.copy()
    env["GARU_MODEL"] = str(model_path)
    proc = subprocess.run(
        ["cargo", "run", "--release", "--example", "analyze_batch", "--", in_path],
        cwd=str(ROOT), capture_output=True, text=True, timeout=1800, env=env,
    )
    os.unlink(in_path)
    if proc.returncode != 0:
        print(f"Garu failed:\n{proc.stderr[-2000:]}", file=sys.stderr)
        sys.exit(1)

    analyses = []
    current = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line == "---":
            analyses.append(current)
            current = []
        elif line == "[]":
            analyses.append([])
        elif "\t" in line:
            form, pos = line.split("\t", 1)
            current.append((form, pos))
    return analyses


# ---------------------------------------------------------------------------
# 5. Correction value (set-based morpheme F1 component, as in build script)
# ---------------------------------------------------------------------------
def correction_value(pred, gold):
    p, g = set(pred), set(gold)
    return len(p - g) + len(g - p)


# ---------------------------------------------------------------------------
# 6. Pattern grouping for bottom-N
# ---------------------------------------------------------------------------
def group_patterns(rows):
    """Group cache entries by 'failure signature' to find patternable groups.
    Returns Counter of pattern → count.
    """
    patterns = Counter()
    for row in rows:
        gold = row["gold"]
        pred = row["pred"]
        if not gold:
            continue
        last_gold = gold[-1]
        last_pred = pred[-1] if pred else ("", "")

        # 1) last morpheme tag mismatch
        if last_gold[1] != last_pred[1]:
            patterns[f"마지막 형태소 태그 불일치: gold={last_gold[1]} vs pred={last_pred[1]}"] += 1

        # 2) missing tag in pred
        gold_tags = {t for _, t in gold}
        pred_tags = {t for _, t in pred}
        missing = gold_tags - pred_tags
        for t in missing:
            patterns[f"누락 태그: {t}"] += 1

        # 3) extra tag in pred
        extra = pred_tags - gold_tags
        for t in extra:
            patterns[f"잘못 태깅된 태그: {t}"] += 1

        # 4) segmentation count mismatch
        if len(gold) != len(pred):
            patterns[f"형태소 개수 불일치: gold={len(gold)} vs pred={len(pred)}"] += 1
    return patterns


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("[1/5] 캐시 바이너리 파싱...")
    entries = parse_cache_bin(CACHE_BIN)
    print(f"  엔트리 수: {len(entries)}")

    print("\n[2/5] NIKL gold 빈도 로딩...")
    freq, gold_analyses = load_eojeol_frequencies()
    print(f"  NIKL eojeol 수: {len(freq)}")

    print("\n[3/5] cacheless GMDL 생성...")
    tmp_gmdl = Path(tempfile.mkstemp(suffix=".gmdl")[1])
    build_cacheless_gmdl(SOURCE_GMDL, tmp_gmdl)
    print(f"  → {tmp_gmdl}")

    print(f"\n[4/5] Garu (cacheless) 분석 — {len(entries)} 어절...")
    eojeols = [e for e, _ in entries]
    preds = run_garu_cacheless(eojeols, tmp_gmdl)
    os.unlink(tmp_gmdl)
    assert len(preds) == len(entries), f"pred 수 불일치: {len(preds)} vs {len(entries)}"

    print("\n[5/5] 기여도 계산 + 분석...")
    rows = []
    for (eojeol, gold_morphs), pred in zip(entries, preds):
        f = freq.get(eojeol, 0)
        cv = correction_value(pred, gold_morphs)
        rows.append({
            "eojeol": eojeol,
            "freq": f,
            "cv": cv,
            "weighted_cv": cv * f,
            "gold": gold_morphs,
            "pred": pred,
        })

    rows.sort(key=lambda r: -r["weighted_cv"])
    total_wcv = sum(r["weighted_cv"] for r in rows)
    n = len(rows)

    # Distribution
    print(f"\n=== 캐시 기여도 분포 (n={n}) ===")
    print(f"  weighted_cv 합계: {total_wcv:,}")
    print(f"  cv == 0 (캐시 없어도 정답): {sum(1 for r in rows if r['cv'] == 0):,}")
    print(f"  freq == 0 (NIKL에 없음): {sum(1 for r in rows if r['freq'] == 0):,}")
    print()

    cum = 0
    milestones = [100, 500, 1000, 2000, 3000, 5000, 7000, 10000]
    print(f"  {'상위 N':>8s}  {'누적 weighted_cv':>18s}  {'%':>6s}")
    for r in rows:
        cum += r["weighted_cv"]
    cum2 = 0
    for i, r in enumerate(rows, 1):
        cum2 += r["weighted_cv"]
        if i in milestones or i == n:
            pct = (cum2 / total_wcv * 100) if total_wcv else 0
            print(f"  {i:>8,}  {cum2:>18,}  {pct:>5.1f}%")

    # Top 20 + Bottom 20
    print("\n=== 상위 20 항목 ===")
    for i, r in enumerate(rows[:20], 1):
        g = " + ".join(f"{f}/{p}" for f, p in r["gold"])
        print(f"  {i:>3d}. {r['eojeol']:<10s}  freq={r['freq']:>4d} cv={r['cv']} wcv={r['weighted_cv']:>5d}  → {g}")

    print("\n=== 하위 20 항목 (절단 후보) ===")
    for i, r in enumerate(rows[-20:], n - 19):
        g = " + ".join(f"{f}/{p}" for f, p in r["gold"])
        print(f"  {i:>5d}. {r['eojeol']:<10s}  freq={r['freq']:>4d} cv={r['cv']} wcv={r['weighted_cv']:>3d}  → {g}")

    # Bottom 5000 pattern analysis
    print("\n=== 하위 5000 패턴 그룹 (상위 25) ===")
    bottom = rows[-5000:] if n > 5000 else rows[n // 2:]
    patterns = group_patterns(bottom)
    for pat, cnt in patterns.most_common(25):
        print(f"  {cnt:>5d}  {pat}")

    # CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "eojeol", "freq", "cv", "weighted_cv", "gold", "pred"])
        for i, r in enumerate(rows, 1):
            w.writerow([
                i, r["eojeol"], r["freq"], r["cv"], r["weighted_cv"],
                " + ".join(f"{f}/{p}" for f, p in r["gold"]),
                " + ".join(f"{f}/{p}" for f, p in r["pred"]),
            ])
    print(f"\nCSV 저장: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
