"""Scenario A 정확 측정: 캐시 10K → top-5K 컷 후 골드셋 F1 변화.

1) cache_analysis.csv 에서 상위 5000 eojeol 선별
2) eojeol_cache.bin (v2) 에서 그 5000개 추출 → 트림된 캐시 바이너리 생성
3) base.gmdl 의 section 13 만 교체 → /tmp/base_trimmed.gmdl
4) 골드셋 5K 문장에 대해 baseline / trimmed 각각 F1 측정
5) 비교 리포트
"""
import csv
import gzip
import json
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "training" / "codebook_data"
GOLD_PATH = ROOT / "training" / "gold_testset" / "gold_testset.jsonl"
CSV_PATH = DATA_DIR / "cache_analysis.csv"
CACHE_BIN = DATA_DIR / "eojeol_cache.bin"
BASELINE_GMDL = ROOT / "js" / "models" / "base.gmdl"

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


def load_top_n_eojeols(n: int) -> set:
    """cache_analysis.csv 의 rank ≤ n 인 eojeol 셋."""
    keep = set()
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["rank"]) <= n:
                keep.add(row["eojeol"])
    return keep


def parse_cache_bin(path: Path):
    data = path.read_bytes()
    marker = struct.unpack_from("<I", data, 0)[0]
    assert marker == 0xFFFFFFFF
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


def encode_cache_v2(selected_entries) -> bytes:
    """build_eojeol_cache.py 와 동일한 v2 포맷으로 인코딩."""
    all_forms = set()
    for _, morphs in selected_entries:
        for form, _ in morphs:
            all_forms.add(form)
    sorted_forms = sorted(all_forms, key=lambda s: s.encode("utf-8"))
    form_to_idx = {f: i for i, f in enumerate(sorted_forms)}
    string_table = bytearray()
    string_offsets = []
    for form in sorted_forms:
        string_offsets.append(len(string_table))
        string_table.extend(form.encode("utf-8"))
    string_offsets.append(len(string_table))

    buf = bytearray()
    buf.extend(struct.pack("<I", 0xFFFFFFFF))
    buf.extend(struct.pack("B", 2))
    buf.extend(struct.pack("<I", len(string_table)))
    buf.extend(string_table)
    buf.extend(struct.pack("<H", len(sorted_forms)))
    for off in string_offsets:
        buf.extend(struct.pack("<I", off))
    buf.extend(struct.pack("<I", len(selected_entries)))
    for eojeol, morphs in selected_entries:
        eb = eojeol.encode("utf-8")
        buf.extend(struct.pack("B", len(eb)))
        buf.extend(eb)
        buf.extend(struct.pack("B", len(morphs)))
        for form, pos in morphs:
            buf.extend(struct.pack("<H", form_to_idx[form]))
            buf.extend(struct.pack("B", POS_TO_BYTE.get(pos, 0)))
    return bytes(buf)


def replace_section_13(src_gmdl: Path, new_section_data: bytes, dst_gmdl: Path):
    raw = src_gmdl.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    assert raw[:4] == b"GMDL"
    out = bytearray(raw[:8])
    pos = 8
    replaced = False
    while pos < len(raw):
        sid = raw[pos]
        slen = struct.unpack_from("<I", raw, pos + 1)[0]
        section_end = pos + 5 + slen
        if sid == 13:
            out.append(13)
            out.extend(struct.pack("<I", len(new_section_data)))
            out.extend(new_section_data)
            replaced = True
        else:
            out.extend(raw[pos:section_end])
        pos = section_end
    if not replaced:
        # Append at the end if not found
        out.append(13)
        out.extend(struct.pack("<I", len(new_section_data)))
        out.extend(new_section_data)
    dst_gmdl.write_bytes(gzip.compress(bytes(out), 9))


def run_garu(texts, model_path: Path):
    binary = ROOT / "target" / "release" / "examples" / "analyze_batch"
    if not binary.exists():
        print("Building analyze_batch...", file=sys.stderr)
        subprocess.run(
            ["cargo", "build", "--release", "--example", "analyze_batch"],
            cwd=str(ROOT), check=True,
        )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        for t in texts:
            f.write(t + "\n")
        in_path = f.name
    env = {**os.environ, "GARU_MODEL": str(model_path)}
    proc = subprocess.run(
        [str(binary), in_path, "--json"],
        capture_output=True, text=True, env=env, timeout=600,
    )
    os.unlink(in_path)
    if proc.returncode != 0:
        print(f"Garu failed:\n{proc.stderr[-2000:]}", file=sys.stderr)
        sys.exit(1)
    return [json.loads(l) for l in proc.stdout.strip().split("\n")]


def compute_f1(pred, gold):
    tp = fp = fn = 0
    for p_tokens, g_tokens in zip(pred, gold):
        p_set, g_set = {}, {}
        for form, pos in p_tokens:
            k = (form, pos)
            p_set[k] = p_set.get(k, 0) + 1
        for form, pos in g_tokens:
            k = (form, pos)
            g_set[k] = g_set.get(k, 0) + 1
        for k in set(list(p_set.keys()) + list(g_set.keys())):
            pc, gc = p_set.get(k, 0), g_set.get(k, 0)
            m = min(pc, gc)
            tp += m
            fp += pc - m
            fn += gc - m
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    return prec, rec, f1, tp, fp, fn


def main():
    N_KEEP = int(sys.argv[1]) if len(sys.argv) > 1 else 5000

    print(f"[1/5] 상위 {N_KEEP} eojeol 선별...")
    keep_set = load_top_n_eojeols(N_KEEP)
    print(f"  선택된 eojeol: {len(keep_set)}")

    print("\n[2/5] 캐시 바이너리 파싱 + 필터링...")
    all_entries = parse_cache_bin(CACHE_BIN)
    selected = [(e, m) for e, m in all_entries if e in keep_set]
    print(f"  전체 {len(all_entries)} → 트림 {len(selected)}")

    print("\n[3/5] 트림된 cache + 트림된 GMDL 생성...")
    trimmed_section = encode_cache_v2(selected)
    print(f"  트림된 section 13: {len(trimmed_section):,} bytes (원본 235,260)")
    trimmed_gmdl = Path(tempfile.mkstemp(suffix="_trimmed.gmdl")[1])
    replace_section_13(BASELINE_GMDL, trimmed_section, trimmed_gmdl)
    print(f"  trimmed gmdl: {trimmed_gmdl} ({trimmed_gmdl.stat().st_size:,} bytes)")
    print(f"  baseline gmdl: {BASELINE_GMDL.stat().st_size:,} bytes")

    print("\n[4/5] 골드셋 로드...")
    records = []
    with open(GOLD_PATH) as f:
        for line in f:
            records.append(json.loads(line))
    texts = [r["text"] for r in records]
    gold = [r["morphemes"] for r in records]
    print(f"  문장 수: {len(texts)}")

    print("\n[5/5] F1 평가 (baseline + trimmed)...")
    print("  → baseline 분석 중...")
    baseline_preds = run_garu(texts, BASELINE_GMDL)
    print("  → trimmed 분석 중...")
    trimmed_preds = run_garu(texts, trimmed_gmdl)

    # Compute
    bp, br, bf1, btp, bfp, bfn = compute_f1(baseline_preds, gold)
    tp, tr, tf1, ttp, tfp, tfn = compute_f1(trimmed_preds, gold)

    print("\n=== 결과 ===\n")
    print(f"{'설정':<25} {'P':>8} {'R':>8} {'F1':>8} {'TP':>8} {'FP':>8} {'FN':>8}")
    print("-" * 80)
    print(f"{'Baseline (10K cache)':<25} {bp:>8.4f} {br:>8.4f} {bf1:>8.4f} {btp:>8} {bfp:>8} {bfn:>8}")
    print(f"{'Trimmed (' + str(N_KEEP) + ' cache)':<25} {tp:>8.4f} {tr:>8.4f} {tf1:>8.4f} {ttp:>8} {tfp:>8} {tfn:>8}")
    print()
    delta_f1 = (tf1 - bf1) * 100
    delta_size = trimmed_gmdl.stat().st_size - BASELINE_GMDL.stat().st_size
    print(f"  F1 변화: {delta_f1:+.3f}%p")
    print(f"  GMDL 사이즈 변화: {delta_size:+,} bytes ({delta_size / 1024:+.1f} KB)")

    # Domain별
    print("\n=== Domain별 F1 비교 ===")
    domains = {}
    for i, rec in enumerate(records):
        d = rec.get("domain", "?")
        domains.setdefault(d, []).append(i)
    print(f"\n{'Domain':<12} {'Baseline':>10} {'Trimmed':>10} {'Δ':>8}")
    print("-" * 44)
    for d in ["뉴스", "일상", "SNS", "기술", "문학", "엣지케이스"]:
        if d not in domains:
            continue
        idxs = domains[d]
        b_pred = [baseline_preds[i] for i in idxs]
        t_pred = [trimmed_preds[i] for i in idxs]
        g = [gold[i] for i in idxs]
        _, _, bf1d, *_ = compute_f1(b_pred, g)
        _, _, tf1d, *_ = compute_f1(t_pred, g)
        print(f"  {d:<10} {bf1d:>10.4f} {tf1d:>10.4f} {(tf1d - bf1d) * 100:>+7.3f}%p")

    # Sentence-level regression analysis
    print("\n=== 문장별 회귀 분석 ===")
    regressed = 0
    improved = 0
    for bp_s, tp_s, g_s in zip(baseline_preds, trimmed_preds, gold):
        _, _, bf1s, *_ = compute_f1([bp_s], [g_s])
        _, _, tf1s, *_ = compute_f1([tp_s], [g_s])
        if tf1s < bf1s - 1e-6:
            regressed += 1
        elif tf1s > bf1s + 1e-6:
            improved += 1
    print(f"  악화 문장: {regressed}")
    print(f"  개선 문장: {improved}")
    print(f"  변화 없음: {len(texts) - regressed - improved}")

    os.unlink(trimmed_gmdl)


if __name__ == "__main__":
    main()
