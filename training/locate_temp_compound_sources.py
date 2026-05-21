"""임시 합성명사(unlisted 91개)가 Garu 데이터 소스 어디에 등재돼 있는지 추적.

출력: training/temp_compound_sources.tsv
  surface | freq | in_cache | in_dict | cache_form | split
"""
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "training" / "codebook_data"
VERIFIED_JSONL = ROOT / "training" / "temp_compound_verified.jsonl"
CACHE_BIN = DATA_DIR / "eojeol_cache.bin"
CONTENT_DICT = DATA_DIR / "content_dict.txt"
OUTPUT_TSV = ROOT / "training" / "temp_compound_sources.tsv"

POS_TAGS = [
    "NNG", "NNP", "NNB", "NR", "NP",
    "VV", "VA", "VX", "VCP", "VCN",
    "MAG", "MAJ", "MM", "IC",
    "JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ", "JX", "JC",
    "EP", "EF", "EC", "ETN", "ETM",
    "XPN", "XSN", "XSV", "XSA", "XR",
    "SF", "SP", "SS", "SE", "SO", "SW", "SH", "SL", "SN",
]


# ---------------------------------------------------------------------------
# 1. 91개 미등재 임시 합성명사 로드
# ---------------------------------------------------------------------------
def load_unlisted():
    items = []
    with open(VERIFIED_JSONL, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("stdict") == "unlisted":
                items.append(obj)
    return items


# ---------------------------------------------------------------------------
# 2. eojeol_cache.bin 파싱 (analyze_eojeol_cache.py 와 동일 포맷)
# ---------------------------------------------------------------------------
def parse_cache_bin(path: Path):
    """Returns dict: eojeol -> list of (form, pos) morphemes."""
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
    cache = {}
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
        cache[eojeol] = morphs
    return cache


# ---------------------------------------------------------------------------
# 3. content_dict.txt — surface -> set of POS lines
# ---------------------------------------------------------------------------
def load_content_dict(path: Path):
    """Returns dict: surface -> list of (pos, freq) for each line."""
    entries = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            surface = parts[0]
            pos = parts[1] if len(parts) > 1 else ""
            freq = int(parts[2]) if len(parts) > 2 else 0
            if surface not in entries:
                entries[surface] = []
            entries[surface].append((pos, freq))
    return entries


# ---------------------------------------------------------------------------
# 4. 캐시에서 합성명사 검색: 어절 자체 + 어절+조사 형태
# ---------------------------------------------------------------------------
def find_in_cache(surface: str, cache: dict):
    """
    Returns (found: bool, cache_form: str)
    cache_form describes how the surface appears:
      - "bare:NNG"       — 어절 자체가 단일 NNG
      - "bare:NNG+NNG"   — 어절이 복수 NNG로 분석 (이미 복합)
      - "bare:<morphs>"  — 다른 분석
      - "josa:<eojeol>"  — surface+조사 형태로 등재
      - ""               — 미발견
    """
    # 1) 어절 자체
    if surface in cache:
        morphs = cache[surface]
        if len(morphs) == 1 and morphs[0][1] == "NNG":
            return True, f"bare:NNG"
        else:
            desc = "+".join(f"{f}/{p}" for f, p in morphs)
            return True, f"bare:{desc}"

    # 2) surface+조사 형태 (surface가 prefix인 캐시 항목)
    hits = []
    for eojeol in cache:
        if eojeol.startswith(surface) and len(eojeol) > len(surface):
            suffix = eojeol[len(surface):]
            morphs = cache[eojeol]
            # 첫 형태소가 surface 자체인지 확인
            if morphs and morphs[0][0] == surface:
                hits.append(eojeol)
    if hits:
        sample = hits[0]
        return True, f"josa:{sample}"

    return False, ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("[1/4] 미등재 91개 로딩...", file=sys.stderr)
    items = load_unlisted()
    print(f"  unlisted 수: {len(items)}", file=sys.stderr)

    print("[2/4] eojeol_cache.bin 파싱...", file=sys.stderr)
    cache = parse_cache_bin(CACHE_BIN)
    print(f"  캐시 엔트리: {len(cache):,}", file=sys.stderr)

    print("[3/4] content_dict.txt 로딩...", file=sys.stderr)
    cdict = load_content_dict(CONTENT_DICT)
    print(f"  content_dict 표제어: {len(cdict):,}", file=sys.stderr)

    print("[4/4] 교차 검색...", file=sys.stderr)

    rows = []
    for item in items:
        surface = item["surface"]
        freq = item["freq"]
        split = item.get("split", [])
        split_str = "+".join(f"{s[0]}/{s[1]}" for s in split)

        # 캐시 조회
        in_cache, cache_form = find_in_cache(surface, cache)

        # content_dict 조회 (NNG 태그 있는지)
        dict_entries = cdict.get(surface, [])
        in_dict = any(pos == "NNG" for pos, _ in dict_entries)

        rows.append({
            "surface": surface,
            "freq": freq,
            "in_cache": "Y" if in_cache else "N",
            "in_dict": "Y" if in_dict else "N",
            "cache_form": cache_form,
            "split": split_str,
        })

    # 통계
    cache_only = sum(1 for r in rows if r["in_cache"] == "Y" and r["in_dict"] == "N")
    dict_only  = sum(1 for r in rows if r["in_cache"] == "N" and r["in_dict"] == "Y")
    both       = sum(1 for r in rows if r["in_cache"] == "Y" and r["in_dict"] == "Y")
    neither    = sum(1 for r in rows if r["in_cache"] == "N" and r["in_dict"] == "N")

    print("\n=== 데이터 소스 통계 ===", file=sys.stderr)
    print(f"  cache only : {cache_only}", file=sys.stderr)
    print(f"  dict only  : {dict_only}", file=sys.stderr)
    print(f"  both       : {both}", file=sys.stderr)
    print(f"  neither    : {neither}", file=sys.stderr)
    print(f"  합계       : {len(rows)}", file=sys.stderr)

    # 출력 (freq 내림차순)
    rows.sort(key=lambda r: -r["freq"])

    with open(OUTPUT_TSV, "w", encoding="utf-8") as f:
        f.write("surface\tfreq\tin_cache\tin_dict\tcache_form\tsplit\n")
        for r in rows:
            f.write(f"{r['surface']}\t{r['freq']}\t{r['in_cache']}\t{r['in_dict']}\t{r['cache_form']}\t{r['split']}\n")

    print(f"\nTSV 저장: {OUTPUT_TSV}", file=sys.stderr)

    # 상위 20 출력 (stdout)
    print("\nsurface\tfreq\tin_cache\tin_dict\tcache_form")
    for r in rows[:20]:
        print(f"{r['surface']}\t{r['freq']}\t{r['in_cache']}\t{r['in_dict']}\t{r['cache_form']}")


if __name__ == "__main__":
    main()
