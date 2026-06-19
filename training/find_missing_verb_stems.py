#!/usr/bin/env python3
"""
박/VV 클래스 탐지: NNP/NNG 동형이의에 가려 동사·형용사 읽기가 사전에서 빠진 어간 수집.

신호: Kiwi = [VV/VA/VX 어간(1~2음절)] + 동사성 어미,
      Garu = 동사 토큰(VV/VA/VX) 0개 (NNG span 또는 NNP+어미로 붕괴).
어간을 빈도순 집계 → 사전에 해당 VV/VA가 없는 것만 후보로 출력.
"""
import json, os, re, subprocess, sys, tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
MODEL = ROOT / "js" / "models" / "base.gmdl"
ANALYZE = ROOT / "target" / "release" / "examples" / "analyze_batch"
KOWIKI = Path(__file__).parent / "kowikitext.txt"
GOLD = Path(__file__).parent / "gold_testset" / "gold_testset.jsonl"
CONTENT_DICT = Path(__file__).parent / "codebook_data" / "content_dict.txt"

N_LINES = 120_000
VERBAL_ENDINGS = {"EP", "EF", "EC", "ETM", "ETN"}
PRED_POS = {"VV", "VA", "VX"}


def load_eojeols(n):
    seen, out = set(), []
    if KOWIKI.exists():
        with open(KOWIKI, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= n:
                    break
                for tok in line.split():
                    if re.search(r"[가-힣]", tok) and "\\" not in tok and tok not in seen:
                        seen.add(tok)
                        out.append(tok)
    if GOLD.exists():
        with open(GOLD, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                for tok in json.loads(line)["text"].split():
                    if re.search(r"[가-힣]", tok) and "\\" not in tok and tok not in seen:
                        seen.add(tok)
                        out.append(tok)
    return out


def run_garu(eojeols):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="utf-8", delete=False) as f:
        path = f.name
        f.write("\n".join(eojeols))
    try:
        env = dict(os.environ, GARU_MODEL=str(MODEL))
        r = subprocess.run([str(ANALYZE), path, "--json"], capture_output=True, text=True, env=env)
        if r.returncode != 0:
            print("garu failed:", r.stderr, file=sys.stderr); sys.exit(1)
        out = []
        for ln in r.stdout.strip().split("\n"):
            ln = ln.strip()
            out.append([(t[0], t[1]) for t in json.loads(ln)] if ln else [])
        return out
    finally:
        os.unlink(path)


def run_kiwi(eojeols):
    from kiwipiepy import Kiwi
    kiwi = Kiwi()
    res = []
    for item in kiwi.analyze(eojeols):
        toks = item[0][0]
        res.append([(t.form, t.tag if isinstance(t.tag, str) else t.tag.name) for t in toks])
    return res


def load_dict_pos():
    """word -> set of POS in content_dict.txt"""
    d = {}
    with open(CONTENT_DICT, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                d.setdefault(p[0], set()).add(p[1])
    return d


def main():
    print("[1/4] 어절 로딩...", file=sys.stderr)
    eojeols = load_eojeols(N_LINES)
    print(f"  고유 한글 어절: {len(eojeols):,}", file=sys.stderr)

    print("[2/4] Garu 분석...", file=sys.stderr)
    g = run_garu(eojeols)
    print("[3/4] Kiwi 분석...", file=sys.stderr)
    k = run_kiwi(eojeols)

    print("[4/4] collapse 탐지...", file=sys.stderr)
    dict_pos = load_dict_pos()
    # stem -> Counter of evidence eojeols; also count freq
    stem_freq = Counter()
    stem_examples = {}
    for eoj, gm, km in zip(eojeols, g, k):
        if not gm or not km:
            continue
        # Kiwi: first morpheme is predicate stem (1~2음절 순한글) + at least one verbal ending after
        first_surf, first_pos = km[0]
        if first_pos not in PRED_POS:
            continue
        if not (1 <= len(first_surf) <= 2 and all('가' <= c <= '힣' for c in first_surf)):
            continue
        if not any(p in VERBAL_ENDINGS for _, p in km[1:]):
            continue
        # Garu: zero predicate tokens (fully collapsed to noun reading)
        if any(p in PRED_POS for _, p in gm):
            continue
        # Garu first token noun-ish
        if gm[0][1] not in ("NNG", "NNP", "NNB"):
            continue
        # dict lacks this predicate reading
        existing = dict_pos.get(first_surf, set())
        if existing & PRED_POS:
            continue  # verb reading exists in dict but lost ranking — not this class
        stem_freq[first_surf] += 1
        if first_surf not in stem_examples:
            kparse = " ".join(f"{s}/{p}" for s, p in km)
            gparse = " ".join(f"{s}/{p}" for s, p in gm)
            stem_examples[first_surf] = (eoj, kparse, gparse)

    print(f"\n=== 후보 어간 (collapse 빈도순, 사전에 VV/VA/VX 없음) ===", file=sys.stderr)
    print(f"{'어간':>4} {'빈도':>5} {'사전POS':>10}  예시(어절 → Kiwi // Garu)", file=sys.stderr)
    print("-" * 100, file=sys.stderr)
    rows = []
    for stem, cnt in stem_freq.most_common(60):
        eoj, kp, gp = stem_examples[stem]
        dpos = ",".join(sorted(dict_pos.get(stem, set()))) or "(없음)"
        print(f"{stem:>4} {cnt:>5} {dpos:>10}  {eoj} → {kp} // {gp}", file=sys.stderr)
        rows.append({"stem": stem, "collapse_freq": cnt, "dict_pos": sorted(dict_pos.get(stem, set())),
                     "example_eojeol": eoj, "kiwi": kp, "garu": gp})
    print(json.dumps({"total_candidates": len(stem_freq), "top": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
