"""Garu-only NIKL MP F1 eval — uses binary path, no cargo run, no Kiwi.

Usage: python3 training/eval_nikl_garu_only.py [binary_name] [n_sentences]
"""
import json, os, random, subprocess, sys, tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NIKL_DIR = Path(os.environ.get("NIKL_MP_DIR", str(Path.home() / "workspace" / "data" / "nikl_mp_2021")))
MODEL = ROOT / "js" / "models" / "base.gmdl"
CNN = ROOT / "js" / "models" / "cnn2.bin"

POS_TAGS = set([
    "NNG", "NNP", "NNB", "NR", "NP",
    "VV", "VA", "VX", "VCP", "VCN",
    "MAG", "MAJ", "MM", "IC",
    "JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ", "JX", "JC",
    "EP", "EF", "EC", "ETN", "ETM",
    "XPN", "XSN", "XSV", "XSA", "XR",
    "SF", "SP", "SS", "SE", "SO", "SW", "SH", "SL", "SN",
])

NIKL_MAP = {'MMD': 'MM', 'MMN': 'MM', 'MMA': 'MM', 'NA': 'NNG', 'NAP': 'NNG', 'NF': 'NNG', 'NV': 'VV'}

def normalize_pos(tag):
    if tag in POS_TAGS: return tag
    if tag in NIKL_MAP: return NIKL_MAP[tag]
    base = tag.split('-')[0]
    if base in POS_TAGS: return base
    for p, d in [('V', 'VV'), ('N', 'NNG'), ('J', 'JX'), ('E', 'EF'), ('X', 'XR')]:
        if tag.startswith(p): return d
    return 'SW'

def load_nikl(max_n=2000):
    sentences = []
    for path in sorted(NIKL_DIR.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        for doc in data["document"]:
            for sent in doc["sentence"]:
                text = sent["form"]
                if not text or len(text) < 5 or len(text) > 200: continue
                morphs = []
                for m in (sent.get("MP") or sent.get("morpheme") or []):
                    if m["form"].strip():
                        morphs.append((m["form"], normalize_pos(m["label"])))
                if morphs:
                    sentences.append((text, morphs))
    random.seed(42)
    if len(sentences) > max_n:
        sentences = random.sample(sentences, max_n)
    return sentences

def run_garu(texts, binary):
    inp = ROOT / "training" / f"_nikl_input_{os.getpid()}.txt"
    inp.write_text("\n".join(texts))
    env = {**os.environ, "GARU_MODEL": str(MODEL), "GARU_CNN": str(CNN)}
    r = subprocess.run([str(binary), str(inp), "--json"], capture_output=True, text=True, env=env)
    inp.unlink()
    if r.returncode != 0:
        sys.exit(f"binary failed: {r.stderr[:500]}")
    return [json.loads(l) for l in r.stdout.strip().split("\n")]

def f1(pred, gold):
    tp = fp = fn = 0
    for p_tokens, g_tokens in zip(pred, gold):
        ps, gs = Counter(), Counter()
        for f, p in p_tokens: ps[(f, p)] += 1
        for f, p in g_tokens: gs[(f, p)] += 1
        for k in set(list(ps) + list(gs)):
            pc, gc = ps[k], gs[k]
            m = min(pc, gc)
            tp += m; fp += pc - m; fn += gc - m
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    return prec, rec, (2 * prec * rec / (prec + rec) if (prec + rec) else 0)

def main():
    binary_name = sys.argv[1] if len(sys.argv) > 1 else "analyze_batch"
    n_arg = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    binary = ROOT / "target" / "release" / "examples" / binary_name
    if not binary.exists():
        sys.exit(f"binary not found: {binary}")

    sentences = load_nikl(n_arg)
    print(f"Loaded {len(sentences)} NIKL sentences", file=sys.stderr)
    texts = [s[0] for s in sentences]
    gold = [s[1] for s in sentences]
    pred = run_garu(texts, binary)
    p, r, fv = f1(pred, gold)
    print(f"{binary_name:30} P={p:.4f} R={r:.4f} F1={fv:.4f}  ({len(pred)} NIKL sentences)")

if __name__ == "__main__":
    main()
