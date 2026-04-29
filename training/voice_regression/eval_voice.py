"""Evaluate Garu on voice-assistant regression suite.

Usage:
    python3 training/voice_regression/eval_voice.py [path/to/seed.jsonl]

Compares (text, pos) tuples token-by-token. Reports exact-match per sentence
and overall token F1.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
SEED = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent / "voice_seed.jsonl")
EXAMPLE = ROOT / "target" / "release" / "examples" / "analyze_batch"
MODEL = ROOT / "js" / "models" / "base.gmdl"
CNN = ROOT / "js" / "models" / "cnn2.bin"

cases = []
with open(SEED) as f:
    for line in f:
        line = line.strip()
        if not line: continue
        cases.append(json.loads(line))

# Write inputs to temp file
import tempfile
with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
    for c in cases:
        f.write(c["text"] + "\n")
    input_path = f.name

env = {"GARU_MODEL": str(MODEL), "GARU_CNN": str(CNN), "PATH": "/usr/bin:/bin"}
proc = subprocess.run(
    [str(EXAMPLE), input_path, "--json"],
    capture_output=True, text=True, env=env,
)
preds = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]

assert len(preds) == len(cases), f"pred {len(preds)} != cases {len(cases)}"

exact = 0
tp = fp = fn = 0
fails = []
for c, p in zip(cases, preds):
    g = [tuple(x) for x in c["gold"]]
    pp = [tuple(x) for x in p]
    if g == pp:
        exact += 1
    else:
        fails.append((c["text"], g, pp))
    g_set = set(g)
    p_set = set(pp)
    tp += len(g_set & p_set)
    fp += len(p_set - g_set)
    fn += len(g_set - p_set)

prec = tp / (tp + fp) if tp + fp else 0.0
rec = tp / (tp + fn) if tp + fn else 0.0
f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

print(f"Voice regression suite ({len(cases)} sentences)")
print(f"  Exact match: {exact}/{len(cases)} ({100 * exact / len(cases):.1f}%)")
print(f"  Token F1: {f1:.4f} (P={prec:.4f}, R={rec:.4f})")
if fails and "--fails" in sys.argv:
    print(f"\nFailures ({len(fails)}):")
    for text, g, pp in fails[:30]:
        print(f"  {text!r}")
        print(f"    GOLD: {g}")
        print(f"    PRED: {pp}")
