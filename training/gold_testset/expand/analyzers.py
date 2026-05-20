"""Garu binary + kiwipiepy wrapper."""
import json
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
GARU_MODEL = ROOT / "js" / "models" / "base.gmdl"
GARU_BIN = ROOT / "target" / "release" / "examples" / "analyze_batch"


def run_garu(texts: list[str]) -> list[list]:
    if not GARU_BIN.exists():
        raise FileNotFoundError(
            f"{GARU_BIN} not found. Run: cargo build --release --example analyze_batch"
        )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for t in texts:
            f.write(t + "\n")
        in_path = f.name
    try:
        env = {**os.environ, "GARU_MODEL": str(GARU_MODEL)}
        result = subprocess.run(
            [str(GARU_BIN), in_path, "--json"],
            capture_output=True, text=True, env=env, check=True,
        )
        return [json.loads(line) for line in result.stdout.strip().split("\n")]
    finally:
        os.unlink(in_path)


def run_kiwi(texts: list[str]) -> list[list]:
    from kiwipiepy import Kiwi
    kw = Kiwi()
    out = []
    for t in texts:
        r = kw.analyze(t)
        if not r:
            out.append([])
            continue
        tokens = [[m.form, m.tag.replace("-I", "").replace("-R", "")]
                  for m in r[0][0]]
        out.append(tokens)
    return out


if __name__ == "__main__":
    samples = ["오늘 날씨가 좋다.", "밥을 먹었다."]
    g = run_garu(samples)
    assert len(g) == 2
    print(f"Garu[0]: {g[0]}")
    k = run_kiwi(samples)
    assert len(k) == 2
    print(f"Kiwi[0]: {k[0]}")
    print("OK")
