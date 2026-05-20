"""Compare Garu (WASM in Node) vs Kiwi (kiwipiepy / C++) throughput.

Same gold test sentences, warm-up + timed loop, reports sentences/sec, chars/sec,
and per-sentence latency.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "training" / "gold_testset" / "gold_testset.jsonl"


def bench_kiwi(sentences):
    from kiwipiepy import Kiwi

    t0 = time.perf_counter()
    kiwi = Kiwi()
    load_ms = (time.perf_counter() - t0) * 1000

    for s in sentences[: min(50, len(sentences))]:
        kiwi.analyze(s)

    t0 = time.perf_counter()
    for s in sentences:
        kiwi.analyze(s)
    elapsed = time.perf_counter() - t0

    chars = sum(len(s) for s in sentences)
    return {
        "tool": "kiwi",
        "version": __import__("kiwipiepy").__version__,
        "load_ms": load_ms,
        "total_ms": elapsed * 1000,
        "sentences": len(sentences),
        "chars": chars,
        "sentences_per_sec": len(sentences) / elapsed,
        "chars_per_sec": chars / elapsed,
    }


def bench_garu(n):
    out = subprocess.run(
        ["node", str(ROOT / "training" / "bench_garu.mjs"), str(GOLD), str(n)],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def fmt(r):
    return (
        f"  load:      {r['load_ms']:8.1f} ms\n"
        f"  total:     {r['total_ms']:8.1f} ms for {r['sentences']} sentences ({r['chars']} chars)\n"
        f"  speed:     {r['sentences_per_sec']:8.1f} sent/s   {r['chars_per_sec']:10.1f} char/s\n"
        f"  per sent:  {r['total_ms'] / r['sentences']:8.3f} ms"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=1000)
    args = ap.parse_args()

    with GOLD.open() as f:
        sentences = [json.loads(l)["text"] for l in f][: args.n]

    print(f"Benchmarking {len(sentences)} sentences from gold testset")
    print(f"(Garu: WASM in Node; Kiwi: native C++ via kiwipiepy)\n")

    print("=== Garu ===")
    g = bench_garu(args.n)
    print(fmt(g))

    print("\n=== Kiwi ===")
    k = bench_kiwi(sentences)
    print(fmt(k))

    print("\n=== Ratio ===")
    sps = k["sentences_per_sec"] / g["sentences_per_sec"]
    print(f"  Kiwi is {sps:.2f}x faster than Garu (sentences/sec)")


if __name__ == "__main__":
    main()
