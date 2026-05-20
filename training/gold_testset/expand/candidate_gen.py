"""raw txt → Garu+Kiwi + ep_norm 비교 → pairs.jsonl."""
import argparse
import json
from pathlib import Path

from analyzers import run_garu, run_kiwi
from ep_norm import normalize_ep_morphemes


def _key(morphemes: list) -> tuple:
    norm = normalize_ep_morphemes(morphemes)
    return tuple(tuple(m) for m in norm)


def build_candidates(texts: list[str]) -> list[dict]:
    garu_out = run_garu(texts)
    kiwi_out = run_kiwi(texts)
    pairs = []
    for text, g, k in zip(texts, garu_out, kiwi_out):
        pairs.append({
            "text": text,
            "garu": g,
            "kiwi": k,
            "agree": _key(g) == _key(k),
        })
    return pairs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    with open(args.input) as f:
        texts = [line.strip() for line in f if line.strip()]
    pairs = build_candidates(texts)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    n_agree = sum(p["agree"] for p in pairs)
    print(f"Wrote {len(pairs)} to {args.output}")
    print(f"  agree: {n_agree} ({100*n_agree/len(pairs):.1f}%)")
    print(f"  disagree: {len(pairs)-n_agree} ({100*(len(pairs)-n_agree)/len(pairs):.1f}%)")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main()
    else:
        pairs = build_candidates(["오늘 날씨가 좋다."])
        assert len(pairs) == 1
        p = pairs[0]
        assert "text" in p and "garu" in p and "kiwi" in p and "agree" in p
        assert isinstance(p["agree"], bool)
        print(f"sample: {p}")
        print("OK")
