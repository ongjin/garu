"""NIKL NXMP/SXMP 코퍼스에서 raw 문장만 추출. gold 미사용."""
import json
import random
from pathlib import Path

NIKL_DIR = Path.home() / "Downloads" / "NIKL_MP(v1.1)"
FILES = {"NXMP": "NXMP1902008040.json", "SXMP": "SXMP1902008031.json"}


def load_nikl_raw(corpus: str, n: int, seed: int = 42,
                  min_len: int = 5, max_len: int = 50) -> list[str]:
    if corpus not in FILES:
        raise ValueError(f"corpus must be one of {list(FILES.keys())}")
    path = NIKL_DIR / FILES[corpus]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")

    with open(path) as f:
        data = json.load(f)
    candidates = []
    for doc in data["document"]:
        for sent in doc.get("sentence", []):
            text = sent.get("form", "").strip()
            if min_len <= len(text) <= max_len:
                candidates.append(text)

    rng = random.Random(seed)
    unique = list(dict.fromkeys(candidates))
    rng.shuffle(unique)
    return unique[:n]


if __name__ == "__main__":
    sents = load_nikl_raw(corpus="NXMP", n=20, seed=42)
    assert len(sents) == 20
    assert all(5 <= len(s) <= 50 for s in sents)
    assert len(set(sents)) == len(sents)
    print(f"NXMP[0]: {sents[0]}")
    sents_s = load_nikl_raw(corpus="SXMP", n=10, seed=42)
    assert len(sents_s) == 10
    print(f"SXMP[0]: {sents_s[0]}")
    print("OK")
