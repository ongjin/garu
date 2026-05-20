"""kowikitext에서 raw 문장 추출."""
import re
import random
from pathlib import Path

KOWIKI_PATH = Path.home() / "Korpora" / "kowikitext" / "kowikitext_20200920.train"
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def load_kowiki_raw(n: int, seed: int = 42,
                    min_len: int = 5, max_len: int = 50) -> list[str]:
    if not KOWIKI_PATH.exists():
        raise FileNotFoundError(f"{KOWIKI_PATH} not found")

    candidates = []
    with open(KOWIKI_PATH) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("="):
                continue
            for sent in _SENT_SPLIT.split(line):
                sent = sent.strip()
                if min_len <= len(sent) <= max_len:
                    candidates.append(sent)
            if len(candidates) > n * 50:
                break

    rng = random.Random(seed)
    unique = list(dict.fromkeys(candidates))
    rng.shuffle(unique)
    return unique[:n]


if __name__ == "__main__":
    sents = load_kowiki_raw(n=20, seed=42)
    assert len(sents) == 20
    assert all(5 <= len(s) <= 50 for s in sents)
    assert len(set(sents)) == len(sents)
    print(f"kowiki[0]: {sents[0]}")
    print("OK")
