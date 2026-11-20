"""Replay polite promise endings on the retained NIKL 2021 document splits."""
import gzip
import hashlib
import json
import os
import re
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

from prepare import documents, normalized, normalize_pos

ROOT = Path(__file__).resolve().parents[2]
DATA = Path.home() / "workspace/data"
REFS = DATA / "korean_context_pos/20261020"
BINARY = ROOT / "target/release/examples/analyze_batch"
ENDINGS = {"ㄹ게", "을게", "ㄹ게요", "을게요"}
SURFACE = re.compile(r"([가-힣])게요(?:$|[^\w])")


def promise_index(tokens):
    return next((i for i, (form, pos) in enumerate(tokens)
                 if pos == "EF" and form.replace("ᆯ", "ㄹ") in ENDINGS), None)


def verb_stem(tokens, ending):
    if ending is None:
        return None
    for form, pos in reversed(tokens[:ending]):
        if pos != "EP":
            return form if pos in ("VV", "VX", "XSV") else None
    return None


def read_examples():
    refs = {}
    for split in ("train", "dev"):
        with gzip.open(REFS / f"{split}.refs.jsonl.gz", "rt") as source:
            for line in source:
                row = json.loads(line)
                refs[row["id"]] = split, row["text_sha256"]

    rows = []
    found = 0
    for path in sorted((DATA / "nikl_mp_2021").glob("*.json")):
        for doc in documents(path):
            for sentence in (doc or {}).get("sentence", []):
                ref = refs.get(sentence["id"])
                if ref is None:
                    continue
                found += 1
                text = sentence["form"]
                if "게요" not in text:
                    continue
                split, digest = ref
                assert hashlib.sha256(text.encode()).hexdigest() == digest
                for word in sentence["word"]:
                    if "게요" not in word["form"]:
                        continue
                    gold = [(m["form"], normalize_pos(m["label"])) for m in (sentence.get("MP") or sentence["morpheme"])
                            if m["word_id"] == word["id"]]
                    match = SURFACE.search(word["form"])
                    # The modern Hangul syllable index ends in 8 (modulo 28) for a ㄹ coda.
                    eligible = bool(match and (ord(match[1]) - ord("가")) % 28 == 8)
                    rows.append((text, word["begin"], word["end"], split, gold, eligible))
    assert found == len(refs), "Some retained sentence IDs were missing from the corpus"
    return rows


def main():
    rows = read_examples()
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as source:
        source.writelines(r[0] + "\n" for r in rows)
        source.flush()
        for mode in ("legacy", "standard"):
            env = {**os.environ, "GARU_MODEL": str(ROOT / "js/models/base.gmdl"), "GARU_PROMISE_ENDINGS": mode}
            output = subprocess.check_output([BINARY, source.name, "--json-spans"], env=env, text=True).splitlines()
            assert len(output) == len(rows)
            stats = {"train": Counter(), "dev": Counter()}
            for (_, start, end, split, gold, eligible), line in zip(rows, output):
                pred = [(f, p) for f, p, a, b in json.loads(line) if start <= a and b <= end]
                expected = promise_index(gold)
                actual = promise_index(pred)
                tally = stats[split]
                tally["gold"] += expected is not None
                tally["other"] += expected is None
                tally["eligible"] += eligible
                tally["eligible_gold"] += eligible and expected is not None
                tally["recognized"] += expected is not None and actual is not None
                tally["false_positive"] += expected is None and actual is not None
                if expected is not None:
                    stem = verb_stem(gold, expected)
                    tally["stem_match"] += stem is not None and stem == verb_stem(pred, actual)
                    tally["word_match"] += normalized(gold) == normalized(pred)
            print(mode, json.dumps(stats, sort_keys=True))


if __name__ == "__main__":
    main()
