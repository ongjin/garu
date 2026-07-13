"""NIKL 2021 → 재순위 학습용 train/dev 준비.

- NIKL MP 2021 (NX 뉴스 / SX 구어) 파싱, 태그는 eval_nikl_mp.normalize_pos로 세종 정규화
- v15k 골드 테스트셋과 문장 단위 일치(공백 무시 포함)하는 문장 전부 제외 (오염 차단)
- 중복 문장 5회 초과분 제거 (구어 상투구 편중 방지)
- train 40K NX + 40K SX / dev 3K NX + 3K SX (seed 42, 상호 배타)
- 출력: <out>/train.jsonl, dev.jsonl ({text, morphemes, src}) + *_texts.txt (dump_topk 입력)
"""
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent))
from eval_nikl_mp import normalize_pos  # noqa: E402

NIKL_DIR = Path(os.environ.get("NIKL_MP_DIR", Path.home() / "workspace/data/nikl_mp_2021"))
GOLD = BASE.parent / "gold_testset/gold_testset.jsonl"
TRAIN_PER_SRC = int(os.environ.get("RERANK_TRAIN_PER_SRC", 40_000))  # 10**9 = 전체
DEV_PER_SRC = 3_000
DUP_CAP = 5


def collapse(text):
    return "".join(text.split())


def load_gold_texts():
    exact, collapsed = set(), set()
    with open(GOLD) as f:
        for line in f:
            t = json.loads(line)["text"]
            exact.add(t)
            collapsed.add(collapse(t))
    return exact, collapsed


def load_src(path, src, gold_exact, gold_collapsed):
    with open(path) as f:
        data = json.load(f)
    out, contaminated, dup_counter = [], 0, Counter()
    for doc in data["document"]:
        for sent in doc["sentence"]:
            text = sent["form"]
            if not text or len(text) < 5 or len(text) > 200:
                continue
            if text in gold_exact or collapse(text) in gold_collapsed:
                contaminated += 1
                continue
            key = collapse(text)
            if dup_counter[key] >= DUP_CAP:
                continue
            morphemes = [
                [m["form"], normalize_pos(m["label"])]
                for m in (sent.get("MP") or sent.get("morpheme") or [])
                if m["form"].strip()
            ]
            if not morphemes:
                continue
            dup_counter[key] += 1
            out.append({"text": text, "morphemes": morphemes, "src": src})
    print(f"{src}: {len(out)} usable, {contaminated} contaminated(excluded)")
    return out


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    gold_exact, gold_collapsed = load_gold_texts()

    pools = {}
    for path in sorted(NIKL_DIR.glob("*.json")):
        src = "NX" if path.name.startswith("NX") else "SX"
        pools[src] = load_src(path, src, gold_exact, gold_collapsed)

    rng = random.Random(42)
    train, dev = [], []
    for src, pool in pools.items():
        rng.shuffle(pool)
        dev.extend(pool[:DEV_PER_SRC])
        train.extend(pool[DEV_PER_SRC:DEV_PER_SRC + TRAIN_PER_SRC])
    rng.shuffle(train)
    rng.shuffle(dev)

    for name, recs in [("train", train), ("dev", dev)]:
        with open(out_dir / f"{name}.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(out_dir / f"{name}_texts.txt", "w") as f:
            for r in recs:
                assert "\n" not in r["text"]
                f.write(r["text"] + "\n")
        print(f"{name}: {len(recs)} sentences")


if __name__ == "__main__":
    main()
