"""고정 세트 예측과 문장별 채점.

    # 모델로 예측하고 문장별 (tp, pred, gold)를 기록
    python3 training/gate/score.py predict --model js/models/base.gmdl --out DIR [--sets gold,sx25,nx25]
    # 이미 있는 예측(DIR/<set>.pred.jsonl)만 다시 채점
    python3 training/gate/score.py count --out DIR

산출물: DIR/<set>.pred.jsonl ({id, tokens}), DIR/<set>.counts.jsonl ({id, doc, stratum, tp, pred, gold}).
양쪽 모두 ep_norm 후 (형태, 품사) 다중집합으로 비교한다. 입력 줄 trim은 analyze_batch가 한다.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sets import ROOT, SETS, load  # noqa: E402

sys.path.insert(0, str(ROOT / "training" / "gold_testset" / "expand"))
from ep_norm import normalize_ep_morphemes  # noqa: E402

BIN = ROOT / "target" / "release" / "examples" / "analyze_batch"


def run_analyzer(texts, model, binary=BIN):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("\n".join(texts))
        path = f.name
    try:
        r = subprocess.run([str(binary), path, "--json"], capture_output=True, text=True,
                           env={**os.environ, "GARU_MODEL": str(model)})
    finally:
        os.unlink(path)
    if r.returncode != 0:
        sys.exit(f"analyze_batch failed: {r.stderr[:500]}")
    preds = [json.loads(line) for line in r.stdout.strip().split("\n")]
    assert len(preds) == len(texts), (len(preds), len(texts))
    return preds


def counts(pred, gold):
    """ep_norm 후 다중집합 (tp, 예측 수, 정답 수)."""
    p = Counter(map(tuple, normalize_ep_morphemes([list(t[:2]) for t in pred])))
    g = Counter(map(tuple, normalize_ep_morphemes([list(t[:2]) for t in gold])))
    tp = sum((p & g).values())
    return tp, sum(p.values()), sum(g.values())


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def score_set(name, preds_by_id, out_dir, records=None):
    records = records or load(name)
    rows = []
    for rec in records:
        tp, npred, ngold = counts(preds_by_id[rec["id"]], rec["gold"])
        rows.append({"id": rec["id"], "doc": rec["doc"], "stratum": rec["stratum"],
                     "tp": tp, "pred": npred, "gold": ngold})
    write_jsonl(Path(out_dir) / f"{name}.counts.jsonl", rows)
    return rows


def micro_f1(rows):
    tp = sum(r["tp"] for r in rows)
    denom = sum(r["pred"] + r["gold"] for r in rows)
    return 2 * tp / denom if denom else 0.0


def predict_set(name, model, out_dir, binary=BIN):
    records = load(name)
    preds = run_analyzer([r["text"] for r in records], model, binary)
    write_jsonl(Path(out_dir) / f"{name}.pred.jsonl",
                [{"id": r["id"], "tokens": p} for r, p in zip(records, preds)])
    return score_set(name, {r["id"]: p for r, p in zip(records, preds)}, out_dir, records)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["predict", "count"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", type=Path)
    ap.add_argument("--binary", type=Path, default=BIN)
    ap.add_argument("--sets", default=",".join(SETS))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for name in args.sets.split(","):
        if args.mode == "predict":
            rows = predict_set(name, args.model, args.out, args.binary)
        else:
            preds = {r["id"]: r["tokens"] for r in read_jsonl(args.out / f"{name}.pred.jsonl")}
            rows = score_set(name, preds, args.out)
        print(f"{name}: n={len(rows)} F1={micro_f1(rows) * 100:.6f}%")


if __name__ == "__main__":
    main()
