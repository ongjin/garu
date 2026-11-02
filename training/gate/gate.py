"""고정 세트 paired bootstrap 게이트 (gate v2).

    python3 training/gate/gate.py BASE_DIR CAND_DIR [--json OUT]

BASE_DIR·CAND_DIR는 score.py가 만든 <set>.counts.jsonl을 담는다. 판정 규칙은 GATE에
고정돼 있고 docs/wiki/benchmarks.md의 '게이트 v2' 절과 같다. 결과를 본 뒤 규칙을 바꾸지 않는다.

Δ = 후보 micro F1 − 기준 micro F1 (ep_norm, 2·TP/(예측+정답)).
재표집은 층(gold=도메인, nx25=주제, sx25=단일) 안에서 단위(gold=문장, 2025=문서)를
복원추출한다. 기준·후보는 같은 표본을 공유한다(paired).
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score import read_jsonl  # noqa: E402
from sets import HELDOUT, SETS  # noqa: E402

GATE = {
    "version": 2,
    "replicates": 10_000,
    "seed": 20261102,
    "benefit_alpha": 0.025,      # held-out 합동 Δ의 단측 하한
    "regression_alpha": 0.05,    # 하위 집합 전체에 Bonferroni로 나눔
}
CHUNK = 200
FIELDS = ("tp", "pred", "gold")


def _units(base_rows, cand_rows):
    """(층, 단위)별 [tp_b, pred_b, gold, tp_c, pred_c] 합."""
    assert [r["id"] for r in base_rows] == [r["id"] for r in cand_rows], "id 순서 불일치"
    acc = defaultdict(lambda: np.zeros(5, np.int64))
    for b, c in zip(base_rows, cand_rows):
        assert b["gold"] == c["gold"], f"정답 수 불일치: {b['id']}"
        acc[(b["stratum"], b["doc"])] += (b["tp"], b["pred"], b["gold"], c["tp"], c["pred"])
    strata = defaultdict(list)
    for (stratum, _), vec in acc.items():
        strata[stratum].append(vec)
    return {s: np.array(v) for s, v in strata.items()}


def _delta(tot):
    """tot[..., 5] → Δ F1 (후보 − 기준)."""
    tot = tot.astype(np.float64)
    base = 2 * tot[..., 0] / (tot[..., 1] + tot[..., 2])
    cand = 2 * tot[..., 3] / (tot[..., 4] + tot[..., 2])
    return cand - base


def _replicate_totals(strata, rng, replicates):
    """층별 복원추출 합 [replicates, 5]. 층마다 독립 난수열을 쓴다."""
    out = np.zeros((replicates, 5), np.int64)
    for name in sorted(strata):
        units = strata[name]
        for start in range(0, replicates, CHUNK):
            n = min(CHUNK, replicates - start)
            idx = rng.integers(0, len(units), size=(n, len(units)))
            out[start:start + n] += units[idx].sum(axis=1)
    return out


def subsets(base, cand):
    """판정 하위 집합 이름 → 해당 층 딕셔너리."""
    out = {}
    for name in SETS:
        if name not in base:
            continue
        strata = _units(base[name], cand[name])
        out[f"{name}/all"] = strata
        if len(strata) > 1:
            for s in sorted(strata):
                out[f"{name}/{s}"] = {s: strata[s]}
    pooled = {f"{name}:{s}": u for name in HELDOUT if name in base
              for s, u in _units(base[name], cand[name]).items()}
    return out, pooled


def _summary(strata, rng, replicates, lower_q, upper_q):
    point = sum(u.sum(axis=0) for u in strata.values())
    reps = _delta(_replicate_totals(strata, rng, replicates))
    lo, hi = np.quantile(reps, [lower_q, upper_q])
    return {"delta_pp": float(_delta(point)) * 100, "lo_pp": float(lo) * 100,
            "hi_pp": float(hi) * 100}


def judge(base, cand, gate=GATE):
    named, pooled = subsets(base, cand)
    m = len(named)
    reg_q = gate["regression_alpha"] / m
    rng = np.random.default_rng(gate["seed"])
    report = {"gate": gate, "subsets": {}, "regression_quantile": reg_q}

    for name in sorted(named):
        s = _summary(named[name], rng, gate["replicates"], reg_q, 1 - reg_q)
        s["significant_regression"] = s["hi_pp"] < 0
        report["subsets"][name] = s

    benefit = _summary(pooled, rng, gate["replicates"], gate["benefit_alpha"],
                       1 - gate["benefit_alpha"])
    report["heldout_pooled"] = benefit
    regressions = [n for n, s in report["subsets"].items() if s["significant_regression"]]
    report["regressions"] = regressions
    report["benefit"] = benefit["lo_pp"] > 0
    report["pass"] = report["benefit"] and not regressions
    return report


def changed(base_rows, cand_rows):
    win = loss = 0
    for b, c in zip(base_rows, cand_rows):
        fb = 2 * b["tp"] / (b["pred"] + b["gold"]) if b["pred"] + b["gold"] else 1.0
        fc = 2 * c["tp"] / (c["pred"] + c["gold"]) if c["pred"] + c["gold"] else 1.0
        win += fc > fb + 1e-12
        loss += fc < fb - 1e-12
    return win, loss


def load_dir(path):
    out = {}
    for name in SETS:
        f = Path(path) / f"{name}.counts.jsonl"
        if f.exists():
            out[name] = read_jsonl(f)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base", type=Path)
    ap.add_argument("cand", type=Path)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    base, cand = load_dir(args.base), load_dir(args.cand)
    assert base.keys() == cand.keys() and base, "두 디렉터리의 세트 구성이 다름"
    report = judge(base, cand)
    report["changes"] = {name: dict(zip(("win", "loss"), changed(base[name], cand[name])))
                         for name in base}

    q = report["regression_quantile"]
    print(f"gate v{GATE['version']}: 회귀 판정 단측 분위 {q:.5f} (하위 집합 {len(report['subsets'])}개)")
    print(f"{'subset':<22} {'Δpp':>9} {'lo':>9} {'hi':>9}")
    for name, s in report["subsets"].items():
        mark = "  ← 유의 회귀" if s["significant_regression"] else ""
        print(f"{name:<22} {s['delta_pp']:>+9.4f} {s['lo_pp']:>+9.4f} {s['hi_pp']:>+9.4f}{mark}")
    b = report["heldout_pooled"]
    print(f"{'held-out pooled':<22} {b['delta_pp']:>+9.4f} {b['lo_pp']:>+9.4f} {b['hi_pp']:>+9.4f}"
          f"  (단측 {GATE['benefit_alpha']} 하한)")
    for name, c in report["changes"].items():
        print(f"  {name}: 개선 {c['win']} / 악화 {c['loss']} 문장")
    print("PASS" if report["pass"] else
          f"FAIL (이득 {'있음' if report['benefit'] else '미확인'}, 유의 회귀 {report['regressions']})")
    if args.json:
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
