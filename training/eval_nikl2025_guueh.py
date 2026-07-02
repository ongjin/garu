"""Held-out 구어 평가: NIKL MP 2025 SX(구어)를 2021 컨벤션으로 역변환한 골드에 대해
Garu F1을 잰다. **학습용 아님** — 2025는 학습에 주입하지 않는다(track1 리포트: 주입 시
v15k 회귀). 현대 구어(게임/스트리밍 레지스터)에서 Garu 실측 성능 게이지 용도.

golд·pred 모두 ep_norm(gold_testset/expand/ep_norm.py) 후 다중집합 F1. 2025 골드는
reverse_2025_to_2021.reverse_convert로 2021 입도로 내린다. POS는 42-태그로 정규화
(MMD/MMN→MM 등) — 누락 시 관형사 하위분류가 거짓 오류로 잡히니 주의(track1 교훈).

Usage:
    python3 training/eval_nikl2025_guueh.py
    NIKL_MP_2025_SX=/path/to/SXMP....json python3 training/eval_nikl2025_guueh.py
    python3 training/eval_nikl2025_guueh.py --check-invertibility   # 역함수 성질 검증
"""
import json, os, subprocess, sys, tempfile
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "training" / "gold_testset" / "expand"))
sys.path.insert(0, str(ROOT / "training"))
from ep_norm import normalize_ep_morphemes
from reverse_2025_to_2021 import reverse_convert
from eval_nikl_mp import normalize_for_2025

SX = Path(os.environ.get(
    "NIKL_MP_2025_SX",
    str(Path.home() / "workspace" / "data" / "nikl_mp_2025" / "SXMP2502512312.json")))
BIN = ROOT / "target" / "release" / "examples" / "analyze_batch"
MODEL = os.environ.get("GARU_MODEL", str(ROOT / "js" / "models" / "base.gmdl"))

POS_SET = set("NNG NNP NNB NR NP VV VA VX VCP VCN MAG MAJ MM IC JKS JKC JKG JKO JKB JKV "
              "JKQ JX JC EP EF EC ETN ETM XPN XSN XSV XSA XR SF SP SS SE SO SW SH SL SN".split())
NIKL_MAP = {'MMD': 'MM', 'MMN': 'MM', 'MMA': 'MM', 'NA': 'NNG', 'NAP': 'NNG', 'NF': 'NNG', 'NV': 'VV'}


def npos(t):
    if t in POS_SET:
        return t
    if t in NIKL_MAP:
        return NIKL_MAP[t]
    b = t.split('-')[0]
    return b if b in POS_SET else 'SW'


def load_sx():
    """→ list of (feed_text, gold_2025_morphs[[f,p]...])"""
    data = json.load(open(SX))
    out = []
    for doc in data["document"]:
        if doc is None:
            continue
        for sent in (doc.get("sentence") or []):
            text = sent.get("form", "")
            if not text or len(text) < 2 or len(text) > 200:
                continue
            ms = [[m["form"], npos(m["label"])]
                  for m in (sent.get("MP") or []) if m.get("form", "").strip()]
            if ms:
                out.append((text.replace("_", " "), ms))
    return out


def run_garu(texts):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("\n".join(texts))
        inp = f.name
    r = subprocess.run([str(BIN), inp, "--json"], capture_output=True, text=True,
                       env={**os.environ, "GARU_MODEL": MODEL})
    os.unlink(inp)
    if r.returncode != 0:
        sys.exit(f"analyze_batch failed: {r.stderr[:500]}")
    return [json.loads(l) for l in r.stdout.strip().split("\n")]


def epn(ms):
    return normalize_ep_morphemes([list(m) for m in ms])


def f1(pred, gold_2021):
    tp = fp = fn = 0
    for p, g in zip(pred, gold_2021):
        ps, gs = Counter(), Counter()
        for f, t in epn(p):
            ps[(f, t)] += 1
        for f, t in epn(g):
            gs[(f, t)] += 1
        for k in set(ps) | set(gs):
            m = min(ps[k], gs[k]); tp += m; fp += ps[k] - m; fn += gs[k] - m
    P = tp / (tp + fp) if tp + fp else 0
    R = tp / (tp + fn) if tp + fn else 0
    return P, R, (2 * P * R / (P + R) if P + R else 0)


def main():
    sents = load_sx()
    gold25 = [g for _, g in sents]
    gold21 = [reverse_convert(g) for g in gold25]

    if "--check-invertibility" in sys.argv:
        ok = sum(1 for g in gold25
                 if {(f, t) for f, t in normalize_for_2025(reverse_convert(g))}
                 == {(f, t) for f, t in normalize_for_2025(g)})
        print(f"invertibility normalize_for_2025(reverse(g))==normalize_for_2025(g): "
              f"{ok}/{len(gold25)} = {ok / len(gold25):.4%}")
        return

    print(f"NIKL 2025 SX: {SX.name}  ({len(sents)} sentences)")
    pred = run_garu([t for t, _ in sents])
    P, R, F = f1(pred, gold21)
    print(f"Garu (reverse→2021 conv, ep_norm both):  P={P:.4f} R={R:.4f} F1={F:.4f}")


if __name__ == "__main__":
    main()
