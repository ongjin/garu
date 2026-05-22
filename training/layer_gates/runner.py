"""한 layer 전체 파이프라인: filter → backup → merge → build → F1 측정 → decide.

스펙: .specs/2026-05-21-dict-expansion-design.md 섹션 3.5.
"""
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from filter import LayerConfig, filter_candidates
from merge import append_entries_to_dict

ROOT = Path(__file__).resolve().parents[2]
POOL_PATH = ROOT / "training" / "codebook_data" / "candidates_pool.jsonl"
DICT_PATH = ROOT / "training" / "codebook_data" / "content_dict.txt"
LOG_PATH = ROOT / "training" / "dict_expansion_log.tsv"
PYTHON = os.environ.get("GARU_ENSEMBLE_PYTHON", "/opt/homebrew/bin/python3.14")


def _run_build():
    subprocess.run([PYTHON, str(ROOT / "training" / "build_codebook_model.py")],
                   check=True, cwd=ROOT)


def _run_f1() -> dict[str, float]:
    """F1 스크립트 실행 + stdout 파싱."""
    script = ROOT / "training" / "gold_testset" / "eval_f1.py"
    res = subprocess.run([PYTHON, str(script)],
                         capture_output=True, text=True, check=True, cwd=ROOT)
    out = res.stdout
    f1 = {}
    for line in out.splitlines():
        if line.strip().startswith("Garu"):
            parts = line.split()
            if len(parts) == 4:
                f1["overall"] = float(parts[3])
                break
    for line in out.splitlines():
        s = line.strip()
        if "Garu=" in s and "Kiwi=" in s:
            tok = s.split()
            domain = tok[0]
            garu_part = next(t for t in tok if t.startswith("Garu="))
            f1[domain] = float(garu_part.split("=")[1])
    return f1


def _log_result(layer_name: str, baseline: dict, new: dict, added: int, decision: str):
    line = (f"{datetime.now().isoformat()}\t{layer_name}\t{added}\t{decision}\t"
            f"overall={new.get('overall')}\t"
            + "\t".join(f"{k}={v}" for k, v in new.items() if k != "overall"))
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def _decide(layer_name: str, baseline: dict, new: dict, tolerance_per_domain: float) -> str:
    if new["overall"] < baseline["overall"]:
        return f"reject_overall (delta={new['overall']-baseline['overall']:+.4f})"
    for k, v in new.items():
        if k == "overall":
            continue
        if v < baseline.get(k, 0) - tolerance_per_domain:
            return f"reject_domain_{k} (delta={v-baseline.get(k,0):+.4f})"
    return "pass"


def _backup_dict(suffix: str) -> Path:
    bak = DICT_PATH.with_suffix(f".bak.{suffix}")
    shutil.copy2(DICT_PATH, bak)
    return bak


def _restore_dict(bak: Path):
    shutil.copy2(bak, DICT_PATH)


def run_layer(cfg: LayerConfig, baseline: dict[str, float],
              tolerance_per_domain: float = 0.10) -> dict:
    """한 layer 전체 실행."""
    records = [json.loads(l) for l in open(POOL_PATH)]
    candidates = filter_candidates(records, cfg)
    print(f"[{cfg.name}] {len(candidates)} candidates passed filter")
    if not candidates:
        return {"layer": cfg.name, "added": 0, "decision": "skip_empty",
                "candidates": 0, "f1": None, "baseline": baseline}

    bak = _backup_dict(cfg.name)
    print(f"[{cfg.name}] backup -> {bak}")

    added = append_entries_to_dict(candidates, str(DICT_PATH))
    print(f"[{cfg.name}] appended {added} entries")

    print(f"[{cfg.name}] building model...")
    _run_build()

    print(f"[{cfg.name}] running F1...")
    new = _run_f1()
    print(f"[{cfg.name}] F1: {new}")

    decision = _decide(cfg.name, baseline, new, tolerance_per_domain)
    print(f"[{cfg.name}] decision: {decision}")

    if not decision.startswith("pass"):
        _restore_dict(bak)
        _run_build()
        print(f"[{cfg.name}] reverted content_dict + rebuilt model")

    _log_result(cfg.name, baseline, new, added, decision)
    return {"layer": cfg.name, "added": added, "decision": decision,
            "candidates": len(candidates), "f1": new, "baseline": baseline}


BASELINE_F1 = {
    "overall": 0.9330, "SNS": 0.9175, "구어": 0.9292, "기술": 0.9571,
    "뉴스": 0.9298, "문학": 0.9623, "일상": 0.9209,
}
