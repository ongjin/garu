"""고정 평가 세트 로더.

세트마다 문장 레코드 {id, doc, stratum, text, gold} 목록을 돌려준다.
- gold : v15k 골드 9,000문장. 문서 정보가 없어 문장이 재표집 단위(doc=id), 층=도메인.
- sx25 : NIKL 2025 구어(SX) held-out. 층=SX 하나, 재표집 단위=문서.
- nx25 : NIKL 2025 문어(NX, 2023년 기사) held-out. 층=기사 주제, 재표집 단위=문서.

2025 세트는 `eval_nikl2025_guueh.load_sx`와 같은 필터(길이 2~200, '_'→공백)를 쓰고
골드를 `reverse_2025_to_2021.reverse_convert`로 2021 관례에 맞춘다. sx25의 문장 순서·
내용은 기존 held-out 평가와 같아야 한다(`test_gate.py`가 검증).
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))
from eval_nikl2025_guueh import npos  # noqa: E402
from reverse_2025_to_2021 import reverse_convert  # noqa: E402

GOLD = ROOT / "training" / "gold_testset" / "gold_testset.jsonl"
NIKL_2025 = Path(os.environ.get("NIKL_MP_2025_DIR",
                                Path.home() / "workspace" / "data" / "nikl_mp_2025"))
SX_FILE = "SXMP2502512312.json"
NX_FILE = "NXMP2502512313.json"
MIN_CHARS, MAX_CHARS = 2, 200

SETS = ("gold", "sx25", "nx25")
HELDOUT = ("sx25", "nx25")


def _load_gold():
    out = []
    with open(GOLD, encoding="utf-8") as f:
        for i, line in enumerate(f):
            rec = json.loads(line)
            sid = f"gold:{i}"
            out.append({"id": sid, "doc": sid, "stratum": rec["domain"],
                        "text": rec["text"], "gold": rec["morphemes"]})
    return out


def _load_2025(path, stratum_of):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for doc in data["document"]:
        if doc is None:
            continue
        stratum = stratum_of(doc)
        for sent in (doc.get("sentence") or []):
            text = sent.get("form", "")
            if not text or len(text) < MIN_CHARS or len(text) > MAX_CHARS:
                continue
            morphs = [[m["form"], npos(m["label"])]
                      for m in (sent.get("MP") or []) if m.get("form", "").strip()]
            if not morphs:
                continue
            out.append({"id": sent["id"], "doc": doc["id"], "stratum": stratum,
                        "text": text.replace("_", " "), "gold": reverse_convert(morphs)})
    return out


def load(name):
    if name == "gold":
        return _load_gold()
    if name == "sx25":
        return _load_2025(NIKL_2025 / SX_FILE, lambda doc: "SX")
    if name == "nx25":
        return _load_2025(NIKL_2025 / NX_FILE, lambda doc: doc["metadata"]["topic"])
    raise ValueError(f"unknown set: {name}")
