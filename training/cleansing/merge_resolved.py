"""최종 병합: gold_voted + gold_resolved + still(규칙+Claude 결정) → gold_cleansed_final.

still 문장의 각 어절을 index 순서로 채운다:
  - eojeol_votes (정상 어절): surface 순서대로
  - suspicious_eojeols: resolved 필드 있으면 그것, 없으면 claude decision (uid="<sentidx>_<index>")

모든 최종 morpheme 시퀀스에 결정적 정규화 적용:
  1. 결합 자모(ᆫ 등) → 호환 자모(ㄴ)
  2. 하 어간 여불규칙: 하+EP(았/었/였)→였, 하+EC(아/어/여)→여

사용:
    $GARU_ENSEMBLE_PYTHON training/cleansing/merge_resolved.py \\
        --voted training/gold_testset/gold_voted.jsonl \\
        --resolved training/gold_testset/gold_resolved.jsonl \\
        --still training/gold_testset/gold_still_suspicious.jsonl \\
        --decisions training/gold_testset/claude_decisions.jsonl \\
        --out training/gold_testset/gold_cleansed_final.jsonl
"""
import argparse
import json
from pathlib import Path

# 결합 자모(Hangul Jamo U+11xx) → 호환 자모(U+31xx) 매핑
_CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
_JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
_JONG = "ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"
_JAMO_MAP = {}
for _i, _c in enumerate(_CHO):
    _JAMO_MAP[chr(0x1100 + _i)] = _c
for _i, _c in enumerate(_JUNG):
    _JAMO_MAP[chr(0x1161 + _i)] = _c
for _i, _c in enumerate(_JONG):
    _JAMO_MAP[chr(0x11A8 + _i)] = _c


def _norm_jamo(s: str) -> str:
    return "".join(_JAMO_MAP.get(c, c) for c in s)


def normalize_morphemes(morphemes: list) -> list:
    """최종 morpheme 정규화. 새 리스트 반환.

    1. 결합 자모 → 호환 자모.
    2. 하 어간 여불규칙: 하+EP(았/었/였)→였, 하+EC(아/어/여)→여.
    """
    out = [[_norm_jamo(s), p] for s, p in morphemes]
    for i in range(1, len(out)):
        if out[i - 1][0] == "하":
            s, p = out[i]
            if p == "EP" and s in ("았", "었", "였"):
                out[i][0] = "였"
            elif p == "EC" and s in ("아", "어", "여"):
                out[i][0] = "여"
    return out


def merge_still_sentence(sent: dict, sent_idx: int, decisions: dict) -> dict:
    """still 문장 1개를 정상 문장으로 병합. decisions: {uid: morphemes}."""
    n = len(sent["eojeol_votes"]) + len(sent["suspicious_eojeols"])
    slots = [None] * n
    for se in sent["suspicious_eojeols"]:
        idx = se["index"]
        if se.get("resolved") is not None:
            slots[idx] = se["resolved"]
        else:
            uid = f"{sent_idx}_{idx}"
            if uid not in decisions:
                raise KeyError(f"Claude decision missing for {uid}")
            slots[idx] = decisions[uid]
    ev_iter = iter(sent["eojeol_votes"])
    for i in range(n):
        if slots[i] is None:
            slots[i] = next(ev_iter)["morphemes"]
    flat = []
    for m in slots:
        flat.extend(m)
    return {
        "text": sent["text"],
        "domain": sent["domain"],
        "vote_status": "normal",
        "morphemes": normalize_morphemes(flat),
        "resolved_by": "phase2_claude",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voted", required=True, type=Path)
    ap.add_argument("--resolved", required=True, type=Path)
    ap.add_argument("--still", required=True, type=Path)
    ap.add_argument("--decisions", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"Output exists: {args.out}")

    decisions = {}
    for line in open(args.decisions):
        d = json.loads(line)
        decisions[d["uid"]] = d["morphemes"]

    n_out = 0
    with open(args.out, "w") as fo:
        for line in open(args.voted):
            d = json.loads(line)
            fo.write(json.dumps({"text": d["text"], "domain": d["domain"],
                                 "vote_status": "normal",
                                 "morphemes": normalize_morphemes(d["morphemes"]),
                                 "resolved_by": "phase1_vote"}, ensure_ascii=False) + "\n")
            n_out += 1
        for line in open(args.resolved):
            d = json.loads(line)
            fo.write(json.dumps({"text": d["text"], "domain": d["domain"],
                                 "vote_status": "normal",
                                 "morphemes": normalize_morphemes(d["morphemes"]),
                                 "resolved_by": "phase2_rule"}, ensure_ascii=False) + "\n")
            n_out += 1
        for sent_idx, line in enumerate(open(args.still)):
            sent = json.loads(line)
            out = merge_still_sentence(sent, sent_idx, decisions)
            fo.write(json.dumps(out, ensure_ascii=False) + "\n")
            n_out += 1
    print(f"merged {n_out} sentences → {args.out}")


if __name__ == "__main__":
    main()
