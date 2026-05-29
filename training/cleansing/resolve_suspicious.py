"""의심 문장의 어절별 수렴 판정 → resolved / still_suspicious 분리.

문장의 모든 의심 어절이 수렴하면 정상 문장으로 승격(flat morphemes 재구성).
하나라도 미해소면 still_suspicious — 단 해소된 어절은 resolved 필드로 표시.

사용:
    $GARU_ENSEMBLE_PYTHON training/cleansing/resolve_suspicious.py \\
        --input training/gold_testset/gold_suspicious.jsonl \\
        --resolved training/gold_testset/gold_resolved.jsonl \\
        --still training/gold_testset/gold_still_suspicious.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sejong_normalize import candidates_converge


def resolve_sentence(sent: dict):
    """반환 (out_dict, fully_resolved: bool).

    fully_resolved=True → out은 vote_status=normal + morphemes(flat) + eojeol_votes.
    False → out은 vote_status=suspicious, suspicious_eojeols의 각 항목에
            수렴 시 resolved=<morphemes> 추가, 미수렴은 그대로.
    """
    resolutions = {}  # index → resolved morphemes
    for se in sent["suspicious_eojeols"]:
        conv = candidates_converge(se["candidates"])
        if conv is not None:
            resolutions[se["index"]] = conv

    fully = len(resolutions) == len(sent["suspicious_eojeols"])

    if fully:
        # eojeol 단위로 morphemes 재구성: eojeol_votes(정상) + 해소된 의심 어절
        n = len(sent["eojeol_votes"]) + len(sent["suspicious_eojeols"])
        slots = [None] * n
        for se in sent["suspicious_eojeols"]:
            slots[se["index"]] = resolutions[se["index"]]
        # 남은 슬롯에 eojeol_votes를 순서대로 채움
        ev_iter = iter(sent["eojeol_votes"])
        for i in range(n):
            if slots[i] is None:
                slots[i] = next(ev_iter)["morphemes"]
        flat = []
        for m in slots:
            flat.extend(m)
        out = {
            "text": sent["text"],
            "domain": sent["domain"],
            "vote_status": "normal",
            "morphemes": flat,
            "eojeol_votes": sent["eojeol_votes"],
            "resolved_by": "phase2_rule",
            "resolved_eojeol_count": len(sent["suspicious_eojeols"]),
        }
        return out, True

    # 부분 해소: suspicious 유지, 해소된 어절에 resolved 표시
    new_susp = []
    for se in sent["suspicious_eojeols"]:
        item = dict(se)
        if se["index"] in resolutions:
            item["resolved"] = resolutions[se["index"]]
        new_susp.append(item)
    out = dict(sent)
    out["suspicious_eojeols"] = new_susp
    return out, False


def process_file(input_path: Path, resolved_path: Path, still_path: Path) -> dict:
    if resolved_path.exists() or still_path.exists():
        raise SystemExit(f"Output exists: {resolved_path} / {still_path}. Delete or rename.")
    stats = {"total": 0, "resolved": 0, "still": 0, "eojeols_resolved": 0, "eojeols_remaining": 0}
    with open(input_path) as fin, open(resolved_path, "w") as fr, open(still_path, "w") as fs:
        for line in fin:
            sent = json.loads(line)
            stats["total"] += 1
            out, fully = resolve_sentence(sent)
            if fully:
                stats["resolved"] += 1
                stats["eojeols_resolved"] += len(sent["suspicious_eojeols"])
                fr.write(json.dumps(out, ensure_ascii=False) + "\n")
            else:
                stats["still"] += 1
                for se in out["suspicious_eojeols"]:
                    if se.get("resolved") is not None:
                        stats["eojeols_resolved"] += 1
                    else:
                        stats["eojeols_remaining"] += 1
                fs.write(json.dumps(out, ensure_ascii=False) + "\n")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--resolved", required=True, type=Path)
    ap.add_argument("--still", required=True, type=Path)
    args = ap.parse_args()
    stats = process_file(args.input, args.resolved, args.still)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
