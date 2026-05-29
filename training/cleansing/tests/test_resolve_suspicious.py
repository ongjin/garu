"""resolve_suspicious 파이프라인 smoke 테스트 (분석기 호출 없음, 순수 로직)."""
import json
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from resolve_suspicious import resolve_sentence, process_file


def test_resolve_sentence_all_converge():
    # 단일 의심 어절이 EP 통일로 수렴 → resolved, flat morphemes 재구성
    sent = {
        "text": "그는 하였다",
        "domain": "일상",
        "vote_status": "suspicious",
        "eojeol_votes": [
            {"surface": "그는", "agree": 5, "morphemes": [["그","NP"],["는","JX"]]},
        ],
        "suspicious_eojeols": [
            {"index": 1, "surface": "하였다", "agree": 2, "candidates": [
                {"analyzers": ["kiwi"], "morphemes": [["하","VV"],["았","EP"],["다","EF"]]},
                {"analyzers": ["mecab"], "morphemes": [["하","VV"],["었","EP"],["다","EF"]]},
            ]},
        ],
    }
    out, fully = resolve_sentence(sent)
    assert fully is True
    assert out["vote_status"] == "normal"
    # eojeol 순서대로 flat: 그/NP 는/JX 하/VV 였/EP 다/EF (EP 통일)
    assert out["morphemes"] == [["그","NP"],["는","JX"],["하","VV"],["였","EP"],["다","EF"]]


def test_resolve_sentence_partial():
    # 의심 어절 중 하나는 EP 수렴, 하나는 POS 충돌 → still suspicious
    sent = {
        "text": "그 하였다", "domain": "x", "vote_status": "suspicious",
        "eojeol_votes": [],
        "suspicious_eojeols": [
            {"index": 0, "surface": "그", "agree": 2, "candidates": [
                {"analyzers": ["kiwi","kkma"], "morphemes": [["그","MM"]]},
                {"analyzers": ["garu"], "morphemes": [["그","VA"]]},
            ]},
            {"index": 1, "surface": "하였다", "agree": 2, "candidates": [
                {"analyzers": ["kiwi"], "morphemes": [["하","VV"],["았","EP"],["다","EF"]]},
                {"analyzers": ["mecab"], "morphemes": [["하","VV"],["었","EP"],["다","EF"]]},
            ]},
        ],
    }
    out, fully = resolve_sentence(sent)
    assert fully is False
    assert out["vote_status"] == "suspicious"
    # "그"는 미해소로 남고, "하였다"는 resolved(EP 통일) 표시
    by_idx = {se["index"]: se for se in out["suspicious_eojeols"]}
    assert 0 in by_idx  # 그: 여전히 의심
    assert by_idx[0].get("resolved") is None
    assert 1 in by_idx and by_idx[1].get("resolved") == [["하","VV"],["였","EP"],["다","EF"]]


def test_process_file():
    inputs = [
        {"text": "그는 하였다", "domain": "x", "vote_status": "suspicious",
         "eojeol_votes": [{"surface":"그는","agree":5,"morphemes":[["그","NP"],["는","JX"]]}],
         "suspicious_eojeols": [
             {"index":1,"surface":"하였다","agree":2,"candidates":[
                 {"analyzers":["kiwi"],"morphemes":[["하","VV"],["았","EP"],["다","EF"]]},
                 {"analyzers":["mecab"],"morphemes":[["하","VV"],["었","EP"],["다","EF"]]}]}]},
        {"text": "그 사람", "domain": "x", "vote_status": "suspicious",
         "eojeol_votes": [{"surface":"사람","agree":5,"morphemes":[["사람","NNG"]]}],
         "suspicious_eojeols": [
             {"index":0,"surface":"그","agree":2,"candidates":[
                 {"analyzers":["kiwi","kkma"],"morphemes":[["그","MM"]]},
                 {"analyzers":["garu"],"morphemes":[["그","VA"]]}]}]},
    ]
    d = Path(tempfile.mkdtemp())
    inp = d / "in.jsonl"
    with open(inp, "w") as f:
        for x in inputs:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    resolved = d / "resolved.jsonl"
    still = d / "still.jsonl"
    stats = process_file(inp, resolved, still)
    assert stats["total"] == 2
    assert stats["resolved"] == 1
    assert stats["still"] == 1
