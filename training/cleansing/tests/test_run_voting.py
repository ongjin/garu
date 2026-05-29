"""run_voting 파이프라인 smoke 테스트.

작은 jsonl 입력으로 정상/의심 분리가 올바르게 작동하는지 확인.
실제 5분석기를 호출하므로 약간의 시간 소요.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_voting import process_sentences


def test_pipeline_smoke():
    """3문장 입력 → voted / suspicious 분리, 스키마 검증."""
    inputs = [
        {"text": "정부가 발표했다.", "domain": "뉴스"},
        {"text": "ㅋㅋㅋㅋ 진짜 ㅎㅎ", "domain": "SNS"},  # 의심 가능성↑
        {"text": "철수가 학교에 갔다.", "domain": "일상"},
    ]
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for d in inputs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
        in_path = Path(f.name)

    voted_path = in_path.with_suffix(".voted.jsonl")
    suspicious_path = in_path.with_suffix(".suspicious.jsonl")

    stats = process_sentences(in_path, voted_path, suspicious_path)

    assert stats["total"] == 3
    assert stats["normal"] + stats["suspicious"] == 3

    # 출력 파일 스키마 검증
    voted = [json.loads(l) for l in open(voted_path)] if voted_path.exists() else []
    susp = [json.loads(l) for l in open(suspicious_path)] if suspicious_path.exists() else []

    for d in voted:
        assert d["vote_status"] == "normal"
        assert "morphemes" in d and len(d["morphemes"]) > 0
        assert "eojeol_votes" in d
        for ev in d["eojeol_votes"]:
            assert ev["agree"] >= 3
            assert "morphemes" in ev

    for d in susp:
        assert d["vote_status"] == "suspicious"
        assert "suspicious_eojeols" in d and len(d["suspicious_eojeols"]) >= 1
        for se in d["suspicious_eojeols"]:
            assert "candidates" in se and len(se["candidates"]) >= 2
            for cand in se["candidates"]:
                assert "analyzers" in cand
                assert "morphemes" in cand

    # cleanup
    in_path.unlink()
    if voted_path.exists(): voted_path.unlink()
    if suspicious_path.exists(): suspicious_path.unlink()
