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
    """4문장 입력 → voted / suspicious 분리, 스키마 검증.

    입력은 두 분기를 모두 자극하도록 구성: 단일 명사 2개(5분석기 합의 보장) + SNS/구어 2개(불일치 유도).
    종결어미가 붙은 동사 어절은 5/5 합의가 어려워 (mecab vs kiwi 등) susp로 쏠리므로,
    voted 보장을 위해 단일 명사를 사용.
    """
    inputs = [
        {"text": "사과", "domain": "일상"},                  # 단일 명사 → voted 보장
        {"text": "나무", "domain": "일상"},                  # 단일 명사 → voted 보장
        {"text": "ㅋㅋㅋㅋ 진짜 ㅎㅎ", "domain": "SNS"},        # 의심 보장
        {"text": "녯사람이 꿈에 나타났더라.", "domain": "기타"},  # 고어/희귀형 → 의심 보장
    ]
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for d in inputs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
        in_path = Path(f.name)

    voted_path = in_path.with_suffix(".voted.jsonl")
    suspicious_path = in_path.with_suffix(".suspicious.jsonl")

    try:
        stats = process_sentences(in_path, voted_path, suspicious_path)

        assert stats["total"] == 4
        assert stats["normal"] + stats["suspicious"] == 4

        # 출력 파일 스키마 검증
        voted = [json.loads(l) for l in open(voted_path)] if voted_path.exists() else []
        susp = [json.loads(l) for l in open(suspicious_path)] if suspicious_path.exists() else []

        # 두 분기 모두 실제로 자극되었는지 보장 (한쪽으로 다 쏠리면 스키마 검증 갭 발생)
        assert len(voted) >= 1, f"voted 분기가 자극되지 않음 (전부 suspicious). susp={len(susp)}"
        assert len(susp) >= 1, f"suspicious 분기가 자극되지 않음 (전부 normal). voted={len(voted)}"
        assert len(voted) + len(susp) == 4

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
    finally:
        # cleanup — try/finally 로 단언 실패 시에도 임시파일 정리 보장 (재실행 가능성)
        if in_path.exists(): in_path.unlink()
        if voted_path.exists(): voted_path.unlink()
        if suspicious_path.exists(): suspicious_path.unlink()
