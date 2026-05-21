"""코퍼스 일괄 분석 → candidates_pool 생성 테스트."""
import sys, os, json, tempfile, gzip
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "crawl"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ensemble"))
from analyze_corpus import build_candidates_pool

SAMPLE_SENTENCES = [
    {"text": "오늘 데이터베이스 설계를 했어요.", "source": "test"},
    {"text": "쿠버네티스 클러스터가 잘 동작합니다.", "source": "test"},
    {"text": "그 영화 정말 재미있었어요.", "source": "test"},
]


def test_build_candidates_pool_small():
    """3문장으로 후보 풀 생성, 합리적 구조 검증."""
    with tempfile.TemporaryDirectory() as td:
        in_path = os.path.join(td, "in.jsonl.gz")
        with gzip.open(in_path, "wt", encoding="utf-8") as f:
            for s in SAMPLE_SENTENCES:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        out_path = os.path.join(td, "candidates_pool.jsonl")
        count = build_candidates_pool(
            input_paths=[in_path], output_path=out_path, domain="test",
        )
        assert count >= 1, "no candidates generated"
        records = [json.loads(l) for l in open(out_path)]
        assert len(records) >= 1
        for rec in records:
            assert "surface" in rec
            assert "normalized_pos" in rec
            assert "votes" in rec
            assert "in_garu_dict" in rec
            assert "frequency" in rec
            assert "source_domains" in rec
            assert isinstance(rec["votes"], dict)
            assert set(rec["votes"].keys()) == {"kiwi", "mecab", "kkma", "komoran"}
            assert all(v in (0, 1) for v in rec["votes"].values())
            assert rec["frequency"] >= 1

if __name__ == "__main__":
    test_build_candidates_pool_small()
    print("ANALYZE OK")
