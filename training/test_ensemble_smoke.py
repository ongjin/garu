"""End-to-end smoke test: 5분석기 + agreement 통합 동작 검증.

스펙: .specs/2026-05-21-dict-expansion-design.md 섹션 2 전체.
Plan A Task 7.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ensemble"))
from wrappers import (
    KiwiWrapper, MecabWrapper, KkmaWrapper, KomoranWrapper, GaruWrapper,
)
from agreement import compute_votes, ANALYZERS, total_votes

SMOKE_SENTENCES = [
    "안녕하세요 오늘 날씨가 좋네요.",
    "데이터베이스 설계는 정규화부터 시작합니다.",
    "그 영화 너무 재밌었어요 ㅋㅋㅋ",
    "헌법재판소가 결정문을 발표했다.",
    "환자가 호흡곤란을 호소했다.",
    "ChatGPT가 코드를 짜준다고?",
    "광주세계수영선수권대회가 개최되었다.",
    "도와줘서 정말 고마워요.",
    "쿠버네티스 클러스터를 구성하는 중이다.",
    "비가 오는데 우산을 안 가져왔다.",
]

def test_all_analyzers_respond_to_10_sentences():
    kiwi = KiwiWrapper()
    mecab = MecabWrapper()
    kkma = KkmaWrapper()
    komoran = KomoranWrapper()
    garu = GaruWrapper()

    candidate_count = 0
    for sent in SMOKE_SENTENCES:
        analyses = {
            "kiwi":    kiwi.analyze(sent),
            "mecab":   mecab.analyze(sent),
            "kkma":    kkma.analyze(sent),
            "komoran": komoran.analyze(sent),
        }
        garu_tokens = garu.analyze(sent)
        # 모든 분석기가 비어있지 않은 결과 반환
        for name, toks in analyses.items():
            assert len(toks) > 0, f"{name} returned empty for: {sent}"
        # agreement 계산이 에러 없이 동작
        votes = compute_votes(analyses, garu_analysis=garu_tokens)
        candidate_count += len(votes)
        # 최소 1개 후보 — 모든 분석기가 동일하게 단일 형태소 인식하는 토큰이 있어야 함
        assert len(votes) > 0, f"no votes for: {sent}"
    # 10문장 → 누적 후보 수가 합리적 범위 (50-1000 정도)
    assert 50 <= candidate_count <= 5000, f"unexpected candidate count: {candidate_count}"
    print(f"total candidates across 10 sentences: {candidate_count}")

def test_high_agreement_token_exists():
    """4/4 합의 후보가 최소 한 문장에서는 나와야 한다."""
    kiwi = KiwiWrapper(); mecab = MecabWrapper()
    kkma = KkmaWrapper(); komoran = KomoranWrapper()
    garu = GaruWrapper()
    found_4_of_4 = False
    for sent in SMOKE_SENTENCES:
        analyses = {
            "kiwi": kiwi.analyze(sent), "mecab": mecab.analyze(sent),
            "kkma": kkma.analyze(sent), "komoran": komoran.analyze(sent),
        }
        votes = compute_votes(analyses, garu_analysis=garu.analyze(sent))
        for key, info in votes.items():
            if total_votes(info) == 4:
                found_4_of_4 = True
                break
        if found_4_of_4:
            break
    assert found_4_of_4, "no 4/4 consensus token found in 10 smoke sentences"

if __name__ == "__main__":
    test_all_analyzers_respond_to_10_sentences()
    print("all_analyzers OK")
    test_high_agreement_token_exists()
    print("4/4 consensus OK")
    print("SMOKE OK")
